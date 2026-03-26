#pragma once
// pointsource_calibration.h — Point-source calibration refinement pipeline.
// Uses synchronized video of a green light wand in a dark arena to
// refine camera calibration (intrinsics + extrinsics) via bundle adjustment.
// Reference: github.com/JohnsonLabJanelia/pointSourceCalib (rj branch)

#include "calibration_pipeline.h" // ReprojectionCost, CameraPose, write_calibration
#include "opencv_yaml_io.h"       // opencv_yaml::read
#include "red_math.h"             // triangulatePoints, projectPoint, etc.

#ifdef __APPLE__
#include "FFmpegDemuxer.h"        // Annex-B demuxing for VT decode
#include "vt_async_decoder.h"     // HW decode → BGRA CVPixelBuffer (zero swscale)
#include "pointsource_metal.h"          // GPU-accelerated light spot detection
#include <CoreVideo/CoreVideo.h>
#else
#include "ffmpeg_frame_reader.h"  // FFmpeg + swscale fallback (Linux)
#endif

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace PointSourceCalibration {

// Project persistence moved to CalibrationTool::CalibProject (calibration_tool.h)

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

// Optimization mode for pointsource BA
enum class PointSourceOptMode {
    ExtrinsicsOnly = 0,      // Fix all intrinsics, optimize R + t
    ExtrinsicsAndFocal = 1,  // + optimize fx, fy
    ExtrinsicsAndAll = 2,    // + optimize fx, fy, cx, cy, k1, k2 (lock p1, p2, k3)
    Full = 3                 // All parameters free (fx, fy, cx, cy, k1, k2, p1, p2, k3)
};

inline const char *pointsource_opt_mode_name(PointSourceOptMode m) {
    switch (m) {
        case PointSourceOptMode::ExtrinsicsOnly: return "Extrinsics only";
        case PointSourceOptMode::ExtrinsicsAndFocal: return "Extrinsics + focal length";
        case PointSourceOptMode::ExtrinsicsAndAll: return "Extrinsics + all intrinsics";
        case PointSourceOptMode::Full: return "Full (all parameters free)";
        default: return "Unknown";
    }
}

struct PointSourceConfig {
    std::string media_folder;
    std::vector<std::string> camera_names;
    std::string calibration_folder;
    std::string output_folder;

    // Detection (defaults from parameter sweep, March 26 2026)
    int green_threshold = 30;  // 25-40 all good; <15 causes failure
    int green_dominance = 5;   // no sensitivity in range 3-15
    int min_blob_pixels = 20;  // filters noise; 10 works but 20 gives +23% more points
    int max_blob_pixels = 600; // no wand blobs exceed this; >600 adds overhead

    // Frame range
    int start_frame = 0;    // first frame to process
    int stop_frame = 0;     // last frame (0 = process all)
    int frame_step = 3;     // optimal quality/speed tradeoff at 180fps

    // Filtering
    int min_cameras = 4;
    double reproj_threshold = 15.0;

    // BA
    double ba_outlier_th1 = 20.0;
    double ba_outlier_th2 = 5.0;
    int ba_max_iter = 50;
    PointSourceOptMode opt_mode = PointSourceOptMode::ExtrinsicsOnly;

    // Global registration (optional — Procrustes alignment after BA)
    std::string global_reg_media_folder;
    std::string global_reg_media_type;  // "videos" or "images"
    CalibrationTool::CharucoSetup charuco_setup;
    std::map<std::string, std::vector<std::vector<float>>> gt_pts;
    std::vector<std::string> world_coordinate_imgs;
    Eigen::Matrix3d world_frame_rotation = Eigen::Matrix3d::Identity();
    bool do_global_reg = false;

    // Detection modes
    bool smart_blob = false;  // multi-blob: pick largest instead of rejecting frame

    // Initialization modes
    bool loose_init = false;  // auto-detect and PnP re-init poorly-calibrated cameras
    bool no_init = false;     // bootstrap all params from Global Reg. Media (implies loose_init)
};

inline nlohmann::json config_to_json(const PointSourceConfig &c) {
    return nlohmann::json{
        {"media_folder", c.media_folder},
        {"calibration_folder", c.calibration_folder},
        {"camera_names", c.camera_names},
        {"green_threshold", c.green_threshold},
        {"green_dominance", c.green_dominance},
        {"min_blob_pixels", c.min_blob_pixels},
        {"max_blob_pixels", c.max_blob_pixels},
        {"start_frame", c.start_frame},
        {"stop_frame", c.stop_frame},
        {"frame_step", c.frame_step},
        {"min_cameras", c.min_cameras},
        {"reproj_threshold", c.reproj_threshold},
        {"ba_outlier_th1", c.ba_outlier_th1},
        {"ba_outlier_th2", c.ba_outlier_th2},
        {"ba_max_iter", c.ba_max_iter},
        {"opt_mode", static_cast<int>(c.opt_mode)},
        {"opt_mode_name", pointsource_opt_mode_name(c.opt_mode)},
        {"do_global_reg", c.do_global_reg},
        {"global_reg_media_folder", c.global_reg_media_folder},
        {"global_reg_media_type", c.global_reg_media_type},
        {"smart_blob", c.smart_blob},
        {"loose_init", c.loose_init},
        {"no_init", c.no_init}};
}

struct SpotDetection {
    double cx, cy;
    int pixel_count;
};

// Smart Blob v2: store all valid blobs per frame for deferred resolution
struct BlobCandidate {
    float cx, cy;       // intensity-weighted centroid
    int pixel_count;    // blob area (dilated pixels)
};
using FrameCandidates = std::map<int, std::vector<BlobCandidate>>;

// Smart Blob v3: artifact mask — identifies static blobs before main detection.
// Preferred mode: keyframe pre-scan (spatially distributed samples from throughout
// the video). Fallback: rolling accumulation during detection.
// Once ready, is_artifact() provides O(1) lookup for filtering.
// Unified spatial artifact mask — used by keyframe pre-scan, filter_static_detections,
// and resolve_blob_candidates. Configurable threshold mode and bin growth.
struct ArtifactMask {
    int bw = 0, bh = 0, bin_size = 32;
    std::vector<int> bins;
    std::vector<bool> mask;
    int frames_seen = 0;
    bool ready = false;
    int artifact_bins_flagged = 0;

    enum ThresholdMode {
        Percentage,      // threshold = max(10, pct * frames_seen) — for keyframe pre-scan
        MedianMultiple   // threshold = max(20, mult * median_nonempty) — for post-hoc filtering
    };

    void init(int img_w, int img_h, int bs = 32) {
        bin_size = bs;
        bw = (img_w + bs - 1) / bs;
        bh = (img_h + bs - 1) / bs;
        bins.assign(bw * bh, 0);
        mask.assign(bw * bh, false);
        frames_seen = 0;
        ready = false;
        artifact_bins_flagged = 0;
    }

    void add_frame(const std::vector<BlobCandidate> &blobs) {
        for (const auto &b : blobs) {
            int bx = std::clamp((int)(b.cx / bin_size), 0, bw - 1);
            int by = std::clamp((int)(b.cy / bin_size), 0, bh - 1);
            bins[by * bw + bx]++;
        }
        frames_seen++;
    }

    // Add detections from a DetectionMap (for filter_static_detections compatibility)
    void add_detections(const std::map<int, SpotDetection> &detections) {
        for (const auto &[frame, det] : detections) {
            int bx = std::clamp((int)(det.cx / bin_size), 0, bw - 1);
            int by = std::clamp((int)(det.cy / bin_size), 0, bh - 1);
            bins[by * bw + bx]++;
        }
        frames_seen += (int)detections.size();
    }

    void finalize(ThresholdMode mode = Percentage, double param = 0.7, bool grow = false) {
        int threshold;
        if (mode == Percentage) {
            threshold = std::max(10, (int)(param * frames_seen));
        } else {
            // MedianMultiple: param is the multiplier (e.g., 5.0)
            std::vector<int> nonempty;
            for (int v : bins) if (v > 0) nonempty.push_back(v);
            if (nonempty.empty()) { ready = true; return; }
            std::sort(nonempty.begin(), nonempty.end());
            int median = nonempty[nonempty.size() / 2];
            threshold = std::max(20, (int)(param * median));
        }

        std::vector<bool> flagged(bw * bh, false);
        for (int i = 0; i < bw * bh; i++)
            if (bins[i] > threshold) flagged[i] = true;

        if (grow) {
            mask.assign(bw * bh, false);
            for (int y = 0; y < bh; y++)
                for (int x = 0; x < bw; x++) {
                    if (!flagged[y * bw + x]) continue;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < bw && ny >= 0 && ny < bh)
                                mask[ny * bw + nx] = true;
                        }
                }
        } else {
            mask = flagged;
        }
        artifact_bins_flagged = 0;
        for (bool b : mask) if (b) artifact_bins_flagged++;
        ready = true;
    }

    bool is_artifact(float cx, float cy) const {
        if (!ready) return false;
        int bx = std::clamp((int)(cx / bin_size), 0, bw - 1);
        int by = std::clamp((int)(cy / bin_size), 0, bh - 1);
        return mask[by * bw + bx];
    }

    int filter(std::vector<BlobCandidate> &blobs) const {
        if (!ready) return 0;
        int before = (int)blobs.size();
        blobs.erase(std::remove_if(blobs.begin(), blobs.end(),
            [this](const BlobCandidate &b) { return is_artifact(b.cx, b.cy); }),
            blobs.end());
        return before - (int)blobs.size();
    }

    // Filter a DetectionMap in-place (for filter_static_detections compatibility)
    int filter_detections(std::map<int, SpotDetection> &detections) const {
        if (!ready) return 0;
        int removed = 0;
        for (auto it = detections.begin(); it != detections.end(); ) {
            if (is_artifact((float)it->second.cx, (float)it->second.cy)) {
                it = detections.erase(it);
                removed++;
            } else {
                ++it;
            }
        }
        return removed;
    }
};

struct FrameObservations {
    int frame_num;
    std::vector<int> cam_indices;
    std::vector<Eigen::Vector2d> pixel_coords;
};

struct Observation {
    int cam_idx;
    int point_idx;
    double px, py;
};

struct CameraChange {
    std::string name;
    // Intrinsic deltas
    double dfx = 0, dfy = 0, dcx = 0, dcy = 0;
    // Extrinsic deltas
    double dt_x = 0, dt_y = 0, dt_z = 0;
    double dt_norm = 0;     // ||delta_t||
    double drot_deg = 0;    // rotation change in degrees
    // Per-camera counts
    int detections = 0;     // light spots found
    int observations = 0;   // observations used in BA (after filtering)
};

struct PointSourceResult {
    bool success = false;
    std::string error;
    int valid_3d_points = 0;
    int total_observations = 0;
    int ba_outliers_removed = 0;
    double mean_reproj_before = 0.0;
    double mean_reproj_after = 0.0;
    std::string output_folder;
    std::vector<CameraChange> camera_changes;
    std::string global_reg_status;  // empty if not attempted, else Procrustes report
};

// Per-camera progress counters, shared between detection threads and UI
struct CameraProgress {
    std::atomic<int> frames_processed{0};
    std::atomic<int> spots_detected{0};
    std::atomic<bool> done{false};
};

struct DetectionProgress {
    std::vector<std::unique_ptr<CameraProgress>> cameras;
    std::atomic<int> current_step{0};  // 0=not started, 1-8 pipeline steps
    std::atomic<int> total_steps{8};
    void init(int num_cameras) {
        cameras.clear();
        cameras.reserve(num_cameras);
        for (int i = 0; i < num_cameras; i++)
            cameras.push_back(std::make_unique<CameraProgress>());
        current_step.store(0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Load previous results from disk
// ─────────────────────────────────────────────────────────────────────────────

// Find the most recent timestamped subfolder (YYYY_MM_DD_HH_MM_SS) in a directory.
// Returns empty string if none found.
inline std::string find_latest_timestamped_subfolder(const std::string &dir) {
    namespace fs = std::filesystem;
    if (!fs::exists(dir) || !fs::is_directory(dir))
        return "";
    std::string latest;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        // Expect format: YYYY_MM_DD_HH_MM_SS (19 chars, all digits and underscores)
        if (name.size() == 19 && name[4] == '_' && name[7] == '_' &&
            name[10] == '_' && name[13] == '_' && name[16] == '_') {
            if (name > latest)
                latest = name;
        }
    }
    if (latest.empty())
        return "";
    return dir + "/" + latest;
}

// Load a PointSourceResult from a summary.json file in a timestamped output folder.
// Returns true on success, populating result. The output_folder field is set to
// the timestamped folder containing the summary.
inline bool load_result_from_summary(const std::string &timestamped_folder,
                                     PointSourceResult &result,
                                     std::string *err = nullptr) {
    namespace fs = std::filesystem;
    std::string summary_path = timestamped_folder + "/summary_data/summary.json";
    if (!fs::exists(summary_path)) {
        if (err) *err = "summary.json not found in " + timestamped_folder;
        return false;
    }
    try {
        std::ifstream ifs(summary_path);
        nlohmann::json j;
        ifs >> j;

        result.success = true;
        result.output_folder = timestamped_folder;
        result.mean_reproj_before = j.value("mean_reproj_before", 0.0);
        result.mean_reproj_after = j.value("mean_reproj_after", 0.0);
        result.valid_3d_points = j.value("valid_3d_points", 0);
        result.total_observations = j.value("total_observations", 0);
        result.ba_outliers_removed = j.value("ba_outliers_removed", 0);

        result.camera_changes.clear();
        if (j.contains("camera_changes") && j["camera_changes"].is_array()) {
            for (const auto &cam_j : j["camera_changes"]) {
                CameraChange cc;
                cc.name = cam_j.value("name", "");
                cc.detections = cam_j.value("detections", 0);
                cc.observations = cam_j.value("observations", 0);
                cc.dfx = cam_j.value("dfx", 0.0);
                cc.dfy = cam_j.value("dfy", 0.0);
                cc.dcx = cam_j.value("dcx", 0.0);
                cc.dcy = cam_j.value("dcy", 0.0);
                cc.dt_x = cam_j.value("dt_x", 0.0);
                cc.dt_y = cam_j.value("dt_y", 0.0);
                cc.dt_z = cam_j.value("dt_z", 0.0);
                cc.dt_norm = cam_j.value("dt_norm", 0.0);
                cc.drot_deg = cam_j.value("drot_deg", 0.0);
                result.camera_changes.push_back(cc);
            }
        }

        // Check for global_reg_status in settings.json
        std::string settings_path = timestamped_folder + "/summary_data/settings.json";
        if (fs::exists(settings_path)) {
            std::ifstream sfs(settings_path);
            nlohmann::json sj;
            sfs >> sj;
            if (sj.contains("global_reg_status"))
                result.global_reg_status = sj["global_reg_status"].get<std::string>();
        }

        return true;
    } catch (const std::exception &e) {
        if (err) *err = std::string("Failed to parse summary.json: ") + e.what();
        result = PointSourceResult();
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Get estimated frame count from a video file.
inline int get_video_frame_count(const std::string &path) {
    AVFormatContext *ctx = avformat_alloc_context();
    if (!ctx) return 0;
    ctx->max_analyze_duration = 5 * AV_TIME_BASE;
    if (avformat_open_input(&ctx, path.c_str(), nullptr, nullptr) < 0)
        return 0;
    if (avformat_find_stream_info(ctx, nullptr) < 0) {
        avformat_close_input(&ctx);
        return 0;
    }
    for (unsigned i = 0; i < ctx->nb_streams; i++) {
        if (ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            int64_t nb = ctx->streams[i]->nb_frames;
            if (nb > 0) {
                avformat_close_input(&ctx);
                return (int)nb;
            }
            double dur = ctx->duration / (double)AV_TIME_BASE;
            AVRational rate = ctx->streams[i]->avg_frame_rate;
            double fps = (rate.num > 0 && rate.den > 0)
                             ? (double)rate.num / rate.den
                             : 30.0;
            avformat_close_input(&ctx);
            return (int)(dur * fps);
        }
    }
    avformat_close_input(&ctx);
    return 0;
}

// Find video files in media folder.
// If camera_names is empty, returns all videos found (keyed by stem).
// If camera_names is provided, returns only matching videos.
inline std::map<std::string, std::string>
find_video_files(const std::string &media_folder,
                 const std::vector<std::string> &camera_names) {
    namespace fs = std::filesystem;
    std::map<std::string, std::string> result;
    const std::vector<std::string> exts = {".mp4", ".avi", ".mov", ".mkv"};

    if (!fs::exists(media_folder))
        return result;

    std::map<std::string, std::string> stem_to_path;
    for (const auto &entry : fs::directory_iterator(media_folder)) {
        if (!entry.is_regular_file())
            continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(exts.begin(), exts.end(), ext) == exts.end())
            continue;
        stem_to_path[entry.path().stem().string()] = entry.path().string();
    }

    if (camera_names.empty()) {
        // Return all found videos
        return stem_to_path;
    }

    for (const auto &name : camera_names) {
        // camera_names may be "2002486" while stems are "Cam2002486"
        std::string stem = "Cam" + name;
        auto it = stem_to_path.find(stem);
        if (it == stem_to_path.end())
            it = stem_to_path.find(name); // fallback: try exact match
        if (it != stem_to_path.end())
            result[name] = it->second;
    }
    return result;
}

// Validate cameras: return sorted intersection of video stems and YAML files.
inline std::vector<std::string>
validate_cameras(const std::string &media_folder,
                 const std::string &calibration_folder) {
    namespace fs = std::filesystem;
    auto videos = find_video_files(media_folder, {});

    std::set<std::string> yaml_stems;
    if (fs::exists(calibration_folder)) {
        for (const auto &entry : fs::directory_iterator(calibration_folder)) {
            if (entry.path().extension().string() == ".yaml") {
                yaml_stems.insert(entry.path().stem().string());
            }
        }
    }

    std::vector<std::string> result;
    for (const auto &[stem, _] : videos) {
        if (yaml_stems.count(stem)) {
            // Strip "Cam" prefix to match calibration pipeline convention
            std::string name = stem;
            if (name.size() > 3 && name.substr(0, 3) == "Cam")
                name = name.substr(3);
            result.push_back(name);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

// Derive camera names from video filenames alone (No Init mode — no YAML files).
inline std::vector<std::string>
camera_names_from_videos(const std::string &media_folder) {
    auto videos = find_video_files(media_folder, {});
    std::vector<std::string> result;
    for (const auto &[stem, _] : videos) {
        std::string name = stem;
        if (name.size() > 3 && name.substr(0, 3) == "Cam")
            name = name.substr(3);
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Detect light spot in a single frame
// ─────────────────────────────────────────────────────────────────────────────

// Union-Find for connected components
struct UnionFind {
    std::vector<int> parent, rank_uf, sz;

    UnionFind(int n) : parent(n), rank_uf(n, 0), sz(n, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (rank_uf[a] < rank_uf[b])
            std::swap(a, b);
        parent[b] = a;
        sz[a] += sz[b];
        if (rank_uf[a] == rank_uf[b])
            rank_uf[a]++;
    }
};

// Pixel format for detect_light_spot.
// RGB24: R=0, G=1, B=2 (3 bpp, stride = width*3)
// BGRA:  B=0, G=1, R=2, A=3 (4 bpp, stride from CVPixelBuffer)
enum PointSourcePixelFormat { POINTSOURCE_FMT_RGB24, POINTSOURCE_FMT_BGRA };

// Detect a single green light spot in a frame (RGB24 or BGRA).
// stride = bytes per row (may include padding for BGRA from CVPixelBuffer).
// Returns true if exactly one valid blob found; fills detection.
// smart_blob: when true, pick the largest valid blob instead of rejecting multi-blob frames.
// ── Shared green blob detection pipeline (CPU path) ──
// Threshold → erode → dilate → connected components → size filter → centroids.
// Returns all valid-sized blobs with intensity-weighted centroids.
// Single-pass centroid computation: O(N) instead of O(k*N).
inline std::vector<BlobCandidate> detect_green_blobs(
    const uint8_t *pixels, int width, int height,
    int stride, PointSourcePixelFormat fmt,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels) {

    int npixels = width * height;
    int bpp = (fmt == POINTSOURCE_FMT_BGRA) ? 4 : 3;
    int r_off = (fmt == POINTSOURCE_FMT_BGRA) ? 2 : 0;
    int b_off = (fmt == POINTSOURCE_FMT_BGRA) ? 0 : 2;

    // Step 1: Threshold — binary mask
    std::vector<uint8_t> mask(npixels, 0);
    for (int y = 0; y < height; y++) {
        const uint8_t *row = pixels + y * stride;
        for (int x = 0; x < width; x++) {
            uint8_t r = row[x * bpp + r_off];
            uint8_t g = row[x * bpp + 1];
            uint8_t b = row[x * bpp + b_off];
            if (g > green_threshold && g >= r && g >= b &&
                (  (g > r + green_dominance && g > b + green_dominance)
                || (g > 200 && g >= r && g >= b)  ))
                mask[y * width + x] = 1;
        }
    }

    // Step 2: Erode (3x3) — pixel stays 1 only if all 8 neighbors are also 1
    std::vector<uint8_t> eroded(npixels, 0);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            if (!mask[idx]) continue;
            bool all = true;
            for (int dy = -1; dy <= 1 && all; dy++)
                for (int dx = -1; dx <= 1 && all; dx++)
                    if (!mask[(y + dy) * width + (x + dx)]) all = false;
            if (all) eroded[idx] = 1;
        }
    }

    // Step 3: Dilate (3x3) — pixel becomes 1 if any neighbor is 1
    std::vector<uint8_t> dilated(npixels, 0);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            bool has_neighbor = false;
            for (int dy = -1; dy <= 1 && !has_neighbor; dy++)
                for (int dx = -1; dx <= 1 && !has_neighbor; dx++)
                    if (eroded[(y + dy) * width + (x + dx)]) has_neighbor = true;
            if (has_neighbor) dilated[y * width + x] = 1;
        }
    }

    // Step 4: Connected components (4-connectivity) via union-find
    UnionFind uf(npixels);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (!dilated[idx]) continue;
            if (x > 0 && dilated[idx - 1]) uf.unite(idx, idx - 1);
            if (y > 0 && dilated[idx - width]) uf.unite(idx, idx - width);
        }

    // Step 5: Size filter + single-pass centroid computation
    // Accumulate per-root centroid in one pass over all dilated pixels.
    struct Accum { double sx = 0, sy = 0, sw = 0; int count = 0; };
    std::map<int, Accum> accums;
    for (int i = 0; i < npixels; i++) {
        if (!dilated[i]) continue;
        int root = uf.find(i);
        int sz = uf.sz[root];
        if (sz < min_blob_pixels || sz > max_blob_pixels) continue;
        int px = i % width, py = i / width;
        double w = pixels[py * stride + px * bpp + 1]; // green channel
        auto &a = accums[root];
        a.sx += px * w; a.sy += py * w; a.sw += w; a.count++;
    }

    std::vector<BlobCandidate> result;
    for (auto &[root, a] : accums) {
        if (a.sw > 1e-9)
            result.push_back({(float)(a.sx / a.sw), (float)(a.sy / a.sw), a.count});
    }
    return result;
}

// Single-blob detection (conservative path): reject frames with multiple blobs.
// Used when smart_blob=false.
inline bool detect_light_spot(const uint8_t *pixels, int width, int height,
                              int stride, PointSourcePixelFormat fmt,
                              int green_threshold, int green_dominance,
                              int min_blob_pixels, int max_blob_pixels,
                              SpotDetection &det, bool smart_blob = false) {
    auto blobs = detect_green_blobs(pixels, width, height, stride, fmt,
                                     green_threshold, green_dominance,
                                     min_blob_pixels, max_blob_pixels);
    if (blobs.empty()) return false;
    if (blobs.size() > 1 && !smart_blob) return false;
    // Pick the largest valid blob
    const BlobCandidate *best = &blobs[0];
    for (const auto &b : blobs)
        if (b.pixel_count > best->pixel_count) best = &b;
    det.cx = best->cx;
    det.cy = best->cy;
    det.pixel_count = best->pixel_count;
    return true;
}

// Multi-blob detection (Smart Blob path): return all valid blobs for deferred resolution.
inline std::vector<BlobCandidate> detect_all_blobs(
    const uint8_t *pixels, int width, int height,
    int stride, PointSourcePixelFormat fmt,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels) {
    return detect_green_blobs(pixels, width, height, stride, fmt,
                              green_threshold, green_dominance,
                              min_blob_pixels, max_blob_pixels);
}

// ─────────────────────────────────────────────────────────────────────────────
// Static artifact filter: remove detections at fixed locations.
// The wand sweeps across the image, spreading detections across many spatial
// bins. Artifacts (LEDs, reflections) produce dense clusters at fixed positions.
// ─────────────────────────────────────────────────────────────────────────────

using DetectionMap = std::map<int, SpotDetection>;

// Thin wrapper: filter static detections using ArtifactMask (5x median, with growth).
inline int filter_static_detections(DetectionMap &detections,
                                     int image_width, int image_height,
                                     int bin_size = 32,
                                     double static_threshold = 5.0) {
    if (detections.size() < 50) return 0;
    ArtifactMask am;
    am.init(image_width, image_height, bin_size);
    am.add_detections(detections);
    am.finalize(ArtifactMask::MedianMultiple, static_threshold, /*grow=*/true);
    return am.filter_detections(detections);
}

// ─────────────────────────────────────────────────────────────────────────────
// Smart Blob v2: resolve multi-blob candidates into a single detection per frame.
//
// Stage 1: Build artifact zone map from spatial clustering of ALL candidates
// Stage 2: For each frame, filter artifact-zone candidates
// Stage 3: Tiebreak remaining candidates via reprojection or size consistency
// ─────────────────────────────────────────────────────────────────────────────

inline DetectionMap resolve_blob_candidates(
    const FrameCandidates &candidates,
    int image_width, int image_height,
    int bin_size = 32,
    double static_threshold = 5.0,
    // Optional: for reprojection-based tiebreaking (Refinement mode)
    const std::vector<CalibrationPipeline::CameraPose> *poses = nullptr,
    int cam_idx = -1,
    const std::vector<DetectionMap> *other_detections = nullptr,
    int num_cameras = 0) {

    DetectionMap result;
    if (candidates.empty()) return result;

    // ── Stage 1: Build artifact zone map using ArtifactMask ──
    ArtifactMask am;
    am.init(image_width, image_height, bin_size);
    for (const auto &[frame, blobs] : candidates)
        am.add_frame(blobs);
    am.finalize(ArtifactMask::MedianMultiple, static_threshold, /*grow=*/true);

    auto in_artifact = [&](float cx, float cy) { return am.is_artifact(cx, cy); };

    // ── Compute median blob size from single-candidate frames (for size tiebreaker) ──
    std::vector<int> single_blob_sizes;
    for (const auto &[frame, blobs] : candidates) {
        // Count non-artifact candidates
        std::vector<const BlobCandidate*> clean;
        for (const auto &b : blobs)
            if (!in_artifact(b.cx, b.cy)) clean.push_back(&b);
        if (clean.size() == 1)
            single_blob_sizes.push_back(clean[0]->pixel_count);
    }
    int median_blob_size = 0;
    if (!single_blob_sizes.empty()) {
        std::sort(single_blob_sizes.begin(), single_blob_sizes.end());
        median_blob_size = single_blob_sizes[single_blob_sizes.size() / 2];
    }

    // Count artifact removals for logging
    int artifact_frames_recovered = 0;
    int artifact_detections_removed = 0;
    int reprojection_tiebreaks = 0;
    int temporal_tiebreaks = 0;
    int size_tiebreaks = 0;

    // ── Temporal continuity state (v3) ──
    float track_cx = -1, track_cy = -1;  // last resolved position
    float track_vx = 0, track_vy = 0;    // velocity estimate (px/frame)
    int track_frame = -1;                 // frame number of last resolved

    // ── Stage 2 & 3: Per-frame resolution ──
    for (const auto &[frame, blobs] : candidates) {
        // Filter out artifact-zone candidates
        std::vector<const BlobCandidate*> clean;
        for (const auto &b : blobs) {
            if (in_artifact(b.cx, b.cy))
                artifact_detections_removed++;
            else
                clean.push_back(&b);
        }

        if (clean.empty()) continue;

        const BlobCandidate *winner = nullptr;

        if (clean.size() == 1) {
            winner = clean[0];
            if (blobs.size() > 1) artifact_frames_recovered++;
        } else {
            // Multiple non-artifact candidates — need tiebreaker

            // Tiebreaker A: Reprojection (if poses available)
            if (poses && other_detections && cam_idx >= 0 && num_cameras > 0) {
                // Find this frame in other cameras' resolved detections
                std::vector<std::pair<int, Eigen::Vector2d>> other_obs;
                for (int oc = 0; oc < num_cameras; oc++) {
                    if (oc == cam_idx) continue;
                    auto it = (*other_detections)[oc].find(frame);
                    if (it != (*other_detections)[oc].end()) {
                        other_obs.push_back({oc, Eigen::Vector2d(it->second.cx, it->second.cy)});
                    }
                }

                if ((int)other_obs.size() >= 2) {
                    // Triangulate 3D point from other cameras
                    std::vector<Eigen::Vector2d> pts2d;
                    std::vector<Eigen::Matrix<double, 3, 4>> Ps;
                    for (const auto &[oc, px] : other_obs) {
                        Eigen::Vector2d und = red_math::undistortPoint(
                            px, (*poses)[oc].K, (*poses)[oc].dist);
                        pts2d.push_back(und);
                        Ps.push_back(red_math::projectionFromKRt(
                            (*poses)[oc].K, (*poses)[oc].R, (*poses)[oc].t));
                    }
                    Eigen::Vector3d X = red_math::triangulatePoints(pts2d, Ps);

                    // Project into this camera, pick closest candidate
                    Eigen::Vector3d rvec = red_math::rotationMatrixToVector((*poses)[cam_idx].R);
                    Eigen::Vector2d proj = red_math::projectPoint(
                        X, rvec, (*poses)[cam_idx].t,
                        (*poses)[cam_idx].K, (*poses)[cam_idx].dist);

                    double best_dist = 1e18;
                    for (const auto *c : clean) {
                        double d = std::hypot(c->cx - proj.x(), c->cy - proj.y());
                        if (d < best_dist) { best_dist = d; winner = c; }
                    }
                    reprojection_tiebreaks++;
                }
            }

            // Tiebreaker B: Temporal continuity (v3)
            // The wand moves smoothly — predict position from recent frames
            if (!winner && track_cx >= 0) {
                int dt = frame - track_frame;
                float pred_x = track_cx + track_vx * dt;
                float pred_y = track_cy + track_vy * dt;
                double best_dist = 1e18;
                for (const auto *c : clean) {
                    double d = std::hypot(c->cx - pred_x, c->cy - pred_y);
                    if (d < best_dist) { best_dist = d; winner = c; }
                }
                // Sanity: reject if best candidate is implausibly far from prediction
                // (wand moves at ~0.5-5 px/frame, so dt*50 px is generous)
                double max_dist = std::max(200.0, 50.0 * dt);
                if (best_dist > max_dist) winner = nullptr;
                if (winner) temporal_tiebreaks++;
            }

            // Tiebreaker C: Size consistency (fallback)
            if (!winner && median_blob_size > 0) {
                int best_diff = INT_MAX;
                for (const auto *c : clean) {
                    int diff = std::abs(c->pixel_count - median_blob_size);
                    if (diff < best_diff) { best_diff = diff; winner = c; }
                }
                size_tiebreaks++;
            }

            // Last resort: pick largest
            if (!winner) {
                for (const auto *c : clean)
                    if (!winner || c->pixel_count > winner->pixel_count) winner = c;
            }
        }

        if (winner) {
            result[frame] = {winner->cx, winner->cy, winner->pixel_count};

            // Update temporal tracking state
            if (track_cx >= 0 && track_frame >= 0) {
                int dt = frame - track_frame;
                if (dt > 0 && dt < 100) { // reasonable gap
                    // Exponential smoothing on velocity (alpha=0.3)
                    float new_vx = (winner->cx - track_cx) / dt;
                    float new_vy = (winner->cy - track_cy) / dt;
                    track_vx = 0.3f * new_vx + 0.7f * track_vx;
                    track_vy = 0.3f * new_vy + 0.7f * track_vy;
                }
            }
            track_cx = winner->cx;
            track_cy = winner->cy;
            track_frame = frame;
        }
    }

    int total_frames = (int)candidates.size();
    int resolved = (int)result.size();
    printf("  resolve_blob_candidates: %d/%d frames resolved "
           "(%d artifact removed, %d recovered, "
           "%d reproj, %d temporal, %d size tiebreaks)\n",
           resolved, total_frames, artifact_detections_removed,
           artifact_frames_recovered, reprojection_tiebreaks,
           temporal_tiebreaks, size_tiebreaks);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// DLT PnP: closed-form camera pose from 3D-2D correspondences.
// No initialization needed — solves the projection matrix directly via SVD.
// ─────────────────────────────────────────────────────────────────────────────

inline bool solvePnPDLT(
    const std::vector<Eigen::Vector3d> &pts_3d,
    const std::vector<Eigen::Vector2d> &pts_2d,
    const Eigen::Matrix3d &K,
    Eigen::Matrix3d &R_out, Eigen::Vector3d &t_out) {

    int n = (int)pts_3d.size();
    if (n < 6) return false;

    // Hartley normalization of 2D points
    Eigen::Vector2d mean_2d = Eigen::Vector2d::Zero();
    for (const auto &p : pts_2d) mean_2d += p;
    mean_2d /= n;
    double scale_2d = 0;
    for (const auto &p : pts_2d) scale_2d += (p - mean_2d).norm();
    scale_2d = std::sqrt(2.0) * n / scale_2d;
    Eigen::Matrix3d T2d = Eigen::Matrix3d::Identity();
    T2d(0, 0) = scale_2d; T2d(1, 1) = scale_2d;
    T2d(0, 2) = -scale_2d * mean_2d.x();
    T2d(1, 2) = -scale_2d * mean_2d.y();

    // Normalize 3D points
    Eigen::Vector3d mean_3d = Eigen::Vector3d::Zero();
    for (const auto &p : pts_3d) mean_3d += p;
    mean_3d /= n;
    double scale_3d = 0;
    for (const auto &p : pts_3d) scale_3d += (p - mean_3d).norm();
    scale_3d = std::sqrt(3.0) * n / scale_3d;
    Eigen::Matrix4d T3d = Eigen::Matrix4d::Identity();
    T3d(0, 0) = scale_3d; T3d(1, 1) = scale_3d; T3d(2, 2) = scale_3d;
    T3d(0, 3) = -scale_3d * mean_3d.x();
    T3d(1, 3) = -scale_3d * mean_3d.y();
    T3d(2, 3) = -scale_3d * mean_3d.z();

    // Build 2N x 12 system: for each correspondence (X, x),
    // x cross (P * X_h) = 0 gives two independent equations
    Eigen::MatrixXd A(2 * n, 12);
    for (int i = 0; i < n; i++) {
        // Normalized points
        Eigen::Vector3d xn = T2d * Eigen::Vector3d(pts_2d[i].x(), pts_2d[i].y(), 1.0);
        Eigen::Vector4d Xn = T3d * Eigen::Vector4d(pts_3d[i].x(), pts_3d[i].y(), pts_3d[i].z(), 1.0);
        double u = xn.x(), v = xn.y(), w = xn.z();

        // Row 1: w*X^T*P2 - v*X^T*P3
        A.row(2 * i + 0) << 0, 0, 0, 0,
            -w * Xn.transpose(), v * Xn.transpose();
        // Row 2: -w*X^T*P1 + u*X^T*P3
        A.row(2 * i + 1) << w * Xn.transpose(), 0, 0, 0, 0,
            -u * Xn.transpose();
    }

    // SVD: P is the last column of V
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd p = svd.matrixV().col(11);
    Eigen::Matrix<double, 3, 4> P_norm;
    P_norm.row(0) = p.segment<4>(0).transpose();
    P_norm.row(1) = p.segment<4>(4).transpose();
    P_norm.row(2) = p.segment<4>(8).transpose();

    // Denormalize: P_real = T2d_inv * P_norm * T3d
    Eigen::Matrix<double, 3, 4> P = T2d.inverse() * P_norm * T3d;

    // Extract [R|t] = K_inv * P
    Eigen::Matrix<double, 3, 4> Rt = K.inverse() * P;
    Eigen::Matrix3d M = Rt.block<3, 3>(0, 0);
    Eigen::Vector3d t_raw = Rt.col(3);

    // Resolve DLT sign ambiguity (P is defined up to scale/sign).
    // Ensure M has positive determinant so SVD extracts a proper rotation.
    if (M.determinant() < 0) { M = -M; t_raw = -t_raw; }

    // Closest rotation matrix via SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_r(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R_out = svd_r.matrixU() * svd_r.matrixV().transpose();
    if (R_out.determinant() < 0) {
        Eigen::Matrix3d V = svd_r.matrixV();
        V.col(2) *= -1;
        R_out = svd_r.matrixU() * V.transpose();
    }

    // Scale t by the same factor used to make M → R
    double scale = svd_r.singularValues().mean();
    t_out = t_raw / scale;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1b: Detect light spots across all cameras (parallel)
// ─────────────────────────────────────────────────────────────────────────────

// Per-camera detection results: frame_number → SpotDetection
// (DetectionMap typedef is above, near filter_static_detections)

#ifdef __APPLE__

// ── macOS fast path: FFmpegDemuxer + VTAsyncDecoder → BGRA CVPixelBuffer ──
// Skips swscale entirely; VT does YUV→BGRA on GPU.

inline std::vector<DetectionMap> detect_all_cameras(
    const PointSourceConfig &config,
    const std::map<std::string, std::string> &video_files,
    std::string *status,
    DetectionProgress *progress = nullptr) {

    int num_cameras = (int)config.camera_names.size();
    if (progress)
        progress->init(num_cameras);
    std::vector<DetectionMap> all_detections(num_cameras);
    std::vector<FrameCandidates> all_candidates(num_cameras); // Smart Blob v2
    std::vector<std::pair<int,int>> cam_dims(num_cameras, {0,0}); // width,height per camera
    bool collect_candidates = config.smart_blob;
    std::mutex det_mutex;

    // GPU-accelerated detection — shared across all camera threads
    PointSourceMetalHandle metal_ctx = pointsource_metal_create();
    if (!metal_ctx)
        printf("PointSource: Metal compute init failed, falling back to CPU\n");

    std::vector<std::thread> threads;

    for (int c = 0; c < num_cameras; c++) {
        auto it = video_files.find(config.camera_names[c]);
        if (it == video_files.end())
            continue;

        threads.emplace_back([&config, &all_detections, &all_candidates,
                              &cam_dims, collect_candidates, &det_mutex, c,
                              video_path = it->second, progress, metal_ctx]() {
            CameraProgress *cam_prog = progress ? progress->cameras[c].get() : nullptr;
            // Open demuxer (handles Annex-B conversion for VT)
            std::unique_ptr<FFmpegDemuxer> demuxer;
            try {
                demuxer = std::make_unique<FFmpegDemuxer>(
                    video_path.c_str(),
                    std::map<std::string, std::string>{});
            } catch (...) {
                printf("PointSource: failed to open demuxer: %s\n",
                       video_path.c_str());
                return;
            }

            int width = (int)demuxer->GetWidth();
            int height = (int)demuxer->GetHeight();
            cam_dims[c] = {width, height};

            // Init VT decoder from stream extradata
            VTAsyncDecoder vt;
            if (!vt.init(demuxer->GetExtradata(), demuxer->GetExtradataSize(),
                         demuxer->GetVideoCodec())) {
                printf("PointSource: VT init failed for %s\n", video_path.c_str());
                return;
            }

            int start_fr = config.start_frame;
            int stop_fr = config.stop_frame;
            int step = std::max(1, config.frame_step);
            if (stop_fr <= 0)
                stop_fr = INT_MAX;

            // Seek to nearest keyframe before start_frame to skip
            // decoding the (potentially many) frames before our range.
            int frame = 0;
            bool first_pkt_from_seek = false;
            uint8_t *seek_pkt = nullptr;
            size_t seek_pkt_size = 0;
            PacketData seek_pkt_info;

            if (start_fr > 0) {
                SeekContext seek_ctx((uint64_t)start_fr, PREV_KEY_FRAME, BY_NUMBER);
                if (demuxer->Seek(seek_ctx, seek_pkt, seek_pkt_size, seek_pkt_info)) {
                    // FrameNumberFromTs converts the PTS back to frame number
                    frame = (int)demuxer->FrameNumberFromTs(seek_ctx.out_frame_pts);
                    if (frame < 0) frame = 0;
                    first_pkt_from_seek = true;
                }
            }

            DetectionMap local_detections;
            FrameCandidates local_candidates; // Smart Blob v2/v3
            ArtifactMask artifact_mask;
            int artifact_filtered = 0;

            // ── Smart Blob v3: keyframe pre-scan for artifact detection ──
            // Seek to ~30 evenly-spaced keyframes throughout the video using
            // the main demuxer+decoder (avoids creating extra VT sessions).
            if (collect_candidates) {
                artifact_mask.init(width, height, 32);
                int total_frames = CalibrationPipeline::get_video_frame_count(video_path);
                int n_samples = std::min(40, std::max(20, total_frames / 300));
                int sample_step = std::max(1, total_frames / n_samples);

                for (int si = 0; si < n_samples; si++) {
                    int target = si * sample_step;
                    SeekContext seek_ctx((uint64_t)target, PREV_KEY_FRAME, BY_NUMBER);
                    uint8_t *pkt = nullptr; size_t pkt_sz = 0; PacketData pkt_info;
                    if (!demuxer->Seek(seek_ctx, pkt, pkt_sz, pkt_info)) continue;

                    bool is_key = (pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                    vt.submit_blocking(pkt, pkt_sz, pkt_info.pts,
                                       pkt_info.dts, demuxer->GetTimebase(), is_key);
                    CVPixelBufferRef pb = vt.drain_one();
                    if (!pb) continue;

                    std::vector<BlobCandidate> blobs;
                    if (metal_ctx) {
                        auto mblobs = pointsource_metal_detect_all(
                            metal_ctx, pb,
                            config.green_threshold, config.green_dominance,
                            config.min_blob_pixels, config.max_blob_pixels);
                        for (const auto &mb : mblobs)
                            blobs.push_back({(float)mb.cx, (float)mb.cy, mb.pixel_count});
                    } else {
                        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        const uint8_t *bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                        int bstride = (int)CVPixelBufferGetBytesPerRow(pb);
                        blobs = detect_all_blobs(bgra, width, height, bstride,
                                                 POINTSOURCE_FMT_BGRA,
                                                 config.green_threshold, config.green_dominance,
                                                 config.min_blob_pixels, config.max_blob_pixels);
                        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                    }
                    CFRelease(pb);
                    artifact_mask.add_frame(blobs);
                }
                artifact_mask.finalize();

                // Flush decoder and re-seek to start_frame for main detection
                vt.flush();
                frame = 0;
                first_pkt_from_seek = false;
                if (start_fr > 0) {
                    SeekContext sc((uint64_t)start_fr, PREV_KEY_FRAME, BY_NUMBER);
                    if (demuxer->Seek(sc, seek_pkt, seek_pkt_size, seek_pkt_info)) {
                        frame = (int)demuxer->FrameNumberFromTs(sc.out_frame_pts);
                        if (frame < 0) frame = 0;
                        first_pkt_from_seek = true;
                    }
                }
            }

            int detected = 0;
            int packets_submitted = 0;
            int processed = 0;

            auto process_frame = [&](CVPixelBufferRef pb) {
                // Only detect on frames within [start, stop) at step intervals
                bool should_detect = frame >= start_fr &&
                                     frame < stop_fr &&
                                     ((frame - start_fr) % step == 0);
                if (should_detect) {
                    if (collect_candidates) {
                        // Smart Blob v3: collect ALL blobs, filter with pre-built artifact mask
                        std::vector<BlobCandidate> frame_blobs;
                        if (metal_ctx) {
                            auto mblobs = pointsource_metal_detect_all(
                                metal_ctx, pb,
                                config.green_threshold, config.green_dominance,
                                config.min_blob_pixels, config.max_blob_pixels);
                            for (const auto &mb : mblobs)
                                frame_blobs.push_back({(float)mb.cx, (float)mb.cy, mb.pixel_count});
                        } else {
                            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                            const uint8_t *bgra =
                                (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                            int bstride = (int)CVPixelBufferGetBytesPerRow(pb);
                            frame_blobs = detect_all_blobs(bgra, width, height, bstride,
                                                           POINTSOURCE_FMT_BGRA,
                                                           config.green_threshold,
                                                           config.green_dominance,
                                                           config.min_blob_pixels,
                                                           config.max_blob_pixels);
                            CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        }
                        // v3: filter with keyframe-derived artifact mask
                        artifact_filtered += artifact_mask.filter(frame_blobs);
                        if (!frame_blobs.empty()) {
                            local_candidates[frame] = std::move(frame_blobs);
                            detected++;
                        }
                    } else {
                        // Original path: single detection per frame
                        SpotDetection det;
                        bool found = false;
                        if (metal_ctx) {
                            PointSourceMetalSpot mdet = pointsource_metal_detect(
                                metal_ctx, pb,
                                config.green_threshold, config.green_dominance,
                                config.min_blob_pixels, config.max_blob_pixels,
                                false); // no smart_blob in v1 path
                            if (mdet.found) {
                                det.cx = mdet.cx;
                                det.cy = mdet.cy;
                                det.pixel_count = mdet.pixel_count;
                                found = true;
                            }
                        } else {
                            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                            const uint8_t *bgra =
                                (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                            int bstride = (int)CVPixelBufferGetBytesPerRow(pb);
                            found = detect_light_spot(bgra, width, height, bstride,
                                                      POINTSOURCE_FMT_BGRA,
                                                      config.green_threshold,
                                                      config.green_dominance,
                                                      config.min_blob_pixels,
                                                      config.max_blob_pixels, det,
                                                      false);
                            CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        }
                        if (found) {
                            local_detections[frame] = det;
                            detected++;
                        }
                    }
                    processed++;
                }
                CFRelease(pb);
                frame++;
                if (cam_prog) {
                    cam_prog->frames_processed.store(processed, std::memory_order_relaxed);
                    cam_prog->spots_detected.store(detected, std::memory_order_relaxed);
                }
            };

            while (true) {
                if (frame >= stop_fr)
                    break;

                uint8_t *pkt_data = nullptr;
                size_t pkt_size = 0;
                PacketData pkt_info;

                if (first_pkt_from_seek) {
                    // Use the packet already returned by Seek()
                    pkt_data = seek_pkt;
                    pkt_size = seek_pkt_size;
                    pkt_info = seek_pkt_info;
                    first_pkt_from_seek = false;
                } else {
                    if (!demuxer->Demux(pkt_data, pkt_size, pkt_info))
                        break; // EOF
                }

                bool is_key = (pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                vt.submit_blocking(pkt_data, pkt_size, pkt_info.pts,
                                   pkt_info.dts, demuxer->GetTimebase(),
                                   is_key);
                packets_submitted++;

                // Drain decoded frames (may produce 0 or 1+ frames)
                while (CVPixelBufferRef pb = vt.drain_one()) {
                    process_frame(pb);
                    if (frame >= stop_fr)
                        break;
                }
                if (frame >= stop_fr)
                    break;
            }

            // Drain any remaining buffered frames
            while (frame < stop_fr) {
                CVPixelBufferRef pb = vt.drain_one();
                if (!pb)
                    break;
                process_frame(pb);
            }

            {
                std::lock_guard<std::mutex> lock(det_mutex);
                if (collect_candidates)
                    all_candidates[c] = std::move(local_candidates);
                else
                    all_detections[c] = std::move(local_detections);
            }

            if (cam_prog)
                cam_prog->done.store(true, std::memory_order_relaxed);

            if (collect_candidates && artifact_mask.artifact_bins_flagged > 0) {
                printf("PointSource: Camera %s — %d decoded, %d blobs, "
                       "%d artifact-filtered (%d bins, %d keyframes) [v3]\n",
                       config.camera_names[c].c_str(), frame, detected,
                       artifact_filtered, artifact_mask.artifact_bins_flagged,
                       artifact_mask.frames_seen);
            } else {
                printf("PointSource: Camera %s — %d decoded, %d %s "
                       "(range %d-%s, step %d)\n",
                       config.camera_names[c].c_str(), frame, detected,
                       collect_candidates ? "blobs" : "spots",
                       start_fr,
                       stop_fr == INT_MAX ? "end" : std::to_string(stop_fr).c_str(),
                       step);
            }
        });
    }

    for (auto &t : threads)
        t.join();

    if (metal_ctx)
        pointsource_metal_destroy(metal_ctx);

    // Smart Blob v2: resolve candidates → detections
    if (collect_candidates) {
        printf("PointSource: Smart Blob v2 — resolving blob candidates...\n");
        for (int c = 0; c < num_cameras; c++) {
            printf("  Camera %s:\n", config.camera_names[c].c_str());
            all_detections[c] = resolve_blob_candidates(
                all_candidates[c], cam_dims[c].first, cam_dims[c].second);
        }
    }

    return all_detections;
}

#else

// ── Linux fallback: FrameReader (FFmpeg + swscale → RGB24) ──

inline std::vector<DetectionMap> detect_all_cameras(
    const PointSourceConfig &config,
    const std::map<std::string, std::string> &video_files,
    std::string *status,
    DetectionProgress *progress = nullptr) {

    int num_cameras = (int)config.camera_names.size();
    if (progress)
        progress->init(num_cameras);
    std::vector<DetectionMap> all_detections(num_cameras);
    std::vector<FrameCandidates> all_candidates(num_cameras);
    std::vector<std::pair<int,int>> cam_dims(num_cameras, {0,0});
    bool collect_candidates = config.smart_blob;
    std::mutex det_mutex;

    std::vector<std::thread> threads;

    for (int c = 0; c < num_cameras; c++) {
        auto it = video_files.find(config.camera_names[c]);
        if (it == video_files.end())
            continue;

        threads.emplace_back([&config, &all_detections, &all_candidates,
                              &cam_dims, collect_candidates, &det_mutex, c,
                              video_path = it->second, progress]() {
            CameraProgress *cam_prog = progress ? progress->cameras[c].get() : nullptr;
            ffmpeg_reader::FrameReader reader;
            if (!reader.open(video_path)) {
                printf("PointSource: failed to open video: %s\n",
                       video_path.c_str());
                return;
            }

            int w = reader.width(), h = reader.height();
            cam_dims[c] = {w, h};

            int start_fr = config.start_frame;
            int stop_fr = config.stop_frame;
            int step = std::max(1, config.frame_step);
            if (stop_fr <= 0)
                stop_fr = INT_MAX;

            DetectionMap local_detections;
            FrameCandidates local_candidates;
            ArtifactMask artifact_mask;
            int artifact_filtered = 0;

            // Keyframe pre-scan for artifact detection (Linux)
            if (collect_candidates) {
                artifact_mask.init(w, h, 32);
                int total_frames = reader.frameCount();
                int n_samples = std::min(40, std::max(20, total_frames / 300));
                int sample_step = std::max(1, total_frames / n_samples);
                for (int si = 0; si < n_samples; si++) {
                    int target = si * sample_step;
                    const uint8_t *rgb_s = reader.readFrame(target);
                    if (!rgb_s) continue;
                    auto blobs = detect_all_blobs(rgb_s, w, h, w * 3,
                        POINTSOURCE_FMT_RGB24, config.green_threshold,
                        config.green_dominance, config.min_blob_pixels,
                        config.max_blob_pixels);
                    artifact_mask.add_frame(blobs);
                }
                artifact_mask.finalize();
            }

            int frame = 0;
            int detected = 0;
            int processed = 0;

            while (frame < stop_fr) {
                const uint8_t *rgb = reader.readFrame(frame);
                if (!rgb)
                    break;

                bool should_detect = frame >= start_fr &&
                                     ((frame - start_fr) % step == 0);
                if (should_detect) {
                    int stride = w * 3;
                    if (collect_candidates) {
                        auto frame_blobs = detect_all_blobs(
                            rgb, w, h, stride, POINTSOURCE_FMT_RGB24,
                            config.green_threshold, config.green_dominance,
                            config.min_blob_pixels, config.max_blob_pixels);
                        artifact_filtered += artifact_mask.filter(frame_blobs);
                        if (!frame_blobs.empty()) {
                            local_candidates[frame] = std::move(frame_blobs);
                            detected++;
                        }
                    } else {
                        SpotDetection det;
                        if (detect_light_spot(rgb, w, h, stride,
                                              POINTSOURCE_FMT_RGB24,
                                              config.green_threshold,
                                              config.green_dominance,
                                              config.min_blob_pixels,
                                              config.max_blob_pixels, det,
                                              false)) {
                            local_detections[frame] = det;
                            detected++;
                        }
                    }
                    processed++;
                }
                frame++;
                if (cam_prog) {
                    cam_prog->frames_processed.store(processed, std::memory_order_relaxed);
                    cam_prog->spots_detected.store(detected, std::memory_order_relaxed);
                }
            }

            {
                std::lock_guard<std::mutex> lock(det_mutex);
                if (collect_candidates)
                    all_candidates[c] = std::move(local_candidates);
                else
                    all_detections[c] = std::move(local_detections);
            }

            if (cam_prog)
                cam_prog->done.store(true, std::memory_order_relaxed);

            printf("PointSource: Camera %s — %d frames decoded, %d spots detected "
                   "(range %d-%s, step %d)\n",
                   config.camera_names[c].c_str(), frame, detected,
                   start_fr,
                   stop_fr == INT_MAX ? "end" : std::to_string(stop_fr).c_str(),
                   step);
        });
    }

    for (auto &t : threads)
        t.join();

    // Smart Blob v2: resolve candidates → detections
    if (collect_candidates) {
        printf("PointSource: Smart Blob v2 — resolving blob candidates...\n");
        for (int c = 0; c < num_cameras; c++) {
            printf("  Camera %s:\n", config.camera_names[c].c_str());
            all_detections[c] = resolve_blob_candidates(
                all_candidates[c], cam_dims[c].first, cam_dims[c].second);
        }
    }

    return all_detections;
}

#endif // __APPLE__

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Assemble multi-camera observations
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<FrameObservations>
assemble_observations(const std::vector<DetectionMap> &all_detections,
                      int min_cameras) {
    // Collect all frame numbers across cameras
    std::set<int> all_frames;
    for (const auto &dmap : all_detections)
        for (const auto &[f, _] : dmap)
            all_frames.insert(f);

    std::vector<FrameObservations> result;
    for (int f : all_frames) {
        FrameObservations obs;
        obs.frame_num = f;
        for (int c = 0; c < (int)all_detections.size(); c++) {
            auto it = all_detections[c].find(f);
            if (it != all_detections[c].end()) {
                obs.cam_indices.push_back(c);
                obs.pixel_coords.push_back(
                    Eigen::Vector2d(it->second.cx, it->second.cy));
            }
        }
        if ((int)obs.cam_indices.size() >= min_cameras)
            result.push_back(std::move(obs));
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-camera quality assessment for Loose Init mode
// ─────────────────────────────────────────────────────────────────────────────

struct CameraQuality {
    double median_reproj = 0.0;
    int n_frames = 0;
    bool is_good = true;
};

// Assess each camera's calibration quality by triangulating with all cameras
// and computing per-camera median reprojection error. Cameras with median
// reproj >= quality_threshold are marked as poor (is_good = false).
// Optional valid_cameras mask: when provided, only those cameras participate
// in triangulation (preventing garbage-pose cameras from corrupting 3D points).
// All cameras still get their reproj measured against the clean 3D points.
inline std::vector<CameraQuality> assess_camera_quality(
    const std::vector<FrameObservations> &frame_obs,
    const std::vector<CalibrationPipeline::CameraPose> &poses,
    double quality_threshold = 10.0,
    std::string *status = nullptr,
    const std::vector<bool> *valid_cameras = nullptr) {

    int num_cameras = (int)poses.size();

    // Build projection matrices and rvecs
    std::vector<Eigen::Matrix<double, 3, 4>> Ps(num_cameras);
    std::vector<Eigen::Vector3d> rvecs(num_cameras);
    for (int c = 0; c < num_cameras; c++) {
        Ps[c] = red_math::projectionFromKRt(poses[c].K, poses[c].R, poses[c].t);
        rvecs[c] = red_math::rotationMatrixToVector(poses[c].R);
    }

    // Collect per-camera reproj errors across all frames
    std::vector<std::vector<double>> per_cam_errors(num_cameras);

    for (const auto &fobs : frame_obs) {
        int n = (int)fobs.cam_indices.size();
        if (n < 2) continue;

        // Triangulate using only valid cameras (or all if no mask)
        std::vector<Eigen::Vector2d> tri_pts;
        std::vector<Eigen::Matrix<double, 3, 4>> tri_Ps;
        for (int i = 0; i < n; i++) {
            int c = fobs.cam_indices[i];
            if (valid_cameras && !(*valid_cameras)[c]) continue;
            tri_pts.push_back(red_math::undistortPoint(fobs.pixel_coords[i],
                                                        poses[c].K, poses[c].dist));
            tri_Ps.push_back(Ps[c]);
        }
        if ((int)tri_pts.size() < 2) continue;
        Eigen::Vector3d pt3d = red_math::triangulatePoints(tri_pts, tri_Ps);

        // Compute reproj error for ALL cameras (including invalid ones)
        for (int i = 0; i < n; i++) {
            int c = fobs.cam_indices[i];
            Eigen::Vector2d proj = red_math::projectPoint(
                pt3d, rvecs[c], poses[c].t, poses[c].K, poses[c].dist);
            double err = (proj - fobs.pixel_coords[i]).norm();
            per_cam_errors[c].push_back(err);
        }
    }

    // Compute median per camera
    std::vector<CameraQuality> quality(num_cameras);
    std::vector<double> all_medians;
    for (int c = 0; c < num_cameras; c++) {
        auto &errs = per_cam_errors[c];
        quality[c].n_frames = (int)errs.size();
        if (errs.empty()) continue;
        std::sort(errs.begin(), errs.end());
        quality[c].median_reproj = errs[errs.size() / 2];
        all_medians.push_back(quality[c].median_reproj);
    }

    // Adaptive threshold: if the user's threshold is too strict for this dataset,
    // use a relative approach — cameras in the better half are "good"
    double effective_threshold = quality_threshold;
    if (!all_medians.empty()) {
        std::sort(all_medians.begin(), all_medians.end());
        double best_median = all_medians[0];
        if (best_median >= quality_threshold) {
            // All cameras exceed the absolute threshold. Use relative:
            // "good" = within 2x of the best camera's median reproj.
            effective_threshold = std::max(quality_threshold, best_median * 2.0);
            printf("  Adaptive threshold: best camera median %.2f px > %.0f px, "
                   "using %.1f px (2x best)\n",
                   best_median, quality_threshold, effective_threshold);
        }
    }

    // Classify
    int n_poor = 0;
    for (int c = 0; c < num_cameras; c++) {
        if (per_cam_errors[c].empty()) {
            quality[c].is_good = true;
            continue;
        }
        quality[c].is_good = (quality[c].median_reproj < effective_threshold);
        if (!quality[c].is_good) n_poor++;
        printf("  Camera %d: median reproj %.2f px (%d frames) -> %s\n",
               c, quality[c].median_reproj, quality[c].n_frames,
               quality[c].is_good ? "GOOD" : "POOR");
    }

    if (status)
        *status = "Camera quality: " + std::to_string(num_cameras - n_poor) +
                  " good, " + std::to_string(n_poor) + " poor";
    return quality;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Triangulate and validate with existing calibration
// ─────────────────────────────────────────────────────────────────────────────

inline bool triangulate_and_validate(
    const std::vector<FrameObservations> &frame_obs,
    const std::vector<CalibrationPipeline::CameraPose> &poses,
    int min_cameras, double reproj_threshold,
    std::vector<Eigen::Vector3d> &points_3d,
    std::vector<std::vector<Observation>> &clean_obs_per_point,
    double &mean_reproj_error, std::string *status) {

    int num_cameras = (int)poses.size();

    // Build projection matrices and rvecs
    std::vector<Eigen::Matrix<double, 3, 4>> Ps(num_cameras);
    std::vector<Eigen::Vector3d> rvecs(num_cameras);
    for (int c = 0; c < num_cameras; c++) {
        Ps[c] =
            red_math::projectionFromKRt(poses[c].K, poses[c].R, poses[c].t);
        rvecs[c] = red_math::rotationMatrixToVector(poses[c].R);
    }

    points_3d.clear();
    clean_obs_per_point.clear();
    double total_err = 0.0;
    int total_obs_count = 0;

    for (const auto &fobs : frame_obs) {
        int n = (int)fobs.cam_indices.size();

        // Undistort pixel coords for triangulation
        std::vector<Eigen::Vector2d> undist_pts(n);
        for (int i = 0; i < n; i++) {
            int c = fobs.cam_indices[i];
            undist_pts[i] = red_math::undistortPoint(fobs.pixel_coords[i],
                                                     poses[c].K, poses[c].dist);
        }

        // Triangulate using all cameras that saw this frame
        std::vector<Eigen::Vector2d> tri_pts;
        std::vector<Eigen::Matrix<double, 3, 4>> tri_Ps;
        for (int i = 0; i < n; i++) {
            tri_pts.push_back(undist_pts[i]);
            tri_Ps.push_back(Ps[fobs.cam_indices[i]]);
        }
        Eigen::Vector3d pt3d = red_math::triangulatePoints(tri_pts, tri_Ps);

        // Compute per-camera reprojection error and filter outliers
        std::vector<int> clean_cams;
        std::vector<Eigen::Vector2d> clean_pixels;
        for (int i = 0; i < n; i++) {
            int c = fobs.cam_indices[i];
            Eigen::Vector2d proj = red_math::projectPoint(
                pt3d, rvecs[c], poses[c].t, poses[c].K, poses[c].dist);
            double err = (proj - fobs.pixel_coords[i]).norm();
            if (err <= reproj_threshold) {
                clean_cams.push_back(c);
                clean_pixels.push_back(fobs.pixel_coords[i]);
            }
        }

        if ((int)clean_cams.size() < min_cameras)
            continue;

        // Re-triangulate with clean observations if we dropped any
        if ((int)clean_cams.size() < n) {
            tri_pts.clear();
            tri_Ps.clear();
            for (int i = 0; i < (int)clean_cams.size(); i++) {
                int c = clean_cams[i];
                tri_pts.push_back(red_math::undistortPoint(
                    clean_pixels[i], poses[c].K, poses[c].dist));
                tri_Ps.push_back(Ps[c]);
            }
            pt3d = red_math::triangulatePoints(tri_pts, tri_Ps);
        }

        int point_idx = (int)points_3d.size();
        points_3d.push_back(pt3d);

        std::vector<Observation> obs_list;
        for (int i = 0; i < (int)clean_cams.size(); i++) {
            obs_list.push_back({clean_cams[i], point_idx, clean_pixels[i].x(),
                                clean_pixels[i].y()});

            Eigen::Vector2d proj = red_math::projectPoint(
                pt3d, rvecs[clean_cams[i]], poses[clean_cams[i]].t,
                poses[clean_cams[i]].K, poses[clean_cams[i]].dist);
            total_err += (proj - clean_pixels[i]).norm();
            total_obs_count++;
        }
        clean_obs_per_point.push_back(std::move(obs_list));
    }

    mean_reproj_error =
        (total_obs_count > 0) ? (total_err / total_obs_count) : 0.0;

    if (status)
        *status = "Triangulated " + std::to_string(points_3d.size()) +
                  " 3D points, " + std::to_string(total_obs_count) +
                  " observations, mean reproj: " +
                  std::to_string(mean_reproj_error).substr(0, 5) + " px";
    return !points_3d.empty();
}

// Forward declaration (definition follows after BA section below)
inline bool bundle_adjust_pointsource(
    const std::vector<std::string> &camera_names,
    std::vector<CalibrationPipeline::CameraPose> &poses,
    std::vector<Eigen::Vector3d> &points_3d,
    const std::vector<std::vector<Observation>> &obs_per_point,
    double outlier_th1, double outlier_th2, int max_iter,
    double &mean_reproj_error, std::string *status,
    int *outliers_removed_out,
    PointSourceOptMode opt_mode);

// ─────────────────────────────────────────────────────────────────────────────
// Loose Init: progressive triangulation with PnP re-initialization
// ─────────────────────────────────────────────────────────────────────────────

// Three-phase approach for handling cameras with poor initial calibration:
// Phase A: Triangulate using only good cameras → high-quality 3D points
// Phase B: PnP re-init poor cameras using good 3D points + their 2D detections
// Phase C: Re-triangulate with all cameras using normal thresholds
inline bool triangulate_and_validate_progressive(
    const std::vector<FrameObservations> &frame_obs,
    std::vector<CalibrationPipeline::CameraPose> &poses,  // mutable: poor cameras get re-initialized
    const std::vector<CameraQuality> &quality,
    int min_cameras, double reproj_threshold,
    std::vector<Eigen::Vector3d> &points_3d,
    std::vector<std::vector<Observation>> &clean_obs_per_point,
    double &mean_reproj_error, std::string *status) {

    int num_cameras = (int)poses.size();

    // Identify good and poor cameras
    std::vector<bool> is_good(num_cameras);
    int n_good = 0, n_poor = 0;
    for (int c = 0; c < num_cameras; c++) {
        is_good[c] = quality[c].is_good;
        if (is_good[c]) n_good++; else n_poor++;
    }

    if (n_good < 2) {
        if (status) *status = "Loose Init: fewer than 2 good cameras — cannot proceed";
        return false;
    }

    printf("Loose Init: %d good cameras, %d poor cameras\n", n_good, n_poor);
    if (n_poor == 0) {
        // All cameras are good — fall through to standard triangulation
        return triangulate_and_validate(frame_obs, poses, min_cameras,
                                         reproj_threshold, points_3d,
                                         clean_obs_per_point, mean_reproj_error,
                                         status);
    }

    // ── Phase A: Triangulate using only good cameras ──
    if (status) *status = "Loose Init Phase A: triangulating with good cameras only...";

    // Build projection matrices for all cameras
    std::vector<Eigen::Matrix<double, 3, 4>> Ps(num_cameras);
    std::vector<Eigen::Vector3d> rvecs(num_cameras);
    for (int c = 0; c < num_cameras; c++) {
        Ps[c] = red_math::projectionFromKRt(poses[c].K, poses[c].R, poses[c].t);
        rvecs[c] = red_math::rotationMatrixToVector(poses[c].R);
    }

    // Adaptive Phase A threshold: use the max good-camera median reproj * 3
    // (good cameras may still have large errors with default intrinsics in No Init mode)
    double phase_a_threshold = reproj_threshold;
    {
        double max_good_median = 0;
        for (int c = 0; c < num_cameras; c++)
            if (is_good[c]) max_good_median = std::max(max_good_median, quality[c].median_reproj);
        if (max_good_median > reproj_threshold)
            phase_a_threshold = max_good_median * 3.0;
    }

    // Phase A min cameras: can't require more good cameras per frame than exist
    int phase_a_min = std::min(n_good, std::max(2, min_cameras));

    printf("Loose Init Phase A: triangulating with %d good cameras "
           "(min=%d per frame, reproj threshold=%.1f px)...\n",
           n_good, phase_a_min, phase_a_threshold);

    // Filter frame_obs to only good cameras, triangulate, validate
    std::vector<Eigen::Vector3d> good_pts;
    // Map from good_pts index to frame_obs index (for Phase B lookup)
    std::vector<int> good_pts_frame_idx;

    for (int fi = 0; fi < (int)frame_obs.size(); fi++) {
        const auto &fobs = frame_obs[fi];
        int n = (int)fobs.cam_indices.size();

        // Collect only good camera observations
        std::vector<int> good_cam_idx;
        std::vector<Eigen::Vector2d> good_pixels;
        for (int i = 0; i < n; i++) {
            int c = fobs.cam_indices[i];
            if (is_good[c]) {
                good_cam_idx.push_back(c);
                good_pixels.push_back(fobs.pixel_coords[i]);
            }
        }
        if ((int)good_cam_idx.size() < phase_a_min)
            continue;

        // Triangulate with good cameras
        std::vector<Eigen::Vector2d> tri_pts;
        std::vector<Eigen::Matrix<double, 3, 4>> tri_Ps;
        for (int i = 0; i < (int)good_cam_idx.size(); i++) {
            int c = good_cam_idx[i];
            tri_pts.push_back(red_math::undistortPoint(
                good_pixels[i], poses[c].K, poses[c].dist));
            tri_Ps.push_back(Ps[c]);
        }
        Eigen::Vector3d pt3d = red_math::triangulatePoints(tri_pts, tri_Ps);

        // Validate: check reproj for good cameras (use adaptive threshold)
        bool all_ok = true;
        for (int i = 0; i < (int)good_cam_idx.size(); i++) {
            int c = good_cam_idx[i];
            Eigen::Vector2d proj = red_math::projectPoint(
                pt3d, rvecs[c], poses[c].t, poses[c].K, poses[c].dist);
            double err = (proj - good_pixels[i]).norm();
            if (err > phase_a_threshold) { all_ok = false; break; }
        }
        if (!all_ok) continue;

        good_pts.push_back(pt3d);
        good_pts_frame_idx.push_back(fi);
    }

    printf("Loose Init Phase A: %d good 3D points from good cameras\n", (int)good_pts.size());
    if (good_pts.empty()) {
        if (status) *status = "Loose Init: no valid 3D points from good cameras";
        return false;
    }

    // ── Phase B: PnP re-initialization of poor cameras ──
    if (status) *status = "Loose Init Phase B: re-initializing " +
                          std::to_string(n_poor) + " poor cameras via PnP...";

    for (int c = 0; c < num_cameras; c++) {
        if (is_good[c]) continue;

        // Collect 3D-2D correspondences: good 3D point ↔ this camera's detection
        std::vector<Eigen::Vector3d> obj_pts;
        std::vector<Eigen::Vector2d> img_pts;

        for (int gi = 0; gi < (int)good_pts.size(); gi++) {
            int fi = good_pts_frame_idx[gi];
            const auto &fobs = frame_obs[fi];
            // Find this camera's observation in this frame
            for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                if (fobs.cam_indices[i] == c) {
                    obj_pts.push_back(good_pts[gi]);
                    img_pts.push_back(fobs.pixel_coords[i]);
                    break;
                }
            }
        }

        printf("  Camera %d: %d correspondences for PnP\n", c, (int)obj_pts.size());
        if ((int)obj_pts.size() < 10) {
            printf("  Camera %d: too few correspondences, skipping PnP\n", c);
            continue;
        }

        // Use refinePnPPose with current (bad) extrinsics as initial guess
        Eigen::Matrix3d R_new = poses[c].R;
        Eigen::Vector3d t_new = poses[c].t;
        CalibrationPipeline::refinePnPPose(obj_pts, img_pts,
                                            poses[c].K, poses[c].dist,
                                            R_new, t_new);

        // Compute reproj improvement
        double err_before = 0, err_after = 0;
        Eigen::Vector3d rvec_old = red_math::rotationMatrixToVector(poses[c].R);
        Eigen::Vector3d rvec_new = red_math::rotationMatrixToVector(R_new);
        for (int i = 0; i < (int)obj_pts.size(); i++) {
            Eigen::Vector2d proj_old = red_math::projectPoint(
                obj_pts[i], rvec_old, poses[c].t, poses[c].K, poses[c].dist);
            Eigen::Vector2d proj_new = red_math::projectPoint(
                obj_pts[i], rvec_new, t_new, poses[c].K, poses[c].dist);
            err_before += (proj_old - img_pts[i]).norm();
            err_after += (proj_new - img_pts[i]).norm();
        }
        err_before /= obj_pts.size();
        err_after /= obj_pts.size();

        printf("  Camera %d: PnP reproj %.2f -> %.2f px\n", c, err_before, err_after);

        // Only accept if PnP improved AND result is reasonably good.
        // Accepting cameras with 600+ px reproj corrupts downstream mini-BA.
        if (err_after < err_before && err_after < 100.0) {
            poses[c].R = R_new;
            poses[c].t = t_new;
            printf("  Camera %d: accepted PnP re-initialization\n", c);
        } else {
            printf("  Camera %d: PnP did not improve, keeping original\n", c);
        }
    }

    // ── Phase B2: Mini-BA to refine intrinsics (especially focal length) ──
    // This is critical for No Init mode where default intrinsics (f=image_width)
    // may be ~20% off. We triangulate with all PnP-improved cameras using a very
    // loose threshold, run BA in Full mode to refine all parameters, then
    // re-assess and re-do PnP with the improved 3D points.
    {
        // Triangulate with ALL cameras that have reasonable poses (not identity)
        // using a very loose threshold to get enough observations for BA
        std::vector<Eigen::Vector3d> mini_pts;
        std::vector<std::vector<Observation>> mini_obs;

        // Determine which cameras have been PnP-initialized (not identity pose)
        // Cameras with identity/garbage poses must be EXCLUDED from mini-BA
        // to prevent them from diverging and corrupting the optimization
        std::vector<bool> has_pose(num_cameras, false);
        int n_with_pose = 0;
        for (int c = 0; c < num_cameras; c++) {
            // Camera has a valid pose if: it's good, OR its PnP-reinitialized
            // reproj is reasonable (not thousands of px from identity pose)
            bool is_identity = (poses[c].R.isIdentity(1e-6) && poses[c].t.norm() < 1.0);
            // Also check if PnP improved it to reasonable quality
            bool pnp_reasonable = !is_identity && quality[c].median_reproj < 200.0;
            has_pose[c] = is_good[c] || pnp_reasonable;
            if (has_pose[c]) n_with_pose++;
        }
        printf("Loose Init Phase B2: %d cameras with poses for mini-BA\n", n_with_pose);

        if (n_with_pose >= 2) {
            // Rebuild projection matrices with PnP-updated poses
            for (int c = 0; c < num_cameras; c++) {
                Ps[c] = red_math::projectionFromKRt(poses[c].K, poses[c].R, poses[c].t);
                rvecs[c] = red_math::rotationMatrixToVector(poses[c].R);
            }

            // Triangulate using cameras with poses, very loose threshold
            double loose_th = std::max(reproj_threshold, phase_a_threshold);
            for (const auto &fobs : frame_obs) {
                int n = (int)fobs.cam_indices.size();
                std::vector<int> active_cams;
                std::vector<Eigen::Vector2d> active_pixels;
                for (int i = 0; i < n; i++) {
                    int c = fobs.cam_indices[i];
                    if (has_pose[c]) {
                        active_cams.push_back(c);
                        active_pixels.push_back(fobs.pixel_coords[i]);
                    }
                }
                if ((int)active_cams.size() < 2) continue;

                std::vector<Eigen::Vector2d> tri_pts_l;
                std::vector<Eigen::Matrix<double, 3, 4>> tri_Ps_l;
                for (int i = 0; i < (int)active_cams.size(); i++) {
                    int c = active_cams[i];
                    tri_pts_l.push_back(red_math::undistortPoint(
                        active_pixels[i], poses[c].K, poses[c].dist));
                    tri_Ps_l.push_back(Ps[c]);
                }
                Eigen::Vector3d pt = red_math::triangulatePoints(tri_pts_l, tri_Ps_l);

                // Filter with loose threshold
                std::vector<int> clean_c;
                std::vector<Eigen::Vector2d> clean_px;
                for (int i = 0; i < (int)active_cams.size(); i++) {
                    int c = active_cams[i];
                    Eigen::Vector2d proj = red_math::projectPoint(
                        pt, rvecs[c], poses[c].t, poses[c].K, poses[c].dist);
                    if ((proj - active_pixels[i]).norm() <= loose_th) {
                        clean_c.push_back(c);
                        clean_px.push_back(active_pixels[i]);
                    }
                }
                if ((int)clean_c.size() < 2) continue;

                // Re-triangulate with clean obs
                if ((int)clean_c.size() < (int)active_cams.size()) {
                    tri_pts_l.clear(); tri_Ps_l.clear();
                    for (int i = 0; i < (int)clean_c.size(); i++) {
                        int c = clean_c[i];
                        tri_pts_l.push_back(red_math::undistortPoint(
                            clean_px[i], poses[c].K, poses[c].dist));
                        tri_Ps_l.push_back(Ps[c]);
                    }
                    pt = red_math::triangulatePoints(tri_pts_l, tri_Ps_l);
                }

                int pidx = (int)mini_pts.size();
                mini_pts.push_back(pt);
                std::vector<Observation> obs_list;
                for (int i = 0; i < (int)clean_c.size(); i++)
                    obs_list.push_back({clean_c[i], pidx, clean_px[i].x(), clean_px[i].y()});
                mini_obs.push_back(std::move(obs_list));
            }

            int mini_total_obs = 0;
            for (const auto &ol : mini_obs) mini_total_obs += (int)ol.size();
            printf("Loose Init Phase B2: %d points, %d observations for mini-BA\n",
                   (int)mini_pts.size(), mini_total_obs);

            if ((int)mini_pts.size() >= 50 && mini_total_obs >= 200) {
                if (status) *status = "Loose Init Phase B2: running mini-BA to refine intrinsics...";
                std::vector<std::string> cam_names_vec(num_cameras);
                for (int c = 0; c < num_cameras; c++)
                    cam_names_vec[c] = std::to_string(c);
                double mini_reproj_after = 0;
                int mini_outliers = 0;
                bundle_adjust_pointsource(cam_names_vec, poses, mini_pts,
                                           mini_obs, loose_th, loose_th / 2,
                                           50, mini_reproj_after, status,
                                           &mini_outliers, PointSourceOptMode::Full);
                printf("Loose Init Phase B2: mini-BA reproj -> %.3f px\n", mini_reproj_after);

                // ── Phase B3: Re-assess and re-do PnP with improved 3D points ──
                if (status) *status = "Loose Init Phase B3: re-assessing with improved parameters...";

                // Rebuild projection matrices
                for (int c = 0; c < num_cameras; c++) {
                    Ps[c] = red_math::projectionFromKRt(poses[c].K, poses[c].R, poses[c].t);
                    rvecs[c] = red_math::rotationMatrixToVector(poses[c].R);
                }

                // Re-assess quality (only triangulate with cameras that have valid poses)
                auto quality2 = assess_camera_quality(frame_obs, poses, 10.0, nullptr, &has_pose);
                int n_still_poor = 0;
                for (int c = 0; c < num_cameras; c++) {
                    is_good[c] = quality2[c].is_good;
                    if (!is_good[c]) n_still_poor++;
                }
                printf("Loose Init Phase B3: after mini-BA, %d good, %d poor cameras\n",
                       num_cameras - n_still_poor, n_still_poor);

                if (n_still_poor > 0) {
                    // Re-triangulate with good cameras (now have better intrinsics)
                    good_pts.clear();
                    good_pts_frame_idx.clear();
                    n_good = num_cameras - n_still_poor;
                    int phase_b3_min = std::min(n_good, std::max(2, min_cameras));
                    double phase_b3_th = reproj_threshold;
                    {
                        double max_gm = 0;
                        for (int c = 0; c < num_cameras; c++)
                            if (is_good[c]) max_gm = std::max(max_gm, quality2[c].median_reproj);
                        if (max_gm > reproj_threshold)
                            phase_b3_th = max_gm * 3.0;
                    }

                    for (int fi = 0; fi < (int)frame_obs.size(); fi++) {
                        const auto &fobs = frame_obs[fi];
                        std::vector<int> gc; std::vector<Eigen::Vector2d> gp;
                        for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                            int c = fobs.cam_indices[i];
                            if (is_good[c]) { gc.push_back(c); gp.push_back(fobs.pixel_coords[i]); }
                        }
                        if ((int)gc.size() < phase_b3_min) continue;
                        std::vector<Eigen::Vector2d> tp; std::vector<Eigen::Matrix<double,3,4>> tP;
                        for (int i = 0; i < (int)gc.size(); i++) {
                            tp.push_back(red_math::undistortPoint(gp[i], poses[gc[i]].K, poses[gc[i]].dist));
                            tP.push_back(Ps[gc[i]]);
                        }
                        Eigen::Vector3d pt = red_math::triangulatePoints(tp, tP);
                        bool ok = true;
                        for (int i = 0; i < (int)gc.size(); i++) {
                            double e = (red_math::projectPoint(pt, rvecs[gc[i]], poses[gc[i]].t,
                                        poses[gc[i]].K, poses[gc[i]].dist) - gp[i]).norm();
                            if (e > phase_b3_th) { ok = false; break; }
                        }
                        if (ok) { good_pts.push_back(pt); good_pts_frame_idx.push_back(fi); }
                    }
                    printf("Loose Init Phase B3: %d good 3D points for PnP re-init\n", (int)good_pts.size());

                    // PnP re-init poor cameras with improved 3D points
                    for (int c = 0; c < num_cameras; c++) {
                        if (is_good[c]) continue;
                        std::vector<Eigen::Vector3d> op; std::vector<Eigen::Vector2d> ip;
                        for (int gi = 0; gi < (int)good_pts.size(); gi++) {
                            const auto &fobs = frame_obs[good_pts_frame_idx[gi]];
                            for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                                if (fobs.cam_indices[i] == c) {
                                    op.push_back(good_pts[gi]); ip.push_back(fobs.pixel_coords[i]); break;
                                }
                            }
                        }
                        if ((int)op.size() < 10) {
                            printf("  Camera %d: %d correspondences (too few)\n", c, (int)op.size());
                            continue;
                        }
                        Eigen::Matrix3d Rn = poses[c].R; Eigen::Vector3d tn = poses[c].t;
                        CalibrationPipeline::refinePnPPose(op, ip, poses[c].K, poses[c].dist, Rn, tn);
                        Eigen::Vector3d rv_new = red_math::rotationMatrixToVector(Rn);
                        Eigen::Vector3d rv_old = red_math::rotationMatrixToVector(poses[c].R);
                        double eb = 0, ea = 0;
                        for (int i = 0; i < (int)op.size(); i++) {
                            eb += (red_math::projectPoint(op[i], rv_old, poses[c].t, poses[c].K, poses[c].dist) - ip[i]).norm();
                            ea += (red_math::projectPoint(op[i], rv_new, tn, poses[c].K, poses[c].dist) - ip[i]).norm();
                        }
                        eb /= op.size(); ea /= op.size();
                        printf("  Camera %d: PnP reproj %.2f -> %.2f px (%d corr)\n", c, eb, ea, (int)op.size());
                        if (ea < eb) { poses[c].R = Rn; poses[c].t = tn; }
                    }
                }
            }
        }
    }

    // ── Phase C: Re-triangulate with cameras that have valid calibration ──
    // If some cameras still have garbage poses (e.g., identity from No Init with
    // no board detection), exclude them from frame_obs to prevent corrupting
    // the DLT triangulation.
    int n_still_good = 0;
    for (int c = 0; c < num_cameras; c++)
        if (is_good[c]) n_still_good++;

    if (n_still_good < num_cameras) {
        // Filter frame_obs to only include good cameras
        std::vector<FrameObservations> filtered_obs;
        for (const auto &fobs : frame_obs) {
            FrameObservations fo;
            fo.frame_num = fobs.frame_num;
            for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                if (is_good[fobs.cam_indices[i]]) {
                    fo.cam_indices.push_back(fobs.cam_indices[i]);
                    fo.pixel_coords.push_back(fobs.pixel_coords[i]);
                }
            }
            // Use min(n_still_good, min_cameras) so we don't require more
            // cameras per frame than exist in the good set
            int phase_c_min = std::min(n_still_good, std::max(2, min_cameras));
            if ((int)fo.cam_indices.size() >= phase_c_min)
                filtered_obs.push_back(std::move(fo));
        }
        if (status) *status = "Loose Init Phase C: re-triangulating with " +
                              std::to_string(n_still_good) + " good cameras...";
        printf("Loose Init Phase C: re-triangulating with %d/%d good cameras (%d frames, min=%d)...\n",
               n_still_good, num_cameras, (int)filtered_obs.size(),
               std::min(n_still_good, std::max(2, min_cameras)));

        int phase_c_min = std::min(n_still_good, std::max(2, min_cameras));
        return triangulate_and_validate(filtered_obs, poses, phase_c_min,
                                         reproj_threshold, points_3d,
                                         clean_obs_per_point, mean_reproj_error,
                                         status);
    }

    if (status) *status = "Loose Init Phase C: re-triangulating with all cameras...";
    printf("Loose Init Phase C: re-triangulating with all %d cameras...\n", num_cameras);

    return triangulate_and_validate(frame_obs, poses, min_cameras,
                                     reproj_threshold, points_3d,
                                     clean_obs_per_point, mean_reproj_error,
                                     status);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Bundle adjustment (Ceres)
// ─────────────────────────────────────────────────────────────────────────────

inline bool bundle_adjust_pointsource(
    const std::vector<std::string> &camera_names,
    std::vector<CalibrationPipeline::CameraPose> &poses,
    std::vector<Eigen::Vector3d> &points_3d,
    const std::vector<std::vector<Observation>> &obs_per_point,
    double outlier_th1, double outlier_th2, int max_iter,
    double &mean_reproj_error, std::string *status,
    int *outliers_removed_out = nullptr,
    PointSourceOptMode opt_mode = PointSourceOptMode::ExtrinsicsOnly) {

    int num_cameras = (int)poses.size();

    // Pack camera parameters: 15 per camera
    // [rvec(3), tvec(3), fx, fy, cx, cy, k1, k2, p1, p2, k3]
    std::vector<std::array<double, 15>> camera_params(num_cameras);
    for (int i = 0; i < num_cameras; i++) {
        Eigen::Vector3d rvec = red_math::rotationMatrixToVector(poses[i].R);
        camera_params[i][0] = rvec.x();
        camera_params[i][1] = rvec.y();
        camera_params[i][2] = rvec.z();
        camera_params[i][3] = poses[i].t.x();
        camera_params[i][4] = poses[i].t.y();
        camera_params[i][5] = poses[i].t.z();
        camera_params[i][6] = poses[i].K(0, 0);  // fx
        camera_params[i][7] = poses[i].K(1, 1);  // fy
        camera_params[i][8] = poses[i].K(0, 2);  // cx
        camera_params[i][9] = poses[i].K(1, 2);  // cy
        camera_params[i][10] = poses[i].dist(0);  // k1
        camera_params[i][11] = poses[i].dist(1);  // k2
        camera_params[i][12] = poses[i].dist(2);  // p1
        camera_params[i][13] = poses[i].dist(3);  // p2
        camera_params[i][14] = poses[i].dist(4);  // k3
    }
    auto camera_params_init = camera_params;

    // Pack 3D point parameters
    int num_points = (int)points_3d.size();
    std::vector<std::array<double, 3>> point_params(num_points);
    for (int i = 0; i < num_points; i++)
        point_params[i] = {points_3d[i].x(), points_3d[i].y(),
                           points_3d[i].z()};

    // Flatten observations
    struct FlatObs {
        int cam_idx, point_idx;
        double px, py;
    };
    std::vector<FlatObs> observations;
    for (const auto &obs_list : obs_per_point)
        for (const auto &obs : obs_list)
            observations.push_back(
                {obs.cam_idx, obs.point_idx, obs.px, obs.py});

    if (observations.empty()) {
        if (status)
            *status = "Error: no observations for bundle adjustment";
        return false;
    }

    // Two-pass BA: first with loose threshold, then tight
    for (int pass = 0; pass < 2; pass++) {
        double outlier_th = (pass == 0) ? outlier_th1 : outlier_th2;

        if (status)
            *status = "BA pass " + std::to_string(pass + 1) + "/2 (" +
                      std::to_string(observations.size()) + " observations)...";

        ceres::Problem problem;

        for (const auto &obs : observations) {
            ceres::CostFunction *cost =
                CalibrationPipeline::ReprojectionCost::Create(obs.px, obs.py);
            ceres::LossFunction *loss = new ceres::HuberLoss(1.0);

            problem.AddResidualBlock(cost, loss,
                                     camera_params[obs.cam_idx].data(),
                                     point_params[obs.point_idx].data());
        }

        // Fix camera 0 extrinsics (rvec + tvec, indices 0-5) — gauge freedom
        // Intrinsic indices: 6=fx, 7=fy, 8=cx, 9=cy, 10=k1, 11=k2, 12=p1, 13=p2, 14=k3
        {
            std::vector<int> fixed_cam0 = {0, 1, 2, 3, 4, 5}; // always lock extrinsics of cam 0
            if (opt_mode == PointSourceOptMode::ExtrinsicsOnly) {
                for (int k = 6; k < 15; k++) fixed_cam0.push_back(k);
            } else if (opt_mode == PointSourceOptMode::ExtrinsicsAndFocal) {
                for (int k : {8, 9, 10, 11, 12, 13, 14}) fixed_cam0.push_back(k);
            } else if (opt_mode == PointSourceOptMode::ExtrinsicsAndAll) {
                for (int k : {12, 13, 14}) fixed_cam0.push_back(k);
            }
            // Full mode: only extrinsics of cam 0 are locked (all intrinsics free)
            problem.SetManifold(
                camera_params[0].data(),
                new ceres::SubsetManifold(15, fixed_cam0));
        }

        // Apply parameter locking for all other cameras based on opt_mode
        if (opt_mode == PointSourceOptMode::ExtrinsicsOnly) {
            // Lock all intrinsics (indices 6-14)
            std::vector<int> locked = {6, 7, 8, 9, 10, 11, 12, 13, 14};
            for (int i = 1; i < num_cameras; i++)
                problem.SetManifold(
                    camera_params[i].data(),
                    new ceres::SubsetManifold(15, locked));
        } else if (opt_mode == PointSourceOptMode::ExtrinsicsAndFocal) {
            // Lock cx, cy, distortion (allow fx, fy to vary)
            std::vector<int> locked = {8, 9, 10, 11, 12, 13, 14};
            for (int i = 1; i < num_cameras; i++)
                problem.SetManifold(
                    camera_params[i].data(),
                    new ceres::SubsetManifold(15, locked));
        } else if (opt_mode == PointSourceOptMode::ExtrinsicsAndAll) {
            // Lock p1, p2, k3 (same as aruco BA strategy)
            std::vector<int> locked = {12, 13, 14};
            for (int i = 1; i < num_cameras; i++)
                problem.SetManifold(
                    camera_params[i].data(),
                    new ceres::SubsetManifold(15, locked));
        }
        // Full mode: no intrinsic locking for other cameras (all 15 params free)

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = max_iter;
        options.function_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        options.minimizer_progress_to_stdout = false;
        options.num_threads =
            std::max(1, (int)std::thread::hardware_concurrency());

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        printf("PointSource BA pass %d: %s  initial_cost=%.2f  final_cost=%.2f  "
               "iterations=%d  time=%.2fs\n",
               pass + 1,
               summary.IsSolutionUsable() ? "CONVERGED" : "FAILED",
               summary.initial_cost, summary.final_cost,
               (int)summary.iterations.size(), summary.total_time_in_seconds);

        // Reject outliers after pass 1
        if (pass == 0) {
            std::vector<FlatObs> inliers;
            for (const auto &obs : observations) {
                Eigen::Vector3d rvec(camera_params[obs.cam_idx][0],
                                     camera_params[obs.cam_idx][1],
                                     camera_params[obs.cam_idx][2]);
                Eigen::Vector3d tvec(camera_params[obs.cam_idx][3],
                                     camera_params[obs.cam_idx][4],
                                     camera_params[obs.cam_idx][5]);
                Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
                K(0, 0) = camera_params[obs.cam_idx][6];
                K(1, 1) = camera_params[obs.cam_idx][7];
                K(0, 2) = camera_params[obs.cam_idx][8];
                K(1, 2) = camera_params[obs.cam_idx][9];
                Eigen::Matrix<double, 5, 1> d;
                d << camera_params[obs.cam_idx][10],
                    camera_params[obs.cam_idx][11],
                    camera_params[obs.cam_idx][12],
                    camera_params[obs.cam_idx][13],
                    camera_params[obs.cam_idx][14];

                auto &pp = point_params[obs.point_idx];
                Eigen::Vector3d pt3d(pp[0], pp[1], pp[2]);

                Eigen::Vector2d proj =
                    red_math::projectPoint(pt3d, rvec, tvec, K, d);
                double err = std::sqrt(std::pow(proj.x() - obs.px, 2) +
                                       std::pow(proj.y() - obs.py, 2));

                if (err < outlier_th)
                    inliers.push_back(obs);
            }

            int removed = (int)observations.size() - (int)inliers.size();
            observations = std::move(inliers);
            if (outliers_removed_out)
                *outliers_removed_out = removed;
            printf("PointSource BA pass 1: removed %d outliers (threshold=%.1f)\n",
                   removed, outlier_th);
        }
    }

    // Log intrinsic changes
    printf("\n=== PointSource BA Intrinsic Changes ===\n");
    printf("%-12s  %8s %8s %8s %8s  |  %8s %8s %8s %8s\n", "Camera",
           "fx_init", "fy_init", "cx_init", "cy_init", "dfx", "dfy", "dcx",
           "dcy");
    for (int i = 0; i < num_cameras; i++) {
        std::string name =
            (i < (int)camera_names.size()) ? camera_names[i] : "cam" + std::to_string(i);
        printf("%-12s  %8.1f %8.1f %8.1f %8.1f  |  %+8.2f %+8.2f %+8.2f "
               "%+8.2f\n",
               name.c_str(), camera_params_init[i][6], camera_params_init[i][7],
               camera_params_init[i][8], camera_params_init[i][9],
               camera_params[i][6] - camera_params_init[i][6],
               camera_params[i][7] - camera_params_init[i][7],
               camera_params[i][8] - camera_params_init[i][8],
               camera_params[i][9] - camera_params_init[i][9]);
    }
    printf("==================================\n\n");

    // Log extrinsic changes
    printf("=== PointSource BA Extrinsic Changes ===\n");
    printf("%-12s  %9s %9s %9s  |  %9s %9s %9s  %8s  %8s\n", "Camera",
           "tx_init", "ty_init", "tz_init", "dtx", "dty", "dtz", "|dt|", "drot(d)");
    for (int i = 0; i < num_cameras; i++) {
        std::string name =
            (i < (int)camera_names.size()) ? camera_names[i] : "cam" + std::to_string(i);
        double dtx = camera_params[i][3] - camera_params_init[i][3];
        double dty = camera_params[i][4] - camera_params_init[i][4];
        double dtz = camera_params[i][5] - camera_params_init[i][5];
        double dt_norm = std::sqrt(dtx * dtx + dty * dty + dtz * dtz);

        // Rotation change in degrees
        Eigen::Vector3d rv0(camera_params_init[i][0], camera_params_init[i][1],
                            camera_params_init[i][2]);
        Eigen::Vector3d rv1(camera_params[i][0], camera_params[i][1],
                            camera_params[i][2]);
        Eigen::Matrix3d R0 = red_math::rotationVectorToMatrix(rv0);
        Eigen::Matrix3d R1 = red_math::rotationVectorToMatrix(rv1);
        Eigen::Matrix3d dR = R0.transpose() * R1;
        double trace_val = std::min(3.0, std::max(-1.0, dR.trace()));
        double drot_deg = std::acos((trace_val - 1.0) / 2.0) * 180.0 / M_PI;

        printf("%-12s  %9.2f %9.2f %9.2f  |  %+9.3f %+9.3f %+9.3f  %8.3f  %8.4f\n",
               name.c_str(),
               camera_params_init[i][3], camera_params_init[i][4], camera_params_init[i][5],
               dtx, dty, dtz, dt_norm, drot_deg);
    }
    printf("==================================\n\n");

    // Unpack results back into poses
    for (int i = 0; i < num_cameras; i++) {
        Eigen::Vector3d rvec(camera_params[i][0], camera_params[i][1],
                             camera_params[i][2]);
        poses[i].R = red_math::rotationVectorToMatrix(rvec);
        poses[i].t = Eigen::Vector3d(camera_params[i][3], camera_params[i][4],
                                     camera_params[i][5]);
        poses[i].K = Eigen::Matrix3d::Identity();
        poses[i].K(0, 0) = camera_params[i][6];
        poses[i].K(1, 1) = camera_params[i][7];
        poses[i].K(0, 2) = camera_params[i][8];
        poses[i].K(1, 2) = camera_params[i][9];
        poses[i].dist << camera_params[i][10], camera_params[i][11],
            camera_params[i][12], camera_params[i][13], camera_params[i][14];
    }

    // Update points_3d
    for (int i = 0; i < num_points; i++)
        points_3d[i] = Eigen::Vector3d(point_params[i][0], point_params[i][1],
                                       point_params[i][2]);

    // Compute mean reproj error
    double total_err = 0.0;
    int total_obs = 0;
    for (const auto &obs : observations) {
        Eigen::Vector3d rvec(camera_params[obs.cam_idx][0],
                             camera_params[obs.cam_idx][1],
                             camera_params[obs.cam_idx][2]);
        Eigen::Vector3d tvec(camera_params[obs.cam_idx][3],
                             camera_params[obs.cam_idx][4],
                             camera_params[obs.cam_idx][5]);
        auto &pp = point_params[obs.point_idx];
        Eigen::Vector3d pt3d(pp[0], pp[1], pp[2]);

        Eigen::Vector2d proj = red_math::projectPoint(
            pt3d, rvec, tvec, poses[obs.cam_idx].K, poses[obs.cam_idx].dist);
        double err = std::sqrt(std::pow(proj.x() - obs.px, 2) +
                               std::pow(proj.y() - obs.py, 2));
        total_err += err;
        total_obs++;
    }

    mean_reproj_error = (total_obs > 0) ? (total_err / total_obs) : 0.0;

    if (status)
        *status = "BA complete. Mean reproj: " +
                  std::to_string(mean_reproj_error).substr(0, 5) + " px (" +
                  std::to_string(total_obs) + " observations)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// No Init: bootstrap camera parameters from Global Reg. Media (ChArUco board)
// ─────────────────────────────────────────────────────────────────────────────

// Bootstrap camera extrinsics (and default intrinsics) from a single ChArUco
// board image/video per camera. Each camera's pose is solved via PnP relative
// to the board's known 3D coordinates, so all cameras end up in world frame.
// Returns false if fewer than 2 cameras could be bootstrapped.
inline bool bootstrap_from_global_reg(
    const PointSourceConfig &config,
    std::vector<CalibrationPipeline::CameraPose> &poses,
    int &image_width, int &image_height,
    std::string *status) {

    namespace fs = std::filesystem;
    int num_cameras = (int)config.camera_names.size();
    poses.resize(num_cameras);

    const auto &cs = config.charuco_setup;
    aruco_detect::CharucoBoard board;
    board.squares_x = cs.w; board.squares_y = cs.h;
    board.square_length = cs.square_side_length;
    board.marker_length = cs.marker_side_length;
    board.dictionary_id = cs.dictionary;
    auto aruco_dict = aruco_detect::getDictionary(cs.dictionary);

    // Generate 3D board corner coordinates (same as generate_charuco_gt_pts but as Vector3d)
    int inner_w = cs.w - 1;
    int inner_h = cs.h - 1;
    float half_x = (inner_w - 1) * cs.square_side_length / 2.0f;
    float half_y = (inner_h - 1) * cs.square_side_length / 2.0f;
    std::vector<Eigen::Vector3d> board_pts_3d(inner_w * inner_h);
    std::vector<Eigen::Vector2d> board_pts_2d(inner_w * inner_h); // for solvePnPHomography
    for (int row = 0; row < inner_h; row++) {
        for (int col = 0; col < inner_w; col++) {
            int idx = row * inner_w + col;
            double x = half_x - col * cs.square_side_length;
            double y = -half_y + row * cs.square_side_length;
            board_pts_3d[idx] = Eigen::Vector3d(x, y, 0);
            board_pts_2d[idx] = Eigen::Vector2d(x, y);
        }
    }

    if (status) *status = "No Init: detecting ChArUco board in global reg media...";
    printf("No Init: bootstrapping %d cameras from %s (%s)\n",
           num_cameras, config.global_reg_media_folder.c_str(),
           config.global_reg_media_type.c_str());

    // Per-camera: detect board and solve PnP (parallel)
    struct CamBootstrap {
        std::map<int, Eigen::Vector2d> corners;
        int w = 0, h = 0;
        bool valid = false;
    };
    std::vector<CamBootstrap> cam_boots(num_cameras);
    std::vector<std::future<void>> futures;

    for (int ci = 0; ci < num_cameras; ci++) {
        futures.push_back(std::async(std::launch::async, [&, ci]() {
            const std::string &serial = config.camera_names[ci];
            auto &cb = cam_boots[ci];
            int w = 0, h = 0;
            std::vector<uint8_t> gray;

            if (config.global_reg_media_type == "videos") {
                auto vids = CalibrationTool::discover_aruco_videos(
                    config.global_reg_media_folder, {serial});
                if (vids.empty()) return;
                const std::string &video_path = vids.begin()->second;
#ifdef __APPLE__
                std::unique_ptr<FFmpegDemuxer> demuxer;
                try { demuxer = std::make_unique<FFmpegDemuxer>(
                    video_path.c_str(), std::map<std::string, std::string>{});
                } catch (...) { return; }
                w = (int)demuxer->GetWidth(); h = (int)demuxer->GetHeight();
                gray.resize(w * h);
                VTAsyncDecoder vt;
                if (!vt.init(demuxer->GetExtradata(), demuxer->GetExtradataSize(),
                             demuxer->GetVideoCodec())) return;
                uint8_t *pkt_data = nullptr; size_t pkt_size = 0; PacketData pkt_info;
                CVPixelBufferRef pb = nullptr;
                for (int pkt_i = 0; pkt_i < 16 && !pb; pkt_i++) {
                    if (!demuxer->Demux(pkt_data, pkt_size, pkt_info)) break;
                    bool is_key = (pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                    vt.submit_blocking(pkt_data, pkt_size, pkt_info.pts,
                                       pkt_info.dts, demuxer->GetTimebase(), is_key);
                    pb = vt.drain_one();
                }
                if (!pb) return;
                CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                const uint8_t *bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                int stride = (int)CVPixelBufferGetBytesPerRow(pb);
                for (int y = 0; y < h; y++) {
                    const uint8_t *row = bgra + y * stride;
                    for (int x = 0; x < w; x++)
                        gray[y*w+x] = (uint8_t)((row[x*4+2]*77 + row[x*4+1]*150 + row[x*4]*29) >> 8);
                }
                CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                CFRelease(pb);
#else
                ffmpeg_reader::FrameReader reader;
                if (!reader.open(video_path)) return;
                w = reader.width(); h = reader.height();
                gray.resize(w * h);
                const uint8_t *rgb = reader.readFrame(0);
                if (!rgb) return;
                for (int i = 0; i < w * h; i++)
                    gray[i] = (uint8_t)((rgb[i*3]*77 + rgb[i*3+1]*150 + rgb[i*3+2]*29) >> 8);
#endif
            } else {
                // Images: find image for this camera
                std::string found_path;
                for (const auto &entry : fs::directory_iterator(config.global_reg_media_folder)) {
                    if (!entry.is_regular_file()) continue;
                    std::string fn = entry.path().filename().string();
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
                    if (fn.find(serial) != std::string::npos) {
                        found_path = entry.path().string();
                        break;
                    }
                }
                if (found_path.empty()) return;
#ifdef __APPLE__
                tjhandle tj = tjInitDecompress();
                FILE *fp = fopen(found_path.c_str(), "rb");
                if (!fp) { tjDestroy(tj); return; }
                fseek(fp, 0, SEEK_END); long fsize = ftell(fp); fseek(fp, 0, SEEK_SET);
                std::vector<unsigned char> jpeg_buf(fsize);
                fread(jpeg_buf.data(), 1, fsize, fp); fclose(fp);
                int tj_sub, tj_cs;
                if (tjDecompressHeader3(tj, jpeg_buf.data(), fsize, &w, &h, &tj_sub, &tj_cs) != 0)
                    { tjDestroy(tj); return; }
                gray.resize(w * h);
                tjDecompress2(tj, jpeg_buf.data(), fsize, gray.data(), w, 0, h, TJPF_GRAY, TJFLAG_FASTDCT);
                tjDestroy(tj);
#else
                int channels = 0;
                unsigned char *pixels = stbi_load(found_path.c_str(), &w, &h, &channels, 1);
                if (!pixels) return;
                gray.assign(pixels, pixels + w * h);
                stbi_image_free(pixels);
#endif
            }

            // Detect ChArUco board
            auto charuco = aruco_detect::detectCharucoBoard(
                gray.data(), w, h, board, aruco_dict,
                nullptr, nullptr, nullptr, 0, 1);
            if ((int)charuco.ids.size() < 4) {
                fprintf(stderr, "[NoInit]   %s: only %d corners (need 4)\n",
                        serial.c_str(), (int)charuco.ids.size());
                return;
            }
            aruco_detect::cornerSubPix(gray.data(), w, h, charuco.corners, 3, 100, 0.001f);

            for (int j = 0; j < (int)charuco.ids.size(); j++) {
                cb.corners[charuco.ids[j]] = Eigen::Vector2d(
                    charuco.corners[j].x(), charuco.corners[j].y());
            }
            cb.w = w; cb.h = h;
            cb.valid = true;
            fprintf(stderr, "[NoInit]   %s: %d corners detected (%dx%d)\n",
                    serial.c_str(), (int)cb.corners.size(), w, h);
        }));
    }
    for (auto &f : futures) f.get();

    // PnP per camera: solve pose relative to board (= world frame)
    int n_bootstrapped = 0;
    for (int ci = 0; ci < num_cameras; ci++) {
        auto &cb = cam_boots[ci];
        if (!cb.valid) {
            printf("  Camera %s: no board detected, using identity pose\n",
                   config.camera_names[ci].c_str());
            poses[ci].K = Eigen::Matrix3d::Identity();
            poses[ci].K(0,0) = cb.w > 0 ? cb.w : 3208;
            poses[ci].K(1,1) = cb.w > 0 ? cb.w : 3208;
            poses[ci].K(0,2) = cb.w > 0 ? cb.w / 2.0 : 1604;
            poses[ci].K(1,2) = cb.h > 0 ? cb.h / 2.0 : 1100;
            poses[ci].dist.setZero();
            poses[ci].R = Eigen::Matrix3d::Identity();
            poses[ci].t.setZero();
            continue;
        }

        // Set image dimensions from first valid camera
        if (image_width == 0) {
            image_width = cb.w;
            image_height = cb.h;
        }

        // Default intrinsics: f = image_width, cx = w/2, cy = h/2
        poses[ci].K = Eigen::Matrix3d::Identity();
        poses[ci].K(0,0) = cb.w;  // fx
        poses[ci].K(1,1) = cb.w;  // fy (square pixels assumed)
        poses[ci].K(0,2) = cb.w / 2.0;  // cx
        poses[ci].K(1,2) = cb.h / 2.0;  // cy
        poses[ci].dist.setZero();

        // Build 3D-2D correspondences from detected corners
        std::vector<Eigen::Vector3d> obj_pts;
        std::vector<Eigen::Vector2d> img_pts;
        std::vector<Eigen::Vector2d> obj_pts_2d_for_H;  // for solvePnPHomography (planar)
        for (const auto &[corner_id, px] : cb.corners) {
            if (corner_id >= 0 && corner_id < (int)board_pts_3d.size()) {
                obj_pts.push_back(board_pts_3d[corner_id]);
                obj_pts_2d_for_H.push_back(board_pts_2d[corner_id]);
                img_pts.push_back(px);
            }
        }

        if ((int)obj_pts.size() < 4) continue;

        // Undistort for solvePnPHomography (dist is zero, so this is identity)
        std::vector<Eigen::Vector2d> img_pts_und(img_pts.size());
        for (int i = 0; i < (int)img_pts.size(); i++)
            img_pts_und[i] = red_math::undistortPoint(img_pts[i], poses[ci].K, poses[ci].dist);

        // Step 1: algebraic PnP (homography decomposition for planar target)
        bool pnp_ok = CalibrationPipeline::solvePnPHomography(
            obj_pts_2d_for_H, img_pts_und, poses[ci].K,
            poses[ci].R, poses[ci].t);

        if (!pnp_ok) {
            printf("  Camera %s: solvePnPHomography failed\n",
                   config.camera_names[ci].c_str());
            continue;
        }

        // Step 2: Ceres refinement of pose (with distorted observations)
        CalibrationPipeline::refinePnPPose(obj_pts, img_pts,
                                            poses[ci].K, poses[ci].dist,
                                            poses[ci].R, poses[ci].t);

        // Compute reproj error
        double total_err = 0;
        Eigen::Vector3d rvec = red_math::rotationMatrixToVector(poses[ci].R);
        for (int i = 0; i < (int)obj_pts.size(); i++) {
            Eigen::Vector2d proj = red_math::projectPoint(
                obj_pts[i], rvec, poses[ci].t, poses[ci].K, poses[ci].dist);
            total_err += (proj - img_pts[i]).norm();
        }
        double mean_err = total_err / obj_pts.size();
        printf("  Camera %s: PnP OK, %d corners, reproj %.2f px\n",
               config.camera_names[ci].c_str(), (int)obj_pts.size(), mean_err);
        n_bootstrapped++;
    }

    printf("No Init: bootstrapped %d/%d cameras\n", n_bootstrapped, num_cameras);
    if (status) *status = "No Init: bootstrapped " + std::to_string(n_bootstrapped) +
                          "/" + std::to_string(num_cameras) + " cameras from board";
    return n_bootstrapped >= 2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level pipeline
// ─────────────────────────────────────────────────────────────────────────────

inline PointSourceResult run_pointsource_refinement(const PointSourceConfig &config_in,
                                        std::string *status,
                                        DetectionProgress *progress = nullptr) {
    namespace fs = std::filesystem;
    PointSourceResult result;
    auto t_start = std::chrono::steady_clock::now();

    // No Init: force smart_blob off for initial detection. The quality assessment
    // requires clean single-blob detections to identify good cameras from rough
    // bootstrapped poses. Camera recovery will re-detect with Smart Blob where needed.
    PointSourceConfig config = config_in;
    if (config.no_init && config.smart_blob) {
        printf("PointSource: No Init mode — using conservative detection for initial pass "
               "(Smart Blob will be used during camera recovery)\n");
        config.smart_blob = false;
    }

    // Find video files
    if (status)
        *status = "Finding video files...";
    auto video_files =
        find_video_files(config.media_folder, config.camera_names);
    if (video_files.empty()) {
        result.error = "No video files found in " + config.media_folder;
        return result;
    }
    printf("PointSource: found %d video files\n", (int)video_files.size());

    // Load existing calibration (or bootstrap from Global Reg. Media in No Init mode)
    int num_cameras = (int)config.camera_names.size();
    std::vector<CalibrationPipeline::CameraPose> poses(num_cameras);
    int image_width = 0, image_height = 0;

    if (config.no_init) {
        // No Init mode: bootstrap from Global Reg. Media
        if (config.global_reg_media_folder.empty()) {
            result.error = "No Init mode requires Global Reg. Media folder";
            return result;
        }
        if (status) *status = "No Init: bootstrapping from ChArUco board...";
        if (!bootstrap_from_global_reg(config, poses, image_width, image_height, status)) {
            result.error = "No Init: failed to bootstrap cameras from Global Reg. Media";
            return result;
        }
    } else {
        if (status)
            *status = "Loading calibration from " + config.calibration_folder;
        for (int c = 0; c < num_cameras; c++) {
            std::string yaml_path = config.calibration_folder + "/Cam" +
                                    config.camera_names[c] + ".yaml";
            if (!fs::exists(yaml_path)) {
                result.error = "Missing calibration file: " + yaml_path;
                return result;
            }
            try {
                auto yaml = opencv_yaml::read(yaml_path);
                poses[c].K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
                Eigen::MatrixXd dist_mat =
                    yaml.getMatrix("distortion_coefficients");
                for (int j = 0; j < 5; j++)
                    poses[c].dist(j) = dist_mat(j, 0);
                poses[c].R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
                Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
                poses[c].t =
                    Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
                if (c == 0) {
                    image_width = yaml.getInt("image_width");
                    image_height = yaml.getInt("image_height");
                }
                printf("PointSource: loaded calibration for %s (fx=%.1f fy=%.1f)\n",
                       config.camera_names[c].c_str(), poses[c].K(0, 0),
                       poses[c].K(1, 1));
            } catch (const std::exception &e) {
                result.error = "Error reading " + yaml_path + ": " + e.what();
                return result;
            }
        }
    }

    // Step 1: Detect light spots (parallel, one thread per camera)
    {
        std::string range_str = "frames " + std::to_string(config.start_frame) +
            "-" + (config.stop_frame > 0 ? std::to_string(config.stop_frame) : "end") +
            ", step " + std::to_string(config.frame_step);
        if (status)
            *status = "Detecting light spots across " +
                      std::to_string(num_cameras) + " cameras (" +
                      range_str + ")...";
    }
    if (progress) progress->current_step.store(1, std::memory_order_relaxed);
    auto all_detections = detect_all_cameras(config, video_files, status, progress);

    // Filter static artifacts (skip when smart_blob v2 — resolve_blob_candidates already did this)
    if (progress) progress->current_step.store(2, std::memory_order_relaxed);
    if (!config.smart_blob && image_width > 0 && image_height > 0) {
        for (int c = 0; c < num_cameras; c++) {
            int before = (int)all_detections[c].size();
            int removed = filter_static_detections(all_detections[c], image_width, image_height);
            if (removed > 0) {
                printf("PointSource: static filter on %s: %d → %d detections (%d removed)\n",
                       config.camera_names[c].c_str(), before, (int)all_detections[c].size(), removed);
            }
        }
    }

    // Report per-camera detection counts
    int total_detections = 0;
    for (int c = 0; c < num_cameras; c++)
        total_detections += (int)all_detections[c].size();
    if (total_detections == 0) {
        result.error =
            "No light spots detected in any camera. Try adjusting "
            "green_threshold or green_dominance.";
        return result;
    }

    // Step 3: Assemble multi-camera observations
    if (progress) progress->current_step.store(3, std::memory_order_relaxed);
    if (status)
        *status = "Assembling multi-camera observations...";
    auto frame_obs =
        assemble_observations(all_detections, config.min_cameras);
    printf("PointSource: %d frames with >= %d cameras\n", (int)frame_obs.size(),
           config.min_cameras);

    if (frame_obs.empty()) {
        result.error =
            "No frames with >= " + std::to_string(config.min_cameras) +
            " cameras detecting a light spot. Total detections: " +
            std::to_string(total_detections);
        return result;
    }

    // Step 4: Triangulate and validate
    if (progress) progress->current_step.store(4, std::memory_order_relaxed);
    if (status)
        *status = "Triangulating " + std::to_string(frame_obs.size()) +
                  " 3D points...";
    std::vector<Eigen::Vector3d> points_3d;
    std::vector<std::vector<Observation>> obs_per_point;
    double mean_reproj_before = 0.0;

    bool use_loose = config.loose_init || config.no_init;
    if (use_loose) {
        // Loose Init: assess quality, PnP re-init poor cameras, then triangulate
        if (status) *status = "Assessing per-camera calibration quality...";

        // For No Init: exclude identity-pose cameras from triangulation during
        // quality assessment. Without this, the 5 cameras with garbage poses
        // corrupt every DLT triangulation, inflating reproj for ALL cameras.
        std::vector<bool> valid_for_quality;
        std::vector<bool> *valid_mask = nullptr;
        if (config.no_init) {
            valid_for_quality.resize(num_cameras);
            for (int c = 0; c < num_cameras; c++) {
                bool is_identity = (poses[c].R.isIdentity(1e-6) && poses[c].t.norm() < 1.0);
                valid_for_quality[c] = !is_identity;
            }
            valid_mask = &valid_for_quality;
        }
        auto quality = assess_camera_quality(frame_obs, poses, 10.0, status, valid_mask);
        if (!triangulate_and_validate_progressive(frame_obs, poses, quality,
                                                   config.min_cameras,
                                                   config.reproj_threshold,
                                                   points_3d, obs_per_point,
                                                   mean_reproj_before, status)) {
            result.error = "Triangulation failed (Loose Init) — no valid 3D points";
            return result;
        }
    } else {
        if (!triangulate_and_validate(frame_obs, poses, config.min_cameras,
                                      config.reproj_threshold, points_3d,
                                      obs_per_point, mean_reproj_before, status)) {
            result.error = "Triangulation failed — no valid 3D points";
            return result;
        }
    }
    result.mean_reproj_before = mean_reproj_before;

    int total_obs = 0;
    for (const auto &obs_list : obs_per_point)
        total_obs += (int)obs_list.size();
    result.total_observations = total_obs;
    result.valid_3d_points = (int)points_3d.size();

    printf("PointSource: %d valid 3D points, %d observations, mean reproj before "
           "BA: %.3f px\n",
           result.valid_3d_points, total_obs, mean_reproj_before);

    // Step 5: Bundle adjustment
    if (progress) progress->current_step.store(5, std::memory_order_relaxed);
    if (status)
        *status = "Running bundle adjustment (" +
                  std::to_string(result.valid_3d_points) + " points, " +
                  std::to_string(total_obs) + " observations)...";
    double mean_reproj_after = 0.0;

    // Save poses before BA for computing deltas
    auto poses_before = poses;

    int ba_outliers = 0;
    if (!bundle_adjust_pointsource(config.camera_names, poses, points_3d,
                             obs_per_point, config.ba_outlier_th1,
                             config.ba_outlier_th2, config.ba_max_iter,
                             mean_reproj_after, status, &ba_outliers,
                             config.opt_mode)) {
        result.error = "Bundle adjustment failed";
        return result;
    }
    result.mean_reproj_after = mean_reproj_after;
    result.ba_outliers_removed = ba_outliers;

    // Compute per-camera changes + counts
    {
        // Count per-camera observations
        std::vector<int> cam_obs_count(num_cameras, 0);
        for (const auto &obs_list : obs_per_point)
            for (const auto &obs : obs_list)
                cam_obs_count[obs.cam_idx]++;

        result.camera_changes.resize(num_cameras);
        for (int c = 0; c < num_cameras; c++) {
            auto &cc = result.camera_changes[c];
            cc.name = config.camera_names[c];
            cc.detections = (int)all_detections[c].size();
            cc.observations = cam_obs_count[c];

            // Intrinsic deltas
            cc.dfx = poses[c].K(0, 0) - poses_before[c].K(0, 0);
            cc.dfy = poses[c].K(1, 1) - poses_before[c].K(1, 1);
            cc.dcx = poses[c].K(0, 2) - poses_before[c].K(0, 2);
            cc.dcy = poses[c].K(1, 2) - poses_before[c].K(1, 2);

            // Translation delta
            Eigen::Vector3d dt = poses[c].t - poses_before[c].t;
            cc.dt_x = dt.x();
            cc.dt_y = dt.y();
            cc.dt_z = dt.z();
            cc.dt_norm = dt.norm();

            // Rotation delta in degrees
            Eigen::Matrix3d dR = poses_before[c].R.transpose() * poses[c].R;
            double trace_val = std::min(3.0, std::max(-1.0, dR.trace()));
            cc.drot_deg = std::acos((trace_val - 1.0) / 2.0) * 180.0 / M_PI;
        }
    }

    // Step 6: Recover missing cameras (No Init / Loose Init)
    if (progress) progress->current_step.store(6, std::memory_order_relaxed);
    // After BA with the good cameras, use the refined 3D points to PnP-initialize
    // any cameras that had 0 observations, then re-triangulate and re-run BA.
    {
        // Find cameras with 0 observations after first BA
        std::vector<int> cam_obs_count_check(num_cameras, 0);
        for (const auto &obs_list : obs_per_point)
            for (const auto &obs : obs_list)
                cam_obs_count_check[obs.cam_idx]++;

        std::vector<int> missing_cams;
        for (int c = 0; c < num_cameras; c++)
            if (cam_obs_count_check[c] == 0 && (int)all_detections[c].size() > 0)
                missing_cams.push_back(c);

        if (!missing_cams.empty()) {
            printf("\n=== Camera Recovery: %d cameras with 0 observations but %s detections ===\n",
                   (int)missing_cams.size(),
                   [&]() {
                       std::string s;
                       for (int c : missing_cams)
                           s += (s.empty() ? "" : "/") + std::to_string(all_detections[c].size());
                       return s;
                   }().c_str());

            // Camera recovery: re-detect missing cameras with Smart Blob v2
            // (artifact filtering built into resolve_blob_candidates)
            {
                bool any_changed = false;
                for (int c : missing_cams) {
                    if (status)
                        *status = "Re-detecting camera " + config.camera_names[c] + "...";

                    // Re-detect with smart_blob v2 (collects all candidates, resolves with artifact filter)
                    PointSourceConfig redet_config = config;
                    redet_config.camera_names = {config.camera_names[c]};
                    redet_config.smart_blob = true; // force v2 for recovery
                    auto redet_videos = find_video_files(redet_config.media_folder,
                                                          redet_config.camera_names);
                    if (!redet_videos.empty()) {
                        auto redet = detect_all_cameras(redet_config, redet_videos, status, nullptr);
                        if (!redet.empty() && (int)redet[0].size() > (int)all_detections[c].size()) {
                            printf("  Camera %s: re-detect %d → %d detections (Smart Blob v2)\n",
                                   config.camera_names[c].c_str(),
                                   (int)all_detections[c].size(), (int)redet[0].size());
                            all_detections[c] = std::move(redet[0]);
                            any_changed = true;
                        }
                    }

                    // Fallback static filter for non-smart-blob initial detection
                    if (!config.smart_blob) {
                        int before = (int)all_detections[c].size();
                        int removed = filter_static_detections(all_detections[c], image_width, image_height);
                        if (removed > 0) {
                            printf("  Camera %s: %d static artifacts removed\n",
                                   config.camera_names[c].c_str(), removed);
                            any_changed = true;
                        }
                    }
                }

                // Rebuild frame_obs if any camera's detections changed
                if (any_changed) {
                    frame_obs = assemble_observations(all_detections, config.min_cameras);
                    printf("  Rebuilt frame_obs: %d frames after recovery re-detection\n",
                           (int)frame_obs.size());
                }
            }

            // Build frame_num → point_index map from obs_per_point
            // Each point in points_3d came from a specific frame in frame_obs.
            // We need to find which frame produced which point. The observations
            // in obs_per_point[i] all reference point index i, and their pixel
            // coords match frame_obs entries for the same frame.
            // Build reverse: for each (cam_idx, frame_num) → point_idx
            std::map<int, int> frame_to_point; // frame_obs index → point_idx
            for (int pi = 0; pi < (int)obs_per_point.size(); pi++) {
                if (obs_per_point[pi].empty()) continue;
                // Find the frame in frame_obs that matches this observation
                const auto &first_obs = obs_per_point[pi][0];
                for (int fi = 0; fi < (int)frame_obs.size(); fi++) {
                    const auto &fobs = frame_obs[fi];
                    for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                        if (fobs.cam_indices[i] == first_obs.cam_idx &&
                            std::abs(fobs.pixel_coords[i].x() - first_obs.px) < 0.01 &&
                            std::abs(fobs.pixel_coords[i].y() - first_obs.py) < 0.01) {
                            frame_to_point[fi] = pi;
                            goto found_frame;
                        }
                    }
                }
                found_frame:;
            }
            printf("  Mapped %d/%d points to frame_obs entries\n",
                   (int)frame_to_point.size(), (int)points_3d.size());

            // Compute median intrinsics from good cameras for initial guess
            std::vector<double> good_fx, good_fy, good_cx, good_cy;
            for (int c = 0; c < num_cameras; c++) {
                if (cam_obs_count_check[c] > 0) {
                    good_fx.push_back(poses[c].K(0, 0));
                    good_fy.push_back(poses[c].K(1, 1));
                    good_cx.push_back(poses[c].K(0, 2));
                    good_cy.push_back(poses[c].K(1, 2));
                }
            }
            auto median = [](std::vector<double> &v) -> double {
                if (v.empty()) return 0.0;
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            double med_fx = median(good_fx), med_fy = median(good_fy);
            double med_cx = median(good_cx), med_cy = median(good_cy);
            printf("  Median intrinsics from good cameras: fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n",
                   med_fx, med_fy, med_cx, med_cy);

            int n_recovered = 0;
            for (int c : missing_cams) {
                // Collect 3D-2D correspondences from BA-refined points + this camera's detections
                std::vector<Eigen::Vector3d> obj_pts;
                std::vector<Eigen::Vector2d> img_pts;

                for (const auto &[fi, pi] : frame_to_point) {
                    const auto &fobs = frame_obs[fi];
                    for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                        if (fobs.cam_indices[i] == c) {
                            obj_pts.push_back(points_3d[pi]);
                            img_pts.push_back(fobs.pixel_coords[i]);
                            break;
                        }
                    }
                }

                printf("  Camera %s: %d 3D-2D correspondences\n",
                       config.camera_names[c].c_str(), (int)obj_pts.size());
                if ((int)obj_pts.size() < 20) {
                    printf("  Camera %s: too few correspondences, skipping\n",
                           config.camera_names[c].c_str());
                    continue;
                }

                // Set intrinsics from median of good cameras
                poses[c].K = Eigen::Matrix3d::Identity();
                poses[c].K(0, 0) = med_fx;
                poses[c].K(1, 1) = med_fy;
                poses[c].K(0, 2) = med_cx;
                poses[c].K(1, 2) = med_cy;
                poses[c].dist.setZero();

                // Step 1: DLT PnP — closed-form, no initialization needed
                Eigen::Matrix3d R_dlt;
                Eigen::Vector3d t_dlt;
                bool dlt_ok = solvePnPDLT(obj_pts, img_pts, poses[c].K, R_dlt, t_dlt);

                double mean_err = 1e9;
                if (dlt_ok) {
                    // Step 2: Refine DLT result with Ceres (more iterations than default)
                    CalibrationPipeline::refinePnPPose(obj_pts, img_pts,
                                                        poses[c].K, poses[c].dist,
                                                        R_dlt, t_dlt);
                    double err = 0;
                    Eigen::Vector3d rv = red_math::rotationMatrixToVector(R_dlt);
                    for (int i = 0; i < (int)obj_pts.size(); i++) {
                        Eigen::Vector2d proj = red_math::projectPoint(
                            obj_pts[i], rv, t_dlt, poses[c].K, poses[c].dist);
                        err += (proj - img_pts[i]).norm();
                    }
                    mean_err = err / obj_pts.size();
                    poses[c].R = R_dlt;
                    poses[c].t = t_dlt;
                    printf("  Camera %s: DLT+Ceres PnP reproj = %.2f px (%d pts)\n",
                           config.camera_names[c].c_str(), mean_err, (int)obj_pts.size());
                }

                // Fallback: try each good camera as initial pose for Ceres
                if (mean_err > 75.0) {
                    for (int gc = 0; gc < num_cameras; gc++) {
                        if (cam_obs_count_check[gc] == 0) continue;
                        Eigen::Matrix3d R_try = poses[gc].R;
                        Eigen::Vector3d t_try = poses[gc].t;
                        CalibrationPipeline::refinePnPPose(obj_pts, img_pts,
                                                            poses[c].K, poses[c].dist,
                                                            R_try, t_try);
                        double err = 0;
                        Eigen::Vector3d rv = red_math::rotationMatrixToVector(R_try);
                        for (int i = 0; i < (int)obj_pts.size(); i++) {
                            Eigen::Vector2d proj = red_math::projectPoint(
                                obj_pts[i], rv, t_try, poses[c].K, poses[c].dist);
                            err += (proj - img_pts[i]).norm();
                        }
                        err /= obj_pts.size();
                        if (err < mean_err) {
                            mean_err = err;
                            poses[c].R = R_try;
                            poses[c].t = t_try;
                        }
                    }
                    printf("  Camera %s: multi-pose fallback reproj = %.2f px\n",
                           config.camera_names[c].c_str(), mean_err);
                }

                if (mean_err < 75.0) {
                    n_recovered++;
                    printf("  Camera %s: RECOVERED\n", config.camera_names[c].c_str());
                } else {
                    printf("  Camera %s: PnP did not converge (%.1f px), keeping excluded\n",
                           config.camera_names[c].c_str(), mean_err);
                    // Reset to identity so it gets excluded in re-triangulation
                    poses[c].R = Eigen::Matrix3d::Identity();
                    poses[c].t.setZero();
                }
            }

            if (n_recovered > 0) {
                printf("\n=== Re-running pipeline with %d recovered cameras ===\n", n_recovered);
                if (status)
                    *status = "Re-triangulating with " + std::to_string(n_recovered) +
                              " recovered cameras...";

                // Re-triangulate with ALL cameras that have valid poses
                points_3d.clear();
                obs_per_point.clear();
                double reproj_re = 0;

                // Filter frame_obs: include cameras with good poses or recovered
                std::vector<FrameObservations> recovery_obs;
                for (const auto &fobs : frame_obs) {
                    FrameObservations fo;
                    fo.frame_num = fobs.frame_num;
                    for (int i = 0; i < (int)fobs.cam_indices.size(); i++) {
                        int c2 = fobs.cam_indices[i];
                        bool is_identity = (poses[c2].R.isIdentity(1e-6) && poses[c2].t.norm() < 1.0);
                        if (!is_identity) {
                            fo.cam_indices.push_back(c2);
                            fo.pixel_coords.push_back(fobs.pixel_coords[i]);
                        }
                    }
                    if ((int)fo.cam_indices.size() >= config.min_cameras)
                        recovery_obs.push_back(std::move(fo));
                }

                if (triangulate_and_validate(recovery_obs, poses, config.min_cameras,
                                              config.reproj_threshold, points_3d,
                                              obs_per_point, reproj_re, status)) {
                    int total_obs_re = 0;
                    for (const auto &ol : obs_per_point) total_obs_re += (int)ol.size();
                    printf("  Re-triangulation: %d points, %d observations, %.3f px reproj\n",
                           (int)points_3d.size(), total_obs_re, reproj_re);
                    result.mean_reproj_before = reproj_re;

                    // Re-run BA with all cameras
                    if (status) *status = "Re-running BA with recovered cameras...";
                    double reproj_re_after = 0;
                    int outliers_re = 0;
                    if (bundle_adjust_pointsource(
                            config.camera_names, poses, points_3d,
                            obs_per_point, config.ba_outlier_th1,
                            config.ba_outlier_th2, config.ba_max_iter,
                            reproj_re_after, status, &outliers_re,
                            config.opt_mode)) {
                        result.mean_reproj_after = reproj_re_after;
                        result.ba_outliers_removed += outliers_re;
                        printf("  Re-BA: %.4f px (%d outliers)\n", reproj_re_after, outliers_re);
                    }

                    // Update per-camera stats
                    result.valid_3d_points = (int)points_3d.size();
                    result.total_observations = 0;
                    std::vector<int> cam_obs_final(num_cameras, 0);
                    for (const auto &ol : obs_per_point)
                        for (const auto &obs : ol)
                            cam_obs_final[obs.cam_idx]++;
                    for (int c = 0; c < num_cameras; c++) {
                        result.camera_changes[c].observations = cam_obs_final[c];
                        result.total_observations += cam_obs_final[c];
                        // Update deltas from original poses_before
                        result.camera_changes[c].dfx = poses[c].K(0, 0) - poses_before[c].K(0, 0);
                        result.camera_changes[c].dfy = poses[c].K(1, 1) - poses_before[c].K(1, 1);
                        result.camera_changes[c].dcx = poses[c].K(0, 2) - poses_before[c].K(0, 2);
                        result.camera_changes[c].dcy = poses[c].K(1, 2) - poses_before[c].K(1, 2);
                        Eigen::Vector3d dt = poses[c].t - poses_before[c].t;
                        result.camera_changes[c].dt_x = dt.x();
                        result.camera_changes[c].dt_y = dt.y();
                        result.camera_changes[c].dt_z = dt.z();
                        result.camera_changes[c].dt_norm = dt.norm();
                        Eigen::Matrix3d dR = poses_before[c].R.transpose() * poses[c].R;
                        double trace_val = std::min(3.0, std::max(-1.0, dR.trace()));
                        result.camera_changes[c].drot_deg =
                            std::acos((trace_val - 1.0) / 2.0) * 180.0 / M_PI;
                    }
                }
            }
        }
    }

    // Step 7: Global registration (optional Procrustes alignment to world frame)
    if (progress) progress->current_step.store(7, std::memory_order_relaxed);
    // Skip if No Init — cameras are already in world frame from PnP against the board
    if (config.do_global_reg && !config.global_reg_media_folder.empty() && !config.no_init) {
        if (status)
            *status = "Running global registration (Procrustes alignment)...";

        // Build a CalibConfig with the fields global_registration() needs
        CalibrationTool::CalibConfig greg_config;
        greg_config.charuco_setup = config.charuco_setup;
        greg_config.gt_pts = config.gt_pts;
        greg_config.world_coordinate_imgs = config.world_coordinate_imgs;
        greg_config.world_frame_rotation = config.world_frame_rotation;

        // global_registration() expects map<int, Vector3d> for points_3d
        std::map<int, Eigen::Vector3d> pts_map;
        for (int i = 0; i < (int)points_3d.size(); i++)
            pts_map[i] = points_3d[i];

        std::string greg_status;
        bool greg_ok = CalibrationPipeline::global_registration(
            greg_config, poses, pts_map, &greg_status,
            nullptr,
            config.global_reg_media_folder,
            config.global_reg_media_type,
            &config.camera_names);

        if (greg_ok) {
            result.global_reg_status = "Global registration succeeded. " + greg_status;
            // Copy transformed points back to vector
            for (int i = 0; i < (int)points_3d.size(); i++)
                points_3d[i] = pts_map[i];
            printf("PointSource: global registration succeeded: %s\n", greg_status.c_str());
        } else {
            result.global_reg_status = "Global registration failed: " + greg_status;
            printf("PointSource: global registration failed: %s\n", greg_status.c_str());
        }
    }

    // Step 8: Write refined calibration to timestamped subfolder
    std::string base_folder = config.output_folder;
    if (base_folder.empty())
        base_folder = config.calibration_folder + "_pointsource_refined";

    // Step 8: Write output
    if (progress) progress->current_step.store(8, std::memory_order_relaxed);

    // Create timestamped subfolder
    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y_%m_%d_%H_%M_%S", &tstruct);
    std::string out_folder = base_folder + "/" + tbuf;
    std::string summary_dir = out_folder + "/summary_data";
    fs::create_directories(out_folder);
    fs::create_directories(summary_dir);

    if (status)
        *status = "Writing refined calibration to " + out_folder;
    std::string write_status;
    if (!CalibrationPipeline::write_calibration(poses, config.camera_names,
                                                out_folder, image_width,
                                                image_height, &write_status)) {
        result.error = write_status;
        return result;
    }

    // Write summary_data/settings.json
    {
        nlohmann::json settings = config_to_json(config);
        std::string settings_path = summary_dir + "/settings.json";
        std::ofstream ofs(settings_path, std::ios::binary);
        if (ofs)
            ofs << settings.dump(2);
    }

    // Write summary_data/summary.json — overall stats and per-camera changes
    {
        nlohmann::json j;
        j["mean_reproj_before"] = result.mean_reproj_before;
        j["mean_reproj_after"] = result.mean_reproj_after;
        j["valid_3d_points"] = result.valid_3d_points;
        j["total_observations"] = result.total_observations;
        j["ba_outliers_removed"] = result.ba_outliers_removed;
        j["num_cameras"] = num_cameras;
        j["image_width"] = image_width;
        j["image_height"] = image_height;

        nlohmann::json cams_j = nlohmann::json::array();
        for (int c = 0; c < num_cameras; c++) {
            const auto &cc = result.camera_changes[c];
            nlohmann::json cam_j;
            cam_j["name"] = cc.name;
            cam_j["detections"] = cc.detections;
            cam_j["observations"] = cc.observations;
            cam_j["dfx"] = cc.dfx;
            cam_j["dfy"] = cc.dfy;
            cam_j["dcx"] = cc.dcx;
            cam_j["dcy"] = cc.dcy;
            cam_j["dt_x"] = cc.dt_x;
            cam_j["dt_y"] = cc.dt_y;
            cam_j["dt_z"] = cc.dt_z;
            cam_j["dt_norm"] = cc.dt_norm;
            cam_j["drot_deg"] = cc.drot_deg;
            cams_j.push_back(cam_j);
        }
        j["camera_changes"] = cams_j;

        std::ofstream ofs(summary_dir + "/summary.json", std::ios::binary);
        if (ofs)
            ofs << j.dump(2);
    }

    // Write summary_data/ba_points.json — triangulated 3D points
    {
        nlohmann::json j = nlohmann::json::array();
        for (int i = 0; i < (int)points_3d.size(); i++) {
            j.push_back({points_3d[i].x(), points_3d[i].y(),
                          points_3d[i].z()});
        }
        std::ofstream ofs(summary_dir + "/ba_points.json", std::ios::binary);
        if (ofs)
            ofs << j.dump(2);
    }

    // Write summary_data/observations.json — per-point camera observations
    {
        nlohmann::json j = nlohmann::json::array();
        for (int i = 0; i < (int)obs_per_point.size(); i++) {
            nlohmann::json pt_j = nlohmann::json::array();
            for (const auto &obs : obs_per_point[i]) {
                pt_j.push_back({{"cam", obs.cam_idx},
                                {"px", {obs.px, obs.py}}});
            }
            j.push_back(pt_j);
        }
        std::ofstream ofs(summary_dir + "/observations.json",
                          std::ios::binary);
        if (ofs)
            ofs << j.dump(2);
    }

    result.output_folder = out_folder;
    result.success = true;

    auto t_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(t_end - t_start).count();

    printf("\n=== PointSource Calibration Refinement Summary ===\n");
    printf("Frame range:        %d - %s (step %d)\n",
           config.start_frame,
           config.stop_frame > 0 ? std::to_string(config.stop_frame).c_str() : "end",
           config.frame_step);
    printf("Valid 3D points:    %d\n", result.valid_3d_points);
    printf("Total observations: %d\n", result.total_observations);
    printf("Mean reproj before: %.3f px\n", result.mean_reproj_before);
    printf("Mean reproj after:  %.3f px\n", result.mean_reproj_after);
    printf("Output folder:      %s\n", result.output_folder.c_str());
    printf("Elapsed:            %.1f s\n", elapsed);
    printf("=============================================\n\n");

    if (status)
        *status = "PointSource calibration complete! Reproj: " +
                  std::to_string(mean_reproj_before).substr(0, 5) + " -> " +
                  std::to_string(mean_reproj_after).substr(0, 5) +
                  " px. Output: " + out_folder;

    return result;
}

} // namespace PointSourceCalibration
