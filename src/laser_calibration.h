#pragma once
// laser_calibration.h — Laser spot calibration refinement pipeline.
// Uses synchronized video of a green laser pointer in a dark arena to
// refine camera calibration (intrinsics + extrinsics) via bundle adjustment.
// Reference: github.com/JohnsonLabJanelia/laserCalib (rj branch)

#include "calibration_pipeline.h" // ReprojectionCost, CameraPose, write_calibration
#include "opencv_yaml_io.h"       // opencv_yaml::read
#include "red_math.h"             // triangulatePoints, projectPoint, etc.

#ifdef __APPLE__
#include "FFmpegDemuxer.h"        // Annex-B demuxing for VT decode
#include "vt_async_decoder.h"     // HW decode → BGRA CVPixelBuffer (zero swscale)
#include "laser_metal.h"          // GPU-accelerated laser spot detection
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

namespace LaserCalibration {

// ─────────────────────────────────────────────────────────────────────────────
// Project persistence (mirrors CalibrationTool::CalibProject pattern)
// ─────────────────────────────────────────────────────────────────────────────

struct LaserCalibProject {
    std::string project_name;
    std::string project_path;      // project_root_path / project_name
    std::string project_root_path;
    std::string media_folder;      // folder with laser videos
    std::string calibration_folder; // folder with CamXXXX.yaml
    std::vector<std::string> camera_names; // validated intersection
    std::string output_folder;     // refined YAML output
};

inline void to_json(nlohmann::json &j, const LaserCalibProject &p) {
    j = nlohmann::json{{"type", "laser_calibration"},
                       {"project_name", p.project_name},
                       {"project_path", p.project_path},
                       {"project_root_path", p.project_root_path},
                       {"media_folder", p.media_folder},
                       {"calibration_folder", p.calibration_folder},
                       {"camera_names", p.camera_names},
                       {"output_folder", p.output_folder}};
}

inline void from_json(const nlohmann::json &j, LaserCalibProject &p) {
    p.project_name = j.value("project_name", std::string{});
    p.project_path = j.value("project_path", std::string{});
    p.project_root_path = j.value("project_root_path", std::string{});
    p.media_folder = j.value("media_folder", std::string{});
    p.calibration_folder = j.value("calibration_folder", std::string{});
    p.camera_names = j.value("camera_names", std::vector<std::string>{});
    p.output_folder = j.value("output_folder", std::string{});
}

inline bool save_laser_project(const LaserCalibProject &p,
                                const std::string &file,
                                std::string *err = nullptr) {
    try {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(fs::path(file).parent_path(), ec);
        if (ec) {
            if (err) *err = ec.message();
            return false;
        }
        std::ofstream ofs(file, std::ios::binary);
        if (!ofs) {
            if (err) *err = "Cannot open: " + file;
            return false;
        }
        nlohmann::json j = p;
        ofs << j.dump(2);
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    }
}

inline bool load_laser_project(LaserCalibProject *out,
                                const std::string &file,
                                std::string *err = nullptr) {
    try {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) {
            if (err) *err = "Cannot open: " + file;
            return false;
        }
        nlohmann::json j;
        ifs >> j;
        // Verify it's a laser calibration project
        if (j.value("type", std::string{}) != "laser_calibration") {
            if (err) *err = "Not a laser calibration project";
            return false;
        }
        *out = j.get<LaserCalibProject>();
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct LaserConfig {
    std::string media_folder;
    std::vector<std::string> camera_names;
    std::string calibration_folder;
    std::string output_folder;

    // Detection
    int green_threshold = 40;
    int green_dominance = 5;
    int min_blob_pixels = 10;
    int max_blob_pixels = 600;

    // Frame range
    int start_frame = 0;    // first frame to process
    int stop_frame = 0;     // last frame (0 = process all)
    int frame_step = 1;     // process every Nth frame (1 = every frame)

    // Filtering
    int min_cameras = 4;
    double reproj_threshold = 15.0;

    // BA
    double ba_outlier_th1 = 20.0;
    double ba_outlier_th2 = 5.0;
    int ba_max_iter = 50;
    bool lock_intrinsics = true;
};

inline nlohmann::json config_to_json(const LaserConfig &c) {
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
        {"lock_intrinsics", c.lock_intrinsics}};
}

struct SpotDetection {
    double cx, cy;
    int pixel_count;
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
    int detections = 0;     // laser spots found
    int observations = 0;   // observations used in BA (after filtering)
};

struct LaserResult {
    bool success = false;
    std::string error;
    int total_frames_scanned = 0;
    int valid_3d_points = 0;
    int total_observations = 0;
    int ba_outliers_removed = 0;
    double mean_reproj_before = 0.0;
    double mean_reproj_after = 0.0;
    std::string output_folder;
    std::vector<CameraChange> camera_changes;
};

// Per-camera progress counters, shared between detection threads and UI
struct CameraProgress {
    std::atomic<int> frames_processed{0};
    std::atomic<int> spots_detected{0};
    std::atomic<bool> done{false};
};

struct DetectionProgress {
    std::vector<std::unique_ptr<CameraProgress>> cameras;
    void init(int num_cameras) {
        cameras.clear();
        cameras.reserve(num_cameras);
        for (int i = 0; i < num_cameras; i++)
            cameras.push_back(std::make_unique<CameraProgress>());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Get estimated frame count from a video file.
inline int get_video_frame_count(const std::string &path) {
    AVFormatContext *ctx = nullptr;
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

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Detect laser spot in a single frame
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

// Pixel format for detect_laser_spot.
// RGB24: R=0, G=1, B=2 (3 bpp, stride = width*3)
// BGRA:  B=0, G=1, R=2, A=3 (4 bpp, stride from CVPixelBuffer)
enum LaserPixelFormat { LASER_FMT_RGB24, LASER_FMT_BGRA };

// Detect a single green laser spot in a frame (RGB24 or BGRA).
// stride = bytes per row (may include padding for BGRA from CVPixelBuffer).
// Returns true if exactly one valid blob found; fills detection.
inline bool detect_laser_spot(const uint8_t *pixels, int width, int height,
                              int stride, LaserPixelFormat fmt,
                              int green_threshold, int green_dominance,
                              int min_blob_pixels, int max_blob_pixels,
                              SpotDetection &det) {
    int npixels = width * height;
    int bpp = (fmt == LASER_FMT_BGRA) ? 4 : 3;
    int r_off = (fmt == LASER_FMT_BGRA) ? 2 : 0;
    // g_off = 1 for both formats
    int b_off = (fmt == LASER_FMT_BGRA) ? 0 : 2;

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
            if (!mask[idx])
                continue;
            bool all = true;
            for (int dy = -1; dy <= 1 && all; dy++)
                for (int dx = -1; dx <= 1 && all; dx++)
                    if (!mask[(y + dy) * width + (x + dx)])
                        all = false;
            if (all)
                eroded[idx] = 1;
        }
    }

    // Step 3: Dilate (3x3) — pixel becomes 1 if any neighbor is 1
    std::vector<uint8_t> dilated(npixels, 0);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            bool has_neighbor = false;
            for (int dy = -1; dy <= 1 && !has_neighbor; dy++)
                for (int dx = -1; dx <= 1 && !has_neighbor; dx++)
                    if (eroded[(y + dy) * width + (x + dx)])
                        has_neighbor = true;
            if (has_neighbor)
                dilated[y * width + x] = 1;
        }
    }

    // Step 4: Connected components (4-connectivity) via union-find
    UnionFind uf(npixels);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (!dilated[idx])
                continue;
            if (x > 0 && dilated[idx - 1])
                uf.unite(idx, idx - 1);
            if (y > 0 && dilated[idx - width])
                uf.unite(idx, idx - width);
        }
    }

    // Step 5: Filter by area
    std::map<int, int> component_sizes;
    for (int i = 0; i < npixels; i++) {
        if (dilated[i])
            component_sizes[uf.find(i)] = uf.sz[uf.find(i)];
    }

    std::vector<int> valid_roots;
    for (auto &[root, s] : component_sizes) {
        if (s >= min_blob_pixels && s <= max_blob_pixels)
            valid_roots.push_back(root);
    }

    // Step 6: Decision — exactly 1 valid blob → intensity-weighted centroid
    if (valid_roots.size() != 1)
        return false;

    int root = valid_roots[0];
    double sum_x = 0, sum_y = 0, sum_w = 0;
    int count = 0;
    for (int i = 0; i < npixels; i++) {
        if (dilated[i] && uf.find(i) == root) {
            int px = i % width;
            int py = i / width;
            double w = pixels[py * stride + px * bpp + 1]; // green channel
            sum_x += px * w;
            sum_y += py * w;
            sum_w += w;
            count++;
        }
    }

    if (sum_w < 1e-9)
        return false;

    det.cx = sum_x / sum_w;
    det.cy = sum_y / sum_w;
    det.pixel_count = count;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1b: Detect laser spots across all cameras (parallel)
// ─────────────────────────────────────────────────────────────────────────────

// Per-camera detection results: frame_number → SpotDetection
using DetectionMap = std::map<int, SpotDetection>;

#ifdef __APPLE__

// ── macOS fast path: FFmpegDemuxer + VTAsyncDecoder → BGRA CVPixelBuffer ──
// Skips swscale entirely; VT does YUV→BGRA on GPU.

inline std::vector<DetectionMap> detect_all_cameras(
    const LaserConfig &config,
    const std::map<std::string, std::string> &video_files,
    std::string *status,
    DetectionProgress *progress = nullptr) {

    int num_cameras = (int)config.camera_names.size();
    if (progress)
        progress->init(num_cameras);
    std::vector<DetectionMap> all_detections(num_cameras);
    std::mutex det_mutex;

    // GPU-accelerated detection — shared across all camera threads
    LaserMetalHandle metal_ctx = laser_metal_create();
    if (!metal_ctx)
        printf("Laser: Metal compute init failed, falling back to CPU\n");

    std::vector<std::thread> threads;

    for (int c = 0; c < num_cameras; c++) {
        auto it = video_files.find(config.camera_names[c]);
        if (it == video_files.end())
            continue;

        threads.emplace_back([&config, &all_detections, &det_mutex, c,
                              video_path = it->second, progress, metal_ctx]() {
            CameraProgress *cam_prog = progress ? progress->cameras[c].get() : nullptr;
            // Open demuxer (handles Annex-B conversion for VT)
            std::unique_ptr<FFmpegDemuxer> demuxer;
            try {
                demuxer = std::make_unique<FFmpegDemuxer>(
                    video_path.c_str(),
                    std::map<std::string, std::string>{});
            } catch (...) {
                printf("Laser: failed to open demuxer: %s\n",
                       video_path.c_str());
                return;
            }

            int width = (int)demuxer->GetWidth();
            int height = (int)demuxer->GetHeight();

            // Init VT decoder from stream extradata
            VTAsyncDecoder vt;
            if (!vt.init(demuxer->GetExtradata(), demuxer->GetExtradataSize(),
                         demuxer->GetVideoCodec())) {
                printf("Laser: VT init failed for %s\n", video_path.c_str());
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
            int detected = 0;
            int packets_submitted = 0;

            int processed = 0;

            auto process_frame = [&](CVPixelBufferRef pb) {
                // Only detect on frames within [start, stop) at step intervals
                bool should_detect = frame >= start_fr &&
                                     frame < stop_fr &&
                                     ((frame - start_fr) % step == 0);
                if (should_detect) {
                    SpotDetection det;
                    bool found = false;
                    if (metal_ctx) {
                        // GPU path — process CVPixelBuffer directly (zero-copy)
                        LaserMetalSpot mdet = laser_metal_detect(
                            metal_ctx, pb,
                            config.green_threshold, config.green_dominance,
                            config.min_blob_pixels, config.max_blob_pixels);
                        if (mdet.found) {
                            det.cx = mdet.cx;
                            det.cy = mdet.cy;
                            det.pixel_count = mdet.pixel_count;
                            found = true;
                        }
                    } else {
                        // CPU fallback
                        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        const uint8_t *bgra =
                            (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                        int stride = (int)CVPixelBufferGetBytesPerRow(pb);
                        found = detect_laser_spot(bgra, width, height, stride,
                                                  LASER_FMT_BGRA,
                                                  config.green_threshold,
                                                  config.green_dominance,
                                                  config.min_blob_pixels,
                                                  config.max_blob_pixels, det);
                        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                    }
                    if (found) {
                        local_detections[frame] = det;
                        detected++;
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
                all_detections[c] = std::move(local_detections);
            }

            if (cam_prog)
                cam_prog->done.store(true, std::memory_order_relaxed);

            printf("Laser: Camera %s — %d frames decoded, %d spots detected "
                   "(range %d-%s, step %d) [VT+BGRA]\n",
                   config.camera_names[c].c_str(), frame, detected,
                   start_fr,
                   stop_fr == INT_MAX ? "end" : std::to_string(stop_fr).c_str(),
                   step);
        });
    }

    for (auto &t : threads)
        t.join();

    if (metal_ctx)
        laser_metal_destroy(metal_ctx);

    return all_detections;
}

#else

// ── Linux fallback: FrameReader (FFmpeg + swscale → RGB24) ──

inline std::vector<DetectionMap> detect_all_cameras(
    const LaserConfig &config,
    const std::map<std::string, std::string> &video_files,
    std::string *status,
    DetectionProgress *progress = nullptr) {

    int num_cameras = (int)config.camera_names.size();
    if (progress)
        progress->init(num_cameras);
    std::vector<DetectionMap> all_detections(num_cameras);
    std::mutex det_mutex;

    std::vector<std::thread> threads;

    for (int c = 0; c < num_cameras; c++) {
        auto it = video_files.find(config.camera_names[c]);
        if (it == video_files.end())
            continue;

        threads.emplace_back([&config, &all_detections, &det_mutex, c,
                              video_path = it->second, progress]() {
            CameraProgress *cam_prog = progress ? progress->cameras[c].get() : nullptr;
            ffmpeg_reader::FrameReader reader;
            if (!reader.open(video_path)) {
                printf("Laser: failed to open video: %s\n",
                       video_path.c_str());
                return;
            }

            int start_fr = config.start_frame;
            int stop_fr = config.stop_frame;
            int step = std::max(1, config.frame_step);
            if (stop_fr <= 0)
                stop_fr = INT_MAX;

            DetectionMap local_detections;
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
                    int stride = reader.width() * 3;
                    SpotDetection det;
                    if (detect_laser_spot(rgb, reader.width(), reader.height(),
                                          stride, LASER_FMT_RGB24,
                                          config.green_threshold,
                                          config.green_dominance,
                                          config.min_blob_pixels,
                                          config.max_blob_pixels, det)) {
                        local_detections[frame] = det;
                        detected++;
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
                all_detections[c] = std::move(local_detections);
            }

            if (cam_prog)
                cam_prog->done.store(true, std::memory_order_relaxed);

            printf("Laser: Camera %s — %d frames decoded, %d spots detected "
                   "(range %d-%s, step %d)\n",
                   config.camera_names[c].c_str(), frame, detected,
                   start_fr,
                   stop_fr == INT_MAX ? "end" : std::to_string(stop_fr).c_str(),
                   step);
        });
    }

    for (auto &t : threads)
        t.join();

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

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Bundle adjustment (Ceres)
// ─────────────────────────────────────────────────────────────────────────────

inline bool bundle_adjust_laser(
    const std::vector<std::string> &camera_names,
    std::vector<CalibrationPipeline::CameraPose> &poses,
    std::vector<Eigen::Vector3d> &points_3d,
    const std::vector<std::vector<Observation>> &obs_per_point,
    double outlier_th1, double outlier_th2, int max_iter,
    double &mean_reproj_error, std::string *status,
    int *outliers_removed_out = nullptr,
    bool lock_intrinsics = false) {

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
        // When lock_intrinsics: also fix indices 6-14 (fx,fy,cx,cy,k1-k3,p1,p2)
        {
            std::vector<int> fixed_cam0 = {0, 1, 2, 3, 4, 5};
            if (lock_intrinsics)
                for (int k = 6; k < 15; k++)
                    fixed_cam0.push_back(k);
            problem.SetManifold(
                camera_params[0].data(),
                new ceres::SubsetManifold(15, fixed_cam0));
        }

        // Lock intrinsics for all other cameras
        if (lock_intrinsics) {
            std::vector<int> intrinsic_indices = {6, 7, 8, 9, 10, 11, 12, 13, 14};
            for (int i = 1; i < num_cameras; i++)
                problem.SetManifold(
                    camera_params[i].data(),
                    new ceres::SubsetManifold(15, intrinsic_indices));
        }

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

        printf("Laser BA pass %d: %s  initial_cost=%.2f  final_cost=%.2f  "
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
            printf("Laser BA pass 1: removed %d outliers (threshold=%.1f)\n",
                   removed, outlier_th);
        }
    }

    // Log intrinsic changes
    printf("\n=== Laser BA Intrinsic Changes ===\n");
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
    printf("=== Laser BA Extrinsic Changes ===\n");
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
// Top-level pipeline
// ─────────────────────────────────────────────────────────────────────────────

inline LaserResult run_laser_refinement(const LaserConfig &config,
                                        std::string *status,
                                        DetectionProgress *progress = nullptr) {
    namespace fs = std::filesystem;
    LaserResult result;
    auto t_start = std::chrono::steady_clock::now();

    // Find video files
    if (status)
        *status = "Finding video files...";
    auto video_files =
        find_video_files(config.media_folder, config.camera_names);
    if (video_files.empty()) {
        result.error = "No video files found in " + config.media_folder;
        return result;
    }
    printf("Laser: found %d video files\n", (int)video_files.size());

    // Load existing calibration
    if (status)
        *status = "Loading calibration from " + config.calibration_folder;
    int num_cameras = (int)config.camera_names.size();
    std::vector<CalibrationPipeline::CameraPose> poses(num_cameras);
    int image_width = 0, image_height = 0;

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
            printf("Laser: loaded calibration for %s (fx=%.1f fy=%.1f)\n",
                   config.camera_names[c].c_str(), poses[c].K(0, 0),
                   poses[c].K(1, 1));
        } catch (const std::exception &e) {
            result.error = "Error reading " + yaml_path + ": " + e.what();
            return result;
        }
    }

    // Step 1: Detect laser spots (parallel, one thread per camera)
    {
        std::string range_str = "frames " + std::to_string(config.start_frame) +
            "-" + (config.stop_frame > 0 ? std::to_string(config.stop_frame) : "end") +
            ", step " + std::to_string(config.frame_step);
        if (status)
            *status = "Detecting laser spots across " +
                      std::to_string(num_cameras) + " cameras (" +
                      range_str + ")...";
    }
    auto all_detections = detect_all_cameras(config, video_files, status, progress);

    // Report per-camera detection counts
    int total_detections = 0;
    for (int c = 0; c < num_cameras; c++)
        total_detections += (int)all_detections[c].size();
    result.total_frames_scanned = 0; // set below from actual decode counts

    if (total_detections == 0) {
        result.error =
            "No laser spots detected in any camera. Try adjusting "
            "green_threshold or green_dominance.";
        return result;
    }

    // Step 2: Assemble multi-camera observations
    if (status)
        *status = "Assembling multi-camera observations...";
    auto frame_obs =
        assemble_observations(all_detections, config.min_cameras);
    printf("Laser: %d frames with >= %d cameras\n", (int)frame_obs.size(),
           config.min_cameras);

    if (frame_obs.empty()) {
        result.error =
            "No frames with >= " + std::to_string(config.min_cameras) +
            " cameras detecting a laser spot. Total detections: " +
            std::to_string(total_detections);
        return result;
    }

    // Step 3: Triangulate and validate
    if (status)
        *status = "Triangulating " + std::to_string(frame_obs.size()) +
                  " 3D points...";
    std::vector<Eigen::Vector3d> points_3d;
    std::vector<std::vector<Observation>> obs_per_point;
    double mean_reproj_before = 0.0;

    if (!triangulate_and_validate(frame_obs, poses, config.min_cameras,
                                  config.reproj_threshold, points_3d,
                                  obs_per_point, mean_reproj_before, status)) {
        result.error = "Triangulation failed — no valid 3D points";
        return result;
    }
    result.mean_reproj_before = mean_reproj_before;

    int total_obs = 0;
    for (const auto &obs_list : obs_per_point)
        total_obs += (int)obs_list.size();
    result.total_observations = total_obs;
    result.valid_3d_points = (int)points_3d.size();

    printf("Laser: %d valid 3D points, %d observations, mean reproj before "
           "BA: %.3f px\n",
           result.valid_3d_points, total_obs, mean_reproj_before);

    // Step 4: Bundle adjustment
    if (status)
        *status = "Running bundle adjustment (" +
                  std::to_string(result.valid_3d_points) + " points, " +
                  std::to_string(total_obs) + " observations)...";
    double mean_reproj_after = 0.0;

    // Save poses before BA for computing deltas
    auto poses_before = poses;

    int ba_outliers = 0;
    if (!bundle_adjust_laser(config.camera_names, poses, points_3d,
                             obs_per_point, config.ba_outlier_th1,
                             config.ba_outlier_th2, config.ba_max_iter,
                             mean_reproj_after, status, &ba_outliers,
                             config.lock_intrinsics)) {
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

    // Step 5: Write refined calibration to timestamped subfolder
    std::string base_folder = config.output_folder;
    if (base_folder.empty())
        base_folder = config.calibration_folder + "_laser_refined";

    // Create timestamped subfolder
    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y_%m_%d_%H_%M_%S", &tstruct);
    std::string out_folder = base_folder + "/" + tbuf;
    fs::create_directories(out_folder);

    if (status)
        *status = "Writing refined calibration to " + out_folder;
    std::string write_status;
    if (!CalibrationPipeline::write_calibration(poses, config.camera_names,
                                                out_folder, image_width,
                                                image_height, &write_status)) {
        result.error = write_status;
        return result;
    }

    // Write settings.json
    {
        nlohmann::json settings = config_to_json(config);
        std::string settings_path = out_folder + "/settings.json";
        std::ofstream ofs(settings_path, std::ios::binary);
        if (ofs)
            ofs << settings.dump(2);
    }

    result.output_folder = out_folder;
    result.success = true;

    auto t_end = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration<double>(t_end - t_start).count();

    printf("\n=== Laser Calibration Refinement Summary ===\n");
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
        *status = "Laser calibration complete! Reproj: " +
                  std::to_string(mean_reproj_before).substr(0, 5) + " -> " +
                  std::to_string(mean_reproj_after).substr(0, 5) +
                  " px. Output: " + out_folder;

    return result;
}

} // namespace LaserCalibration
