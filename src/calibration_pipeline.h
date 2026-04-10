#pragma once
// calibration_pipeline.h — Full multiview calibration pipeline (C++).
// Ports the multiview_calib Python pipeline: ChArUco detection → intrinsics →
// pairwise relative poses → chained global poses → bundle adjustment →
// world registration → per-camera YAML output.

#include "aruco_detect.h"
#include "calibration_tool.h"
#include "intrinsic_calibration.h"
#include "opencv_yaml_io.h"
#include "red_math.h"

#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#ifdef __APPLE__
#include <turbojpeg.h>
#include "FFmpegDemuxer.h"
#include "vt_async_decoder.h"
#include "aruco_metal.h"
#elif defined(_WIN32)
#include <turbojpeg.h>
#include "ffmpeg_frame_reader.h"
#include "aruco_cuda.h"
#else
#include "ffmpeg_frame_reader.h"
#endif

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define STDERR_FILENO 2
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace CalibrationPipeline {

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct CameraIntrinsics {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist = Eigen::Matrix<double, 5, 1>::Zero();
    double reproj_error = 0.0;
    int image_width = 0;
    int image_height = 0;
    // Per-image detected corners: image_index → {corner_id → pixel}
    std::map<int, std::vector<Eigen::Vector2f>> corners_per_image;
    std::map<int, std::vector<int>> ids_per_image;
};

struct RelativePose {
    Eigen::Matrix3d Rd = Eigen::Matrix3d::Identity();
    Eigen::Vector3d td = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> triang_points; // 3D points in cam_a's frame
    std::vector<int> point_ids;                 // global landmark IDs
};

struct CameraPose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist = Eigen::Matrix<double, 5, 1>::Zero();
};

// Per-camera detection results (used by both image and video pipelines)
struct DetectionResult {
    CameraIntrinsics cam_data;
    std::vector<std::vector<Eigen::Vector3f>> all_obj_points;
    std::vector<std::vector<Eigen::Vector2f>> all_img_points;
    int det_image_width = 0;
    int det_image_height = 0;
    bool ok = false;
    std::string error;
};

// Video frame range for video-based calibration
struct VideoFrameRange {
    std::string video_folder;
    std::vector<std::string> cam_ordered;
    int start_frame = 0;
    int stop_frame = 0;    // 0 = all
    int frame_step = 1;
};

// Progress tracking for ArUco calibration (shared between pipeline thread and UI)
struct ArucoProgress {
    // Per-camera detection progress
    struct CameraProgress {
        std::atomic<int> frames_processed{0};
        std::atomic<int> corners_detected{0};  // frames with valid board detections
        std::atomic<bool> done{false};
    };
    std::vector<std::unique_ptr<CameraProgress>> cameras;
    std::vector<std::string> camera_names;

    // Pipeline step progress (post-detection)
    std::atomic<int> current_step{0};     // 0=not started, 1-7 = pipeline steps
    std::atomic<int> total_steps{7};
    std::atomic<int> intrinsics_done{0};  // cameras with intrinsics calibrated

    void init(const std::vector<std::string> &cam_names) {
        camera_names = cam_names;
        cameras.clear();
        for (size_t i = 0; i < cam_names.size(); i++)
            cameras.push_back(std::make_unique<CameraProgress>());
        current_step.store(0);
        intrinsics_done.store(0);
    }
};

// Get video frame count using FFmpeg
inline int get_video_frame_count(const std::string &path) {
    AVFormatContext *ctx = avformat_alloc_context();
    if (!ctx) { fprintf(stderr, "[FrameCount] alloc failed\n"); return 0; }
    ctx->max_analyze_duration = 5 * AV_TIME_BASE;
    if (avformat_open_input(&ctx, path.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "[FrameCount] open failed: %s\n", path.c_str());
        return 0;
    }
    if (avformat_find_stream_info(ctx, nullptr) < 0) {
        fprintf(stderr, "[FrameCount] find_stream_info failed\n");
        avformat_close_input(&ctx); return 0;
    }
    for (unsigned i = 0; i < ctx->nb_streams; i++) {
        if (ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            int64_t nb = ctx->streams[i]->nb_frames;
            double dur = (ctx->duration != AV_NOPTS_VALUE)
                ? ctx->duration / (double)AV_TIME_BASE : -1.0;
            AVRational rate = ctx->streams[i]->avg_frame_rate;
            double fps = (rate.num > 0 && rate.den > 0) ? (double)rate.num / rate.den : 30.0;
            fprintf(stderr, "[FrameCount] nb_frames=%lld dur=%.2fs fps=%.1f path=%s\n",
                    (long long)nb, dur, fps, path.c_str());
            if (nb > 0) { avformat_close_input(&ctx); return (int)nb; }
            if (dur > 0) { avformat_close_input(&ctx); return (int)(dur * fps); }
            // Fallback: try companion _meta.csv (one row per frame + header)
            break;
        }
    }
    avformat_close_input(&ctx);
    // Fallback: count lines in companion _meta.csv file
    {
        namespace fs = std::filesystem;
        fs::path vp(path);
        std::string stem = vp.stem().string();
        fs::path meta = vp.parent_path() / (stem + "_meta.csv");
        if (fs::exists(meta)) {
            std::ifstream f(meta);
            int lines = 0;
            std::string line;
            while (std::getline(f, line)) lines++;
            int frames = std::max(0, lines - 1); // subtract header
            fprintf(stderr, "[FrameCount] meta.csv fallback: %d frames from %s\n",
                    frames, meta.string().c_str());
            return frames;
        }
    }
    return 0;
}

struct PerCameraMetrics {
    std::string name;
    int detection_count = 0;
    int observation_count = 0;
    double mean_reproj = 0.0;
    double median_reproj = 0.0;
    double std_reproj = 0.0;
    double max_reproj = 0.0;
    double intrinsic_reproj = 0.0;
};

// Calibration diagnostic database — stores intermediate data for inspection.
// Serialized to calibration_data.json alongside the YAML output.
struct CalibrationDatabase {
    // Per-frame board poses from PnP (camera_name → {frame_idx → pose})
    struct BoardPose {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        double reproj = 0; int num_corners = 0;
    };
    std::map<std::string, std::map<int, BoardPose>> board_poses;

    // Landmark map (camera_name → {landmark_id → pixel})
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;

    // Registration graph (experimental pipeline)
    struct RegistrationStep {
        std::string camera_name;
        int camera_idx = -1;
        std::string parent_camera; // registered against this camera (or "initial_pair")
        int num_shared_frames = 0;
        int num_3d_points = 0;
        std::string method; // "initial_pair", "markley_avg", "bridge"
    };
    std::vector<RegistrationStep> registration_order;

    // BA pass history
    struct BAPassInfo {
        int pass_number = 0;
        int fix_mode = 0;
        double cauchy_scale = 1.0;
        double cost_before = 0, cost_after = 0;
        int iterations = 0;
        double time_sec = 0;
        int outliers_removed = 0;
    };
    std::vector<BAPassInfo> ba_passes;

    // Per-observation residuals (after final BA)
    struct Residual {
        int camera_idx; int landmark_id;
        float obs_x, obs_y, pred_x, pred_y, error;
    };
    std::vector<Residual> residuals;

    // Pipeline timing
    double detection_time_sec = 0;
    double ba_time_sec = 0;
    double total_time_sec = 0;
};

struct CalibrationResult {
    std::vector<CameraPose> cameras;
    std::vector<std::string> cam_names;
    std::map<int, Eigen::Vector3d> points_3d;
    double mean_reproj_error = 0.0;
    int image_width = 0;
    int image_height = 0;
    bool success = false;
    std::string error;
    std::string warning;       // non-fatal issues (e.g., skipped cameras)
    std::string global_reg_status; // "success: N points", "skipped: ...", "failed: ..."
    std::string output_folder;
    std::vector<PerCameraMetrics> per_camera_metrics;
    std::vector<double> all_reproj_errors;
    int ba_rounds = 0;
    int outliers_removed = 0;
    // Diagnostic database (populated during experimental pipeline)
    CalibrationDatabase db;

    // Global multi-view consistency (Phase 1 diagnostic)
    struct GlobalConsistency {
        double mean_reproj = 0.0;
        double median_reproj = 0.0;
        double pct95_reproj = 0.0;
        int landmarks_triangulated = 0;
        int total_observations = 0;
        bool computed = false;
        struct CamResult { std::string name; double mean_reproj = 0.0; int obs = 0; };
        std::vector<CamResult> per_camera;
    };
    GlobalConsistency global_consistency;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

inline double laplacian_variance(const uint8_t *gray, int w, int h) {
    double sum = 0, sum_sq = 0;
    int count = 0;
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            int lap = -4*(int)gray[y*w+x] + (int)gray[(y-1)*w+x]
                      + (int)gray[(y+1)*w+x] + (int)gray[y*w+x-1] + (int)gray[y*w+x+1];
            sum += lap; sum_sq += (double)lap*lap; count++;
        }
    if (count == 0) return 0.0;
    double mean = sum / count;
    return (sum_sq / count) - mean * mean;
}

// Get sorted list of image numbers for a camera serial.
inline std::vector<int>
get_sorted_image_numbers(const std::string &img_path,
                         const std::string &serial) {
    namespace fs = std::filesystem;
    std::vector<int> numbers;
    std::string prefix = serial + "_";
    for (const auto &entry : fs::directory_iterator(img_path)) {
        if (!entry.is_regular_file())
            continue;
        std::string fn = entry.path().filename().string();
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".tiff" && ext != ".tif")
            continue;
        if (fn.substr(0, prefix.size()) != prefix)
            continue;
        // Extract number: everything between prefix and extension
        std::string num_str =
            fn.substr(prefix.size(), fn.size() - prefix.size() - ext.size());
        try {
            numbers.push_back(std::stoi(num_str));
        } catch (...) {
            continue;
        }
    }
    std::sort(numbers.begin(), numbers.end());
    return numbers;
}

// Get image file extension for a camera serial (assumes consistent extension).
inline std::string get_image_extension(const std::string &img_path,
                                       const std::string &serial) {
    namespace fs = std::filesystem;
    std::string prefix = serial + "_";
    for (const auto &entry : fs::directory_iterator(img_path)) {
        if (!entry.is_regular_file())
            continue;
        std::string fn = entry.path().filename().string();
        if (fn.substr(0, prefix.size()) == prefix)
            return entry.path().extension().string();
    }
    return ".jpg";
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Phase 2: calibrate intrinsics from detection results (per camera, parallel)
// ─────────────────────────────────────────────────────────────────────────────

inline bool calibrate_intrinsics_from_detections(
    const std::vector<std::string> &cam_ordered,
    std::vector<DetectionResult> &detections,
    std::map<std::string, CameraIntrinsics> &intrinsics,
    std::string *status,
    std::vector<std::string> *skipped_cameras = nullptr) {

    int total_cameras = (int)cam_ordered.size();
    std::mutex status_mutex;

    // Skip failed cameras
    std::vector<int> valid_cam_indices;
    for (int cam_i = 0; cam_i < total_cameras; cam_i++) {
        if (!detections[cam_i].ok) {
            fprintf(stderr, "[Calibration] Skipping camera %s: %s\n",
                    cam_ordered[cam_i].c_str(), detections[cam_i].error.c_str());
            if (skipped_cameras)
                skipped_cameras->push_back(cam_ordered[cam_i]);
        } else {
            valid_cam_indices.push_back(cam_i);
        }
    }
    if (valid_cam_indices.size() < 2) {
        if (status) *status = "Error: fewer than 2 cameras passed detection";
        return false;
    }
    int num_cameras = (int)valid_cam_indices.size();

    // Calibrate intrinsics in parallel (Ceres is thread-safe)
    std::vector<CameraIntrinsics> results(num_cameras);
    std::vector<bool> result_ok(num_cameras, false);
    std::atomic<int> calib_done{0};

    if (status)
        *status = "Calibrating intrinsics (0/" + std::to_string(num_cameras) + ")...";

    {
        std::vector<std::future<void>> calib_futures;
        for (int vi = 0; vi < num_cameras; vi++) {
            int orig_i = valid_cam_indices[vi];
            calib_futures.push_back(std::async(std::launch::async, [&, vi, orig_i]() {
                auto &det = detections[orig_i];

                auto calib_result = intrinsic_calib::calibrateCamera(
                    det.all_obj_points, det.all_img_points,
                    det.det_image_width, det.det_image_height,
                    /*fix_aspect_ratio=*/true);

                det.cam_data.K = calib_result.K;
                det.cam_data.dist = calib_result.dist;
                det.cam_data.reproj_error = calib_result.reproj_error;
                results[vi] = std::move(det.cam_data);
                result_ok[vi] = true;

                int done = ++calib_done;
                {
                    std::lock_guard<std::mutex> lock(status_mutex);
                    if (status)
                        *status = "Calibrating intrinsics (" +
                                  std::to_string(done) + "/" +
                                  std::to_string(num_cameras) + ")...";
                }
            }));
        }
        for (auto &f : calib_futures)
            f.get();
    }

    // Collect results
    for (int vi = 0; vi < num_cameras; vi++) {
        int orig_i = valid_cam_indices[vi];
        if (!result_ok[vi]) {
            if (status)
                *status = "Error: calibration failed for camera " +
                          cam_ordered[orig_i];
            return false;
        }
        intrinsics[cam_ordered[orig_i]] = std::move(results[vi]);
    }

    if (status) {
        std::string msg = "Intrinsics done. Reproj errors:";
        for (const auto &[name, intr] : intrinsics)
            msg += " " + name + "=" +
                   std::to_string(intr.reproj_error).substr(0, 5);
        *status = msg;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Detect ChArUco corners and calibrate intrinsics (per camera)
// ─────────────────────────────────────────────────────────────────────────────

inline bool
detect_and_calibrate_intrinsics(
    const CalibrationTool::CalibConfig &config,
    std::map<std::string, CameraIntrinsics> &intrinsics,
    std::string *status,
    aruco_detect::GpuThresholdFunc gpu_thresh = nullptr,
    void *gpu_ctx = nullptr,
    std::vector<std::string> *skipped_cameras = nullptr) {

    const auto &cs = config.charuco_setup;
    int max_corners = (cs.w - 1) * (cs.h - 1);

    // Get sorted image numbers (should be same across cameras).
    // Use the first camera to determine the image list, then verify others.
    auto image_numbers =
        get_sorted_image_numbers(config.img_path, config.cam_ordered[0]);
    if (image_numbers.empty()) {
        if (status)
            *status = "Error: No images found for " + config.cam_ordered[0];
        return false;
    }

    // Build image_number → sorted_index map
    std::map<int, int> img_num_to_idx;
    for (int i = 0; i < (int)image_numbers.size(); i++)
        img_num_to_idx[image_numbers[i]] = i;

    int num_cameras = (int)config.cam_ordered.size();
    std::mutex status_mutex;
    std::atomic<int> cameras_done{0};

    std::vector<DetectionResult> detections(num_cameras);

    // ── Phase 1: detect ChArUco corners in parallel (thread-safe) ──
    int total_images = num_cameras * (int)image_numbers.size();
    std::atomic<int> images_done{0};
    auto phase1_start = std::chrono::steady_clock::now();

    if (status) *status = "Detecting ChArUco corners (0/" +
                          std::to_string(total_images) + " images)...";
    fprintf(stderr, "[Calibration] Phase 1: detecting corners in %d images "
            "across %d cameras...\n", total_images, num_cameras);
    {
        std::vector<std::future<void>> futures;
        for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
            futures.push_back(std::async(std::launch::async, [&, cam_i]() {
                const std::string &serial = config.cam_ordered[cam_i];
                std::string ext = get_image_extension(config.img_path, serial);
                auto &det = detections[cam_i];

                auto aruco_dict = aruco_detect::getDictionary(cs.dictionary);
                aruco_detect::CharucoBoard board;
                board.squares_x = cs.w;
                board.squares_y = cs.h;
                board.square_length = cs.square_side_length;
                board.marker_length = cs.marker_side_length;
                board.dictionary_id = cs.dictionary;

#ifdef __APPLE__
                // Per-thread turbojpeg decompressor (SIMD-accelerated)
                tjhandle tj = tjInitDecompress();
#endif
                for (int img_num : image_numbers) {
                    std::string img_file = config.img_path + "/" + serial +
                                           "_" + std::to_string(img_num) + ext;
                    int w = 0, h = 0;
                    unsigned char *pixels = nullptr;

#ifdef __APPLE__
                    // turbojpeg: NEON SIMD, ~2-3x faster than stbi_load
                    FILE *fp = fopen(img_file.c_str(), "rb");
                    if (!fp) { ++images_done; continue; }
                    fseek(fp, 0, SEEK_END);
                    long fsize = ftell(fp);
                    fseek(fp, 0, SEEK_SET);
                    std::vector<unsigned char> jpeg_buf(fsize);
                    fread(jpeg_buf.data(), 1, fsize, fp);
                    fclose(fp);

                    int tj_subsamp, tj_colorspace;
                    if (tjDecompressHeader3(tj, jpeg_buf.data(), fsize,
                                            &w, &h, &tj_subsamp,
                                            &tj_colorspace) != 0) {
                        ++images_done; continue;
                    }
                    pixels = (unsigned char *)malloc(w * h);
                    if (tjDecompress2(tj, jpeg_buf.data(), fsize,
                                      pixels, w, 0, h, TJPF_GRAY,
                                      TJFLAG_FASTDCT) != 0) {
                        free(pixels);
                        ++images_done; continue;
                    }
#else
                    int channels = 0;
                    pixels = stbi_load(img_file.c_str(), &w, &h, &channels, 1);
                    if (!pixels) { ++images_done; continue; }
#endif

                    if (det.det_image_width == 0) {
                        det.det_image_width = w;
                        det.det_image_height = h;
                        det.cam_data.image_width = w;
                        det.cam_data.image_height = h;
                    }

                    auto charuco = aruco_detect::detectCharucoBoard(
                        pixels, w, h, board, aruco_dict,
                        gpu_thresh, gpu_ctx,
                        nullptr, 0, 1);  // full-res contour finding for max accuracy

                    if ((int)charuco.ids.size() < 6) {
#ifdef __APPLE__
                        free(pixels);
#else
                        stbi_image_free(pixels);
#endif
                        int done_img = ++images_done;
                        // Update progress periodically
                        if (done_img % 20 == 0) {
                            auto now = std::chrono::steady_clock::now();
                            double elapsed = std::chrono::duration<double>(now - phase1_start).count();
                            double rate = done_img / elapsed;
                            int remaining = total_images - done_img;
                            double eta = remaining / rate;
                            std::lock_guard<std::mutex> lock(status_mutex);
                            if (status) {
                                char buf[128];
                                snprintf(buf, sizeof(buf),
                                    "Detecting corners (%d/%d images, %.0f img/s, ~%.0fs left)...",
                                    done_img, total_images, rate, eta);
                                *status = buf;
                            }
                        }
                        continue;
                    }

                    aruco_detect::cornerSubPix(
                        pixels, w, h, charuco.corners, 3, 100, 0.001f);

#ifdef __APPLE__
                    free(pixels);
#else
                    stbi_image_free(pixels);
#endif

                    int sorted_idx = img_num_to_idx[img_num];
                    det.cam_data.corners_per_image[sorted_idx] = charuco.corners;
                    det.cam_data.ids_per_image[sorted_idx] = charuco.ids;

                    std::vector<Eigen::Vector3f> obj_pts;
                    std::vector<Eigen::Vector2f> img_pts;
                    aruco_detect::matchImagePoints(
                        board, charuco.corners, charuco.ids, obj_pts, img_pts);

                    if ((int)obj_pts.size() >= 6) {
                        det.all_obj_points.push_back(std::move(obj_pts));
                        det.all_img_points.push_back(std::move(img_pts));
                    }

                    int done_img = ++images_done;
                    if (done_img % 20 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        double elapsed = std::chrono::duration<double>(now - phase1_start).count();
                        double rate = done_img / elapsed;
                        int remaining = total_images - done_img;
                        double eta = remaining / rate;
                        {
                            std::lock_guard<std::mutex> lock(status_mutex);
                            if (status) {
                                char buf[128];
                                snprintf(buf, sizeof(buf),
                                    "Detecting corners (%d/%d images, %.0f img/s, ~%.0fs left)...",
                                    done_img, total_images, rate, eta);
                                *status = buf;
                            }
                        }
                        fprintf(stderr, "[Calibration]   %d/%d images (%.0f img/s, ~%.0fs left)\n",
                                done_img, total_images, rate, eta);
                    }
                }
#ifdef __APPLE__
                tjDestroy(tj);
#endif

                if (det.all_obj_points.size() < 4) {
                    det.error = "Too few valid images for camera " + serial +
                                " (" +
                                std::to_string(det.all_obj_points.size()) + ")";
                } else {
                    det.ok = true;
                }

                int done = ++cameras_done;
                {
                    std::lock_guard<std::mutex> lock(status_mutex);
                    if (status)
                        *status = "Detecting corners — camera " +
                                  std::to_string(done) + "/" +
                                  std::to_string(num_cameras) + " done";
                }
            }));
        }
        for (auto &f : futures)
            f.get();
    }

    {
        auto phase1_end = std::chrono::steady_clock::now();
        double phase1_s = std::chrono::duration<double>(phase1_end - phase1_start).count();
        fprintf(stderr, "[Calibration] Phase 1 done: %d images in %.1fs (%.0f img/s)\n",
                total_images, phase1_s, total_images / phase1_s);
    }

    // ── Phase 2: calibrate intrinsics (shared helper) ──
    return calibrate_intrinsics_from_detections(
        config.cam_ordered, detections, intrinsics, status, skipped_cameras);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 (video path): detect ChArUco corners from video files + calibrate intrinsics
// ─────────────────────────────────────────────────────────────────────────────

inline bool detect_and_calibrate_intrinsics_video(
    const CalibrationTool::CalibConfig &config,
    const VideoFrameRange &vfr,
    std::map<std::string, CameraIntrinsics> &intrinsics,
    std::vector<int> *out_frame_numbers,
    std::string *status,
    aruco_detect::GpuThresholdFunc gpu_thresh = nullptr,
    void *gpu_ctx = nullptr,
    std::vector<std::string> *skipped_cameras = nullptr,
    ArucoProgress *progress = nullptr) {

    const auto &cs = config.charuco_setup;

    // Discover videos
    auto video_files = CalibrationTool::discover_aruco_videos(
        vfr.video_folder, vfr.cam_ordered);
    if (video_files.empty()) {
        if (status) *status = "Error: No aruco videos found in " + vfr.video_folder;
        return false;
    }

    // Probe frame count from first video
    int total_video_frames = 0;
    { auto first_it = video_files.begin();
      total_video_frames = get_video_frame_count(first_it->second); }

    // Build frame list
    int stop_fr = vfr.stop_frame;
    if (stop_fr <= 0) stop_fr = total_video_frames;
    int step = std::max(1, vfr.frame_step);
    std::vector<int> frame_numbers;
    for (int f = vfr.start_frame; f < stop_fr; f += step)
        frame_numbers.push_back(f);
    if (frame_numbers.empty()) {
        if (status) *status = "Error: No frames in range";
        return false;
    }
    std::map<int, int> frame_to_idx;
    for (int i = 0; i < (int)frame_numbers.size(); i++)
        frame_to_idx[frame_numbers[i]] = i;
    if (out_frame_numbers) *out_frame_numbers = frame_numbers;

    int num_cameras = (int)vfr.cam_ordered.size();
    std::mutex status_mutex;
    std::atomic<int> cameras_done{0};
    std::vector<DetectionResult> detections(num_cameras);
    int frames_per_cam = (int)frame_numbers.size();
    int total_frames = num_cameras * frames_per_cam;
    std::atomic<int> frames_done{0};
    auto phase1_start = std::chrono::steady_clock::now();

    if (progress) progress->init(vfr.cam_ordered);

    if (status) *status = "Detecting ChArUco corners from video (0/" +
                          std::to_string(total_frames) + " frames)...";
    fprintf(stderr, "[Calibration] Phase 1 (video): %d frames across %d cameras...\n",
            total_frames, num_cameras);

    {
        std::vector<std::future<void>> futures;
        for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
            auto vid_it = video_files.find(vfr.cam_ordered[cam_i]);
            if (vid_it == video_files.end()) {
                detections[cam_i].error = "No video for " + vfr.cam_ordered[cam_i];
                continue;
            }
            futures.push_back(std::async(std::launch::async,
                [&, cam_i, video_path = vid_it->second]() {
                const std::string &serial = vfr.cam_ordered[cam_i];
                auto &det = detections[cam_i];
                auto aruco_dict = aruco_detect::getDictionary(cs.dictionary);
                aruco_detect::CharucoBoard board;
                board.squares_x = cs.w; board.squares_y = cs.h;
                board.square_length = cs.square_side_length;
                board.marker_length = cs.marker_side_length;
                board.dictionary_id = cs.dictionary;

#ifdef __APPLE__
                // macOS: FFmpegDemuxer + VTAsyncDecoder + Metal GPU
                std::unique_ptr<FFmpegDemuxer> demuxer;
                try { demuxer = std::make_unique<FFmpegDemuxer>(
                    video_path.c_str(), std::map<std::string, std::string>{});
                } catch (...) { det.error = "Failed to open demuxer: " + video_path; return; }

                int w = (int)demuxer->GetWidth(), h = (int)demuxer->GetHeight();
                det.det_image_width = w; det.det_image_height = h;
                det.cam_data.image_width = w; det.cam_data.image_height = h;

                VTAsyncDecoder vt;
                if (!vt.init(demuxer->GetExtradata(), demuxer->GetExtradataSize(),
                             demuxer->GetVideoCodec())) {
                    det.error = "VT init failed for " + video_path; return;
                }

                std::vector<uint8_t> gray(w * h);

                int frame = 0;
                bool first_pkt_from_seek = false;
                uint8_t *seek_pkt = nullptr; size_t seek_pkt_size = 0;
                PacketData seek_pkt_info;
                if (vfr.start_frame > 0) {
                    SeekContext seek_ctx((uint64_t)vfr.start_frame, PREV_KEY_FRAME, BY_NUMBER);
                    if (demuxer->Seek(seek_ctx, seek_pkt, seek_pkt_size, seek_pkt_info)) {
                        frame = (int)demuxer->FrameNumberFromTs(seek_ctx.out_frame_pts);
                        if (frame < 0) frame = 0;
                        first_pkt_from_seek = true;
                    }
                }

                // Unified process_frame: BGRA→gray then full-res detection
                // (same pipeline as image path — proven REDv4 approach)
                auto process_frame = [&](CVPixelBufferRef pb) {
                    bool should_detect = frame_to_idx.count(frame) > 0;
                    if (should_detect) {
                        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
                        const uint8_t *bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pb);
                        int stride = (int)CVPixelBufferGetBytesPerRow(pb);
                        for (int y = 0; y < h; y++) { const uint8_t *row = bgra + y * stride;
                            for (int x = 0; x < w; x++)
                                gray[y*w+x] = (uint8_t)((row[x*4+2]*77 + row[x*4+1]*150 + row[x*4]*29) >> 8); }
                        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

                        // Full-res detection with GPU-accelerated threshold (same as image path)
                        auto charuco = aruco_detect::detectCharucoBoard(
                            gray.data(), w, h, board, aruco_dict,
                            gpu_thresh, gpu_ctx,
                            nullptr, 0, 1);  // full-res contour finding
                        if ((int)charuco.ids.size() >= 6) {
                            aruco_detect::cornerSubPix(gray.data(), w, h, charuco.corners, 3, 100, 0.001f);
                            int sorted_idx = frame_to_idx[frame];
                            det.cam_data.corners_per_image[sorted_idx] = charuco.corners;
                            det.cam_data.ids_per_image[sorted_idx] = charuco.ids;
                            std::vector<Eigen::Vector3f> obj_pts;
                            std::vector<Eigen::Vector2f> img_pts;
                            aruco_detect::matchImagePoints(board, charuco.corners, charuco.ids, obj_pts, img_pts);
                            if ((int)obj_pts.size() >= 6) {
                                det.all_obj_points.push_back(std::move(obj_pts));
                                det.all_img_points.push_back(std::move(img_pts));
                            }
                            if (progress && cam_i < (int)progress->cameras.size())
                                progress->cameras[cam_i]->corners_detected.fetch_add(1, std::memory_order_relaxed);
                        }
                        if (progress && cam_i < (int)progress->cameras.size())
                            progress->cameras[cam_i]->frames_processed.fetch_add(1, std::memory_order_relaxed);
                        int done_f = ++frames_done;
                        if (done_f % 20 == 0) {
                            auto now = std::chrono::steady_clock::now();
                            double elapsed = std::chrono::duration<double>(now - phase1_start).count();
                            double rate = done_f / elapsed;
                            double eta = (total_frames - done_f) / rate;
                            std::lock_guard<std::mutex> lock(status_mutex);
                            if (status) { char buf[128];
                                snprintf(buf, sizeof(buf), "Detecting corners (%d/%d frames, %.0f f/s, ~%.0fs left)...",
                                    done_f, total_frames, rate, eta);
                                *status = buf; }
                        }
                    }
                    CFRelease(pb);
                    frame++;
                };

                while (true) {
                    if (frame >= stop_fr) break;
                    uint8_t *pkt_data = nullptr; size_t pkt_size = 0; PacketData pkt_info;
                    if (first_pkt_from_seek) {
                        pkt_data = seek_pkt; pkt_size = seek_pkt_size;
                        pkt_info = seek_pkt_info; first_pkt_from_seek = false;
                    } else { if (!demuxer->Demux(pkt_data, pkt_size, pkt_info)) break; }
                    bool is_key = (pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                    vt.submit_blocking(pkt_data, pkt_size, pkt_info.pts,
                                       pkt_info.dts, demuxer->GetTimebase(), is_key);
                    while (CVPixelBufferRef pb = vt.drain_one()) {
                        process_frame(pb); if (frame >= stop_fr) break; }
                    if (frame >= stop_fr) break;
                }
                while (frame < stop_fr) {
                    CVPixelBufferRef pb = vt.drain_one();
                    if (!pb) break; process_frame(pb);
                }
#else
                // Linux: ffmpeg_reader::FrameReader (CPU decode)
                ffmpeg_reader::FrameReader reader;
                if (!reader.open(video_path)) {
                    det.error = "Failed to open video: " + video_path; return; }
                int w = reader.width(), h = reader.height();
                det.det_image_width = w; det.det_image_height = h;
                det.cam_data.image_width = w; det.cam_data.image_height = h;
                std::vector<uint8_t> gray(w * h);
                for (int frame_num : frame_numbers) {
                    const uint8_t *rgb = reader.readFrame(frame_num);
                    if (!rgb) continue;
                    for (int i = 0; i < w * h; i++)
                        gray[i] = (uint8_t)((rgb[i*3]*77 + rgb[i*3+1]*150 + rgb[i*3+2]*29) >> 8);
                    auto charuco = aruco_detect::detectCharucoBoard(gray.data(), w, h, board, aruco_dict,
                        gpu_thresh, gpu_ctx, nullptr, 0, 1);  // full-res, with GPU threshold
                    if ((int)charuco.ids.size() >= 6) {
                        aruco_detect::cornerSubPix(gray.data(), w, h, charuco.corners, 3, 100, 0.001f);
                        int sorted_idx = frame_to_idx[frame_num];
                        det.cam_data.corners_per_image[sorted_idx] = charuco.corners;
                        det.cam_data.ids_per_image[sorted_idx] = charuco.ids;
                        std::vector<Eigen::Vector3f> obj_pts;
                        std::vector<Eigen::Vector2f> img_pts;
                        aruco_detect::matchImagePoints(board, charuco.corners, charuco.ids, obj_pts, img_pts);
                        if ((int)obj_pts.size() >= 6) {
                            det.all_obj_points.push_back(std::move(obj_pts));
                            det.all_img_points.push_back(std::move(img_pts));
                        }
                        if (progress && cam_i < (int)progress->cameras.size())
                            progress->cameras[cam_i]->corners_detected.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (progress && cam_i < (int)progress->cameras.size())
                        progress->cameras[cam_i]->frames_processed.fetch_add(1, std::memory_order_relaxed);
                    int done_f = ++frames_done;
                    if (done_f % 20 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        double elapsed = std::chrono::duration<double>(now - phase1_start).count();
                        double rate = done_f / elapsed;
                        double eta = (total_frames - done_f) / rate;
                        std::lock_guard<std::mutex> lk(status_mutex);
                        if (status) { char buf[128];
                            snprintf(buf, sizeof(buf), "Detecting corners (%d/%d frames, %.0f f/s, ~%.0fs left)...",
                                done_f, total_frames, rate, eta);
                            *status = buf; }
                    }
                }
#endif
                if (det.all_obj_points.size() < 4) {
                    det.error = "Too few valid frames for camera " + serial +
                                " (" + std::to_string(det.all_obj_points.size()) + ")";
                } else { det.ok = true; }
                if (progress && cam_i < (int)progress->cameras.size())
                    progress->cameras[cam_i]->done.store(true, std::memory_order_relaxed);
                int done = ++cameras_done;
                { std::lock_guard<std::mutex> lock(status_mutex);
                  if (status) *status = "Detecting corners — camera " +
                    std::to_string(done) + "/" + std::to_string(num_cameras) + " done"; }
            }));
        }
        for (auto &f : futures) f.get();
    }

    { auto phase1_end = std::chrono::steady_clock::now();
      double phase1_s = std::chrono::duration<double>(phase1_end - phase1_start).count();
      fprintf(stderr, "[Calibration] Phase 1 (video) done: %d frames in %.1fs (%.0f f/s)\n",
              total_frames, phase1_s, total_frames / phase1_s);
      for (int cam_i = 0; cam_i < num_cameras; cam_i++)
          fprintf(stderr, "[Calibration]   %s: %d frames with detections\n",
                  vfr.cam_ordered[cam_i].c_str(),
                  (int)detections[cam_i].cam_data.corners_per_image.size()); }

    // ── Phase 2: calibrate intrinsics (shared helper) ──
    return calibrate_intrinsics_from_detections(
        vfr.cam_ordered, detections, intrinsics, status, skipped_cameras);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Build unified landmark map from per-camera detections
// ─────────────────────────────────────────────────────────────────────────────

inline void build_landmarks(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks) {

    int max_corners =
        (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);

    for (const auto &serial : config.cam_ordered) {
        auto it = intrinsics.find(serial);
        if (it == intrinsics.end())
            continue;
        const auto &intr = it->second;
        auto &cam_landmarks = landmarks[serial];

        for (const auto &[img_idx, corners] : intr.corners_per_image) {
            const auto &ids = intr.ids_per_image.at(img_idx);
            for (int j = 0; j < (int)ids.size(); j++) {
                int global_id = img_idx * max_corners + ids[j];
                cam_landmarks[global_id] =
                    Eigen::Vector2d(corners[j].x(), corners[j].y());
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Compute relative pose for each spanning tree edge
// ─────────────────────────────────────────────────────────────────────────────

inline bool compute_relative_poses(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    std::map<std::pair<int, int>, RelativePose> &relative_poses,
    std::string *status) {

    int num_cameras = (int)config.cam_ordered.size();

    // Build spanning tree edges: each second_view connects to its predecessor
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < (int)config.second_view_order.size(); i++) {
        int cam_new = config.second_view_order[i];
        int cam_prev = (i == 0) ? config.first_view
                                : config.second_view_order[i - 1];
        edges.push_back({cam_prev, cam_new});
    }

    for (int e = 0; e < (int)edges.size(); e++) {
        auto [idx_a, idx_b] = edges[e];
        const std::string &serial_a = config.cam_ordered[idx_a];
        const std::string &serial_b = config.cam_ordered[idx_b];

        if (status)
            *status = "Computing relative poses (edge " +
                      std::to_string(e + 1) + "/" +
                      std::to_string(edges.size()) + "): " + serial_a +
                      " → " + serial_b + "...";

        const auto &lm_a = landmarks.at(serial_a);
        const auto &lm_b = landmarks.at(serial_b);
        const auto &intr_a = intrinsics.at(serial_a);
        const auto &intr_b = intrinsics.at(serial_b);

        // Find common landmark IDs
        std::vector<int> common_ids;
        for (const auto &[id, _] : lm_a) {
            if (lm_b.count(id))
                common_ids.push_back(id);
        }

        if (common_ids.size() < 20) {
            if (status)
                *status = "Error: only " + std::to_string(common_ids.size()) +
                          " common points between " + serial_a + " and " +
                          serial_b + " (need ≥20)";
            return false;
        }

        // Undistort 2D points
        std::vector<Eigen::Vector2d> pts_a_undist, pts_b_undist;
        for (int id : common_ids) {
            pts_a_undist.push_back(
                red_math::undistortPoint(lm_a.at(id), intr_a.K, intr_a.dist));
            pts_b_undist.push_back(
                red_math::undistortPoint(lm_b.at(id), intr_b.K, intr_b.dist));
        }

        // Normalize points: undistorted pixel coords → normalized camera coords
        std::vector<Eigen::Vector2d> pts_a_norm, pts_b_norm;
        pts_a_norm.reserve(common_ids.size());
        pts_b_norm.reserve(common_ids.size());
        for (int i = 0; i < (int)common_ids.size(); i++) {
            double fx_a = intr_a.K(0, 0), fy_a = intr_a.K(1, 1);
            double cx_a = intr_a.K(0, 2), cy_a = intr_a.K(1, 2);
            double fx_b = intr_b.K(0, 0), fy_b = intr_b.K(1, 1);
            double cx_b = intr_b.K(0, 2), cy_b = intr_b.K(1, 2);
            pts_a_norm.push_back(Eigen::Vector2d(
                (pts_a_undist[i].x() - cx_a) / fx_a,
                (pts_a_undist[i].y() - cy_a) / fy_a));
            pts_b_norm.push_back(Eigen::Vector2d(
                (pts_b_undist[i].x() - cx_b) / fx_b,
                (pts_b_undist[i].y() - cy_b) / fy_b));
        }

        // Find essential matrix directly from normalized points
        // (more robust than F → E conversion)
        auto e_result = red_math::findEssentialMatRANSAC(
            pts_a_norm, pts_b_norm, 0.999, 0.001);

        std::vector<bool> inlier_mask;
        Eigen::Matrix3d E;

        if (!e_result.success) {
            // Fallback: try with pixel coords and F matrix → derive E
            auto f_result = red_math::findFundamentalMatRANSAC(
                pts_a_undist, pts_b_undist, 0.999, 3.0);
            if (!f_result.success) {
                if (status)
                    *status =
                        "Error: pose estimation failed for " + serial_a +
                        " → " + serial_b + " (" +
                        std::to_string(common_ids.size()) + " common pts)";
                return false;
            }
            // E = Kb^T * F * Ka (Eigen matrix multiply, no conversion needed)
            E = intr_b.K.transpose() * f_result.E * intr_a.K;
            inlier_mask = std::move(f_result.inlier_mask);
        } else {
            E = e_result.E;
            inlier_mask = std::move(e_result.inlier_mask);
        }

        // Decompose essential matrix → 4 candidate poses
        auto decomp = red_math::decomposeEssentialMatrix(E);
        Eigen::Matrix3d R1 = decomp.R1;
        Eigen::Matrix3d R2 = decomp.R2;
        Eigen::Vector3d t_dir = decomp.t;

        // Four candidates: (R1, +t), (R1, -t), (R2, +t), (R2, -t)
        struct Candidate {
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
        };
        Candidate candidates[4] = {
            {R1, t_dir}, {R1, -t_dir}, {R2, t_dir}, {R2, -t_dir}};

        int best_idx = 0;
        int best_count = -1;

        for (int c = 0; c < 4; c++) {
            // Projection matrices for triangulation
            auto P_a = red_math::projectionFromKRt(
                intr_a.K, Eigen::Matrix3d::Identity(),
                Eigen::Vector3d::Zero());
            auto P_b = red_math::projectionFromKRt(intr_b.K, candidates[c].R,
                                                   candidates[c].t);

            int positive_count = 0;
            for (int i = 0; i < (int)common_ids.size(); i++) {
                if (!inlier_mask[i])
                    continue;
                Eigen::Vector3d X = red_math::triangulatePoints(
                    {pts_a_undist[i], pts_b_undist[i]}, {P_a, P_b});

                // Check positive depth in both cameras
                bool front_a = X.z() > 0;
                Eigen::Vector3d X_b = candidates[c].R * X + candidates[c].t;
                bool front_b = X_b.z() > 0;
                if (front_a && front_b)
                    positive_count++;
            }

            if (positive_count > best_count) {
                best_count = positive_count;
                best_idx = c;
            }
        }

        // Triangulate final points with the best pose
        RelativePose rp;
        rp.Rd = candidates[best_idx].R;
        rp.td = candidates[best_idx].t;

        auto P_a = red_math::projectionFromKRt(
            intr_a.K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        auto P_b =
            red_math::projectionFromKRt(intr_b.K, rp.Rd, rp.td);

        for (int i = 0; i < (int)common_ids.size(); i++) {
            if (!inlier_mask[i])
                continue;
            Eigen::Vector3d X =
                red_math::triangulatePoints({pts_a_undist[i], pts_b_undist[i]}, {P_a, P_b});

            // Skip points behind cameras
            Eigen::Vector3d X_b = rp.Rd * X + rp.td;
            if (X.z() <= 0 || X_b.z() <= 0)
                continue;

            rp.triang_points.push_back(X);
            rp.point_ids.push_back(common_ids[i]);
        }

        relative_poses[{idx_a, idx_b}] = std::move(rp);
    }

    if (status)
        *status = "Relative poses computed (" +
                  std::to_string(edges.size()) + " edges)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Chain pairwise poses along spanning tree
// ─────────────────────────────────────────────────────────────────────────────

inline bool concatenate_poses(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    const std::map<std::pair<int, int>, RelativePose> &relative_poses,
    std::vector<CameraPose> &poses, std::string *status) {

    int num_cameras = (int)config.cam_ordered.size();
    poses.resize(num_cameras);

    // Root camera: identity pose
    int root = config.first_view;
    const auto &root_intr = intrinsics.at(config.cam_ordered[root]);
    poses[root].R = Eigen::Matrix3d::Identity();
    poses[root].t = Eigen::Vector3d::Zero();
    poses[root].K = root_intr.K;
    poses[root].dist = root_intr.dist;

    // Accumulated world-frame 3D points for scale estimation
    std::map<int, Eigen::Vector3d> accumulated_pts;

    // Process edges in order
    for (int i = 0; i < (int)config.second_view_order.size(); i++) {
        int cam_new = config.second_view_order[i];
        int cam_prev =
            (i == 0) ? config.first_view : config.second_view_order[i - 1];

        const auto &rp = relative_poses.at({cam_prev, cam_new});
        const auto &intr_new = intrinsics.at(config.cam_ordered[cam_new]);

        double scale = 1.0;

        if (i > 0 && !accumulated_pts.empty()) {
            // Find common points between accumulated and locally-triangulated
            std::vector<std::pair<int, int>> common; // (accum_id, local_idx)
            for (int j = 0; j < (int)rp.point_ids.size(); j++) {
                if (accumulated_pts.count(rp.point_ids[j]))
                    common.push_back({rp.point_ids[j], j});
            }

            if (common.size() >= 2) {
                // Compute pairwise distance ratios
                const auto &R_prev = poses[cam_prev].R;
                const auto &t_prev = poses[cam_prev].t;
                std::vector<double> ratios;

                for (int j = 0; j < (int)common.size(); j++) {
                    for (int k = j + 1; k < (int)common.size(); k++) {
                        // Transform accumulated world points to cam_prev frame
                        Eigen::Vector3d Xj_cam =
                            R_prev * accumulated_pts[common[j].first] +
                            t_prev;
                        Eigen::Vector3d Xk_cam =
                            R_prev * accumulated_pts[common[k].first] +
                            t_prev;
                        double dist_world = (Xj_cam - Xk_cam).norm();

                        // Local triangulated distances
                        double dist_local =
                            (rp.triang_points[common[j].second] -
                             rp.triang_points[common[k].second])
                                .norm();

                        if (dist_local > 1e-10)
                            ratios.push_back(dist_world / dist_local);
                    }
                }

                if (!ratios.empty()) {
                    std::sort(ratios.begin(), ratios.end());
                    scale = ratios[ratios.size() / 2]; // median
                }
            }
        }

        // Chain pose
        const auto &R_prev = poses[cam_prev].R;
        const auto &t_prev = poses[cam_prev].t;
        poses[cam_new].R = rp.Rd * R_prev;
        poses[cam_new].t = rp.Rd * t_prev + scale * rp.td;
        poses[cam_new].K = intr_new.K;
        poses[cam_new].dist = intr_new.dist;

        // Transform locally-triangulated points to world frame and accumulate
        for (int j = 0; j < (int)rp.point_ids.size(); j++) {
            int pid = rp.point_ids[j];
            if (!accumulated_pts.count(pid)) {
                Eigen::Vector3d X_cam_true = scale * rp.triang_points[j];
                Eigen::Vector3d X_world =
                    R_prev.transpose() * (X_cam_true - t_prev);
                accumulated_pts[pid] = X_world;
            }
        }
    }

    if (status)
        *status = "Poses chained (" + std::to_string(num_cameras) +
                  " cameras, " + std::to_string(accumulated_pts.size()) +
                  " 3D points)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5: Bundle adjustment via Ceres
// ─────────────────────────────────────────────────────────────────────────────

// Ceres cost functor: reprojection error for one 2D observation.
struct ReprojectionCost {
    double obs_x, obs_y;

    ReprojectionCost(double x, double y) : obs_x(x), obs_y(y) {}

    template <typename T>
    bool operator()(const T *camera, const T *point, T *residuals) const {
        // camera[0..2]  = rvec (angle-axis)
        // camera[3..5]  = tvec
        // camera[6..9]  = fx, fy, cx, cy
        // camera[10..14]= k1, k2, p1, p2, k3

        // Rotate point
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective division
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Distortion
        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;
        T k1 = camera[10], k2 = camera[11];
        T p1 = camera[12], p2 = camera[13];
        T k3 = camera[14];
        T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        T xpp = xp * radial + T(2) * p1 * xp * yp +
                p2 * (r2 + T(2) * xp * xp);
        T ypp = yp * radial + p1 * (r2 + T(2) * yp * yp) +
                T(2) * p2 * xp * yp;

        // Pixel coordinates
        T fx = camera[6], fy = camera[7], cx = camera[8], cy = camera[9];
        T pred_x = fx * xpp + cx;
        T pred_y = fy * ypp + cy;

        residuals[0] = pred_x - T(obs_x);
        residuals[1] = pred_y - T(obs_y);
        return true;
    }

    static ceres::CostFunction *Create(double obs_x, double obs_y) {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 15, 3>(
            new ReprojectionCost(obs_x, obs_y));
    }
};

inline bool bundle_adjust(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    std::vector<CameraPose> &poses,
    std::map<int, Eigen::Vector3d> &points_3d, std::string *status) {

    int num_cameras = (int)config.cam_ordered.size();

    // Parse BA config
    double th_outliers_early = 1000.0;
    double th_outliers = 15.0;
    int max_nfev = 40;
    int max_nfev2 = 40;
    bool optimize_points = true;
    bool optimize_camera_params = true;
    bool use_bounds = false;
    std::vector<double> bounds_cp, bounds_pt;
    std::string loss_type = "linear";
    double f_scale = 1.0;

    if (!config.ba_config.is_null()) {
        const auto &ba = config.ba_config;
        th_outliers_early = ba.value("th_outliers_early", 1000.0);
        th_outliers = ba.value("th_outliers", 15.0);
        max_nfev = ba.value("max_nfev", 40);
        max_nfev2 = ba.value("max_nfev2", 40);
        optimize_points = ba.value("optimize_points", true);
        optimize_camera_params = ba.value("optimize_camera_params", true);
        use_bounds = ba.value("bounds", false);
        loss_type = ba.value("loss", std::string("linear"));
        f_scale = ba.value("f_scale", 1.0);
        if (ba.contains("bounds_cp") && ba["bounds_cp"].is_array())
            bounds_cp = ba["bounds_cp"].get<std::vector<double>>();
        if (ba.contains("bounds_pt") && ba["bounds_pt"].is_array())
            bounds_pt = ba["bounds_pt"].get<std::vector<double>>();
    }

    // Build initial 3D points from concatenated poses if not already populated
    // (points_3d should already be populated from concatenate_poses via the
    // accumulated points, but let's ensure we have a reasonable set)

    // Pack camera parameters: 15 doubles per camera
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

    // Store initial values for bounds
    auto camera_params_init = camera_params;

    // Pack 3D point parameters: ordered by global ID
    std::vector<int> point_id_order;
    for (const auto &[id, _] : points_3d)
        point_id_order.push_back(id);
    std::sort(point_id_order.begin(), point_id_order.end());

    std::map<int, int> point_id_to_param_idx;
    std::vector<std::array<double, 3>> point_params(point_id_order.size());
    for (int i = 0; i < (int)point_id_order.size(); i++) {
        int pid = point_id_order[i];
        point_id_to_param_idx[pid] = i;
        const auto &pt = points_3d[pid];
        point_params[i] = {pt.x(), pt.y(), pt.z()};
    }

    auto point_params_init = point_params;

    // Build observations: (camera_idx, point_id, pixel_x, pixel_y)
    struct Observation {
        int cam_idx;
        int point_param_idx;
        double px, py;
    };
    std::vector<Observation> observations;

    for (int c = 0; c < num_cameras; c++) {
        const auto &serial = config.cam_ordered[c];
        auto it = landmarks.find(serial);
        if (it == landmarks.end())
            continue;
        for (const auto &[pid, pixel] : it->second) {
            if (point_id_to_param_idx.count(pid)) {
                observations.push_back(
                    {c, point_id_to_param_idx[pid], pixel.x(), pixel.y()});
            }
        }
    }

    if (observations.empty()) {
        if (status)
            *status = "Error: no observations for bundle adjustment";
        return false;
    }

    // Two-pass BA: first with loose outlier threshold, then tight
    for (int pass = 0; pass < 2; pass++) {
        double outlier_th = (pass == 0) ? th_outliers_early : th_outliers;
        int max_iter = (pass == 0) ? max_nfev : max_nfev2;

        if (status)
            *status = "Bundle adjustment (pass " + std::to_string(pass + 1) +
                      "/2, " + std::to_string(observations.size()) +
                      " observations)...";

        ceres::Problem problem;

        // Track which point indices have observations in this pass
        std::set<int> active_point_indices;
        for (const auto &obs : observations) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.px, obs.py);

            ceres::LossFunction *loss = nullptr;
            if (loss_type == "huber")
                loss = new ceres::HuberLoss(f_scale);

            problem.AddResidualBlock(cost, loss,
                                     camera_params[obs.cam_idx].data(),
                                     point_params[obs.point_param_idx].data());
            active_point_indices.insert(obs.point_param_idx);
        }

        // Set bounds if configured.
        // bounds_cp[p] > 0  → constrain parameter to initial ± bound
        // bounds_cp[p] == 0 → fix parameter at initial value
        if (use_bounds && bounds_cp.size() >= 15) {
            for (int c = 0; c < num_cameras; c++) {
                // Collect indices to fix (bound == 0)
                std::vector<int> constant_indices;
                for (int p = 0; p < 15; p++) {
                    double b = bounds_cp[p];
                    if (b > 0) {
                        double init_val = camera_params_init[c][p];
                        problem.SetParameterLowerBound(
                            camera_params[c].data(), p, init_val - b);
                        problem.SetParameterUpperBound(
                            camera_params[c].data(), p, init_val + b);
                    } else if (b == 0) {
                        constant_indices.push_back(p);
                    }
                }
                // Fix zero-bounded parameters via SubsetManifold
                if (!constant_indices.empty()) {
                    problem.SetManifold(
                        camera_params[c].data(),
                        new ceres::SubsetManifold(15, constant_indices));
                }
            }
        }

        if (use_bounds && bounds_pt.size() >= 3 && optimize_points) {
            for (int i : active_point_indices) {
                for (int p = 0; p < 3; p++) {
                    double b = bounds_pt[p];
                    if (b > 0) {
                        double init_val = point_params_init[i][p];
                        problem.SetParameterLowerBound(
                            point_params[i].data(), p, init_val - b);
                        problem.SetParameterUpperBound(
                            point_params[i].data(), p, init_val + b);
                    }
                }
            }
        }

        // Fix parameters if not optimizing
        if (!optimize_camera_params) {
            for (int c = 0; c < num_cameras; c++)
                problem.SetParameterBlockConstant(camera_params[c].data());
        }
        if (!optimize_points) {
            for (int i = 0; i < (int)point_params.size(); i++)
                problem.SetParameterBlockConstant(point_params[i].data());
        }

        // Solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = max_iter;
        options.function_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;
        options.num_threads = std::max(1, (int)std::thread::hardware_concurrency());

        // Suppress CHOLMOD warnings by temporarily redirecting stderr
#ifdef _WIN32
        int saved_stderr2 = _dup(STDERR_FILENO);
        if (saved_stderr2 >= 0) {
            int devnull2 = _open("NUL", O_WRONLY);
            if (devnull2 >= 0) { _dup2(devnull2, STDERR_FILENO); _close(devnull2); }
        }
#else
        int saved_stderr2 = dup(STDERR_FILENO);
        if (saved_stderr2 >= 0) {
            int devnull2 = open("/dev/null", O_WRONLY);
            if (devnull2 >= 0) { dup2(devnull2, STDERR_FILENO); close(devnull2); }
        }
#endif
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
#ifdef _WIN32
        if (saved_stderr2 >= 0) { _dup2(saved_stderr2, STDERR_FILENO); _close(saved_stderr2); }
#else
        if (saved_stderr2 >= 0) { dup2(saved_stderr2, STDERR_FILENO); close(saved_stderr2); }
#endif

        printf("BA pass %d: %s  initial_cost=%.2f  final_cost=%.2f  "
               "iterations=%d  time=%.2fs\n",
               pass + 1, summary.IsSolutionUsable() ? "CONVERGED" : "FAILED",
               summary.initial_cost, summary.final_cost,
               (int)summary.iterations.size(), summary.total_time_in_seconds);

        // Reject outliers after first pass
        if (pass == 0) {
            std::vector<Observation> inlier_obs;
            for (const auto &obs : observations) {
                // Compute reprojection error
                Eigen::Vector3d rvec(camera_params[obs.cam_idx][0],
                                     camera_params[obs.cam_idx][1],
                                     camera_params[obs.cam_idx][2]);
                Eigen::Vector3d tvec(camera_params[obs.cam_idx][3],
                                     camera_params[obs.cam_idx][4],
                                     camera_params[obs.cam_idx][5]);
                Eigen::Matrix3d K_cam = Eigen::Matrix3d::Identity();
                K_cam(0, 0) = camera_params[obs.cam_idx][6];
                K_cam(1, 1) = camera_params[obs.cam_idx][7];
                K_cam(0, 2) = camera_params[obs.cam_idx][8];
                K_cam(1, 2) = camera_params[obs.cam_idx][9];
                Eigen::Matrix<double, 5, 1> d;
                d << camera_params[obs.cam_idx][10],
                    camera_params[obs.cam_idx][11],
                    camera_params[obs.cam_idx][12],
                    camera_params[obs.cam_idx][13],
                    camera_params[obs.cam_idx][14];

                auto &pp = point_params[obs.point_param_idx];
                Eigen::Vector3d pt3d(pp[0], pp[1], pp[2]);

                Eigen::Vector2d projected =
                    red_math::projectPoint(pt3d, rvec, tvec, K_cam, d);
                double err = std::sqrt(std::pow(projected.x() - obs.px, 2) +
                                       std::pow(projected.y() - obs.py, 2));

                if (err < outlier_th)
                    inlier_obs.push_back(obs);
            }

            int removed = (int)observations.size() - (int)inlier_obs.size();
            observations = std::move(inlier_obs);

            if (status)
                *status = "BA pass 1 done. Removed " +
                          std::to_string(removed) + " outliers (threshold=" +
                          std::to_string(outlier_th).substr(0, 6) + ")";
        }
    }

    // Log BA intrinsic changes for diagnostics
    printf("\n=== Bundle Adjustment Intrinsic Changes ===\n");
    printf("%-12s  %8s %8s %8s %8s  |  %8s %8s %8s %8s\n",
           "Camera", "fx_init", "fy_init", "cx_init", "cy_init",
           "dfx", "dfy", "dcx", "dcy");
    for (int i = 0; i < num_cameras; i++) {
        printf("%-12s  %8.1f %8.1f %8.1f %8.1f  |  %+8.2f %+8.2f %+8.2f %+8.2f\n",
               config.cam_ordered[i].c_str(),
               camera_params_init[i][6], camera_params_init[i][7],
               camera_params_init[i][8], camera_params_init[i][9],
               camera_params[i][6] - camera_params_init[i][6],
               camera_params[i][7] - camera_params_init[i][7],
               camera_params[i][8] - camera_params_init[i][8],
               camera_params[i][9] - camera_params_init[i][9]);
    }
    printf("============================================\n\n");

    // Unpack results back into poses and points_3d
    for (int i = 0; i < num_cameras; i++) {
        Eigen::Vector3d rvec(camera_params[i][0], camera_params[i][1],
                             camera_params[i][2]);
        poses[i].R = red_math::rotationVectorToMatrix(rvec);
        poses[i].t =
            Eigen::Vector3d(camera_params[i][3], camera_params[i][4],
                            camera_params[i][5]);
        poses[i].K = Eigen::Matrix3d::Identity();
        poses[i].K(0, 0) = camera_params[i][6];
        poses[i].K(1, 1) = camera_params[i][7];
        poses[i].K(0, 2) = camera_params[i][8];
        poses[i].K(1, 2) = camera_params[i][9];
        poses[i].dist << camera_params[i][10], camera_params[i][11],
            camera_params[i][12], camera_params[i][13], camera_params[i][14];
    }

    for (int i = 0; i < (int)point_id_order.size(); i++) {
        points_3d[point_id_order[i]] =
            Eigen::Vector3d(point_params[i][0], point_params[i][1],
                            point_params[i][2]);
    }

    // Compute mean reprojection error
    double total_err = 0.0;
    int total_obs = 0;
    for (const auto &obs : observations) {
        Eigen::Vector3d rvec(camera_params[obs.cam_idx][0],
                             camera_params[obs.cam_idx][1],
                             camera_params[obs.cam_idx][2]);
        Eigen::Vector3d tvec(camera_params[obs.cam_idx][3],
                             camera_params[obs.cam_idx][4],
                             camera_params[obs.cam_idx][5]);
        auto &pp = point_params[obs.point_param_idx];
        Eigen::Vector3d pt3d(pp[0], pp[1], pp[2]);

        Eigen::Vector2d projected =
            red_math::projectPoint(pt3d, rvec, tvec, poses[obs.cam_idx].K,
                                   poses[obs.cam_idx].dist);
        double err = std::sqrt(std::pow(projected.x() - obs.px, 2) +
                               std::pow(projected.y() - obs.py, 2));
        total_err += err;
        total_obs++;
    }

    double mean_err = (total_obs > 0) ? (total_err / total_obs) : 0.0;
    if (status)
        *status = "Bundle adjustment complete. Mean reproj error: " +
                  std::to_string(mean_err).substr(0, 5) + " px (" +
                  std::to_string(total_obs) + " observations)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6a: Detect ChArUco board in separate global registration media
// Extracts frame 0 from each camera's video (or loads one image per camera),
// detects the board, triangulates corners using calibrated poses, and returns
// 3D points for Procrustes alignment.
// ─────────────────────────────────────────────────────────────────────────────

inline bool detect_global_reg_board(
    const CalibrationTool::CalibConfig &config,
    const std::string &global_reg_folder,
    const std::string &global_reg_type, // "videos" or "images"
    const std::vector<CameraPose> &poses,
    const std::vector<std::string> &cam_names, // cam serials matching poses order
    std::vector<Eigen::Vector3d> &out_pts_3d,  // triangulated board corners
    std::vector<int> &out_corner_ids,          // which corner IDs were triangulated
    std::string *status) {

    const auto &cs = config.charuco_setup;
    aruco_detect::CharucoBoard board;
    board.squares_x = cs.w; board.squares_y = cs.h;
    board.square_length = cs.square_side_length;
    board.marker_length = cs.marker_side_length;
    board.dictionary_id = cs.dictionary;
    auto aruco_dict = aruco_detect::getDictionary(cs.dictionary);
    int max_corners = (cs.w - 1) * (cs.h - 1);

    // Per-camera detections: corner_id → 2D point
    struct CamDetection {
        std::map<int, Eigen::Vector2d> corners; // corner_id → (x,y)
        bool valid = false;
    };
    std::vector<CamDetection> cam_dets(cam_names.size());

    if (status) *status = "Detecting board in global registration media...";
    fprintf(stderr, "[GlobalReg] Detecting board in %s (%s)...\n",
            global_reg_folder.c_str(), global_reg_type.c_str());

    // Detect board in each camera (parallel)
    std::vector<std::future<void>> futures;
    for (int ci = 0; ci < (int)cam_names.size(); ci++) {
        futures.push_back(std::async(std::launch::async, [&, ci]() {
            const std::string &serial = cam_names[ci];
            auto &det = cam_dets[ci];
            int w = 0, h = 0;
            std::vector<uint8_t> gray;

            if (global_reg_type == "videos") {
                // Find the video for this camera
                auto vids = CalibrationTool::discover_aruco_videos(
                    global_reg_folder, {serial});
                if (vids.empty()) {
                    fprintf(stderr, "[GlobalReg]   %s: no video found in %s\n",
                            serial.c_str(), global_reg_folder.c_str());
                    return;
                }
                const std::string &video_path = vids.begin()->second;
                fprintf(stderr, "[GlobalReg]   %s: opening %s\n",
                        serial.c_str(), video_path.c_str());

#ifdef __APPLE__
                // Decode frame 0 via FFmpeg + VideoToolbox
                std::unique_ptr<FFmpegDemuxer> demuxer;
                try { demuxer = std::make_unique<FFmpegDemuxer>(
                    video_path.c_str(), std::map<std::string, std::string>{});
                } catch (...) {
                    fprintf(stderr, "[GlobalReg]   %s: FFmpegDemuxer failed\n", serial.c_str());
                    return;
                }
                w = (int)demuxer->GetWidth(); h = (int)demuxer->GetHeight();
                gray.resize(w * h);

                VTAsyncDecoder vt;
                if (!vt.init(demuxer->GetExtradata(), demuxer->GetExtradataSize(),
                             demuxer->GetVideoCodec())) {
                    fprintf(stderr, "[GlobalReg]   %s: VTAsyncDecoder init failed\n", serial.c_str());
                    return;
                }

                // Decode first frame — submit multiple packets to fill reorder queue
                uint8_t *pkt_data = nullptr; size_t pkt_size = 0; PacketData pkt_info;
                CVPixelBufferRef pb = nullptr;
                for (int pkt_i = 0; pkt_i < 16 && !pb; pkt_i++) {
                    if (!demuxer->Demux(pkt_data, pkt_size, pkt_info)) {
                        fprintf(stderr, "[GlobalReg]   %s: Demux failed at packet %d\n",
                                serial.c_str(), pkt_i);
                        break;
                    }
                    bool is_key = (pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                    vt.submit_blocking(pkt_data, pkt_size, pkt_info.pts,
                                       pkt_info.dts, demuxer->GetTimebase(), is_key);
                    pb = vt.drain_one();
                }
                if (!pb) {
                    fprintf(stderr, "[GlobalReg]   %s: no frame decoded after 16 packets\n",
                            serial.c_str());
                    return;
                }
                fprintf(stderr, "[GlobalReg]   %s: decoded frame %dx%d\n",
                        serial.c_str(), w, h);

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
                // Images: find image for this camera (first image with this serial)
                namespace fs = std::filesystem;
                std::string found_path;
                for (const auto &entry : fs::directory_iterator(global_reg_folder)) {
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

            // Detect ChArUco board (full-res, same as calibration path)
            auto charuco = aruco_detect::detectCharucoBoard(
                gray.data(), w, h, board, aruco_dict,
                nullptr, nullptr, nullptr, 0, 1);
            if ((int)charuco.ids.size() < 4) {
                fprintf(stderr, "[GlobalReg]   %s: only %d corners (need 4), skipping\n",
                        serial.c_str(), (int)charuco.ids.size());
                return;
            }
            aruco_detect::cornerSubPix(gray.data(), w, h, charuco.corners, 3, 100, 0.001f);

            for (int j = 0; j < (int)charuco.ids.size(); j++) {
                det.corners[charuco.ids[j]] = Eigen::Vector2d(
                    charuco.corners[j].x(), charuco.corners[j].y());
            }
            det.valid = true;
            fprintf(stderr, "[GlobalReg]   %s: %d corners detected\n",
                    serial.c_str(), (int)charuco.ids.size());
        }));
    }
    for (auto &f : futures) f.get();

    // Count valid cameras
    int valid_cams = 0;
    for (const auto &d : cam_dets) if (d.valid) valid_cams++;
    if (valid_cams < 2) {
        if (status) *status = "Error: board detected in fewer than 2 cameras for global registration";
        return false;
    }
    fprintf(stderr, "[GlobalReg] Board detected in %d / %d cameras\n",
            valid_cams, (int)cam_names.size());

    // Triangulate each corner from all cameras that see it
    for (int cid = 0; cid < max_corners; cid++) {
        // Collect 2D observations and corresponding camera indices
        std::vector<Eigen::Vector2d> obs;
        std::vector<int> cam_indices;
        for (int ci = 0; ci < (int)cam_names.size(); ci++) {
            if (!cam_dets[ci].valid) continue;
            auto it = cam_dets[ci].corners.find(cid);
            if (it != cam_dets[ci].corners.end()) {
                obs.push_back(it->second);
                cam_indices.push_back(ci);
            }
        }
        if ((int)obs.size() < 2) continue;

        // DLT triangulation from multiple views
        Eigen::MatrixXd A(2 * (int)obs.size(), 4);
        for (int k = 0; k < (int)obs.size(); k++) {
            int ci = cam_indices[k];
            Eigen::Matrix<double, 3, 4> P;
            P.block<3,3>(0,0) = poses[ci].R;
            P.col(3) = poses[ci].t;
            P = poses[ci].K * P;
            double u = obs[k].x(), v = obs[k].y();
            A.row(2*k)   = u * P.row(2) - P.row(0);
            A.row(2*k+1) = v * P.row(2) - P.row(1);
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d X = svd.matrixV().col(3);
        if (std::abs(X(3)) < 1e-10) continue;
        Eigen::Vector3d pt = X.head<3>() / X(3);

        out_pts_3d.push_back(pt);
        out_corner_ids.push_back(cid);
    }

    fprintf(stderr, "[GlobalReg] Triangulated %d board corners from global reg media\n",
            (int)out_pts_3d.size());
    return !out_pts_3d.empty();
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6: Global registration (Procrustes alignment to world frame)
// ─────────────────────────────────────────────────────────────────────────────

inline bool global_registration(
    const CalibrationTool::CalibConfig &config,
    std::vector<CameraPose> &poses,
    std::map<int, Eigen::Vector3d> &points_3d, std::string *status,
    const std::vector<int> *frame_numbers = nullptr,
    const std::string &global_reg_folder = {},
    const std::string &global_reg_type = {},
    const std::vector<std::string> *cam_names_for_global_reg = nullptr) {

    if (config.gt_pts.empty() || config.world_coordinate_imgs.empty()) {
        if (status)
            *status = "Skipping global registration (no gt_pts or "
                      "world_coordinate_imgs in config)";
        return true; // Not an error, just skip
    }

    int max_corners =
        (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);

    // Collect corresponding pairs: BA 3D point ↔ world coordinate
    std::vector<Eigen::Vector3d> src_pts; // BA frame
    std::vector<Eigen::Vector3d> dst_pts; // world frame (gt_pts)

    // Try separate global registration media first (if provided)
    fprintf(stderr, "[GlobalReg] global_reg_folder='%s' type='%s' cam_names=%s\n",
            global_reg_folder.c_str(), global_reg_type.c_str(),
            cam_names_for_global_reg ? std::to_string(cam_names_for_global_reg->size()).c_str() : "null");
    if (!global_reg_folder.empty() && cam_names_for_global_reg) {
        std::vector<Eigen::Vector3d> reg_pts_3d;
        std::vector<int> reg_corner_ids;
        if (detect_global_reg_board(config, global_reg_folder, global_reg_type,
                                     poses, *cam_names_for_global_reg,
                                     reg_pts_3d, reg_corner_ids, status)) {
            // Match triangulated corners against gt_pts
            // gt_pts keys are world_coordinate_imgs names; corners are indexed 0..max_corners-1
            for (const auto &img_name : config.world_coordinate_imgs) {
                auto gt_it = config.gt_pts.find(img_name);
                if (gt_it == config.gt_pts.end()) continue;
                const auto &gt = gt_it->second;
                for (int k = 0; k < (int)reg_corner_ids.size(); k++) {
                    int cid = reg_corner_ids[k];
                    if (cid < (int)gt.size()) {
                        src_pts.push_back(reg_pts_3d[k]);
                        dst_pts.push_back(Eigen::Vector3d(gt[cid][0], gt[cid][1], gt[cid][2]));
                    }
                }
            }
            if (src_pts.size() >= 3) {
                fprintf(stderr, "[GlobalReg] Using %d points from separate global reg media\n",
                        (int)src_pts.size());
            } else {
                fprintf(stderr, "[GlobalReg] Only %d points from global reg media (need 3), falling back\n",
                        (int)src_pts.size());
            }
        } else {
            fprintf(stderr, "[GlobalReg] detect_global_reg_board FAILED\n");
        }
    } else {
        fprintf(stderr, "[GlobalReg] No global_reg_folder provided, using calibration frames\n");
    }

    // Fallback: try using frames from the calibration data itself
    if (src_pts.size() < 3) {
        fprintf(stderr, "[GlobalReg] Trying fallback: frames from calibration data...\n");
        std::map<int, int> img_num_to_idx;
        if (frame_numbers && !frame_numbers->empty()) {
            for (int i = 0; i < (int)frame_numbers->size(); i++)
                img_num_to_idx[(*frame_numbers)[i]] = i;
        } else {
            auto image_numbers =
                get_sorted_image_numbers(config.img_path, config.cam_ordered[0]);
            for (int i = 0; i < (int)image_numbers.size(); i++)
                img_num_to_idx[image_numbers[i]] = i;
        }

        for (const auto &img_name : config.world_coordinate_imgs) {
            int img_num = std::stoi(img_name);
            auto it = img_num_to_idx.find(img_num);
            if (it == img_num_to_idx.end())
                continue;
            int img_idx = it->second;

            auto gt_it = config.gt_pts.find(img_name);
            if (gt_it == config.gt_pts.end())
                continue;
            const auto &gt = gt_it->second;

            for (int corner_id = 0;
                 corner_id < max_corners && corner_id < (int)gt.size();
                 corner_id++) {
                int global_id = img_idx * max_corners + corner_id;
                auto pt_it = points_3d.find(global_id);
                if (pt_it != points_3d.end()) {
                    src_pts.push_back(pt_it->second);
                    dst_pts.push_back(Eigen::Vector3d(gt[corner_id][0],
                                                      gt[corner_id][1],
                                                      gt[corner_id][2]));
                }
            }
        }
    }

    if (src_pts.size() < 3) {
        if (frame_numbers && !frame_numbers->empty()) {
            if (status) *status = "Skipping global registration (world_coordinate_imgs "
                                  "not in video frame range and no separate global reg media)";
            return true;
        }
        if (status)
            *status = "Error: only " + std::to_string(src_pts.size()) +
                      " matching points for global registration (need ≥3)";
        return false;
    }

    if (status)
        *status = "Aligning to world frame (" +
                  std::to_string(src_pts.size()) + " ground truth points)...";

    // Procrustes alignment: find R, s, t such that dst ≈ s*R*src + t
    int n = (int)src_pts.size();

    // 1. Compute centroids
    Eigen::Vector3d src_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d dst_mean = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; i++) {
        src_mean += src_pts[i];
        dst_mean += dst_pts[i];
    }
    src_mean /= n;
    dst_mean /= n;

    // 2. Center the points
    Eigen::MatrixXd S(n, 3), D(n, 3);
    for (int i = 0; i < n; i++) {
        S.row(i) = (src_pts[i] - src_mean).transpose();
        D.row(i) = (dst_pts[i] - dst_mean).transpose();
    }

    // 3. Cross-covariance matrix M = D^T * S
    Eigen::Matrix3d M = D.transpose() * S;

    // 4. SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation (det > 0)
    Eigen::Matrix3d R_reg = U * V.transpose();
    if (R_reg.determinant() < 0) {
        V.col(2) *= -1;
        R_reg = U * V.transpose();
    }

    // 5. Scale: s = trace(Sigma) / trace(S^T * S)
    double src_var = 0.0;
    for (int i = 0; i < n; i++)
        src_var += S.row(i).squaredNorm();
    if (src_var < 1e-12) {
        if (status) *status = "Error: 3D point variance too small for alignment";
        return false;
    }
    double scale_reg = svd.singularValues().sum() / src_var;

    // 6. Translation: t = dst_mean - s * R * src_mean
    Eigen::Vector3d t_reg = dst_mean - scale_reg * R_reg * src_mean;

    // Apply transformation to all camera poses and 3D points
    // Original: X_cam = R_old * X_world_old + t_old
    // New world: X_world_new = scale * R_reg * X_world_old + t_reg
    // So: X_world_old = R_reg^T * (X_world_new - t_reg) / scale
    // X_cam = R_old * R_reg^T * (X_world_new - t_reg) / scale + t_old
    //       = (R_old * R_reg^T / scale) * X_world_new
    //         + t_old - R_old * R_reg^T * t_reg / scale
    // But that mixes scale into the extrinsics. The standard approach:
    //
    // We want the new extrinsics (R_new, t_new) such that:
    //   X_cam = R_new * X_world_new + t_new
    // where X_world_new = s * R_reg * X_world_old + t_reg
    //
    // X_cam = R_old * X_world_old + t_old
    //       = R_old * R_reg^T * (X_world_new - t_reg) / s + t_old
    //       = (R_old * R_reg^T / s) * X_world_new
    //         + t_old - R_old * R_reg^T * t_reg / s
    //
    // So: R_new = R_old * R_reg^T  (ignoring scale in R)
    //     t_new = scale_reg * R_new * ( ... )
    //
    // Actually the cleaner formulation: the camera center in old world coords
    // is C_old = -R_old^T * t_old. In new world coords:
    // C_new = scale_reg * R_reg * C_old + t_reg
    // Then: t_new = -R_new * C_new
    //       R_new = R_old * R_reg^T

    for (int i = 0; i < (int)poses.size(); i++) {
        // Camera center in old world frame
        Eigen::Vector3d C_old = -poses[i].R.transpose() * poses[i].t;
        // Transform to new world frame
        Eigen::Vector3d C_new = scale_reg * R_reg * C_old + t_reg;
        // New rotation
        Eigen::Matrix3d R_new = poses[i].R * R_reg.transpose();
        // New translation
        Eigen::Vector3d t_new = -R_new * C_new;

        poses[i].R = R_new;
        poses[i].t = t_new;
    }

    // Transform 3D points
    for (auto &[id, pt] : points_3d) {
        pt = scale_reg * R_reg * pt + t_reg;
    }

    // Apply optional post-Procrustes world frame rotation (e.g., MVC convention)
    if (!config.world_frame_rotation.isIdentity(1e-10)) {
        Eigen::Matrix3d W = config.world_frame_rotation;
        fprintf(stderr, "[Experimental] Applying world_frame_rotation from config\n");
        for (int i = 0; i < (int)poses.size(); i++) {
            Eigen::Vector3d C = -poses[i].R.transpose() * poses[i].t;
            C = W * C;  // rotate camera center
            poses[i].R = poses[i].R * W.transpose();
            poses[i].t = -poses[i].R * C;
        }
        for (auto &[id, pt] : points_3d)
            pt = W * pt;
    }

    // Report registration error (Procrustes alignment quality, before world_frame_rotation)
    double reg_err = 0.0;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d transformed =
            scale_reg * R_reg * src_pts[i] + t_reg;
        reg_err += (transformed - dst_pts[i]).norm();
    }
    reg_err /= n;

    if (status)
        *status = "Global registration complete. Mean error: " +
                  std::to_string(reg_err).substr(0, 6) + " mm (" +
                  std::to_string(n) + " points, scale=" +
                  std::to_string(scale_reg).substr(0, 6) + ")";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 7: Write per-camera YAML calibration files
// ─────────────────────────────────────────────────────────────────────────────

inline bool write_calibration(
    const std::vector<CameraPose> &poses,
    const std::vector<std::string> &cam_names,
    const std::string &output_folder, int image_width, int image_height,
    std::string *status) {

    namespace fs = std::filesystem;

    // Create output directory
    std::error_code ec;
    fs::create_directories(output_folder, ec);
    if (ec) {
        if (status)
            *status = "Error: cannot create output folder: " + ec.message();
        return false;
    }

    for (int i = 0; i < (int)poses.size(); i++) {
        std::string filename =
            output_folder + "/Cam" + cam_names[i] + ".yaml";

        try {
            opencv_yaml::YamlWriter writer(filename);

            writer.writeScalar("image_width", image_width);
            writer.writeScalar("image_height", image_height);

            // Camera matrix (3×3)
            writer.writeMatrix("camera_matrix", poses[i].K);

            // Distortion coefficients (5×1)
            Eigen::MatrixXd dist_mat(5, 1);
            for (int j = 0; j < 5; j++)
                dist_mat(j, 0) = poses[i].dist(j);
            writer.writeMatrix("distortion_coefficients", dist_mat);

            // Translation (3×1)
            Eigen::MatrixXd t_mat(3, 1);
            t_mat(0, 0) = poses[i].t.x();
            t_mat(1, 0) = poses[i].t.y();
            t_mat(2, 0) = poses[i].t.z();
            writer.writeMatrix("tc_ext", t_mat);

            // Rotation (3×3)
            writer.writeMatrix("rc_ext", poses[i].R);

            writer.close();
        } catch (const std::exception &e) {
            if (status)
                *status =
                    "Error writing " + filename + ": " + std::string(e.what());
            return false;
        }
    }

    if (status)
        *status = "Wrote " + std::to_string(poses.size()) +
                  " calibration files to " + output_folder;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Write intermediate output (matches multiview_calib output/ structure)
// ─────────────────────────────────────────────────────────────────────────────

// Write run_info.json to the summary_data folder.
// Records all calibration settings for reproducibility.
inline void write_run_info(
    const CalibrationTool::CalibConfig &config,
    const std::string &output_folder,
    const VideoFrameRange *vfr,
    int total_video_frames,
    int cameras_used,
    double mean_reproj,
    double detection_sec,
    double ba_sec,
    double total_sec) {

    namespace fs = std::filesystem;
    std::string dir = output_folder + "/summary_data";
    std::error_code ec;
    fs::create_directories(dir, ec);

    nlohmann::json j;
    // Timestamp
    {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char buf[64];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        j["timestamp"] = buf;
    }

    // Input source
    j["input_type"] = vfr ? "videos" : "images";
    j["media_folder"] = vfr ? vfr->video_folder : config.img_path;
    j["cam_ordered"] = config.cam_ordered;
    j["cameras_used"] = cameras_used;

    // Video frame parameters
    if (vfr) {
        j["start_frame"] = vfr->start_frame;
        j["stop_frame"] = vfr->stop_frame;
        j["frame_step"] = vfr->frame_step;
        j["total_video_frames"] = total_video_frames;
        int eff_stop = vfr->stop_frame > 0 ? vfr->stop_frame : total_video_frames;
        int step = std::max(1, vfr->frame_step);
        j["sampled_frames_per_camera"] = (eff_stop - vfr->start_frame + step - 1) / step;
    }

    // Board configuration
    j["charuco_setup"] = {
        {"w", config.charuco_setup.w},
        {"h", config.charuco_setup.h},
        {"square_side_length", config.charuco_setup.square_side_length},
        {"marker_side_length", config.charuco_setup.marker_side_length},
        {"dictionary", config.charuco_setup.dictionary}
    };

    // Global registration
    j["global_reg_media_folder"] = config.global_reg_media_folder;
    j["global_reg_media_type"] = config.global_reg_media_type;
    j["world_coordinate_imgs"] = config.world_coordinate_imgs;
    j["has_gt_pts"] = !config.gt_pts.empty();

    // Results summary
    j["mean_reproj_error"] = mean_reproj;

    // Timing
    j["timing"] = {
        {"detection_sec", detection_sec},
        {"ba_sec", ba_sec},
        {"total_sec", total_sec}
    };

    std::string path = dir + "/run_info.json";
    std::ofstream f(path);
    if (f.is_open()) {
        f << j.dump(2);
        fprintf(stderr, "[Calibration] Wrote %s\n", path.c_str());
    }
}

inline bool write_intermediate_output(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    const std::vector<CameraPose> &poses,
    const std::map<int, Eigen::Vector3d> &points_3d,
    const std::string &output_folder) {

    namespace fs = std::filesystem;
    std::string output_dir = output_folder + "/summary_data";
    std::string ba_dir = output_dir + "/bundle_adjustment";
    std::string intr_dir = output_dir + "/intrinsics";

    std::error_code ec;
    fs::create_directories(ba_dir, ec);
    fs::create_directories(intr_dir, ec);

    int num_cameras = (int)config.cam_ordered.size();

    // ── intrinsics.json ──
    {
        nlohmann::json j;
        for (const auto &serial : config.cam_ordered) {
            auto it = intrinsics.find(serial);
            if (it == intrinsics.end()) continue;
            const auto &intr = it->second;
            nlohmann::json cam_j;
            cam_j["K"] = {
                {intr.K(0,0), intr.K(0,1), intr.K(0,2)},
                {intr.K(1,0), intr.K(1,1), intr.K(1,2)},
                {intr.K(2,0), intr.K(2,1), intr.K(2,2)}
            };
            cam_j["dist"] = {intr.dist(0), intr.dist(1), intr.dist(2),
                             intr.dist(3), intr.dist(4)};
            cam_j["reprojection_error"] = intr.reproj_error;
            cam_j["image_width"] = intr.image_width;
            cam_j["image_height"] = intr.image_height;
            j[serial] = cam_j;
        }
        std::ofstream f(output_dir + "/intrinsics.json");
        f << j.dump(2);
    }

    // ── Per-camera intrinsic YAML files ──
    for (const auto &serial : config.cam_ordered) {
        auto it = intrinsics.find(serial);
        if (it == intrinsics.end()) continue;
        const auto &intr = it->second;
        std::string filename = intr_dir + "/" + serial + ".yaml";
        opencv_yaml::YamlWriter writer(filename);
        writer.writeScalar("image_width", intr.image_width);
        writer.writeScalar("image_height", intr.image_height);
        writer.writeMatrix("camera_matrix", intr.K);
        Eigen::MatrixXd dist_mat(5, 1);
        for (int j = 0; j < 5; j++) dist_mat(j, 0) = intr.dist(j);
        writer.writeMatrix("distortion_coefficients", dist_mat);
        writer.close();
    }

    // ── landmarks.json ──
    {
        nlohmann::json j;
        for (const auto &serial : config.cam_ordered) {
            auto it = landmarks.find(serial);
            if (it == landmarks.end()) continue;
            nlohmann::json cam_j;
            std::vector<int> ids;
            std::vector<std::vector<double>> pts;
            for (const auto &[pid, pixel] : it->second) {
                ids.push_back(pid);
                pts.push_back({pixel.x(), pixel.y()});
            }
            cam_j["ids"] = ids;
            cam_j["landmarks"] = pts;
            j[serial] = cam_j;
        }
        std::ofstream f(output_dir + "/landmarks.json");
        f << j.dump(2);
    }

    // ── bundle_adjustment/ba_poses.json ──
    {
        nlohmann::json j;
        for (int i = 0; i < num_cameras; i++) {
            const auto &serial = config.cam_ordered[i];
            nlohmann::json cam_j;
            // R as 3x3 row-major array
            nlohmann::json R_j = nlohmann::json::array();
            for (int r = 0; r < 3; r++) {
                nlohmann::json row = nlohmann::json::array();
                for (int c = 0; c < 3; c++) row.push_back(poses[i].R(r, c));
                R_j.push_back(row);
            }
            cam_j["R"] = R_j;
            cam_j["t"] = {poses[i].t.x(), poses[i].t.y(), poses[i].t.z()};
            nlohmann::json K_j = nlohmann::json::array();
            for (int r = 0; r < 3; r++) {
                nlohmann::json row = nlohmann::json::array();
                for (int c = 0; c < 3; c++) row.push_back(poses[i].K(r, c));
                K_j.push_back(row);
            }
            cam_j["K"] = K_j;
            cam_j["dist"] = {poses[i].dist(0), poses[i].dist(1),
                             poses[i].dist(2), poses[i].dist(3),
                             poses[i].dist(4)};
            j[serial] = cam_j;
        }
        std::ofstream f(ba_dir + "/ba_poses.json");
        f << j.dump(2);
    }

    // ── bundle_adjustment/ba_points.json ──
    {
        nlohmann::json j;
        for (const auto &[id, pt] : points_3d) {
            j[std::to_string(id)] = {pt.x(), pt.y(), pt.z()};
        }
        std::ofstream f(ba_dir + "/ba_points.json");
        f << j.dump(2);
    }

    // ── bundle_adjustment/bundle_adjustment.log ──
    {
        std::ofstream f(ba_dir + "/bundle_adjustment.log");
        f << "Reprojection errors (mean+-std pixels):\n";

        double global_total = 0;
        int global_count = 0;

        for (int c = 0; c < num_cameras; c++) {
            const auto &serial = config.cam_ordered[c];
            auto lm_it = landmarks.find(serial);
            if (lm_it == landmarks.end()) continue;

            Eigen::Vector3d rvec =
                red_math::rotationMatrixToVector(poses[c].R);

            std::vector<double> errs;
            for (const auto &[pid, pixel] : lm_it->second) {
                auto pt_it = points_3d.find(pid);
                if (pt_it == points_3d.end()) continue;
                Eigen::Vector2d projected = red_math::projectPoint(
                    pt_it->second, rvec, poses[c].t, poses[c].K,
                    poses[c].dist);
                errs.push_back((projected - pixel).norm());
            }

            if (errs.empty()) continue;

            double mean = 0;
            for (double e : errs) mean += e;
            mean /= errs.size();

            double var = 0;
            for (double e : errs) var += (e - mean) * (e - mean);
            double std_dev = std::sqrt(var / errs.size());

            // Median
            std::sort(errs.begin(), errs.end());
            double median = errs[errs.size() / 2];

            f << "\t " << serial << " n_points=" << errs.size()
              << ": " << std::fixed << std::setprecision(3)
              << mean << "+-" << std_dev
              << " (median=" << median << ")\n";

            global_total += mean * errs.size();
            global_count += errs.size();
        }

        double global_mean = (global_count > 0) ?
            (global_total / global_count) : 0.0;
        f << "Average absolute residual: " << std::fixed
          << std::setprecision(2) << global_mean
          << " over " << global_count << " points.\n";
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Write CalibrationDatabase to JSON for 3D viewer inspection
// ─────────────────────────────────────────────────────────────────────────────

inline void write_calibration_database(const CalibrationDatabase &db,
    const std::string &output_folder,
    const std::vector<PerCameraMetrics> *metrics = nullptr) {
    nlohmann::json j;
    j["version"] = 1;

    // Board poses: camera → {frame → {R, t, reproj, corners}}
    nlohmann::json bp_j;
    for (const auto &[cam, frames] : db.board_poses) {
        nlohmann::json cam_j;
        for (const auto &[fi, bp] : frames) {
            nlohmann::json pose_j;
            pose_j["reproj"] = bp.reproj;
            pose_j["corners"] = bp.num_corners;
            pose_j["t"] = {bp.t.x(), bp.t.y(), bp.t.z()};
            nlohmann::json R_j = nlohmann::json::array();
            for (int r = 0; r < 3; r++) {
                nlohmann::json row = nlohmann::json::array();
                for (int c = 0; c < 3; c++) row.push_back(bp.R(r, c));
                R_j.push_back(row);
            }
            pose_j["R"] = R_j;
            cam_j[std::to_string(fi)] = pose_j;
        }
        bp_j[cam] = cam_j;
    }
    j["board_poses"] = bp_j;

    // Registration order
    nlohmann::json reg_j = nlohmann::json::array();
    for (const auto &step : db.registration_order) {
        reg_j.push_back({{"camera", step.camera_name}, {"idx", step.camera_idx},
            {"parent", step.parent_camera}, {"frames", step.num_shared_frames},
            {"points", step.num_3d_points}, {"method", step.method}});
    }
    j["registration_order"] = reg_j;

    // BA passes
    nlohmann::json ba_j = nlohmann::json::array();
    for (const auto &p : db.ba_passes) {
        ba_j.push_back({{"pass", p.pass_number}, {"mode", p.fix_mode},
            {"cauchy", p.cauchy_scale}, {"cost_before", p.cost_before},
            {"cost_after", p.cost_after}, {"iters", p.iterations},
            {"time", p.time_sec}, {"outliers", p.outliers_removed}});
    }
    j["ba_passes"] = ba_j;

    // Per-observation residuals (compact: parallel arrays)
    if (!db.residuals.empty()) {
        nlohmann::json res_j;
        std::vector<int> cam_idxs, lm_ids;
        std::vector<float> errors;
        for (const auto &r : db.residuals) {
            cam_idxs.push_back(r.camera_idx);
            lm_ids.push_back(r.landmark_id);
            errors.push_back(r.error);
        }
        res_j["camera_idx"] = cam_idxs;
        res_j["landmark_id"] = lm_ids;
        res_j["error"] = errors;
        res_j["count"] = (int)db.residuals.size();
        j["residuals"] = res_j;
    }

    j["timing"] = {{"detection_sec", db.detection_time_sec},
                    {"ba_sec", db.ba_time_sec},
                    {"total_sec", db.total_time_sec}};

    // Per-camera metrics (intrinsic reproj, detection count)
    if (metrics && !metrics->empty()) {
        nlohmann::json mcam_j;
        for (const auto &m : *metrics) {
            mcam_j[m.name] = {{"intrinsic_reproj", m.intrinsic_reproj},
                              {"detection_count", m.detection_count}};
        }
        j["per_camera_metrics"] = mcam_j;
    }

    std::ofstream f(output_folder + "/calibration_data.json");
    if (f.is_open()) f << j.dump(2);
}

// Check if a calibration_data.json exists in a folder
inline bool has_calibration_database(const std::string &folder) {
    if (folder.empty()) return false;
    // Check the folder itself and the most recent timestamped subfolder
    namespace fs = std::filesystem;
    if (fs::exists(folder + "/calibration_data.json")) return true;
    // Search for most recent timestamped subfolder
    if (!fs::is_directory(folder)) return false;
    std::string latest;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            if (name.size() >= 10 && name[4] == '_') { // timestamp format
                if (name > latest) latest = name;
            }
        }
    }
    if (!latest.empty() && fs::exists(folder + "/" + latest + "/calibration_data.json"))
        return true;
    return false;
}

// Find the most recent calibration_data.json in a folder
inline std::string find_calibration_database_path(const std::string &folder) {
    namespace fs = std::filesystem;
    if (fs::exists(folder + "/calibration_data.json"))
        return folder + "/calibration_data.json";
    if (!fs::is_directory(folder)) return "";
    std::string latest;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            if (name.size() >= 10 && name[4] == '_') {
                if (name > latest) latest = name;
            }
        }
    }
    if (!latest.empty()) {
        std::string path = folder + "/" + latest + "/calibration_data.json";
        if (fs::exists(path)) return path;
    }
    return "";
}

// Load a CalibrationResult from an output folder (YAML files + calibration_data.json)
// ─────────────────────────────────────────────────────────────────────────────
// Global multi-view triangulation consistency.
// For each landmark seen by 2+ cameras, triangulate from all views (DLT),
// then reproject to each camera. This measures global extrinsic consistency —
// the metric that actually matters for 3D tracking accuracy.
// ─────────────────────────────────────────────────────────────────────────────
inline CalibrationResult::GlobalConsistency compute_global_consistency(
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    const std::vector<CameraPose> &poses,
    const std::vector<std::string> &cam_names) {

    CalibrationResult::GlobalConsistency gc;
    int nc = (int)cam_names.size();
    if (nc == 0 || landmarks.empty()) return gc;

    // Group observations by landmark ID
    std::map<int, std::vector<std::pair<int, Eigen::Vector2d>>> pobs;
    for (int c = 0; c < nc; c++) {
        auto it = landmarks.find(cam_names[c]);
        if (it == landmarks.end()) continue;
        for (const auto &[pid, px] : it->second)
            pobs[pid].push_back({c, px});
    }

    // Per-camera error accumulation
    std::vector<double> cam_err_sum(nc, 0.0);
    std::vector<int> cam_err_count(nc, 0);
    std::vector<double> all_errors;

    for (const auto &[pid, obs] : pobs) {
        if ((int)obs.size() < 2) continue;

        // Triangulate from all views
        std::vector<Eigen::Vector2d> pts_undist;
        std::vector<Eigen::Matrix<double, 3, 4>> Ps;
        for (const auto &[ci, px] : obs) {
            pts_undist.push_back(red_math::undistortPoint(px, poses[ci].K, poses[ci].dist));
            Ps.push_back(red_math::projectionFromKRt(poses[ci].K, poses[ci].R, poses[ci].t));
        }
        Eigen::Vector3d X = red_math::triangulatePoints(pts_undist, Ps);

        // Reproject to each camera
        for (const auto &[ci, px] : obs) {
            auto rv = red_math::rotationMatrixToVector(poses[ci].R);
            auto pr = red_math::projectPoint(X, rv, poses[ci].t, poses[ci].K, poses[ci].dist);
            double err = (pr - px).norm();
            all_errors.push_back(err);
            cam_err_sum[ci] += err;
            cam_err_count[ci]++;
        }
        gc.landmarks_triangulated++;
    }

    if (all_errors.empty()) return gc;

    // Compute global statistics
    gc.total_observations = (int)all_errors.size();
    std::sort(all_errors.begin(), all_errors.end());
    double sum = 0;
    for (double e : all_errors) sum += e;
    gc.mean_reproj = sum / all_errors.size();
    gc.median_reproj = all_errors[all_errors.size() / 2];
    gc.pct95_reproj = all_errors[(int)(all_errors.size() * 0.95)];

    // Per-camera
    gc.per_camera.resize(nc);
    for (int c = 0; c < nc; c++) {
        gc.per_camera[c].name = cam_names[c];
        gc.per_camera[c].obs = cam_err_count[c];
        gc.per_camera[c].mean_reproj = cam_err_count[c] > 0
            ? cam_err_sum[c] / cam_err_count[c] : 0.0;
    }

    gc.computed = true;
    fprintf(stderr, "[Calib] Global consistency: mean=%.2f px, median=%.2f px, "
            "95pct=%.2f px (%d landmarks, %d obs)\n",
            gc.mean_reproj, gc.median_reproj, gc.pct95_reproj,
            gc.landmarks_triangulated, gc.total_observations);
    return gc;
}

inline CalibrationResult load_calibration_from_folder(
    const std::string &folder, const std::vector<std::string> &cam_names) {
    CalibrationResult result;
    namespace fs = std::filesystem;

    // Find the actual timestamped subfolder
    std::string data_folder = folder;
    if (!fs::exists(folder + "/calibration_data.json")) {
        std::string latest;
        if (fs::is_directory(folder)) {
            for (const auto &entry : fs::directory_iterator(folder)) {
                if (entry.is_directory()) {
                    std::string name = entry.path().filename().string();
                    if (name.size() >= 10 && name[4] == '_' && name > latest)
                        latest = name;
                }
            }
        }
        if (!latest.empty()) data_folder = folder + "/" + latest;
    }

    std::string db_path = data_folder + "/calibration_data.json";
    if (!fs::exists(db_path)) { result.error = "No calibration_data.json"; return result; }

    // Load YAML camera files — skip cameras without YAML files gracefully
    std::vector<std::string> skipped_cameras;
    for (int i = 0; i < (int)cam_names.size(); i++) {
        std::string yaml_path = data_folder + "/Cam" + cam_names[i] + ".yaml";
        if (!fs::exists(yaml_path)) {
            skipped_cameras.push_back(cam_names[i]);
            continue;
        }
        try {
            auto yf = opencv_yaml::read(yaml_path);
            CameraPose pose;
            pose.K = yf.getMatrix("camera_matrix");
            Eigen::MatrixXd dist_mat = yf.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5 && j < (int)dist_mat.rows(); j++)
                pose.dist(j) = dist_mat(j, 0);
            pose.R = yf.getMatrix("rc_ext");
            Eigen::MatrixXd t_mat = yf.getMatrix("tc_ext");
            pose.t = Eigen::Vector3d(t_mat(0,0), t_mat(1,0), t_mat(2,0));
            int iw = yf.getInt("image_width"), ih = yf.getInt("image_height");
            if (result.cameras.empty()) { result.image_width = iw; result.image_height = ih; }
            result.cameras.push_back(pose);
            result.cam_names.push_back(cam_names[i]);
        } catch (const std::exception &e) {
            skipped_cameras.push_back(cam_names[i]);
            fprintf(stderr, "[load_calibration] Skipping %s: %s\n",
                    cam_names[i].c_str(), e.what());
        }
    }
    if (result.cameras.empty()) {
        result.error = "No camera YAML files found in " + data_folder;
        return result;
    }
    if (!skipped_cameras.empty()) {
        std::string skip_msg = "Skipped " + std::to_string(skipped_cameras.size()) +
            " camera(s) (no YAML): ";
        for (size_t i = 0; i < skipped_cameras.size(); i++) {
            if (i > 0) skip_msg += ", ";
            skip_msg += skipped_cameras[i];
        }
        result.warning = skip_msg;
        fprintf(stderr, "[load_calibration] %s\n", skip_msg.c_str());
    }

    // Load database JSON
    try {
        std::ifstream dbf(db_path);
        nlohmann::json j;
        dbf >> j;

        // Board poses
        if (j.contains("board_poses")) {
            for (auto &[cam, frames_j] : j["board_poses"].items()) {
                for (auto &[fi_str, pose_j] : frames_j.items()) {
                    CalibrationDatabase::BoardPose bp;
                    bp.reproj = pose_j.value("reproj", 0.0);
                    bp.num_corners = pose_j.value("corners", 0);
                    auto &t_arr = pose_j["t"];
                    bp.t = Eigen::Vector3d(t_arr[0], t_arr[1], t_arr[2]);
                    auto &R_arr = pose_j["R"];
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            bp.R(r, c) = R_arr[r][c];
                    result.db.board_poses[cam][std::stoi(fi_str)] = bp;
                }
            }
        }

        // Registration order
        if (j.contains("registration_order")) {
            for (const auto &s : j["registration_order"]) {
                result.db.registration_order.push_back({
                    s.value("camera", ""), s.value("idx", -1),
                    s.value("parent", ""), s.value("frames", 0),
                    s.value("points", 0), s.value("method", "")});
            }
        }

        // BA passes
        if (j.contains("ba_passes")) {
            for (const auto &p : j["ba_passes"]) {
                result.db.ba_passes.push_back({
                    p.value("pass", 0), p.value("mode", 0),
                    p.value("cauchy", 1.0), p.value("cost_before", 0.0),
                    p.value("cost_after", 0.0), p.value("iters", 0),
                    p.value("time", 0.0), p.value("outliers", 0)});
            }
        }

        // Landmarks from summary_data (if available)
        std::string lm_path = data_folder + "/summary_data/landmarks.json";
        if (fs::exists(lm_path)) {
            std::ifstream lmf(lm_path);
            nlohmann::json lm_j;
            lmf >> lm_j;
            for (auto &[cam, cam_j] : lm_j.items()) {
                auto &ids = cam_j["ids"];
                auto &pts = cam_j["landmarks"];
                for (int i = 0; i < (int)ids.size(); i++) {
                    result.db.landmarks[cam][ids[i].get<int>()] =
                        Eigen::Vector2d(pts[i][0].get<double>(), pts[i][1].get<double>());
                }
            }
        }

        // 3D points from summary_data
        std::string pts_path = data_folder + "/summary_data/bundle_adjustment/ba_points.json";
        if (fs::exists(pts_path)) {
            std::ifstream pf(pts_path);
            nlohmann::json pts_j;
            pf >> pts_j;
            for (auto &[id_str, pt] : pts_j.items()) {
                result.points_3d[std::stoi(id_str)] =
                    Eigen::Vector3d(pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>());
            }
        }

        // Load saved per-camera metrics (intrinsic reproj, etc.) if available
        nlohmann::json saved_metrics;
        if (j.contains("per_camera_metrics"))
            saved_metrics = j["per_camera_metrics"];

        // Compute per-camera metrics from landmarks + points.
        // Use result.cam_names (loaded cameras only, may be fewer than input).
        int nc = (int)result.cam_names.size();
        result.per_camera_metrics.resize(nc);
        for (int c = 0; c < nc; c++) {
            auto &m = result.per_camera_metrics[c];
            m.name = result.cam_names[c];
            auto bp_it = result.db.board_poses.find(result.cam_names[c]);
            if (bp_it != result.db.board_poses.end())
                m.detection_count = (int)bp_it->second.size();
            // Load intrinsic reproj from saved data (not recomputable from YAMLs)
            if (saved_metrics.contains(m.name))
                m.intrinsic_reproj = saved_metrics[m.name].value("intrinsic_reproj", 0.0);
            auto lm_it = result.db.landmarks.find(result.cam_names[c]);
            if (lm_it != result.db.landmarks.end()) {
                m.observation_count = (int)lm_it->second.size();
                // Recompute reprojection error from loaded poses + 3D points.
                // This uses camera poses from YAML (may have been flipped) and
                // 3D points from ba_points.json (may not be flipped), so only
                // recompute if both are available and consistent.
                if (!result.points_3d.empty() && c < (int)result.cameras.size()) {
                    // Use projectPointR (matrix-based) instead of projectPoint
                    // (Rodrigues-based) to handle improper rotations (det=-1)
                    // from Z-flip transforms safely.
                    std::vector<double> errs;
                    for (const auto &[pid, px] : lm_it->second) {
                        auto pt_it = result.points_3d.find(pid);
                        if (pt_it == result.points_3d.end()) continue;
                        double e = (red_math::projectPointR(pt_it->second,
                            result.cameras[c].R, result.cameras[c].t,
                            result.cameras[c].K,
                            result.cameras[c].dist) - px).norm();
                        errs.push_back(e);
                    }
                    if (!errs.empty()) {
                        double s = 0; for (double e : errs) s += e;
                        m.mean_reproj = s / errs.size();
                        std::sort(errs.begin(), errs.end());
                        m.median_reproj = errs[errs.size() / 2];
                        m.max_reproj = errs.back();
                    }
                }
            }
        }

        // Compute overall mean
        double total_err = 0; int total_obs = 0;
        for (const auto &m : result.per_camera_metrics) {
            total_err += m.mean_reproj * m.observation_count;
            total_obs += m.observation_count;
        }
        result.mean_reproj_error = (total_obs > 0) ? total_err / total_obs : 0;

    } catch (const std::exception &e) {
        result.error = std::string("JSON parse error: ") + e.what();
        return result;
    }

    // Compute global multi-view consistency if landmarks are available
    if (!result.db.landmarks.empty() && !result.cameras.empty()) {
        result.global_consistency = compute_global_consistency(
            result.db.landmarks, result.cameras, result.cam_names);
    }

    result.output_folder = data_folder;
    result.success = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Experimental pipeline: PnP-all-then-BA (no spanning tree)
// ═══════════════════════════════════════════════════════════════════════════════

// Ceres cost functor for per-frame PnP refinement (6 DOF: angle-axis + translation).
// Intrinsics are fixed; only extrinsics are optimized.
struct PnPRefineCost {
    double obs_x, obs_y, obj_x, obj_y, obj_z;
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;
    PnPRefineCost(double ox, double oy, double objx, double objy, double objz,
                  const Eigen::Matrix3d &K, const Eigen::Matrix<double,5,1> &d)
        : obs_x(ox), obs_y(oy), obj_x(objx), obj_y(objy), obj_z(objz),
          fx(K(0,0)), fy(K(1,1)), cx(K(0,2)), cy(K(1,2)),
          k1(d(0)), k2(d(1)), p1(d(2)), p2(d(3)), k3(d(4)) {}
    template<typename T> bool operator()(const T *pose, T *residuals) const {
        T pt[3] = {T(obj_x), T(obj_y), T(obj_z)};
        T p[3]; ceres::AngleAxisRotatePoint(pose, pt, p);
        p[0]+=pose[3]; p[1]+=pose[4]; p[2]+=pose[5];
        T xp=p[0]/p[2], yp=p[1]/p[2];
        T r2=xp*xp+yp*yp, r4=r2*r2, r6=r4*r2;
        T rad=T(1)+T(k1)*r2+T(k2)*r4+T(k3)*r6;
        T xpp=xp*rad+T(2)*T(p1)*xp*yp+T(p2)*(r2+T(2)*xp*xp);
        T ypp=yp*rad+T(p1)*(r2+T(2)*yp*yp)+T(2)*T(p2)*xp*yp;
        residuals[0]=T(fx)*xpp+T(cx)-T(obs_x);
        residuals[1]=T(fy)*ypp+T(cy)-T(obs_y);
        return true;
    }
};

// Refine PnP pose using Ceres LM (minimizes geometric reprojection error).
// obj_pts_3d: board 3D points, img_pts: pixel observations (distorted).
// K, dist: intrinsics (fixed). R, t: initial pose (refined in-place).
inline void refinePnPPose(
    const std::vector<Eigen::Vector3d> &obj_pts_3d,
    const std::vector<Eigen::Vector2d> &img_pts,
    const Eigen::Matrix3d &K, const Eigen::Matrix<double,5,1> &dist,
    Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    if ((int)obj_pts_3d.size() < 4) return;
    Eigen::Vector3d rvec = red_math::rotationMatrixToVector(R);
    double pose[6] = {rvec.x(), rvec.y(), rvec.z(), t.x(), t.y(), t.z()};
    ceres::Problem problem;
    for (int i = 0; i < (int)obj_pts_3d.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PnPRefineCost, 2, 6>(
                new PnPRefineCost(img_pts[i].x(), img_pts[i].y(),
                    obj_pts_3d[i].x(), obj_pts_3d[i].y(), obj_pts_3d[i].z(), K, dist)),
            nullptr, pose);
    }
    ceres::Solver::Options opt;
    opt.linear_solver_type = ceres::DENSE_QR;
    opt.max_num_iterations = 20;
    opt.function_tolerance = 1e-12;
    opt.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary sum;
    ceres::Solve(opt, &problem, &sum);
    R = red_math::rotationVectorToMatrix(Eigen::Vector3d(pose[0], pose[1], pose[2]));
    t = Eigen::Vector3d(pose[3], pose[4], pose[5]);
}

inline bool solvePnPHomography(
    const std::vector<Eigen::Vector2d> &obj_pts_2d,
    const std::vector<Eigen::Vector2d> &img_pts_und,
    const Eigen::Matrix3d &K,
    Eigen::Matrix3d &R_out, Eigen::Vector3d &t_out) {
    if ((int)obj_pts_2d.size() < 4) return false;
    Eigen::Matrix3d H = intrinsic_calib::estimateHomography(obj_pts_2d, img_pts_und);
    Eigen::Matrix3d M = K.inverse() * H;
    Eigen::Vector3d r1 = M.col(0), r2 = M.col(1), t_raw = M.col(2);
    double avg_norm = (r1.norm() + r2.norm()) * 0.5;
    if (avg_norm < 1e-10) return false;
    r1 /= avg_norm; r2 /= avg_norm; t_raw /= avg_norm;
    Eigen::Matrix3d R_approx;
    R_approx.col(0) = r1; R_approx.col(1) = r2; R_approx.col(2) = r1.cross(r2);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_approx, Eigen::ComputeFullU|Eigen::ComputeFullV);
    R_out = svd.matrixU() * svd.matrixV().transpose();
    if (R_out.determinant() < 0) { auto V=svd.matrixV(); V.col(2)*=-1; R_out=svd.matrixU()*V.transpose(); }
    t_out = t_raw;
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (const auto &p : obj_pts_2d) c += p;
    c /= obj_pts_2d.size();
    if ((R_out * Eigen::Vector3d(c.x(),c.y(),0) + t_out).z() < 0) { R_out=-R_out; t_out=-t_out; if(R_out.determinant()<0) R_out=-R_out; }
    return true;
}

inline bool initialize_extrinsics_pnp(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    std::vector<CameraPose> &poses, std::string *status,
    CalibrationDatabase *db = nullptr) {
    int num_cameras = (int)config.cam_ordered.size();
    const auto &cs = config.charuco_setup;
    poses.resize(num_cameras);
    int cx = cs.w-1, cy = cs.h-1, tc = cx*cy;
    std::vector<Eigen::Vector2d> board2d(tc);
    std::vector<Eigen::Vector3d> board3d(tc);
    for (int y=0;y<cy;y++) for (int x=0;x<cx;x++) {
        int i=y*cx+x; double bx=x*cs.square_side_length, by=y*cs.square_side_length;
        board2d[i]=Eigen::Vector2d(bx,by); board3d[i]=Eigen::Vector3d(bx,by,0);
    }
    struct FramePose { Eigen::Matrix3d R; Eigen::Vector3d t; double reproj; };
    std::vector<std::map<int,FramePose>> pnp(num_cameras);
    for (int ci=0;ci<num_cameras;ci++) {
        const auto &intr = intrinsics.at(config.cam_ordered[ci]);
        for (const auto &[fi,corners] : intr.corners_per_image) {
            const auto &ids = intr.ids_per_image.at(fi);
            if ((int)ids.size()<4) continue;
            std::vector<Eigen::Vector2d> o2d,iund; std::vector<Eigen::Vector2d> iraw;
            std::vector<Eigen::Vector3d> o3d;
            for (int j=0;j<(int)ids.size();j++) { int c=ids[j]; if(c<0||c>=tc) continue;
                o2d.push_back(board2d[c]); o3d.push_back(board3d[c]);
                Eigen::Vector2d px(corners[j].x(),corners[j].y());
                iund.push_back(red_math::undistortPoint(px,intr.K,intr.dist)); iraw.push_back(px);
            }
            if ((int)o2d.size()<4) continue;
            Eigen::Matrix3d Rf; Eigen::Vector3d tf;
            if (!solvePnPHomography(o2d,iund,intr.K,Rf,tf)) continue;
            // Refine algebraic PnP with Ceres LM (geometric reprojection minimization)
            refinePnPPose(o3d, iraw, intr.K, intr.dist, Rf, tf);
            auto rv=red_math::rotationMatrixToVector(Rf);
            auto pr=red_math::projectPoints(o3d,rv,tf,intr.K,intr.dist);
            double es=0; for(int i=0;i<(int)pr.size();i++) es+=(pr[i]-iraw[i]).norm();
            double me=es/pr.size();
            if (me<10.0) pnp[ci][fi]={Rf,tf,me};
        }
    }
    // Save board poses to database
    if (db) {
        for (int ci = 0; ci < num_cameras; ci++) {
            const auto &serial = config.cam_ordered[ci];
            for (const auto &[fi, fp] : pnp[ci]) {
                db->board_poses[serial][fi] = {fp.R, fp.t, fp.reproj,
                    (int)intrinsics.at(serial).corners_per_image.count(fi) ?
                    (int)intrinsics.at(serial).corners_per_image.at(fi).size() : 0};
            }
        }
    }
    // ── Greedy incremental registration (COLMAP-inspired) ──
    // Step 1: Find best initial pair (most shared frames with both reproj < 2.0)
    int best_a = -1, best_b = -1, best_shared = 0;
    for (int a = 0; a < num_cameras; a++) {
        for (int b = a + 1; b < num_cameras; b++) {
            int shared = 0;
            for (const auto &[fi, fp_a] : pnp[a]) {
                auto it = pnp[b].find(fi);
                if (it != pnp[b].end() && fp_a.reproj < 2.0 && it->second.reproj < 2.0)
                    shared++;
            }
            if (shared > best_shared) { best_shared = shared; best_a = a; best_b = b; }
        }
    }
    if (best_a < 0 || best_shared < 1) {
        if (status) *status = "Error: no camera pair shares frames";
        return false;
    }

    // Initialize pair: camera A at identity, B via multi-frame Markley averaging
    std::vector<bool> init(num_cameras, false);
    int ic = 0;

    // Helper: compute relative pose from camera ci to camera bi using Markley averaging
    auto compute_relative_markley = [&](int ci, int bi, Eigen::Matrix3d &Rout, Eigen::Vector3d &tout) -> bool {
        struct RelP { Eigen::Matrix3d R; Eigen::Vector3d t; double err; };
        std::vector<RelP> cands;
        for (const auto &[fi, fp] : pnp[ci]) {
            auto it = pnp[bi].find(fi);
            if (it == pnp[bi].end()) continue;
            if (fp.reproj >= 2.0 || it->second.reproj >= 2.0) continue;
            Eigen::Matrix3d Rr = fp.R * it->second.R.transpose();
            Eigen::Vector3d tr = fp.t - Rr * it->second.t;
            cands.push_back({Rr, tr, fp.reproj + it->second.reproj});
        }
        if (cands.empty()) {
            // Fallback: best single frame without filter
            double be = 1e10;
            for (const auto &[fi, fp] : pnp[ci]) {
                auto it = pnp[bi].find(fi);
                if (it == pnp[bi].end()) continue;
                Eigen::Matrix3d Rr = fp.R * it->second.R.transpose();
                Eigen::Vector3d tr = fp.t - Rr * it->second.t;
                double ce = fp.reproj + it->second.reproj;
                if (ce < be) { be = ce; Rout = Rr; tout = tr; }
            }
            return be < 1e9;
        }
        if (cands.size() == 1) { Rout = cands[0].R; tout = cands[0].t; return true; }
        // Markley weighted quaternion averaging
        Eigen::Matrix4d Mqat = Eigen::Matrix4d::Zero();
        for (size_t i = 0; i < cands.size(); i++) {
            Eigen::Quaterniond qi(cands[i].R); qi.normalize();
            if (i > 0) { Eigen::Quaterniond q0(cands[0].R); q0.normalize();
                if (qi.dot(q0) < 0.0) qi.coeffs() = -qi.coeffs(); }
            double w = 1.0 / (1.0 + cands[i].err * cands[i].err);
            Eigen::Vector4d qv = qi.coeffs();
            Mqat += w * qv * qv.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(Mqat);
        Eigen::Vector4d bq = eig.eigenvectors().col(3);
        Eigen::Quaterniond avgQ(bq(3), bq(0), bq(1), bq(2));
        avgQ.normalize(); Rout = avgQ.toRotationMatrix();
        Eigen::Vector3d avgT = Eigen::Vector3d::Zero(); double ws = 0;
        for (size_t i = 0; i < cands.size(); i++) {
            double w = 1.0 / (1.0 + cands[i].err * cands[i].err);
            avgT += w * cands[i].t; ws += w;
        }
        tout = avgT / ws;
        return true;
    };

    // Initialize camera A at identity
    const auto &intr_a = intrinsics.at(config.cam_ordered[best_a]);
    poses[best_a] = {Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), intr_a.K, intr_a.dist};
    init[best_a] = true; ic++;

    // Initialize camera B via relative pose to A
    Eigen::Matrix3d Rab; Eigen::Vector3d tab;
    if (!compute_relative_markley(best_b, best_a, Rab, tab)) {
        if (status) *status = "Error: cannot compute relative pose for initial pair";
        return false;
    }
    const auto &intr_b = intrinsics.at(config.cam_ordered[best_b]);
    poses[best_b] = {Rab, tab, intr_b.K, intr_b.dist};
    init[best_b] = true; ic++;
    fprintf(stderr, "[Experimental]   Initial pair: %s + %s (%d shared frames)\n",
        config.cam_ordered[best_a].c_str(), config.cam_ordered[best_b].c_str(), best_shared);
    if (db) {
        db->registration_order.push_back({config.cam_ordered[best_a], best_a, "origin", best_shared, 0, "initial_pair"});
        db->registration_order.push_back({config.cam_ordered[best_b], best_b, config.cam_ordered[best_a], best_shared, 0, "initial_pair"});
    }

    // Step 2: Triangulate initial 3D point cloud from the pair
    // Build landmarks for just these two cameras
    int max_corners = (cs.w - 1) * (cs.h - 1);
    std::map<int, Eigen::Vector3d> init_points;
    {
        const auto &intr_aa = intrinsics.at(config.cam_ordered[best_a]);
        const auto &intr_bb = intrinsics.at(config.cam_ordered[best_b]);
        // For each frame where both cameras detect the board
        for (const auto &[fi, corners_a] : intr_aa.corners_per_image) {
            auto it_b = intr_bb.corners_per_image.find(fi);
            if (it_b == intr_bb.corners_per_image.end()) continue;
            const auto &ids_a = intr_aa.ids_per_image.at(fi);
            const auto &ids_b = intr_bb.ids_per_image.at(fi);
            const auto &corners_b = it_b->second;
            // Find common corner IDs
            std::map<int, Eigen::Vector2d> obs_a, obs_b;
            for (int j = 0; j < (int)ids_a.size(); j++)
                obs_a[ids_a[j]] = Eigen::Vector2d(corners_a[j].x(), corners_a[j].y());
            for (int j = 0; j < (int)ids_b.size(); j++)
                obs_b[ids_b[j]] = Eigen::Vector2d(corners_b[j].x(), corners_b[j].y());
            for (const auto &[cid, px_a] : obs_a) {
                auto it = obs_b.find(cid);
                if (it == obs_b.end()) continue;
                int global_id = fi * max_corners + cid;
                if (init_points.count(global_id)) continue;
                // Triangulate
                Eigen::Vector2d ua = red_math::undistortPoint(px_a, poses[best_a].K, poses[best_a].dist);
                Eigen::Vector2d ub = red_math::undistortPoint(it->second, poses[best_b].K, poses[best_b].dist);
                auto Pa = red_math::projectionFromKRt(poses[best_a].K, poses[best_a].R, poses[best_a].t);
                auto Pb = red_math::projectionFromKRt(poses[best_b].K, poses[best_b].R, poses[best_b].t);
                auto X = red_math::triangulatePoints({ua, ub}, {Pa, Pb});
                // Check reproj in both cameras
                auto rva = red_math::rotationMatrixToVector(poses[best_a].R);
                auto rvb = red_math::rotationMatrixToVector(poses[best_b].R);
                double ea = (red_math::projectPoint(X, rva, poses[best_a].t, poses[best_a].K, poses[best_a].dist) - px_a).norm();
                double eb = (red_math::projectPoint(X, rvb, poses[best_b].t, poses[best_b].K, poses[best_b].dist) - it->second).norm();
                if (ea < 10.0 && eb < 10.0) init_points[global_id] = X;
            }
        }
    }
    fprintf(stderr, "[Experimental]   Initial triangulation: %d points\n", (int)init_points.size());

    // Step 3: Greedily register remaining cameras
    // Each iteration: pick unregistered camera with most 2D correspondences to existing 3D points,
    // compute its pose via multi-frame Markley averaging against ALL initialized cameras,
    // triangulate new points, optionally run quick BA.
    while (ic < num_cameras) {
        // Score each unregistered camera: count landmarks it observes that are in init_points
        int best_ci = -1, best_count = 0;
        for (int ci = 0; ci < num_cameras; ci++) {
            if (init[ci]) continue;
            const auto &intr_ci = intrinsics.at(config.cam_ordered[ci]);
            int count = 0;
            for (const auto &[fi, corners_ci] : intr_ci.corners_per_image) {
                const auto &ids_ci = intr_ci.ids_per_image.at(fi);
                for (int j = 0; j < (int)ids_ci.size(); j++) {
                    int global_id = fi * max_corners + ids_ci[j];
                    if (init_points.count(global_id)) count++;
                }
            }
            if (count > best_count) { best_count = count; best_ci = ci; }
        }

        if (best_ci < 0 || best_count < 4) {
            // No camera has enough 3D correspondences — fall back to multi-frame
            // relative pose against any initialized camera
            for (int ci = 0; ci < num_cameras; ci++) {
                if (init[ci]) continue;
                const auto &intr_ci = intrinsics.at(config.cam_ordered[ci]);
                // Try all initialized cameras, pick best
                double best_err = 1e10;
                Eigen::Matrix3d best_R; Eigen::Vector3d best_t;
                for (int bi = 0; bi < num_cameras; bi++) {
                    if (!init[bi]) continue;
                    Eigen::Matrix3d Rcb; Eigen::Vector3d tcb;
                    if (!compute_relative_markley(ci, bi, Rcb, tcb)) continue;
                    // Chain: ci in world = Rcb * bi_world
                    Eigen::Matrix3d Rw = Rcb * poses[bi].R;
                    Eigen::Vector3d tw = Rcb * poses[bi].t + tcb;
                    // Score by counting inlier reproj against init_points
                    auto rv = red_math::rotationMatrixToVector(Rw);
                    int inliers = 0;
                    for (const auto &[fi, corners_ci] : intr_ci.corners_per_image) {
                        const auto &ids_ci = intr_ci.ids_per_image.at(fi);
                        for (int j = 0; j < (int)ids_ci.size(); j++) {
                            int gid = fi * max_corners + ids_ci[j];
                            auto pt = init_points.find(gid);
                            if (pt == init_points.end()) continue;
                            double e = (red_math::projectPoint(pt->second, rv, tw, intr_ci.K, intr_ci.dist) -
                                       Eigen::Vector2d(corners_ci[j].x(), corners_ci[j].y())).norm();
                            if (e < 5.0) inliers++;
                        }
                    }
                    double err = (inliers > 0) ? 1.0 / inliers : 1e10;
                    if (err < best_err) { best_err = err; best_R = Rw; best_t = tw; }
                }
                if (best_err < 1e9) {
                    poses[ci] = {best_R, best_t, intr_ci.K, intr_ci.dist};
                    init[ci] = true; ic++;
                    fprintf(stderr, "[Experimental]   PnP %s: via bridge (fallback)\n",
                        config.cam_ordered[ci].c_str());
                }
            }
            break; // exit greedy loop
        }

        // Register best_ci via multi-frame Markley averaging against ALL initialized cameras
        const auto &intr_ci = intrinsics.at(config.cam_ordered[best_ci]);
        struct RelP { Eigen::Matrix3d R; Eigen::Vector3d t; double err; };
        std::vector<RelP> all_cands;
        for (int bi = 0; bi < num_cameras; bi++) {
            if (!init[bi]) continue;
            for (const auto &[fi, fp_ci] : pnp[best_ci]) {
                auto it = pnp[bi].find(fi);
                if (it == pnp[bi].end()) continue;
                if (fp_ci.reproj >= 2.0 || it->second.reproj >= 2.0) continue;
                // Relative pose ci→bi in board frame, then chain to world
                Eigen::Matrix3d Rcb = fp_ci.R * it->second.R.transpose();
                Eigen::Vector3d tcb = fp_ci.t - Rcb * it->second.t;
                Eigen::Matrix3d Rw = Rcb * poses[bi].R;
                Eigen::Vector3d tw = Rcb * poses[bi].t + tcb;
                all_cands.push_back({Rw, tw, fp_ci.reproj + it->second.reproj});
            }
        }

        if (all_cands.empty()) {
            // No good frames with any initialized camera — try unfiltered
            double be = 1e10; Eigen::Matrix3d bR; Eigen::Vector3d bt;
            for (int bi = 0; bi < num_cameras; bi++) {
                if (!init[bi]) continue;
                for (const auto &[fi, fp_ci] : pnp[best_ci]) {
                    auto it = pnp[bi].find(fi);
                    if (it == pnp[bi].end()) continue;
                    Eigen::Matrix3d Rcb = fp_ci.R * it->second.R.transpose();
                    Eigen::Vector3d tcb = fp_ci.t - Rcb * it->second.t;
                    Eigen::Matrix3d Rw = Rcb * poses[bi].R;
                    Eigen::Vector3d tw = Rcb * poses[bi].t + tcb;
                    double ce = fp_ci.reproj + it->second.reproj;
                    if (ce < be) { be = ce; bR = Rw; bt = tw; }
                }
            }
            if (be < 1e9) {
                poses[best_ci] = {bR, bt, intr_ci.K, intr_ci.dist};
                init[best_ci] = true; ic++;
                fprintf(stderr, "[Experimental]   PnP %s: fallback single frame (err %.2f)\n",
                    config.cam_ordered[best_ci].c_str(), be);
            }
        } else if (all_cands.size() == 1) {
            poses[best_ci] = {all_cands[0].R, all_cands[0].t, intr_ci.K, intr_ci.dist};
            init[best_ci] = true; ic++;
            fprintf(stderr, "[Experimental]   PnP %s: 1 frame from init cameras (err %.2f)\n",
                config.cam_ordered[best_ci].c_str(), all_cands[0].err);
        } else {
            // Markley averaging across ALL candidates from ALL initialized cameras
            Eigen::Matrix4d Mqat = Eigen::Matrix4d::Zero();
            for (size_t i = 0; i < all_cands.size(); i++) {
                Eigen::Quaterniond qi(all_cands[i].R); qi.normalize();
                if (i > 0) { Eigen::Quaterniond q0(all_cands[0].R); q0.normalize();
                    if (qi.dot(q0) < 0.0) qi.coeffs() = -qi.coeffs(); }
                double w = 1.0 / (1.0 + all_cands[i].err * all_cands[i].err);
                Eigen::Vector4d qv = qi.coeffs();
                Mqat += w * qv * qv.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(Mqat);
            Eigen::Vector4d bq = eig.eigenvectors().col(3);
            Eigen::Quaterniond avgQ(bq(3), bq(0), bq(1), bq(2));
            avgQ.normalize();
            Eigen::Vector3d avgT = Eigen::Vector3d::Zero(); double ws = 0;
            for (size_t i = 0; i < all_cands.size(); i++) {
                double w = 1.0 / (1.0 + all_cands[i].err * all_cands[i].err);
                avgT += w * all_cands[i].t; ws += w;
            }
            poses[best_ci] = {avgQ.toRotationMatrix(), avgT / ws, intr_ci.K, intr_ci.dist};
            init[best_ci] = true; ic++;
            fprintf(stderr, "[Experimental]   PnP %s: %d frames from %d+ init cameras (%d 3D pts)\n",
                config.cam_ordered[best_ci].c_str(), (int)all_cands.size(),
                ic - 1, best_count);
            if (db) db->registration_order.push_back({config.cam_ordered[best_ci], best_ci,
                config.cam_ordered[best_a], (int)all_cands.size(), best_count, "markley_avg"});
        }

        // Triangulate new points visible from newly registered camera and existing cameras
        if (init[best_ci]) {
            const auto &intr_new = intrinsics.at(config.cam_ordered[best_ci]);
            for (const auto &[fi, corners_new] : intr_new.corners_per_image) {
                const auto &ids_new = intr_new.ids_per_image.at(fi);
                for (int j = 0; j < (int)ids_new.size(); j++) {
                    int global_id = fi * max_corners + ids_new[j];
                    if (init_points.count(global_id)) continue; // already have this point
                    Eigen::Vector2d px_new(corners_new[j].x(), corners_new[j].y());
                    // Find another initialized camera that also sees this corner in this frame
                    for (int bi = 0; bi < num_cameras; bi++) {
                        if (!init[bi] || bi == best_ci) continue;
                        const auto &intr_bi = intrinsics.at(config.cam_ordered[bi]);
                        auto fi_it = intr_bi.corners_per_image.find(fi);
                        if (fi_it == intr_bi.corners_per_image.end()) continue;
                        const auto &ids_bi = intr_bi.ids_per_image.at(fi);
                        const auto &corners_bi = fi_it->second;
                        for (int k = 0; k < (int)ids_bi.size(); k++) {
                            if (ids_bi[k] != ids_new[j]) continue;
                            Eigen::Vector2d px_bi(corners_bi[k].x(), corners_bi[k].y());
                            auto ua = red_math::undistortPoint(px_new, poses[best_ci].K, poses[best_ci].dist);
                            auto ub = red_math::undistortPoint(px_bi, poses[bi].K, poses[bi].dist);
                            auto Pa = red_math::projectionFromKRt(poses[best_ci].K, poses[best_ci].R, poses[best_ci].t);
                            auto Pb = red_math::projectionFromKRt(poses[bi].K, poses[bi].R, poses[bi].t);
                            auto X = red_math::triangulatePoints({ua, ub}, {Pa, Pb});
                            auto rva = red_math::rotationMatrixToVector(poses[best_ci].R);
                            double e = (red_math::projectPoint(X, rva, poses[best_ci].t, poses[best_ci].K, poses[best_ci].dist) - px_new).norm();
                            if (e < 10.0) { init_points[global_id] = X; break; }
                        }
                        if (init_points.count(global_id)) break;
                    }
                }
            }
        }
    }

    if (ic < num_cameras) {
        if (status) *status = "Error: not all cameras initialized (" + std::to_string(ic) + "/" + std::to_string(num_cameras) + ")";
        return false;
    }
    if (status) *status = "PnP init done (" + std::to_string(num_cameras) + " cameras, " + std::to_string(init_points.size()) + " points)";
    return true;
}

inline int triangulate_landmarks_multiview(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string,std::map<int,Eigen::Vector2d>> &landmarks,
    const std::vector<CameraPose> &poses,
    std::map<int,Eigen::Vector3d> &points_3d, double max_reproj=5.0) {
    int nc=(int)config.cam_ordered.size();
    std::map<int,std::vector<std::pair<int,Eigen::Vector2d>>> pobs;
    for (int c=0;c<nc;c++){auto it=landmarks.find(config.cam_ordered[c]);if(it==landmarks.end())continue;
        for(const auto&[pid,px]:it->second) pobs[pid].push_back({c,px});}
    int acc=0;
    for (const auto&[pid,obs]:pobs){
        if((int)obs.size()<2)continue;
        std::vector<Eigen::Vector2d> p2; std::vector<Eigen::Matrix<double,3,4>> Ps;
        for(const auto&[ci,px]:obs){p2.push_back(red_math::undistortPoint(px,poses[ci].K,poses[ci].dist));
            Ps.push_back(red_math::projectionFromKRt(poses[ci].K,poses[ci].R,poses[ci].t));}
        auto X=red_math::triangulatePoints(p2,Ps);
        bool ok=true;
        for(int i=0;i<(int)obs.size();i++){int ci=obs[i].first;
            auto rv=red_math::rotationMatrixToVector(poses[ci].R);
            auto pr=red_math::projectPoint(X,rv,poses[ci].t,poses[ci].K,poses[ci].dist);
            if((pr-obs[i].second).norm()>max_reproj){ok=false;break;}}
        if(ok){points_3d[pid]=X;acc++;}
    }
    return acc;
}

inline bool bundle_adjust_experimental(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string,std::map<int,Eigen::Vector2d>> &landmarks,
    std::vector<CameraPose> &poses,
    std::map<int,Eigen::Vector3d> &points_3d,
    CalibrationResult &result, std::string *status) {
    int nc=(int)config.cam_ordered.size();

    // Helper: pack camera params from poses
    auto pack_cameras = [&](std::vector<std::array<double,15>> &cp) {
        cp.resize(nc);
        for(int i=0;i<nc;i++){auto rv=red_math::rotationMatrixToVector(poses[i].R);
            cp[i]={rv.x(),rv.y(),rv.z(),poses[i].t.x(),poses[i].t.y(),poses[i].t.z(),
                   poses[i].K(0,0),poses[i].K(1,1),poses[i].K(0,2),poses[i].K(1,2),
                   poses[i].dist(0),poses[i].dist(1),poses[i].dist(2),poses[i].dist(3),poses[i].dist(4)};}
    };
    // Helper: unpack camera params back to poses
    auto unpack_cameras = [&](const std::vector<std::array<double,15>> &cp) {
        for(int i=0;i<nc;i++){Eigen::Vector3d rv(cp[i][0],cp[i][1],cp[i][2]);
            poses[i].R=red_math::rotationVectorToMatrix(rv);poses[i].t=Eigen::Vector3d(cp[i][3],cp[i][4],cp[i][5]);
            poses[i].K=Eigen::Matrix3d::Identity();poses[i].K(0,0)=cp[i][6];poses[i].K(1,1)=cp[i][7];poses[i].K(0,2)=cp[i][8];poses[i].K(1,2)=cp[i][9];
            poses[i].dist<<cp[i][10],cp[i][11],cp[i][12],cp[i][13],cp[i][14];}
    };
    // Helper: pack 3D points
    auto pack_points = [&](std::vector<int> &pio, std::map<int,int> &pidx,
                           std::vector<std::array<double,3>> &pp) {
        pio.clear(); for(const auto&[id,_]:points_3d) pio.push_back(id);
        std::sort(pio.begin(),pio.end());
        pidx.clear(); pp.resize(pio.size());
        for(int i=0;i<(int)pio.size();i++){pidx[pio[i]]=i;const auto&pt=points_3d[pio[i]];pp[i]={pt.x(),pt.y(),pt.z()};}
    };
    // Helper: unpack points
    auto unpack_points = [&](const std::vector<int> &pio, const std::vector<std::array<double,3>> &pp) {
        for(int i=0;i<(int)pio.size();i++)points_3d[pio[i]]=Eigen::Vector3d(pp[i][0],pp[i][1],pp[i][2]);
    };

    struct Obs{int ci,pi;double px,py;};
    auto build_observations = [&](const std::map<int,int> &pidx) {
        std::vector<Obs> obs;
        for(int c=0;c<nc;c++){auto it=landmarks.find(config.cam_ordered[c]);if(it==landmarks.end())continue;
            for(const auto&[pid,px]:it->second){auto pit=pidx.find(pid);if(pit!=pidx.end())obs.push_back({c,pit->second,px.x(),px.y()});}}
        return obs;
    };

    // Helper: compute reprojection errors for all observations
    auto compute_errors = [&](const std::vector<Obs> &obs, const std::vector<std::array<double,15>> &cp,
                              const std::vector<std::array<double,3>> &pp) {
        std::vector<double> errors; errors.reserve(obs.size());
        for(const auto&o:obs){
            Eigen::Vector3d rv(cp[o.ci][0],cp[o.ci][1],cp[o.ci][2]),tv(cp[o.ci][3],cp[o.ci][4],cp[o.ci][5]);
            Eigen::Matrix3d K=Eigen::Matrix3d::Identity();K(0,0)=cp[o.ci][6];K(1,1)=cp[o.ci][7];K(0,2)=cp[o.ci][8];K(1,2)=cp[o.ci][9];
            Eigen::Matrix<double,5,1> d;d<<cp[o.ci][10],cp[o.ci][11],cp[o.ci][12],cp[o.ci][13],cp[o.ci][14];
            auto&p=pp[o.pi];auto pr=red_math::projectPoint(Eigen::Vector3d(p[0],p[1],p[2]),rv,tv,K,d);
            errors.push_back((pr-Eigen::Vector2d(o.px,o.py)).norm());}
        return errors;
    };

    std::vector<std::array<double,15>> cp;
    std::vector<int> pio; std::map<int,int> pidx; std::vector<std::array<double,3>> pp;
    pack_cameras(cp); pack_points(pio,pidx,pp);
    auto observations = build_observations(pidx);
    if(observations.empty()){if(status)*status="Error: no observations";return false;}

    int total_outliers=0,total_rounds=0;

    // 3-stage hierarchical BA with GNC:
    //   Stage 0: Extrinsics only, GNC CauchyLoss 16→4→1
    //   (Re-triangulate after stage 0)
    //   Stage 1: Extrinsics + points, CauchyLoss 4→1
    //   Stage 2: Full joint (extrinsics + intrinsics + points), CauchyLoss 1.0
    // Outlier rejection: progressive Anipose-style thresholds
    struct BAPass {
        int fix_mode; // 0=fix intrinsics+points, 1=fix intrinsics, 2=free all
        double cauchy_scale;
        double outlier_px; // fixed px threshold (0 = skip outlier rejection)
    };
    std::vector<BAPass> passes = {
        // Stage 0: extrinsics only, GNC
        {0, 16.0, 0},
        {0,  4.0, 0},
        {0,  1.0, 20.0},
        // Re-triangulation happens here (handled below)
        // Stage 1: extrinsics + points
        {1,  4.0, 0},
        {1,  1.0, 10.0},
        // Stage 2: full joint (intrinsics + extrinsics + points)
        // First pass: moderate Cauchy for outlier robustness
        // Second pass: near-linear (scale=50) to give all observations equal weight
        // for best 3D metric accuracy (matches MVC's linear loss strategy).
        {2,  4.0, 15.0},
        {2, 50.0, 10.0},
    };

    bool did_retri = false;
    for(int pi_pass=0;pi_pass<(int)passes.size();pi_pass++){
        auto &bp = passes[pi_pass];

        // Re-triangulate after stage 0 (between fix_mode 0 and 1)
        if(!did_retri && bp.fix_mode >= 1) {
            did_retri = true;
            unpack_cameras(cp);
            unpack_points(pio,pp);
            // Re-triangulate with updated extrinsics
            points_3d.clear();
            int np = triangulate_landmarks_multiview(config,landmarks,poses,points_3d,10.0);
            fprintf(stderr,"[Experimental]   Re-triangulated: %d landmarks\n",np);
            pack_points(pio,pidx,pp);
            observations = build_observations(pidx);
        }

        total_rounds++;
        if(status){char b[128];snprintf(b,sizeof(b),"BA pass %d/%d (mode=%d cauchy=%.0f %d obs)...",
            pi_pass+1,(int)passes.size(),bp.fix_mode,bp.cauchy_scale,(int)observations.size());*status=b;}

        ceres::Problem problem;
        for(const auto&obs:observations)
            problem.AddResidualBlock(ReprojectionCost::Create(obs.px,obs.py),
                new ceres::CauchyLoss(bp.cauchy_scale),cp[obs.ci].data(),pp[obs.pi].data());

        if(bp.fix_mode==0){
            // Fix intrinsics + points
            for(int c=0;c<nc;c++){std::vector<int> fix={6,7,8,9,10,11,12,13,14};
                problem.SetManifold(cp[c].data(),new ceres::SubsetManifold(15,fix));}
            for(int i=0;i<(int)pp.size();i++)if(problem.HasParameterBlock(pp[i].data()))problem.SetParameterBlockConstant(pp[i].data());
        } else if(bp.fix_mode==1){
            // Fix intrinsics only, points free
            for(int c=0;c<nc;c++){std::vector<int> fix={6,7,8,9,10,11,12,13,14};
                problem.SetManifold(cp[c].data(),new ceres::SubsetManifold(15,fix));}
        }
        // fix_mode==2: free extrinsics + fx,fy,cx,cy,k1,k2 + 3D points.
        // Lock tangential distortion (p1,p2) and k3 — these are poorly
        // constrained with typical ChArUco data and cause 3D accuracy loss
        // when freed. Only unlock if user provides explicit ba_config bounds.
        if(bp.fix_mode==2){
            std::vector<int> obs_per_cam(nc, 0);
            for(const auto&obs:observations) obs_per_cam[obs.ci]++;
            for(int c=0;c<nc;c++){
                if(obs_per_cam[c]<30){
                    // Too few observations — fix all intrinsics
                    std::vector<int> fix={6,7,8,9,10,11,12,13,14};
                    problem.SetManifold(cp[c].data(),new ceres::SubsetManifold(15,fix));
                    fprintf(stderr,"[Experimental]   Camera %s: %d obs < 30, fixing intrinsics\n",
                            config.cam_ordered[c].c_str(), obs_per_cam[c]);
                } else {
                    // Lock p1(12), p2(13), k3(14) — prevents distortion overfitting
                    std::vector<int> fix={12,13,14};
                    problem.SetManifold(cp[c].data(),new ceres::SubsetManifold(15,fix));
                }
            }
        }

        // Apply ba_config bounds in full joint mode
        if(bp.fix_mode==2&&!config.ba_config.is_null()){
            bool ub=config.ba_config.value("bounds",false);std::vector<double> bcp;
            if(config.ba_config.contains("bounds_cp")&&config.ba_config["bounds_cp"].is_array())bcp=config.ba_config["bounds_cp"].get<std::vector<double>>();
            if(ub&&bcp.size()>=15)for(int c=0;c<nc;c++){std::vector<int> ci;
                for(int p=0;p<15;p++){double b=bcp[p];if(b>0){problem.SetParameterLowerBound(cp[c].data(),p,cp[c][p]-b);problem.SetParameterUpperBound(cp[c].data(),p,cp[c][p]+b);}else if(b==0)ci.push_back(p);}
                if(!ci.empty())problem.SetManifold(cp[c].data(),new ceres::SubsetManifold(15,ci));}}

        ceres::Solver::Options opt;
        opt.linear_solver_type=ceres::SPARSE_SCHUR;
        opt.max_num_iterations=100;
        opt.function_tolerance=1e-10; opt.parameter_tolerance=1e-10; opt.gradient_tolerance=1e-12;
        opt.use_inner_iterations=true;
        opt.minimizer_progress_to_stdout=false;
        opt.logging_type=ceres::SILENT;
        opt.num_threads=std::max(1,(int)std::thread::hardware_concurrency());

        // Try solving, fall back to DENSE_SCHUR if SPARSE_SCHUR fails
        // (vcpkg Ceres on Windows may lack SuiteSparse)
        ceres::Solver::Summary sum;
        {
            // Suppress CHOLMOD warnings by temporarily redirecting stderr.
            // CHOLMOD writes "not positive definite" via SuiteSparse's printf,
            // which is normal LM behavior (Ceres increases damping and retries).
#ifdef _WIN32
            int saved_stderr = _dup(STDERR_FILENO);
            if (saved_stderr >= 0) {
                int devnull = _open("NUL", O_WRONLY);
                if (devnull >= 0) { _dup2(devnull, STDERR_FILENO); _close(devnull); }
            }
#else
            int saved_stderr = dup(STDERR_FILENO);
            if (saved_stderr >= 0) {
                int devnull = open("/dev/null", O_WRONLY);
                if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
            }
#endif
            ceres::Solve(opt,&problem,&sum);
#ifdef _WIN32
            if (saved_stderr >= 0) { _dup2(saved_stderr, STDERR_FILENO); _close(saved_stderr); }
#else
            if (saved_stderr >= 0) { dup2(saved_stderr, STDERR_FILENO); close(saved_stderr); }
#endif
        }
        // Fallback: if SPARSE_SCHUR failed (no SuiteSparse), try DENSE_SCHUR
        if (!sum.IsSolutionUsable() && opt.linear_solver_type == ceres::SPARSE_SCHUR) {
            fprintf(stderr,"[Experimental]   SPARSE_SCHUR failed (%s), retrying with DENSE_SCHUR...\n",
                    sum.message.c_str());
            opt.linear_solver_type = ceres::DENSE_SCHUR;
            opt.use_inner_iterations = false;  // inner iterations can also fail without sparse backend
            ceres::Solve(opt,&problem,&sum);
            if (!sum.IsSolutionUsable()) {
                fprintf(stderr,"[Experimental]   DENSE_SCHUR also failed (%s), trying ITERATIVE_SCHUR...\n",
                        sum.message.c_str());
                opt.linear_solver_type = ceres::ITERATIVE_SCHUR;
                opt.preconditioner_type = ceres::SCHUR_JACOBI;
                ceres::Solve(opt,&problem,&sum);
            }
        }
        fprintf(stderr,"[Experimental] Pass %d/%d (mode=%d cauchy=%.0f): %s cost %.2f→%.2f iters=%d time=%.2fs\n",
            pi_pass+1,(int)passes.size(),bp.fix_mode,bp.cauchy_scale,
            sum.IsSolutionUsable()?"OK":"FAIL",sum.initial_cost,sum.final_cost,
            (int)sum.iterations.size(),sum.total_time_in_seconds);

        // Progressive outlier rejection (Anipose-style fixed px threshold)
        int pass_outliers = 0;
        if(bp.outlier_px > 0) {
            auto errors = compute_errors(observations,cp,pp);
            std::vector<Obs> inl;
            for(int i=0;i<(int)observations.size();i++)
                if(errors[i]<bp.outlier_px) inl.push_back(observations[i]);
            pass_outliers=(int)observations.size()-(int)inl.size();total_outliers+=pass_outliers;
            fprintf(stderr,"[Experimental]   Outlier rejection: threshold=%.1f px removed=%d\n",bp.outlier_px,pass_outliers);
            if(pass_outliers>0) observations=std::move(inl);
        }

        // Record BA pass info in database
        result.db.ba_passes.push_back({pi_pass+1, bp.fix_mode, bp.cauchy_scale,
            sum.initial_cost, sum.final_cost, (int)sum.iterations.size(),
            sum.total_time_in_seconds, pass_outliers});
    }

    // Unpack final results
    unpack_cameras(cp); unpack_points(pio,pp);

    // Collect per-camera metrics
    result.per_camera_metrics.resize(nc);result.all_reproj_errors.clear();
    for(int c=0;c<nc;c++){auto&m=result.per_camera_metrics[c];m.name=config.cam_ordered[c];
        auto it=landmarks.find(m.name);if(it==landmarks.end())continue;
        auto rv=red_math::rotationMatrixToVector(poses[c].R);std::vector<double>ce;
        for(const auto&[pid,px]:it->second){auto pt=points_3d.find(pid);if(pt==points_3d.end())continue;
            double e=(red_math::projectPoint(pt->second,rv,poses[c].t,poses[c].K,poses[c].dist)-px).norm();
            ce.push_back(e);result.all_reproj_errors.push_back(e);}
        m.observation_count=(int)ce.size();if(!ce.empty()){double s=0;for(double e:ce)s+=e;m.mean_reproj=s/ce.size();
            std::sort(ce.begin(),ce.end());m.median_reproj=ce[ce.size()/2];m.max_reproj=ce.back();
            double v=0;for(double e:ce)v+=(e-m.mean_reproj)*(e-m.mean_reproj);m.std_reproj=std::sqrt(v/ce.size());}}
    result.ba_rounds=total_rounds;result.outliers_removed=total_outliers;
    if(!result.all_reproj_errors.empty()){double s=0;for(double e:result.all_reproj_errors)s+=e;result.mean_reproj_error=s/result.all_reproj_errors.size();}
    if(status)*status="Experimental BA done. Mean: "+std::to_string(result.mean_reproj_error).substr(0,5)+" px";
    return true;
}

inline CalibrationResult run_experimental_pipeline(
    const CalibrationTool::CalibConfig &config_in, const std::string &base_folder,
    std::string *status, const VideoFrameRange *vfr=nullptr,
    aruco_detect::GpuThresholdFunc gpu_thresh=nullptr, void *gpu_ctx=nullptr,
    ArucoProgress *progress=nullptr) {
    CalibrationTool::CalibConfig config = config_in; // mutable copy (cameras may be removed)
    CalibrationResult result; result.cam_names=config.cam_ordered;
    auto now=std::chrono::system_clock::now();auto t=std::chrono::system_clock::to_time_t(now);
    std::tm ts;
#ifdef _WIN32
    localtime_s(&ts,&t);
#else
    localtime_r(&t,&ts);
#endif
    char tb[64];std::strftime(tb,sizeof(tb),"%Y_%m_%d_%H_%M_%S",&ts);
    std::string outf=base_folder+"/"+tb;
    {namespace fs=std::filesystem;std::error_code ec;fs::create_directories(outf,ec);if(ec){result.error="Cannot create: "+ec.message();return result;}}
    fprintf(stderr,"\n[Experimental] === Starting experimental pipeline ===\n");
    auto pipeline_start = std::chrono::steady_clock::now();
    if(progress) progress->current_step.store(1, std::memory_order_relaxed);
    if(status)*status="Step 1: Detecting + calibrating intrinsics...";
    std::map<std::string,CameraIntrinsics> intrinsics;std::vector<int> video_frame_numbers;
    std::vector<std::string> skipped_cams;
    if(vfr){if(!detect_and_calibrate_intrinsics_video(config,*vfr,intrinsics,&video_frame_numbers,status,gpu_thresh,gpu_ctx,&skipped_cams,progress)){result.error=status?*status:"failed";return result;}}
    else{if(!detect_and_calibrate_intrinsics(config,intrinsics,status,gpu_thresh,gpu_ctx,&skipped_cams)){result.error=status?*status:"failed";return result;}}
    // Remove skipped cameras from cam_ordered for subsequent pipeline steps
    if (!skipped_cams.empty()) {
        std::set<std::string> skip_set(skipped_cams.begin(), skipped_cams.end());
        std::vector<std::string> filtered;
        for (const auto &cam : config.cam_ordered)
            if (!skip_set.count(cam)) filtered.push_back(cam);
        config.cam_ordered = filtered;
        // The experimental pipeline uses incremental PnP registration
        // (not spanning tree), so no need to remap first_view/second_view_order.
        std::string skip_msg;
        for (const auto &s : skipped_cams) skip_msg += (skip_msg.empty() ? "" : ", ") + s;
        result.warning = "Skipped " + std::to_string(skipped_cams.size()) +
            " camera(s): " + skip_msg;
        result.cam_names = config.cam_ordered; // Update to filtered list
        fprintf(stderr, "[Experimental] %s\n", result.warning.c_str());
    }
    if (intrinsics.empty()) { result.error = "No cameras passed detection"; return result; }
    result.image_width=intrinsics.begin()->second.image_width;result.image_height=intrinsics.begin()->second.image_height;
    int nc=(int)config.cam_ordered.size();result.per_camera_metrics.resize(nc);
    for(int i=0;i<nc;i++){auto&m=result.per_camera_metrics[i];m.name=config.cam_ordered[i];
        auto it=intrinsics.find(m.name);if(it!=intrinsics.end()){m.detection_count=(int)it->second.corners_per_image.size();m.intrinsic_reproj=it->second.reproj_error;}}

    // Step 2: Intrinsic quality gate — re-calibrate cameras with high reproj error
    if(progress) progress->current_step.store(2, std::memory_order_relaxed);
    if(status)*status="Step 2: Intrinsic quality gate...";
    for(int ci=0;ci<nc;ci++){
        const auto &serial=config.cam_ordered[ci];
        auto &intr=intrinsics[serial];
        if (intr.reproj_error > 1.0 && intr.corners_per_image.size() > 8) {
            fprintf(stderr,"[Experimental]   Quality gate: %s reproj=%.2f px > 1.0, re-calibrating with best 50%% frames\n",
                serial.c_str(), intr.reproj_error);
            // Compute per-frame reproj and keep only the best 50%
            // Re-run intrinsic calibration on remaining corners
            // We need obj_points and img_points from the detection — reconstruct from corners_per_image
            const auto &cs = config.charuco_setup;
            aruco_detect::CharucoBoard board;
            board.squares_x=cs.w; board.squares_y=cs.h;
            board.square_length=cs.square_side_length; board.marker_length=cs.marker_side_length;
            board.dictionary_id=cs.dictionary;
            // Rebuild obj/img points from corners_per_image
            std::vector<std::pair<int,double>> frame_errors; // (frame_idx, per-frame reproj)
            std::vector<std::vector<Eigen::Vector3f>> all_obj;
            std::vector<std::vector<Eigen::Vector2f>> all_img;
            std::vector<int> frame_indices;
            for (const auto &[fi, corners] : intr.corners_per_image) {
                const auto &ids = intr.ids_per_image.at(fi);
                std::vector<Eigen::Vector3f> obj_pts;
                std::vector<Eigen::Vector2f> img_pts;
                aruco_detect::matchImagePoints(board, corners, ids, obj_pts, img_pts);
                if ((int)obj_pts.size() >= 6) {
                    all_obj.push_back(obj_pts);
                    all_img.push_back(img_pts);
                    frame_indices.push_back(fi);
                }
            }
            // Compute per-frame reproj with current K
            for (int f=0;f<(int)all_obj.size();f++) {
                double err=0;
                for (int j=0;j<(int)all_obj[f].size();j++) {
                    Eigen::Vector3d pt3d(all_obj[f][j].x(), all_obj[f][j].y(), 0);
                    // Project using current intrinsics (identity extrinsics per frame)
                }
                // Simpler: use corner count as quality proxy (more corners = more constrained)
                frame_errors.push_back({f, -(double)all_obj[f].size()}); // negative so sort ascending = best first
            }
            std::sort(frame_errors.begin(), frame_errors.end(),
                [](const auto&a, const auto&b){return a.second < b.second;});
            // Keep best 50%
            int keep = std::max(4, (int)frame_errors.size() / 2);
            std::vector<std::vector<Eigen::Vector3f>> best_obj;
            std::vector<std::vector<Eigen::Vector2f>> best_img;
            for (int f=0;f<keep;f++) {
                int idx = frame_errors[f].first;
                best_obj.push_back(all_obj[idx]);
                best_img.push_back(all_img[idx]);
            }
            auto recalib = intrinsic_calib::calibrateCamera(
                best_obj, best_img, intr.image_width, intr.image_height, true);
            fprintf(stderr,"[Experimental]   Re-calibrated %s: %.3f → %.3f px (%d/%d frames)\n",
                serial.c_str(), intr.reproj_error, recalib.reproj_error, keep, (int)all_obj.size());
            if (recalib.reproj_error < intr.reproj_error) {
                intr.K = recalib.K;
                intr.dist = recalib.dist;
                intr.reproj_error = recalib.reproj_error;
                result.per_camera_metrics[ci].intrinsic_reproj = recalib.reproj_error;
            }
        }
    }

    if(progress) progress->current_step.store(3, std::memory_order_relaxed);
    if(status)*status="Step 3: PnP initialization...";
    std::map<std::string,std::map<int,Eigen::Vector2d>> landmarks;
    build_landmarks(config,intrinsics,landmarks);
    result.db.landmarks = landmarks; // Save landmark map for 3D viewer
    auto detection_end = std::chrono::steady_clock::now();
    result.db.detection_time_sec = std::chrono::duration<double>(detection_end - pipeline_start).count();
    std::vector<CameraPose> poses;
    if(!initialize_extrinsics_pnp(config,intrinsics,poses,status,&result.db)){result.error=status?*status:"PnP failed";return result;}
    if(progress) progress->current_step.store(4, std::memory_order_relaxed);
    if(status)*status="Step 4: Triangulation...";
    std::map<int,Eigen::Vector3d> points_3d;
    int np=triangulate_landmarks_multiview(config,landmarks,poses,points_3d,50.0);
    fprintf(stderr,"[Experimental] Triangulated %d landmarks (threshold=50px)\n",np);
    if(np<10){result.error="Too few points ("+std::to_string(np)+")";return result;}
    if(progress) progress->current_step.store(5, std::memory_order_relaxed);
    if(status)*status="Step 5: Bundle adjustment...";
    { auto ba_start = std::chrono::steady_clock::now();
    if(!bundle_adjust_experimental(config,landmarks,poses,points_3d,result,status)){result.error=status?*status:"BA failed";return result;}
    result.db.ba_time_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - ba_start).count(); }
    if(progress) progress->current_step.store(6, std::memory_order_relaxed);
    if(status)*status="Step 6: Global registration...";
    { std::string gr_status;
      if(!global_registration(config,poses,points_3d,&gr_status,vfr?&video_frame_numbers:nullptr,
        config.global_reg_media_folder,config.global_reg_media_type,
        &config.cam_ordered)){result.error=gr_status.empty()?"Registration failed":gr_status;return result;}
      result.global_reg_status = gr_status;
      fprintf(stderr,"[Experimental] Global reg: %s\n", gr_status.c_str());
    }
    if(progress) progress->current_step.store(7, std::memory_order_relaxed);
    if(status)*status="Step 7: Writing files...";
    if(!write_calibration(poses,config.cam_ordered,outf,result.image_width,result.image_height,status)){result.error=status?*status:"Write failed";return result;}
    write_intermediate_output(config,intrinsics,landmarks,poses,points_3d,outf);
    // Collect final residuals and write database
    double te=0;int to=0;result.all_reproj_errors.clear();
    result.db.residuals.clear();
    for(int c=0;c<nc;c++){auto li=landmarks.find(config.cam_ordered[c]);if(li==landmarks.end())continue;
        auto rv=red_math::rotationMatrixToVector(poses[c].R);std::vector<double>ce;
        for(const auto&[pid,px]:li->second){auto pt=points_3d.find(pid);if(pt==points_3d.end())continue;
            auto proj=red_math::projectPoint(pt->second,rv,poses[c].t,poses[c].K,poses[c].dist);
            double e=(proj-px).norm();
            te+=e;to++;ce.push_back(e);result.all_reproj_errors.push_back(e);
            result.db.residuals.push_back({c, pid, (float)px.x(), (float)px.y(),
                (float)proj.x(), (float)proj.y(), (float)e});}
        auto&m=result.per_camera_metrics[c];m.observation_count=(int)ce.size();if(!ce.empty()){
            double s=0;for(double e:ce)s+=e;m.mean_reproj=s/ce.size();std::sort(ce.begin(),ce.end());
            m.median_reproj=ce[ce.size()/2];m.max_reproj=ce.back();double v=0;for(double e:ce)v+=(e-m.mean_reproj)*(e-m.mean_reproj);m.std_reproj=std::sqrt(v/ce.size());}}
    result.mean_reproj_error=(to>0)?(te/to):0;result.output_folder=outf;result.cameras=std::move(poses);result.points_3d=std::move(points_3d);result.success=true;
    // Recompute board poses using post-BA intrinsics (the original PnP board poses
    // were computed with pre-BA K, which can differ significantly after BA refinement)
    {
        int mc = (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);
        std::vector<Eigen::Vector3d> bd3(mc);
        std::vector<Eigen::Vector2d> bd2(mc);
        for (int y = 0; y < config.charuco_setup.h - 1; y++)
            for (int x = 0; x < config.charuco_setup.w - 1; x++) {
                int i = y * (config.charuco_setup.w - 1) + x;
                double bx = x * config.charuco_setup.square_side_length;
                double by = y * config.charuco_setup.square_side_length;
                bd3[i] = Eigen::Vector3d(bx, by, 0);
                bd2[i] = Eigen::Vector2d(bx, by);
            }
        for (int ci = 0; ci < nc; ci++) {
            const auto &serial = config.cam_ordered[ci];
            const auto &cam = result.cameras[ci];
            auto intr_it = intrinsics.find(serial);
            if (intr_it == intrinsics.end()) continue;
            auto bp_it = result.db.board_poses.find(serial);
            if (bp_it == result.db.board_poses.end()) continue;
            for (auto &[fi, bp] : bp_it->second) {
                auto c_it = intr_it->second.corners_per_image.find(fi);
                auto i_it = intr_it->second.ids_per_image.find(fi);
                if (c_it == intr_it->second.corners_per_image.end() ||
                    i_it == intr_it->second.ids_per_image.end()) continue;
                std::vector<Eigen::Vector3d> o3d;
                std::vector<Eigen::Vector2d> o2d, iund, iraw;
                for (int j = 0; j < (int)i_it->second.size(); j++) {
                    int c = i_it->second[j];
                    if (c < 0 || c >= mc) continue;
                    o3d.push_back(bd3[c]);
                    o2d.push_back(bd2[c]);
                    Eigen::Vector2d px(c_it->second[j].x(), c_it->second[j].y());
                    iund.push_back(red_math::undistortPoint(px, cam.K, cam.dist));
                    iraw.push_back(px);
                }
                if ((int)o3d.size() < 4) continue;
                Eigen::Matrix3d Rf; Eigen::Vector3d tf;
                if (!solvePnPHomography(o2d, iund, cam.K, Rf, tf)) continue;
                refinePnPPose(o3d, iraw, cam.K, cam.dist, Rf, tf);
                auto rv = red_math::rotationMatrixToVector(Rf);
                auto pr = red_math::projectPoints(o3d, rv, tf, cam.K, cam.dist);
                double es = 0;
                for (int i = 0; i < (int)pr.size(); i++)
                    es += (pr[i] - iraw[i]).norm();
                bp.R = Rf; bp.t = tf; bp.reproj = es / pr.size();
            }
        }
    }
    // Write run info for reproducibility
    { int tvf=0; if(vfr){auto vids=CalibrationTool::discover_aruco_videos(vfr->video_folder,vfr->cam_ordered);
      if(!vids.empty())tvf=get_video_frame_count(vids.begin()->second);}
      auto pipeline_end = std::chrono::steady_clock::now();
      double total_s = std::chrono::duration<double>(pipeline_end - pipeline_start).count();
      result.db.total_time_sec = total_s;
      write_run_info(config,outf,vfr,tvf,nc,result.mean_reproj_error,
                     result.db.detection_time_sec, result.db.ba_time_sec, total_s); }
    // Write calibration database for 3D viewer inspection
    write_calibration_database(result.db, outf, &result.per_camera_metrics);

    // Compute global multi-view triangulation consistency
    // This is the real quality metric: how well all cameras agree on 3D point locations.
    result.global_consistency = compute_global_consistency(
        result.db.landmarks, result.cameras, result.cam_names);

    fprintf(stderr,"[Experimental] === Done: %.3f px per-board, %.2f px multi-view (%d obs, %d cameras) ===\n\n",
        result.mean_reproj_error, result.global_consistency.mean_reproj,
        to, nc);
    if(status){
        std::string gc_str = result.global_consistency.computed
            ? " | Multi-view: " + std::to_string(result.global_consistency.mean_reproj).substr(0, 5) + " px"
            : "";
        *status="Experimental done! "+std::to_string(result.mean_reproj_error).substr(0,5)+" px" + gc_str +
            " ("+std::to_string(to)+" obs, "+std::to_string(result.ba_rounds)+" BA rounds, "+
            std::to_string(result.outliers_removed)+" outliers)";
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
} // namespace CalibrationPipeline

// Legacy run_full_pipeline removed — use run_experimental_pipeline() instead.
// The experimental pipeline uses incremental PnP registration (not spanning tree)
// and 7-pass GNC Cauchy bundle adjustment. It produces strictly better results.

#if 0 // DEAD CODE — kept as reference only
inline CalibrationResult
run_full_pipeline_REMOVED(const CalibrationTool::CalibConfig &config,
                  const std::string &base_folder,
                  std::string *status,
                  const VideoFrameRange *vfr = nullptr,
                  aruco_detect::GpuThresholdFunc gpu_thresh = nullptr,
                  void *gpu_ctx = nullptr) {
    CalibrationResult result;
    result.cam_names = config.cam_ordered;

    // Create timestamped output folder inside base_folder
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tstruct;
    localtime_r(&t, &tstruct);
    char tbuf[64];
    std::strftime(tbuf, sizeof(tbuf), "%Y_%m_%d_%H_%M_%S", &tstruct);
    std::string output_folder = base_folder + "/" + tbuf;

    {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(output_folder, ec);
        if (ec) {
            result.error = "Cannot create output folder: " + ec.message();
            return result;
        }
    }

    // Step 1: Detect ChArUco corners + calibrate intrinsics
    if (status)
        *status = "Step 1/7: Detecting ChArUco corners...";
    std::map<std::string, CameraIntrinsics> intrinsics;
    std::vector<int> video_frame_numbers;
    if (vfr) {
        if (!detect_and_calibrate_intrinsics_video(
                config, *vfr, intrinsics, &video_frame_numbers, status,
                gpu_thresh, gpu_ctx)) {
            result.error = status ? *status : "Intrinsic calibration failed";
            return result;
        }
    } else {
        if (!detect_and_calibrate_intrinsics(config, intrinsics, status,
                                             gpu_thresh, gpu_ctx)) {
            result.error = status ? *status : "Intrinsic calibration failed";
            return result;
        }
    }

    // Get image dimensions from first camera
    result.image_width = intrinsics.begin()->second.image_width;
    result.image_height = intrinsics.begin()->second.image_height;

    // Step 2: Build landmarks
    if (status)
        *status = "Step 2/7: Building landmarks...";
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
    build_landmarks(config, intrinsics, landmarks);

    // Step 3: Compute relative poses
    if (status)
        *status = "Step 3/7: Computing relative poses...";
    std::map<std::pair<int, int>, RelativePose> relative_poses;
    if (!compute_relative_poses(config, intrinsics, landmarks, relative_poses,
                                status)) {
        result.error = status ? *status : "Relative pose estimation failed";
        return result;
    }

    // Step 4: Chain poses
    if (status)
        *status = "Step 4/7: Chaining poses along spanning tree...";
    std::vector<CameraPose> poses;
    if (!concatenate_poses(config, intrinsics, relative_poses, poses,
                           status)) {
        result.error = status ? *status : "Pose concatenation failed";
        return result;
    }

    // Collect all accumulated 3D points for BA
    // Re-triangulate using the chained poses for a complete point set
    std::map<int, Eigen::Vector3d> points_3d;
    {
        int max_corners =
            (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);

        // For each 3D point, triangulate from all cameras that see it
        std::map<int, std::vector<std::pair<int, Eigen::Vector2d>>>
            point_observations;
        for (int c = 0; c < (int)config.cam_ordered.size(); c++) {
            const auto &serial = config.cam_ordered[c];
            auto it = landmarks.find(serial);
            if (it == landmarks.end())
                continue;
            for (const auto &[pid, pixel] : it->second) {
                point_observations[pid].push_back({c, pixel});
            }
        }

        for (const auto &[pid, obs] : point_observations) {
            if (obs.size() < 2)
                continue;
            std::vector<Eigen::Vector2d> pts2d;
            std::vector<Eigen::Matrix<double, 3, 4>> Ps;
            for (const auto &[cam_idx, pixel] : obs) {
                // Undistort the point
                Eigen::Vector2d und = red_math::undistortPoint(
                    pixel, poses[cam_idx].K, poses[cam_idx].dist);
                pts2d.push_back(und);
                Ps.push_back(red_math::projectionFromKRt(
                    poses[cam_idx].K, poses[cam_idx].R, poses[cam_idx].t));
            }
            Eigen::Vector3d X = red_math::triangulatePoints(pts2d, Ps);
            points_3d[pid] = X;
        }
    }

    // Step 5: Bundle adjustment
    if (status)
        *status = "Step 5/7: Bundle adjustment...";
    if (!bundle_adjust(config, landmarks, poses, points_3d, status)) {
        result.error = status ? *status : "Bundle adjustment failed";
        return result;
    }

    // Step 6: Global registration
    if (status)
        *status = "Step 6/7: Global registration...";
    if (!global_registration(config, poses, points_3d, status,
                             vfr ? &video_frame_numbers : nullptr,
                             config.global_reg_media_folder,
                             config.global_reg_media_type,
                             &config.cam_ordered)) {
        result.error = status ? *status : "Global registration failed";
        return result;
    }

    // Step 7: Write calibration files
    if (status)
        *status = "Step 7/7: Writing calibration files...";
    if (!write_calibration(poses, config.cam_ordered, output_folder,
                           result.image_width, result.image_height, status)) {
        result.error = status ? *status : "Failed to write calibration files";
        return result;
    }

    // Write summary data for validation/comparison
    write_intermediate_output(config, intrinsics, landmarks, poses,
                              points_3d, output_folder);
    // Write run info for reproducibility
    // Compute final mean reprojection error
    double total_err = 0.0;
    int total_obs = 0;
    for (int c = 0; c < (int)config.cam_ordered.size(); c++) {
        const auto &serial = config.cam_ordered[c];
        auto lm_it = landmarks.find(serial);
        if (lm_it == landmarks.end())
            continue;
        Eigen::Vector3d rvec = red_math::rotationMatrixToVector(poses[c].R);
        for (const auto &[pid, pixel] : lm_it->second) {
            auto pt_it = points_3d.find(pid);
            if (pt_it == points_3d.end())
                continue;
            Eigen::Vector2d projected = red_math::projectPoint(
                pt_it->second, rvec, poses[c].t, poses[c].K, poses[c].dist);
            double err = (projected - pixel).norm();
            total_err += err;
            total_obs++;
        }
    }
    result.mean_reproj_error =
        (total_obs > 0) ? (total_err / total_obs) : 0.0;

    result.output_folder = output_folder;
    result.cameras = std::move(poses);
    result.points_3d = std::move(points_3d);
    result.success = true;

    // Write run info for reproducibility
    { int tvf = 0;
      if (vfr) {
          auto vids = CalibrationTool::discover_aruco_videos(
              vfr->video_folder, vfr->cam_ordered);
          if (!vids.empty()) tvf = get_video_frame_count(vids.begin()->second);
      }
      write_run_info(config, output_folder, vfr, tvf,
                     (int)config.cam_ordered.size(), result.mean_reproj_error, 0, 0, 0);
    }

    if (status)
        *status = "Calibration complete! Mean reproj error: " +
                  std::to_string(result.mean_reproj_error).substr(0, 5) +
                  " px (" + std::to_string(total_obs) + " observations, " +
                  std::to_string(result.cameras.size()) + " cameras)";

    return result;
}
#endif // DEAD CODE
