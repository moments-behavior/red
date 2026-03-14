#pragma once
// superpoint_refinement.h — Unified SuperPoint calibration refinement pipeline.
// Header-only, namespace-scoped. Pattern follows laser_calibration.h.
//
// Pipeline:
//   1. Frame selection   (C++ pixel-level diversity analysis via FrameReader)
//   2. Frame extraction  (C++ multi-threaded FrameReader + turbojpeg)
//   3. Feature matching  (Python subprocess → calibration_refinement.py)
//   4. Bundle adjustment (calls FeatureRefinement::run_feature_refinement)

#include "calibration_pipeline.h"  // CalibrationResult, CameraPose, write_calibration
#include "feature_refinement.h"    // FeatureRefinement::run_feature_refinement
#include "ffmpeg_frame_reader.h"   // FrameReader
#include "opencv_yaml_io.h"       // opencv_yaml::read
#include "json.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#ifdef __APPLE__
#include <mach-o/dyld.h>  // _NSGetExecutablePath
#include <turbojpeg.h>
#else
// On Linux, stb_image_write.h is included elsewhere with IMPLEMENTATION.
// We only need the declaration.
extern "C" int stbi_write_jpg(const char *filename, int x, int y, int comp,
                               const void *data, int quality);
#endif

namespace SuperPointRefinement {
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// JPEG writer (copied from jarvis_export.h, in its own namespace to avoid ODR)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef __APPLE__
inline bool sp_write_jpeg(const char *path, int w, int h, int channels,
                          const uint8_t *data, int quality) {
    tjhandle tj = tjInitCompress();
    if (!tj) return false;
    unsigned char *buf = nullptr;
    unsigned long buf_size = 0;
    int pf = (channels == 3) ? TJPF_RGB : TJPF_RGBA;
    int rc = tjCompress2(tj, data, w, 0, h, pf, &buf, &buf_size,
                         TJSAMP_420, quality, TJFLAG_FASTDCT);
    bool ok = (rc == 0);
    if (ok) {
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(buf, 1, buf_size, f); fclose(f); }
        else ok = false;
    }
    tjFree(buf);
    tjDestroy(tj);
    return ok;
}
#else
inline bool sp_write_jpeg(const char *path, int w, int h, int channels,
                          const uint8_t *data, int quality) {
    return stbi_write_jpg(path, w, h, channels, data, quality) != 0;
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Detect negative PTS offset (pre-roll frames in some encoders)
// ─────────────────────────────────────────────────────────────────────────────
inline int sp_detect_negative_pts_offset(const std::string &video_path, double fps) {
    std::string cmd = "ffprobe -v quiet -select_streams v:0 -show_packets "
                      "-show_entries packet=pts_time -of csv=p=0 \"" +
                      video_path + "\" 2>/dev/null | head -1";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) return 0;
    char buf[256];
    int offset = 0;
    if (fgets(buf, sizeof(buf), fp)) {
        try {
            double first_pts = std::stod(buf);
            if (first_pts < -0.001) {
                offset = static_cast<int>(std::round(std::abs(first_pts) * fps));
            }
        } catch (...) {}
    }
    pclose(fp);
    return offset;
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct SPConfig {
    std::string video_folder;        // folder with CamXXXXXXX.mp4 files
    std::string calibration_folder;  // folder with CamXXXXXXX.yaml files
    std::string output_folder;       // where to write results (auto-generated if empty)
    std::vector<std::string> camera_names;

    // Frame selection
    std::string ref_camera;          // camera serial for diversity scoring (e.g., "2002490")
    int num_frame_sets = 50;
    float scan_interval_sec = 2.0f;
    float min_separation_sec = 5.0f;

    // Python feature matching config
    std::string python_path = "python3";
    std::string script_path;         // path to calibration_refinement.py (auto-detected)
    int max_keypoints = 4096;
    int resize = 1600;
    float match_threshold = 0.2f;
    float reproj_thresh = 15.0f;
    int workers = 12;                // parallel Python workers

    // BA config
    double ba_outlier_th1 = 10.0;
    double ba_outlier_th2 = 3.0;
    int ba_max_iter = 100;
    bool lock_intrinsics = true;
    bool lock_distortion = true;
    double prior_rot_weight = 10.0;
    double prior_trans_weight = 100.0;
};

struct SPProgress {
    std::atomic<int> current_step{0};  // 0=idle, 1=selecting, 2=extracting, 3=matching, 4=BA, 5=done

    // Step 1: frame selection
    std::atomic<int> frames_scanned{0};
    std::atomic<int> total_scan_frames{0};

    // Step 2: extraction
    std::atomic<int> frames_extracted{0};
    std::atomic<int> total_extract_frames{0};

    // Step 3: Python matching
    std::atomic<int> sets_matched{0};
    std::atomic<int> total_sets{0};
};

struct SPResult {
    bool success = false;
    std::string error;

    int frames_selected = 0;
    int total_tracks = 0;
    int valid_3d_points = 0;
    int total_observations = 0;
    int ba_outliers_removed = 0;
    double mean_reproj_before = 0.0;
    double mean_reproj_after = 0.0;
    std::string output_folder;

    // For 3D viewer
    CalibrationPipeline::CalibrationResult calib_result;
    // Also store the initial calibration for comparison
    CalibrationPipeline::CalibrationResult init_calib_result;

    std::vector<FeatureRefinement::FeatureResult::CameraChange> camera_changes;
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. check_python_deps
// ─────────────────────────────────────────────────────────────────────────────
inline bool check_python_deps(const std::string &python_path, std::string *error) {
    std::string cmd = python_path + " -c \"from lightglue import LightGlue, SuperPoint; print('OK')\" 2>&1";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) {
        if (error) *error = "Failed to run python: " + python_path;
        return false;
    }
    char buf[1024];
    std::string output;
    while (fgets(buf, sizeof(buf), fp))
        output += buf;
    int status = pclose(fp);

    if (output.find("OK") != std::string::npos && status == 0)
        return true;

    if (error) {
        *error = "Python dependencies not found. Install with:\n"
                 "  pip install torch\n"
                 "  pip install git+https://github.com/cvg/LightGlue.git\n\n"
                 "Python output: " + output;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. find_script_path
// ─────────────────────────────────────────────────────────────────────────────
inline std::string find_script_path() {
    // Try paths relative to the executable
    std::vector<std::string> candidates = {
        "../data_exporter/calibration_refinement.py",
        "../../data_exporter/calibration_refinement.py",
    };

    // Get executable directory
    std::string exe_dir;
#ifdef __APPLE__
    char path_buf[4096];
    uint32_t size = sizeof(path_buf);
    if (_NSGetExecutablePath(path_buf, &size) == 0) {
        exe_dir = fs::path(path_buf).parent_path().string();
    }
#else
    try {
        exe_dir = fs::read_symlink("/proc/self/exe").parent_path().string();
    } catch (...) {}
#endif

    if (!exe_dir.empty()) {
        for (const auto &rel : candidates) {
            fs::path p = fs::path(exe_dir) / rel;
            if (fs::exists(p))
                return fs::canonical(p).string();
        }
    }

    // Also try from cwd
    for (const auto &rel : candidates) {
        if (fs::exists(rel))
            return fs::canonical(rel).string();
    }

    return "";
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. select_diverse_frames
// ─────────────────────────────────────────────────────────────────────────────

struct FrameCandidate {
    int frame_num;
    double content_score;    // stddev of downsampled grayscale
    double diff_from_first;  // mean absolute diff from frame 0
    double combined_score;
};

inline std::vector<int> select_diverse_frames(
    const std::string &video_path,
    int num_frames, float scan_interval_sec, float min_separation_sec,
    SPProgress *progress, std::string *status) {

    if (status) *status = "Opening video for frame selection...";

    ffmpeg_reader::FrameReader reader;
    if (!reader.open(video_path)) {
        if (status) *status = "Failed to open video: " + video_path;
        return {};
    }

    double fps = reader.fps();
    int frame_step = std::max(1, (int)(scan_interval_sec * fps));
    int min_sep_frames = (int)(min_separation_sec * fps);
    int w = reader.width();
    int h = reader.height();

    // Estimate total candidate frames from video duration
    // Use AVFormatContext duration indirectly: just scan until readFrame returns null
    // For now, estimate from a reasonable upper bound
    int max_frame = 0;
    {
        // Try to get duration via a quick probe
        std::string cmd = "ffprobe -v quiet -show_entries format=duration -of csv=p=0 \""
                          + video_path + "\" 2>/dev/null";
        FILE *fp = popen(cmd.c_str(), "r");
        if (fp) {
            char buf[256];
            if (fgets(buf, sizeof(buf), fp)) {
                try { max_frame = (int)(std::stod(buf) * fps); } catch (...) {}
            }
            pclose(fp);
        }
    }
    if (max_frame <= 0) max_frame = (int)(3600 * fps); // fallback: assume 1 hour max

    int total_candidates = max_frame / frame_step;
    if (progress) progress->total_scan_frames.store(total_candidates);

    if (status) *status = "Scanning " + std::to_string(total_candidates) + " candidate frames...";
    fprintf(stderr, "[SuperPoint] Scanning %d candidates (step=%d, fps=%.1f) from %s\n",
            total_candidates, frame_step, fps, video_path.c_str());

    // Downsampled dimensions (4x)
    int dw = w / 4;
    int dh = h / 4;
    std::vector<uint8_t> first_gray;
    std::vector<FrameCandidate> candidates;

    for (int i = 0; i < total_candidates; i++) {
        int frame_num = i * frame_step;
        const uint8_t *rgb = reader.readFrame(frame_num);
        if (!rgb) break;

        // Downsample 4x and convert to grayscale
        std::vector<uint8_t> gray(dw * dh);
        for (int y = 0; y < dh; y++) {
            for (int x = 0; x < dw; x++) {
                int sx = x * 4;
                int sy = y * 4;
                const uint8_t *p = rgb + (sy * w + sx) * 3;
                gray[y * dw + x] = (uint8_t)((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
            }
        }

        // Content score: stddev of grayscale values
        double sum = 0, sum_sq = 0;
        int n = dw * dh;
        for (int j = 0; j < n; j++) {
            double v = gray[j];
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / n;
        double variance = (sum_sq / n) - (mean * mean);
        double content_score = std::sqrt(std::max(0.0, variance));

        // Diff from first frame
        double diff_score = 0.0;
        if (first_gray.empty()) {
            first_gray = gray;
        } else {
            double diff_sum = 0;
            for (int j = 0; j < n; j++)
                diff_sum += std::abs((int)gray[j] - (int)first_gray[j]);
            diff_score = diff_sum / n;
        }

        candidates.push_back({frame_num, content_score, diff_score, 0.0});

        if (progress) progress->frames_scanned.store(i + 1);
    }

    if (candidates.empty()) {
        if (status) *status = "No frames could be read from video";
        return {};
    }

    fprintf(stderr, "[SuperPoint] Scanned %d candidates\n", (int)candidates.size());

    // Normalize scores to [0,1]
    double max_content = 0, min_content = 1e9;
    double max_diff = 0, min_diff = 1e9;
    for (const auto &c : candidates) {
        max_content = std::max(max_content, c.content_score);
        min_content = std::min(min_content, c.content_score);
        max_diff = std::max(max_diff, c.diff_from_first);
        min_diff = std::min(min_diff, c.diff_from_first);
    }
    double content_range = max_content - min_content;
    double diff_range = max_diff - min_diff;

    for (auto &c : candidates) {
        double content_norm = (content_range > 1e-6) ? (c.content_score - min_content) / content_range : 0.5;
        double diff_norm = (diff_range > 1e-6) ? (c.diff_from_first - min_diff) / diff_range : 0.5;
        c.combined_score = 0.5 * content_norm + 0.5 * diff_norm;
    }

    // Greedy selection: sort by combined score descending, pick top that respect min separation
    std::sort(candidates.begin(), candidates.end(),
              [](const FrameCandidate &a, const FrameCandidate &b) {
                  return a.combined_score > b.combined_score;
              });

    std::vector<int> selected;
    for (const auto &c : candidates) {
        if ((int)selected.size() >= num_frames) break;

        bool too_close = false;
        for (int s : selected) {
            if (std::abs(c.frame_num - s) < min_sep_frames) {
                too_close = true;
                break;
            }
        }
        if (!too_close)
            selected.push_back(c.frame_num);
    }

    // Sort by frame number for sequential extraction
    std::sort(selected.begin(), selected.end());

    fprintf(stderr, "[SuperPoint] Selected %d diverse frames\n", (int)selected.size());
    if (status) *status = "Selected " + std::to_string(selected.size()) + " diverse frames";
    return selected;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. extract_frame_sets
// ─────────────────────────────────────────────────────────────────────────────
inline bool extract_frame_sets(
    const SPConfig &config,
    const std::vector<int> &frame_numbers,
    const std::string &output_dir,
    SPProgress *progress, std::string *status) {

    int num_cams = (int)config.camera_names.size();
    int num_sets = (int)frame_numbers.size();

    if (status) *status = "Extracting " + std::to_string(num_sets) + " frame sets from " +
                          std::to_string(num_cams) + " cameras...";

    // Create set directories
    for (int s = 0; s < num_sets; s++) {
        char dirname[64];
        snprintf(dirname, sizeof(dirname), "set_%03d", s + 1);
        fs::create_directories(fs::path(output_dir) / dirname);
    }

    if (progress) progress->total_extract_frames.store(num_sets * num_cams);

    std::mutex err_mutex;
    std::string first_error;
    std::vector<std::thread> threads;

    for (int c = 0; c < num_cams; c++) {
        threads.emplace_back([&, c]() {
            std::string cam_name = config.camera_names[c];
            std::string video_path = config.video_folder + "/Cam" + cam_name + ".mp4";

            if (!fs::exists(video_path)) {
                std::lock_guard<std::mutex> lock(err_mutex);
                if (first_error.empty())
                    first_error = "Video not found: " + video_path;
                return;
            }

            ffmpeg_reader::FrameReader reader;
            if (!reader.open(video_path)) {
                std::lock_guard<std::mutex> lock(err_mutex);
                if (first_error.empty())
                    first_error = "Failed to open video: " + video_path;
                return;
            }

            int pts_offset = sp_detect_negative_pts_offset(video_path, reader.fps());
            int w = reader.width();
            int h = reader.height();

            for (int s = 0; s < num_sets; s++) {
                int seek_frame = frame_numbers[s] - pts_offset;
                if (seek_frame < 0) seek_frame = 0;

                const uint8_t *rgb = reader.readFrame(seek_frame);
                if (!rgb) {
                    fprintf(stderr, "[SuperPoint] WARNING: failed to read frame %d from %s\n",
                            seek_frame, cam_name.c_str());
                    if (progress) progress->frames_extracted.fetch_add(1);
                    continue;
                }

                char dirname[64];
                snprintf(dirname, sizeof(dirname), "set_%03d", s + 1);
                std::string jpg_path = (fs::path(output_dir) / dirname /
                                        ("Cam" + cam_name + ".jpg")).string();

                if (!sp_write_jpeg(jpg_path.c_str(), w, h, 3, rgb, 95)) {
                    fprintf(stderr, "[SuperPoint] WARNING: failed to write %s\n", jpg_path.c_str());
                }

                if (progress) progress->frames_extracted.fetch_add(1);
            }

            fprintf(stderr, "[SuperPoint] Extracted %d frames for Cam%s\n", num_sets, cam_name.c_str());
        });
    }

    for (auto &t : threads) t.join();

    if (!first_error.empty()) {
        if (status) *status = first_error;
        return false;
    }

    if (status) *status = "Extracted " + std::to_string(num_sets * num_cams) + " frames";
    fprintf(stderr, "[SuperPoint] Extraction complete: %d sets x %d cameras\n", num_sets, num_cams);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. run_python_matching
// ─────────────────────────────────────────────────────────────────────────────
inline bool run_python_matching(
    const SPConfig &config,
    const std::string &frames_dir,
    const std::string &match_output_dir,
    SPProgress *progress, std::string *status) {

    int num_sets = config.num_frame_sets;
    if (progress) progress->total_sets.store(num_sets);
    if (status) *status = "Running SuperPoint+LightGlue matching...";

    fs::create_directories(match_output_dir);

    // Build list of set directories
    std::string image_dirs;
    for (int s = 1; s <= num_sets; s++) {
        char dirname[64];
        snprintf(dirname, sizeof(dirname), "set_%03d", s);
        fs::path set_dir = fs::path(frames_dir) / dirname;
        if (fs::exists(set_dir))
            image_dirs += " " + set_dir.string();
    }

    if (image_dirs.empty()) {
        if (status) *status = "No frame set directories found in " + frames_dir;
        return false;
    }

    // Build command
    char cmd_buf[8192];
    snprintf(cmd_buf, sizeof(cmd_buf),
             "%s \"%s\" --image_dir%s --calib_dir \"%s\" --output_dir \"%s\" "
             "--max_keypoints %d --resize %d --match_threshold %.2f "
             "--reproj_thresh %.1f --min_matches 5 --workers %d --device cpu 2>&1",
             config.python_path.c_str(),
             config.script_path.c_str(),
             image_dirs.c_str(),
             config.calibration_folder.c_str(),
             match_output_dir.c_str(),
             config.max_keypoints,
             config.resize,
             config.match_threshold,
             config.reproj_thresh,
             config.workers);

    fprintf(stderr, "[SuperPoint] Running: %s\n", cmd_buf);

    FILE *fp = popen(cmd_buf, "r");
    if (!fp) {
        if (status) *status = "Failed to run Python script";
        return false;
    }

    char line[2048];
    std::string full_output;
    while (fgets(line, sizeof(line), fp)) {
        full_output += line;
        fprintf(stderr, "[Python] %s", line);

        // Parse progress: look for "[N]" pattern indicating set completion
        // e.g., "  [3] set_003: 142 tracks"
        std::string sline(line);
        auto bracket_pos = sline.find('[');
        auto bracket_end = sline.find(']');
        if (bracket_pos != std::string::npos && bracket_end != std::string::npos &&
            bracket_end > bracket_pos) {
            try {
                int set_idx = std::stoi(sline.substr(bracket_pos + 1, bracket_end - bracket_pos - 1));
                if (progress) progress->sets_matched.store(set_idx);
            } catch (...) {}
        }

        // Update status with last meaningful line
        if (sline.find("tracks") != std::string::npos || sline.find("Merging") != std::string::npos) {
            // Trim trailing newline
            while (!sline.empty() && (sline.back() == '\n' || sline.back() == '\r'))
                sline.pop_back();
            if (status) *status = sline;
        }
    }

    int exit_status = pclose(fp);
    if (exit_status != 0) {
        if (status) *status = "Python matching failed (exit code " + std::to_string(exit_status) + ")";
        fprintf(stderr, "[SuperPoint] Python output:\n%s\n", full_output.c_str());
        return false;
    }

    // Verify landmarks.json was produced
    std::string landmarks_path = match_output_dir + "/landmarks.json";
    if (!fs::exists(landmarks_path)) {
        if (status) *status = "Python matching completed but landmarks.json not found in " + match_output_dir;
        return false;
    }

    if (progress) progress->sets_matched.store(num_sets);
    if (status) *status = "Feature matching complete";
    fprintf(stderr, "[SuperPoint] Feature matching complete, landmarks at %s\n", landmarks_path.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Load calibration into CalibrationResult (for 3D viewer)
// ─────────────────────────────────────────────────────────────────────────────
inline CalibrationPipeline::CalibrationResult load_calib_result_from_folder(
    const std::string &calib_folder,
    const std::vector<std::string> &camera_names) {

    CalibrationPipeline::CalibrationResult result;
    result.cam_names = camera_names;
    int nc = (int)camera_names.size();
    result.cameras.resize(nc);

    for (int c = 0; c < nc; c++) {
        std::string yaml_path = calib_folder + "/Cam" + camera_names[c] + ".yaml";
        if (!fs::exists(yaml_path)) continue;
        try {
            auto yaml = opencv_yaml::read(yaml_path);
            result.cameras[c].K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dist_mat = yaml.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) result.cameras[c].dist(j) = dist_mat(j, 0);
            result.cameras[c].R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
            result.cameras[c].t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
            if (c == 0) {
                result.image_width = yaml.getInt("image_width");
                result.image_height = yaml.getInt("image_height");
            }
        } catch (...) {}
    }
    result.success = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. run_superpoint_refinement — main entry point
// ─────────────────────────────────────────────────────────────────────────────
inline SPResult run_superpoint_refinement(
    const SPConfig &config,
    std::string *status,
    std::shared_ptr<SPProgress> progress) {

    SPResult result;
    auto t0 = std::chrono::steady_clock::now();

    // ── Pre-flight checks ──────────────────────────────────────────────────

    if (status) *status = "Checking Python dependencies...";
    if (progress) progress->current_step.store(0);

    std::string dep_error;
    if (!check_python_deps(config.python_path, &dep_error)) {
        result.error = dep_error;
        if (status) *status = "Missing Python dependencies";
        return result;
    }

    // Find Python script
    std::string script = config.script_path;
    if (script.empty()) script = find_script_path();
    if (script.empty() || !fs::exists(script)) {
        result.error = "Cannot find calibration_refinement.py. Set script_path in config.";
        if (status) *status = result.error;
        return result;
    }
    fprintf(stderr, "[SuperPoint] Using script: %s\n", script.c_str());

    // Validate paths
    if (!fs::exists(config.video_folder)) {
        result.error = "Video folder not found: " + config.video_folder;
        if (status) *status = result.error;
        return result;
    }
    if (!fs::exists(config.calibration_folder)) {
        result.error = "Calibration folder not found: " + config.calibration_folder;
        if (status) *status = result.error;
        return result;
    }

    // Create timestamped output folder
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y_%m_%d_%H_%M_%S", &tm_buf);

    std::string base_output = config.output_folder;
    if (base_output.empty())
        base_output = config.calibration_folder + "/superpoint_refined";
    std::string output_dir = base_output + "/" + ts;
    fs::create_directories(output_dir);
    result.output_folder = output_dir;

    fprintf(stderr, "[SuperPoint] Output directory: %s\n", output_dir.c_str());

    // ── Step 1: Frame selection ────────────────────────────────────────────

    if (progress) progress->current_step.store(1);
    if (status) *status = "Step 1/4: Selecting diverse frames...";

    // Determine reference camera (use config or first camera)
    std::string ref_cam = config.ref_camera;
    if (ref_cam.empty() && !config.camera_names.empty())
        ref_cam = config.camera_names[0];

    std::string ref_video = config.video_folder + "/Cam" + ref_cam + ".mp4";
    if (!fs::exists(ref_video)) {
        result.error = "Reference video not found: " + ref_video;
        if (status) *status = result.error;
        return result;
    }

    std::vector<int> frame_numbers = select_diverse_frames(
        ref_video, config.num_frame_sets,
        config.scan_interval_sec, config.min_separation_sec,
        progress.get(), status);

    if (frame_numbers.empty()) {
        result.error = "Frame selection failed — no frames selected";
        if (status) *status = result.error;
        return result;
    }
    result.frames_selected = (int)frame_numbers.size();

    // Save frame_selection.json
    {
        nlohmann::json j;
        j["ref_camera"] = ref_cam;
        j["num_frames"] = (int)frame_numbers.size();
        j["scan_interval_sec"] = config.scan_interval_sec;
        j["min_separation_sec"] = config.min_separation_sec;
        j["frame_numbers"] = frame_numbers;
        std::ofstream f(output_dir + "/frame_selection.json");
        if (f.is_open()) f << j.dump(2);
    }

    // ── Step 2: Frame extraction ───────────────────────────────────────────

    if (progress) progress->current_step.store(2);
    if (status) *status = "Step 2/4: Extracting frames...";

    std::string frames_dir = output_dir + "/frames";
    fs::create_directories(frames_dir);

    if (!extract_frame_sets(config, frame_numbers, frames_dir, progress.get(), status)) {
        result.error = "Frame extraction failed: " + (status ? *status : "unknown error");
        return result;
    }

    // ── Step 3: Python feature matching ────────────────────────────────────

    if (progress) progress->current_step.store(3);
    if (status) *status = "Step 3/4: Running SuperPoint+LightGlue matching...";

    std::string match_dir = output_dir + "/matches";

    // Build a mutable config with the resolved script path
    SPConfig run_config = config;
    run_config.script_path = script;
    // Update num_frame_sets to actual count
    run_config.num_frame_sets = (int)frame_numbers.size();

    if (!run_python_matching(run_config, frames_dir, match_dir, progress.get(), status)) {
        result.error = "Feature matching failed: " + (status ? *status : "unknown error");
        return result;
    }

    // ── Step 4: Bundle adjustment ──────────────────────────────────────────

    if (progress) progress->current_step.store(4);
    if (status) *status = "Step 4/4: Running bundle adjustment...";

    std::string landmarks_file = match_dir + "/landmarks.json";
    std::string points_3d_file = match_dir + "/points_3d.json";
    std::string refined_dir = output_dir + "/calibration";

    FeatureRefinement::FeatureConfig feat_config;
    feat_config.landmarks_file = landmarks_file;
    feat_config.points_3d_file = fs::exists(points_3d_file) ? points_3d_file : "";
    feat_config.calibration_folder = config.calibration_folder;
    feat_config.output_folder = refined_dir;
    feat_config.camera_names = config.camera_names;
    feat_config.ba_outlier_th1 = config.ba_outlier_th1;
    feat_config.ba_outlier_th2 = config.ba_outlier_th2;
    feat_config.ba_max_iter = config.ba_max_iter;
    feat_config.lock_intrinsics = config.lock_intrinsics;
    feat_config.lock_distortion = config.lock_distortion;
    feat_config.prior_rot_weight = config.prior_rot_weight;
    feat_config.prior_trans_weight = config.prior_trans_weight;

    FeatureRefinement::FeatureResult feat_result =
        FeatureRefinement::run_feature_refinement(feat_config, status);

    if (!feat_result.success) {
        result.error = "Bundle adjustment failed: " + feat_result.error;
        if (status) *status = result.error;
        return result;
    }

    // ── Populate result ────────────────────────────────────────────────────

    if (progress) progress->current_step.store(5);

    result.total_tracks = feat_result.total_tracks;
    result.valid_3d_points = feat_result.valid_3d_points;
    result.total_observations = feat_result.total_observations;
    result.ba_outliers_removed = feat_result.ba_outliers_removed;
    result.mean_reproj_before = feat_result.mean_reproj_before;
    result.mean_reproj_after = feat_result.mean_reproj_after;
    result.camera_changes = feat_result.camera_changes;

    // Load refined calibration for 3D viewer
    result.calib_result = load_calib_result_from_folder(refined_dir, config.camera_names);
    result.calib_result.mean_reproj_error = feat_result.mean_reproj_after;
    result.calib_result.output_folder = refined_dir;

    // Load initial calibration for comparison
    result.init_calib_result = load_calib_result_from_folder(config.calibration_folder, config.camera_names);
    result.init_calib_result.mean_reproj_error = feat_result.mean_reproj_before;

    // Load 3D points into calib_result
    if (fs::exists(points_3d_file)) {
        try {
            std::ifstream pf(points_3d_file);
            nlohmann::json pj;
            pf >> pj;
            for (auto &[id_str, pt] : pj.items()) {
                result.calib_result.points_3d[std::stoi(id_str)] = Eigen::Vector3d(
                    pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>());
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperPoint] WARNING: failed to load points_3d.json: %s\n", e.what());
        }
    }

    // Load landmarks into calib_result.db
    if (fs::exists(landmarks_file)) {
        try {
            std::ifstream lf(landmarks_file);
            nlohmann::json lj;
            lf >> lj;
            for (auto &[cam, cam_j] : lj.items()) {
                auto &ids = cam_j["ids"];
                auto &pts = cam_j["landmarks"];
                for (int i = 0; i < (int)ids.size(); i++) {
                    result.calib_result.db.landmarks[cam][ids[i].get<int>()] =
                        Eigen::Vector2d(pts[i][0].get<double>(), pts[i][1].get<double>());
                }
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperPoint] WARNING: failed to load landmarks.json: %s\n", e.what());
        }
    }

    // Write summary calibration_data.json
    {
        nlohmann::json j;
        j["pipeline"] = "superpoint_refinement";
        j["timestamp"] = ts;
        j["frames_selected"] = result.frames_selected;
        j["total_tracks"] = result.total_tracks;
        j["valid_3d_points"] = result.valid_3d_points;
        j["total_observations"] = result.total_observations;
        j["ba_outliers_removed"] = result.ba_outliers_removed;
        j["mean_reproj_before"] = result.mean_reproj_before;
        j["mean_reproj_after"] = result.mean_reproj_after;
        j["calibration_folder"] = config.calibration_folder;
        j["video_folder"] = config.video_folder;

        nlohmann::json changes = nlohmann::json::array();
        for (const auto &ch : result.camera_changes) {
            changes.push_back({
                {"camera", ch.name},
                {"rotation_deg", ch.d_rot_deg},
                {"translation_mm", ch.d_trans_mm},
                {"d_fx", ch.d_fx}, {"d_fy", ch.d_fy},
                {"d_cx", ch.d_cx}, {"d_cy", ch.d_cy}
            });
        }
        j["camera_changes"] = changes;

        std::ofstream f(output_dir + "/calibration_data.json");
        if (f.is_open()) f << j.dump(2);
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    result.success = true;

    fprintf(stderr, "\n[SuperPoint] === Pipeline complete in %.1f s ===\n", elapsed);
    fprintf(stderr, "[SuperPoint]   Frames: %d, Tracks: %d, 3D points: %d\n",
            result.frames_selected, result.total_tracks, result.valid_3d_points);
    fprintf(stderr, "[SuperPoint]   Reproj: %.3f px -> %.3f px\n",
            result.mean_reproj_before, result.mean_reproj_after);
    fprintf(stderr, "[SuperPoint]   Output: %s\n\n", output_dir.c_str());

    if (status) *status = "SuperPoint refinement done: " +
        std::to_string(result.mean_reproj_after).substr(0, 5) + " px (" +
        std::to_string(result.valid_3d_points) + " points, " +
        std::to_string(result.frames_selected) + " frames)";

    return result;
}

} // namespace SuperPointRefinement
