#pragma once
// superpoint_refinement.h — Native SuperPoint calibration refinement pipeline.
// Header-only, namespace-scoped. Pattern follows pointsource_calibration.h.
//
// Pipeline (macOS, native — no Python):
//   1. Feature extraction  (VT decode → CoreML SuperPoint on ANE)
//   2. Descriptor matching (BLAS cblas_sgemm, mutual NN + ratio test)
//   3. Track building      (union-find, instant)
//   4. Bundle adjustment   (Ceres multi-round BA)

#include "calibration_pipeline.h"  // CalibrationResult, CameraPose, write_calibration
#include "feature_refinement.h"    // FeatureRefinement::run_feature_refinement
#include "track_builder.h"         // TrackBuilder::build_tracks, PairwiseMatch
#include "descriptor_matcher.h"    // DescriptorMatcher::match_descriptors, etc.
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
#include <map>

#ifdef __APPLE__
#include "superpoint_coreml.h"     // CoreML SuperPoint inference
#include "FFmpegDemuxer.h"         // Annex-B demuxing for VT decode
#include "vt_async_decoder.h"      // HW decode → BGRA CVPixelBuffer
#include <CoreVideo/CoreVideo.h>
#endif

namespace SuperPointRefinement {
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct SPConfig {
    std::string video_folder;        // folder with CamXXXXXXX.mp4 files
    std::string calibration_folder;  // folder with CamXXXXXXX.yaml files
    std::string output_folder;       // where to write results (auto-generated if empty)
    std::vector<std::string> camera_names;

    // Frame selection
    std::string ref_camera;          // camera serial for diversity scoring
    int num_frame_sets = 50;
    float scan_interval_sec = 2.0f;
    float min_separation_sec = 5.0f;

    // Native feature matching config
    std::string model_path;          // path to superpoint.mlpackage
    int max_keypoints = 4096;
    float ratio_threshold = 0.8f;    // Lowe's ratio test threshold
    float reproj_thresh = 15.0f;

    // BA config
    double ba_outlier_th1 = 10.0;
    double ba_outlier_th2 = 3.0;
    int ba_max_iter = 100;
    bool lock_intrinsics = true;
    bool lock_distortion = true;
    double prior_rot_weight = 10.0;
    double prior_trans_weight = 100.0;

    // Multi-round BA convergence
    int ba_max_rounds = 5;
    double ba_convergence_eps = 0.001;
};

struct SPProgress {
    std::atomic<int> current_step{0};  // 0=idle, 1=extracting, 2=matching, 3=BA, 4=done

    // Step 1: feature extraction
    std::atomic<int> frames_extracted{0};
    std::atomic<int> total_extract_frames{0};

    // Step 2: matching
    std::atomic<int> pairs_matched{0};
    std::atomic<int> total_pairs{0};

    // Step 3: BA rounds
    std::atomic<int> ba_round{0};
    std::atomic<int> ba_total_rounds{0};
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

    // Multi-round BA diagnostics
    int ba_rounds_completed = 0;
    std::vector<double> per_round_reproj;

    // For 3D viewer
    CalibrationPipeline::CalibrationResult calib_result;
    CalibrationPipeline::CalibrationResult init_calib_result;

    std::vector<FeatureRefinement::FeatureResult::CameraChange> camera_changes;

    // Cross-validation holdout diagnostics
    double train_reproj = 0.0;
    double holdout_reproj = 0.0;
    double holdout_ratio = 0.0;
    int holdout_observations = 0;
    int train_observations = 0;
};

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

#ifdef __APPLE__

// ─────────────────────────────────────────────────────────────────────────────
// Select keyframes uniformly distributed across the video.
// Only keyframes are selected — this makes VT decode instant (no forward-
// decode from the previous keyframe needed). For 180fps H.264 with keyframes
// every 1s, a 25-minute video has ~1500 keyframes.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<int> select_keyframes(
    const std::string &video_path,
    int num_frames,
    float min_separation_sec) {

    // Use ffprobe to enumerate keyframe timestamps (fast, ~1s for any video).
    std::string cmd = "ffprobe -v quiet -select_streams v:0 -show_packets "
        "-show_entries packet=pts_time,flags -of csv=p=0 \"" + video_path + "\" 2>/dev/null";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "[SuperPoint] Failed to run ffprobe for keyframe scan\n");
        return {};
    }

    // Get FPS for frame number conversion
    double fps = 30.0;
    {
        std::string fps_cmd = "ffprobe -v quiet -show_entries stream=r_frame_rate "
            "-of csv=p=0 -select_streams v:0 \"" + video_path + "\" 2>/dev/null";
        FILE *fps_fp = popen(fps_cmd.c_str(), "r");
        if (fps_fp) {
            char buf[256];
            if (fgets(buf, sizeof(buf), fps_fp)) {
                std::string s(buf);
                auto slash = s.find('/');
                if (slash != std::string::npos) {
                    try {
                        double num = std::stod(s.substr(0, slash));
                        double den = std::stod(s.substr(slash + 1));
                        if (den > 0) fps = num / den;
                    } catch (...) {}
                }
            }
            pclose(fps_fp);
        }
    }

    int min_sep_frames = std::max(1, (int)(min_separation_sec * fps));

    // Parse ffprobe output: "pts_time,flags" lines, keyframes have 'K' in flags
    std::vector<int> keyframes;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        std::string sline(line);
        // Format: "1.000000,K__" or "0.005556,___"
        if (sline.find('K') != std::string::npos) {
            auto comma = sline.find(',');
            if (comma != std::string::npos) {
                try {
                    double pts = std::stod(sline.substr(0, comma));
                    int frame_num = (int)(pts * fps + 0.5);
                    keyframes.push_back(frame_num);
                } catch (...) {}
            }
        }
    }
    pclose(fp);

    if (keyframes.empty()) {
        fprintf(stderr, "[SuperPoint] No keyframes found in %s\n", video_path.c_str());
        return {};
    }

    fprintf(stderr, "[SuperPoint] Found %d keyframes in %s (fps=%.1f)\n",
            (int)keyframes.size(), video_path.c_str(), fps);

    // Select uniformly distributed subset respecting min separation
    int step = std::max(1, (int)keyframes.size() / std::max(1, num_frames));
    std::vector<int> selected;
    for (int i = 0; i < (int)keyframes.size() && (int)selected.size() < num_frames; i += step) {
        if (!selected.empty() && std::abs(keyframes[i] - selected.back()) < min_sep_frames)
            continue;
        selected.push_back(keyframes[i]);
    }

    fprintf(stderr, "[SuperPoint] Selected %d keyframes (step=%d)\n",
            (int)selected.size(), step);
    return selected;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-camera feature cache: features extracted from each frame
// ─────────────────────────────────────────────────────────────────────────────
struct PerCameraFeatures {
    std::string camera_name;
    // features[i] = features from frame_numbers[i]
    std::vector<SuperPointFeatures> features;
};

// ─────────────────────────────────────────────────────────────────────────────
// Native extract + match pipeline (replaces Python)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<TrackBuilder::PairwiseMatch> extract_and_match_native(
    const SPConfig &config,
    const std::vector<int> &frame_numbers,
    const std::vector<CalibrationPipeline::CameraPose> &poses,
    SPProgress *progress,
    std::string *status) {

    int nc = (int)config.camera_names.size();
    int nf = (int)frame_numbers.size();

    if (progress) progress->total_extract_frames.store(nf * nc);

    // ── Step 1: Feature extraction (per-camera parallel VT decode, serial CoreML) ──

    if (status) *status = "Step 1/4: Extracting features (CoreML SuperPoint)...";
    if (progress) progress->current_step.store(1);

    // Load CoreML model (single instance, ANE is single-pipeline)
    SuperPointCoreMLHandle sp_model = superpoint_coreml_create(
        config.model_path.c_str());
    if (!superpoint_coreml_available(sp_model)) {
        if (status) *status = "Failed to load SuperPoint CoreML model: " + config.model_path;
        superpoint_coreml_destroy(sp_model);
        return {};
    }

    // Per-camera features: VT decode runs in parallel across cameras (I/O bound),
    // CoreML inference serialized via mutex (ANE is single-pipeline).
    std::vector<PerCameraFeatures> all_features(nc);
    std::mutex model_mutex; // serialize CoreML inference
    std::vector<std::thread> extract_threads;

    for (int c = 0; c < nc; c++) {
        extract_threads.emplace_back([&, c]() {
            std::string cam_name = config.camera_names[c];
            std::string video_path = config.video_folder + "/Cam" + cam_name + ".mp4";

            if (!fs::exists(video_path)) {
                fprintf(stderr, "[SuperPoint] WARNING: video not found: %s\n", video_path.c_str());
                return;
            }

            all_features[c].camera_name = cam_name;
            all_features[c].features.resize(nf);

            // Open demuxer
            std::unique_ptr<FFmpegDemuxer> demuxer;
            try {
                demuxer = std::make_unique<FFmpegDemuxer>(
                    video_path.c_str(), std::map<std::string, std::string>{});
            } catch (...) {
                fprintf(stderr, "[SuperPoint] Failed to open demuxer: %s\n", video_path.c_str());
                return;
            }

            // Init VT decoder
            VTAsyncDecoder vt;
            if (!vt.init(demuxer->GetExtradata(), demuxer->GetExtradataSize(),
                         demuxer->GetVideoCodec())) {
                fprintf(stderr, "[SuperPoint] VT init failed for %s\n", video_path.c_str());
                return;
            }

            int decoded_frame = 0;

            for (int fi = 0; fi < nf; fi++) {
                int target_frame = frame_numbers[fi];

                // Seek to nearest keyframe before target
                {
                    uint8_t *seek_pkt = nullptr;
                    size_t seek_pkt_size = 0;
                    PacketData seek_pkt_info;

                    SeekContext seek_ctx((uint64_t)target_frame, PREV_KEY_FRAME, BY_NUMBER);
                    if (demuxer->Seek(seek_ctx, seek_pkt, seek_pkt_size, seek_pkt_info)) {
                        decoded_frame = (int)demuxer->FrameNumberFromTs(seek_ctx.out_frame_pts);
                        if (decoded_frame < 0) decoded_frame = 0;

                        bool is_key = (seek_pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                        vt.submit_blocking(seek_pkt, seek_pkt_size, seek_pkt_info.pts,
                                           seek_pkt_info.dts, demuxer->GetTimebase(), is_key);
                        while (CVPixelBufferRef pb = vt.drain_one()) {
                            if (decoded_frame >= target_frame) {
                                // Got our target frame — extract features (mutex for ANE)
                                std::lock_guard<std::mutex> lock(model_mutex);
                                all_features[c].features[fi] = superpoint_coreml_extract(
                                    sp_model, pb, config.max_keypoints);
                                CFRelease(pb);
                                decoded_frame++;
                                if (progress) progress->frames_extracted.fetch_add(1);
                                goto next_frame;
                            }
                            CFRelease(pb);
                            decoded_frame++;
                        }
                    }
                }

                // Forward decode to target frame
                while (decoded_frame <= target_frame) {
                    uint8_t *pkt_data = nullptr;
                    size_t pkt_size = 0;
                    PacketData pkt_info;

                    if (!demuxer->Demux(pkt_data, pkt_size, pkt_info))
                        break;

                    bool is_key = (pkt_info.flags & AV_PKT_FLAG_KEY) != 0;
                    vt.submit_blocking(pkt_data, pkt_size, pkt_info.pts,
                                       pkt_info.dts, demuxer->GetTimebase(), is_key);

                    while (CVPixelBufferRef pb = vt.drain_one()) {
                        if (decoded_frame >= target_frame) {
                            std::lock_guard<std::mutex> lock(model_mutex);
                            all_features[c].features[fi] = superpoint_coreml_extract(
                                sp_model, pb, config.max_keypoints);
                            CFRelease(pb);
                            decoded_frame++;
                            if (progress) progress->frames_extracted.fetch_add(1);
                            goto next_frame;
                        }
                        CFRelease(pb);
                        decoded_frame++;
                    }
                }

                if (progress) progress->frames_extracted.fetch_add(1);
                next_frame:;
            }

            fprintf(stderr, "[SuperPoint] Cam%s: extracted features for %d frames\n",
                    cam_name.c_str(), nf);
        });
    }

    for (auto &t : extract_threads) t.join();
    superpoint_coreml_destroy(sp_model);

    // ── Step 2: Descriptor matching across viable camera pairs ──

    if (status) *status = "Step 2/4: Matching descriptors...";
    if (progress) progress->current_step.store(2);

    auto viable_pairs = DescriptorMatcher::select_viable_pairs(poses);
    int total_match_tasks = (int)viable_pairs.size() * nf;
    if (progress) progress->total_pairs.store(total_match_tasks);

    fprintf(stderr, "[SuperPoint] Matching %d camera pairs x %d frames = %d tasks\n",
            (int)viable_pairs.size(), nf, total_match_tasks);

    std::vector<TrackBuilder::PairwiseMatch> all_pairwise;
    std::mutex match_mutex;

    // Parallel matching across pairs (descriptor matching is CPU-bound, benefits from threads)
    std::vector<std::thread> match_threads;
    int pairs_per_thread = std::max(1, (int)viable_pairs.size() / (int)std::thread::hardware_concurrency());

    for (int pi = 0; pi < (int)viable_pairs.size(); pi++) {
        match_threads.emplace_back([&, pi]() {
            auto [ci, cj] = viable_pairs[pi];
            const auto &feats_i = all_features[ci];
            const auto &feats_j = all_features[cj];

            std::vector<TrackBuilder::PairwiseMatch> local_matches;

            for (int fi = 0; fi < nf; fi++) {
                const auto &fa = feats_i.features[fi];
                const auto &fb = feats_j.features[fi];

                if (fa.num_keypoints < 10 || fb.num_keypoints < 10) {
                    if (progress) progress->pairs_matched.fetch_add(1);
                    continue;
                }

                // BLAS descriptor matching
                auto matches = DescriptorMatcher::match_descriptors(
                    fa.descriptors_flat.data(), fa.num_keypoints,
                    fb.descriptors_flat.data(), fb.num_keypoints,
                    256, config.ratio_threshold);

                if (matches.empty()) {
                    if (progress) progress->pairs_matched.fetch_add(1);
                    continue;
                }

                // Reprojection filter
                matches = DescriptorMatcher::filter_by_reprojection(
                    matches, fa.keypoints, fb.keypoints,
                    poses[ci], poses[cj], config.reproj_thresh);

                if ((int)matches.size() >= 5) {
                    TrackBuilder::PairwiseMatch pm;
                    pm.cam_a = feats_i.camera_name;
                    pm.cam_b = feats_j.camera_name;
                    pm.pts_a.reserve(matches.size());
                    pm.pts_b.reserve(matches.size());
                    pm.scores.reserve(matches.size());
                    for (const auto &m : matches) {
                        pm.pts_a.push_back(fa.keypoints[m.idx_a]);
                        pm.pts_b.push_back(fb.keypoints[m.idx_b]);
                        pm.scores.push_back(m.score);
                    }
                    local_matches.push_back(std::move(pm));
                }

                if (progress) progress->pairs_matched.fetch_add(1);
            }

            if (!local_matches.empty()) {
                std::lock_guard<std::mutex> lock(match_mutex);
                all_pairwise.insert(all_pairwise.end(),
                    std::make_move_iterator(local_matches.begin()),
                    std::make_move_iterator(local_matches.end()));
            }
        });
    }

    for (auto &t : match_threads) t.join();

    int total_match_count = 0;
    for (const auto &pm : all_pairwise)
        total_match_count += (int)pm.pts_a.size();

    fprintf(stderr, "[SuperPoint] Matching complete: %d pairwise sets, %d total matches\n",
            (int)all_pairwise.size(), total_match_count);

    return all_pairwise;
}

#endif // __APPLE__

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────
inline SPResult run_superpoint_refinement(
    const SPConfig &config,
    std::string *status,
    std::shared_ptr<SPProgress> progress) {

    SPResult result;
    auto t0 = std::chrono::steady_clock::now();

    // ── Pre-flight checks ──────────────────────────────────────────────────

    if (progress) progress->current_step.store(0);

#ifdef __APPLE__
    // Check CoreML model
    if (config.model_path.empty() || !fs::exists(config.model_path)) {
        result.error = "SuperPoint CoreML model not found: " + config.model_path +
            "\nRun: python scripts/convert_superpoint_coreml.py --output_dir models/superpoint";
        if (status) *status = result.error;
        return result;
    }
#else
    result.error = "Native SuperPoint pipeline requires macOS with CoreML. "
                   "Use the Python pipeline on Linux.";
    if (status) *status = result.error;
    return result;
#endif

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
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[64];
    strftime(ts, sizeof(ts), "%Y_%m_%d_%H_%M_%S", &tm_buf);

    std::string base_output = config.output_folder;
    if (base_output.empty())
        base_output = config.calibration_folder + "/superpoint_refined";
    std::string output_dir = base_output + "/" + ts;
    fs::create_directories(output_dir);
    result.output_folder = output_dir;

    fprintf(stderr, "[SuperPoint] Output directory: %s\n", output_dir.c_str());

#ifdef __APPLE__

    // ── Load calibration for viable pair selection + reprojection filter ──

    int nc = (int)config.camera_names.size();
    std::vector<CalibrationPipeline::CameraPose> poses(nc);
    for (int c = 0; c < nc; c++) {
        std::string yaml_path = config.calibration_folder + "/Cam" + config.camera_names[c] + ".yaml";
        if (!fs::exists(yaml_path)) {
            result.error = "Missing calibration: " + yaml_path;
            if (status) *status = result.error;
            return result;
        }
        try {
            auto yaml = opencv_yaml::read(yaml_path);
            poses[c].K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dist_mat = yaml.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) poses[c].dist(j) = dist_mat(j, 0);
            poses[c].R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
            poses[c].t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
        } catch (const std::exception &e) {
            result.error = "Error reading " + yaml_path + ": " + e.what();
            if (status) *status = result.error;
            return result;
        }
    }

    // ── Compute uniform frame grid ──

    std::string ref_cam = config.ref_camera;
    if (ref_cam.empty() && !config.camera_names.empty())
        ref_cam = config.camera_names[0];

    std::string ref_video = config.video_folder + "/Cam" + ref_cam + ".mp4";
    if (!fs::exists(ref_video)) {
        result.error = "Reference video not found: " + ref_video;
        if (status) *status = result.error;
        return result;
    }

    std::vector<int> frame_numbers = select_keyframes(
        ref_video, config.num_frame_sets, config.min_separation_sec);

    if (frame_numbers.empty()) {
        result.error = "Frame grid computation failed";
        if (status) *status = result.error;
        return result;
    }
    result.frames_selected = (int)frame_numbers.size();

    // Save frame_selection.json
    {
        nlohmann::json j;
        j["ref_camera"] = ref_cam;
        j["num_frames"] = (int)frame_numbers.size();
        j["min_separation_sec"] = config.min_separation_sec;
        j["frame_numbers"] = frame_numbers;
        j["method"] = "uniform_grid";
        std::ofstream f(output_dir + "/frame_selection.json");
        if (f.is_open()) f << j.dump(2);
    }

    // ── Steps 1-2: Native extract + match ──

    auto pairwise_matches = extract_and_match_native(
        config, frame_numbers, poses, progress.get(), status);

    if (pairwise_matches.empty()) {
        result.error = "No matches found. Check model path and video files.";
        if (status) *status = result.error;
        return result;
    }

    // ── Step 3 (instant): Track building ──

    if (status) *status = "Step 3/4: Building tracks...";
    auto track_result = TrackBuilder::build_tracks(pairwise_matches);

    if (track_result.num_tracks < 10) {
        result.error = "Too few tracks (" + std::to_string(track_result.num_tracks) + ")";
        if (status) *status = result.error;
        return result;
    }

    fprintf(stderr, "[SuperPoint] Built %d tracks, %d observations\n",
            track_result.num_tracks, track_result.num_observations);

    // Save landmarks.json for diagnostics
    std::string landmarks_file = output_dir + "/landmarks.json";
    {
        nlohmann::json lj;
        for (const auto &[cam, pts] : track_result.landmarks) {
            nlohmann::json ids_arr = nlohmann::json::array();
            nlohmann::json lm_arr = nlohmann::json::array();
            for (const auto &[tid, px] : pts) {
                ids_arr.push_back(tid);
                lm_arr.push_back({px.x(), px.y()});
            }
            lj[cam] = {{"ids", ids_arr}, {"landmarks", lm_arr}};
        }
        std::ofstream f(landmarks_file);
        if (f.is_open()) f << lj.dump(2);
    }

    // ── Step 4: Bundle adjustment ──

    if (progress) progress->current_step.store(3);
    if (status) *status = "Step 4/4: Running multi-round bundle adjustment...";

    std::string refined_dir = output_dir + "/calibration";

    FeatureRefinement::FeatureConfig feat_config;
    feat_config.landmarks_file = landmarks_file;
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
    feat_config.ba_max_rounds = config.ba_max_rounds;
    feat_config.ba_convergence_eps = config.ba_convergence_eps;

    FeatureRefinement::FeatureResult feat_result =
        FeatureRefinement::run_feature_refinement(feat_config, status);

    if (!feat_result.success) {
        result.error = "Bundle adjustment failed: " + feat_result.error;
        if (status) *status = result.error;
        return result;
    }

    // ── Populate result ────────────────────────────────────────────────────

    if (progress) progress->current_step.store(4);

    result.total_tracks = feat_result.total_tracks;
    result.valid_3d_points = feat_result.valid_3d_points;
    result.total_observations = feat_result.total_observations;
    result.ba_outliers_removed = feat_result.ba_outliers_removed;
    result.mean_reproj_before = feat_result.mean_reproj_before;
    result.mean_reproj_after = feat_result.mean_reproj_after;
    result.camera_changes = feat_result.camera_changes;
    result.ba_rounds_completed = feat_result.ba_rounds_completed;
    result.per_round_reproj = feat_result.per_round_reproj;
    result.train_reproj = feat_result.train_reproj;
    result.holdout_reproj = feat_result.holdout_reproj;
    result.holdout_ratio = feat_result.holdout_ratio;
    result.holdout_observations = feat_result.holdout_observations;
    result.train_observations = feat_result.train_observations;

    // Load refined calibration for 3D viewer
    result.calib_result = load_calib_result_from_folder(refined_dir, config.camera_names);
    result.calib_result.mean_reproj_error = feat_result.mean_reproj_after;
    result.calib_result.output_folder = refined_dir;

    // Load initial calibration for comparison
    result.init_calib_result = load_calib_result_from_folder(config.calibration_folder, config.camera_names);
    result.init_calib_result.mean_reproj_error = feat_result.mean_reproj_before;

    // Load landmarks into calib_result.db
    for (const auto &[cam, pts] : track_result.landmarks)
        for (const auto &[tid, px] : pts)
            result.calib_result.db.landmarks[cam][tid] = px;

    // Write summary calibration_data.json
    {
        nlohmann::json j;
        j["pipeline"] = "superpoint_native";
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
        j["model_path"] = config.model_path;

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

#endif // __APPLE__

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
