#pragma once
#include "imgui.h"
#include "calibration_tool.h"
#include "calibration_pipeline.h"
#include "pointsource_calibration.h"
#include "superpoint_refinement.h"
#include "telecentric_dlt.h"
#include "calib_viewer_window.h"
#include "tele_viewer_window.h"
#include "pointsource_metal.h"
#include "skeleton.h"
#include "project.h"
#include <atomic>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Forward declarations
struct AppContext;
struct DeferredQueue;

// Async detection visualization state (light spot overlay)
struct PointSourceVizState {
    // Double-buffered: "ready" results for display, "pending" being computed
    struct CamResult {
        int num_blobs = 0;
        int total_mask_pixels = 0;
        std::vector<uint8_t> rgba;       // w*h*4 RGBA mask
        int frame_num = -1;              // frame this result corresponds to
        bool uploaded = false;           // has this been uploaded to GPU?
    };
    std::vector<CamResult> ready;        // display these
    std::vector<CamResult> pending;      // being computed in background
    std::atomic<bool> computing{false};  // background work in flight
    std::thread worker;

#ifdef __APPLE__
    // Metal context for GPU-accelerated viz (macOS only)
    PointSourceMetalHandle metal_ctx = nullptr;
#endif

    // Params that triggered the current computation
    int last_green_th = -1, last_green_dom = -1;
    int last_min_blob = -1, last_max_blob = -1;

    ~PointSourceVizState() {
        if (worker.joinable()) worker.join();
#ifdef __APPLE__
        if (metal_ctx) pointsource_metal_destroy(metal_ctx);
#endif
    }
};

struct CalibrationToolState {
    // Core project state
    bool show = false;
    CalibrationTool::CalibProject project;
    bool project_loaded = false;
    bool dock_pending = false;
    bool show_create_dialog = true;
    bool subtype_chosen = false;  // workflow chooser step completed
    std::string config_path;
    CalibrationTool::CalibConfig config;
    bool config_loaded = false;
    bool images_loaded = false;
    std::string status;

    // Aruco media auto-detection results (for Create Project dialog display)
    CalibrationTool::ArucoMediaInfo calib_aruco_media_info;
    CalibrationTool::ArucoMediaInfo calib_global_reg_info;

    // Per-camera enable/disable (populated from config.cam_ordered)
    std::vector<bool> camera_enabled;

    // Initialize camera_enabled from config (call after config is loaded)
    void init_camera_enabled() {
        camera_enabled.assign(config.cam_ordered.size(), true);
    }

    // Build filtered cam_ordered containing only enabled cameras
    std::vector<std::string> enabled_cameras() const {
        std::vector<std::string> result;
        for (size_t i = 0; i < config.cam_ordered.size(); i++)
            if (i < camera_enabled.size() && camera_enabled[i])
                result.push_back(config.cam_ordered[i]);
        return result;
    }

    // ── Unified aruco calibration pipeline ──
    bool aruco_running_flag = false;
    bool aruco_done = false;
    CalibrationPipeline::CalibrationResult aruco_result;
    std::future<CalibrationPipeline::CalibrationResult> aruco_future;

    // Aruco media state (shared for images and videos)
    bool aruco_media_loaded = false;
    int aruco_start_frame = 0;
    int aruco_stop_frame = 0;      // 0 = all (only used for videos)
    int aruco_frame_step = 10;     // 30fps -> 3fps effective (only used for videos)
    int aruco_total_frames = 0;
    int aruco_video_count = 0;     // cached number of matched videos

    // Global registration: which frame from calibration media to use (0-based)
    int global_reg_frame = 0;      // user-selectable in Calibration Tool UI

    // Helper: is any aruco pipeline running?
    bool aruco_running() const {
        return aruco_running_flag;
    }

    // Telecentric
    bool tele_videos_loaded = false;
    bool tele_dlt_running = false;
    bool tele_dlt_done = false;
    std::string tele_dlt_status;
    TelecentricDLT::DLTResult tele_dlt_result;
    std::future<TelecentricDLT::DLTResult> tele_dlt_future;
    // DLT options (UI state)
    bool tele_flip_y = true;
    bool tele_square_pixels = false;
    bool tele_zero_skew = false;
    bool tele_do_ba = true;
    int tele_method = 0; // 0=Linear DLT, 1=DLT+k1, 2=DLT+k1k2
    // History of calibration runs for comparison
    std::vector<TelecentricDLT::DLTResult> tele_run_history;
    // Deferred label import (waits N frames for dock layout to stabilize)
    int tele_deferred_label_frames = 0;

    // PointSource refinement
    bool pointsource_ready = false;
    PointSourceCalibration::PointSourceConfig pointsource_config;
    CalibrationTool::ArucoMediaInfo pointsource_global_reg_info; // auto-detection for PS global reg media
    int pointsource_total_frames = 0;
    bool pointsource_running = false;
    bool pointsource_done = false;
    std::string pointsource_status;
    PointSourceCalibration::PointSourceResult pointsource_result;
    std::shared_ptr<PointSourceCalibration::DetectionProgress> pointsource_progress =
        std::make_shared<PointSourceCalibration::DetectionProgress>();
    std::future<PointSourceCalibration::PointSourceResult> pointsource_future;
    bool pointsource_show_detection = false;
    bool pointsource_focus_window = false;

    // PointSource visualization
    PointSourceVizState pointsource_viz;

    // SuperPoint refinement
    bool sp_running = false;
    bool sp_done = false;
    std::string sp_status;
    SuperPointRefinement::SPResult sp_result;
    std::shared_ptr<SuperPointRefinement::SPProgress> sp_progress =
        std::make_shared<SuperPointRefinement::SPProgress>();
    std::future<SuperPointRefinement::SPResult> sp_future;
    // UI input fields
    std::string sp_video_folder;
    std::string sp_ref_camera;
    int sp_num_sets = 50;
    float sp_scan_interval = 2.0f;
    float sp_min_separation = 5.0f;
    std::string sp_model_path;  // path to superpoint.mlpackage (auto-detected)
    float sp_reproj_thresh = 15.0f;
    float sp_rot_prior = 10.0f;
    float sp_trans_prior = 100.0f;
    int sp_max_keypoints = 4096;
    bool sp_lock_intrinsics = true;
    bool sp_lock_distortion = true;
    float sp_outlier_th1 = 10.0f;
    float sp_outlier_th2 = 3.0f;
    int sp_ba_max_rounds = 5;

    // Manual keypoint refinement — evaluation result
    struct KPCamEval { std::string serial; int labeled = 0; double mean_reproj = 0.0; };
    struct KPEvalState {
        bool success = false;
        std::string error;
        int total_keypoints = 0, total_cameras = 0, triangulated = 0;
        double mean_reproj = 0.0, max_reproj = 0.0;
        std::vector<KPCamEval> per_camera;
    };

    bool kp_skeleton_ready = false;
    bool kp_videos_loaded = false;
    bool kp_running = false;
    bool kp_refine_done = false;
    std::string kp_status;
    std::string kp_video_folder;
    int kp_num_points = 4;
    float kp_rot_prior = 50.0f;
    float kp_trans_prior = 500.0f;
    bool kp_lock_intrinsics = true;
    bool kp_lock_distortion = true;
    KPEvalState kp_eval;
    FeatureRefinement::FeatureResult kp_feat_result;
    std::future<FeatureRefinement::FeatureResult> kp_future;
    std::map<std::string, std::map<int, Eigen::Vector2d>> kp_landmarks; // saved for async

    // 3D calibration viewer (perspective)
    CalibViewerState calib_viewer;
    CalibrationPipeline::CalibrationResult loaded_result; // for loading from disk

    // 3D telecentric viewer
    TeleViewerState tele_viewer;

};

struct CalibrationToolCallbacks {
    // Load calibration images into camera windows
    std::function<void(std::map<std::string, std::string> &files)> load_images;
    // Load laser videos into camera windows
    std::function<void()> load_videos;
    // Unload all media (on close/reset)
    std::function<void()> unload_media;
    // Copy default layout ini to project folder (if not already present)
    std::function<void(const std::string &project_path)> copy_default_layout;
    // Switch imgui_layout.ini to project folder
    std::function<void(const std::string &project_path)> switch_ini;
    // Print video metadata to console
    std::function<void()> print_metadata;
    // Deferred queue for scheduling main-thread work from callbacks
    DeferredQueue *deferred = nullptr;
};

// Helper: set up a labeling skeleton with one keypoint per landmark.
// Used when opening a telecentric project, loading videos, or starting labeling.
inline void setup_landmark_skeleton(SkeletonContext &skel, int n_landmarks,
                                     ProjectManager &pm,
                                     const std::string &project_path) {
    skel.name = "Target";
    skel.num_nodes = n_landmarks;
    skel.num_edges = 0;
    skel.has_skeleton = true;
    skel.node_colors.clear();
    skel.edges.clear();
    skel.node_names.clear();
    for (int i = 0; i < n_landmarks; i++) {
        skel.node_names.push_back("Pt" + std::to_string(i));
        skel.node_colors.push_back(
            (ImVec4)ImColor::HSV(i / (float)n_landmarks, 0.8f, 0.8f));
    }
    pm.keypoints_root_folder =
        (std::filesystem::path(project_path) / "labeled_data").string();
    std::error_code ec;
    std::filesystem::create_directories(pm.keypoints_root_folder, ec);
    pm.camera_params.clear();
    pm.plot_keypoints_flag = true;
}
