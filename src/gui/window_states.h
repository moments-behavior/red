#pragma once
#include "gui/labeling_tool_window.h"
#include "gui/calib_tool_state.h"
#include "gui/annotation_dialog.h"
#include "gui/settings_window.h"
#include "gui/transport_bar.h"
#include "gui/jarvis_export_window.h"
#include "gui/jarvis_import_window.h"
#include "gui/jarvis_predict_window.h"
#include "gui/export_window.h"
#include "gui/bbox_tool.h"
#include "gui/obb_tool.h"
#include "gui/sam_tool.h"
#ifdef RED_HAS_MUJOCO
#include "gui/body_model_window.h"
#endif

// Bundle of all tool-window states.  Inference-engine states (JarvisState,
// JarvisCoreMLState, SamState) are intentionally excluded — those are
// heavyweight runtime objects, not UI window states.
struct WindowStates {
    LabelingToolState labeling;
    CalibrationToolState calibration;
    AnnotationDialogState annotation;
    SettingsState settings;
    TransportBarState transport;
    JarvisExportState jarvis_export;
    JarvisImportState jarvis_import;
    JarvisPredictState jarvis_predict;
    ExportWindowState export_win;
    BBoxToolState bbox;
    OBBToolState obb;
    SamToolState sam_tool;
#ifdef RED_HAS_MUJOCO
    BodyModelState body_model;
#endif
    bool show_help = false;

    // Reset all tool window state for project switching.
    // Waits on async futures, joins threads, clears all project-specific data.
    void reset() {
        labeling = LabelingToolState{};
        // CalibrationToolState has futures + thread — wait then clear fields
        if (calibration.pointsource_viz.worker.joinable())
            calibration.pointsource_viz.worker.join();
        calibration.pointsource_viz.ready.clear();
        calibration.pointsource_viz.pending.clear();
        // Note: future destructors from std::async block until complete
        calibration.show = false;
        calibration.project_loaded = false;
        calibration.show_create_dialog = true;
        calibration.subtype_chosen = false;
        calibration.config_loaded = false;
        calibration.images_loaded = false;
        // Unified aruco pipeline
        calibration.aruco_running_flag = false;
        calibration.aruco_done = false;
        calibration.aruco_media_loaded = false;
        // PointSource refinement
        calibration.pointsource_ready = false;
        calibration.pointsource_running = false;
        calibration.pointsource_done = false;
        calibration.pointsource_status.clear();
        calibration.pointsource_show_detection = false;
        // SuperPoint
        calibration.sp_running = false;
        calibration.sp_done = false;
        calibration.sp_status.clear();
        // Manual keypoint
        calibration.kp_skeleton_ready = false;
        calibration.kp_running = false;
        calibration.kp_refine_done = false;
        calibration.kp_videos_loaded = false;
        calibration.kp_status.clear();
        // Telecentric
        calibration.tele_videos_loaded = false;
        calibration.tele_dlt_running = false;
        calibration.tele_dlt_done = false;
        calibration.tele_dlt_status.clear();
        calibration.tele_run_history.clear();
        calibration.tele_deferred_label_frames = 0;
        // General
        calibration.status.clear();
        // Clear stale result data (can be large)
        calibration.aruco_result = {};
        calibration.tele_dlt_result = {};
        calibration.pointsource_result = {};
        calibration.loaded_result = {};
        // Null raw pointers to prevent dangling references
        calibration.tele_viewer.show = false;
        calibration.tele_viewer.dlt_result = nullptr;
        calibration.tele_viewer.landmarks_3d.clear();
        calibration.calib_viewer.show = false;
        calibration.calib_viewer.result = nullptr;
        // Clear stale project/config data
        calibration.project = {};
        calibration.config = {};
        calibration.config_path.clear();
        calibration.pointsource_config = {};
        calibration.dock_pending = false;
        calibration.aruco_start_frame = 0;
        calibration.aruco_stop_frame = 0;
        calibration.aruco_frame_step = 10;
        calibration.aruco_total_frames = 0;
        calibration.aruco_video_count = 0;
        calibration.tele_flip_y = true;
        calibration.tele_square_pixels = false;
        calibration.tele_zero_skew = false;
        calibration.tele_do_ba = true;
        calibration.tele_method = 0;
        calibration.pointsource_total_frames = 0;
        calibration.pointsource_focus_window = false;
        calibration.pointsource_progress = std::make_shared<PointSourceCalibration::DetectionProgress>();
        annotation.show = false;
        annotation.video_folder.clear();
        annotation.discovered_cameras.clear();
        annotation.camera_selected.clear();
        annotation.status.clear();
        settings.show = false;
        transport = TransportBarState{};
        jarvis_export.show = false;
        jarvis_export.status.clear();
        jarvis_export.output_dir.clear();
        jarvis_export.in_progress = false;
        jarvis_export.label_folder.clear();
        jarvis_export.label_display.clear();
        jarvis_export.label_cache_key.clear();
        jarvis_import.show = false;
        jarvis_import.data3d_path.clear();
        jarvis_import.conf_threshold = 0.0f;
        jarvis_import.done = false;
        jarvis_import.result = {};
        jarvis_predict.show = false;
        jarvis_predict.predict_requested = false;
        jarvis_predict.models_folder.clear();
        jarvis_predict.confidence_threshold = 0.1f;
        jarvis_predict.convert_job.reset();
        jarvis_predict.convert_status.clear();
        jarvis_predict.cached_models_folder.clear();
        jarvis_predict.cached_has_onnx = false;
        jarvis_predict.cached_has_pth = false;
        jarvis_predict.cached_has_coreml = false;
        jarvis_predict.cached_center_path.clear();
        jarvis_predict.cached_keypoint_path.clear();
        jarvis_predict.cached_info_path.clear();
        jarvis_predict.model_dir_display.clear();
        export_win.show = false;
        export_win.format_idx = 0;
        export_win.include_video_index = false;
        export_win.status.clear();
        export_win.output_dir.clear();
        export_win.margin = 50.0f;
        export_win.train_ratio = 0.9f;
        export_win.seed = 42;
        export_win.jpeg_quality = 95;
        export_win.in_progress.store(false);
        export_win.images_saved.store(0);
        export_win.images_total = 0;
        export_win.finished_status.reset();
        export_win.finished.store(false);
        export_win.label_folder.clear();
        export_win.label_display.clear();
        export_win.label_cache_key.clear();
        bbox.show = false;
        bbox.enabled = false;
        bbox.drawing = false;
        bbox.class_names.clear();
        bbox.class_colors.clear();
        bbox.current_class = 0;
        bbox.current_instance = 0;
        obb.show = false;
        obb.enabled = false;
        obb.draw_state = OBBDrawState::Idle;
#ifdef RED_HAS_MUJOCO
        if (body_model.renderer) {
            mujoco_renderer_destroy(body_model.renderer);
            body_model.renderer = nullptr;
        }
        body_model.show = false;
        body_model.auto_solve = false;
        body_model.last_solved_frame = -1;
        body_model.model_path.clear();
        body_model.show_site_markers = true;
        body_model.show_target_lines = true;
        body_model.cam_lookat[0] = 0; body_model.cam_lookat[1] = 0; body_model.cam_lookat[2] = 0.05f;
        body_model.cam_distance = 0.5f;
        body_model.cam_azimuth = 135.0f;
        body_model.cam_elevation = -25.0f;
        body_model.dragging = false;
        mujoco_ik_reset(body_model.ik_state);
#endif
        sam_tool.show = false;
        sam_tool.enabled = false;
        sam_tool.fg_points.clear();
        sam_tool.bg_points.clear();
        sam_tool.multi_mask = {};
        sam_tool.selected_mask = 0;
        sam_tool.current_polygons.clear();
        sam_tool.has_pending_mask = false;
        sam_tool.prompt_frame = 0;
        sam_tool.prompt_cam = -1;
        sam_tool.model_idx = 0;
        sam_tool.encoder_path = "models/mobilesam/mobile_sam_encoder.onnx";
        sam_tool.decoder_path = "models/mobilesam/mobile_sam_decoder.onnx";
        show_help = false;
    }
};
