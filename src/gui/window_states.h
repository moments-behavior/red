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
    bool show_help = false;

    // Reset all tool window state for project switching.
    // Waits on async futures, joins threads, clears all project-specific data.
    void reset() {
        labeling = LabelingToolState{};
        // CalibrationToolState has futures + thread — wait then clear fields
        if (calibration.laser_viz.worker.joinable())
            calibration.laser_viz.worker.join();
        calibration.laser_viz.ready.clear();
        calibration.laser_viz.pending.clear();
        // Note: future destructors from std::async block until complete
        calibration.show = false;
        calibration.project_loaded = false;
        calibration.show_create_dialog = true;
        calibration.config_loaded = false;
        calibration.images_loaded = false;
        calibration.img_running = false;
        calibration.img_done = false;
        calibration.vid_running = false;
        calibration.vid_done = false;
        calibration.exp_img_running = false;
        calibration.exp_img_done = false;
        calibration.exp_vid_running = false;
        calibration.exp_vid_done = false;
        calibration.aruco_videos_loaded = false;
        calibration.tele_videos_loaded = false;
        calibration.tele_dlt_running = false;
        calibration.tele_dlt_done = false;
        calibration.tele_dlt_status.clear();
        calibration.tele_run_history.clear();
        calibration.tele_deferred_label_frames = 0;
        calibration.laser_ready = false;
        calibration.laser_running = false;
        calibration.laser_done = false;
        calibration.laser_status.clear();
        calibration.laser_show_detection = false;
        calibration.status.clear();
        // Clear stale result data (can be large)
        calibration.img_result = {};
        calibration.vid_result = {};
        calibration.exp_img_result = {};
        calibration.exp_vid_result = {};
        calibration.tele_dlt_result = {};
        calibration.laser_result = {};
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
        calibration.laser_config = {};
        annotation.show = false;
        annotation.video_folder.clear();
        annotation.discovered_cameras.clear();
        annotation.camera_selected.clear();
        annotation.status.clear();
        settings.show = false;
        transport = TransportBarState{};
        jarvis_export.show = false;
        jarvis_export.status.clear();
        jarvis_import.show = false;
        jarvis_predict.show = false;
        jarvis_predict.predict_requested = false;
        export_win.show = false;
        export_win.status.clear();
        bbox.show = false;
        bbox.enabled = false;
        obb.show = false;
        obb.enabled = false;
        sam_tool.show = false;
        sam_tool.fg_points.clear();
        sam_tool.bg_points.clear();
        sam_tool.has_pending_mask = false;
        show_help = false;
    }
};
