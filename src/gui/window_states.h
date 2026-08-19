#pragma once
#include "gui/labeling_tool_window.h"
#include "gui/annotation_dialog.h"
#include "gui/settings_window.h"
#include "gui/transport_bar.h"
#include "gui/jarvis_export_window.h"
#include "gui/jarvis_import_window.h"
#include "gui/pose_stats_window.h"
#include "gui/frame_drops_window.h"
#include "gui/pump_events_window.h"
#include "gui/export_window.h"
#include "gui/group_export_window.h"
#include "gui/bbox_tool.h"
#include "gui/obb_tool.h"
#include "gui/midline_tool.h"
#include "gui/triangulation_diagnostics_window.h"
#include "gui/switch_skeleton_window.h"

// Bundle of all tool-window states.
struct WindowStates {
    LabelingToolState labeling;
    AnnotationDialogState annotation;
    SettingsState settings;
    TransportBarState transport;
    JarvisExportState jarvis_export;
    JarvisImportState jarvis_import;
    PoseStatsState pose_stats;
    FrameDropsState frame_drops;
    PumpEventsState pump_events;
    ExportWindowState export_win;
    GroupExportState group_export;
    BBoxToolState bbox;
    OBBToolState obb;
    MidlineToolState midline;
    TriangulationDiagnosticsState triangulation_diag;
    SwitchSkeletonState switch_skeleton;
    bool show_help = false;

    // Reset all tool window state for project switching.
    // Waits on async futures, joins threads, clears all project-specific data.
    void reset() {
        labeling = LabelingToolState{};
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
        jarvis_import.done = false;
        jarvis_import.summary.clear();
        jarvis_import.error.clear();
        jarvis_import.store_to_load.clear();
        pose_stats = PoseStatsState{};
        frame_drops = FrameDropsState{};
        pump_events = PumpEventsState{};
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
        // Group export is standalone (not tied to the open project), but clear
        // its transient state on switch. The merge thread holds its own copy of
        // the source list, so clearing here is safe.
        group_export.show = false;
        group_export.sources.clear();
        group_export.output_dir.clear();
        group_export.status.clear();
        group_export.in_progress.store(false);
        group_export.images_saved.store(0);
        group_export.images_total = 0;
        group_export.finished.store(false);
        group_export.finished_status.reset();
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
        // Reset midline tool but keep any user default cam selection re-derived
        // on next project open (inited=false forces re-pick).
        midline = MidlineToolState{};
        triangulation_diag = TriangulationDiagnosticsState{};
        switch_skeleton = SwitchSkeletonState{};
        show_help = false;
    }
};
