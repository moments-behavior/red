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
};
