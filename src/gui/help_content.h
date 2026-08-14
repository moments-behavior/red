#pragma once
// Single source of truth for the in-app Help window (help_window.h renders
// this). Every shortcut / mouse action / tool blurb / workflow / concept lives
// here as data, tagged with a Gate so the window can hide or reorder content to
// match the current build and the open project. Keeping it as plain data (no
// ImGui) means it can later also drive the input layer so behavior and help
// cannot drift apart.
//
// All content was audited against the code (2026-08); notes on overloaded keys
// (F/T/Backspace) and the calibration-vs-body 3D rotate convention reflect the
// current, real behavior.
#include "gui/shortcuts.h"
#include <string>
#include <vector>

namespace help {

// Built at the call site from WindowStates + ProjectManager. A plain struct so
// this header stays decoupled from those types.
struct Context {
    bool project_open = false;
    bool is_3d        = false;  // calibrated multi-camera (triangulation live)
    bool bbox_on      = false;  // Bbox tool enabled
    bool obb_on       = false;  // OBB tool enabled
    bool sam_on       = false;  // SAM Assist enabled
    bool midline_on   = false;  // Midline tool enabled
};

// Availability of an entry.
enum class Gate {
    Always,       // always shown
    Need3D,       // only meaningful for a calibrated 3D project
    ToolBbox,     // belongs to a tool mode: always listed in its own group, and
    ToolObb,      // that group floats to the top of Shortcuts when the tool is on
    ToolSam,
    ToolMidline,
};

struct Shortcut {
    keys::Sc sc;                 // a bound key: label is derived from shortcuts.h
    const char *lit;             // literal label when sc == keys::Sc::COUNT
    const char *action;
    Gate gate = Gate::Always;
    const char *note = "";       // caveat / overload note (optional)
};
struct Group {
    const char *title;
    const char *subtitle;        // context line (optional)
    Gate group_gate = Gate::Always; // whole-group gate (used for tool groups)
    std::vector<Shortcut> items;
};

struct MouseAction { const char *input; const char *effect; const char *note = ""; };
struct MouseGroup  { const char *title; const char *subtitle; std::vector<MouseAction> items; };

struct Tool {
    const char *name;
    const char *open;            // how to open it
    const char *desc;            // one-liner
    Gate gate = Gate::Always;
    const char *needs = "";      // precondition (optional)
};

struct Workflow { const char *title; std::vector<const char *> steps; };
struct Concept  { const char *term; const char *def; };

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

inline const std::vector<Group> &shortcut_groups() {
    using S = keys::Sc;
    static const std::vector<Group> g = {
        {"Global", "Available any time", Gate::Always, {
            {S::ToggleHelp, nullptr, "Toggle this Help window"},
            {S::PlayPause, nullptr, "Play / pause"},
            {S::SeekBack, nullptr, "Step back one frame", Gate::Always, "Hold Shift for x10"},
            {S::SeekFwd, nullptr, "Step forward one frame", Gate::Always, "Hold Shift for x10"},
            {S::COUNT, "[  /  ]", "Previous / next pump dispense", Gate::Always,
                 "Needs a loaded pumpctl log; skips pumps hidden in the Pump Events filter"},
            {S::SaveLabels, nullptr, "Save labels (writes a new timestamped labeled_data folder)"},
            {S::PredictCurrent, nullptr, "Run JARVIS prediction on the current frame", Gate::Need3D,
                 "Paused, with a JARVIS model loaded"},
        }},
        {"When paused", "Stepping through the frame buffer", Gate::Always, {
            {S::BufferPrev, nullptr, "Previous buffered frame"},
            {S::BufferNext, nullptr, "Next buffered frame"},
        }},
        {"Labeling \xE2\x80\x94 hovering an image", "With a skeleton loaded; hover a camera view", Gate::Always, {
            {S::CreateFrame, nullptr, "Create the keypoint set for this frame"},
            {S::PlaceKeypoint, nullptr, "Place the active keypoint at the cursor, then advance to the next node"},
            {S::ActivePrev, nullptr, "Previous active keypoint"},
            {S::ActiveNext, nullptr, "Next active keypoint"},
            {S::ActiveFirst, nullptr, "Jump active keypoint to the first node"},
            {S::ActiveLast, nullptr, "Jump active keypoint to the last node"},
            {S::DeleteAllKp, nullptr, "Delete all keypoints on this frame", Gate::Always,
                 "While SAM has prompts, Backspace undoes a SAM point instead"},
            {S::Triangulate, nullptr, "Triangulate the current frame", Gate::Need3D,
                 "Needs the same keypoint in \xE2\x89\xA5 2 cameras"},
            {S::PlotMenu, nullptr, "Open the image context menu (fit axes, toggle keypoint/mask/bbox layers)"},
            {S::PeekRaw, nullptr, "Hide this view's labels to peek at the raw image underneath"},
        }},
        {"Labeling \xE2\x80\x94 hovering a keypoint", "Hover an existing (drawn) keypoint", Gate::Always, {
            {S::COUNT, "Click", "Activate that keypoint for this camera (does not create one)"},
            {S::COUNT, "Left-drag", "Move the keypoint (clears its triangulated 3D)"},
            {S::COUNT, "R", "Delete this keypoint on this camera"},
            {S::COUNT, "F", "Delete this keypoint on all cameras"},
        }},
        {"Bbox tool", "When the Bbox tool is enabled", Gate::ToolBbox, {
            {S::COUNT, "Shift + drag", "Draw a box (committed when you release Shift)"},
            {S::COUNT, "F", "Delete the hovered box on this camera"},
            {S::COUNT, "O", "Delete the hovered box on all cameras"},
            {S::COUNT, "Z  /  X", "Previous / next class"},
            {S::COUNT, "N", "New class"},
            {S::COUNT, "C  /  V", "Previous / next instance id"},
        }},
        {"OBB tool", "When the OBB tool is enabled", Gate::ToolObb, {
            {S::COUNT, "G  (\xC3\x97""3)", "Place axis point 1, axis point 2, then the corner"},
            {S::COUNT, "Esc", "Cancel the current box"},
            {S::COUNT, "Delete", "Delete the hovered box"},
        }},
        {"SAM Assist", "When SAM Assist is enabled (needs ONNX Runtime)", Gate::ToolSam, {
            {S::COUNT, "Left-click", "Add a foreground point"},
            {S::COUNT, "Right-click", "Add a background point"},
            {S::COUNT, "Shift + scroll", "Cycle mask candidates"},
            {S::COUNT, "Enter", "Accept the mask"},
            {S::COUNT, "Backspace", "Undo the last point"},
            {S::COUNT, "Esc", "Clear all prompts"},
        }},
        {"Midline tool", "When the Midline tool is enabled", Gate::ToolMidline, {
            {S::COUNT, "Click, click", "Place the two line endpoints (in the line camera)"},
            {S::COUNT, "W", "Label the midline keypoints in the side camera (as usual)"},
        }},
    };
    return g;
}

inline const std::vector<MouseGroup> &mouse_groups() {
    static const std::vector<MouseGroup> g = {
        {"Camera / image views", "Every camera view is a zoomable plot", {
            {"Left-drag (empty area)", "Pan the image"},
            {"Scroll", "Zoom at the cursor"},
            {"Double-click", "Fit the image to the view"},
            {"Right-drag", "Box-zoom to a rectangle"},
            {"Click a keypoint", "Activate it", "Does not create one \xE2\x80\x94 press W to place"},
            {"Drag a keypoint", "Move it", "Clears its triangulated 3D"},
            {"Hover a keypoint", "Show its 3D coordinate (if triangulated)"},
        }},
        {"3D calibration viewers", "Calibration / Telecentric 3D scenes", {
            {"Right-drag", "Rotate the scene"},
            {"Left-drag", "Pan"},
            {"Scroll / Middle-drag", "Zoom"},
            {"Click a camera or point", "Select / isolate it"},
            {"Double-click (empty)", "Deselect and reset the view"},
        }},
        {"3D body-model viewport", "MuJoCo body-model view", {
            {"Left-drag", "Orbit / rotate", "Matches MuJoCo's native viewer; the calibration viewers rotate on right-drag"},
            {"Right-drag", "Pan"},
            {"Middle-drag / Scroll", "Zoom"},
            {"Double-click", "Reset the view"},
        }},
        {"Timelines & plots", "Labeling strip, Pose Stats, Frame Drops, transport slider, frame buffer", {
            {"Click", "Seek to that frame"},
            {"Drag", "Pan the plot"},
            {"Scroll", "Zoom"},
            {"Double-click", "Reset to the full range"},
            {"Ctrl/Cmd + click the transport slider", "Type an exact frame number"},
        }},
    };
    return g;
}

inline const std::vector<Tool> &tools() {
    static const std::vector<Tool> v = [] {
        std::vector<Tool> t = {
            // Projects
            {"Create Annotation Project", "Annotate menu / Welcome",
                "Define a new project over per-camera videos: skeleton, camera model, calibration.", Gate::Always, "A loaded video/folder"},
            {"Load Project", "File \xE2\x86\x92 Load Project",
                "Open a .redproj (auto-routes calibration vs annotation projects).", Gate::Always, "\xE2\x80\x94"},
            {"Switch Skeleton", "File \xE2\x86\x92 Switch Skeleton\xE2\x80\xA6",
                "Change an open project's skeleton.", Gate::Always, "No manual labels yet (re-indexes keypoints)"},
            // Annotation
            {"Labeling Tool", "Always-on panel",
                "The core keypoint panel: save, triangulate, jump between labeled frames, copy previous.", Gate::Always, "An open project"},
            {"Bbox Tool", "Tools \xE2\x86\x92 Bbox Tool",
                "Axis-aligned bounding boxes with multi-class and instance ids."},
            {"OBB Tool", "Tools \xE2\x86\x92 OBB Tool",
                "Oriented (rotated) bounding boxes via 3-click construction."},
            {"Midline Tool", "Tools \xE2\x86\x92 Midline Tool",
                "Reconstruct a midline (e.g. a proboscis) from one side camera + one line camera.", Gate::Need3D, "Calibrated project"},
            {"SAM Assist", "Tools \xE2\x86\x92 SAM Assist",
                "Point-prompt segmentation (MobileSAM / SAM 2.1).", Gate::Always, "Built with ONNX Runtime + a model"},
            // Prediction / JARVIS
            {"JARVIS Predict", "Tools \xE2\x86\x92 JARVIS Predict",
                "Load a trained model and run pose inference (current frame or batch).", Gate::Need3D, "Calibrated project + a model"},
            {"Export Tool", "Tools \xE2\x86\x92 Export Tool",
                "Export labels: JARVIS, COCO, DeepLabCut, YOLO Pose/Detection, Nerfstudio.", Gate::Always, "Labeled frames"},
            {"Group JARVIS Export", "Tools \xE2\x86\x92 Group JARVIS Export",
                "Merge many projects/datasets into one JARVIS dataset (shared keypoints).", Gate::Always, "\xE2\x80\x94"},
            {"Import 3D Predictions", "Tools \xE2\x86\x92 Import 3D Predictions",
                "Stream a cluster keypoints3d.csv into a read-only prediction store.", Gate::Need3D, "Calibrated project"},
            // Analysis
            {"Pose Stats", "View \xE2\x86\x92 Pose Stats",
                "Confidence over time for a prediction store; promote frames to fix.", Gate::Always, "An active prediction store"},
            {"Bouts", "View \xE2\x86\x92 Bouts",
                "Segment the video into contiguous well-tracked ranges.", Gate::Always, "An active prediction store"},
            {"Bout Filter", "View \xE2\x86\x92 Bout Filter",
                "Validated walking-bout pipeline with a per-candidate accept/reject reason.", Gate::Always, "An active store + a profile"},
            {"Frame Drops", "View \xE2\x86\x92 Frame Drops",
                "Visualize dropped frames and the camera sync plan.", Gate::Always, "Sync metadata (Cam*_meta.csv)"},
            {"Pump Events", "View \xE2\x86\x92 Pump Events",
                "Place pumpctl dispenses on the video timeline; click a row to seek, or jump with [ / ].", Gate::Always, "A pumpctl dispense log + frame timestamps"},
            {"Triangulation Diagnostics", "Tools \xE2\x86\x92 Triangulation Diagnostics",
                "Per-keypoint reprojection-error report (read-only).", Gate::Need3D, "Calibration + labeled frames"},
            // Calibration
            {"Calibration Tool", "Calibrate menu / Welcome",
                "Produce multi-camera calibration: ArUco, PointSource, Telecentric DLT, SuperPoint, Manual.", Gate::Always, "\xE2\x80\x94"},
            // Settings
            {"Settings", "View \xE2\x86\x92 Settings",
                "Paths, display, keypoint colors, playback, hardware, and which annotation tools are enabled.", Gate::Always, "\xE2\x80\x94"},
        };
#ifdef RED_HAS_MUJOCO
        t.push_back({"Body Model", "Tools \xE2\x86\x92 Body Model",
            "Fit a MuJoCo body model to the project's 3D keypoints via inverse kinematics.", Gate::Need3D, "A model file + triangulated 3D"});
#endif
        return t;
    }();
    return v;
}

inline const std::vector<Workflow> &workflows() {
    static const std::vector<Workflow> w = {
        {"Create an annotation project", {
            "Welcome \xE2\x86\x92 Create Annotation Project (or Annotate menu).",
            "Pick the video folder \xE2\x80\x94 RED auto-discovers one .mp4 per camera.",
            "Set project name, root path, and skeleton (preset or a .json file).",
            "For multi-camera: choose the camera model (Projective or Telecentric) and a calibration folder.",
            "Create Project \xE2\x80\x94 videos load, one decoder per camera.",
        }},
        {"Label a frame", {
            "Navigate to a frame (Space, \xE2\x86\x90/\xE2\x86\x92, Shift+\xE2\x86\x90/\xE2\x86\x92, or the timeline).",
            "Hover a camera view. Press W to place the active keypoint; A/D switch which node is active.",
            "Place the same keypoint in \xE2\x89\xA5 2 cameras, then press T (or Triangulate) for the 3D point.",
            "Prev/Next jump between labeled frames; Copy Prev seeds from the previous labeled frame.",
            "Ctrl+S saves labels to a new timestamped labeled_data folder.",
        }},
        {"Run JARVIS prediction", {
            "Tools \xE2\x86\x92 JARVIS Predict \xE2\x86\x92 Import Model (RED auto-detects the backend from the files).",
            "Import to Project copies + registers the model; pick it under Project Models.",
            "Predict Current Frame (key 6), or Batch Predict a range \xE2\x86\x92 a prediction store.",
            "Toggle the video overlay; review in Pose Stats / Bouts.",
            "Promote a frame to editable labels via Pose Stats \xE2\x80\x9C" "Fix this frame\xE2\x80\x9D, correct, re-export.",
        }},
        {"Calibrate cameras", {
            "Calibrate \xE2\x86\x92 Create Calibration Project (or a Welcome button) and pick a method.",
            "Provide the media (ChArUco board / light-wand / landmarks) the method needs.",
            "Run the pipeline and check the reprojection error (< 3 px excellent, 3\xE2\x80\x93" "8 ok).",
            "Calibration YAML/DLT files are written into the project for triangulation.",
        }},
        {"Find behavior bouts", {
            "Batch Predict a range with Send to: Predictions store.",
            "Open Bouts (generic good-tracking) or Bout Filter (validated walking bouts).",
            "Tune thresholds; each candidate shows why it was accepted/rejected.",
            "Export CSV.",
        }},
        {"Export training data", {
            "Tools \xE2\x86\x92 Export Tool.",
            "Pick a format (JARVIS, COCO, DeepLabCut, YOLO, Nerfstudio).",
            "Set the output directory and split options; Export (labels auto-save first).",
        }},
    };
    return w;
}

inline const std::vector<Concept> &concepts() {
    static const std::vector<Concept> c = {
        {"Camera models",
            "Projective (pinhole, <cam>.yaml) vs Telecentric (affine DLT, <cam>_dlt.csv). Chosen at project creation; it drives triangulation and export."},
        {"Triangulation",
            "A 3D point exists only once the same keypoint is labeled in \xE2\x89\xA5 2 cameras. Press T, or it auto-triangulates."},
        {"Sync / desync fix",
            "Cameras are hardware-triggered off a shared clock; a dropped frame desyncs everything after it. The Sync toggle maps frame index i to the same trigger instant across all cameras."},
        {"Pump dispense alignment",
            "pumpctl stamps each dispense with the same PTP hardware clock the cameras write into Cam*_meta.csv, so a dispense maps to a frame by lookup, not by fitting. The Offset (ms) slider is for rig latency (tubing, valve), not clock drift."},
        {"Prediction stores (.rpred)",
            "Read-only, on-disk 3D + confidence, kept separate from manual labels so whole-video prediction never floods the Labeling Tool. They power the overlay, Pose Stats, and Bouts."},
        {"Model sets",
            "Several models, each covering a different keypoint group, run over the same frames and are concatenated into one combined-skeleton store."},
        {"Scale factor",
            "JARVIS's voxel grid is integer-mm, too coarse for a ~3 mm fly. Exports inflate world units \xC3\x97N so the animal resolves; divide predicted 3D back down by N."},
        {"Bout vs Bout Filter",
            "Bouts = generic ranges of good tracking. Bout Filter = a validated walking-bout pipeline with arena geometry and a rejection reason per candidate."},
    };
    return c;
}

} // namespace help
