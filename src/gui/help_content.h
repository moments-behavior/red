#pragma once
// Single source of truth for the in-app Help window (help_window.h renders
// this). Every shortcut / mouse action / tool blurb / workflow / concept lives
// here as data, tagged with a Gate so the window can hide or reorder content to
// match the current build and the open project. Keeping it as plain data (no
// ImGui) means it can later also drive the input layer so behavior and help
// cannot drift apart.
//
// All content was audited against the code (2026-08); notes on overloaded keys
// (F/T/Backspace) reflect the current, real behavior.
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
    bool midline_on   = false;  // Midline tool enabled
};

// Availability of an entry.
enum class Gate {
    Always,       // always shown
    Need3D,       // only meaningful for a calibrated 3D project
    ToolBbox,     // belongs to a tool mode: always listed in its own group, and
    ToolObb,      // that group floats to the top of Shortcuts when the tool is on
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
            {S::SaveLabels, nullptr, "Save labels (writes a new timestamped labeled_data folder)"},
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
            {S::DeleteAllKp, nullptr, "Delete all keypoints on this frame"},
            {S::Triangulate, nullptr, "Triangulate the current frame", Gate::Need3D,
                 "Needs the same keypoint in \xE2\x89\xA5 2 cameras"},
            {S::PlotMenu, nullptr, "Open the image context menu (fit axes, toggle keypoint/bbox layers)"},
            {S::PeekRaw, nullptr, "Hide this view's labels to peek at the raw image underneath"},
        }},
        {"Labeling \xE2\x80\x94 hovering a keypoint", "Hover an existing (drawn) keypoint", Gate::Always, {
            {S::COUNT, "Click", "Activate that keypoint for this camera (does not create one)"},
            {S::COUNT, "Left-drag", "Move the keypoint (clears its triangulated 3D)"},
            {S::COUNT, "R", "Delete this keypoint on this camera"},
            {S::COUNT, "F", "Delete this keypoint on all cameras"},
        }},
        {"Keypoints window", "Selecting, copying & deleting keypoint columns", Gate::Always, {
            {S::COUNT, "Click a name", "Select that keypoint column (and set it active in all cameras)"},
            {S::COUNT, "Shift / Ctrl + click", "Range-select / toggle keypoint columns in the set"},
            {S::SelectAllKeypoints, nullptr, "Select all keypoint columns (press again to clear)", Gate::Always,
                 "Then Ctrl+C copies the whole frame; Ctrl+V pastes it onto another"},
            {S::CopyKeypoints, nullptr, "Copy the selected keypoints from this frame"},
            {S::PasteKeypoints, nullptr, "Paste the copied keypoints onto this frame", Gate::Always,
                 "Overwrites those keypoints on the target frame"},
            {S::DeleteKeypoint, nullptr, "Delete keypoints", Gate::Always,
                 "Over a cell: that camera. Over a name: all cameras. Otherwise: the whole selection, all cameras"},
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
        {"Timelines & plots", "Labeling strip, Frame Drops, transport slider, frame buffer", {
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
            {"Load Project", "File > Load Project",
                "Open an annotation .redproj.", Gate::Always, "\xE2\x80\x94"},
            {"Switch Skeleton", "File > Switch Skeleton\xE2\x80\xA6",
                "Change an open project's skeleton.", Gate::Always, "No manual labels yet (re-indexes keypoints)"},
            // Annotation
            {"Labeling Tool", "Always-on panel",
                "The core keypoint panel: save, triangulate, jump between labeled frames, copy previous.", Gate::Always, "An open project"},
            {"Bbox Tool", "Tools > Bbox Tool",
                "Axis-aligned bounding boxes with multi-class and instance ids."},
            {"OBB Tool", "Tools > OBB Tool",
                "Oriented (rotated) bounding boxes via 3-click construction."},
            {"Midline Tool", "Tools > Midline Tool",
                "Reconstruct a midline (e.g. a proboscis) from one side camera + one line camera.", Gate::Need3D, "Calibrated project"},
            // Export
            {"Export Tool", "Tools > Export Tool",
                "Export labels: JARVIS, COCO, DeepLabCut, YOLO Pose/Detection, Nerfstudio.", Gate::Always, "Labeled frames"},
            {"Group JARVIS Export", "Tools > Group JARVIS Export",
                "Merge many projects/datasets into one JARVIS dataset (shared keypoints).", Gate::Always, "\xE2\x80\x94"},
            {"Import JARVIS Predictions", "Tools > Import JARVIS Predictions",
                "Read a JARVIS data3D.csv into a read-only prediction store, or straight into editable labels.", Gate::Need3D, "Calibrated project"},
            // Analysis
            {"Pose Stats", "View > Pose Stats",
                "Confidence over time for the active prediction store; promote a frame to fix it.", Gate::Always, "An active prediction store"},
            {"Frame Drops", "View > Frame Drops",
                "Visualize dropped frames and the camera sync plan.", Gate::Always, "Sync metadata (Cam*_meta.csv)"},
            {"Triangulation Diagnostics", "Tools > Triangulation Diagnostics",
                "Per-keypoint reprojection-error report (read-only).", Gate::Need3D, "Calibration + labeled frames"},
            // Settings
            {"Settings", "View > Settings",
                "Paths, display, keypoint colors, playback, hardware, and which annotation tools are enabled.", Gate::Always, "\xE2\x80\x94"},
        };
        return t;
    }();
    return v;
}

inline const std::vector<Workflow> &workflows() {
    static const std::vector<Workflow> w = {
        {"Create an annotation project", {
            "Welcome > Create Annotation Project (or Annotate menu).",
            "Pick the video folder \xE2\x80\x94 RED auto-discovers one .mp4 per camera.",
            "Set project name, root path, and skeleton (preset or a .json file).",
            "For multi-camera: choose the camera model (Projective or Telecentric) and an existing calibration folder.",
            "Create Project \xE2\x80\x94 videos load, one decoder per camera.",
        }},
        {"Label a frame", {
            "Navigate to a frame (Space, the arrow keys, Shift+arrow for x10, or the timeline).",
            "Hover a camera view. Press W to place the active keypoint; A/D switch which node is active.",
            "Place the same keypoint in \xE2\x89\xA5 2 cameras, then press T (or Triangulate) for the 3D point.",
            "Prev/Next jump between labeled frames; Copy Prev seeds from the previous labeled frame.",
            "Ctrl+S saves labels to a new timestamped labeled_data folder.",
        }},
        {"Export training data", {
            "Tools > Export Tool.",
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
        {"Camera alignment (desync fix)",
            "Cameras are hardware-triggered off a shared clock; a dropped frame desyncs everything after it. The transport bar reports the condition \xE2\x80\x94 Aligned, Uneven Ends, or Dropped Frames \xE2\x80\x94 and its Realign checkbox remaps frame index i to the same trigger instant across all cameras."},
        {"Prediction stores (.rpred)",
            "Read-only, on-disk 3D + confidence, kept separate from manual labels so a whole-video import never floods the Labeling Tool. They power the overlay and Pose Stats; \"Fix this frame\" promotes one frame into editable labels."},
        {"Scale factor",
            "JARVIS's voxel grid is integer-mm, too coarse for a ~3 mm fly. Exports inflate world units \xC3\x97N so the animal resolves; divide predicted 3D back down by N."},
    };
    return c;
}

} // namespace help
