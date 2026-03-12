# RED User Guide

This guide covers every action available to annotators in RED, from creating a
project through exporting labeled data.

---

## Table of Contents

1. [What you need before starting](#what-you-need-before-starting)
2. [Creating a new project](#creating-a-new-project)
3. [Loading an existing project](#loading-an-existing-project)
4. [Opening videos or images](#opening-videos-or-images)
5. [Playback controls](#playback-controls)
6. [Labeling keypoints](#labeling-keypoints)
7. [Triangulating 3D points](#triangulating-3d-points)
8. [Bounding box annotation](#bounding-box-annotation)
9. [Oriented bounding boxes](#oriented-bounding-boxes)
10. [Reprojection error analysis](#reprojection-error-analysis)
11. [YOLO auto-labeling](#yolo-auto-labeling)
12. [Saving labeled data](#saving-labeled-data)
13. [Loading previously saved labels](#loading-previously-saved-labels)
14. [Exporting for deep learning](#exporting-for-deep-learning)
15. [Keyboard shortcut reference](#keyboard-shortcut-reference)

---

## What you need before starting

Before launching RED you need:

1. **Video files** — one `.mp4` per camera, all synchronized to the same frame
   numbers. Image sequences (`.jpg`, `.png`, etc.) are also supported.

2. **Calibration files** — one OpenCV YAML file per camera containing the
   intrinsic matrix (`K`), distortion coefficients (`dist`), rotation matrix
   (`R`), and translation vector (`T`). These are used for 3D triangulation.

3. **A skeleton definition** — either select a built-in preset (rat, mouse,
   fly, etc.) or provide a custom JSON file describing your keypoint names and
   the edges between them (see the skeleton JSON format below).

---

## Creating a new project

When RED starts, the **Project** window appears automatically.

1. **Project name** — type a name for the project (e.g., `rat_session_01`).

2. **Project path** — click the folder icon to choose where the project will be
   saved. RED will create a `<project_name>.redproj` JSON file and a
   `labeled_data/` subdirectory here.

3. **Calibration folder** — click the folder icon to select the directory
   containing your per-camera YAML calibration files.

4. **Skeleton** — choose a built-in preset from the dropdown, or check
   "Load from JSON" and select a custom skeleton JSON file.

5. **Camera names** — type the ordered list of camera identifiers separated by
   commas or newlines. These must match the base names of the calibration YAML
   files (e.g., camera names `cam0`, `cam1` expect `cam0.yaml`, `cam1.yaml`).

6. Click **Create Project**. RED validates the configuration and creates the
   project file. The project window closes when successful.

### Skeleton JSON format

Custom skeletons are defined in a JSON file:

```json
{
    "name": "my_animal",
    "num_nodes": 5,
    "num_edges": 4,
    "node_names": ["Snout", "EarL", "EarR", "SpineBase", "TailBase"],
    "edges": [[0,1],[0,2],[0,3],[3,4]],
    "has_skeleton": true,
    "has_bbox": false,
    "has_obb": false
}
```

Set `has_bbox: true` to enable axis-aligned bounding box annotation. Set
`has_obb: true` to enable oriented bounding box annotation. Set
`has_skeleton: false` if you only want bounding boxes with no skeleton.

See `example/skeleton.json` for a complete example.

---

## Loading an existing project

**File → Load Project**, then select the `.redproj` file. RED loads the project
configuration, initializes the skeleton, and immediately opens the video file
dialog so you can select your media.

If a `labeled_data/` folder exists in the project directory and contains saved
CSV files, RED automatically loads the most recent labeling session.

---

## Opening videos or images

After creating or loading a project:

- **File → Open Video(s)** — opens a file dialog. Select one `.mp4` per camera
  (multi-select is supported). The videos are matched to cameras by the order
  you select them.

- **File → Open Images** — opens a file dialog to select an image sequence
  directory per camera.

Once loaded, all camera views appear in the main window and playback begins.

---

## Playback controls

### Play / Pause

| Action | How |
|---|---|
| Toggle play / pause | Click the **Play/Pause** button, or press **Space** |
| Seek forward | **Right Arrow** |
| Seek backward | **Left Arrow** |
| Seek forward (large step) | **Shift + Right Arrow** |
| Seek backward (large step) | **Shift + Left Arrow** |
| Jump to next labeled frame | Click **"Jump to next labeled frame"** button in the Labeling Tool |
| Jump to previous labeled frame | Click **"Copy from previous labeled frame"** button in the Labeling Tool |

### Playback speed

A **log-scale speed slider** in the playback controls panel lets you slow down
to 1/16× real-time for fine-grained inspection, or play at 1× (real-time).
The instantaneous playback speed is displayed next to the slider.

### Frame slider

Drag the **frame slider** at the top of the playback controls to jump directly
to any frame number. The slider is disabled during playback; pause first.

### Buffer view (paused mode)

When paused, a list of recently decoded frames appears in the "Display Controls"
panel. Use **,** (comma) and **.** (period) to step through the decode buffer
or click any entry to jump to that frame.

---

## Labeling keypoints

Keypoint labeling is performed with the mouse in any camera view. The
application uses the **active keypoint** concept: one keypoint is active at a
time, and clicking places that keypoint.

### The active keypoint

The current keypoint index is shown above each camera view and in the
**Labeling Tool** panel. To change which keypoint is active:

| Action | How |
|---|---|
| Select previous keypoint | **A** (while hovering any camera view) |
| Select next keypoint | **D** (while hovering any camera view) |
| Select first keypoint | **Q** (while hovering any camera view) |
| Select last keypoint | **E** (while hovering any camera view) |

### Placing a keypoint

| Action | How |
|---|---|
| Place active keypoint | **Left click** in a camera view |
| Place active keypoint at mouse | **W** (while hovering a camera view) — same as clicking |

After placing a keypoint, the active index automatically advances to the next
keypoint so you can label sequentially without changing the selection manually.

### Deleting keypoints

| Action | How |
|---|---|
| Delete hovered keypoint (current camera) | **R** (while hovering a keypoint) |
| Delete hovered keypoint (all cameras) | **F** (while hovering a keypoint) |
| Delete all keypoints on current frame | **Backspace** (while hovering a camera view) |

Individual keypoints can be moved by clicking to place them at a new position
(the existing label is replaced).

### Keypoint colors

Each keypoint has a unique color derived from HSV color space. The colored dot
appears on all cameras that have that keypoint labeled. Unlabeled keypoints show
no dot.

### Skeleton lines

If `has_skeleton: true` in the skeleton definition, lines are drawn between
connected keypoints according to the `edges` list, forming the skeleton overlay
on each camera view.

---

## Triangulating 3D points

Triangulation computes the 3D world position of a keypoint from its 2D
positions in ≥ 2 camera views.

**Requirement:** the same keypoint must be labeled in at least 2 cameras for
the current frame.

| Action | How |
|---|---|
| Triangulate all labeled keypoints | Press **T** (works globally), or click **"Triangulate"** in the Labeling Tool panel |

After triangulation:
- The keypoint appears in the **Keypoints Table** with a green color (fully
  triangulated) or yellow (labeled 2D but not yet triangulated).
- The white "T" badge appears in the table cell.
- The 3D position is stored in `KeyPoints3D` for the current frame.

You should triangulate each frame before saving to get 3D data in the output
CSV. Frames with 2D labels but no triangulation will have `NaN` in
`keypoints3d.csv`.

---

## Bounding box annotation

Bounding boxes are available when `has_bbox: true` in the skeleton definition.

### Drawing a bounding box

| Action | How |
|---|---|
| Draw a new bounding box | **Shift + drag** in a camera view |

Hold Shift and drag from one corner to the opposite corner of the region you
want to annotate. Release to finish.

### Bounding box class and ID selection

| Action | How |
|---|---|
| Switch to previous class | **Z** |
| Switch to next class (creates new if at end) | **X** |
| Create new class | **N** |
| Previous bbox ID within class | **C** |
| Next bbox ID within class | **V** |

The current class and ID are shown in the Labeling Tool panel.

### Deleting a bounding box

| Action | How |
|---|---|
| Delete hovered bounding box (current camera) | Hover over the box, then press **F** |
| Delete all instances of hovered class | Hover over the box, then press **O** |

### Keypoints inside bounding boxes

If the skeleton has both `has_bbox: true` and `has_skeleton: true` with
`has_bbox_keypoints: true`, you can label keypoints within the bounding box
region. Hover the bounding box and press **W** to place the active keypoint at
the mouse position (within the box boundaries).

---

## Oriented bounding boxes

Oriented bounding boxes (OBBs) are available when `has_obb: true` in the
skeleton definition. An OBB is defined by three click points:

1. **First axis point** — one end of the primary axis.
2. **Second axis point** — the other end of the primary axis.
3. **Perpendicular corner** — a point defining the width of the box.

| Action | How |
|---|---|
| Place next OBB corner point | **W** (while hovering a camera view) |
| Cancel incomplete OBB | **Escape** |
| Delete hovered OBB (current camera) | Hover over the OBB, then press **T** |
| Delete hovered OBB (all cameras) | Hover over the OBB, then press **F** |

---

## Reprojection error analysis

After triangulating keypoints, RED can compute and visualize reprojection
errors. Access the reprojection tool from the **View** menu or the **Labeling
Tool** panel.

The reprojection error for a keypoint is the pixel distance between:
- The original 2D label placed by the annotator.
- The 2D projection of the triangulated 3D point back through the camera.

A large reprojection error (typically > 5–10 pixels) suggests:
- Mislabeled keypoint in one or more views.
- Miscalibrated camera.
- The keypoint is genuinely ambiguous from one angle.

The tool shows:
- **Per-camera bar charts** — mean error per camera across all labeled frames.
- **Per-keypoint bar charts** — mean error per keypoint across all cameras.
- **Scatter plots** — error per frame, exposing outliers.
- **Error bars** — switch between SD and SEM via a dropdown.

---

## YOLO auto-labeling

If you have a trained YOLO model (`.pt` file in LibTorch format), RED can run
it to automatically suggest bounding boxes and keypoint positions.

1. **View → YOLO** (or the YOLO menu item) to open the YOLO settings panel.
2. Click **Load Model** to select your `.pt` model file.
3. Set the **Confidence threshold** — predictions below this are ignored.
4. Enable **Auto-labeling** to apply YOLO predictions to new frames as you
   navigate.

YOLO predictions appear as semi-transparent overlays. You can accept them as-is
or click to manually adjust individual keypoints.

---

## Saving labeled data

| Action | How |
|---|---|
| Save | **Ctrl+S**, or click **"Save Labeled Data"** in the Labeling Tool panel |

Labels are saved in `<project>/labeled_data/<timestamp>/` where `<timestamp>`
is a datetime string (e.g., `2026_02_25_11_58_40`). Each save creates a new
timestamped folder so previous saves are never overwritten.

### Output files

**`keypoints3d.csv`** — 3D triangulated keypoints, one row per labeled frame:
```
frame_id, kp0_x, kp0_y, kp0_z, kp1_x, kp1_y, kp1_z, ...
42, 12.3, -5.1, 200.4, NaN, NaN, NaN, ...
```
Un-triangulated keypoints appear as `NaN`.

**`<camera_name>.csv`** — 2D keypoints per camera, one row per labeled frame:
```
frame_id, kp0_x, kp0_y, kp0_conf, kp1_x, kp1_y, kp1_conf, ...
42, 310.5, 220.1, 1.0, NaN, NaN, 0.0, ...
```
Un-labeled keypoints have `NaN` coordinates and `0.0` confidence. Manually
labeled points have confidence `1.0`; YOLO-predicted points have the model's
confidence score.

---

## Loading previously saved labels

When you open a project that has a `labeled_data/` folder, RED automatically
loads the **most recent** timestamped save. A notification appears in the status
bar.

To load a different session, use **File → Load Labels** (if available) or
manually move or rename folders to change which session is most recent.

---

## Exporting for deep learning

Python export scripts are provided in the `data_exporter/` directory.

### JARVIS / COCO format

```bash
python data_exporter/red3d2jarvis.py \
    --input <project>/labeled_data/<timestamp>/ \
    --output <output_dir>/ \
    --calibration <calibration_folder>/ \
    --skeleton rat
```

Creates a 90/10 train/val split with one COCO JSON per camera view.

### YOLO detection format

```bash
python data_exporter/red3d2yolo.py \
    --input <project>/labeled_data/<timestamp>/ \
    --output <output_dir>/ \
    --videos <media_folder>/
```

Produces one `.txt` label file per frame per camera with normalized bounding
box coordinates.

### YOLO pose format

```bash
python data_exporter/red2yolopose.py \
    --input <project>/labeled_data/<timestamp>/ \
    --output <output_dir>/ \
    --videos <media_folder>/
```

Produces YOLO pose format labels: `class cx cy w h x0 y0 v0 x1 y1 v1 …`

See `data_exporter/README.md` for full argument documentation.

---

## Keyboard shortcut reference

### General

| Key | Action | Scope |
|---|---|---|
| `H` | Toggle help window | Global |
| `Space` | Play / pause | Global |
| `Ctrl+S` | Save labeled data | Global |
| `T` | Triangulate all labeled keypoints on current frame | Global |
| `←` | Seek backward | Global |
| `→` | Seek forward | Global |
| `Shift+←` | Seek backward (x10) | Global |
| `Shift+→` | Seek forward (x10) | Global |
| `,` | Previous entry in decode buffer | Global (paused) |
| `.` | Next entry in decode buffer | Global (paused) |

### Keypoint labeling

| Key | Action | Scope |
|---|---|---|
| `B` | Create keypoints on frame | Hover: camera view |
| Left click | Place active keypoint at mouse position | Hover: camera view |
| `W` | Place active keypoint at mouse position | Hover: camera view |
| `A` | Select previous keypoint | Hover: camera view |
| `D` | Select next keypoint | Hover: camera view |
| `Q` | Select first keypoint | Hover: camera view |
| `E` | Select last keypoint | Hover: camera view |
| `Backspace` | Delete all keypoints on current frame | Hover: camera view |
| `R` | Delete hovered keypoint (current camera) | Hover: keypoint |
| `F` | Delete hovered keypoint (all cameras) | Hover: keypoint |
| Click | Activate a keypoint | Hover: keypoint |

### Bounding boxes

| Key | Action | Scope |
|---|---|---|
| `Shift + drag` | Draw a new bounding box | Hover: camera view |
| `Z` | Switch to previous bbox class | Global |
| `X` | Switch to next bbox class (creates new if at end) | Global |
| `N` | Create new bbox class | Global |
| `C` | Previous bbox ID within class | Global |
| `V` | Next bbox ID within class | Global |
| `F` | Delete hovered bounding box (current camera) | Hover: bbox |
| `O` | Delete all instances of hovered class | Hover: bbox |
| `W` | Place keypoint inside bounding box | Hover: bbox |
| `A` | Select previous keypoint inside bbox | Hover: bbox |
| `D` | Select next keypoint inside bbox | Hover: bbox |

### Oriented bounding boxes

| Key | Action | Scope |
|---|---|---|
| `W` | Place next OBB corner point (3 clicks to complete) | Hover: camera view |
| `Escape` | Cancel incomplete OBB | Hover: camera view |
| `T` | Delete hovered OBB (current camera) | Hover: OBB |
| `F` | Delete hovered OBB (all cameras) | Hover: OBB |
| `A` | Switch OBB class backward | Hover: OBB |
| `D` | Switch OBB class forward | Hover: OBB |

### Skeleton editor

| Key | Action | Scope |
|---|---|---|
| Left click (in plot) | Add a new skeleton node | Skeleton editor |
| Drag (in plot) | Move an existing node | Skeleton editor |
| `Ctrl+Click` (node) | Select a node for edge creation | Skeleton editor |
| `Ctrl+Click` (another node) | Create or remove an edge between two nodes | Skeleton editor |
| `R` | Delete hovered node | Hover: node |
| `Escape` | Cancel edge selection | Skeleton editor |
