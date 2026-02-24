# RED Software Architecture

This document describes the internal design of RED for developers who want to understand, extend, or contribute to the codebase.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [High-Level Architecture](#high-level-architecture)
3. [Threading Model](#threading-model)
4. [GPU Pipeline](#gpu-pipeline)
5. [Component Reference](#component-reference)
   - [Entry Point — red.cpp](#entry-point--redcpp)
   - [Project Management — project.h](#project-management--projecth)
   - [Camera Calibration — camera.h](#camera-calibration--camerah)
   - [Skeleton System — skeleton.h / skeleton.cpp](#skeleton-system--skeletonh--skeletoncpp)
   - [Decoding — decoder.h / decoder.cpp](#decoding--decoderh--decodercpp)
   - [Rendering — render.h / render.cpp](#rendering--renderh--rendercpp)
   - [GUI — gui.h](#gui--guih)
   - [Keypoints Table — keypoints_table.h](#keypoints-table--keypoints_tableh)
   - [Reprojection Tool — reprojection_tool.h](#reprojection-tool--reprojection_toolh)
   - [YOLO Inference — yolo_torch.h / yolo_torch.cpp](#yolo-inference--yolo_torchh--yolo_torchcpp)
   - [YOLO Export — yolo_export.h / yolo_export.cpp](#yolo-export--yolo_exporth--yolo_exportcpp)
   - [Global State — global.h / global.cpp](#global-state--globalh--globalcpp)
6. [Key Data Structures](#key-data-structures)
7. [Data Flow: Annotation Session](#data-flow-annotation-session)
8. [Triangulation](#triangulation)
9. [Saved Data Format](#saved-data-format)
10. [Python Export Pipeline](#python-export-pipeline)
11. [Adding a New Skeleton Preset](#adding-a-new-skeleton-preset)
12. [Adding a New Export Format](#adding-a-new-export-format)
13. [Build System](#build-system)
14. [External Libraries](#external-libraries)

---

## Repository Layout

```
red/
├── src/                        # C++ source (application)
│   ├── red.cpp                 # Main entry point and UI event loop
│   ├── project.h               # Project file load/save and setup
│   ├── camera.h                # Camera calibration parameter types
│   ├── skeleton.h / .cpp       # Skeleton presets and keypoint structures
│   ├── decoder.h / .cpp        # NVIDIA hardware video decoder wrapper
│   ├── render.h / .cpp         # OpenGL + CUDA-GL interop, texture management
│   ├── gui.h                   # All ImGui drawing and interaction logic
│   ├── keypoints_table.h       # ImGui table for keypoint status
│   ├── reprojection_tool.h     # Reprojection-error computation and plots
│   ├── yolo_torch.h / .cpp     # LibTorch YOLO inference
│   ├── yolo_export.h / .cpp    # YOLO-format dataset export
│   ├── global.h / .cpp         # Global state shared across modules
│   ├── types.h                 # Primitive type aliases and small structs
│   ├── utils.h / .cpp          # File/path utilities
│   ├── kernel.cu / .cuh        # CUDA kernels (colorspace conversion)
│   ├── ColorSpace.cu / .h      # YUV→RGB colorspace conversion
│   ├── create_image_cuda.cu/h  # CUDA frame assembly
│   ├── FFmpegDemuxer.h / .cpp  # FFmpeg container demuxer
│   ├── NvDecoder.h / .cpp      # NVIDIA NVDEC decoder
│   ├── json.hpp                # nlohmann/json (single-header)
│   └── (imgui, implot, …)      # Third-party UI submodule files
│
├── data_exporter/              # Python export pipeline
│   ├── keypoints.py            # Skeleton definitions for Python side
│   ├── red3d2jarvis.py         # Export to JARVIS / COCO format
│   ├── red3d2yolo.py           # Export to YOLO detection format
│   ├── red2yolopose.py         # Export to YOLO pose format
│   ├── jarvis2red3d.py         # Import from JARVIS back to RED CSVs
│   ├── red3d2jarvis_with_phases.py  # Phase-aware JARVIS export
│   ├── export_yolo_detection.py     # Standalone YOLO detection export
│   ├── export_yolo_pose.py          # Standalone YOLO pose export
│   └── utils.py                # Shared Python helpers
│
├── example/
│   └── skeleton.json           # Example custom skeleton (zebrafish)
├── fonts/                      # Icon and UI fonts
├── lib/                        # Third-party C++ libraries (libtorch, etc.)
├── CMakeLists.txt
└── build.sh / run.sh
```

---

## High-Level Architecture

RED is a single-process, multi-threaded application. One thread owns the
OpenGL context and runs the ImGui event loop; all other threads are decoder
workers, one per camera.

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Thread                          │
│                                                             │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────────┐ │
│  │ ImGui UI │──▶│  gui.h      │──▶│  OpenGL render       │ │
│  │ event    │   │  (draw,     │   │  render.h/cpp        │ │
│  │ loop     │   │   interact) │   │  (textures, PBOs)    │ │
│  └──────────┘   └──────┬──────┘   └──────────────────────┘ │
│                        │                                    │
│              reads frames from display_buffer[]             │
│                        │                                    │
└────────────────────────┼────────────────────────────────────┘
                         │  (shared circular buffer, atomic sync)
         ┌───────────────┼────────────────────────┐
         ▼               ▼                         ▼
   ┌───────────┐   ┌───────────┐           ┌───────────┐
   │ Decoder   │   │ Decoder   │    …      │ Decoder   │
   │ Thread 0  │   │ Thread 1  │           │ Thread N  │
   │ (cam 0)   │   │ (cam 1)   │           │ (cam N)   │
   │           │   │           │           │           │
   │ FFmpeg    │   │ FFmpeg    │           │ FFmpeg    │
   │ demuxer   │   │ demuxer   │           │ demuxer   │
   │   ↓       │   │   ↓       │           │   ↓       │
   │ NVDEC     │   │ NVDEC     │           │ NVDEC     │
   │ (GPU h/w) │   │ (GPU h/w) │           │ (GPU h/w) │
   └───────────┘   └───────────┘           └───────────┘
```

---

## Threading Model

| Thread | Count | Responsibility |
|---|---|---|
| Main (UI) | 1 | ImGui event loop, rendering, user interaction, triangulation, save |
| Decoder | 1 per camera | Demux packets → NVDEC → write decoded YUV frames to `display_buffer` |

**Synchronization**

- Each camera has a `SeekContext` struct with atomic `frame_number` and
  `available` flag.
- The UI thread sets a target frame number; each decoder thread seeks to that
  frame and sets `available = true` when the decoded frame is in the buffer.
- `window_need_decoding` (in `global.h`) is a per-camera atomic bool that
  gates whether a decoder thread should be running.
- No locks are held while copying frames to the display buffer; the atomic
  `available` flag prevents torn reads.

---

## GPU Pipeline

```
MP4 / H264 bitstream
       │
       ▼
FFmpegDemuxer          (src/FFmpegDemuxer.h)
  – Reads NAL packets from container
       │  raw compressed packets
       ▼
NvDecoder              (src/NvDecoder.h)
  – NVIDIA NVDEC hardware decode
  – Output: NV12 (YUV 4:2:0) on GPU memory
       │  GPU NV12 surface
       ▼
CUDA kernels           (src/ColorSpace.cu, create_image_cuda.cu)
  – NV12 → interleaved RGB or RGBA
  – Optional resize / crop
       │  RGB frame in CUDA device memory
       ▼
PBO (Pixel Buffer Object)   (src/render.h)
  – cudaMemcpy into OpenGL PBO (zero-copy path when possible)
  – glTexSubImage2D uploads PBO → GL texture
       │
       ▼
OpenGL texture         displayed via ImGui ImPlot image
```

The `PictureBuffer` struct in `decoder.h` owns one slot in the circular
frame ring. Each slot has:
- `data` — pointer to the RGB byte array (CPU or GPU depending on
  `use_cpu_buffer`)
- `frame_number` — which video frame is stored here
- `available` — atomic bool; true when the decoder has written a valid frame

---

## Component Reference

### Entry Point — `red.cpp`

`red.cpp` (~4 000 lines) contains:

- **`main()`** — initializes GLFW, OpenGL, ImGui, and ImPlot; enters the
  render loop.
- **Render loop** — calls `DrawGUI()` and all subsidiary draw functions each
  frame.
- **Playback logic** — advances `current_frame` based on elapsed time and
  FPS; triggers decoder seeks.
- **Save logic** — writes per-frame CSV files on `Ctrl+S`.
- **Triangulation** — calls `cv::sfm` (or similar) when the user presses `T`;
  stores result in `KeyPoints3D`.
- **Keyboard / mouse dispatch** — maps raw GLFW events to application actions.

### Project Management — `project.h`

`ProjectManager` is a plain struct serialized to/from JSON (`.redproj`) using
nlohmann/json. It stores:

| Field | Type | Description |
|---|---|---|
| `project_name` | `string` | Human-readable identifier |
| `project_path` | `string` | Absolute path to the project directory |
| `project_root_path` | `string` | Parent of the project directory |
| `skeleton_name` | `string` | Built-in preset name or empty |
| `load_skeleton_from_json` | `bool` | Use JSON file instead of preset |
| `skeleton_file` | `string` | Path to custom skeleton JSON |
| `calibration_folder` | `string` | Directory containing per-camera YAML files |
| `media_folder` | `string` | Directory containing video / image files |
| `keypoints_root_folder` | `string` | Set automatically to `project_path/labeled_data` |
| `camera_names` | `vector<string>` | Ordered list of camera identifiers |
| `camera_params` | `vector<CameraParams>` | Populated at load time from YAML files |

**`DrawProjectWindow()`** renders the creation dialog; **`setup_project()`**
validates fields, loads camera YAML files, initializes the skeleton, and
creates the `labeled_data/` directory.

### Camera Calibration — `camera.h`

`CameraParams` holds the OpenCV intrinsic and extrinsic matrices for one
camera:

```cpp
struct CameraParams {
    cv::Mat K;        // 3×3 intrinsic matrix
    cv::Mat dist;     // Distortion coefficients
    cv::Mat R;        // 3×3 rotation matrix  (world → camera)
    cv::Mat T;        // 3×1 translation vector
    cv::Mat P;        // 3×4 projection matrix  P = K[R|T]
};
```

`camera_load_params_from_yaml()` reads a standard OpenCV YAML calibration
file. The projection matrix `P` is computed once at load time and reused for
triangulation.

### Skeleton System — `skeleton.h` / `skeleton.cpp`

**`SkeletonContext`** describes the anatomy being labeled:

```cpp
struct SkeletonContext {
    int num_nodes;                     // Number of keypoints
    int num_edges;                     // Number of limb connections
    vector<string> node_names;         // e.g., {"Snout", "EarL", ...}
    vector<tuple_i> edges;             // Pairs: {(0,1), (1,2), ...}
    vector<ImVec4> node_colors;        // HSV-generated, one per node
    string name;
    bool has_bbox;                     // Supports bounding boxes
    bool has_obb;                      // Supports oriented bounding boxes
    bool has_skeleton;                 // Draw skeleton edges
};
```

Built-in presets are enumerated as `SkeletonPrimitive` and populated by
`skeleton_initialize()`. Custom skeletons are loaded by
`load_skeleton_json()`.

**Keypoint memory** is managed through:
- `allocate_keypoints()` — allocates `KeyPoints2D**` (cameras × nodes) and
  `KeyPoints3D*` (nodes) arrays.
- `free_keypoints()` / `free_all_keypoints()` — release memory when the frame
  map is cleared.

### Decoding — `decoder.h` / `decoder.cpp`

`DecoderContext` holds shared playback state (target frame, FPS, seek
interval). Each camera runs `decoder_process()` in its own thread:

1. `FFmpegDemuxer` reads compressed packets from the MP4.
2. `NvDecoder` submits packets to NVDEC and retrieves decoded NV12 surfaces.
3. CUDA kernels convert NV12 → RGB and write into a `PictureBuffer` slot.
4. The slot's `available` atomic is set to `true`.

For image-sequence inputs, `image_loader()` replaces step 1–3 with an OpenCV
`imread` + colorspace conversion.

The **circular buffer** size is configurable (`label_buffer_size` in
`red.cpp`). A larger buffer allows the decoder to prefetch more frames but
requires more VRAM.

### Rendering — `render.h` / `render.cpp`

`RenderScene` owns all GPU-side resources:

| Field | Description |
|---|---|
| `num_cams` | Number of active cameras |
| `image_width[]`, `image_height[]` | Per-camera resolution |
| `display_buffer[]` | Array of circular `PictureBuffer` rings, one per camera |
| `seek_context[]` | Per-camera seek state (target frame, available flag) |
| `use_cpu_buffer` | If true, frames are in CPU memory (for debugging or non-CUDA systems) |

`render_allocate_scene_memory()` initializes all buffers. OpenGL PBOs are
created once per camera; each frame, the latest decoded buffer is mapped and
copied to the PBO, which is then uploaded to a GL texture via
`glTexSubImage2D`.

### GUI — `gui.h`

`gui.h` (~2 500 lines) is the largest single file. It contains inline functions
that are called from the render loop in `red.cpp`:

| Function | Purpose |
|---|---|
| `DrawMainMenuBar()` | File, View, YOLO menus |
| `DrawCameraViews()` | Lays out camera image panels with ImPlot |
| `DrawKeypointOverlay()` | Draws colored dots and skeleton lines on camera images |
| `DrawBoundingBoxOverlay()` | Draws axis-aligned and oriented bounding boxes |
| `DrawSkeletonCreator()` | Interactive node/edge editor for custom skeletons |
| `DrawYoloExportTool()` | Batch export configuration UI |
| `DrawSpreadsheet()` | Summary table of labeled frames |
| `DrawPlaybackControls()` | Frame slider, play/pause, FPS display |

**Interaction model:** ImPlot is used as the image canvas. Mouse clicks within
a camera view are transformed from screen to image coordinates using the plot's
axis transform. A click within the snap radius of an existing keypoint selects
it; otherwise the active keypoint is placed at the click position.

### Keypoints Table — `keypoints_table.h`

Renders an ImGui table with one row per labeled frame and one column per
keypoint (or camera). Cells are color-coded:

| Color | Meaning |
|---|---|
| Green | Labeled and triangulated |
| Yellow | Labeled in 2D but not triangulated |
| Red / empty | Not labeled |

### Reprojection Tool — `reprojection_tool.h`

After triangulation, the 3D point is projected back into each camera using the
projection matrix `P`. The Euclidean distance between the projected point and
the original 2D label is the **reprojection error** in pixels.

The tool renders:
- Bar charts of mean error per camera or per keypoint.
- Scatter plots to expose camera-specific or keypoint-specific biases.
- Error bars for SD or SEM, selectable via a dropdown.

### YOLO Inference — `yolo_torch.h` / `yolo_torch.cpp`

A LibTorch model (`.pt`) is loaded once. Each frame, `runYoloInference()`
runs the model on the decoded RGB frame and returns a vector of
`YoloPrediction` structs:

```cpp
struct YoloPrediction {
    float x, y, w, h;    // Bounding box center + size (image coords)
    float confidence;
    int class_id;
};
```

Predictions above the confidence threshold are converted to `BoundingBox`
objects and optionally used to seed keypoint positions (auto-labeling mode).

### YOLO Export — `yolo_export.h` / `yolo_export.cpp`

Implements the in-app YOLO dataset export dialog. Wraps the logic from the
Python scripts for cases where the user wants to export directly from the GUI.

### Global State — `global.h` / `global.cpp`

Declares extern variables shared across translation units:

| Variable | Type | Purpose |
|---|---|---|
| `window_need_decoding` | `unordered_map<string, atomic<bool>>` | Per-camera decode enable flag |
| `yolo_predictions` | `vector<YoloPrediction>` | Latest YOLO detections |
| `yolo_confidence_threshold` | `float` | Minimum confidence to display |
| `current_frame` | `u32` | The frame index currently displayed |

---

## Key Data Structures

### `KeyPoints2D`

One instance per (camera, keypoint) pair per frame.

```cpp
struct KeyPoints2D {
    tuple_d position;       // (x, y) in image pixel coordinates
    tuple_d last_position;  // Previous position, for reprojection error delta
    bool is_labeled;        // True when the user has placed this point
    bool last_is_labeled;
    float confidence;       // 1.0 for manual; YOLO score for auto
};
```

### `KeyPoints3D`

One instance per keypoint per frame.

```cpp
struct KeyPoints3D {
    triple_d position;      // (x, y, z) in world coordinates (mm or camera units)
    bool is_triangulated;
    float confidence;
};
```

### `KeyPoints`

One instance per labeled frame, stored in a `map<u32, KeyPoints*>` keyed by
frame number.

```cpp
struct KeyPoints {
    KeyPoints3D *kp3d;                           // [num_nodes]
    KeyPoints2D **kp2d;                          // [num_cams][num_nodes]
    u32 *active_id;                              // Active keypoint index per camera
    vector<vector<BoundingBox>> bbox2d_list;     // [cam][bbox_idx]
    vector<vector<OrientedBoundingBox>> obb2d_list;
};
```

### `BoundingBox`

```cpp
struct BoundingBox {
    ImPlotRect *rect;            // (x_min, x_max, y_min, y_max) in image coords
    RectState state;             // Drawing state machine
    int class_id, id;
    float confidence;
    KeyPoints2D **bbox_keypoints2d;  // Optional per-bbox keypoints (e.g., sub-object)
    bool has_bbox_keypoints;
    u32 *active_kp_id;
};
```

### `OrientedBoundingBox`

Defined by two axis points and a perpendicular corner point. The rotation
angle and dimensions are computed from these three points.

---

## Data Flow: Annotation Session

```
User opens project (.redproj)
        │
        ▼
ProjectManager loaded from JSON
Camera YAML files parsed → CameraParams[]
Skeleton initialized → SkeletonContext
        │
        ▼
User loads videos/images
  FFmpegDemuxer × N opened
  decode threads spawned
  RenderScene buffers allocated
        │
        ▼
Render loop (60 Hz)
  ├── For each camera:
  │     check display_buffer for new frame
  │     upload to GL texture via PBO
  │     draw texture in ImPlot canvas
  │
  ├── Draw keypoint overlays (dots + skeleton lines)
  │
  ├── Handle mouse click → update KeyPoints2D[cam][node].position
  │
  ├── User presses T
  │     collect all is_labeled 2D points for current frame
  │     call cv::triangulatePoints / sfm for each keypoint
  │     store result in KeyPoints3D[node]
  │     set is_triangulated = true
  │
  └── User presses Ctrl+S
        write keypoints3d.csv
        write <cam_name>.csv for each camera
```

---

## Triangulation

Triangulation uses OpenCV's linear triangulation. For each keypoint that has
been labeled in ≥ 2 cameras:

1. Collect the 2D pixel coordinates from each labeled camera.
2. Apply `cv::undistortPoints()` using the camera's `K` and `dist` matrices.
3. Call `cv::triangulatePoints()` with the projection matrices `P` for each
   camera pair (or all cameras via DLT).
4. Convert from homogeneous coordinates to 3D.

The result is stored in `KeyPoints3D.position` and displayed as a white "T"
badge in the keypoints table.

Reprojection error is computed post-hoc by projecting the 3D point back
through each `P` and comparing to the original 2D label.

---

## Saved Data Format

Labels are saved in `<project>/labeled_data/<timestamp>/` where
`<timestamp>` is a UNIX timestamp string, allowing multiple save sessions to
coexist.

### `keypoints3d.csv`

One row per labeled frame.

```
frame_id, kp0_x, kp0_y, kp0_z, kp1_x, kp1_y, kp1_z, ...
42, 12.3, -5.1, 200.4, NaN, NaN, NaN, ...
```

Columns are ordered by keypoint index (matching `SkeletonContext.node_names`).
Un-triangulated keypoints are written as `NaN`.

### `<camera_name>.csv`

One row per frame, one triplet per keypoint.

```
frame_id, kp0_x, kp0_y, kp0_conf, kp1_x, kp1_y, kp1_conf, ...
42, 310.5, 220.1, 1.0, NaN, NaN, 0.0, ...
```

Un-labeled keypoints have `NaN` coordinates and `0.0` confidence.

---

## Python Export Pipeline

All export scripts share a common pattern:

1. **Discover** CSV files in `labeled_data/<timestamp>/`.
2. **Load** `keypoints3d.csv` and per-camera CSVs into pandas DataFrames.
3. **Filter** frames that have sufficient labels (configurable minimum).
4. **Extract** JPEG frames from source videos (using FFmpeg or PyNvCodec).
5. **Write** the target format (COCO JSON, YOLO TXT, etc.).

### `red3d2jarvis.py` — JARVIS / COCO export

Key steps:
- Creates 90 / 10 train / val split by random frame selection.
- Computes a bounding box from labeled keypoints with a configurable margin.
- Writes calibration YAML files (transposing rotation matrices to match
  JARVIS convention).
- Produces one COCO-format JSON per camera view.

### `red3d2yolo.py` — YOLO detection export

- Derives bounding boxes from keypoint extents.
- Normalizes box coordinates to [0, 1] relative to image dimensions.
- Writes one `.txt` file per frame per camera.

### `red2yolopose.py` — YOLO pose export

- Combines bounding box with normalized keypoint coordinates and visibility
  flags (0 = not labeled, 1 = labeled but potentially occluded, 2 = labeled
  and visible).
- Format: `class cx cy w h x0 y0 v0 x1 y1 v1 …`

---

## Adding a New Skeleton Preset

1. Add a new value to the `SkeletonPrimitive` enum in `skeleton.h`.
2. In `skeleton.cpp`, add a case to `skeleton_initialize()` that fills in
   `num_nodes`, `num_edges`, `node_names`, `edges`, and `node_colors`.
3. Register the name in `skeleton_get_all()` so it appears in the UI dropdown.
4. Add the corresponding skeleton definition to `data_exporter/keypoints.py`
   if you want Python export scripts to understand the new preset.

For one-off experiments, using a custom JSON skeleton (the `SP_LOAD` path) is
preferred over modifying `skeleton.cpp`.

---

## Adding a New Export Format

1. Create a new Python script in `data_exporter/`.
2. Use `utils.py` for shared helpers (CSV loading, frame extraction, path
   handling).
3. Import skeleton definitions from `keypoints.py`.
4. Follow the argument convention of existing scripts (`--input`, `--output`,
   `--calibration`, `--skeleton`).

If the format also needs an in-app export UI, implement it in
`yolo_export.h/cpp` and wire a menu item in `gui.h`.

---

## Build System

RED uses CMake. `build.sh` wraps the CMake configure + build steps. Key
CMake targets:

| Target | Output |
|---|---|
| `redgui` | Main application executable |

Important CMake variables:

| Variable | Purpose |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | Set to match your GPU (e.g., `75` for Turing) |
| `TORCH_DIR` | Path to LibTorch `lib/libtorch/share/cmake/Torch` |
| `OpenCV_DIR` | Path to OpenCV cmake config |

ImGui and ImPlot object files must be compiled separately on the first build
(see lines 16–26 of `build.sh`).

---

## External Libraries

| Library | Location | Role |
|---|---|---|
| [Dear ImGui](https://github.com/ocornut/imgui) | `src/imgui/` | Immediate-mode GUI |
| [ImPlot](https://github.com/epezent/implot) | `src/implot/` | Interactive plots (used as image canvas) |
| [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog) | `src/ImGuiFileDialog/` | File picker dialogs |
| [imgui-filebrowser](https://github.com/AirGuanZ/imgui-filebrowser) | `src/imgui-filebrowser/` | Alternative file browser |
| [nlohmann/json](https://github.com/nlohmann/json) | `src/json.hpp` | JSON serialization (single header) |
| [NVDEC / nvcodec](https://developer.nvidia.com/nvidia-video-codec-sdk) | `src/nvcodec/` | NVIDIA hardware video decode |
| [stb_image_write](https://github.com/nothings/stb) | `src/stb_image_write.h` | PNG/JPEG writing (single header) |
| LibTorch | `lib/libtorch/` | PyTorch C++ frontend for YOLO inference |
| OpenCV | system | Triangulation, image I/O, calibration |
| FFmpeg | system | Video demuxing |
| GLFW | system | Window management and OpenGL context |
