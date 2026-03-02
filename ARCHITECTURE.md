# RED Software Architecture

This document describes the internal design of RED for developers who want to
understand, extend, or contribute to the codebase.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [High-Level Architecture](#high-level-architecture)
3. [Threading Model](#threading-model)
4. [GPU Pipelines](#gpu-pipelines)
   - [Linux / NVIDIA](#linux--nvidia-pipeline)
   - [macOS / Apple Silicon](#macos--apple-silicon-pipeline)
5. [Component Reference](#component-reference)
   - [Entry Point — red.cpp](#entry-point--redcpp)
   - [Project Management — project.h](#project-management--projecth)
   - [Camera Calibration — camera.h](#camera-calibration--camerah)
   - [Skeleton System — skeleton.h / skeleton.cpp](#skeleton-system--skeletonh--skeletoncpp)
   - [Decoding — decoder.h / decoder.cpp](#decoding--decoderh--decodercpp)
   - [macOS Async Decoder — vt_async_decoder.h / .mm](#macos-async-decoder--vt_async_decoderh--mm)
   - [Rendering — render.h / render.cpp](#rendering--renderh--rendercpp)
   - [macOS Metal Context — metal_context.h / .mm](#macos-metal-context--metal_contexth--mm)
   - [GUI — gui.h](#gui--guih)
   - [Keypoints Table — keypoints_table.h](#keypoints-table--keypoints_tableh)
   - [Reprojection Tool — reprojection_tool.h](#reprojection-tool--reprojection_toolh)
   - [YOLO Inference — yolo_torch.h / yolo_torch.cpp](#yolo-inference--yolo_torchh--yolo_torchcpp)
   - [YOLO Export — yolo_export.h / yolo_export.cpp](#yolo-export--yolo_exporth--yolo_exportcpp)
   - [JARVIS Export — jarvis_export.h](#jarvis-export--jarvis_exporth)
   - [Camera Math — red_math.h](#camera-math--red_mathh)
   - [OpenCV YAML I/O — opencv_yaml_io.h](#opencv-yaml-io--opencv_yaml_ioh)
   - [FFmpeg Frame Reader — ffmpeg_frame_reader.h](#ffmpeg-frame-reader--ffmpeg_frame_readerh)
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
├── src/                            # C++ source (application)
│   ├── red.cpp                     # Main entry point and UI event loop (~4000 lines)
│   ├── project.h                   # Project file load/save and setup
│   ├── camera.h                    # Camera calibration parameter types
│   ├── skeleton.h / .cpp           # Skeleton presets and keypoint structures
│   ├── decoder.h / .cpp            # Video decoder wrapper (platform-dispatch)
│   ├── render.h / .cpp             # Rendering abstraction (OpenGL or Metal)
│   ├── gui.h                       # All ImGui drawing and interaction logic (~2500 lines)
│   ├── keypoints_table.h           # ImGui table for keypoint status
│   ├── reprojection_tool.h         # Reprojection-error computation and plots
│   ├── yolo_torch.h / .cpp         # LibTorch YOLO inference
│   ├── yolo_export.h / .cpp        # YOLO-format dataset export
│   ├── global.h / .cpp             # Global state shared across modules
│   ├── types.h                     # Primitive type aliases and small structs
│   ├── utils.h / .cpp              # File/path utilities
│   │
│   ├── — Linux / NVIDIA only —
│   ├── kernel.cu / .cuh            # CUDA kernels (colorspace conversion)
│   ├── ColorSpace.cu / .h          # YUV→RGB colorspace conversion
│   ├── create_image_cuda.cu / .h   # CUDA frame assembly
│   ├── NvDecoder.h / .cpp          # NVIDIA NVDEC decoder wrapper
│   ├── FFmpegDemuxer.h / .cpp      # FFmpeg container demuxer
│   │
│   ├── — macOS / Apple Silicon only —
│   ├── metal_context.h / .mm       # Metal init, CAMetalLayer, CVMetalTextureCache
│   ├── vt_async_decoder.h / .mm    # Async VideoToolbox decoder + PTS reorder queue
│   ├── red_math.h                  # Eigen-based camera math (replaces OpenCV sfm/calib3d)
│   ├── opencv_yaml_io.h            # OpenCV YAML format reader/writer (no OpenCV dep)
│   ├── ffmpeg_frame_reader.h       # FFmpeg C API frame reader (JARVIS export)
│   ├── jarvis_export.h             # Built-in JARVIS/COCO export tool
│   │
│   └── json.hpp                    # nlohmann/json (single-header)
│
├── data_exporter/                  # Python export pipeline
│   ├── keypoints.py                # Skeleton definitions for Python side
│   ├── red3d2jarvis.py             # Export to JARVIS / COCO format
│   ├── red3d2yolo.py               # Export to YOLO detection format
│   ├── red2yolopose.py             # Export to YOLO pose format
│   ├── jarvis2red3d.py             # Import from JARVIS back to RED CSVs
│   ├── red3d2jarvis_with_phases.py # Phase-aware JARVIS export
│   ├── export_yolo_detection.py    # Standalone YOLO detection export
│   ├── export_yolo_pose.py         # Standalone YOLO pose export
│   └── utils.py                   # Shared Python helpers
│
├── packaging/
│   └── homebrew/
│       ├── red.rb                 # Homebrew formula
│       └── README.md              # Tap documentation
├── example/
│   └── skeleton.json              # Example custom skeleton (zebrafish)
├── fonts/                         # Icon and UI fonts
├── lib/                           # Third-party C++ libraries (libtorch, etc.)
├── .github/workflows/
│   └── release.yml                # GitHub Actions release workflow
├── CMakeLists.txt
├── build.sh / run.sh              # Linux build and run scripts
└── development_notes.md           # Full macOS port history
```

---

## High-Level Architecture

RED is a single-process, multi-threaded application. One thread owns the GPU
context (OpenGL on Linux, Metal on macOS) and runs the ImGui event loop. All
other threads are decoder workers, one per camera.

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Main Thread                               │
│                                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────────────┐   │
│  │ ImGui UI │──▶│  gui.h       │──▶│  GPU render                │   │
│  │ event    │   │  (draw,      │   │  Linux:  render.h (OpenGL) │   │
│  │ loop     │   │   interact)  │   │  macOS:  metal_context.mm  │   │
│  └──────────┘   └──────┬───────┘   └────────────────────────────┘   │
│                        │                                             │
│              reads frames from display_buffer[]                      │
│              uploads decoded frame to GPU texture each frame         │
│                        │                                             │
└────────────────────────┼─────────────────────────────────────────────┘
                         │  (shared circular buffer, atomic sync)
         ┌───────────────┼─────────────────────────────┐
         ▼               ▼                              ▼
   ┌───────────┐   ┌───────────┐                 ┌───────────┐
   │ Decoder   │   │ Decoder   │      …          │ Decoder   │
   │ Thread 0  │   │ Thread 1  │                 │ Thread N  │
   │ (cam 0)   │   │ (cam 1)   │                 │ (cam N)   │
   │           │   │           │                 │           │
   │ FFmpeg    │   │ FFmpeg    │                 │ FFmpeg    │
   │ demuxer   │   │ demuxer   │                 │ demuxer   │
   │    ↓      │   │    ↓      │                 │    ↓      │
   │ NVDEC     │   │ NVDEC     │  (Linux only)   │ NVDEC     │
   │ — or —    │   │ — or —    │                 │ — or —    │
   │ VTAsync   │   │ VTAsync   │  (macOS only)   │ VTAsync   │
   └───────────┘   └───────────┘                 └───────────┘
```

---

## Threading Model

| Thread | Count | Responsibility |
|---|---|---|
| Main (UI) | 1 | ImGui event loop, frame upload to GPU, user interaction, triangulation, save |
| Decoder | 1 per camera | Demux packets → hardware decode → write decoded frame to `display_buffer` |

**Synchronization**

Each camera has a `SeekInfo` struct with flags to coordinate seek requests
between the main thread and the decoder thread:

```cpp
struct SeekInfo {
    bool use_seek;       // Main thread sets this to request a seek
    bool seek_done;      // Decoder sets this when seek is complete
    uint64_t seek_frame; // Target frame number
    bool seek_accurate;  // Whether frame-accurate seek is needed
};
```

- The main thread writes `seek_frame` + sets `use_seek = true`.
- The decoder thread detects `use_seek`, performs the seek, then sets
  `seek_done = true`.
- `window_need_decoding` (in `global.h`) is a per-camera atomic bool that
  gates whether a decoder thread is actively decoding.
- `PictureBuffer.available_to_write` is a bool that prevents the decoder
  from overwriting a slot the main thread is still reading.

---

## GPU Pipelines

### Linux / NVIDIA Pipeline

```
MP4 / H.264 or H.265 bitstream
        │
        ▼
FFmpegDemuxer                    (src/FFmpegDemuxer.h)
  — reads compressed NAL packets from MP4 container
        │  raw compressed packets (CPU memory)
        ▼
NvDecoder                        (src/NvDecoder.h)
  — submits packets to NVIDIA NVDEC hardware engine
  — output: NV12 (YUV 4:2:0 planar) surfaces in GPU memory
        │  NV12 surface (GPU / CUDA device memory)
        ▼
CUDA kernels                     (src/ColorSpace.cu, create_image_cuda.cu)
  — NV12 → interleaved BGR or BGRA (GPU-side)
  — optional resize / crop
        │  BGRA frame in CUDA device memory
        ▼
PBO (Pixel Buffer Object)        (src/render.h)
  — cudaMemcpy into OpenGL PBO (CUDA–GL interop, zero-copy when possible)
  — glTexSubImage2D uploads PBO → OpenGL texture
        │
        ▼
OpenGL texture                   displayed via ImGui / ImPlot image
```

The `PictureBuffer` struct owns one slot in the circular frame ring:

```cpp
struct PictureBuffer {
    unsigned char *frame;       // CPU or GPU BGRA byte array (Linux)
    int frame_number;
    bool available_to_write;
#ifdef __APPLE__
    CVPixelBufferRef pixel_buffer;  // macOS: retained CVPixelBuffer
#endif
};
```

The circular buffer depth (`label_buffer_size` in `red.cpp`) is currently 64
frames (~355 ms at 180 fps), giving the decoder ample lookahead.

---

### macOS / Apple Silicon Pipeline

The macOS pipeline is fully GPU-side from decode through display. No CPU color
conversion or CPU memory copies occur after the demuxer stage.

```
MP4 / H.264 or H.265 bitstream
        │
        ▼
FFmpegDemuxer                    (src/FFmpegDemuxer.h)
  — reads compressed NAL packets (CPU memory, Annex-B format)
        │  raw compressed packets
        ▼
VTAsyncDecoder                   (src/vt_async_decoder.mm)
  — converts Annex-B → AVCC (4-byte length prefix) for VideoToolbox
  — builds CMSampleBufferRef with timing metadata
  — submits to VTDecompressionSession (async hardware decode)
  — output callback fires on VT internal queue with decoded CVPixelBuffer
  — PTS-sorted min-heap re-orders frames for B-frame correctness
  — main decoder thread calls pop_next() to retrieve frames in PTS order
        │  CVPixelBufferRef (BGRA, IOSurface-backed, GPU memory)
        │  VideoToolbox applies correct YUV→RGB matrix (BT.601/BT.709)
        │  internally based on stream color space metadata
        ▼
CVMetalTextureCache              (src/metal_context.mm)
  — CVMetalTextureCacheCreateTextureFromImage wraps the IOSurface
    as a MTLTexture with zero CPU involvement
  — pixel format: MTLPixelFormatBGRA8Unorm
        │  MTLTexture wrapping the same GPU memory as CVPixelBuffer
        ▼
MTLBlitCommandEncoder            (src/metal_context.mm)
  — copies BGRA IOSurface-backed texture → stable per-camera MTLTexture
  — zero CPU copy; GPU blit only
        │  stable MTLTexture (BGRA8Unorm, MTLStorageModeShared)
        ▼
imgui_impl_metal                 (lib/imgui/backends/imgui_impl_metal.mm)
  — samples MTLTexture in ImGui render pass
        │
        ▼
CAMetalLayer drawable            displayed at 1:1 pixel on Retina display
```

**Key design decisions in the macOS pipeline:**

1. **BGRA from VideoToolbox**: Requesting `kCVPixelFormatType_32BGRA` instead
   of NV12 means VT performs YUV→RGB conversion internally using the correct
   color matrix from the stream's metadata. This is correct for any camera
   source (BT.601, BT.709, full range, video range) without any shader changes.

2. **IOSurface-backed buffers**: The `kCVPixelBufferIOSurfacePropertiesKey`
   option ensures VT allocates CVPixelBuffers on IOSurface memory, which can
   be imported as Metal textures via `CVMetalTextureCache` without any copy.

3. **Async decode**: `kVTDecodeFrame_EnableAsynchronousDecompression` allows
   VT to use its internal hardware decode queue. The output callback fires
   when a frame is ready, typically within microseconds of submission on
   Apple Silicon.

4. **B-frame reorder queue**: VT may deliver frames out of PTS order when
   B-frames are present. `VTAsyncDecoder` maintains a `std::priority_queue`
   (min-heap by PTS) and only emits frames once `REORDER_DEPTH = 8` later
   frames have been received, guaranteeing correct display order.

5. **Retina display**: `CAMetalLayer.contentsScale` is set to
   `nswin.screen.backingScaleFactor` (2.0 on Retina) so the drawable matches
   the physical pixel resolution. Without this, the layer defaults to 1.0
   and ImGui renders at 2× logical size.

6. **CVMetalTextureCache flush**: Called every 60 frames to release stale
   cache entries from recycled CVPixelBuffers and prevent memory accumulation
   during long labeling sessions.

---

## Component Reference

### Entry Point — `red.cpp`

`red.cpp` (~4 000 lines) is the application entry point and contains the
entire render loop:

- **`main()`** — initializes GLFW, GPU context (OpenGL or Metal), ImGui, and
  ImPlot; enters the render loop.
- **Render loop** — calls `DrawGUI()` and subsidiary draw functions each frame.
- **Playback logic** — advances `current_frame` based on elapsed time and FPS;
  computes a log-scale playback speed (1/16× to 1×); triggers decoder seeks.
- **Frame upload** — on macOS: calls `metal_upload_pixelbuf()` for each camera
  that has a new `CVPixelBufferRef` in its `display_buffer` slot, then releases
  the retained buffer. On Linux: maps PBO and calls `glTexSubImage2D`.
- **Save logic** — writes per-frame CSV files on `Ctrl+S`.
- **Triangulation** — triangulates when the user presses `T` (Eigen DLT on
  macOS via `red_math.h`, OpenCV on Linux); stores results in `KeyPoints3D`.
- **Keyboard / mouse dispatch** — maps raw GLFW events to application actions.

---

### Project Management — `project.h`

`ProjectManager` is a plain struct serialized to/from JSON (`.redproj`) using
nlohmann/json:

| Field | Type | Description |
|---|---|---|
| `project_name` | `string` | Human-readable identifier |
| `project_path` | `string` | Absolute path to project directory |
| `project_root_path` | `string` | Parent of the project directory |
| `skeleton_name` | `string` | Built-in preset name or empty |
| `load_skeleton_from_json` | `bool` | Use JSON file instead of preset |
| `skeleton_file` | `string` | Path to custom skeleton JSON |
| `calibration_folder` | `string` | Directory with per-camera YAML files |
| `media_folder` | `string` | Directory with video files |
| `keypoints_root_folder` | `string` | Set to `project_path/labeled_data` |
| `camera_names` | `vector<string>` | Ordered list of camera identifiers |
| `camera_params` | `vector<CameraParams>` | Populated at load time from YAML |

`DrawProjectWindow()` renders the project creation dialog. `setup_project()`
validates all fields, loads camera YAML files, initializes the skeleton, and
creates the `labeled_data/` directory.

---

### Camera Calibration — `camera.h`

`CameraParams` holds the intrinsic and extrinsic matrices for one camera.
On Linux, these are `cv::Mat`; on macOS, they are Eigen types (no OpenCV
dependency):

```cpp
struct CameraParams {
    // Linux: cv::Mat; macOS: Eigen types
    Matrix3d K;                  // 3×3 intrinsic matrix
    Matrix<double,5,1> dist;     // Distortion coefficients
    Matrix3d R;                  // 3×3 rotation matrix  (world → camera)
    Vector3d rvec;               // Rodrigues rotation vector
    Vector3d tvec;               // 3×1 translation vector
    Matrix<double,3,4> P;        // 3×4 projection matrix  P = K[R|T]
};
```

`camera_load_params_from_yaml()` reads a standard OpenCV-format YAML calibration
file. On macOS, parsing uses `opencv_yaml_io.h` (a standalone YAML parser with
no OpenCV dependency). The projection matrix `P` is computed once at load time
and reused for all subsequent triangulation calls.

---

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
    bool has_skeleton;                 // Draw skeleton edges between keypoints
};
```

Built-in presets are enumerated as `SkeletonPrimitive` (rat, mouse, fly, etc.)
and populated by `skeleton_initialize()`. Custom skeletons are loaded by
`load_skeleton_json()`.

**Keypoint memory** is managed through:
- `allocate_keypoints()` — allocates `KeyPoints2D**` (cameras × nodes) and
  `KeyPoints3D*` (nodes) arrays for one frame.
- `free_keypoints()` / `free_all_keypoints()` — release memory when the frame
  map is cleared.

---

### Decoding — `decoder.h` / `decoder.cpp`

`DecoderContext` holds shared playback state:

```cpp
struct DecoderContext {
    bool decoding_flag;         // Decoder should continue running
    bool stop_flag;             // Signal to stop
    int total_num_frame;
    int estimated_num_frames;
    int gpu_index;
    int seek_interval;          // Keyframe interval for seek
    double video_fps;
};
```

Each camera runs `decoder_process()` in its own thread. The behavior is
platform-specific:

**Linux path:**
1. `FFmpegDemuxer` reads compressed packets from the MP4.
2. `NvDecoder` submits packets to NVDEC and retrieves decoded NV12 surfaces.
3. CUDA kernels convert NV12 → BGRA in GPU memory.
4. Frame is written into a `PictureBuffer` slot; `available_to_write = false`.

**macOS path:**
1. `FFmpegDemuxer` reads compressed packets (Annex-B format).
2. `VTAsyncDecoder.submit()` converts to AVCC, wraps in `CMSampleBufferRef`,
   and submits to `VTDecompressionSession` asynchronously.
3. VT fires the output callback (on an internal queue) with a decoded
   `CVPixelBufferRef` (BGRA, IOSurface-backed).
4. Callback inserts `{CVPixelBufferRef, PTS}` into the PTS min-heap.
5. `VTAsyncDecoder.pop_next()` returns frames in PTS order once the reorder
   queue depth exceeds `REORDER_DEPTH = 8`.
6. Decoder retains the `CVPixelBufferRef` and stores it in
   `PictureBuffer.pixel_buffer`.

For image-sequence inputs, `image_loader()` replaces the video path with
image loading on CPU (`stbi_load` on macOS, `cv::imread` on Linux).

---

### macOS Async Decoder — `vt_async_decoder.h` / `.mm`

`VTAsyncDecoder` wraps one `VTDecompressionSessionRef` per camera.

**Initialization** (`init`):
- Parses SPS/PPS (H.264) or VPS/SPS/PPS (HEVC) from FFmpeg extradata to build
  a `CMVideoFormatDescriptionRef`.
- Creates the `VTDecompressionSession` requesting `kCVPixelFormatType_32BGRA`
  output with `kCVPixelBufferIOSurfacePropertiesKey` for Metal-importable
  IOSurface memory.

**Annex-B → AVCC conversion** (`annexb_to_avcc`):
FFmpeg demuxes packets in Annex-B format (start codes `00 00 00 01`).
VideoToolbox requires AVCC (4-byte big-endian length prefix). The converter
scans for start codes and replaces each with the length of the following NAL
unit. A `found_next` boolean flag correctly handles the last NAL in a packet
(which has no following start code).

**Async submission** (`submit`):
- Allocates a `CMBlockBufferRef` from a malloc'd copy of the AVCC data
  (released by `kCFAllocatorMalloc` when the block is freed).
- Submits with `kVTDecodeFrame_EnableAsynchronousDecompression`.

**Output callback** (static, any thread):
- Fired by VT when a frame finishes decoding.
- Retains the `CVPixelBufferRef` and inserts `{buf, PTS}` into the mutex-
  protected `std::priority_queue` (min-heap by PTS).

**Frame retrieval** (`pop_next`):
- Returns the lowest-PTS frame from the heap only when `queue_.size() >=
  REORDER_DEPTH` or `flushing_` is set.
- Returns `nullptr` if not enough frames have arrived yet.

**Seek / flush**:
- `submit_blocking()` submits a frame then calls
  `VTDecompressionSessionFinishDelayedFrames()` to wait synchronously — used
  for frame-accurate seeks.
- `flush()` calls `VTDecompressionSessionFinishDelayedFrames()`, sets
  `flushing_ = true` so `pop_next()` drains completely, then releases all
  retained `CVPixelBufferRef`s.

---

### Rendering — `render.h` / `render.cpp`

`RenderScene` owns the per-camera GPU resources:

```cpp
struct RenderScene {
    u32 num_cams;
    u32 *image_width;
    u32 *image_height;
    u32 size_of_buffer;                // Circular buffer depth (64 frames)
#ifdef __APPLE__
    ImTextureID *image_descriptor;     // Per-camera ImTextureID (MTLTexture ptr)
#else
    GLuint *image_texture;             // Per-camera OpenGL texture handle
#endif
    PBO_CUDA *pbo_cuda;                // Linux: CUDA–GL interop PBO
    PictureBuffer **display_buffer;    // [cam][slot] circular frame rings
    SeekInfo *seek_context;            // Per-camera seek state
    bool use_cpu_buffer;
};
```

`render_allocate_scene_memory()` initializes all buffers and textures.

On macOS, `image_descriptor[j]` holds an `ImTextureID` cast from the
`id<MTLTexture>` pointer:
```objc
(ImTextureID)(intptr_t)(__bridge void*)g_ctx.textures[cam_idx]
```
`ImGui::Image()` in `red.cpp` passes this value directly to `imgui_impl_metal`,
which samples the MTLTexture in the Metal render pass.

---

### macOS Metal Context — `metal_context.h` / `.mm`

`metal_context.mm` owns all Metal objects through an Objective-C class
(`MetalCtx`) managed by ARC:

**Initialization** (`metal_init`):
- Creates `MTLDevice` (Apple Silicon GPU).
- Creates `MTLCommandQueue`.
- Attaches `CAMetalLayer` to the GLFW `NSWindow`'s `contentView`.
- Sets `layer.pixelFormat = MTLPixelFormatBGRA8Unorm` to match VT output.
- Sets `layer.contentsScale = screen.backingScaleFactor` for Retina display.
- Sets `layer.drawableSize` to the physical pixel resolution.
- Creates `CVMetalTextureCache` for zero-copy `CVPixelBuffer` → `MTLTexture`.

**Per-camera textures** (`metal_allocate_textures`):
- Allocates one `MTLTexture` per camera:
  - Format: `MTLPixelFormatBGRA8Unorm` (matches VT output)
  - Usage: `ShaderRead | ShaderWrite` (blit destination + ImGui sampling)
  - Storage: `MTLStorageModeShared` (unified CPU/GPU memory on Apple Silicon)

**Frame loop** (`metal_begin_frame` / `metal_end_frame`):
- `metal_begin_frame()`: acquires `CAMetalDrawable`, builds
  `MTLRenderPassDescriptor`, creates `MTLCommandBuffer`, calls
  `ImGui_ImplMetal_NewFrame(rpd)`, and flushes the texture cache every 60
  frames.
- `metal_end_frame()`: creates a `MTLRenderCommandEncoder`, calls
  `ImGui_ImplMetal_RenderDrawData()`, ends encoding, presents the drawable,
  and commits the command buffer.
- A single `MTLCommandBuffer` is shared between the blit pass (VT frame
  upload) and the render pass (ImGui display) to minimize driver overhead.

**Frame upload** (`metal_upload_pixelbuf`):
```
CVPixelBufferRef (BGRA, IOSurface)
        │
CVMetalTextureCacheCreateTextureFromImage  →  CVMetalTextureRef (MTLPixelFormatBGRA8Unorm)
        │
CVMetalTextureGetTexture  →  id<MTLTexture> src  (wraps same IOSurface memory)
        │
MTLBlitCommandEncoder copyFromTexture:toTexture:  →  g_ctx.textures[cam_idx]
        │
CFRelease(bgra_ref)  (IOSurface ref released; VT can recycle buffer)
```

The blit encoder is appended to the existing command buffer (created in
`metal_begin_frame`) so all GPU work for a frame is in one command buffer.

---

### GUI — `gui.h`

`gui.h` (~2 500 lines) contains all ImGui drawing functions called from the
render loop in `red.cpp`:

| Function | Purpose |
|---|---|
| `DrawMainMenuBar()` | File, View, YOLO menus |
| `DrawCameraViews()` | Lays out camera image panels using ImPlot |
| `DrawKeypointOverlay()` | Draws colored dots and skeleton lines on camera images |
| `DrawBoundingBoxOverlay()` | Draws axis-aligned and oriented bounding boxes |
| `DrawSkeletonCreator()` | Interactive node/edge editor for custom skeletons |
| `DrawYoloExportTool()` | Batch export configuration UI |
| `DrawSpreadsheet()` | Summary table of labeled frames |
| `DrawPlaybackControls()` | Frame slider, play/pause, FPS display, speed slider |

**Interaction model:** ImPlot is used as the image canvas. Mouse clicks within
a camera view are transformed from screen to image coordinates using the plot's
axis transform. A click within the snap radius of an existing keypoint selects
it; otherwise the active keypoint is placed at the click position.

---

### Keypoints Table — `keypoints_table.h`

Renders an ImGui table with one row per labeled frame and one column per
keypoint (or camera). Cells are color-coded:

| Color | Meaning |
|---|---|
| Green | Labeled and triangulated |
| Yellow | Labeled in 2D but not yet triangulated |
| Red / empty | Not labeled |

---

### Reprojection Tool — `reprojection_tool.h`

After triangulation, the 3D point is projected back into each camera via `P`.
The Euclidean distance between the projected point and the original 2D label is
the **reprojection error** in pixels.

The tool renders:
- Bar charts of mean error per camera or per keypoint.
- Scatter plots to expose camera-specific or keypoint-specific biases.
- Error bars for SD or SEM, selectable via dropdown.

---

### YOLO Inference — `yolo_torch.h` / `yolo_torch.cpp`

A LibTorch model (`.pt`) is loaded once. Each frame, `runYoloInference()` runs
the model on the decoded frame and returns a vector of `YoloPrediction` structs:

```cpp
struct YoloPrediction {
    float x, y, w, h;    // Bounding box center + size (image coords)
    float confidence;
    int class_id;
};
```

Predictions above the confidence threshold are converted to `BoundingBox`
objects and optionally used to seed keypoint positions (auto-labeling mode).

---

### YOLO Export — `yolo_export.h` / `yolo_export.cpp`

Implements the in-app YOLO dataset export dialog (Linux only). Wraps the logic
from the Python scripts for cases where the user wants to export directly from
within the GUI without running Python.

---

### JARVIS Export — `jarvis_export.h`

Header-only module (namespace `JarvisExport`) accessible via **Tools -> JARVIS
Export Tool** in the GUI. Exports labeled data to
[JARVIS-HybridNet](https://github.com/JARVIS-MoCap/JARVIS-HybridNet) COCO
training format directly from within RED, without requiring Python.

Features:
- Reads calibration YAML files and writes JARVIS-format YAML output
- Extracts JPEG frames from source videos using `ffmpeg_frame_reader.h`
- Writes COCO JSON annotation files with automatic 90/10 train/val split
- Detects and corrects **negative PTS frame offset** (some MP4s have pre-roll
  frames with negative PTS; RED's frame counter includes these but OpenCV/JARVIS
  start from PTS 0). Auto-detected via ffprobe.
- JPEG quality slider, auto output directory, timestamped export folders

On macOS, JPEG encoding uses turbojpeg (`tjCompress2`, SIMD-accelerated).
On Linux, falls back to stb_image_write.

---

### Camera Math — `red_math.h`

macOS-only header (~130 lines) providing Eigen-based replacements for OpenCV
camera math functions. Used by `gui.h` for triangulation and reprojection:

| Function | Replaces |
|---|---|
| `triangulatePoints()` | `cv::sfm::triangulatePoints` — DLT via `Eigen::JacobiSVD` |
| `projectionFromKRt()` | `cv::sfm::projectionFromKRt` — builds 3x4 P matrix |
| `rotationMatrixToVector()` / `rotationVectorToMatrix()` | `cv::Rodrigues` — via `Eigen::AngleAxisd` |
| `undistortPoint()` | `cv::undistortPoints` — iterative Brown-Conrady (10 iterations, k1-k3 + p1-p2) |
| `projectPoints()` / `projectPoint()` | `cv::projectPoints` — forward projection with full distortion model |

---

### OpenCV YAML I/O — `opencv_yaml_io.h`

macOS-only standalone parser and writer (~245 lines) for OpenCV's YAML format
(`%YAML:1.0` header, `!!opencv-matrix` tags). Allows RED to read and write
OpenCV-format calibration files without linking OpenCV.

- `read(path)` — parses a YAML file into a `YamlFile` struct with `scalars`
  and `matrices` maps
- `getInt()` / `getDouble()` / `getMatrix()` — typed accessors
- `YamlWriter` class — writes OpenCV-compatible YAML output (used by JARVIS
  export for calibration files)

---

### FFmpeg Frame Reader — `ffmpeg_frame_reader.h`

macOS-only FFmpeg C API frame reader (~220 lines) used by the JARVIS export
tool to extract JPEG frames from source videos. Three performance optimizations:

1. **VideoToolbox hardware decode** — automatically enabled via
   `AV_HWDEVICE_TYPE_VIDEOTOOLBOX`
2. **Sequential decode optimization** — when frames are requested in order,
   decodes forward instead of seeking for each frame
3. **Lazy swscale context** — created on first decoded frame so the pixel format
   matches what the decoder actually produces

---

### Global State — `global.h` / `global.cpp`

Declares `extern` variables shared across translation units:

| Variable | Type | Purpose |
|---|---|---|
| `window_need_decoding` | `unordered_map<string, atomic<bool>>` | Per-camera decode enable flag |
| `yolo_predictions` | `vector<YoloPrediction>` | Latest YOLO detections |
| `yolo_confidence_threshold` | `float` | Minimum confidence to display |
| `current_frame` | `u32` | The frame index currently displayed |

---

## Key Data Structures

### `KeyPoints2D`

One instance per (camera, keypoint) pair per labeled frame.

```cpp
struct KeyPoints2D {
    tuple_d position;       // (x, y) in image pixel coordinates
    tuple_d last_position;  // Previous position, for reprojection error delta
    bool is_labeled;        // True when the user has placed this point
    bool last_is_labeled;
    float confidence;       // 1.0 for manual; YOLO score for auto-labeled
};
```

### `KeyPoints3D`

One instance per keypoint per labeled frame.

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
    KeyPoints2D **bbox_keypoints2d;  // Optional per-bbox keypoints
    bool has_bbox_keypoints;
    u32 *active_kp_id;
};
```

### `OrientedBoundingBox`

Defined by two axis points and a perpendicular corner point. The rotation angle
and dimensions are computed from these three points at draw time.

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
User loads videos / images
  FFmpegDemuxer × N opened (one per camera)
  Decoder threads spawned (one per camera)
  RenderScene buffers allocated
  Metal textures allocated (macOS) / OpenGL textures allocated (Linux)
        │
        ▼
Render loop (display refresh rate, typically 60 Hz)
  │
  ├── For each camera:
  │     Check display_buffer for a new decoded frame
  │     macOS: call metal_upload_pixelbuf(j, pixel_buffer)
  │              → blit CVPixelBuffer → MTLTexture via CVMetalTextureCache
  │              → release CVPixelBufferRef
  │     Linux:  map PBO, cudaMemcpy, glTexSubImage2D
  │     Draw MTLTexture / GL texture in ImPlot canvas
  │
  ├── Draw keypoint overlays (colored dots + skeleton lines)
  │
  ├── Handle mouse click
  │     transform screen → image coordinates via ImPlot axis
  │     update KeyPoints2D[cam][node].position + is_labeled = true
  │
  ├── Handle keyboard input
  │     Tab / number keys: cycle active keypoint
  │     T: triangulate current frame
  │     Ctrl+S: save CSV files
  │     Space: play / pause
  │     Arrow keys / scroll: seek ±1 frame
  │     J: jump to next labeled frame
  │
  ├── User presses T
  │     collect all is_labeled 2D points for current frame (≥2 cameras required)
  │     undistort with K and dist for each labeled camera
  │     triangulate with projection matrices P (Eigen DLT on macOS, OpenCV on Linux)
  │     convert homogeneous → 3D; store in KeyPoints3D[node]
  │     set is_triangulated = true
  │
  └── User presses Ctrl+S
        write labeled_data/<timestamp>/keypoints3d.csv
        write labeled_data/<timestamp>/<cam_name>.csv for each camera
```

---

## Triangulation

Triangulation uses the Direct Linear Transform (DLT) algorithm. For each
keypoint labeled in >= 2 cameras:

1. Collect 2D pixel coordinates from all labeled cameras.
2. Undistort points using each camera's `K` and distortion coefficients.
3. Triangulate using projection matrices `P` (SVD on the constructed
   measurement matrix).
4. Convert from homogeneous coordinates to 3D Euclidean.

On Linux, steps 2-3 use OpenCV (`cv::undistortPoints`, `cv::triangulatePoints`).
On macOS, the same math is implemented in Eigen via `red_math.h`
(`red_math::undistortPoint`, `red_math::triangulatePoints`) with no OpenCV
dependency.

The result is stored in `KeyPoints3D.position` and displayed as a white "T"
badge in the keypoints table.

Reprojection error is computed post-hoc by projecting the 3D point back through
each camera's `P` and computing the Euclidean distance to the original 2D label.

---

## Saved Data Format

Labels are saved in `<project>/labeled_data/<timestamp>/` where `<timestamp>`
is a datetime string (e.g., `2026_02_25_11_58_40`), allowing multiple save
sessions to coexist in the same project.

### `keypoints3d.csv`

One row per labeled frame, `NaN` for un-triangulated keypoints.

```
frame_id, kp0_x, kp0_y, kp0_z, kp1_x, kp1_y, kp1_z, ...
42, 12.3, -5.1, 200.4, NaN, NaN, NaN, ...
```

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
3. **Filter** frames with sufficient labels (configurable minimum).
4. **Extract** JPEG frames from source videos (using FFmpeg or PyNvCodec).
5. **Write** the target format (COCO JSON, YOLO TXT, etc.).

### `red3d2jarvis.py` — JARVIS / COCO export

- Creates 90/10 train/val split by random frame selection.
- Computes bounding box from labeled keypoints with configurable margin.
- Writes calibration YAML files (transposing rotation matrices to match
  JARVIS convention).
- Produces one COCO-format JSON per camera view.

### `red3d2yolo.py` — YOLO detection export

- Derives bounding boxes from keypoint extents.
- Normalizes box coordinates to [0, 1] relative to image dimensions.
- Writes one `.txt` label file per frame per camera.

### `red2yolopose.py` — YOLO pose export

- Combines bounding box with normalized keypoint coordinates and visibility
  flags (0 = not labeled, 2 = labeled).
- Format per line: `class cx cy w h x0 y0 v0 x1 y1 v1 …`

---

## Adding a New Skeleton Preset

1. Add a new value to the `SkeletonPrimitive` enum in `skeleton.h`.
2. In `skeleton.cpp`, add a case to `skeleton_initialize()` that fills
   `num_nodes`, `num_edges`, `node_names`, `edges`, and `node_colors`.
3. Register the name in `skeleton_get_all()` so it appears in the UI dropdown.
4. Add the skeleton definition to `data_exporter/keypoints.py` if you want
   Python export scripts to understand the new preset.

For one-off experiments, use a custom JSON skeleton (the `SP_LOAD` path)
rather than modifying `skeleton.cpp`.

---

## Adding a New Export Format

1. Create a new Python script in `data_exporter/`.
2. Use `utils.py` for shared helpers (CSV loading, frame extraction, paths).
3. Import skeleton definitions from `keypoints.py`.
4. Follow the argument convention of existing scripts (`--input`, `--output`,
   `--calibration`, `--skeleton`).

To add an in-app export UI, see `jarvis_export.h` as a reference
implementation — it is a header-only module wired into the **Tools** menu in
`red.cpp`.

---

## Build System

### macOS

```bash
cmake -S . -B release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build release --config Release -j$(sysctl -n hw.logicalcpu)
```

The CMake configuration detects `APPLE` automatically. On macOS, CUDA and
OpenCV are excluded. Metal and VideoToolbox are linked as system frameworks:
`-framework Metal -framework QuartzCore -framework CoreVideo
-framework CoreMedia -framework VideoToolbox`. Eigen and libjpeg-turbo are
the only non-system dependencies beyond FFmpeg and GLFW.

Objective-C++ source files (`.mm`) are compiled with the Xcode C++ compiler.

### Linux

```bash
./build.sh
```

`build.sh` wraps the CMake configure + build steps. On first build, uncomment
lines 16–26 to compile ImGui and ImPlot object files.

### Key CMake variables (Linux)

| Variable | Purpose |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | Set to match your GPU (e.g., `75` for Turing) |
| `TORCH_DIR` | Path to LibTorch `lib/libtorch/share/cmake/Torch` |
| `OpenCV_DIR` | Path to OpenCV cmake config |

---

## External Libraries

| Library | Location | Role |
|---|---|---|
| [Dear ImGui](https://github.com/ocornut/imgui) | `lib/imgui/` | Immediate-mode GUI |
| [ImPlot](https://github.com/epezent/implot) | `lib/implot/` | Interactive plots (image canvas) |
| [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog) | `lib/ImGuiFileDialog/` | File picker dialogs |
| [imgui-filebrowser](https://github.com/AirGuanZ/imgui-filebrowser) | `lib/imgui-filebrowser/` | Alternative file browser |
| [nlohmann/json](https://github.com/nlohmann/json) | `src/json.hpp` | JSON serialization |
| [NVDEC / nvcodec](https://developer.nvidia.com/nvidia-video-codec-sdk) | `src/nvcodec/` | NVIDIA hardware decode (Linux) |
| [stb_image / stb_image_write](https://github.com/nothings/stb) | `src/stb_image.h`, `src/stb_image_write.h` | Image loading and PNG/JPEG writing |
| LibTorch | `lib/libtorch/` | PyTorch C++ frontend for YOLO inference (Linux) |
| OpenCV | system (Linux only) | Triangulation, image I/O, calibration |
| Eigen | system (macOS) | Linear algebra, triangulation, camera math |
| libjpeg-turbo | system (macOS) | SIMD-accelerated JPEG encoding (JARVIS export) |
| FFmpeg | system | Video demuxing (both platforms) |
| GLFW | system | Window and input management |
| Metal / QuartzCore | system (macOS) | GPU rendering on Apple Silicon |
| VideoToolbox | system (macOS) | Hardware video decode on Apple Silicon |
| CoreVideo / CoreMedia | system (macOS) | CVPixelBuffer, CMSampleBuffer types |
