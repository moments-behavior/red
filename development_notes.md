# RED — macOS Apple Silicon Port: Development Notes

This document records the full history of porting RED from a Linux/NVIDIA-only
application to a native macOS Apple Silicon build. The work was done by Rob
Johnson (Johnson Lab, HHMI Janelia) with Claude Code assistance across five
development phases spanning February 24 – March 1, 2026.

All macOS changes use `#ifdef __APPLE__` guards. The Linux/NVIDIA build is
completely untouched throughout.

---

## Timeline and Branches

| Branch | Dates | Purpose |
|---|---|---|
| `rob_dev` | Feb 24–25 | Initial macOS port (FFmpeg + VideoToolbox + OpenGL) |
| `rob_dev_vulkan` | Feb 25–26 | Vulkan/MoltenVK rendering backend |
| `rob_dev_metal` | Feb 26 – Mar 1 | Native Metal rendering, async VT decode, Homebrew tap, JARVIS export |
| `rob_dev_no_opencv` | Mar 1 | Remove OpenCV dependency, replace with Eigen + turbojpeg |

Each branch builds on the previous one (linear history, no merges to main).
Main branch merging is handled by Jinyao Yan.

---

## Phase 1 — Initial macOS Port (branch: `rob_dev`)

**Commits:** `a7f371e` through `0289aa9` (Feb 24–25)

### Goal

Get RED compiling and running on macOS Apple Silicon with minimal changes.
Replace NVIDIA-specific subsystems with macOS equivalents.

### Replacements

| Linux/NVIDIA | macOS replacement |
|---|---|
| NVDEC hardware decode | FFmpeg `avcodec` with VideoToolbox hwaccel |
| CUDA NV12→RGB kernels | FFmpeg `sws_scale` (CPU, single-threaded) |
| `cudaMalloc` frame buffers | `malloc` CPU buffers |
| CUDA-GL interop / PBO upload | `glTexSubImage2D` from CPU buffer |
| TensorRT / LibTorch YOLO | Stub functions (YOLO not needed on Mac) |

### Key changes

- **`CMakeLists.txt`**: Split into `if(APPLE) ... else() ... endif()` blocks.
  macOS block uses `pkg_check_modules` for FFmpeg, `find_package` for OpenCV
  and GLFW, links Apple frameworks (VideoToolbox, CoreMedia, etc.).
- **`src/decoder.h/cpp`**: Entire Linux decode path wrapped in `#ifndef __APPLE__`.
  New macOS path: FFmpeg demux → `avcodec_send_packet`/`avcodec_receive_frame` with
  VideoToolbox hwaccel → `av_hwframe_transfer_data` (GPU→CPU) → `sws_scale` (NV12→RGBA).
- **`src/render.h/cpp`**: On macOS, always `malloc` for frame buffers; skip all
  PBO/CUDA interop.
- **`src/red.cpp`**: Texture upload uses `glTexSubImage2D` on macOS. YOLO blocks
  guarded out.
- **`src/gx_helper.h`**: OpenGL 3.3 core profile on macOS (`GLFW_OPENGL_FORWARD_COMPAT`),
  `GL_SILENCE_DEPRECATION`, GLSL `#version 150`.
- **`src/Logger.h`**: Fixed `sockaddr_in` aggregate init — macOS struct has an
  extra `sin_len` field that broke the Linux initializer. Replaced with explicit
  field assignment.
- **`src/FFmpegDemuxer.cpp`**: `AVInputFormat::read_seek` removed in FFmpeg 6+.
  Guarded with `#ifdef __APPLE__` to use `pb->seekable` instead.
- Various CUDA headers/types guarded out: `ColorSpace.h`, `kernel.cuh`,
  `create_image_cuda.h`, `AppDecUtils.h`.

### Performance

~0.30x real-time (180 fps, 3-camera, 3208x2200). The bottleneck was
`sws_scale` running single-threaded on CPU.

---

## Phase 2 — Vulkan Rendering Backend (branch: `rob_dev_vulkan`)

**Commits:** `8d33169`, `4f174c4`, `f2cdff6`, `1ee518b`, `1993147`, `c0959b9`
(Feb 25–26)

### Motivation

macOS deprecated OpenGL in 2018. Apple Silicon runs OpenGL through a Metal
compatibility layer with overhead. Vulkan via MoltenVK translates to native
Metal calls with less overhead.

### New files

- **`src/vulkan_context.h/cpp`**: Complete Vulkan state and operations.
  `VulkanContext` struct owns instance, device, queue, swapchain, command pool,
  per-camera `VulkanTexture` (VkImage + persistently-mapped staging VkBuffer +
  VkDescriptorSet as ImTextureID).
- Used `VK_KHR_dynamic_rendering` (no explicit renderpass/framebuffer objects).
- 1 frame in flight, shared sampler, persistently-mapped staging buffers.

### vImage color conversion (0.30x → 0.88x)

Replaced `sws_scale` with Apple's vImage framework (`vImageConvert_420Yp8_CbCr8ToARGB8888`):
- Eliminated `av_hwframe_transfer_data` — `CVPixelBufferLockBaseAddress` reads
  the IOSurface-backed VideoToolbox output in-place (zero copy).
- vImage uses NEON SIMD + automatic GCD tiling across all CPU cores.
  `sws_scale` was single-threaded.

### Other improvements in this phase

- **Directory argument fix**: `./red /path/to/project_dir` now resolves the
  `.redproj` file automatically instead of crashing.
- **Font and ini paths**: Resolved using `argv[0]` absolute paths, independent
  of working directory.
- **Decoder thread QoS**: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)`
  ensures decoder threads run on performance (P) cores.
- **Log-scale playback speed slider**: 1/16x to 1x for frame-by-frame review.

### Profiling (macOS `sample` tool)

| Thread | Blocked in | % of time |
|---|---|---|
| Main (render) | `CAMetalLayer nextDrawable` vsync | ~47% (expected) |
| Decoder (per camera) | `mach_msg2_trap` (VT XPC IPC) | ~93% |
| vImage GCD pool | `vImageConvert_420Yp8_CbCr8ToARGB8888` | active |

The bottleneck was VideoToolbox synchronous IPC, not CPU conversion.

---

## Phase 3 — Native Metal + Async VideoToolbox (branch: `rob_dev_metal`)

**Commits:** `4a4e491` through `d127aeb` (Feb 26)

This was the largest single change, eliminating all three remaining bottlenecks
in one commit followed by iterative refinements.

### 3a. Native Metal rendering (replaces Vulkan/MoltenVK)

**New files:**
- **`src/metal_context.h`**: C++ interface — no Objective-C types exposed.
- **`src/metal_context.mm`**: Obj-C++ implementation. Internal `MetalCtx` class
  owns `MTLDevice`, `MTLCommandQueue`, `CAMetalLayer`, per-camera `MTLTexture`
  array, `CVMetalTextureCacheRef`.

Frame loop: `CAMetalLayer.nextDrawable` → `MTLRenderPassDescriptor` →
`ImGui_ImplMetal_NewFrame(rpd)` → ImGui render → `presentDrawable` → `commit`.

On Apple Silicon, `MTLTexture` with `MTLStorageModeShared` is directly
CPU-writable — `[tex replaceRegion:withBytes:]` replaces the entire Vulkan
staging buffer pipeline.

**Retina fix**: `CAMetalLayer.contentsScale` defaults to 1.0. Without setting it
to `screen.backingScaleFactor`, all UI elements render at 2x their intended size
on Retina displays.

### 3b. GPU NV12→RGBA via Metal compute shader

VideoToolbox returns `CVPixelBuffer` backed by `IOSurface`. These can be
imported as Metal textures via `CVMetalTextureCacheCreateTextureFromImage` with
zero CPU involvement.

```
VideoToolbox → CVPixelBuffer (IOSurface)
    → CVMetalTextureCache → MTLTexture (Y plane, R8Unorm)
    → CVMetalTextureCache → MTLTexture (UV plane, RG8Unorm)
    → Metal compute shader → MTLTexture (RGBA output for ImGui)
```

No CPU color conversion. No staging buffer. No `memcpy`. The compute encoder
runs on the same `MTLCommandBuffer` as the ImGui render pass.

### 3c. Async VideoToolbox decode

**New files:**
- **`src/vt_async_decoder.h`**: C++ class interface.
- **`src/vt_async_decoder.mm`**: Obj-C++ implementation.

Replaced FFmpeg's synchronous `avcodec_receive_frame` (which blocked in
`mach_msg2_trap` 93% of the time) with direct VideoToolbox API calls:
`VTDecompressionSessionCreate` + `kVTDecodeFrame_EnableAsynchronousDecompression`.

Key design:
- **Format description**: Created from SPS/PPS (H.264) or VPS/SPS/PPS (HEVC)
  extracted from container extradata.
- **Annex-B → AVCC conversion**: FFmpegDemuxer outputs Annex-B start codes;
  VideoToolbox requires AVCC length-prefixed NALs. `annexb_to_avcc()` handles
  the conversion.
- **Async output callback**: VT delivers decoded frames on an internal thread;
  they're pushed into a mutex-protected priority queue.
- **B-frame reorder queue**: `std::priority_queue` sorted by PTS, emits frames
  only when queue depth exceeds `REORDER_DEPTH` (initially 4, later increased
  to 8).
- **Seek**: Flush + destroy + recreate VT session. Uses `submit_blocking()` +
  `drain_one()` for frame-accurate synchronous seek.

### Critical bug fix: AVCC NAL truncation

`annexb_to_avcc()` had a scanning loop `while (j + 2 < size)` that stopped 2
bytes before buffer end. The last 2 bytes of every packet were silently dropped,
causing `kVTVideoDecoderBadDataErr` (-12909) on every frame. Fixed with a
`found_next` flag to distinguish "found next start code" from "reached end of
buffer."

### Later refinement: BGRA direct output (commit `d127aeb`)

Discovered that VideoToolbox can output BGRA directly (not just NV12) by
setting the `CVPixelBuffer` format to `kCVPixelFormatType_32BGRA` in the
decoder configuration. This eliminated the Metal compute shader entirely —
VideoToolbox applies the correct YUV→RGB matrix (BT.601/BT.709, full/video
range) internally based on stream metadata.

Final pipeline:
```
FFmpeg demux → VTDecompressionSession (async) → CVPixelBuffer (IOSurface, BGRA)
    → CVMetalTextureCache (zero-copy) → MTLBlitCommandEncoder
    → MTLTexture (BGRA) → imgui_impl_metal → CAMetalLayer → display
```

### Performance result

| Pipeline | Speed |
|---|---|
| Phase 1: FFmpeg + sws_scale + OpenGL | ~0.30x |
| Phase 2: FFmpeg + vImage + Vulkan/MoltenVK | ~0.88x |
| Phase 3: Async VT + native Metal | **1.0x real-time** |

---

## Phase 4 — Homebrew Tap and Distribution (branch: `rob_dev_metal`)

**Commits:** `acc6697` through `d0ee527` (Feb 26–28)

### Homebrew formula (`packaging/homebrew/red.rb`)

Created a Homebrew formula for one-command install:
```bash
brew tap JohnsonLabJanelia/red
brew install --HEAD JohnsonLabJanelia/red/red
```

The formula:
- Declares build deps (`cmake`, `pkg-config`) and runtime deps
- Fetches submodules for HEAD installs
- Passes `HOMEBREW_PREFIX` to CMake (works on both Apple Silicon `/opt/homebrew`
  and Intel `/usr/local`)
- Installs fonts to `share/red/fonts/`

### Tap repository

`JohnsonLabJanelia/homebrew-red` (currently private, will be made public before
preprint release). Lives at `/opt/homebrew/Library/Taps/johnsonlabjanelia/homebrew-red/`.

### GitHub Actions release workflow (`.github/workflows/release.yml`)

Triggered by version tag push (`v*`). On `macos-14` runner:
1. Checkout with submodules
2. Install Homebrew deps
3. CMake configure + build
4. Package binary tarball (`red-v1.0.0-macos-arm64.tar.gz`)
5. Create fat source tarball with bundled submodules (for Homebrew formula)
6. Compute SHA256 hashes
7. Publish to GitHub Releases

### Other fixes

- **PATH-based launch** (`ec34af7`): When installed via Homebrew, `argv[0]` is
  just `"red"` (not a full path). Used `_NSGetExecutablePath` to resolve the
  real executable path for font/resource lookup.
- **Multi-location font search** (`gx_helper.h`): Checks `exe_dir/../fonts`
  then `exe_dir/../share/red/fonts` (Homebrew layout).
- **Homebrew 5.0 compatibility** (`4be6678`): `bottle :unneeded` was removed in
  Homebrew 5.0. Added `uses_from_macos "xcode" => :build`.
- **Installation test instructions** (`Install_Instructions.md`): Step-by-step
  guide for beta testers covering SSH keys, Xcode CLT, tap, install, Gatekeeper.

---

## Phase 5 — JARVIS Export Tool (branch: `rob_dev_metal`)

**Commits:** `764b494` through `d0d1409` (Feb 28 – Mar 1)

### Built-in C++ JARVIS export (`src/jarvis_export.h`)

Header-only module (namespace `JarvisExport`) that exports labeled data to
JARVIS-HybridNet COCO training format directly from the GUI (Tools menu).
Previously this required running a separate Python script.

Features:
- Reads calibration YAML files and writes JARVIS-format YAML output
- Extracts JPEG frames from source videos at labeled frame positions
- Writes COCO JSON annotation files with train/val split
- Detects and corrects negative PTS frame offset (some MP4s have pre-roll
  frames with negative PTS; RED's frame counter includes these but OpenCV
  skips them)
- JPEG quality slider, auto output directory, timestamped export folders

### Negative PTS bug (`69a28f3`)

Some H.264 MP4 files have pre-roll frames with negative PTS. RED's `nFrame`
counter starts from the first packet (frame 0 = first packet, which may have
negative PTS), but OpenCV and JARVIS start from PTS 0. Auto-detected via
`detect_negative_pts_offset()` using ffprobe.

### Frame extraction performance

Used `src/ffmpeg_frame_reader.h` (FFmpeg C API) with three optimizations:
1. VideoToolbox hardware decoding (automatic on macOS)
2. Sequential decode optimization — when frames are in order, decode forward
   instead of seeking for each frame
3. Lazy swscale context for pixel format flexibility

---

## Phase 6 — Remove OpenCV (branch: `rob_dev_no_opencv`)

**Commits:** `aa2df0d`, `dbef2e4` (Mar 1)

### Motivation

OpenCV was RED's heaviest dependency. On macOS via Homebrew, `opencv` pulls in
~65 transitive packages (ceres-solver, boost, vtk, qt, protobuf, etc.), making
installs slow and fragile. RED used only a small fraction of OpenCV.

### Replacements

| OpenCV usage | Replacement | Location |
|---|---|---|
| `cv::sfm::triangulatePoints` | Eigen DLT (JacobiSVD) | `src/red_math.h` |
| `cv::sfm::projectionFromKRt` | Eigen matrix concat + multiply | `src/red_math.h` |
| `cv::Rodrigues` | `Eigen::AngleAxisd` | `src/red_math.h` |
| `cv::undistortPoints` | Iterative Brown-Conrady (10 iters) | `src/red_math.h` |
| `cv::projectPoints` | Forward projection with distortion | `src/red_math.h` |
| `cv::FileStorage` (read/write) | Custom `!!opencv-matrix` YAML parser/writer | `src/opencv_yaml_io.h` |
| `cv::VideoCapture` | FFmpeg C API + VideoToolbox HW decode | `src/ffmpeg_frame_reader.h` |
| `cv::imwrite` (JPEG) | turbojpeg (`tjCompress2`, SIMD) | `src/jarvis_export.h` |
| `cv::imread` | `stbi_load` / `stbi_info` | `src/decoder.cpp`, `src/project.h` |
| `cv::Mat` (camera params) | Eigen types | `src/camera.h` |

### New files

- **`src/red_math.h`** (~130 lines): Eigen-based camera math — projection,
  Rodrigues, undistortion, triangulation, reprojection.
- **`src/opencv_yaml_io.h`** (~245 lines): Parser and writer for OpenCV's YAML
  format (`%YAML:1.0` header, `!!opencv-matrix` tags).
- **`src/ffmpeg_frame_reader.h`** (~220 lines): FFmpeg C API frame reader with
  VideoToolbox HW decode and sequential decode optimization.

### Dependency impact

Before: `brew deps opencv` → ~65 packages.
After: `eigen` (0 deps, header-only) + `ffmpeg` (~10, already required) +
`glfw` (0) + `jpeg-turbo` (0).

### JARVIS export performance

| Version | Time | Notes |
|---|---|---|
| OpenCV baseline | 6.3s | `cv::VideoCapture` + `cv::imwrite` |
| FFmpeg + stb (initial) | 27.1s | Per-frame seeking, software decode |
| + Sequential decode | 12.1s | Skip seek when frames are in order |
| + VideoToolbox HW | 9.1s | Hardware-accelerated decoding |
| + turbojpeg SIMD | 3.7s | SIMD-accelerated JPEG encoding |

---

## Lessons Learned and Gotchas

### VideoToolbox
- `annexb_to_avcc` conversion must use a `found_next` flag to detect
  end-of-buffer vs start-code break — otherwise last 2 bytes of each NAL are
  truncated, causing `kVTVideoDecoderBadDataErr` (-12909).
- VT can output BGRA directly (not just NV12) — set `kCVPixelFormatType_32BGRA`
  in pixel buffer attributes. VT applies the correct YUV→RGB matrix from stream
  metadata internally.
- Seek requires destroy + recreate the VT session; use
  `VTDecompressionSessionFinishDelayedFrames` for synchronous seek behavior.

### Metal / ImGui
- `CAMetalLayer.contentsScale` must be set to `screen.backingScaleFactor` or
  Retina displays render at wrong scale.
- `ImGui_ImplMetal_NewFrame(rpd)` must precede `ImGui::NewFrame()`.
- `ImTextureID` is `ImU64`; bridge:
  `(ImTextureID)(intptr_t)(__bridge void*)tex`.
- `CVMetalTextureCacheFlush` should be called periodically to avoid memory
  accumulation during long sessions.

### macOS platform
- `sockaddr_in` has an extra `sin_len` field — aggregate initializers that work
  on Linux silently map to wrong fields on macOS.
- `AVInputFormat::read_seek` was removed in FFmpeg 6+ — use
  `pb->seekable & AVIO_SEEKABLE_NORMAL` instead.
- `std::ifstream` silently "opens" a directory on macOS but reads zero bytes —
  must detect and resolve `.redproj` files when given a directory argument.
- When launched via PATH (`argv[0]` = `"red"`), use `_NSGetExecutablePath` to
  find the real executable location for resource loading.
- Font search must check both `exe_dir/../fonts` (dev) and
  `exe_dir/../share/red/fonts` (Homebrew install).

### Homebrew
- `bottle :unneeded` was removed in Homebrew 5.0; use
  `uses_from_macos "xcode" => :build`.
- CMakeLists.txt must not hardcode `/opt/homebrew` — use a `HOMEBREW_PREFIX`
  variable auto-detected via `brew --prefix` to support both Apple Silicon and
  Intel Macs.
- `brew install opencv` can silently upgrade FFmpeg between cmake configure and
  build, causing version mismatch. Always reconfigure from a clean build dir.

### Negative PTS in MP4 files
- Some H.264 MP4s have pre-roll frames with negative PTS. RED's frame counter
  includes these (frame 0 = first packet), but OpenCV skips them (frame 0 =
  PTS 0). The offset must be auto-detected and corrected during data export.

---

## File Summary

New files added during the port (macOS only):

| File | Purpose |
|---|---|
| `src/metal_context.h/mm` | Native Metal rendering context |
| `src/vt_async_decoder.h/mm` | Async VideoToolbox decoder |
| `src/red_math.h` | Eigen-based camera math (replaces OpenCV sfm/calib3d) |
| `src/opencv_yaml_io.h` | OpenCV YAML format reader/writer (no OpenCV dep) |
| `src/ffmpeg_frame_reader.h` | FFmpeg C API frame reader |
| `src/jarvis_export.h` | Built-in JARVIS/COCO export tool |
| `packaging/homebrew/red.rb` | Homebrew formula |
| `.github/workflows/release.yml` | GitHub Actions release workflow |
