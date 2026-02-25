# RED – macOS Apple Silicon Port

This document describes the work required to port RED from Linux/NVIDIA to macOS Apple Silicon (M-series).

---

## Strategy

The port uses `#ifdef __APPLE__` guards throughout the codebase so that the Linux/NVIDIA path is completely untouched. On macOS, four subsystems are replaced:

| Linux/NVIDIA | macOS replacement |
|---|---|
| NVDEC hardware decode | FFmpeg + VideoToolbox hwaccel |
| CUDA NV12→RGB kernels | FFmpeg `libswscale` (`sws_scale`) |
| `cudaMalloc` frame buffers | `malloc` (CPU buffers) |
| CUDA-GL interop / PBO upload | `glTexSubImage2D` from CPU buffer |
| TensorRT / LibTorch YOLO | Stub functions (YOLO not needed on Mac) |

---

## macOS Dependencies

Install with Homebrew:

```bash
brew install ffmpeg opencv glew glfw pkg-config
```

That's it. No CUDA, no TensorRT, no LibTorch needed on macOS.

---

## Build

```bash
./build_mac.sh
```

Then run from the repo root (so the `fonts/` directory is in the working path):

```bash
./release/red
```

---

## Files Changed

### `CMakeLists.txt`
Split into `if(APPLE) … else() … endif()` blocks.

- **macOS block**: `project(red LANGUAGES CXX)` (no CUDA language). Uses `pkg_check_modules` to find FFmpeg, `find_package` for OpenCV / GLEW / glfw3. Links `-framework VideoToolbox`, `-framework CoreMedia`, `-framework CoreFoundation`, and `-framework OpenGL`. Explicitly links `opencv_sfm` (needed for `cv::sfm::triangulatePoints`). Excludes `NvDecoder.cpp`, `yolo_torch.cpp`, `yolo_export.cpp` from the build. Adds `target_link_directories(${FFMPEG_LIBRARY_DIRS})` so the linker can find the FFmpeg dylibs.
- **Linux block**: original content preserved exactly.

### `build_mac.sh` *(new file)*
Convenience script:
```bash
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/homebrew"
cmake --build release -j $(sysctl -n hw.logicalcpu)
```

### `src/decoder.h` / `src/decoder.cpp`
The biggest change. The entire Linux `decoder_process()` function is wrapped in `#ifndef __APPLE__`. A new macOS implementation is added under `#else`:

1. `avcodec_find_decoder()` to get the codec for whatever format is in the file.
2. Attempt VideoToolbox hardware acceleration via `av_hwdevice_ctx_create(..., AV_HWDEVICE_TYPE_VIDEOTOOLBOX, ...)`. Falls back silently to software decode if unavailable.
3. A `mac_get_hw_format` callback that prefers `AV_PIX_FMT_VIDEOTOOLBOX` and falls back to the first available software format.
4. Decode loop: `demuxer->Demux()` → copy into `AVPacket` → `avcodec_send_packet` / `avcodec_receive_frame`.
5. If the decoded frame is a VideoToolbox surface, `av_hwframe_transfer_data()` moves it to CPU as NV12/YUV420P.
6. Lazy-init `SwsContext` on first frame; `sws_scale()` converts directly into the RGBA CPU display buffer.
7. Seeking: `FindClosestKeyFrameFNI` + `demuxer->Seek()` + `avcodec_flush_buffers()`.

Also guarded: the `decoder_get_image_from_gpu()` function (CUDA-only), and the `AppDecUtils.h` include.

### `src/render.h` / `src/render.cpp`
- `PBO_CUDA` struct: `cudaGraphicsResource_t` and `cuda_pbo_storage_buffer_size` fields guarded with `#ifndef __APPLE__`.
- `render_allocate_scene_memory()`: on macOS, always uses `malloc()` for frame buffers; skips all `create_pbo` / `register_pbo_to_cuda` / `map_cuda_resource` calls.

### `src/red.cpp`
- `#include "kernel.cuh"` guarded with `#ifndef __APPLE__`.
- Texture upload block replaced with `#ifdef __APPLE__` / `#else` / `#endif`: macOS uses `glTexSubImage2D()` directly from the CPU frame buffer; Linux uses the existing CUDA PBO path.
- Auto-YOLO labeling block (`~130 lines`) guarded with `#ifndef __APPLE__`.
- "Run YOLO Prediction" button block (`~175 lines`) guarded with `#ifndef __APPLE__`.

### `src/gx_helper.h`
- `GL_SILENCE_DEPRECATION` defined on macOS (suppresses Apple OpenGL deprecation warnings).
- OpenGL 3.3 **core profile** window hints added on macOS (`GLFW_OPENGL_CORE_PROFILE`, `GLFW_OPENGL_FORWARD_COMPAT`). Required for OpenGL 3.x on macOS.
- GLSL version string set to `#version 150` on macOS (core profile minimum), `#version 130` on Linux.
- `glewExperimental = GL_TRUE` set before `glewInit()` on macOS (required for GLEW to work with core profile).
- `register_pbo_to_cuda`, `map_cuda_resource`, `cuda_pointer_from_resource`, `unmap_cuda_resource` all guarded with `#ifndef __APPLE__`.
- `#include <cuda_gl_interop.h>` guarded with `#ifndef __APPLE__`.

### `src/global.h` / `src/global.cpp`
- `#include "yolo_torch.h"` guarded (not needed by other consumers on macOS).
- `yolo_bboxes` and `yolo_predictions` extern declarations and definitions guarded with `#ifndef __APPLE__`.

### `src/yolo_torch.h`
- `#include <torch/script.h>` and `#include <torch/torch.h>` guarded.
- `YoloPrediction` and `YoloBBox` structs left always-visible (pure C++, no CUDA).
- Inline macOS stubs added for `calculateIoU`, `applyNMS`, `runYoloInference`, `frameHasYoloDetections` (all return empty/false).

### `src/yolo_export.h`
- `#include "NvDecoder.h"` and `#include <cuda.h>` guarded.
- Inline macOS stubs for `export_yolo_detection_dataset`, `export_yolo_pose_dataset`, `export_yolo_obb_dataset`.

### `src/AppDecUtils.h`
Entire file content wrapped in `#ifndef __APPLE__` (contains only CUDA context creation utilities unused on macOS).

### `src/ColorSpace.h`
- `#include <cuda_runtime.h>` guarded.
- `uchar4` and `ushort4` CUDA vector type stubs added for macOS (used in BGRA32/RGBA32 unions).

### `src/create_image_cuda.h`
Entire content wrapped in `#ifndef __APPLE__`.

### `src/kernel.cuh`
All CUDA includes and function declarations wrapped in `#ifndef __APPLE__`.

### `src/FFmpegDemuxer.h`
`#include "cuviddec.h"` and `#include "nvcuvid.h"` guarded. `FFmpeg2NvCodecId()` inline function guarded.

### `src/FFmpegDemuxer.cpp`
`AVInputFormat::read_seek` and `read_seek2` were removed from the public API in FFmpeg 6+. Guarded with `#ifdef __APPLE__` to use `fmtc->pb->seekable & AVIO_SEEKABLE_NORMAL` instead.

### `src/Logger.h`
`sockaddr_in` aggregate initializer `{AF_INET, htons(uPort), addr}` assumes Linux struct layout. macOS `sockaddr_in` has an extra `sin_len` field at offset 0, making the aggregate map to wrong fields. Fixed by replacing with explicit field assignment:
```cpp
struct sockaddr_in s = {};
s.sin_family = AF_INET;
s.sin_port   = htons(uPort);
s.sin_addr   = addr;
```

---

## Build Errors Encountered and Fixed

| Error | Cause | Fix |
|---|---|---|
| `Could NOT find PkgConfig` | `pkg-config` not installed | `brew install pkg-config` |
| `Could not find OpenCV` | OpenCV not installed | `brew install opencv` (also installed `glew` at the same time) |
| `library 'avformat' not found` (linker) | CMake had `FFMPEG_LIBRARIES` but not the `-L` search path | Added `target_link_directories(red PRIVATE ${FFMPEG_LIBRARY_DIRS})` |
| `undefined symbol cv::sfm::triangulatePoints` | `opencv_sfm` not in link list | Added `opencv_sfm` to `target_link_libraries` |
| `non-constant-expression cannot be narrowed … sa_family_t` | `Logger.h` sockaddr_in aggregate init, macOS struct layout difference | Replaced with explicit field assignment |
| `no member named 'read_seek' in AVInputFormat` | Fields removed in FFmpeg 6+ | Use `pb->seekable` on macOS |
| FFmpeg version mismatch at runtime (`libavformat.61` vs `62`) | `brew install opencv` silently upgraded ffmpeg 7→8 between cmake configure and build | Deleted build dir, reconfigured clean |
