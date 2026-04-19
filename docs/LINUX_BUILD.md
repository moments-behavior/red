# Building RED on Ubuntu (rob_ui_overhaul branch)

This branch was developed primarily on macOS. The Linux build was brought to parity in April 2026, pulling in the OpenGL MuJoCo renderer from `rob_windows`. This doc lists every dependency, the CMake flags that matter, and which features are still Apple-only.

## Quick build (assuming deps are installed)

```bash
cmake -S . -B release -DCMAKE_CUDA_ARCHITECTURES=86   # 86=RTX30xx, 89=RTX40xx, 80=A100
cmake --build release -j
./release/red
```

## Dependencies

### System packages (apt)

```bash
sudo apt install \
    libeigen3-dev \
    libturbojpeg0-dev \
    libcholmod3 libccolamd2 libcamd2 libcolamd2 libamd2 libspqr2 libsuitesparseconfig5 \
    liblapack-dev libblas-dev \
    libglew-dev libglfw3-dev libglu1-mesa-dev
```

### Built from source (one-time setup under `/home/user/build/` or equivalent)

| Dep | Version | Install location |
|---|---|---|
| CUDA Toolkit | 12.0–12.6 | `/usr/local/cuda/` |
| cuDNN | 8.9.3+ | `/usr/local/cuda/{include,lib64}/` |
| OpenCV | 4.8.0–4.10.0 with contrib + SFM | `/usr/local/` |
| Ceres Solver | 2.x (static .a) | `/usr/local/lib/libceres.a` + `/usr/local/lib/libabsl_*.a` |
| FFmpeg | 4.4+ (NVIDIA patched) | `~/nvidia/ffmpeg/build/` |
| TensorRT | 8.6.1.6 | `~/nvidia/TensorRT-8.6.1.6/` |
| LibTorch | **2.5.1+cu121** (CUDA build — not CPU-only) | `lib/libtorch/` |
| MuJoCo | 3.3.6 | `/home/user/build/mujoco-3.3.6/` (override with `-DMUJOCO_DIR=...`) |

### Optional (enables extra features)

- **ONNX Runtime 1.18.x GPU** — enables `RED_HAS_ONNXRUNTIME` (SAM segmentation, JARVIS ONNX path):
  ```bash
  cd lib
  curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
  tar -xzf onnxruntime-linux-x64-gpu-1.18.1.tgz
  mv onnxruntime-linux-x64-gpu-1.18.1 onnxruntime
  ```

CMake auto-detects both ONNX Runtime at `lib/onnxruntime/` and MuJoCo at the configured `MUJOCO_DIR`. Missing deps just disable the corresponding features with a `-- ... disabled` message at configure time.

## CMake options that matter

| Flag | Default | Why |
|---|---|---|
| `-DCMAKE_CUDA_ARCHITECTURES=86` | `80;86;89` fat binary | Override to your GPU for faster builds. 80=A100, 86=RTX 30xx, 89=RTX 40xx, 90=H100. |
| `-DMUJOCO_DIR=/path/to/mujoco` | `/home/user/build/mujoco-3.3.6` | If MuJoCo is elsewhere. Supports both `include/mujoco/mujoco.h` (standard) and `include/mujoco.h` (flat) layouts. |

## Features: what works on Linux

| Feature | Status |
|---|---|
| Video decode (NVIDIA NvDecoder) | Working |
| ArUco detection (CPU path) | Working |
| Intrinsic / multi-cam calibration (Ceres BA) | Working |
| JARVIS 2D keypoint detection (LibTorch or TensorRT) | Working |
| JARVIS ONNX Runtime inference | Working (requires `lib/onnxruntime/`) |
| SAM segmentation | Working (requires `lib/onnxruntime/`) |
| CoTracker3 propagation | Integrated but buggy — see `JARVIS_COTRACKER_INTEGRATION.md` |
| MuJoCo body model load / IK / STAC | Working (requires `MUJOCO_DIR`) |
| MuJoCo body resize | Working |
| MuJoCo 3D rendering (OpenGL offscreen) | Working (via `mujoco_opengl_renderer.cpp` from rob_windows) |
| pre-computed qpos playback | Working |

## Features that are Apple-only (stubbed on Linux)

These features are guarded by `#ifdef __APPLE__` and will silently disable on Linux. Porting each is possible but out of scope so far:

| Feature | Apple impl | Porting effort |
|---|---|---|
| JARVIS CoreML inference (fast path) | `jarvis_coreml.mm` | Not needed — the ONNX Runtime + LibTorch paths work on Linux. |
| SuperPoint calibration refinement | `superpoint_coreml.mm` + `vt_async_decoder.mm` | ~500 LoC — needs ONNX SuperPoint + NvDecoder plumbing. |
| Metal point-source (green LED) detection | `pointsource_metal.mm` | ~50 LoC — replace with OpenCV CPU blob detection. |
| ArUco Metal GPU threshold | `aruco_metal.mm` | Not needed — `aruco_detect.h` has a CPU fallback. |
| VideoToolbox async decoder | `vt_async_decoder.mm` | Not needed — `NvDecoder.cpp` is already used on Linux. |
| Learned-IK CoreML (ANE bypass) | `learned_ik_coreml.mm` | Not needed — standard MuJoCo iterative IK works. |

## Common build errors on a new Ubuntu machine

- **`Eigen/Core: No such file or directory`** → `apt install libeigen3-dev`.
- **`implot3d.h: No such file`** → submodule wasn't initialized; `git submodule update --init --recursive`.
- **`GLEW error: ‘PFNGLMINMAXPROC’ does not name a type`** → `<GL/glew.h>` is being included after another GL header. `src/gx_helper.h` now includes it first on non-Apple; don't reorder.
- **CMake error about `GTest::gmock`** when finding Ceres → Ceres is bypassed via direct linkage (`find_library(CERES_LIBRARY ceres)`) in the Linux CMake block. Do not replace with `find_package(Ceres)` unless you also `apt install libgmock-dev`.
- **Linker: `undefined reference to absl::lts_...`** → abseil static libs not found. CMake globs `/usr/local/lib/libabsl_*.a`; verify those exist (they come with the Ceres source build).
- **Linker: `undefined reference to cholmod_*`** → SuiteSparse libs not installed; apt packages listed above.
- **Segfault in `gx_context::operator=` at startup** → malloc+designated-initializer with a `std::string` member. Fixed by using `new gx_context{}`.
- **CUDA arch mismatch at runtime** → pass `-DCMAKE_CUDA_ARCHITECTURES=<your_compute>` to `cmake`.

## What changed in this branch (vs main)

The Linux CMake block (`CMakeLists.txt` after the `else()` around line 1320) was expanded to:

1. Multi-arch CUDA default (`80;86;89`) that's overridable.
2. `find_package(Eigen3)` + manual Ceres linkage (bypasses the broken CeresConfig).
3. `pkg-config libturbojpeg`.
4. `implot3d` sources + include path.
5. Optional auto-detection for ONNX Runtime (`lib/onnxruntime/`) and MuJoCo (`MUJOCO_DIR`) with `RED_HAS_ONNXRUNTIME` / `RED_HAS_MUJOCO` defines.
6. `-Wl,--start-group` around Ceres + abseil to resolve circular refs.
7. `RPATH` entries so libtorch / onnxruntime / mujoco resolve at runtime.

Source-level fixes:

- `src/gx_helper.h` — GLEW included before GLFW; `<filesystem>` header added.
- `src/red.cpp` — `gx_context` allocated with `new{}` (C++ construction) instead of malloc+designated-initializer (UB). Apple-only `pixel_buffer` and `jarvis_coreml_state` usages gated by `#ifdef __APPLE__`.
- `src/gui/body_model_window.h` — video bg tex uses `image_texture` (GLuint) on Linux, `image_descriptor` (ImTextureID) on Apple.
- `src/mujoco_opengl_renderer.cpp/.h` — cherry-picked from `rob_windows` (OpenGL port of the Metal renderer, ~1100 LoC).
- `src/learned_ik_coreml.h` — now provides a Linux/Windows stub struct so `body_model_window.h` compiles everywhere.
- `src/ffmpeg_frame_reader.h` — `frameCount()` helper added (from rob_windows) for Linux/Windows code paths.
