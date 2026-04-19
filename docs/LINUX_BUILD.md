# Building RED on Ubuntu (quan_dev_linux branch)

This branch was developed primarily on macOS (as `rob_ui_overhaul`). Linux parity was added in April 2026, pulling in the OpenGL MuJoCo renderer from `rob_windows`. This doc lists every dependency, the CMake flags that matter, and which features are still Apple-only.

---

## Fresh-machine build (start here)

Run these in order on a clean Ubuntu 22.04 box with an NVIDIA GPU. Each step is standalone. Skip a step only if you're sure it's already done.

### 1. Clone the repo

```bash
git clone --recursive https://github.com/JohnsonLabJanelia/red.git
cd red
git checkout quan_dev_linux
git submodule update --init --recursive
```

### 2. Install apt dependencies

```bash
sudo apt install -y \
    build-essential cmake pkg-config git curl \
    libeigen3-dev libturbojpeg0-dev \
    libcholmod3 libccolamd2 libcamd2 libcolamd2 libamd2 libspqr2 libsuitesparseconfig5 \
    liblapack-dev libblas-dev \
    libglew-dev libglfw3-dev libglu1-mesa-dev
```

### 3. Install NVIDIA stack (if not already)

- **NVIDIA driver** ≥ 525 (`nvidia-smi` should print your GPU)
- **CUDA Toolkit 12.0 – 12.6** at `/usr/local/cuda/`
- **cuDNN 8.9.3** copied into `/usr/local/cuda/{include,lib64}/`

Steps for CUDA / cuDNN / TensorRT are in the top-level `README.md`. For this branch, **TensorRT 8.6.1.6 must live at `~/nvidia/TensorRT-8.6.1.6/`** and **FFmpeg must be built at `~/nvidia/ffmpeg/build/`** — the CMake has these paths hardcoded. If you already have them at other locations, symlink them:

```bash
mkdir -p ~/nvidia
ln -sf /path/to/your/TensorRT-8.6.1.6 ~/nvidia/TensorRT-8.6.1.6
ln -sf /path/to/your/ffmpeg-build     ~/nvidia/ffmpeg/build
```

### 4. Install OpenCV 4.x with SFM + CUDA

Follow the top-level `README.md` OpenCV section. Do **not** install `libopencv-dev` from apt — it doesn't include the SFM module.

### 5. Build and install Ceres Solver from source

The apt `libceres-dev` package doesn't ship the abseil static libs we need. Build from source so you get `libceres.a` + `libabsl_*.a` under `/usr/local/lib/`:

```bash
cd ~/build   # or anywhere
git clone --depth 1 https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF ..
make -j && sudo make install
```

### 6. Drop in LibTorch (CUDA build)

```bash
cd /path/to/red
curl -L -o /tmp/libtorch.zip \
  https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
unzip /tmp/libtorch.zip -d lib/
# results in lib/libtorch/
```

### 7. Install MuJoCo 3.3.6 (for body model features)

```bash
curl -L -o /tmp/mujoco.tgz \
  https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco.tgz -C ~/build/
# results in ~/build/mujoco-3.3.6/
```

### 8. Install ONNX Runtime (for JARVIS + SAM) — pick the CUDA 12 variant

**Important:** the default `onnxruntime-linux-x64-gpu-*.tgz` on GitHub is the **CUDA 11.8** build. It links against `libcublasLt.so.11` which won't exist on a CUDA 12 system — ONNX Runtime will silently fall back to CPU and you'll get this warning in the log:

```
[JARVIS] CUDA EP unavailable, using CPU: ...
libcublasLt.so.11: cannot open shared object file
```

Download the **cuda12** variant instead:

```bash
cd /path/to/red
curl -L -o /tmp/ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz
tar -xzf /tmp/ort.tgz -C lib/
# Note: the CUDA 12 tarball extracts to the same folder name as the CUDA 11 one
mv lib/onnxruntime-linux-x64-gpu-1.18.1 lib/onnxruntime
```

This variant also needs **cuDNN 9**. We get it for free from libtorch — `lib/libtorch/lib/libcudnn.so.9` is shipped with the libtorch-2.5.1+cu121 build from step 6, and the binary's RPATH resolves it at runtime. No separate cuDNN install needed.

You can verify the CUDA EP links correctly after extracting:

```bash
LD_LIBRARY_PATH=lib/libtorch/lib ldd lib/onnxruntime/lib/libonnxruntime_providers_cuda.so | grep -iE 'cuda|cublas|cudnn'
```

Every line should show `=> /some/path` with no `not found`.

### 9. Configure + build

Pass the correct CUDA arch for your GPU. Don't guess — check `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and drop the dot (e.g. `8.6` → `86`).

```bash
cmake -S . -B release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build release -j
```

If MuJoCo isn't at `/home/user/build/mujoco-3.3.6`, add `-DMUJOCO_DIR=/your/path`.

At configure time you should see:

```
-- ONNX Runtime found at .../lib/onnxruntime      # if step 8 done
-- MuJoCo found at .../mujoco-3.3.6               # if step 7 done
```

### 10. Run

```bash
./release/red
```

---

## One-liner (once all deps are in place)

```bash
cmake -S . -B release -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build release -j && ./release/red
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

- **ONNX Runtime 1.18.x GPU (CUDA 12 build)** — enables `RED_HAS_ONNXRUNTIME` (SAM segmentation, JARVIS ONNX path on GPU):
  ```bash
  cd lib
  curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz
  tar -xzf onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz
  # tarball extracts to same folder name regardless of CUDA variant
  mv onnxruntime-linux-x64-gpu-1.18.1 onnxruntime
  ```
  The default `onnxruntime-linux-x64-gpu-*.tgz` (without `-cuda12-`) links against CUDA 11 and will silently fall back to CPU on a CUDA 12 machine.

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
- **`[JARVIS] CUDA EP unavailable, using CPU: libcublasLt.so.11: cannot open shared object file`** → you installed the CUDA 11 build of ONNX Runtime on a CUDA 12 machine. Replace with the `cuda12` variant (`onnxruntime-linux-x64-gpu-cuda12-1.18.1.tgz`, see step 8 of the install guide).
- **`[JARVIS] CUDA EP unavailable, using CPU: libcudnn.so.9: cannot open shared object file`** → cuDNN 9 not reachable. libtorch 2.5.1+cu121 ships it at `lib/libtorch/lib/libcudnn.so.9`, so make sure `run.sh` sets `LD_LIBRARY_PATH` to include that dir, or the binary's RPATH points at it.
- **`CUDA_ERROR_OUT_OF_MEMORY at line 145 in AppDecUtils.h`** → too many decoder threads each creating their own CUDA context. Fixed in `src/AppDecUtils.h` by switching to `cuDevicePrimaryCtxRetain` (shared refcounted context).
- **`cuvidCreateDecoder returned error 2` or `cudaErrorIllegalAddress` when opening a project with many high-res cameras** → display buffer exceeded available VRAM. Fixed in `src/render.cpp` by auto-shrinking `size_of_buffer` to fit (50% of free VRAM). Override with a smaller `default_buffer_size` in `~/.config/red/config.json` if you want finer control.

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
