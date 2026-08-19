# RED — Cross-Platform Branch Refresher for Claude

A bootstrap document for a fresh Claude session working on `red`. Read this first; then
consult the deeper docs linked at the bottom for whichever area you're touching.

---

## Identity

- **User:** Rob Johnson (johnsonr@janelia.hhmi.org), Johnson Lab, HHMI Janelia.
- **Project:** `red` — GPU-accelerated multi-camera 3D keypoint labeling for behavioral
  neuroscience. Repo: `git@github.com:moments-behavior/red.git`.
- **Working tree:** `/home/user/src_dev/red`.
- **Cross-platform branch:** `xp` on GitHub. The canonical cross-platform
  integration branch — macOS (primary, Apple Silicon) + Windows (RTX 3090) +
  Linux (NVIDIA, modern Eigen/Ceres/no-OpenCV build). Mac-only branch lineage
  lives in `rob_ui_overhaul` (merged in).
- **Sibling branch (`rob_windows`):** Some collaborators are still working from
  `rob_windows`. Its Linux port (commits `f28d6f3` + `ebb4029`) was cherry-picked
  onto `xp` so both branches now share the modern Linux build, but the branches
  remain independent. Don't push changes into `rob_windows` and don't merge it
  into `xp` without coordinating with Rob first.
- **Companion app:** `orange` at `/home/user/src/orange/`
  (github.com/moments-behavior/orange). Shares this machine's NVIDIA toolchain.
  **Never touch orange's installs or the system NVIDIA driver/CUDA/TensorRT.**

---

## Non-Negotiable Constraints

1. **No OpenCV — ever** (all three platforms). Use `stb_image`, `Eigen`, `Ceres`,
   and the in-repo math headers instead. If you find yourself reaching
   for `cv::`, stop. The pre-rewrite Linux block that referenced OpenCV/LibTorch
   has been replaced — current `CMakeLists.txt else()` block (Linux, ~line 1643+)
   uses Eigen + Ceres + bundled ORT/cuDNN with RPATH isolation.
2. **Orange toolchain is read-only.** Driver 535.x, CUDA 12.2 at `/usr/local/cuda`,
   custom FFmpeg at `$HOME/nvidia/ffmpeg`, TensorRT at `$HOME/nvidia/TensorRT-8.6.1.6`.
   Red links against these read-only; it never installs into them, never upgrades them.
   Red's own ML libs (ONNX Runtime, cuDNN) live under `lib/` with RPATH
   `$ORIGIN/../lib/...` so they cannot collide.
3. **Don't break Mac or Windows when extending Linux** (and vice versa). The three
   platforms branch via `if(APPLE) / elseif(WIN32) / else()` in `CMakeLists.txt` and
   via `#ifdef __APPLE__ / _WIN32 / __linux__` in source. Keep the guards tight.
4. **No new features beyond the task.** No surrounding refactors, abstractions for
   hypothetical futures, comment narration, or backward-compat shims. Edit the code
   directly.

---

## Architecture in 60 Seconds

**Threading model:** 1 main thread (ImGui render loop, input, triangulation) + N
decoder threads (one per camera), synchronizing via circular `PictureBuffer` + atomics.

**Per-platform GPU pipeline:**

| Stage     | macOS (Apple Silicon)             | Linux (NVIDIA)              | Windows (NVIDIA)                              |
|-----------|-----------------------------------|-----------------------------|-----------------------------------------------|
| Decode    | VideoToolbox (async, PTS reorder) | NVDEC via FFmpeg            | NVDEC via FFmpeg                              |
| Convert   | CVPixelBuffer (BGRA)              | CUDA NV12→RGB kernel        | CUDA NV12→RGB kernel                          |
| Upload    | CVMetalTextureCache (zero-copy)   | CUDA-GL interop PBO         | CUDA-GL interop PBO                           |
| Render    | Metal (ImGui Metal backend)       | OpenGL (ImGui OpenGL3)      | OpenGL (ImGui OpenGL3)                        |
| Inference | CoreML (native ANE/GPU)           | ONNX Runtime + CUDA EP      | ONNX Runtime + CUDA EP (+ optional TensorRT)  |

**Data model centerpiece:** `AnnotationMap` (`std::map<u32, FrameAnnotation>`) in
`src/annotation.h`. `FrameAnnotation` holds per-camera 2D keypoints, triangulated 3D
keypoints, and optional bboxes/OBBs/masks behind `unique_ptr`. Per-keypoint provenance
(`Manual`/`Predicted`/`Triangulated`/`Imported`) is tracked.

**Single `AppContext` bundle** (`src/app_context.h`) is passed by reference everywhere
— skeleton, annotations, project, render scene, decoder threads, popups, toasts, etc.
Each GUI window is a state struct + an inline `Draw(AppContext &ctx)` function.

---

## Where Things Live

```
red/
├── CMakeLists.txt          # 3-arm: APPLE (1–1319) / WIN32 (1320–1640) / else=Linux (1641–1896)
├── src/                    # ~85 files; mixed .cpp / .h / .mm / .cu
│   ├── red.cpp             # entry point, render loop, ~1994 lines
│   ├── app_context.h       # AppContext, DisplayState (the reference bundle)
│   ├── annotation.h        # AnnotationMap, FrameAnnotation (v2 data model)
│   ├── annotation_csv.h    # `#red_csv v2` persistence
│   ├── project.h           # .redproj JSON load/save, ProjectManager
│   ├── skeleton.{h,cpp}    # presets (Rat4..Rat24Target, Fly50, …)
│   ├── camera.h            # CameraParams (Eigen-based on all platforms now)
│   ├── red_math.h          # DLT triangulation + camera math
│   ├── opencv_yaml_io.h    # OpenCV-format YAML reader (no OpenCV dependency)
│   ├── decoder.{h,cpp} + NvDecoder.{h,cpp} + FFmpegDemuxer.{h,cpp}   # NVDEC (Linux/Win)
│   ├── vt_async_decoder.{h,mm}    # VideoToolbox async decoder (macOS only)
│   ├── render.{h,cpp}             # OpenGL render path
│   ├── metal_context.{h,mm}       # Metal context + texture cache (macOS only)
│   ├── jarvis_coreml.{h,mm}       # CoreML inference (macOS only)
│   ├── jarvis_inference.h         # ONNX Runtime inference (cross-platform)
│   ├── jarvis_tensorrt.h          # optional Windows TensorRT
│   ├── imgui_impl_glfw_patched.cpp # patched GLFW backend (mac modifier-key + Win fixes)
│   ├── kernel.cu / ColorSpace.cu / create_image_cuda.cu   # CUDA kernels (Linux/Win)
│   └── gui/                       # ~38 modular window/panel files (state + Draw())
├── lib/                    # bundled deps; submodules + optional ML libs
│   ├── imgui/ implot/ implot3d/ ImGuiFileDialog/ IconFontCppHeaders/   # submodules
│   ├── FFmpeg/ GL/ GLFW/ nvcodec/                                      # Windows headers
│   ├── onnxruntime/        # OPTIONAL (auto-detected) — drop release here for JARVIS
│   └── cudnn/              # OPTIONAL (Linux/Windows) — bundled for ORT CUDA EP
├── tests/                  # ~30 test files; 2 main binaries: test_gui, test_annotation
├── scripts/                # Python helpers: pth_to_coreml, nerfstudio_export, …
├── tools/                  # convert_labels_v1_to_v2 (one-off CSV migration)
├── packaging/              # Homebrew formula and friends
├── dev_docs/               # RedPortToWindows.md (full Windows port log)
├── mac_dev.md              # macOS development summary (rob_ui_overhaul history)
├── legacy_md/              # archived docs
├── ROADMAP.md              # forward-looking research plan
├── README.md               # user-facing docs
├── build.sh / build.bat / build.ps1 / build_cmd.bat
└── claude_refresher.md     # this file
```

---

## Building

### macOS (Apple Silicon)
```bash
brew install eigen ffmpeg glfw jpeg-turbo pkg-config ceres-solver
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build release -j$(sysctl -n hw.ncpu)
```
Binary: `release/red`. Optional `lib/onnxruntime/` auto-detected.

### Linux — two supported targets

The Linux `else()` block in `CMakeLists.txt` builds on both the original
**22.04 / CUDA-12.2** reference box and the newer **24.04 / gcc-13 / CUDA-13.1**
(Blackwell) box. The same source compiles on both — the 24.04 fixes are
backward-compatible and not `#if`-gated (see commit `d9d2a09`).

Common to both:
- CUDA arch defaults to `86;89` (A16 + RTX 4000 Ada). Override with
  `-DCMAKE_CUDA_ARCHITECTURES=...`.
- `find_package(Ceres)` requires `GTest::gmock` (via absl) — apt
  `libgtest-dev` lacks gmock, so CMakeLists explicitly finds GTest at
  `/usr/local/lib/cmake/GTest` first. Build googletest+gmock to `/usr/local`:
  `cmake -S /usr/src/googletest -B /tmp/gt -DCMAKE_BUILD_TYPE=Release && sudo cmake --build /tmp/gt --target install -j`.
- Fresh clone: `git submodule update --init lib/implot3d` (it's a submodule).
- Custom CUDA FFmpeg at `$HOME/nvidia/ffmpeg/build/lib/pkgconfig` (shared with
  orange) is preferred over any system FFmpeg — don't install a system one.
- Optional bundled libs (auto-detected, RPATH-isolated via
  `$ORIGIN/../lib/...`): `lib/onnxruntime/`, `lib/cudnn/`.
- Binaries: `release/red`, `release/test_gui`, `release/test_annotation`.
  Run tests headless: `DISPLAY= ./release/test_annotation` (673) /
  `DISPLAY= ./release/test_gui` (178).

**22.04 / CUDA-12.2 reference box** (driver 535.x):
```bash
cmake -S . -B release && cmake --build release -j
```
Pre-reqs already on machine: Eigen3, Ceres (`/usr/local`), glfw3, GLEW, OpenGL,
`cblas`, CUDAToolkit 12.2 (`/usr/local/cuda`), driver-provided `libnvcuvid.so`.

**24.04 / CUDA-13.1 (Blackwell) box** (driver 590; shares orange's toolchain
read-only). Verified building + passing all tests + GUI playback (June 2026):
```bash
# apt deps (online); 24.04 dropped libcblas.so → install openblas (exports cblas_*)
sudo apt-get install -y libeigen3-dev libceres-dev libopenblas-dev \
    patchelf libgtest-dev libgmock-dev
# nvcc is NOT on PATH → pass it explicitly. 120 = Blackwell display GPU.
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
cmake --build release -j$(nproc)
```
The three 24.04/CUDA-13 build fixes (all backward-compatible, in `d9d2a09`):
1. `src/global.h` includes `<string>` — gcc-13's libstdc++ no longer pulls it
   transitively through `<map>`/`<unordered_map>`.
2. CMake finds a CBLAS provider (`find_library(NAMES cblas openblas blas)`)
   instead of hardcoding `-lcblas` — 24.04 has `libopenblas.so`, not `libcblas.so`.
3. `-Wl,--disable-new-dtags` on red + test targets → DT_RPATH so the shared
   FFmpeg's transitive `libswresample.so.3` resolves at runtime.

The CUDA-13 source hazards (cuCtxCreate `_v4`, NPP `_Ctx`, NVTX) were already
handled in red, so no source porting was needed beyond fix #1. ML inference
(JARVIS) is still compiled out on the 24.04 box — see the runbook in the
`moments_setup` repo (`RED_2404_NOTES.md`) for the full build path and the
Phase-B (TensorRT) inference plan, including running JARVIS on the Blackwell
via TensorRT 10.

### Windows (RTX 3090)
Use `build.bat` / `build_cmd.bat` (vcvars64 + CMake). Toolchain via vcpkg
(`$USERPROFILE/vcpkg`). CUDA 12.6.2 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\`.
Full play-by-play: `dev_docs/RedPortToWindows.md`.

### Testing

The two main test binaries auto-build with `red`:
- `release/test_gui` — ~178 tests covering GUI infrastructure
- `release/test_annotation` — ~673 tests covering the v2 annotation/CSV layer

Plus targeted binaries: `test_jarvis_*`, `test_ort_*`, `test_sync_plan`,
`test_nerfstudio_export`, `test_pump_events`. Run headless on
Linux with `DISPLAY= ./release/<binary>`.

---

## Conventions & Style

- **C++17**, header-only modules where practical. `.h` files often contain
  implementations behind anonymous namespaces — that's intentional, don't break them.
- **GUI:** state struct + inline `Draw(AppContext &ctx)`. Lives in `src/gui/`. New
  windows register through `panel_registry.h` and use the `DrawPanel()` wrapper for
  consistent sizing/docking. Cross-cutting infra: `popup_stack.h`, `toast.h`,
  `deferred_queue.h`.
- **Platform guards:** `if(APPLE)/elseif(WIN32)/else()` in CMake;
  `__APPLE__ / _WIN32 / __linux__` in C++. CUDA `.cu` files are only compiled on
  Linux/Windows.
- **Optional features** are gated by CMake `HAS_*` flags that map to
  `RED_HAS_ONNXRUNTIME`, `USE_TENSORRT`
  compile definitions. Code paths use `#ifdef` guards so the build stays green when
  optional libs are missing.
- **Comments:** minimal. Only when the *why* is non-obvious. No multi-paragraph
  docstrings.
- **Commit messages:** lowercase, action-first, present tense, often with a colon
  prefix for subsystem (`red: …`, `posetail: …`, `jarvis: …`, `render: …`). Look at
  recent `git log --oneline -30` for the house style.

---

## Recent Active Areas (as of May 2026)

Scan `git log --oneline -20` on `xp` for the current frontier. Recent themes:

- **Triangulation Diagnostics window** — new annotation-project tool
  (`src/gui/triangulation_diagnostics_window.h`, `src/annotation_diagnostics.h`).
- **Reprojection diagnostics for telecentric DLT** calibration
  (`src/reprojection_diagnostics.h`).
- **CoreML conversion robustness** — config overrides from `.redproj`, flat `.pth`
  layout support (`scripts/pth_to_coreml.py`).
- **Project lifecycle fixes** — skeleton initialization after `close_project`.
- **MJB / STAC interaction fixes** — auto-reset STAC, clear active_calibration on
  load.
- **Body model IK / STAC / per-segment body resize** (MuJoCo) — large series of
  commits earlier in the branch.
- **Windows MuJoCo support** (OpenGL renderer, build fixes).
- **macOS screen-recording fix** — ImGui input was broken during screen capture.

Themes happening on `rob_windows` (not yet on `xp`): PoseTail integration, CUDA-GL
brightness/contrast on display path. (Linux build modernization + ArUco progress
bars cherry-picked into xp on 2026-05-22.)

Forward research roadmap: `ROADMAP.md` (SuperAnimal bootstrap, Lightning Pose-style
semi-supervised signals, multi-animal tracking, behavior classification, live
closed-loop streaming).

The forward research roadmap lives in `ROADMAP.md` (SuperAnimal bootstrap, Lightning
Pose-style semi-supervised signals, multi-animal tracking, behavior classification,
live closed-loop streaming).

---

## Memory & Auto-Recall

Claude has persistent memory at
`/home/user/.claude/projects/-home-user-src-dev-red/memory/`. The index
(`MEMORY.md`) loads automatically each session and points to:

- `user_role.md` — who Rob is, what red is for, what's non-negotiable.
- `project_linux_build.md` — toolchain paths and the orange-coexistence story.
- `feedback_no_opencv.md` — the "do not re-add OpenCV" guardrail.
- `feedback_orange_isolation.md` — never touch system NVIDIA stack; bundle via RPATH.

Memories are point-in-time and may go stale. Verify file paths and function names
against current code before recommending action on them.

---

## Deeper Docs

| Doc                              | When to read                                                |
|----------------------------------|-------------------------------------------------------------|
| `README.md`                      | User-facing install/usage; full feature inventory.          |
| `mac_dev.md`                     | Why the codebase looks the way it does — Mac port history.  |
| `dev_docs/RedPortToWindows.md`   | Anything Windows: CUDA versions, vcpkg, DLL deps, fallbacks.|
| `ROADMAP.md`                     | Where this is going — research integrations, priorities.    |
| `rat_calib_MVC_vs_REDexp.md`     | Calibration methodology comparison (vs MATLAB ground truth).|
| `legacy_md/`                     | Archive of older design notes; usually not relevant.        |

---

## Fast Sanity Checks Before You Start Coding

```bash
git status                         # confirm you're on xp with a clean tree
git branch --show-current          # should print: xp
git log --oneline -10              # what just changed?
ls release/red 2>/dev/null         # is there a build to iterate on?
cmake --build release -j           # incremental rebuild (Mac/Windows)
```

On the Janelia Linux box, you can also: `DISPLAY= ./release/test_annotation`
(~673 tests, headless-safe). If a build fails with a Ceres/absl/gmock error,
re-read the Ceres note above. If FFmpeg can't be found, check that
`$HOME/nvidia/ffmpeg/build/lib/pkgconfig` exists — do NOT install a system
FFmpeg as a workaround.
