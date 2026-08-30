# red labeling 📍

A 3D multi-camera labeling tool for fast review and triangulation across many synchronized video streams.

![gui](images/gui.png)

## Overview

`red` is the labeling counterpart to [orange](https://github.com/moments-behavior/orange). It takes multi-view video (typically recorded with `orange`) and lets you label keypoints across all camera views simultaneously, with real-time hardware-accelerated decoding of h264 / hevc (NVDEC + CUDA on Linux and Windows, VideoToolbox + Metal on macOS), synchronized playback across all cameras, and multi-view triangulation. Labeled data can be exported for downstream training (YOLO detection, YOLO pose, JARVIS).

## Documentation

Full documentation — installation, configuration, data export — lives at the [moments-behavior docs site](https://moments-behavior.github.io/docs/red/). The site currently documents the Linux build; see Build below for macOS and Windows.

[Video demo](https://www.youtube.com/watch?v=9eOJaadE1Nc)

## Dependencies

| | macOS | Linux | Windows |
|---|---|---|---|
| Build | CMake, pkg-config | CMake, pkg-config | CMake, Visual Studio 2022 (C++ toolset) |
| Video | FFmpeg (avcodec, avformat, avutil, swscale) | ← same | ← same (not from vcpkg) |
| Graphics | GLFW | GLFW, GLEW, OpenGL | GLFW, GLEW |
| Math | Eigen3, Ceres Solver | Eigen3, Ceres Solver, CBLAS/OpenBLAS | Eigen3, Ceres Solver |
| Images | libjpeg-turbo | — | libjpeg-turbo |
| GPU | — (VideoToolbox + Metal are part of macOS) | CUDA Toolkit with NVDEC | CUDA Toolkit 12.x with NVDEC |
| *Optional* | Apache Arrow + Parquet | ← same | ← same |

red is built as C++20 throughout, so every build needs a compiler that
supports it — GCC 10+, Clang 10+, or Visual Studio 2022. (The standard moved
up for Arrow, whose headers use `std::span`, but it applies whether or not
Arrow is installed.)

Apache Arrow is optional and only enables the tailcycle-dataset export format
(see [below](#optional-tailcycle-dataset-export)). Without it red builds and
runs normally; that one entry simply does not appear in the export window.

Homebrew, apt and vcpkg are the paths below, but nothing requires them — any
install CMake can find via `find_package` / `pkg-config` works.

## Build

```bash
git clone --recursive https://github.com/moments-behavior/red.git
cd red
```

**macOS** (Apple Silicon)

```bash
brew install cmake pkg-config ffmpeg glfw eigen ceres-solver jpeg-turbo
./build.sh          # builds release/red
./release/red
```

**Linux** (NVIDIA GPU with NVDEC, GTX 1060+)

```bash
sudo apt install cmake pkg-config libglfw3-dev libglew-dev libeigen3-dev \
    libceres-dev libopenblas-dev libgtest-dev nvidia-cuda-toolkit \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
./build.sh
./release/red
```

**Windows** (NVIDIA GPU with NVDEC)

```powershell
vcpkg install glfw3 glew eigen3 ceres libjpeg-turbo --triplet x64-windows
.\build.bat                              # builds release\red.exe
$env:PATH = "C:\ffmpeg\bin;$env:PATH"
.\release\red.exe
```

FFmpeg does not come from vcpkg — that port compiles FFmpeg from source under
MSVC and frequently fails. Use any win64 *shared* build (one with `include\`
and `lib\`, not just `ffmpeg.exe`), unpacked to `C:\ffmpeg` or pointed at by
`FFMPEG_ROOT`; its `bin\` must be on `PATH` at runtime, as above. Last tested
2026-08 with the `win64-lgpl-shared` build from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases).

`build.bat` locates Visual Studio, CUDA, vcpkg and FFmpeg itself. Set
`VCPKG_ROOT` if vcpkg is not on `PATH` or at `$env:USERPROFILE\vcpkg`.

**Windows toolchain versions:**

- **MSVC 14.4x** (VS 2022 17.14 or newer). Older toolsets fail to build Arrow.
- **CUDA** new enough for that MSVC — `nvcc` rejects newer host compilers, and
  CMake reports it as *no CUDA compiler found*.
- **CMake** comes from Visual Studio; `build.bat` prefers it over any
  standalone install.

Updating Visual Studio breaks a working CUDA install until the toolkit is
updated too. Check `nvidia-smi` (top right) for your driver's ceiling first —
if it already covers the toolkit you want, install with the bundled display
driver **unticked**.

### Building for a machine without a GPU

red decodes and renders on the GPU by default. On a machine with no NVIDIA GPU,
build the CPU version instead:

```bash
./build.sh -DRED_ENABLE_CUDA=OFF
./release/red
```

It does the same things, just slower — expect a large frame-rate drop with many
cameras.

Only playback is affected. Every export format decodes with FFmpeg in software
on Linux and Windows regardless of this flag, so a CPU-only build extracts
frames at full capability.

On Linux this has to be a separate build: a normal build needs the NVIDIA
driver's libraries just to start up, so without them red will not launch at
all. Windows builds run on either kind of machine.

### Choosing the decoder

A GPU build uses the GPU, and switches to the CPU decoder by itself if the GPU
turns out to be unusable. To choose explicitly — mainly to exercise the CPU
path on a machine that does have a GPU:

```bash
RED_DECODE_BACKEND=sw ./release/red      # CPU decoder
RED_DECODE_BACKEND=hw ./release/red      # GPU decoder
RED_SW_DECODE_THREADS=2 ./release/red    # CPU decode threads per camera
```

In PowerShell the variable is set separately, and stays set until you clear it:

```powershell
$env:RED_DECODE_BACKEND = "sw"
.\release\red.exe
Remove-Item Env:RED_DECODE_BACKEND       # back to the default
```

red prints which backend it chose at startup.

## Authors

**Red** is developed by Jinyao Yan, with contributions from Wilson Chen, Diptodip Deb, Ratan Othayoth, and Rob Johnson.

Contact [Jinyao Yan](mailto:yanj11@janelia.hhmi.org) with questions about the software.

## Citation

If you use **Red**, please cite the software:

```bibtex
@software{moments_behavior_red_2026,
  author       = {Yan, Jinyao and
                  Deb, Diptodip and
                  Chen, Wilson and
                  Othayoth, Ratan and
                  Johnson, Rob},
  title        = {moments-behavior/red: v1.1.0},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.19688190},
  url          = {https://doi.org/10.5281/zenodo.19688190},
}
```

## Contribute

Please open an issue for bug fixes or feature requests. If you wish to make changes to the source code, fork the repo and open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
