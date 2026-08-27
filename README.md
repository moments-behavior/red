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

```bat
vcpkg install glfw3 glew eigen3 ceres libjpeg-turbo --triplet x64-windows
build.bat     REM builds build_win\red.exe
```

FFmpeg does not come from vcpkg — that port compiles FFmpeg from source under
MSVC and frequently fails. Use any win64 *shared* build (one with `include\`
and `lib\`, not just `ffmpeg.exe`), unpacked to `C:\ffmpeg` or pointed at by
`FFMPEG_ROOT`; its `bin\` must be on `PATH` at runtime. Last tested 2026-08
with the `win64-lgpl-shared` build from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases).

`build.bat` locates Visual Studio, CUDA, vcpkg and FFmpeg itself. Set
`VCPKG_ROOT` if vcpkg is not on `PATH` or at `%USERPROFILE%\vcpkg`.

### Running without a GPU

Decode and render use the GPU by default — NVDEC + CUDA on Linux and Windows,
VideoToolbox + Metal on macOS. There is also an FFmpeg software path, chosen
automatically when no usable GPU is found. Expect a large frame-rate drop with
many cameras.

```bash
RED_DECODE_BACKEND=sw ./release/red     # force software (works on macOS too)
RED_SW_DECODE_THREADS=2 ./release/red   # decode threads per camera
```

A machine with no NVIDIA driver gets a CUDA-free build automatically. That
matters because linking CUDA there produces a binary that cannot start at all:
Linux resolves `libcuda.so.1` and `libnvcuvid.so.1` before `main()` runs, so it
never reaches the code that would have fallen back to software.

Force it when the build machine and the target machine differ — building on a
GPU box to run somewhere without one:

```bash
./build.sh -DRED_ENABLE_CUDA=OFF
```

Windows never needs this; that build delay-loads the CUDA DLLs instead.

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
