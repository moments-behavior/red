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
| Video | FFmpeg (avcodec, avformat, avutil, swscale) | ← same | ← same |
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

**Windows** (NVIDIA GPU with NVDEC) — dependencies come from
[vcpkg](https://vcpkg.io); `build.bat` finds Visual Studio, CUDA and vcpkg
itself. Set `VCPKG_ROOT` if vcpkg is not at `%USERPROFILE%\vcpkg`.

```bat
vcpkg install glfw3 glew eigen3 ceres ffmpeg libjpeg-turbo
build.bat     REM builds build_win\red.exe
```

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
