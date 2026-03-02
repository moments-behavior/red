# Removing OpenCV from the macOS Build

## Motivation

OpenCV was RED's heaviest dependency. On macOS via Homebrew, `opencv` pulls in ~65 transitive packages (ceres-solver, boost, vtk, qt, protobuf, etc.), making installs slow and fragile. RED used a small fraction of OpenCV:

- Camera math (5 functions from `cv::sfm` and core)
- YAML I/O for calibration files (`cv::FileStorage`)
- Video frame extraction (`cv::VideoCapture`)
- JPEG writing (`cv::imwrite`)
- Image loading (`cv::imread`)

All were replaceable with lighter alternatives. The Linux/NVIDIA build path is untouched — all changes are behind `#ifdef __APPLE__` guards or in files excluded from the macOS CMake build.

## Replacement Strategy

| OpenCV usage | Replacement | Notes |
|---|---|---|
| `cv::sfm::triangulatePoints` | Eigen DLT (JacobiSVD) | `src/red_math.h` |
| `cv::sfm::projectionFromKRt` | Eigen matrix concat + multiply | `src/red_math.h` |
| `cv::Rodrigues` | `Eigen::AngleAxisd` | `src/red_math.h` |
| `cv::undistortPoints` | Iterative Brown-Conrady (10 iters) | `src/red_math.h` |
| `cv::projectPoints` | Forward projection with distortion | `src/red_math.h` |
| `cv::FileStorage` (read) | Custom `!!opencv-matrix` YAML parser | `src/opencv_yaml_io.h` |
| `cv::FileStorage` (write) | Custom YAML writer | `src/opencv_yaml_io.h` |
| `cv::VideoCapture` | FFmpeg C API + VideoToolbox HW decode | `src/ffmpeg_frame_reader.h` |
| `cv::imwrite` (JPEG) | turbojpeg (`tjCompress2`, SIMD) | `src/jarvis_export.h` |
| `cv::imread` | `stbi_load` / `stbi_info` | `src/decoder.cpp`, `src/project.h` |
| `cv::cvtColor(BGR2RGBA)` | `stbi_load(..., 4)` (direct RGBA) | `src/decoder.cpp` |
| `cv::Mat` (camera params) | Eigen types | `src/camera.h` |

## New Files

### `src/red_math.h` (~130 lines)

Eigen-based replacements for 5 OpenCV camera math functions:

- `projectionFromKRt(K, R, t)` — builds 3x4 projection matrix
- `rotationMatrixToVector(R)` / `rotationVectorToMatrix(rvec)` — Rodrigues conversion via `Eigen::AngleAxisd`
- `undistortPoint(pt, K, dist)` — iterative Brown-Conrady undistortion (10 iterations, handles k1-k3 radial + p1-p2 tangential)
- `triangulatePoints(points2d, projections)` — DLT algorithm: builds 2N x 4 matrix per point, solves with `JacobiSVD`, returns last column of V
- `projectPoints(pts3d, rvec, tvec, K, dist)` — forward projection with full distortion model
- `projectPoint(pt3d, rvec, tvec, K, dist)` — single-point variant
- `projectPointNoDist(pt3d, K, R, t)` — distortion-free projection for FOV checks

### `src/opencv_yaml_io.h` (~245 lines)

Parser and writer for OpenCV's YAML format (`%YAML:1.0` header, `!!opencv-matrix` tags):

- `YamlFile` struct with `scalars` map and `matrices` map
- `read(path)` — parses file, returns `YamlFile`
- `getInt()` / `getDouble()` / `getMatrix()` — typed accessors
- `YamlWriter` class — writes OpenCV-compatible YAML (used by JARVIS export for calibration output)

### `src/ffmpeg_frame_reader.h` (~220 lines)

FFmpeg C API frame reader with three performance optimizations:

1. **VideoToolbox hardware decoding** — automatically enabled on macOS via `AV_HWDEVICE_TYPE_VIDEOTOOLBOX`. HW frames are transferred to CPU with `av_hwframe_transfer_data()`.
2. **Sequential decode optimization** — tracks current decode position. When the next requested frame is ahead (within 300 frames), continues decoding forward without seeking. Only seeks when going backward or jumping far ahead.
3. **Lazy swscale context** — created on first decoded frame so the pixel format matches what the decoder actually produces (NV12 from VideoToolbox vs yuv420p from software decode).

## Modified Files

### `src/camera.h`

- `CameraParams` struct: all `cv::Mat` fields replaced with Eigen types
  - `cv::Mat k` (3x3) -> `Eigen::Matrix3d`
  - `cv::Mat dist_coeffs` (5x1) -> `Eigen::Matrix<double,5,1>`
  - `cv::Mat r` (3x3) -> `Eigen::Matrix3d`
  - `cv::Mat rvec` (3x1) -> `Eigen::Vector3d`
  - `cv::Mat tvec` (3x1) -> `Eigen::Vector3d`
  - `cv::Mat projection_mat` (3x4) -> `Eigen::Matrix<double,3,4>`
- `camera_load_params_from_yaml()` — `cv::FileStorage` replaced with `opencv_yaml::read()`
- `camera_print_parameters()` — `cv::format()` replaced with `Eigen::IOFormat`

### `src/gui.h`

~15 call sites updated across these functions:

- `reprojection()` — `cv::sfm::triangulatePoints` -> `red_math::triangulatePoints`, `cv::undistortPoints` -> `red_math::undistortPoint`, `cv::projectPoints` -> `red_math::projectPoint`
- `is_in_camera_fov()` — uses `red_math::projectPointNoDist()`
- `world_coordinates_projection_points()` — `cv::projectPoints` -> `red_math::projectPoints`
- `gui_arena_projection_points()` — same pattern
- `triangulate_bounding_boxes()` — full Eigen replacement of cv::Mat triangulation + reprojection
- Removed `draw_cv_contours()` (unused dead code)

### `src/jarvis_export.h`

- `cv::FileStorage` -> `opencv_yaml::read()` + `opencv_yaml::YamlWriter`
- `cv::VideoCapture` -> `ffmpeg_reader::FrameReader`
- `cv::imwrite` -> `write_jpeg()` (turbojpeg on macOS, stb fallback on Linux)

### `src/decoder.cpp`

- `cv::imread` + `cv::cvtColor(BGR2RGBA)` -> `stbi_load(..., 4)` on macOS (direct RGBA output)

### `src/project.h`

- `cv::imread` for dimension query -> `stbi_info()` on macOS

### `src/decoder.h`, `src/yolo_export.h`

- `#include <opencv2/opencv.hpp>` wrapped in `#ifndef __APPLE__` guards

### `src/red.cpp`

- Added `STB_IMAGE_WRITE_IMPLEMENTATION` define (single compilation unit for stb)

### `tests/test_gui.cpp`

- Added `STB_IMAGE_IMPLEMENTATION` and `STB_IMAGE_WRITE_IMPLEMENTATION` defines (test binary doesn't include `red.cpp`)

### `CMakeLists.txt` (macOS section only)

- Removed: `find_package(OpenCV)`, `${OpenCV_LIBS}`, `opencv_sfm`
- Added: `find_package(Eigen3 REQUIRED)`, `pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)`, `libswscale` to FFmpeg modules
- Linux section unchanged

### `packaging/homebrew/red.rb`

- Removed: `depends_on "opencv"`
- Added: `depends_on "eigen"`, `depends_on "jpeg-turbo"`

## Performance

JARVIS export benchmark (3 cameras x 12 frames = 36 JPEGs, 180fps H.264 video):

| Version | Time | Notes |
|---|---|---|
| OpenCV baseline | 6.3s | cv::VideoCapture + cv::imwrite |
| Initial FFmpeg (v1) | 27.1s | Per-frame seeking, software decode, stb JPEG |
| + Sequential decode | 12.1s | Skip seek when frames are in order |
| + VideoToolbox HW decode | 9.1s | Hardware-accelerated H.264 decoding |
| + turbojpeg SIMD | 3.7s | SIMD-accelerated JPEG encoding |

Final result is ~40% faster than the OpenCV version.

## Dependency Impact

Before (Homebrew):
```
brew deps opencv  →  ~65 packages (ceres-solver, boost, vtk, qt, protobuf, ...)
```

After (Homebrew):
```
eigen       →  0 transitive deps (header-only)
ffmpeg      →  ~10 transitive deps (already required)
glfw        →  0 transitive deps
jpeg-turbo  →  0 transitive deps
```
