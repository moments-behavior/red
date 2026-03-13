# macOS Development Summary

**Branch lineage:** `rob_dev` -> `rob_dev_vulkan` -> `rob_dev_metal` -> `rob_dev_no_opencv` -> `rob_dev_calib` -> `rob_calib_no_opencv` -> `rob_ui_overhaul`

**Period:** Feb 24, 2026 - Mar 13, 2026 (18 days)
**Commits:** 211
**Files changed:** 136 (104 new, 26 modified, 6 deleted)
**Net code change:** +47,508 / -11,194 lines

## Codebase Transformation

| Metric | Before (main) | After (rob_ui_overhaul) |
|--------|--------------|------------------------|
| `red.cpp` (main file) | 3,982 lines | 1,402 lines (-65%) |
| Source files (src/) | 87 | 93 |
| Source LOC (src/) | ~87,000 | 34,819 |
| Test files | 0 | 12 |
| Test LOC | 0 | 8,654 |
| Scripts | 0 | 895 LOC |
| GUI files (src/gui/) | 0 | 25 files |

## Changes by Area

| Area | Files | Lines Added | Lines Removed | Description |
|------|-------|-------------|---------------|-------------|
| Calibration | 12 | +9,902 | -0 | Full calibration pipeline from scratch |
| GUI modularization | 31 | +8,921 | -0 | Extracted from red.cpp monolith |
| Documentation | 16 | +8,796 | -143 | README, guides, legacy docs organized |
| Tests & tools | 12 | +4,256 | -0 | 852 tests across 12 test files |
| Annotation system | 6 | +3,851 | -633 | Unified data model + CSV persistence |
| JARVIS inference | 7 | +2,497 | -0 | CoreML + ONNX pose estimation |
| Core app | 7 | +1,898 | -4,292 | Extracted into modules |
| Segmentation (SAM) | 2 | +1,883 | -0 | Segment Anything integration |
| Video/GPU pipeline | 10 | +1,378 | -107 | macOS Metal/VT pipeline, fixes |
| Scripts & packaging | 4 | +1,063 | -0 | Homebrew, CoreML conversion |
| Other | 29 | +3,063 | -6,019 | Cleanup, removed OpenCV, misc |

## Major Features Added

### 1. macOS Native Port
- Metal rendering pipeline replacing OpenGL
- VideoToolbox hardware decoding with async PTS reorder queue
- CVPixelBuffer -> CVMetalTextureCache -> MTLTexture (zero-copy)
- Replaced OpenCV dependency entirely with Eigen + custom code

### 2. GUI Modularization (25 files)
Extracted the 3,982-line `red.cpp` monolith into a modular GUI system:
- **Pattern:** State struct + inline draw function, all take `AppContext &ctx`
- **Panel system:** `DrawPanel()` wrapper with consistent sizing/docking
- **Windows:** labeling tool, calibration tool, JARVIS predict/export/import, SAM tool, export, settings, help, annotation dialog, project, calib viewer
- **Infrastructure:** panel registry, popup stack, toast notifications, transport bar
- **WindowStates:** bundles all 12 tool states with comprehensive `reset()` for project switching
- `red.cpp` reduced to 1,402 lines (render loop + coordination)

### 3. Calibration System (from scratch)
- **ArUco detection** (1,449 LOC): Custom OpenCV-free ChArUco detector with Metal GPU adaptive thresholding
- **Calibration pipeline** (3,286 LOC): Full intrinsic + extrinsic calibration
- **Laser refinement** (1,441 LOC): Sub-pixel laser line detection with Metal GPU acceleration
- **Telecentric DLT** (904 LOC): Ported from MATLAB — linear DLT, K2 constraints, bundle adjustment
- **3D viewer** (446 LOC): ImPlot3D visualization of calibration results
- Achieved 0.447 px reprojection error on images, 0.586 px on video

### 4. JARVIS AI Pose Estimation
- **CoreML inference** (391 LOC): Native Apple Silicon GPU/ANE acceleration (~6-20ms/frame)
- **ONNX Runtime inference** (382 LOC): Cross-platform fallback
- **Model config** (58 LOC): Unified config parsed from model_info.json
- **Predict UI** (753 LOC): Model selector, Convert to CoreML button, auto-import
- **Export** (1,187 LOC): Built-in JARVIS/COCO format export
- **Import** (274 LOC): Prediction import from external tools
- ImageNet normalization in C++ for both mouse (24 joints) and fly (50 joints)
- Convert to CoreML button: .pth -> .mlpackage with auto-import into project

### 5. Segment Anything (SAM) Integration
- **SAM inference** (803 LOC): MobileSAM encoder/decoder via ONNX Runtime
- **SAM tool** (386 LOC): Point prompt UI, multi-mask selection, polygon extraction

### 6. Annotation System Rewrite
- **Unified data model** (336 LOC): `AnnotationMap` replacing scattered keypoint arrays
- **CSV persistence** (725 LOC): `#red_csv v2` format with per-camera files
- **Export formats** (1,187 LOC): JARVIS, COCO, DLC, YOLO Pose, YOLO Detect
- Per-keypoint confidence and source tracking (Manual, Predicted, Triangulated)

### 7. Project Management
- **Project system** (253 LOC): JSON project files (.redproj)
- **App context** (346 LOC): Centralized state bundle with close/open lifecycle
- **Media loader** (305 LOC): Extracted video loading with graceful error handling
- **Project switching:** Full teardown + reload without app restart
- **User settings** (107 LOC): Persistent preferences across sessions

### 8. Video Pipeline Hardening
- Async VideoToolbox decoder with PTS reorder queue
- FindKeyFrameInterval fix (seek back after probing)
- Framerate detection fix (avg_frame_rate vs r_frame_rate for H.264)
- Graceful handling of broken video files (missing co64 atom)
- select_corr_head recomputation after mid-frame seek

### 9. Skeleton System
- Fly50 preset (50 keypoints, 44 edges) for fruit fly behavioral tracking
- Skeleton presets: Rat16, Mouse22, Fly50 (expandable)
- Per-keypoint memory across frames

### 10. Scripts & Packaging
- **pth_to_coreml.py** (395 LOC): PyTorch .pth -> CoreML .mlpackage conversion
- **convert_onnx_to_coreml.py** (500 LOC): ONNX -> CoreML conversion
- **Homebrew formula:** `brew install --HEAD JohnsonLabJanelia/red/red`

## Key Architectural Decisions

1. **No OpenCV on macOS:** Replaced with Eigen (math), custom ArUco detector, custom YAML parser. Eliminates 200MB+ dependency.
2. **Metal over OpenGL:** Native GPU pipeline, CVMetalTextureCache for zero-copy video display.
3. **CoreML over ONNX RT:** 3-5x faster inference on Apple Silicon via GPU/ANE.
4. **Header-only GUI modules:** Each window is a self-contained .h file with state struct + draw function. No .cpp files needed.
5. **Async everything:** Calibration, laser detection, model conversion all run on background threads with future/atomic polling.

## Test Coverage

| Test File | Tests | Description |
|-----------|-------|-------------|
| test_annotation.cpp | 673 | Annotation model, CSV, export formats, migration |
| test_gui.cpp | 179 | Panel registry, toast, keypoint table, skeleton |
| test_sam_integration.cpp | - | SAM mask-to-polygon conversion |
| test_jarvis_golden.cpp | - | JARVIS prediction golden tests |
| test_jarvis_import.cpp | - | Prediction import parsing |
| test_video_pipeline.cpp | - | Demuxer, framerate, seek |
| test_pipeline_run.cpp | - | End-to-end calibration pipeline |
| test_calib_crossval.cpp | - | Calibration cross-validation |
| test_calib_debug.cpp | - | Calibration debugging |
| test_detect_timing.cpp | - | ArUco detection performance |
| test_detect_granular.cpp | - | ArUco detection accuracy |
| test_contour_drill.cpp | - | Contour detection |
| bench_jarvis.cpp | - | JARVIS inference benchmarks |
| bench_coreml_accuracy.mm | - | CoreML accuracy benchmarks |

**Total: 852+ tests, all passing**

## Largest New Files

| LOC | File | Description |
|-----|------|-------------|
| 3,286 | calibration_pipeline.h | Full calibration algorithm |
| 1,449 | aruco_detect.h | OpenCV-free ChArUco detector |
| 1,441 | laser_calibration.h | Laser line refinement |
| 904 | telecentric_dlt.h | Telecentric DLT calibration |
| 803 | sam_inference.h | Segment Anything integration |
| 753 | jarvis_predict_window.h | JARVIS model selector UI |
| 725 | annotation_csv.h | CSV persistence layer |
| 603 | laser_metal.mm | Metal GPU laser detection |
| 596 | aruco_metal.mm | Metal GPU adaptive threshold |
| 568 | labeling_tool_window.h | Labeling tool UI |

## Development Velocity

| Date | Commits | Notable |
|------|---------|---------|
| Feb 24-25 | 9 | macOS port, initial docs |
| Feb 26 | 16 | Metal pipeline, VideoToolbox |
| Feb 28 - Mar 2 | 18 | OpenCV removal, calibration start |
| Mar 3-6 | 12 | Calibration pipeline, laser refinement |
| Mar 7-8 | 33 | GUI modularization, annotation rewrite |
| Mar 9-10 | 45 | Project system, SAM, export formats |
| Mar 11 | 26 | Telecentric DLT, project switching |
| Mar 12 | 42 | JARVIS CoreML, video sync fixes, audit |
| Mar 13 | 10 | Convert button, audit fixes |
