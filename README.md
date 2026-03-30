# RED: Real-time, GPU-Accelerated 3D Multi-Camera Keypoint Labeling

A complete pipeline for multi-camera calibration, 2D/3D annotation, and AI-assisted pose estimation in behavioral neuroscience.

Developed at the [Johnson Lab](https://www.janelia.org/lab/johnson-lab), HHMI Janelia Research Campus.

---

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Calibration](#calibration)
- [Annotation Workflow](#annotation-workflow)
- [AI-Assisted Labeling (JARVIS)](#ai-assisted-labeling-jarvis)
- [Architecture Overview](#architecture-overview)
- [Supported Animals and Skeletons](#supported-animals-and-skeletons)
- [Building from Source](#building-from-source)
- [Project File Format](#project-file-format)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Key Features

- **GPU-accelerated real-time multi-camera playback** -- simultaneous decoding and display of 6-16+ synchronized camera views at full frame rate
- **Integrated multi-camera calibration** -- ArUco/ChArUco board detection, laser refinement, and telecentric DLT calibration, all without OpenCV
- **2D keypoint annotation with automatic 3D triangulation** -- label keypoints in any camera view and triangulate 3D positions in real time via Eigen-based DLT
- **AI-assisted labeling via JARVIS pose estimation** -- top-down pose estimation with native CoreML inference on Apple Silicon and ONNX Runtime as a cross-platform fallback
- **SAM (Segment Anything) assisted segmentation** -- MobileSAM integration for interactive segmentation masks via click prompts
- **Bounding box and oriented bounding box annotation** -- axis-aligned and rotated bounding box tools for detection tasks
- **Active learning loop** -- label, export, train, predict, correct: iteratively improve model accuracy with minimal manual effort
- **Native Apple Silicon support** -- Metal GPU rendering, VideoToolbox hardware decode, CoreML inference (6-20 ms/frame)
- **Linux support with NVIDIA NVDEC + CUDA pipeline** -- hardware-accelerated decode and NV12-to-RGB conversion on NVIDIA GPUs with OpenGL rendering
- **Project management** -- switch between projects without restarting the application; per-project layout persistence
- **Preset skeletons** -- built-in skeleton definitions for rats (4-24 keypoints), flies (50 keypoints), and calibration targets, plus custom JSON skeleton import
- **Export to standard formats** -- JARVIS/COCO JSON, YOLO, and DeepLabCut-compatible CSV exports via bundled Python utilities

---

## Quick Start

### Install via Homebrew (macOS)

```bash
brew tap JohnsonLabJanelia/red
brew install --HEAD JohnsonLabJanelia/red/red
```

### Launch

```bash
red /path/to/project.redproj
```

### Create Your First Project

1. **File > New Project** -- choose a project directory, name, skeleton preset, and (optionally) a calibration folder.
2. **File > Load Videos** -- select a folder containing one video file per camera. Camera names are derived from filenames.
3. **Begin labeling** -- click keypoints in any camera view. When two or more cameras have a labeled keypoint, 3D triangulation runs automatically.

---

## Installation

### macOS (Homebrew) -- Recommended

Requires macOS 12 (Monterey) or later on Apple Silicon (M1/M2/M3/M4/M5).

**Step 1: Install Homebrew** (skip if already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Step 2: Add the RED tap**

```bash
# Public repo:
brew tap JohnsonLabJanelia/red

# Private repo (requires GitHub SSH key):
brew tap JohnsonLabJanelia/red git@github.com:JohnsonLabJanelia/homebrew-red.git
```

**Step 3: Install RED**

```bash
brew install --HEAD JohnsonLabJanelia/red/red
```

This installs all dependencies (Eigen, FFmpeg, GLFW, Ceres Solver, etc.) and builds from the latest source. Typical install time: 3-5 minutes.

**Step 4: Launch**

```bash
red                                    # prints usage
red /path/to/my_project.redproj        # open a project
```

On first launch, macOS Gatekeeper may show a security warning. If so:

```bash
xattr -dr com.apple.quarantine "$(brew --prefix)/bin/red"
```

**Updating to the latest version:**

```bash
brew uninstall red
brew install --HEAD JohnsonLabJanelia/red/red
```

### What Gets Installed

| File | Location | Purpose |
|------|----------|---------|
| `red` binary | `/opt/homebrew/bin/red` | Main application |
| Font files (4) | `/opt/homebrew/share/red/fonts/` | UI fonts (Roboto, FontAwesome) |
| `default_imgui_layout.ini` | `/opt/homebrew/share/red/` | Default window layout |
| `pth_to_coreml.py` | `/opt/homebrew/share/red/scripts/` | PyTorch to CoreML model converter |

### macOS (Build from Source)

Install dependencies:

```bash
brew install eigen ffmpeg glfw jpeg-turbo pkg-config ceres-solver
```

Build:

```bash
git clone --recurse-submodules https://github.com/JohnsonLabJanelia/red.git
cd red
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build release -j$(sysctl -n hw.ncpu)
```

The binary is at `release/red`.

### Linux (Build from Source)

Linux builds require an NVIDIA GPU with NVDEC support.

Install system dependencies:

```bash
# Ubuntu/Debian
sudo apt install cmake pkg-config libglfw3-dev libglew-dev \
    libopencv-dev nvidia-cuda-toolkit
```

Additional dependencies installed manually:

| Dependency    | Location                         |
|---------------|----------------------------------|
| FFmpeg        | `~/nvidia/ffmpeg/build`          |
| TensorRT      | `~/nvidia/TensorRT-8.6.1.6`     |
| LibTorch      | `lib/libtorch` (in source tree)  |
| CUDA toolkit  | `/usr/local/cuda`                |

Build:

```bash
git clone --recurse-submodules https://github.com/JohnsonLabJanelia/red.git
cd red
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release
cmake --build release -j$(nproc)
```

### Hardware Requirements

| Component | macOS                                  | Linux                                   |
|-----------|----------------------------------------|-----------------------------------------|
| GPU       | Apple Silicon (M1/M2/M3/M4/M5)        | NVIDIA with NVDEC (GTX 1060+)          |
| RAM       | 16 GB recommended                      | 16 GB recommended                       |
| Storage   | SSD recommended for video playback      | SSD recommended for video playback      |
| Display   | 1920x1080 minimum                      | 1920x1080 minimum                       |

### Optional: ONNX Runtime (SAM / JARVIS ONNX inference)

Download the [ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases) and extract it to `lib/onnxruntime/` in the source tree. The build system detects it automatically and enables SAM inference and ONNX-based JARVIS prediction.

---

## Calibration

RED includes a complete multi-camera calibration pipeline that does not depend on OpenCV. Three calibration methods are supported.

### Telecentric DLT Calibration

Best for rigs with telecentric (parallel-projection) cameras, or as a quick initial calibration using known 3D landmarks.

1. **Create a calibration project** -- File > New Project. Choose the `Table3Corners` or `Target` skeleton. Point the media folder at your calibration recordings.
2. **Label known 3D landmarks** -- in each camera view, click the landmark locations. The 3D coordinates of the landmarks must be known in advance (e.g., corners of a table at measured positions).
3. **Run DLT calibration** -- the tool computes a DLT projection matrix per camera from the 2D-3D correspondences.
4. **Create annotation project** -- start a new project that references the DLT calibration output folder. Set the project to telecentric mode.

Output: one `<camera_name>_dlt.csv` file per camera in the calibration folder.

### Projective Calibration (ArUco + Laser Refinement)

For standard projective (pinhole) camera models with lens distortion.

1. **Record a ChArUco calibration board** from each camera. RED detects ArUco markers using a custom GPU-accelerated detector (Metal on macOS) that requires no OpenCV.
2. **Run the calibration pipeline** from the Calibration Tool window. The pipeline estimates intrinsics (focal length, principal point, distortion) and extrinsics (rotation, translation) for each camera using Ceres Solver bundle adjustment.
3. **Optional laser refinement** -- use a laser pointer visible in multiple cameras simultaneously to refine extrinsic parameters. The laser detector runs on the GPU (Metal compute shaders on macOS).

Output: one `<camera_name>.yaml` file per camera containing intrinsic and extrinsic parameters in OpenCV-compatible YAML format.

### Calibration Viewer

The built-in 3D calibration viewer (powered by ImPlot3D) displays camera positions, orientations, and reprojection error statistics to help evaluate calibration quality.

---

## Annotation Workflow

### 1. Create an Annotation Project

From **File > New Project**:

- Choose a project directory and name.
- Select a skeleton preset (e.g., `Rat20` for a 20-keypoint rat skeleton) or load a custom skeleton JSON file.
- Point to an existing calibration folder (produced by the calibration pipeline above).
- Enable optional annotation types: bounding boxes, oriented bounding boxes, segmentation masks.

### 2. Load Videos

**File > Load Videos** -- select the folder containing one video per camera. Filenames must match the camera names from calibration. RED launches one hardware-accelerated decoder thread per camera and begins streaming frames.

### 3. Label Keypoints

- Use the **Labeling Tool** window to see the active skeleton and keypoint list.
- Click in any camera view to place a 2D keypoint. Cycle through keypoints with keyboard shortcuts.
- When the same keypoint is labeled in two or more cameras, RED triangulates the 3D position automatically using DLT and displays it in the 3D viewer.
- Navigate frames with the transport bar (play/pause, frame step, seek slider, direct frame number entry via Cmd+click).

### 4. SAM-Assisted Segmentation

If ONNX Runtime is available and a MobileSAM model is present in `models/mobilesam/`:

- Open the **SAM Tool** window.
- Click a point in a camera view to generate a segmentation mask.
- The mask is stored as polygon contours in the annotation data.

### 5. Export Training Data

Use the built-in JARVIS/COCO exporter (**File > Export**) or the Python scripts in `data_exporter/` to export annotations in standard formats:

| Format        | Script / Tool                        | Description                                   |
|---------------|--------------------------------------|-----------------------------------------------|
| JARVIS JSON   | Built-in exporter                    | Top-down pose estimation format               |
| COCO JSON     | `data_exporter/coco_export.py`       | Standard COCO keypoints format                |
| YOLO          | `data_exporter/yolo_export.py`       | YOLO detection/pose format                    |
| DLC CSV       | `data_exporter/dlc_export.py`        | DeepLabCut-compatible CSV                     |

---

## AI-Assisted Labeling (JARVIS)

RED integrates JARVIS, a top-down pose estimation framework, to accelerate annotation through an active learning loop.

### The Active Learning Loop

```
Label seed frames (manual)
        |
        v
Export training data (JARVIS/COCO format)
        |
        v
Train JARVIS model (external)
        |
        v
Import model into RED project
        |
        v
Run predictions on unlabeled frames
        |
        v
Review and correct predictions (manual)
        |
        v
Re-export with corrections --> iterate
```

### Model Formats

| Format   | Platform                | Inference Engine       | Typical Speed       |
|----------|-------------------------|------------------------|---------------------|
| CoreML   | macOS (Apple Silicon)   | Native CoreML          | 6-20 ms/frame       |
| ONNX     | macOS, Linux            | ONNX Runtime           | Varies by hardware   |

CoreML is the preferred backend on Apple Silicon. Models are stored per-project under a `jarvis_models/` subdirectory and registered in the `.redproj` file.

### Using JARVIS in RED

1. **Import a model** -- from the JARVIS Predict window, select a model directory containing the CoreML or ONNX model files and a `config.json` describing the model architecture (number of joints, input sizes).
2. **Run predictions** -- choose a frame range and click Predict. RED runs the center detector and keypoint estimator on each frame.
3. **Review results** -- predicted keypoints appear with a confidence score. Accept, adjust, or delete predictions frame by frame.

---

## Architecture Overview

### Threading Model

RED uses **1 main thread + N decoder threads** (one per camera):

- The **main thread** runs the ImGui render loop (Metal on macOS, OpenGL on Linux), handles user input, and performs triangulation.
- Each **decoder thread** manages a circular `PictureBuffer` and uses atomic flags to synchronize frame availability with the main thread.

### GPU Pipeline

**macOS (Apple Silicon):**

```
FFmpeg demux --> async VideoToolbox decode --> BGRA CVPixelBuffer
    --> CVMetalTextureCache --> MTLTexture --> ImGui Metal backend
```

**Linux (NVIDIA):**

```
FFmpeg demux --> NVDEC hardware decode --> CUDA NV12-to-RGB kernel
    --> OpenGL PBO --> GL texture --> ImGui OpenGL3 backend
```

### Data Model

The annotation data model centers on two structures:

- **`AnnotationMap`** (`std::map<u32, FrameAnnotation>`) -- maps frame numbers to per-frame annotation data.
- **`FrameAnnotation`** -- contains per-camera 2D keypoints, triangulated 3D keypoints, and optional extras (bounding boxes, oriented bounding boxes, segmentation masks) behind `std::unique_ptr` for lightweight allocation.

Label provenance is tracked per keypoint (`Manual`, `Predicted`, `Imported`).

### Key Source Files

| File                        | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `src/red.cpp`               | Main entry point and render loop                         |
| `src/annotation.h`          | Unified annotation data model (v2)                       |
| `src/annotation_csv.h`      | CSV persistence layer for annotations                    |
| `src/app_context.h`         | AppContext reference bundle, DisplayState                 |
| `src/project.h`             | Project JSON load/save, ProjectManager struct             |
| `src/skeleton.h` / `.cpp`   | Skeleton presets and keypoint memory management           |
| `src/camera.h`              | CameraParams (Eigen on macOS, OpenCV on Linux)           |
| `src/red_math.h`            | Eigen-based camera math and DLT triangulation            |
| `src/calibration_pipeline.h`| Full calibration pipeline (ArUco + bundle adjustment)    |
| `src/laser_calibration.h`   | Laser-based extrinsic refinement                         |
| `src/aruco_detect.h`        | Custom OpenCV-free ChArUco detection                     |
| `src/metal_context.h` / `.mm` | macOS Metal context and texture management            |
| `src/vt_async_decoder.h` / `.mm` | Async VideoToolbox decoder with PTS reorder queue  |
| `src/decoder.h` / `.cpp`    | NVIDIA NVDEC hardware decoder (Linux)                    |
| `src/render.h` / `.cpp`     | Texture/PBO pipeline (Metal on macOS, OpenGL on Linux)   |
| `src/jarvis_coreml.h` / `.mm` | Native CoreML inference on Apple Silicon              |
| `src/jarvis_inference.h`    | ONNX Runtime inference                                   |
| `src/jarvis_export.h`       | JARVIS/COCO export                                       |
| `src/sam_inference.h`       | Segment Anything (MobileSAM) integration                 |
| `src/gui/`                  | 25 modular GUI files (ImGui windows, menus, panels)      |

---

## Body Model & Inverse Kinematics (MuJoCo)

RED integrates MuJoCo body models for inverse kinematics, allowing you to fit articulated 3D body models to labeled keypoints.

### Supported Models

| Model | Skeleton | Sites | Source |
|-------|----------|-------|--------|
| Rodent (dm_control) | `Rat24Target` (24 keypoints) | 24 `_kpsite` markers | Included in `models/rodent/` |
| Fruitfly v2.1 | `Fly50` (50 keypoints) | 50 joint markers | [janelia-anibody/fruitfly](https://github.com/janelia-anibody/fruitfly) |

### Rodent Model

The rodent model is included in the repository and works out of the box:

- `models/rodent/rodent_no_collision.xml` — IK model with 24 keypoint sites
- `models/rodent/rodent.xml` — variant with skin mesh for visualization
- `models/rodent/rodent_walker_skin.skn` — skin mesh (auto-loaded when present)

### Fruitfly Model (Build Required)

The fruitfly model requires a one-time build step because the model uses OBJ mesh files that need the Python MuJoCo decoder:

```bash
pip install mujoco
python3 scripts/build_fly_model.py
```

This script:
1. Clones [janelia-anibody/fruitfly](https://github.com/janelia-anibody/fruitfly) to `lib/fruitfly/`
2. Adds 50 keypoint sites matching the `Fly50` skeleton (site definitions from [TuragaLab/fly-body-tuning](https://github.com/TuragaLab/fly-body-tuning))
3. Compiles and saves to `models/fruitfly/fruitfly_fly50.mjb` (~103 MB)

Load the generated `.mjb` file in the Body Model panel.

### IK Features

- **IK_dm_control** — gradient descent with momentum on analytical site Jacobians, following the `qpos_from_site_xpos` algorithm from dm_control
- **STAC site calibration** — iteratively adjusts keypoint site positions on the body model to minimize aggregate IK residual across many frames (Wu et al. 2013, talmolab/stac-mjx)
- **Symmetric KP Sites** — enforces bilateral symmetry during STAC calibration (midline sites stay on midline, L/R pairs get mirrored offsets)
- **Arena alignment** — SVD Procrustes alignment from labeled arena corners to the MuJoCo arena
- **Parallel qpos export** — multi-threaded batch IK solve with full metadata for reproducibility
- **Camera view overlay** — render the MuJoCo scene through calibration camera perspectives with video background for alignment verification

### MuJoCo Dependency

RED requires MuJoCo 3.6.0 as a macOS framework in `lib/mujoco.framework/`. This is an optional dependency — RED compiles and runs without it, but the Body Model panel is disabled.

---

## Supported Animals and Skeletons

### Built-in Presets

| Skeleton Name     | Keypoints | Description                                            |
|-------------------|-----------|--------------------------------------------------------|
| `Target`          | 1         | Single target point                                    |
| `RatTarget`       | 2         | Snout + target                                         |
| `Rat3Target`      | 4         | Snout, ears + target                                   |
| `Rat4`            | 4         | Snout, ears, tail                                      |
| `Rat4Target`      | 5         | Snout, ears, tail + target                             |
| `Rat6`            | 6         | Head, spine, tail base                                 |
| `Rat6Target`      | 7         | Head, spine, tail base + target                        |
| `Rat6Target2`     | 8         | Head, spine, tail base + 2 targets                     |
| `Rat7Target`      | 9         | Head, full spine, tail base + target                   |
| `Rat10Target2`    | 12        | Head, spine, hands, feet + 2 targets                   |
| `Rat20`           | 20        | Full body: head, spine, all limb joints                |
| `Rat20Target`     | 21        | Full body + target                                     |
| `Rat22`           | 22        | Full body with hip joints                              |
| `Rat24`           | 24        | Full body with detailed tail segments                  |
| `Rat24Target`     | 25        | Full body with tail + target                           |
| `Fly50`           | 50        | Drosophila: head, thorax, abdomen, wings, all 6 legs   |
| `Table3Corners`   | 3         | Calibration: 3 table corner landmarks                  |

### Custom Skeleton JSON Format

To define a custom skeleton, create a JSON file:

```json
{
    "num_nodes": 4,
    "num_edges": 3,
    "has_skeleton": true,
    "node_names": ["Head", "Neck", "Body", "Tail"],
    "edges": [[0, 1], [1, 2], [2, 3]]
}
```

Load it in the New Project dialog by selecting "Load from JSON" and pointing to the file.

---

## Building from Source

### Requirements

| Dependency      | macOS (Homebrew)            | Linux                              |
|-----------------|-----------------------------|------------------------------------|
| CMake           | `cmake` (>= 3.10)          | `cmake` (>= 3.10)                 |
| Eigen3          | `brew install eigen`        | `apt install libeigen3-dev`        |
| FFmpeg          | `brew install ffmpeg`       | Custom build in `~/nvidia/ffmpeg`  |
| GLFW            | `brew install glfw`         | `apt install libglfw3-dev`         |
| libjpeg-turbo   | `brew install jpeg-turbo`   | `apt install libturbojpeg0-dev`    |
| Ceres Solver    | `brew install ceres-solver` | `apt install libceres-dev`         |
| pkg-config      | `brew install pkg-config`   | `apt install pkg-config`           |
| GLEW            | Not needed (Metal backend)  | `apt install libglew-dev`          |
| OpenCV          | Not needed                  | `apt install libopencv-dev`        |
| CUDA toolkit    | Not needed                  | NVIDIA CUDA toolkit                |
| TensorRT        | Not needed                  | `~/nvidia/TensorRT-8.6.1.6`       |

### Build Commands

**macOS:**

```bash
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build release -j$(sysctl -n hw.ncpu)
```

**Linux:**

```bash
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release
cmake --build release -j$(nproc)
```

### Bundled Libraries (in `lib/`)

The following are included as Git submodules:

- [Dear ImGui](https://github.com/ocornut/imgui) -- immediate-mode GUI
- [ImPlot](https://github.com/epezent/implot) -- 2D plotting
- [ImPlot3D](https://github.com/brenocq/implot3d) -- 3D plotting (calibration viewer)
- [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog) -- native file/folder picker
- [nlohmann/json](https://github.com/nlohmann/json) -- JSON parsing
- [IconFontCppHeaders](https://github.com/juliettef/IconFontCppHeaders) -- icon fonts

---

## Project File Format

### `.redproj` JSON Structure

Each project is stored as a single JSON file (typically `project.redproj`) alongside its data:

```
my_project/
  project.redproj          # Project configuration
  labeled_data/            # Annotation data (one subfolder per video set)
    cam1_frame_0001.csv    # Per-frame CSV with 2D/3D keypoints
    cam1_frame_0002.csv
    ...
  jarvis_models/           # Imported JARVIS models (optional)
    mouseJan30/
      config.json
      model.mlpackage/     # CoreML model
  imgui_layout.ini         # Per-project UI layout
```

The `.redproj` file contains:

```json
{
    "project_root_path": "/path/to/projects",
    "project_path": "/path/to/projects/my_project",
    "project_name": "my_project",
    "skeleton_name": "Rat20",
    "load_skeleton_from_json": false,
    "skeleton_file": "",
    "calibration_folder": "/path/to/calibration",
    "media_folder": "/path/to/videos",
    "camera_names": ["cam1", "cam2", "cam3", "cam4"],
    "telecentric": false,
    "annotation_config": {
        "enable_keypoints": true,
        "enable_bboxes": false,
        "enable_obbs": false,
        "enable_segmentation": false,
        "class_names": ["animal"]
    },
    "jarvis_models": [],
    "active_jarvis_model": -1
}
```

### Calibration Output Formats

| Method       | File Format                   | Contents                                            |
|--------------|-------------------------------|-----------------------------------------------------|
| Projective   | `<camera>.yaml`               | OpenCV-compatible YAML with intrinsics + extrinsics |
| Telecentric  | `<camera>_dlt.csv`            | DLT projection matrix coefficients                  |

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository and create a feature branch.
2. Follow the existing code style: C++17 with header-only modules where practical.
3. GUI code goes in `src/gui/` as state struct + inline draw function, taking `AppContext &ctx`.
4. Test calibration changes against the included test suite (`cmake --build release --target test_pipeline_run`).
5. Submit a pull request with a clear description of the change.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use RED in your research, please cite:

```bibtex
@software{red2026,
    title     = {RED: Real-time, GPU-Accelerated 3D Multi-Camera Keypoint Labeling},
    author    = {Johnson Lab},
    year      = {2026},
    institution = {HHMI Janelia Research Campus},
    url       = {https://github.com/JohnsonLabJanelia/red}
}
```
