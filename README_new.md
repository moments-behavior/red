# RED — Real-time 3D Labeling Tool

RED is an open-source, GPU-accelerated application for annotating 3D keypoints across synchronized multi-camera video. Built for behavioral neuroscience, RED lets researchers label body landmarks on animals (including rats, mice, and fruit flies) viewed from multiple angles, triangulate those 2D labels into 3D world coordinates in real time, and export the resulting datasets for downstream pose-estimation model training.

![RED GUI](images/gui.png)

**[Video demo](https://www.youtube.com/watch?v=9eOJaadE1Nc)**

---

## Features

| Feature | Details |
|---|---|
| GPU video decoding | NVIDIA hardware (h264/h265) via NVDEC — no CPU decode bottleneck |
| Synchronized playback | All camera streams seek and play in lockstep |
| Multi-view keypoint labeling | Click to place 2D keypoints in each camera view |
| Real-time 3D triangulation | Keypress triangulates labeled 2D points into world coordinates |
| Bounding box support | Axis-aligned and oriented bounding boxes |
| Built-in skeleton presets | Rat, mouse, and custom skeletons; define your own in JSON |
| YOLO inference | Load a PyTorch YOLO model for assisted auto-labeling |
| Reprojection error visualization | Interactive per-camera and per-keypoint error plots |
| Image and video support | MP4 (h264/h265) and directory-of-images workflows |
| Export pipeline | Python scripts export to JARVIS (COCO), YOLO detection, and YOLO pose formats |

---

## System Requirements

- Linux (tested on Ubuntu 20.04/22.04)
- NVIDIA GPU with hardware decode support (Pascal or later recommended)
- NVIDIA driver ≥ 525, CUDA 12.0, cuDNN 8.9
- ~8 GB GPU VRAM for typical multi-camera setups

---

## Dependencies

The following must be installed before building RED.

| Library | Tested version | Purpose |
|---|---|---|
| CUDA Toolkit | 12.0 | GPU compute |
| cuDNN | 8.9.3 | Deep learning primitives |
| NVIDIA Video Codec SDK | — | Hardware decode (bundled via nvcodec submodule) |
| FFmpeg | system | Video demuxing |
| OpenCV (with SfM module) | 4.8.0 | Triangulation, image I/O |
| LibTorch (CPU) | latest | YOLO model inference |
| OpenGL / GLFW | system | Rendering |

---

## Installation

### 1. Clone the repository (with submodules)

```bash
git clone --recursive https://github.com/JohnsonLabJanelia/red.git
cd red
```

### 2. Install cuDNN

Download `cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz` from the
[cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive), then:

```bash
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Verify:

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 3. Install OpenCV with SfM

Download and unzip `opencv-4.8.0` and `opencv_contrib-4.8.0` (use 4.10 for CUDA 12.2+).
See [OpenCV SfM installation](https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html) for libgflags/libglog dependencies.

```bash
cd opencv-4.8.0 && mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON \
  -D CUDA_ARCH_BIN=7.5 \
  -D WITH_TBB=ON -D WITH_QT=ON -D WITH_OPENGL=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=~/build/opencv_contrib-4.8.0/modules \
  -D BUILD_opencv_cudacodec=OFF ..

make -j $(nproc) && sudo make install
```

### 4. Install TensorRT (optional, for YOLO inference)

Download `TensorRT-8.6.1.6` for CUDA 12.0 from NVIDIA, extract it, rename the folder to `TensorRT`, and add its `lib` directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/home/$USER/nvidia/TensorRT/lib:$LD_LIBRARY_PATH
```

### 5. Install LibTorch

Download the CPU LibTorch zip from [pytorch.org](https://pytorch.org/get-started/locally/)
(select Linux → LibTorch → C++/Java → CPU). Unzip into the `lib/` folder:

```
red/
  lib/
    libtorch/
      include/
      lib/
      ...
```

### 6. Build RED

On the first build, ensure the ImGui/ImPlot object-file compilation lines in `build.sh` are **uncommented** (lines 16–26). Then:

```bash
./build.sh
```

Subsequent builds can comment those lines out to save time.

The executable `redgui` is placed in `release/`. Launch it with:

```bash
./run.sh
```

---

## Quick Start

### Step 1 — Prepare your data

Organize your videos so each camera has its own MP4 file, named by camera
(e.g., `cam1.mp4`, `cam2.mp4`, …) in a single folder.

Camera calibration files must be YAML files named `<camera_name>.yaml` in a
calibration folder. Each file should contain the standard OpenCV fields:
`camera_matrix`, `distortion_coefficients`, `rotation_matrix`,
`translation_vector`.

### Step 2 — Create a project

1. Launch RED (`./run.sh`).
2. Open **File → New Project**.
3. Fill in:
   - **Project Name** — a short identifier.
   - **Project Root Path** — parent directory for the project folder.
   - **Skeleton** — choose a built-in preset (e.g., `Rat20`) or point to a
     custom JSON skeleton file.
   - **Calibration Folder** — directory containing per-camera YAML files.
4. Click **Create Project**. RED saves a `.redproj` file and creates a
   `labeled_data/` subdirectory.

### Step 3 — Load media

- **Videos**: Go to **File → Open Videos** and select all camera MP4s.
- **Image sequences**: Go to **File → Open Images** and select the image files
  for all cameras in a single pass. Image filenames must be zero-padded frame
  numbers (e.g., `cam1_00001.jpg`).

### Step 4 — Label keypoints

1. Use **Space** to play/pause. Use **← / →** to step one frame at a time, or
   the seek slider at the top.
2. Click on a camera view to make it active (highlighted in red).
3. Select a keypoint from the keypoints panel.
4. Click the location on the image where that landmark appears.
5. Repeat for each camera that has a clear view of the landmark.
6. Press **T** to triangulate — RED computes the 3D position from all labeled
   2D views.

### Step 5 — Save

Press **Ctrl+S** at any time. Labels are written to
`<project>/labeled_data/<timestamp>/`.

### Step 6 — Export for model training

See [Data Export](#data-export).

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `←` / `→` | Step one frame backward / forward |
| `Ctrl+S` | Save all labels |
| `T` | Triangulate 3D keypoints from labeled 2D points |
| `Tab` | Cycle to the next keypoint |
| `Delete` | Remove the active label |
| `Shift+drag` | Draw a bounding box |

---

## Skeleton Configuration

RED comes with built-in skeleton presets suited for common experimental animals.
Custom skeletons are defined in a simple JSON file:

```json
{
  "name": "MyAnimal",
  "num_nodes": 7,
  "num_edges": 7,
  "edges": [[0,1],[0,2],[1,3],[2,3],[3,4],[4,5],[5,6]],
  "node_names": ["Nose","EyeL","EyeR","Neck","Spine","TailBase","TailEnd"],
  "has_skeleton": true,
  "has_bbox": false,
  "has_obb": false
}
```

| Field | Description |
|---|---|
| `node_names` | Ordered list of landmark names |
| `edges` | Pairs of node indices defining the skeleton connectivity |
| `has_skeleton` | Whether the skeleton lines are drawn |
| `has_bbox` | Enable axis-aligned bounding box annotation |
| `has_obb` | Enable oriented bounding box annotation |

To use a custom skeleton, select **File** mode in the Skeleton row of the
Create Project dialog and browse to your JSON file.

---

## Data Output Format

### Per-frame CSV files

After saving, labels appear in `<project>/labeled_data/<timestamp>/`:

| File | Contents |
|---|---|
| `keypoints3d.csv` | 3D world-space coordinates: `frame_id, kp0_x, kp0_y, kp0_z, kp1_x, ...` |
| `<camera_name>.csv` | 2D image-space coordinates per camera: `frame_id, kp0_x, kp0_y, kp0_conf, ...` |

Unlabeled keypoints are stored as `NaN`. Confidence values for manually placed
labels are `1.0`; YOLO-assisted labels carry the model's confidence score.

---

## Data Export

Python export scripts live in `data_exporter/`. Install dependencies once:

```bash
cd data_exporter
pip install -r requirements.txt
```

### Export to JARVIS (COCO format)

[JARVIS](https://github.com/JARVIS-MoCap/JARVIS-HybridNet) is a hybrid 2D/3D
pose-estimation framework that uses COCO-formatted datasets.

```bash
python red3d2jarvis.py \
  --input  /path/to/labeled_data/<timestamp> \
  --output /path/to/jarvis_dataset \
  --calibration /path/to/calibration_folder \
  --skeleton Rat20
```

The script performs a 90/10 train/validation split, extracts JPEG frames from
the source videos, and writes COCO JSON annotation files alongside YAML
calibration files.

Use the `-s` flag to export a subset of keypoints if your labeling skeleton
contains more nodes than your model requires.

### Export to YOLO detection

```bash
python red3d2yolo.py \
  --input  /path/to/labeled_data/<timestamp> \
  --output /path/to/yolo_dataset
```

Writes YOLO-format `.txt` label files (normalized center-x, center-y, width,
height) alongside extracted frames.

### Export to YOLO pose

```bash
python red2yolopose.py \
  --input  /path/to/labeled_data/<timestamp> \
  --output /path/to/yolo_pose_dataset
```

Writes YOLO pose `.txt` files with bounding box plus normalized keypoint
coordinates and visibility flags.

---

## YOLO-Assisted Labeling

RED can run a YOLO object-detection or pose-estimation model in the background
to provide initial labels that you refine manually.

1. Go to **YOLO → Load Model** and select a `.pt` file (LibTorch format).
2. Set the confidence threshold slider.
3. Enable **Auto YOLO Labeling** to process a range of frames automatically.
4. Review and correct predictions frame by frame.

---

## Reprojection Error

After triangulation you can inspect the accuracy of your 3D labels via
**View → Reprojection Error**. The panel shows:

- A bar chart of mean reprojection error per camera or per keypoint.
- Error bars indicating SD or SEM.
- Scatter plots that reveal systematic biases.

High reprojection error in a particular camera often indicates a calibration
issue or an incorrectly placed 2D label.

---

## Contributing

Bug reports and feature requests are welcome — please open a GitHub Issue.

To contribute code:

1. Fork the repository.
2. Create a feature branch.
3. Submit a Pull Request with a clear description of the change.

Please follow the existing code style (C++17, clang-format default settings).

---

## License

See [LICENSE](LICENSE).

---

## Citation

If you use RED in your research, please cite:

> [Citation will be added upon publication]

---

## Contact

Questions about the software: [Jinyao Yan](mailto:yanj11@janelia.hhmi.org)
Johnson Lab, HHMI Janelia Research Campus
