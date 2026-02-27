# RED — Real-time multi-camera keypoint labeling

RED is a GPU-accelerated, multi-camera 3D keypoint labeling tool written in C++.
It is designed for behavioral neuroscience datasets where multiple synchronized
cameras capture an animal at high frame rate.

Contact [Jinyao Yan](yanj11@janelia.hhmi.org) with questions about the software.

![gui](images/gui.png)

## Video demo

See this [link](https://www.youtube.com/watch?v=9eOJaadE1Nc) for a video demo of the application.

---

## Features

- **Real-time GPU-accelerated decoding** — H.264 and H.265 (HEVC) at up to 180+ fps
- **Synchronized multi-camera playback** — all cameras seek and advance together
- **2D keypoint labeling** — click to place keypoints on each camera view
- **3D triangulation** — multi-view linear triangulation using OpenCV (DLT)
- **Reprojection error visualization** — bar charts and scatter plots per camera / keypoint
- **Bounding box annotation** — axis-aligned and oriented bounding boxes
- **YOLO auto-labeling** — load a LibTorch model to seed keypoint positions
- **Custom skeleton support** — define any anatomy via JSON or built-in presets
- **Data export** — CSV output; Python scripts for COCO / JARVIS / YOLO formats
- **Cross-platform** — Linux with NVIDIA CUDA, macOS with Apple Silicon Metal

---

## Platform support

| Platform | GPU backend | Decoder |
|---|---|---|
| Linux (NVIDIA GPU) | OpenGL + CUDA | NVIDIA NVDEC (`NvDecoder`) |
| macOS (Apple Silicon) | Metal + VideoToolbox | `VTDecompressionSession` (async) |

The Linux and macOS code paths are entirely separate under `#ifdef __APPLE__`
guards. No cross-platform compromises were needed.

---

## macOS — Quick Start (Apple Silicon)

### Dependencies

```bash
brew install ffmpeg opencv glfw pkg-config
```

No CUDA, no Vulkan, and no MoltenVK are required on macOS.

### Build

```bash
git clone --recursive https://github.com/JohnsonLabJanelia/red.git
cd red
cmake -S . -B release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build release --config Release -j$(sysctl -n hw.logicalcpu)
```

### Run

```bash
./release/red
```

### macOS performance notes

On Apple Silicon (M1/M2/M3) with a 3-camera, 180 fps, 3208×2200 dataset,
the application plays back at **1.0× real-time**. The pipeline is fully
GPU-side:

```
FFmpeg demux → VTDecompressionSession (async) → CVPixelBuffer (IOSurface)
  → CVMetalTextureCache (zero-copy) → MTLBlitCommandEncoder
  → MTLTexture (BGRA) → imgui_impl_metal → CAMetalLayer → display
```

VideoToolbox reads the color space metadata from the stream (BT.601 / BT.709,
full / video range) and applies the correct YUV→RGB matrix internally, so the
pipeline is correct for any camera source.

---

## Linux / NVIDIA — Dependencies and Build

### 1. Install CUDA and cuDNN

We use `cuDNN 8.9.3` with `driver 525.105.17` and `CUDA 12.0`. Download the
appropriate TAR from the
[cuDNN version archive](https://developer.nvidia.com/rdp/cudnn-archive) and
install:

```bash
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Verify:
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 2. Install OpenCV with SfM

Download `opencv-4.8.0.zip` and `opencv_contrib-4.8.0.zip` (use 4.10 for
CUDA 12.2+). Follow the
[SfM installation guide](https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html)
for the optional Ceres solver dependency, then build:

```bash
cd opencv-4.8.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=7.5 \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=~/build/opencv_contrib-4.8.0/modules \
      -D OPENCV_GENERATE_PKGCONFIG=ON ..
make -j$(nproc)
sudo make install
```

### 3. Install TensorRT

We use `TensorRT-8.6.1.6` with `CUDA 12.0` (use TensorRT 10 for CUDA 12.2+).

```bash
cd ~/nvidia
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
mv TensorRT-8.6.1.6 TensorRT
echo 'export LD_LIBRARY_PATH=~/nvidia/TensorRT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
cd ~/nvidia/TensorRT/samples/trtexec && make
~/nvidia/TensorRT/bin/trtexec
```

### 4. Install LibTorch (optional — required for YOLO inference only)

Download the CPU or CUDA version from [PyTorch](https://pytorch.org/get-started/locally/)
(Linux → LibTorch → C++) and unzip into `lib/`:

```bash
unzip libtorch-*.zip -d lib/
```

### 5. Build RED (Linux)

```bash
git clone --recursive https://github.com/JohnsonLabJanelia/red.git
cd red
./build.sh
```

On first build, ensure lines 16–26 of `build.sh` are uncommented to compile
ImGui and ImPlot object files. Comment them out afterwards to reduce build time.

### 6. Run (Linux)

```bash
./run.sh
```

---

## Project file format

RED projects are stored as `.redproj` JSON files. Key fields:

| Field | Description |
|---|---|
| `project_name` | Human-readable identifier |
| `project_path` | Absolute path to the project directory |
| `skeleton_name` | Built-in preset (e.g., `"rat"`) or empty |
| `load_skeleton_from_json` | Set to `true` to use a custom skeleton file |
| `skeleton_file` | Path to custom skeleton JSON |
| `calibration_folder` | Directory containing per-camera OpenCV YAML files |
| `media_folder` | Directory containing video files (`.mp4`) |
| `camera_names` | Ordered list of camera identifiers |

---

## Custom skeleton format

```json
{
    "name": "zebrafish",
    "num_nodes": 7,
    "num_edges": 6,
    "node_names": ["Head", "Neck", "Spine1", "Spine2", "Spine3", "TailBase", "TailTip"],
    "edges": [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]],
    "has_skeleton": true,
    "has_bbox": false,
    "has_obb": false
}
```

See `example/skeleton.json` for a complete example.

---

## Output data format

Labels are saved in `<project>/labeled_data/<timestamp>/`.

### `keypoints3d.csv`
```
frame_id, kp0_x, kp0_y, kp0_z, kp1_x, kp1_y, kp1_z, ...
42, 12.3, -5.1, 200.4, NaN, NaN, NaN, ...
```
Un-triangulated keypoints are written as `NaN`.

### `<camera_name>.csv`
```
frame_id, kp0_x, kp0_y, kp0_conf, kp1_x, kp1_y, kp1_conf, ...
42, 310.5, 220.1, 1.0, NaN, NaN, 0.0, ...
```

---

## Deep learning export

Python export scripts are provided in `data_exporter/`:

| Script | Purpose |
|---|---|
| `red3d2jarvis.py` | Export to [JARVIS](https://github.com/JARVIS-MoCap/JARVIS-HybridNet) / COCO format |
| `red3d2yolo.py` | Export to YOLO detection format |
| `red2yolopose.py` | Export to YOLO pose format |
| `jarvis2red3d.py` | Import JARVIS predictions back to RED CSVs |

See [data_exporter/README.md](data_exporter/README.md) for usage details.

---

## Contribute

Please open an issue for bug reports or feature requests. To contribute code,
fork the repository and open a pull request.
