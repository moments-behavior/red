# Red Labeling App
3D labeling tool for multiple cameras in C++

[![DOI](https://zenodo.org/badge/544234536.svg)](https://doi.org/10.5281/zenodo.19688189)

![gui](images/gui.png)

## Video demo
Please see this [link](https://www.youtube.com/watch?v=9eOJaadE1Nc) for a video demo of the app. 

## Features
1. Real-time GPU accelerated decoding (h264, h265)
2. Synchronized decoding
3. Multi-view keypoints labeling and triangulation  

## Dependencies
1. NVIDIA Video Codec SDK
1. CUDA Toolkit and cuDNN
2. FFmpeg 
3. OpenCV
4. LibTorch
5. OpenGL

## Build instructions 

### Install cuDNN (depends on CUDA installation)
- download the cudnn install files (we use `cudnn 8.9.3` with `driver 525.105.17` and `cuda 12.0` )
- you may run the commands below for the exact version or download a TAR file for `cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz` from the [cudnn version archives](https://developer.nvidia.com/rdp/cudnn-archive)
- extract the file
- copy cudnn files to where your `cuda` is installed -- we assume it is installed at `/usr/local/cuda` 
  ```
  sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
  sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
  sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
  ```
- verify installation and cudnn version
  ```
  source ~/.bashrc
  cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
  ```
  you can expect an output like:
  ```
  #define CUDNN_MAJOR 8
  #define CUDNN_MINOR 9
  #define CUDNN_PATCHLEVEL 3
  --
  #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
  ```

### Install OpenCV
- download and upzip `opencv-4.8.0.zip` and `opencv_contrib-4.8.0.zip`. Unzip the folders to `~/build/`, for instance. Note, if you are using cuda 12.2, please download opencv-4.10 instead.

- to build OpenCV with opencv sfm, please follow instructions from: https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html first to install sfm dependency. Ceres solver is optional. If you wish to install ceres solver, a more detailed installation instruction can be found at: http://ceres-solver.org/installation.html#linux. At the time of test, one need to set CMake flag USE_CUDA=OFF for ceres.  

- build OpenCV using

```
cd opencv-4.8.0/ 
mkdir build
cd build 
```

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=7.5 \
-D WITH_V4L=ON \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/build/opencv_contrib-4.8.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON ..
```

```
make -j $(nproc) 
sudo make install
```
### Install TensorRT
- [tensor-rt (depends on nvidia-driver and CUDA)](#install-tensor-rt)

### install tensor-rt
this is based on [these instructions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) (has more details if needed)

**0. download and extract tensor-rt installation file**
  - we use `TensorRT-8.6.1.6` with `cuda 12.0`  -- you can directly download this (or from this page). But if you are using `cuda 12.2` and above, please use TensorRT 10, for instance `TensorRT-10.6.0.26.Linux.x86_64-gnu.cuda-12.6`. The installation steps are similar.
    ```
    cd /home/$USER/nvidia
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
    tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
    ```
  - this should extract a folder `TensorRT-8.6.1.6` with following subdirectories
    ```
    bin  data  doc  include  lib  python  samples  targets
    ```
  - rename the folder `TensorRT`.

**1. add tensor-rt path in bashrc**
  - Add the absolute path to the TensorRT lib directory to the environment variable `LD_LIBRARY_PATH`:
    ```
    export LD_LIBRARY_PATH=/home/$USER/nvidia/TensorRT/lib:$LD_LIBRARY_PATH 
    source ~/.bashrc
    ```
**2. verify installation**
  - try to build one of the sample programs (say, `trtexec`) to verify installation
    ```
    cd /home/$USER/nvidia/TensorRT/samples/trtexec
    make
    ```
  - run the program built above
    ```
    cd home/$USER/nvidia/TensorRT/bin/
    ./trtexec
    ```

### Install LibTorch
Download libtorch cpu version from [PyTorch](https://pytorch.org/get-started/locally/) selecting Linux, LibTorch, C++. Unzip it into `lib` folder.


### Install RED 

- Clone the repo and submodules

```
git clone --recursive https://github.com/JohnsonLabJanelia/red.git
```
To build the project, 
```
./build.sh
```
We use CMake to build the project. Make sure you have CMake installed. 

Once built, it will make a folder called `release`. The executable `redgui` is the application. Start the program using the run script. 

```
./run.sh
```
To have RED show up as other installed programs with a desktop entry, install it with:
```
./install.sh
```

## Config RED (optional)
If you want to open videos with a default media folder (instead of navigating to the folder everytime), and save red projects to a default folder (otherwise it is saved default to `$HOME/red_data`), you can set up a config.json file in `$HOME/.config/red`.

Example config file:
```
{
	"media_folder": "/nfs/exports/ratlv",
	"project_folder": "/nfs/exports/ratlv/fetch_runs"
}
```

## Format data for Deep Learning
Currently we are saving labeled keypoints simply as a plain csv file. We provide python scripts for formating data as [COCO format](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html), which is used by [JARVIS](https://github.com/JARVIS-MoCap/JARVIS-HybridNet). Please refer to [data_exporter](https://github.com/JohnsonLabJanelia/red/tree/main/data_exporter).

## Citation
**Red** is devloped by Jinyao Yan, with contributions from Wilson Chen, Diptodip Deb, Ratan Othayoth and Rob Johnson. If you use **Red**, please cite the software 

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

Contact [Jinyao Yan](yanj11@janelia.hhmi.org) if you have questions about the software 

## Contribute

Please open an issue for bug fix or feature request. If you wish to make changes to the source code, you can fork the repo. To contribute to the project, please create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
