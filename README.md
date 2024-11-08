# Red Labeling App
3D labeling tool for multiple cameras in C++

Contact [Jinyao Yan](yanj11@janelia.hhmi.org) if you have questions about the software 

![gui](images/gui.png)

## Features
1. Real-time GPU accelerated decoding (h264, h265)
2. Synchronized decoding
3. Multi-view keypoints labeling and triangulation  
4. YOLOv5 (OpenCV cuDNN ONNX model) and YOLOv8 (TensoRT) inference  

## Dependencies
1. NVIDIA Video Codec SDK
1. CUDA Toolkit and cuDNN
2. FFmpeg 
3. OpenCV
4. TensorRT
5. OpenGL

## Build instructions 

### Install OpenCV
- download and upzip `opencv-4.8.0.zip` and `opencv_contrib-4.8.0.zip`. Unzip the folders to `~/build/`, for instance. 

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
make -j8 
sudo make install
```

### Install cuDNN (depends on CUDA installation)
- download the cudnn install files (we use `cudnn 8.9.3` with `driver 525.105.17` and `cuda 12` )
- you may run the commands below for the exact version or download a TAR file for Linux_x86_64 from the [cudnn version archives](https://developer.nvidia.com/rdp/cudnn-archive)
    ```
    cd /home/$USER/setup_files
    wget https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.3/local_installers/12.x/cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz/
- extract the file
  ```
  tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
  ```
- copy cudnn files to where your `cuda` is installed -- we assume it is installed at `/usr/local/cuda` 
  ```
  cd <to where files where extracted above>
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

### Install RED 

- Clone the repo and submodules

```
git clone https://github.com/JohnsonLabJanelia/orange.git
git submodule init
git submodule update
```

If you are building the project for the first time, uncomment [`line 16 ~ line 26`](https://github.com/JohnsonLabJanelia/red/blob/0829b09d20b0dbccb0ea6df7a20e5ee4e23f635f/build_linux.sh#L16) for building `ImGui` and `ImPlot` object files. Run
```
./build.sh
```
Comment out Line 16 ~ line 26 to reduce compiling time afterwards. 

Once built, it will make a folder called `release`. The executable `redgui` is the application. Start the program using the run script. 

```
./run.sh
```

## Use the App
A video demo is coming...

## Format data for Deep Learning
Currently we are saving labeled keypoints simply as a plain csv file. We provide python scripts for formating data as [COCO format](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html), which is used by [JARVIS](https://github.com/JARVIS-MoCap/JARVIS-HybridNet). Please refer to [data_exporter](https://github.com/JohnsonLabJanelia/red/tree/main/data_exporter).

## Contribute

Please open an issue for bug fix or feature request. If you wish to make changes to the source code, you can fork the repo. To contribute to the project, please create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
