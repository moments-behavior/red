# Red Labeling App
This is based on Dear ImGui examples using GLFW, and just adds the minimum for rendering the result of a CUDA computation with PBO.

# Building 
## Build opencv with dnn support 
1. Download and upzip `opencv-4.6.0.zip` and `opencv_contrib-4.6.0.zip`. 
2. Install Ceres Solver with instruction from https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html

If you want to build with opencv sfm, please follow instruction from: https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html. 

Ceres is optional, but if you want to enable the solver, make sure you are building with Ceres 1.14.x:  https://ceres-solver.googlesource.com/ceres-solver/+/refs/heads/1.14.x, by 

```
git checkout facb199f3eda902360f9e1d5271372b7e54febe1
```
otherwise you will have build issue, and have to turn off Ceres. 
3. Install cudnn follow instruction from cudnn website. The version I am currently using is `cudnn-linux-x86_64-8.7.0.84_cuda11-archive/`. 
4. Build opencv using 

```
cd opencv-4.6.0/ 
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
-D OPENCV_EXTRA_MODULES_PATH=~/Build/opencv_contrib-4.6.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON ..
```

```
make -j8 
sudo make install
```