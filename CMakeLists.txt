cmake_minimum_required(VERSION 3.10)
project(redgui LANGUAGES CXX CUDA)

# Set CUDA architecture and options
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-msse4.1)

# Set Directories
set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
set(DIR_FFMPEG "$ENV{HOME}/nvidia/ffmpeg/build")
set(DIR_TENSORRT "$ENV{HOME}/build/TensorRT-8.6.1.6")
set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")
set(DIR_ICONFONT "${CMAKE_SOURCE_DIR}/lib/IconFontCppHeaders")

# Source Files
file(GLOB SRC_FILES src/*.cpp lib/ImGuiFileDialog/ImGuiFileDialog.cpp)
file(GLOB IMGUI_SRC
    ${DIR_IMGUI}/*.cpp
    ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
    ${DIR_IMGUI}/backends/imgui_impl_opengl3.cpp
)
file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)

# CUDA Source Files
set(CUDA_SRC_FILES
    src/create_image_cuda.cu
    src/ColorSpace.cu
    src/kernel.cu
)

# Main Executable
add_executable(redgui 
    ${SRC_FILES}
    ${IMGUI_SRC}
    ${IMPLOT_SRC}
    ${CUDA_SRC_FILES}
)
# Include Directories
target_include_directories(redgui PRIVATE
        ${SRC_DIR}
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        lib/nvcodec
        lib/IconFontCppHeaders
        lib/ImGuiFileDialog
        /usr/local/include/opencv4
        ${DIR_FFMPEG}/include
        ${DIR_TENSORRT}/include
        /usr/local/cuda/include
)

# Link Libraries
target_link_libraries(redgui PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${OPENGL_LIBRARIES}
    glfw
    GLEW GLU GL
    pthread
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnvcuvid -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial
    -L${DIR_FFMPEG}/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec
        -lopencv_sfm -lopencv_core -lopencv_bgsegm -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio -lopencv_calib3d -lopencv_dnn
    -L${DIR_TENSORRT}/lib/ -lnvinfer -lnvinfer_plugin
)
