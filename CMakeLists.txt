cmake_minimum_required(VERSION 3.26)
project(red_refactor)

set(CMAKE_CXX_STANDARD 17)

include_directories(lib/imgui)
include_directories(lib/implot)
include_directories(/usr/local/cuda/include)
include_directories(lib/nvcodec)
include_directories(lib/imgui/backends)
include_directories(lib/IconFontCppHeaders)
include_directories(lib/imgui-filebrowser)
include_directories(/usr/local/include/opencv4)

include_directories(src)
add_executable(red_refactor src/main.cpp)

target_link_libraries(red_refactor opencv_sfm opencv_core opencv_bgsegm opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio opencv_highgui opencv_calib3d opencv_dnn)