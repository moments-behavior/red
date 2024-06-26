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

include_directories(src)
add_executable(red_refactor src/main.cpp)