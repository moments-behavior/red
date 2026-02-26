cmake_minimum_required(VERSION 3.10)

if(APPLE)
    project(red LANGUAGES CXX)

    # Homebrew prefix for Apple Silicon
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

    find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc imgcodecs videoio)
    find_package(glfw3 REQUIRED)
    find_package(Vulkan REQUIRED)

    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")

    file(GLOB SRC_FILES src/*.cpp)
    list(REMOVE_ITEM SRC_FILES
        "${CMAKE_SOURCE_DIR}/src/NvDecoder.cpp"
        "${CMAKE_SOURCE_DIR}/src/yolo_torch.cpp"
        "${CMAKE_SOURCE_DIR}/src/yolo_export.cpp"
    )

    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
        ${DIR_IMGUI}/backends/imgui_impl_vulkan.cpp
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)

    add_executable(red
        ${SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
    )

    target_include_directories(red PRIVATE
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${OpenCV_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_DIRS}
        ${Vulkan_INCLUDE_DIRS}
        /opt/homebrew/include
    )

    target_link_directories(red PRIVATE ${FFMPEG_LIBRARY_DIRS})

    target_link_libraries(red PRIVATE
        ${OpenCV_LIBS}
        opencv_sfm
        glfw
        Vulkan::Vulkan
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        ${FFMPEG_LIBRARIES}
        pthread
    )

    target_compile_definitions(red PRIVATE GLFW_INCLUDE_VULKAN)

    set_target_properties(red PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Test target: same source/deps as red, replacing red.cpp with tests/test_gui.cpp
    set(TEST_GUI_SRC_FILES ${SRC_FILES})
    list(REMOVE_ITEM TEST_GUI_SRC_FILES "${CMAKE_SOURCE_DIR}/src/red.cpp")

    add_executable(test_gui
        ${TEST_GUI_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        tests/test_gui.cpp
    )

    target_include_directories(test_gui PRIVATE
        src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${OpenCV_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_DIRS}
        ${Vulkan_INCLUDE_DIRS}
        /opt/homebrew/include
    )

    target_link_directories(test_gui PRIVATE ${FFMPEG_LIBRARY_DIRS})

    target_link_libraries(test_gui PRIVATE
        ${OpenCV_LIBS}
        opencv_sfm
        glfw
        Vulkan::Vulkan
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        ${FFMPEG_LIBRARIES}
        pthread
    )

    target_compile_definitions(test_gui PRIVATE GLFW_INCLUDE_VULKAN)

    set_target_properties(test_gui PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

else()
    # Linux build with NVIDIA CUDA support
    project(red LANGUAGES CXX CUDA)

    # Set CUDA architecture and options
    set(CMAKE_CUDA_ARCHITECTURES 80)

    add_compile_options(-msse4.1)

    # Set Directories
    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
    set(DIR_FFMPEG "$ENV{HOME}/nvidia/ffmpeg/build")
    set(DIR_TENSORRT "$ENV{HOME}/nvidia/TensorRT-8.6.1.6")
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")
    set(DIR_ICONFONT "${CMAKE_SOURCE_DIR}/lib/IconFontCppHeaders")

    # Added for LibTorch
    set(DIR_LIBTORCH "${CMAKE_SOURCE_DIR}/lib/libtorch")
    set(LIBTORCH_INCLUDE "${DIR_LIBTORCH}/include")
    set(LIBTORCH_INCLUDE_API "${DIR_LIBTORCH}/include/torch/csrc/api/include")
    set(LIBTORCH_LIB "${DIR_LIBTORCH}/lib")

    # Source Files
    file(GLOB SRC_FILES src/*.cpp)
    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
        ${DIR_IMGUI}/backends/imgui_impl_opengl3.cpp
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)

    # CUDA Source Files
    set(CUDA_SRC_FILES
        src/create_image_cuda.cu
        src/ColorSpace.cu
        src/kernel.cu
    )

    # Main Executable
    add_executable(red
        ${SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${CUDA_SRC_FILES}
    )

    # Include Directories
    target_include_directories(red PRIVATE
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
            ${LIBTORCH_INCLUDE}
            ${LIBTORCH_INCLUDE_API}
    )

    # Link Libraries
    target_link_libraries(red PRIVATE
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
        -L${LIBTORCH_LIB} -ltorch -lc10 -ltorch_cpu
    )

    # Set RPATH so runtime finds libtorch
    set_target_properties(red PROPERTIES
        BUILD_RPATH ${LIBTORCH_LIB}
        INSTALL_RPATH ${LIBTORCH_LIB}
    )

endif()
