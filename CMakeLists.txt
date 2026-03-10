cmake_minimum_required(VERSION 3.10)

if(APPLE)
    project(red LANGUAGES CXX OBJCXX)

    # Homebrew prefix — defaults to /opt/homebrew (Apple Silicon).
    # Override with -DHOMEBREW_PREFIX=/usr/local for Intel Macs, or let
    # the Homebrew formula pass the correct value via CMAKE_PREFIX_PATH.
    if(NOT DEFINED HOMEBREW_PREFIX)
        execute_process(
            COMMAND brew --prefix
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NOT HOMEBREW_PREFIX)
            set(HOMEBREW_PREFIX "/opt/homebrew")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

    find_package(Eigen3 REQUIRED)
    find_package(glfw3 REQUIRED)
    pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)
    find_package(Ceres REQUIRED)

    # Optional ONNX Runtime for SAM inference (use official release, not Homebrew)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")
    if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        set(HAS_ONNXRUNTIME TRUE)
        message(STATUS "ONNX Runtime found at ${ONNXRUNTIME_DIR}")
    else()
        set(HAS_ONNXRUNTIME FALSE)
        message(STATUS "ONNX Runtime not found — SAM inference disabled")
    endif()

    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
    set(DIR_IMPLOT3D "${CMAKE_SOURCE_DIR}/lib/implot3d")
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")

    file(GLOB SRC_FILES src/*.cpp)
    list(REMOVE_ITEM SRC_FILES
        "${CMAKE_SOURCE_DIR}/src/NvDecoder.cpp"
    )

    # Objective-C++ source files (.mm)
    file(GLOB MM_SRC_FILES src/*.mm)

    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
        ${DIR_IMGUI}/backends/imgui_impl_metal.mm
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)
    file(GLOB IMPLOT3D_SRC ${DIR_IMPLOT3D}/*.cpp)
    list(REMOVE_ITEM IMPLOT3D_SRC "${DIR_IMPLOT3D}/implot3d_demo.cpp")

    add_executable(red
        ${SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
    )

    target_include_directories(red PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_IMPLOT3D}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )

    target_link_directories(red PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})

    target_link_libraries(red PRIVATE
        Eigen3::Eigen
        glfw
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        "-framework CoreVideo"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    # Removed: Vulkan::Vulkan, -framework Accelerate (vImage no longer used)

    # Optional ONNX Runtime for SAM inference
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(red PRIVATE RED_HAS_ONNXRUNTIME)
        target_include_directories(red PRIVATE "${ONNXRUNTIME_DIR}/include")
        target_link_libraries(red PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endif()

    # Enable ARC for all Objective-C++ translation units
    set_source_files_properties(
        ${MM_SRC_FILES}
        ${DIR_IMGUI}/backends/imgui_impl_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_target_properties(red PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Test target: same source/deps as red, replacing red.cpp with tests/test_gui.cpp
    set(TEST_GUI_SRC_FILES ${SRC_FILES})
    list(REMOVE_ITEM TEST_GUI_SRC_FILES "${CMAKE_SOURCE_DIR}/src/red.cpp")

    add_executable(test_gui
        ${TEST_GUI_SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
        tests/test_gui.cpp
    )

    target_include_directories(test_gui PRIVATE
        src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_IMPLOT3D}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${FFMPEG_INCLUDE_DIRS}
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )

    target_link_directories(test_gui PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})

    target_link_libraries(test_gui PRIVATE
        Eigen3::Eigen
        glfw
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        "-framework CoreVideo"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )

    set_source_files_properties(
        ${MM_SRC_FILES}
        ${DIR_IMGUI}/backends/imgui_impl_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    if(HAS_ONNXRUNTIME)
        target_compile_definitions(test_gui PRIVATE RED_HAS_ONNXRUNTIME)
        target_include_directories(test_gui PRIVATE "${ONNXRUNTIME_DIR}/include")
        target_link_libraries(test_gui PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endif()

    set_target_properties(test_gui PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Annotation system test: same deps as test_gui
    add_executable(test_annotation
        ${TEST_GUI_SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
        tests/test_annotation.cpp
    )
    target_include_directories(test_annotation PRIVATE
        src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_IMPLOT3D}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${FFMPEG_INCLUDE_DIRS}
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )
    target_link_directories(test_annotation PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_annotation PRIVATE
        Eigen3::Eigen
        glfw
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        "-framework CoreVideo"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(test_annotation PRIVATE RED_HAS_ONNXRUNTIME)
        target_include_directories(test_annotation PRIVATE "${ONNXRUNTIME_DIR}/include")
        target_link_libraries(test_annotation PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endif()
    set_target_properties(test_annotation PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Calibration debug test: Eigen + Ceres only (no OpenCV)
    add_executable(test_calib_debug tests/test_calib_debug.cpp)
    target_include_directories(test_calib_debug PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_calib_debug PRIVATE ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_calib_debug PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${TURBOJPEG_LIBRARIES}
        pthread
    )
    set_target_properties(test_calib_debug PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Full pipeline test: runs all 7 steps, validates reproj error
    add_executable(test_pipeline_run
        tests/test_pipeline_run.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
    )
    target_include_directories(test_pipeline_run PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_pipeline_run PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_pipeline_run PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework CoreVideo"
        "-framework CoreMedia"
        "-framework VideoToolbox"
        "-framework IOKit"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_pipeline_run PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # JARVIS import test: header-only Eigen math (no OpenCV, no GUI, no Metal)
    add_executable(test_jarvis_import tests/test_jarvis_import.cpp)
    target_include_directories(test_jarvis_import PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/lib/eigen
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )
    target_link_libraries(test_jarvis_import PRIVATE
        Eigen3::Eigen
    )
    set_target_properties(test_jarvis_import PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Video pipeline test: tests video-based aruco calibration
    add_executable(test_video_pipeline
        tests/test_video_pipeline.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
    )
    target_include_directories(test_video_pipeline PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_video_pipeline PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_video_pipeline PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework CoreVideo"
        "-framework CoreMedia"
        "-framework VideoToolbox"
        "-framework IOKit"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_video_pipeline PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Detection timing benchmark
    add_executable(test_detect_timing
        tests/test_detect_timing.cpp
        src/aruco_metal.mm
    )
    target_include_directories(test_detect_timing PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_detect_timing PRIVATE ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_detect_timing PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${TURBOJPEG_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework IOKit"
        "-framework CoreVideo"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_detect_timing PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )


    # Granular detection timing benchmark
    add_executable(test_detect_granular
        tests/test_detect_granular.cpp
        src/aruco_metal.mm
    )
    target_include_directories(test_detect_granular PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_detect_granular PRIVATE ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_detect_granular PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${TURBOJPEG_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework IOKit"
        "-framework CoreVideo"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_detect_granular PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Contour finding drill-down benchmark
    add_executable(test_contour_drill
        tests/test_contour_drill.cpp
        src/aruco_metal.mm
    )
    target_include_directories(test_contour_drill PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_contour_drill PRIVATE ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_contour_drill PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${TURBOJPEG_LIBRARIES}
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
        "-framework IOKit"
        "-framework CoreVideo"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_contour_drill PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    # Cross-validation: laser-refined vs ChArUco landmarks (Eigen-only, no OpenCV)
    add_executable(test_calib_crossval tests/test_calib_crossval.cpp)
    target_include_directories(test_calib_crossval PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )
    target_link_libraries(test_calib_crossval PRIVATE
        Eigen3::Eigen
    )
    set_target_properties(test_calib_crossval PROPERTIES
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
        -L${DIR_FFMPEG}/lib/ -lavformat -lavutil -lavcodec
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
