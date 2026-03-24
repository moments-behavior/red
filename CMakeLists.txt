cmake_minimum_required(VERSION 3.18...4.2)

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
        "-framework CoreML"
        "-framework Accelerate"
        "-framework CoreVideo"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    # Removed: Vulkan::Vulkan (Accelerate still used by superpoint_coreml.mm / jarvis_coreml.mm)

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
        "-framework CoreML"
        "-framework Accelerate"
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
        "-framework CoreML"
        "-framework Accelerate"
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

    # SAM integration test: same deps as test_annotation
    add_executable(test_sam_integration
        ${TEST_GUI_SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
        tests/test_sam_integration.cpp
    )
    target_include_directories(test_sam_integration PRIVATE
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
    target_link_directories(test_sam_integration PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_sam_integration PRIVATE
        Eigen3::Eigen
        glfw
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        "-framework CoreVideo"
        "-framework CoreML"
        "-framework Accelerate"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(test_sam_integration PRIVATE RED_HAS_ONNXRUNTIME)
        target_include_directories(test_sam_integration PRIVATE "${ONNXRUNTIME_DIR}/include")
        target_link_libraries(test_sam_integration PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endif()
    set_target_properties(test_sam_integration PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # JARVIS golden-file comparison test: same deps as test_annotation
    add_executable(test_jarvis_golden
        ${TEST_GUI_SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
        tests/test_jarvis_golden.cpp
    )
    target_include_directories(test_jarvis_golden PRIVATE
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
    target_link_directories(test_jarvis_golden PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_jarvis_golden PRIVATE
        Eigen3::Eigen
        glfw
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        "-framework CoreVideo"
        "-framework CoreML"
        "-framework Accelerate"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(test_jarvis_golden PRIVATE RED_HAS_ONNXRUNTIME)
        target_include_directories(test_jarvis_golden PRIVATE "${ONNXRUNTIME_DIR}/include")
        target_link_libraries(test_jarvis_golden PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
    endif()
    set_target_properties(test_jarvis_golden PROPERTIES
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

    # Calibration reload roundtrip test (rat dataset)
    add_executable(test_calib_reload
        tests/test_calib_reload.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
    )
    target_include_directories(test_calib_reload PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_calib_reload PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_calib_reload PRIVATE
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
    set_target_properties(test_calib_reload PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # SuperPoint unified pipeline test
    add_executable(test_superpoint_pipeline
        tests/test_superpoint_pipeline.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
    )
    target_include_directories(test_superpoint_pipeline PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_superpoint_pipeline PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_superpoint_pipeline PRIVATE
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
        "-framework CoreML"
        "-framework Accelerate"
        "-framework IOKit"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_superpoint_pipeline PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # SuperPoint native pipeline benchmark (full end-to-end)
    add_executable(bench_superpoint_native
        tests/bench_superpoint_native.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
    )
    target_include_directories(bench_superpoint_native PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(bench_superpoint_native PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(bench_superpoint_native PRIVATE
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
        "-framework CoreML"
        "-framework Accelerate"
        "-framework IOKit"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(bench_superpoint_native PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Run experimental calibration from CLI
    add_executable(run_experimental_calib
        tests/run_experimental_calib.cpp
        src/aruco_metal.mm src/FFmpegDemuxer.cpp src/vt_async_decoder.mm src/superpoint_coreml.mm
    )
    target_include_directories(run_experimental_calib PRIVATE
        src lib/nvcodec ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS} ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(run_experimental_calib PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(run_experimental_calib PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES} ${TURBOJPEG_LIBRARIES}
        "-framework Metal" "-framework Foundation" "-framework CoreGraphics"
        "-framework CoreVideo" "-framework CoreMedia" "-framework VideoToolbox"
        "-framework CoreML" "-framework Accelerate" "-framework IOKit" pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm src/vt_async_decoder.mm src/superpoint_coreml.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(run_experimental_calib PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    # Calibration evaluation tool (global consistency + per-board reproj)
    add_executable(eval_calibration
        tests/eval_calibration.cpp
        src/aruco_metal.mm src/FFmpegDemuxer.cpp src/vt_async_decoder.mm src/superpoint_coreml.mm
    )
    target_include_directories(eval_calibration PRIVATE
        src lib/nvcodec ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS} ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(eval_calibration PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(eval_calibration PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES} ${TURBOJPEG_LIBRARIES}
        "-framework Metal" "-framework Foundation" "-framework CoreGraphics"
        "-framework CoreVideo" "-framework CoreMedia" "-framework VideoToolbox"
        "-framework CoreML" "-framework Accelerate" "-framework IOKit" pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm src/vt_async_decoder.mm src/superpoint_coreml.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(eval_calibration PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    # Holdout cross-validation comparison test
    add_executable(test_holdout_compare
        tests/test_holdout_compare.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
    )
    target_include_directories(test_holdout_compare PRIVATE
        src lib/nvcodec
        ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS} ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_holdout_compare PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_holdout_compare PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES} ${TURBOJPEG_LIBRARIES}
        "-framework Metal" "-framework Foundation" "-framework CoreGraphics"
        "-framework CoreVideo" "-framework CoreMedia" "-framework VideoToolbox"
        "-framework CoreML" "-framework Accelerate" "-framework IOKit" pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm src/vt_async_decoder.mm src/superpoint_coreml.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_holdout_compare PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    # Feature-based calibration refinement test
    add_executable(test_feature_refinement
        tests/test_feature_refinement.cpp
        src/aruco_metal.mm
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
    )
    target_include_directories(test_feature_refinement PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_feature_refinement PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_feature_refinement PRIVATE
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
        "-framework CoreML"
        "-framework Accelerate"
        "-framework IOKit"
        pthread
    )
    set_source_files_properties(
        src/aruco_metal.mm
        src/vt_async_decoder.mm
        src/superpoint_coreml.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_feature_refinement PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # DICT_5X5 detection test (rat calibration board)
    add_executable(test_dict5x5
        tests/test_dict5x5.cpp
        src/aruco_metal.mm
    )
    target_include_directories(test_dict5x5 PRIVATE
        src lib
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_dict5x5 PRIVATE ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_dict5x5 PRIVATE
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
    set_target_properties(test_dict5x5 PROPERTIES
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

    # Cross-validate laser modes against aruco detections
    add_executable(test_crossval_modes /Users/johnsonr/red_dev/quan_v_rob_calib/test_crossval_modes.cpp)
    target_include_directories(test_crossval_modes PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )
    target_link_libraries(test_crossval_modes PRIVATE Eigen3::Eigen)
    set_target_properties(test_crossval_modes PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Laser refinement mode comparison test
    add_executable(test_laser_modes
        /Users/johnsonr/red_dev/quan_v_rob_calib/test_laser_modes.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_laser_modes PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_laser_modes PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_laser_modes PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework Foundation"
    )
    set_target_properties(test_laser_modes PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    add_executable(test_detection_diag
        /Users/johnsonr/red_dev/Mar22/test_detection_diag.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_detection_diag PRIVATE
        src ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3 ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(test_detection_diag PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_detection_diag PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox" "-framework CoreMedia" "-framework CoreVideo"
        "-framework CoreFoundation" "-framework Metal" "-framework Foundation")
    set_target_properties(test_detection_diag PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    add_executable(test_no_init
        /Users/johnsonr/red_dev/Mar22/test_no_init.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_no_init PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_no_init PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_no_init PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework Foundation"
    )
    set_target_properties(test_no_init PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    add_executable(test_loose_init
        /Users/johnsonr/red_dev/Mar22/test_loose_init.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_loose_init PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_loose_init PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_loose_init PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework Foundation"
    )
    set_target_properties(test_loose_init PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # JARVIS inference pipeline benchmark (ONNX Runtime only, no GUI deps)
    if(HAS_ONNXRUNTIME)
        add_executable(bench_jarvis tests/bench_jarvis.cpp)
        target_include_directories(bench_jarvis PRIVATE
            src
            ${DIR_FILEBROWSER}
            "${ONNXRUNTIME_DIR}/include"
            ${HOMEBREW_PREFIX}/include
            ${HOMEBREW_PREFIX}/include/eigen3
        )
        target_compile_definitions(bench_jarvis PRIVATE RED_HAS_ONNXRUNTIME)
        target_link_libraries(bench_jarvis PRIVATE
            "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
        )
        set_target_properties(bench_jarvis PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()

    # CoreML vs ONNX accuracy comparison benchmark
    if(HAS_ONNXRUNTIME)
        add_executable(bench_coreml_accuracy tests/bench_coreml_accuracy.mm)
        target_include_directories(bench_coreml_accuracy PRIVATE
            src
            ${DIR_FILEBROWSER}
            "${ONNXRUNTIME_DIR}/include"
            ${HOMEBREW_PREFIX}/include
            ${HOMEBREW_PREFIX}/include/eigen3
        )
        target_compile_definitions(bench_coreml_accuracy PRIVATE RED_HAS_ONNXRUNTIME)
        target_link_libraries(bench_coreml_accuracy PRIVATE
            "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
            "-framework CoreML"
            "-framework CoreVideo"
            "-framework Accelerate"
            "-framework Foundation"
        )
        set_target_properties(bench_coreml_accuracy PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
        set_source_files_properties(
            tests/bench_coreml_accuracy.mm
            PROPERTIES COMPILE_FLAGS "-fobjc-arc"
        )
    endif()

elseif(WIN32)
    # Windows build with NVIDIA CUDA support (RTX 3090)
    project(red LANGUAGES CXX CUDA)

    # CUDA architecture: 86 for RTX 3090 (Ampere)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # vcpkg integration
    set(VCPKG_ROOT "$ENV{USERPROFILE}/vcpkg")
    if(EXISTS "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    endif()

    # Find packages via vcpkg
    find_package(Eigen3 REQUIRED)
    find_package(Ceres REQUIRED)

    # FFmpeg: find via vcpkg pkg-config or manual paths
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFMPEG IMPORTED_TARGET libavcodec libavformat libavutil libswscale)
    endif()
    if(NOT FFMPEG_FOUND)
        # Fallback: use vcpkg installed paths directly
        find_path(FFMPEG_INCLUDE_DIR libavcodec/avcodec.h)
        find_library(AVCODEC_LIB avcodec)
        find_library(AVFORMAT_LIB avformat)
        find_library(AVUTIL_LIB avutil)
        find_library(SWSCALE_LIB swscale)
        set(FFMPEG_LIBRARIES ${AVCODEC_LIB} ${AVFORMAT_LIB} ${AVUTIL_LIB} ${SWSCALE_LIB})
        set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})
    endif()

    # libjpeg-turbo
    find_package(libjpeg-turbo CONFIG QUIET)
    if(NOT libjpeg-turbo_FOUND)
        find_library(TURBOJPEG_LIB turbojpeg)
        find_path(TURBOJPEG_INCLUDE turbojpeg.h)
    endif()

    # CUDA paths
    set(CUDA_DIR "$ENV{CUDA_PATH}")
    if(NOT CUDA_DIR)
        set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
    endif()

    # Bundled Windows libraries
    set(DIR_GL "${CMAKE_SOURCE_DIR}/lib/GL")
    set(DIR_GLFW "${CMAKE_SOURCE_DIR}/lib/GLFW")
    set(DIR_NVCODEC "${CMAKE_SOURCE_DIR}/lib/nvcodec")

    # ImGui / ImPlot / etc
    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
    set(DIR_IMPLOT3D "${CMAKE_SOURCE_DIR}/lib/implot3d")
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")
    set(DIR_ICONFONT "${CMAKE_SOURCE_DIR}/lib/IconFontCppHeaders")

    # Source files — use the macOS source list (most complete, no legacy OpenCV deps)
    file(GLOB SRC_FILES src/*.cpp)

    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
        ${DIR_IMGUI}/backends/imgui_impl_opengl3.cpp
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)
    file(GLOB IMPLOT3D_SRC ${DIR_IMPLOT3D}/*.cpp)
    list(REMOVE_ITEM IMPLOT3D_SRC "${DIR_IMPLOT3D}/implot3d_demo.cpp")

    # CUDA source files
    set(CUDA_SRC_FILES
        src/create_image_cuda.cu
        src/ColorSpace.cu
        src/kernel.cu
        src/aruco_cuda.cu
        src/pointsource_cuda.cu
    )

    # Main executable
    add_executable(red
        ${SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
        ${CUDA_SRC_FILES}
    )

    # ONNX Runtime (GPU build with CUDA execution provider)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")
    if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        set(HAS_ONNXRUNTIME TRUE)
    else()
        set(HAS_ONNXRUNTIME FALSE)
        message(WARNING "ONNX Runtime not found in lib/onnxruntime/ — JARVIS prediction disabled")
    endif()

    target_compile_definitions(red PRIVATE NOMINMAX _USE_MATH_DEFINES USE_CUDA_POINTSOURCE)
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(red PRIVATE RED_HAS_ONNXRUNTIME)
    endif()
    target_compile_options(red PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj /EHsc>)

    # Use vcpkg GLFW and GLEW (version-matched headers + DLLs)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(GLEW REQUIRED)

    target_include_directories(red PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_IMPLOT3D}
        ${DIR_FILEBROWSER}
        ${DIR_ICONFONT}
        ${DIR_NVCODEC}
        ${CUDA_DIR}/include
        ${FFMPEG_INCLUDE_DIRS}
    )
    if(TURBOJPEG_INCLUDE)
        target_include_directories(red PRIVATE ${TURBOJPEG_INCLUDE})
    endif()
    if(HAS_ONNXRUNTIME)
        target_include_directories(red PRIVATE "${ONNXRUNTIME_DIR}/include")
    endif()

    # Link libraries
    target_link_libraries(red PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        # vcpkg GLFW 3.4 + GLEW
        glfw
        GLEW::GLEW
        # System OpenGL
        opengl32
        # CUDA
        "${CUDA_DIR}/lib/x64/cudart.lib"
        "${CUDA_DIR}/lib/x64/cuda.lib"
        "${CUDA_DIR}/lib/x64/nppc.lib"
        "${CUDA_DIR}/lib/x64/nppicc.lib"
        "${CUDA_DIR}/lib/x64/nppidei.lib"
        "${CUDA_DIR}/lib/x64/nppig.lib"
        "${CUDA_DIR}/lib/x64/nppial.lib"
        # Bundled NVDEC
        "${DIR_NVCODEC}/x64/nvcuvid.lib"
        "${DIR_NVCODEC}/x64/nvencodeapi.lib"
    )

    # FFmpeg libraries
    if(FFMPEG_FOUND AND TARGET PkgConfig::FFMPEG)
        target_link_libraries(red PRIVATE PkgConfig::FFMPEG)
    else()
        target_link_libraries(red PRIVATE ${FFMPEG_LIBRARIES})
    endif()

    # libjpeg-turbo
    if(libjpeg-turbo_FOUND)
        target_link_libraries(red PRIVATE libjpeg-turbo::turbojpeg)
    elseif(TURBOJPEG_LIB)
        target_link_libraries(red PRIVATE ${TURBOJPEG_LIB})
    endif()

    # ONNX Runtime (GPU with CUDA EP)
    if(HAS_ONNXRUNTIME)
        target_link_libraries(red PRIVATE "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    endif()

    # Windows system libraries
    target_link_libraries(red PRIVATE
        shell32
        ole32
    )

    # Copy ONNX Runtime DLLs to build directory
    if(HAS_ONNXRUNTIME)
        file(GLOB ORT_DLLS "${ONNXRUNTIME_DIR}/lib/*.dll")
        foreach(dll ${ORT_DLLS})
            add_custom_command(TARGET red POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:red>")
        endforeach()
    endif()

    # vcpkg automatically copies dependent DLLs via its toolchain

    set_target_properties(red PROPERTIES
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
