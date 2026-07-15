cmake_minimum_required(VERSION 3.18...3.31)

# Default to an optimized build. Without this the build type is empty and the
# compiler runs at -O0, which cripples hot host loops (e.g. the HybridNet
# reprojection: 3200ms at -O0 vs ~450ms at -O3). Respects an explicit
# -DCMAKE_BUILD_TYPE=... and multi-config generators.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug/Release/RelWithDebInfo/MinSizeRel)" FORCE)
endif()

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

    # Optional MuJoCo for body model IK (use official macOS framework release)
    set(MUJOCO_FRAMEWORK_DIR "${CMAKE_SOURCE_DIR}/lib/mujoco.framework")
    set(MUJOCO_DYLIB_VERSION "3.6.0")
    set(MUJOCO_DYLIB "${MUJOCO_FRAMEWORK_DIR}/Versions/A/libmujoco.${MUJOCO_DYLIB_VERSION}.dylib")
    if(EXISTS "${MUJOCO_FRAMEWORK_DIR}/Headers/mujoco.h")
        set(HAS_MUJOCO TRUE)
        message(STATUS "MuJoCo framework found at ${MUJOCO_FRAMEWORK_DIR}")
    else()
        set(HAS_MUJOCO FALSE)
        message(STATUS "MuJoCo not found — body model IK disabled")
    endif()

    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
    set(DIR_IMPLOT3D "${CMAKE_SOURCE_DIR}/lib/implot3d")

    # Use 32-bit vertex indices for ImGui/ImPlot3D to support large meshes.
    # Default 16-bit indices cap at 65K vertices — with circle markers (~50
    # verts each), this silently drops 3D points after ~1,300 points.
    # Applied via compile definition so we don't modify the imgui submodule.
    add_definitions(-DImDrawIdx=unsigned\ int)
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")

    file(GLOB SRC_FILES src/*.cpp)
    list(REMOVE_ITEM SRC_FILES
        "${CMAKE_SOURCE_DIR}/src/NvDecoder.cpp"
    )

    # Objective-C++ source files (.mm)
    file(GLOB MM_SRC_FILES src/*.mm)

    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        ${CMAKE_SOURCE_DIR}/src/imgui_impl_glfw_patched.cpp  # patched: macOS modifier key fix
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

    # Optional MuJoCo for body model IK
    if(HAS_MUJOCO)
        target_compile_definitions(red PRIVATE RED_HAS_MUJOCO)
        target_include_directories(red PRIVATE "${MUJOCO_FRAMEWORK_DIR}/Headers")
        target_link_libraries(red PRIVATE "${MUJOCO_DYLIB}")
        # Set rpath so the binary finds the MuJoCo dylib at runtime
        set_target_properties(red PROPERTIES BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")
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

    if(HAS_MUJOCO)
        target_compile_definitions(test_gui PRIVATE RED_HAS_MUJOCO)
        target_include_directories(test_gui PRIVATE "${MUJOCO_FRAMEWORK_DIR}/Headers")
        target_link_libraries(test_gui PRIVATE "${MUJOCO_DYLIB}")
        set_target_properties(test_gui PROPERTIES BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")
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
    if(HAS_MUJOCO)
        target_compile_definitions(test_annotation PRIVATE RED_HAS_MUJOCO)
        target_include_directories(test_annotation PRIVATE "${MUJOCO_FRAMEWORK_DIR}/Headers")
        target_link_libraries(test_annotation PRIVATE "${MUJOCO_DYLIB}")
        set_target_properties(test_annotation PROPERTIES BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")
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

    # Sync-plan unit test: header-only logic, stdlib + vendored json.hpp only
    add_executable(test_sync_plan tests/test_sync_plan.cpp)
    target_include_directories(test_sync_plan PRIVATE src)
    set_target_properties(test_sync_plan PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # MuJoCo IK integration tests
    if(HAS_MUJOCO)
        # Unit/scale diagnostics
        add_executable(test_mujoco_units
            tests/test_mujoco_units.cpp
            src/skeleton.cpp
            ${DIR_IMGUI}/imgui.cpp ${DIR_IMGUI}/imgui_draw.cpp
            ${DIR_IMGUI}/imgui_tables.cpp ${DIR_IMGUI}/imgui_widgets.cpp
        )
        target_include_directories(test_mujoco_units PRIVATE
            src ${DIR_IMGUI} ${DIR_IMGUI}/backends ${DIR_IMPLOT} ${DIR_IMPLOT3D}
            lib/nvcodec lib/IconFontCppHeaders
            ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3
            ${FFMPEG_INCLUDE_DIRS} "${MUJOCO_FRAMEWORK_DIR}/Headers"
        )
        target_link_libraries(test_mujoco_units PRIVATE Eigen3::Eigen
            "${MUJOCO_DYLIB}")
        target_compile_definitions(test_mujoco_units PRIVATE RED_HAS_MUJOCO)
        set_target_properties(test_mujoco_units PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")

        # Geometry integrity test (joints don't break after IK)
        add_executable(test_mujoco_geometry
            tests/test_mujoco_geometry.cpp
            src/skeleton.cpp
            ${DIR_IMGUI}/imgui.cpp ${DIR_IMGUI}/imgui_draw.cpp
            ${DIR_IMGUI}/imgui_tables.cpp ${DIR_IMGUI}/imgui_widgets.cpp
        )
        target_include_directories(test_mujoco_geometry PRIVATE
            src ${DIR_IMGUI} ${DIR_IMGUI}/backends ${DIR_IMPLOT} ${DIR_IMPLOT3D}
            lib/nvcodec lib/IconFontCppHeaders
            ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3
            ${FFMPEG_INCLUDE_DIRS} "${MUJOCO_FRAMEWORK_DIR}/Headers"
        )
        target_link_libraries(test_mujoco_geometry PRIVATE Eigen3::Eigen
            "${MUJOCO_DYLIB}")
        target_compile_definitions(test_mujoco_geometry PRIVATE RED_HAS_MUJOCO)
        set_target_properties(test_mujoco_geometry PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")

        # Joint connectivity test
        add_executable(test_mujoco_joints
            tests/test_mujoco_joints.cpp
            src/skeleton.cpp
            ${DIR_IMGUI}/imgui.cpp ${DIR_IMGUI}/imgui_draw.cpp
            ${DIR_IMGUI}/imgui_tables.cpp ${DIR_IMGUI}/imgui_widgets.cpp
        )
        target_include_directories(test_mujoco_joints PRIVATE
            src ${DIR_IMGUI} ${DIR_IMGUI}/backends ${DIR_IMPLOT} ${DIR_IMPLOT3D}
            lib/nvcodec lib/IconFontCppHeaders
            ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3
            ${FFMPEG_INCLUDE_DIRS} "${MUJOCO_FRAMEWORK_DIR}/Headers"
        )
        target_link_libraries(test_mujoco_joints PRIVATE Eigen3::Eigen
            "${MUJOCO_DYLIB}")
        target_compile_definitions(test_mujoco_joints PRIVATE RED_HAS_MUJOCO)
        set_target_properties(test_mujoco_joints PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")

        # Full IK test suite
        add_executable(test_mujoco_ik
            tests/test_mujoco_ik.cpp
            src/skeleton.cpp
            ${DIR_IMGUI}/imgui.cpp
            ${DIR_IMGUI}/imgui_draw.cpp
            ${DIR_IMGUI}/imgui_tables.cpp
            ${DIR_IMGUI}/imgui_widgets.cpp
        )
        target_include_directories(test_mujoco_ik PRIVATE
            src
            ${DIR_IMGUI}
            ${DIR_IMGUI}/backends
            ${DIR_IMPLOT}
            ${DIR_IMPLOT3D}
            lib/nvcodec
            lib/IconFontCppHeaders
            ${HOMEBREW_PREFIX}/include
            ${HOMEBREW_PREFIX}/include/eigen3
            ${FFMPEG_INCLUDE_DIRS}
            "${MUJOCO_FRAMEWORK_DIR}/Headers"
        )
        target_link_libraries(test_mujoco_ik PRIVATE
            Eigen3::Eigen
            "${MUJOCO_DYLIB}"
        )
        target_compile_definitions(test_mujoco_ik PRIVATE RED_HAS_MUJOCO)
        set_target_properties(test_mujoco_ik PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
        )

        # STAC comparison test: IK_dm_control vs IK+STAC
        add_executable(test_stac_comparison
            tests/test_stac_comparison.cpp
            src/skeleton.cpp
            ${DIR_IMGUI}/imgui.cpp
            ${DIR_IMGUI}/imgui_draw.cpp
            ${DIR_IMGUI}/imgui_tables.cpp
            ${DIR_IMGUI}/imgui_widgets.cpp
        )
        target_include_directories(test_stac_comparison PRIVATE
            src
            ${DIR_IMGUI}
            ${DIR_IMGUI}/backends
            ${DIR_IMPLOT}
            ${DIR_IMPLOT3D}
            lib/nvcodec
            lib/IconFontCppHeaders
            ${HOMEBREW_PREFIX}/include
            ${HOMEBREW_PREFIX}/include/eigen3
            ${FFMPEG_INCLUDE_DIRS}
            "${MUJOCO_FRAMEWORK_DIR}/Headers"
        )
        target_link_libraries(test_stac_comparison PRIVATE
            Eigen3::Eigen
            "${MUJOCO_DYLIB}"
        )
        target_compile_definitions(test_stac_comparison PRIVATE RED_HAS_MUJOCO)
        set_target_properties(test_stac_comparison PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
        )
    endif()

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

    # CPU (hn_reproject) vs Metal (HNMetalReproject) reprojection parity.
    # Enforces the "bit-exact" claim shared by jarvis_hn_reproject.h and
    # jarvis_hn_metal.mm. Metal-only, so Apple-block only.
    add_executable(test_jarvis_hn_parity
        tests/test_jarvis_hn_parity.mm
        src/jarvis_hn_metal.mm
    )
    target_include_directories(test_jarvis_hn_parity PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_jarvis_hn_parity PRIVATE
        "-framework Metal" "-framework Foundation" "-framework CoreGraphics"
    )
    set_target_properties(test_jarvis_hn_parity PROPERTIES
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

    # Nerfstudio / 3DGS export test: extracts frames + writes transforms.json
    add_executable(test_nerfstudio_export
        tests/test_nerfstudio_export.cpp
        src/FFmpegDemuxer.cpp
        src/vt_async_decoder.mm
    )
    target_include_directories(test_nerfstudio_export PRIVATE
        src
        lib/nvcodec
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_nerfstudio_export PRIVATE
        ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_nerfstudio_export PRIVATE
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
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    set_target_properties(test_nerfstudio_export PROPERTIES
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

    add_executable(test_smart_blob_v2
        /Users/johnsonr/red_dev/Mar25_fulltest/test_smart_blob_v2.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_smart_blob_v2 PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_smart_blob_v2 PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_smart_blob_v2 PRIVATE
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
    set_target_properties(test_smart_blob_v2 PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    add_executable(test_all_opt_modes
        /Users/johnsonr/red_dev/Mar25_fulltest/test_all_opt_modes.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_all_opt_modes PRIVATE
        src
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
        ${FFMPEG_INCLUDE_DIRS}
    )
    target_link_directories(test_all_opt_modes PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_all_opt_modes PRIVATE
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
    set_target_properties(test_all_opt_modes PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    add_executable(test_ps_scratch_v3
        /Users/johnsonr/red_dev/Mar25_fulltest/test_ps_scratch_v3.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_ps_scratch_v3 PRIVATE
        src ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3 ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(test_ps_scratch_v3 PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_ps_scratch_v3 PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox" "-framework CoreMedia" "-framework CoreVideo"
        "-framework CoreFoundation" "-framework Metal" "-framework Foundation")
    set_target_properties(test_ps_scratch_v3 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    add_executable(test_ps_sweep
        /Users/johnsonr/red_dev/Mar25_fulltest/test_ps_sweep.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_ps_sweep PRIVATE
        src ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3 ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(test_ps_sweep PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_ps_sweep PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox" "-framework CoreMedia" "-framework CoreVideo"
        "-framework CoreFoundation" "-framework Metal" "-framework Foundation")
    set_target_properties(test_ps_sweep PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    add_executable(test_no_init_ui
        /Users/johnsonr/red_dev/Mar24/test_no_init_ui.cpp
        src/pointsource_metal.mm
        src/vt_async_decoder.mm
        src/FFmpegDemuxer.cpp)
    target_include_directories(test_no_init_ui PRIVATE
        src ${HOMEBREW_PREFIX}/include ${HOMEBREW_PREFIX}/include/eigen3 ${FFMPEG_INCLUDE_DIRS})
    target_link_directories(test_no_init_ui PRIVATE ${FFMPEG_LIBRARY_DIRS})
    target_link_libraries(test_no_init_ui PRIVATE
        Eigen3::Eigen Ceres::ceres ${FFMPEG_LIBRARIES}
        "-L/opt/homebrew/opt/jpeg-turbo/lib" "-lturbojpeg"
        "-framework VideoToolbox" "-framework CoreMedia" "-framework CoreVideo"
        "-framework CoreFoundation" "-framework Metal" "-framework Foundation")
    set_target_properties(test_no_init_ui PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

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
        set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
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
        # Use patched GLFW backend from src/ (picked up by SRC_FILES glob)
        # instead of stock imgui_impl_glfw.cpp — has Windows + macOS fixes
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
        src/jarvis_hybridnet_cuda.cu
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

    # Optional MuJoCo for body model IK (Windows: DLL release in lib/mujoco/)
    # Supports both layouts: include/mujoco/mujoco.h (official) and include/mujoco.h (flat)
    set(MUJOCO_DIR "${CMAKE_SOURCE_DIR}/lib/mujoco")
    if(EXISTS "${MUJOCO_DIR}/include/mujoco/mujoco.h")
        set(HAS_MUJOCO TRUE)
        set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include/mujoco")
        message(STATUS "MuJoCo found at ${MUJOCO_DIR} (standard layout)")
    elseif(EXISTS "${MUJOCO_DIR}/include/mujoco.h")
        set(HAS_MUJOCO TRUE)
        set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include")
        message(STATUS "MuJoCo found at ${MUJOCO_DIR} (flat layout)")
    else()
        set(HAS_MUJOCO FALSE)
        message(STATUS "MuJoCo not found in lib/mujoco/ — body model IK disabled")
    endif()

    target_compile_definitions(red PRIVATE NOMINMAX _USE_MATH_DEFINES USE_CUDA_POINTSOURCE)
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(red PRIVATE RED_HAS_ONNXRUNTIME)
    endif()
    if(HAS_MUJOCO)
        target_compile_definitions(red PRIVATE RED_HAS_MUJOCO)
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
    if(HAS_MUJOCO)
        target_include_directories(red PRIVATE "${MUJOCO_DIR}/include" "${MUJOCO_DIR}/include/mujoco")
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

    # MuJoCo body model IK
    if(HAS_MUJOCO)
        target_link_libraries(red PRIVATE "${MUJOCO_DIR}/lib/mujoco.lib")
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

    # Copy cuDNN DLLs to build directory (needed by ONNX Runtime CUDA EP)
    set(CUDNN_DIR "${CMAKE_SOURCE_DIR}/lib/cudnn")
    if(EXISTS "${CUDNN_DIR}/bin/cudnn64_9.dll")
        file(GLOB CUDNN_DLLS "${CUDNN_DIR}/bin/*.dll")
        foreach(dll ${CUDNN_DLLS})
            add_custom_command(TARGET red POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:red>")
        endforeach()
    endif()

    # Copy MuJoCo DLL to build directory
    if(HAS_MUJOCO)
        file(GLOB MUJOCO_DLLS "${MUJOCO_DIR}/bin/*.dll")
        foreach(dll ${MUJOCO_DLLS})
            add_custom_command(TARGET red POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:red>")
        endforeach()
    endif()

    # vcpkg automatically copies dependent DLLs via its toolchain

    set_target_properties(red PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # ── Test targets (same deps as red, replacing red.cpp) ──────────────────
    set(TEST_GUI_SRC_FILES ${SRC_FILES})
    list(REMOVE_ITEM TEST_GUI_SRC_FILES "${CMAKE_SOURCE_DIR}/src/red.cpp")

    foreach(TEST_NAME test_gui test_annotation test_jarvis_hybridnet_smoke)
        add_executable(${TEST_NAME}
            ${TEST_GUI_SRC_FILES}
            ${IMGUI_SRC}
            ${IMPLOT_SRC}
            ${IMPLOT3D_SRC}
            ${CUDA_SRC_FILES}
            tests/${TEST_NAME}.cpp
        )
        target_compile_definitions(${TEST_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES USE_CUDA_POINTSOURCE)
        target_compile_options(${TEST_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj /EHsc>)
        target_include_directories(${TEST_NAME} PRIVATE
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
            target_include_directories(${TEST_NAME} PRIVATE ${TURBOJPEG_INCLUDE})
        endif()
        target_link_libraries(${TEST_NAME} PRIVATE
            Eigen3::Eigen
            Ceres::ceres
            glfw
            GLEW::GLEW
            opengl32
            "${CUDA_DIR}/lib/x64/cudart.lib"
            "${CUDA_DIR}/lib/x64/cuda.lib"
            "${CUDA_DIR}/lib/x64/nppc.lib"
            "${CUDA_DIR}/lib/x64/nppicc.lib"
            "${CUDA_DIR}/lib/x64/nppidei.lib"
            "${CUDA_DIR}/lib/x64/nppig.lib"
            "${CUDA_DIR}/lib/x64/nppial.lib"
            "${DIR_NVCODEC}/x64/nvcuvid.lib"
            "${DIR_NVCODEC}/x64/nvencodeapi.lib"
            shell32
            ole32
        )
        if(FFMPEG_FOUND AND TARGET PkgConfig::FFMPEG)
            target_link_libraries(${TEST_NAME} PRIVATE PkgConfig::FFMPEG)
        else()
            target_link_libraries(${TEST_NAME} PRIVATE ${FFMPEG_LIBRARIES})
        endif()
        if(libjpeg-turbo_FOUND)
            target_link_libraries(${TEST_NAME} PRIVATE libjpeg-turbo::turbojpeg)
        elseif(TURBOJPEG_LIB)
            target_link_libraries(${TEST_NAME} PRIVATE ${TURBOJPEG_LIB})
        endif()
        if(HAS_ONNXRUNTIME)
            target_compile_definitions(${TEST_NAME} PRIVATE RED_HAS_ONNXRUNTIME)
            target_include_directories(${TEST_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")
            target_link_libraries(${TEST_NAME} PRIVATE "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
        endif()
        if(HAS_MUJOCO)
            target_compile_definitions(${TEST_NAME} PRIVATE RED_HAS_MUJOCO)
            target_include_directories(${TEST_NAME} PRIVATE "${MUJOCO_DIR}/include" "${MUJOCO_DIR}/include/mujoco")
            target_link_libraries(${TEST_NAME} PRIVATE "${MUJOCO_DIR}/lib/mujoco.lib")
        endif()
        set_target_properties(${TEST_NAME} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endforeach()

    # Windows default 1MB stack is too small for MuJoCo XML parsing
    # (complex rodent/fly models) and STB image static initializers.
    # Use 64MB stack for all targets.
    target_link_options(red PRIVATE /STACK:67108864)
    target_link_options(test_gui PRIVATE /STACK:67108864)
    target_link_options(test_annotation PRIVATE /STACK:67108864)

else()
    # ────────────────────────────────────────────────────────────────────────
    # Linux build with NVIDIA CUDA + NVDEC
    #
    # Coexists with the orange app (same machine): uses the same CUDA 12.2
    # at /usr/local/cuda, the same custom FFmpeg at $HOME/nvidia/ffmpeg, and
    # the same driver 535.x. Orange's toolchain is UNTOUCHED — red just
    # links read-only against those shared installs. ML inference libs
    # (ONNX Runtime, cuDNN) are bundled in red/lib/ with RPATH isolation so
    # they cannot collide with orange's TensorRT 8.6 / cuDNN 8.9. The JARVIS
    # HybridNet path links read-only against orange's shared TRT install at
    # $HOME/nvidia/TensorRT-8.6.1.6 (build-tree RPATH only — never bundled).
    # No OpenCV, no LibTorch.
    # ────────────────────────────────────────────────────────────────────────
    project(red LANGUAGES CXX CUDA)

    # CUDA arch: 86 = A16 (Ampere), 89 = RTX 4000 Ada. Override with
    # -DCMAKE_CUDA_ARCHITECTURES=... on the cmake command line if needed.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 89)
    endif()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # System packages (apt / distro)
    find_package(Eigen3 REQUIRED)
    # Find the GTest install that ships GMock (/usr/local) before Ceres —
    # Ceres → absl's config requires the GTest::gmock imported target.
    # The apt libgtest-dev at /usr/lib/x86_64-linux-gnu ships gtest only.
    find_package(GTest REQUIRED CONFIG
        HINTS /usr/local/lib/cmake/GTest
        NO_CMAKE_SYSTEM_PATH)
    find_package(Ceres REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(Threads REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    find_package(PkgConfig REQUIRED)

    # CBLAS provider (used by descriptor_matcher.h). Ubuntu 22.04 ships a literal
    # libcblas.so (netlib/atlas); 24.04 dropped that name but libopenblas.so
    # exports the full cblas_* API. Find whichever exists instead of hardcoding
    # -lcblas, so the build works on both the 22.04 reference box and 24.04.
    find_library(RED_CBLAS_LIB NAMES cblas openblas blas REQUIRED)
    message(STATUS "Using CBLAS provider: ${RED_CBLAS_LIB}")

    # FFmpeg via pkg-config — prefer the custom CUDA-enabled build at
    # $HOME/nvidia/ffmpeg (shared with orange) over any system FFmpeg.
    set(RED_FFMPEG_PKGCONFIG "$ENV{HOME}/nvidia/ffmpeg/build/lib/pkgconfig")
    if(EXISTS "${RED_FFMPEG_PKGCONFIG}")
        set(ENV{PKG_CONFIG_PATH} "${RED_FFMPEG_PKGCONFIG}:$ENV{PKG_CONFIG_PATH}")
        message(STATUS "Using custom FFmpeg from ${RED_FFMPEG_PKGCONFIG}")
    endif()
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavcodec libavformat libavutil libswscale)

    # Bundled third-party sources
    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")
    set(DIR_IMPLOT3D "${CMAKE_SOURCE_DIR}/lib/implot3d")
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")
    set(DIR_ICONFONT "${CMAKE_SOURCE_DIR}/lib/IconFontCppHeaders")
    set(DIR_NVCODEC "${CMAKE_SOURCE_DIR}/lib/nvcodec")

    # Source files (mirrors Windows — uses patched GLFW backend from src/)
    file(GLOB SRC_FILES src/*.cpp)

    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        # Patched GLFW backend picked up via src/*.cpp glob
        ${DIR_IMGUI}/backends/imgui_impl_opengl3.cpp
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)
    file(GLOB IMPLOT3D_SRC ${DIR_IMPLOT3D}/*.cpp)
    list(REMOVE_ITEM IMPLOT3D_SRC "${DIR_IMPLOT3D}/implot3d_demo.cpp")

    set(CUDA_SRC_FILES
        src/create_image_cuda.cu
        src/ColorSpace.cu
        src/kernel.cu
        src/aruco_cuda.cu
        src/pointsource_cuda.cu
        src/jarvis_hybridnet_cuda.cu
    )

    add_executable(red
        ${SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${IMPLOT3D_SRC}
        ${CUDA_SRC_FILES}
    )

    # Optional ONNX Runtime (GPU, CUDA EP) bundled at lib/onnxruntime/
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime")
    if(EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
        set(HAS_ONNXRUNTIME TRUE)
        message(STATUS "ONNX Runtime found at ${ONNXRUNTIME_DIR}")
    else()
        set(HAS_ONNXRUNTIME FALSE)
        message(STATUS "ONNX Runtime not found in lib/onnxruntime/ — JARVIS prediction disabled")
    endif()

    # Optional TensorRT for JARVIS HybridNet direct-runtime inference.
    # Engine files for the loaded JARVIS model are compiled offline via
    # scripts/compile_tensorrt_engines.sh. At runtime, jarvis_hybridnet.h
    # uses TRT when .engine files are present and falls back to the ORT CUDA EP.
    #
    # Two install layouts are supported (tried in this order):
    #   1. Self-contained tarball — the 22.04 reference box: TRT 8.6.1.6 under
    #      $HOME/nvidia/TensorRT-8.6.1.6 (read-only, orange-shared, never
    #      bundled/modified), or any dir passed via -DTENSORRT_DIR=. Layout:
    #      <dir>/include/NvInfer.h + <dir>/lib/libnvinfer.so.
    #   2. System / apt install — the 24.04 / CUDA-13 boxes (e.g. flyrig): TRT
    #      10 from the NVIDIA CUDA apt repo, headers in /usr/include/<triplet>,
    #      libs in /usr/lib/<triplet>. Found via find_path/find_library.
    # red only loads prebuilt .engine files (no ONNX parsing, no builder at
    # runtime) and uses the modern name-based API (createInferRuntime,
    # deserializeCudaEngine, enqueueV3, setTensorAddress, setInputShape,
    # getTensorShape, getIOTensorName) shared across TRT 8.5–11 — none of the
    # binding-index APIs TRT 10 removed — so either TRT major works unchanged.
    set(TENSORRT_DIR "$ENV{HOME}/nvidia/TensorRT-8.6.1.6" CACHE PATH
        "TensorRT tarball root; tried before the system/apt install")
    set(HAS_TENSORRT_HN FALSE)
    if(EXISTS "${TENSORRT_DIR}/include/NvInfer.h" AND
       EXISTS "${TENSORRT_DIR}/lib/libnvinfer.so")
        # Layout 1: self-contained tarball.
        set(HAS_TENSORRT_HN TRUE)
        set(TENSORRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
        set(TENSORRT_LIB         "${TENSORRT_DIR}/lib/libnvinfer.so")
        set(TENSORRT_PLUGIN_LIB  "${TENSORRT_DIR}/lib/libnvinfer_plugin.so")
        set(TENSORRT_LIB_DIR     "${TENSORRT_DIR}/lib")
        message(STATUS "TensorRT found at ${TENSORRT_DIR} (tarball; HybridNet direct runtime enabled)")
    else()
        # Layout 2: system / apt (e.g. TRT 10 from the CUDA repo).
        find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h PATH_SUFFIXES x86_64-linux-gnu)
        find_library(TENSORRT_LIB NAMES nvinfer)
        find_library(TENSORRT_PLUGIN_LIB NAMES nvinfer_plugin)
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIB AND TENSORRT_PLUGIN_LIB)
            set(HAS_TENSORRT_HN TRUE)
            get_filename_component(TENSORRT_LIB_DIR "${TENSORRT_LIB}" DIRECTORY)
            message(STATUS "TensorRT found in system paths: ${TENSORRT_LIB} "
                           "(HybridNet direct runtime enabled)")
        else()
            message(STATUS "TensorRT not found (tarball ${TENSORRT_DIR} or system paths) "
                           "— JARVIS HybridNet will use ORT CUDA EP only")
        endif()
    endif()
    # Report the detected TRT major version (informational — the app is
    # version-agnostic across the modern API described above).
    if(HAS_TENSORRT_HN AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
        file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_ver_line
             REGEX "define +NV_TENSORRT_MAJOR")
        message(STATUS "  ${_trt_ver_line}")
    endif()

    # Optional MuJoCo for body-model IK (bundled at lib/mujoco/)
    set(MUJOCO_DIR "${CMAKE_SOURCE_DIR}/lib/mujoco")
    if(EXISTS "${MUJOCO_DIR}/include/mujoco/mujoco.h")
        set(HAS_MUJOCO TRUE)
        set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include/mujoco")
        message(STATUS "MuJoCo found at ${MUJOCO_DIR} (standard layout)")
    elseif(EXISTS "${MUJOCO_DIR}/include/mujoco.h")
        set(HAS_MUJOCO TRUE)
        set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include")
        message(STATUS "MuJoCo found at ${MUJOCO_DIR} (flat layout)")
    else()
        set(HAS_MUJOCO FALSE)
        message(STATUS "MuJoCo not found in lib/mujoco/ — body model IK disabled")
    endif()

    target_compile_definitions(red PRIVATE USE_CUDA_POINTSOURCE)
    if(HAS_ONNXRUNTIME)
        target_compile_definitions(red PRIVATE RED_HAS_ONNXRUNTIME)
    endif()
    if(HAS_TENSORRT_HN)
        target_compile_definitions(red PRIVATE RED_HAS_TENSORRT_HN)
    endif()
    if(HAS_MUJOCO)
        target_compile_definitions(red PRIVATE RED_HAS_MUJOCO)
    endif()

    target_include_directories(red PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_IMPLOT3D}
        ${DIR_FILEBROWSER}
        ${DIR_ICONFONT}
        ${DIR_NVCODEC}
    )
    if(HAS_ONNXRUNTIME)
        target_include_directories(red PRIVATE "${ONNXRUNTIME_DIR}/include")
    endif()
    if(HAS_TENSORRT_HN)
        target_include_directories(red PRIVATE "${TENSORRT_INCLUDE_DIR}")
    endif()
    if(HAS_MUJOCO)
        target_include_directories(red PRIVATE "${MUJOCO_DIR}/include" "${MUJOCO_DIR}/include/mujoco")
    endif()

    # Link libraries
    target_link_libraries(red PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        glfw
        GLEW::GLEW
        OpenGL::GL
        PkgConfig::FFMPEG
        # CUDA runtime + NPP (imaging primitives)
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nppc
        CUDA::nppicc
        CUDA::nppidei
        CUDA::nppig
        CUDA::nppial
        # NVDEC (driver-provided: libnvcuvid.so from /usr/lib/x86_64-linux-gnu)
        nvcuvid
        # cblas: used by descriptor_matcher.h on non-Windows/non-Mac
        ${RED_CBLAS_LIB}
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )

    if(HAS_ONNXRUNTIME)
        target_link_libraries(red PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
    endif()
    if(HAS_TENSORRT_HN)
        # libnvinfer_plugin.so is wrapped in --no-as-needed because red has no
        # direct symbol references to it — initLibNvInferPlugins is called via
        # an inline helper in jarvis_hybridnet.h, which is reachable but the
        # linker's default --as-needed drops the DT_NEEDED entry. Without the
        # plugin lib loaded, hybrid3d.engine deserialization fails with
        # "getPluginCreator could not find plugin: InstanceNormalization_TRT".
        target_link_libraries(red PRIVATE
            "${TENSORRT_LIB}"
            "-Wl,--no-as-needed"
            "${TENSORRT_PLUGIN_LIB}"
            "-Wl,--as-needed")
    endif()
    if(HAS_MUJOCO)
        target_link_libraries(red PRIVATE "${MUJOCO_DIR}/lib/libmujoco.so")
    endif()

    # RPATH: isolate bundled ML libs from system / orange's toolchain.
    # $ORIGIN so the installed binary finds them relative to itself; build
    # rpath points at the absolute bundle paths for running from build/.
    set(RED_RPATH_BUILD "")
    set(RED_RPATH_INSTALL "$ORIGIN")
    if(HAS_ONNXRUNTIME)
        list(APPEND RED_RPATH_BUILD "${ONNXRUNTIME_DIR}/lib")
        list(APPEND RED_RPATH_INSTALL "$ORIGIN/../lib/onnxruntime/lib")
    endif()
    # cuDNN 9 for ONNX Runtime CUDA EP (bundled, never system-installed)
    set(CUDNN_DIR "${CMAKE_SOURCE_DIR}/lib/cudnn")
    if(EXISTS "${CUDNN_DIR}/lib")
        list(APPEND RED_RPATH_BUILD "${CUDNN_DIR}/lib")
        list(APPEND RED_RPATH_INSTALL "$ORIGIN/../lib/cudnn/lib")

        # ORT's CUDA provider library is dlopen'd at runtime by libonnxruntime.so
        # and doesn't inherit red's RUNPATH. Without a patched RPATH on the
        # provider .so itself, libcudnn.so.9 fails to resolve and ORT silently
        # falls back to CPU EP. Patch idempotently with patchelf if available.
        set(ORT_CUDA_PROVIDER "${ONNXRUNTIME_DIR}/lib/libonnxruntime_providers_cuda.so")
        if(HAS_ONNXRUNTIME AND EXISTS "${ORT_CUDA_PROVIDER}")
            find_program(PATCHELF_EXE patchelf)
            if(PATCHELF_EXE)
                execute_process(
                    COMMAND "${PATCHELF_EXE}" --set-rpath
                            "\$ORIGIN:\$ORIGIN/../../cudnn/lib:/usr/local/cuda/lib64"
                            "${ORT_CUDA_PROVIDER}"
                    RESULT_VARIABLE PATCHELF_RESULT
                    OUTPUT_QUIET ERROR_QUIET)
                if(PATCHELF_RESULT EQUAL 0)
                    message(STATUS "Patched ORT CUDA provider RPATH to find bundled cuDNN")
                else()
                    message(WARNING "patchelf on libonnxruntime_providers_cuda.so failed; "
                                    "set LD_LIBRARY_PATH=${CUDNN_DIR}/lib at runtime")
                endif()
            else()
                message(WARNING "patchelf not found; set LD_LIBRARY_PATH=${CUDNN_DIR}/lib "
                                "at runtime to enable CUDA EP")
            endif()
        endif()
    endif()
    if(HAS_MUJOCO)
        list(APPEND RED_RPATH_BUILD "${MUJOCO_DIR}/lib")
        list(APPEND RED_RPATH_INSTALL "$ORIGIN/../lib/mujoco/lib")
    endif()
    # TensorRT: build-tree RPATH only — read-only orange-shared install.
    # Users running `cmake --install` should set LD_LIBRARY_PATH at runtime
    # (or symlink the install to a fixed location) since the absolute path
    # under $HOME is not portable.
    if(HAS_TENSORRT_HN)
        list(APPEND RED_RPATH_BUILD "${TENSORRT_LIB_DIR}")
    endif()

    set_target_properties(red PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        BUILD_RPATH "${RED_RPATH_BUILD}"
        INSTALL_RPATH "${RED_RPATH_INSTALL}"
    )

    # Emit DT_RPATH (old dtags) instead of DT_RUNPATH. The shared FFmpeg's
    # libavcodec/libavformat have a transitive DT_NEEDED on libswresample.so.3
    # (same dir). DT_RUNPATH does not apply to a dependency's own NEEDED libs,
    # so libswresample fails to resolve at runtime; DT_RPATH propagates. Same
    # fix orange uses for its bundled FFmpeg.
    target_link_options(red PRIVATE "-Wl,--disable-new-dtags")

    # ── Test targets (same deps as red, replacing red.cpp) ──────────────────
    set(TEST_GUI_SRC_FILES ${SRC_FILES})
    list(REMOVE_ITEM TEST_GUI_SRC_FILES "${CMAKE_SOURCE_DIR}/src/red.cpp")

    foreach(TEST_NAME test_gui test_annotation test_jarvis_hybridnet_smoke)
        add_executable(${TEST_NAME}
            ${TEST_GUI_SRC_FILES}
            ${IMGUI_SRC}
            ${IMPLOT_SRC}
            ${IMPLOT3D_SRC}
            ${CUDA_SRC_FILES}
            tests/${TEST_NAME}.cpp
        )
        target_compile_definitions(${TEST_NAME} PRIVATE USE_CUDA_POINTSOURCE)
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${DIR_IMGUI}
            ${DIR_IMGUI}/backends
            ${DIR_IMPLOT}
            ${DIR_IMPLOT3D}
            ${DIR_FILEBROWSER}
            ${DIR_ICONFONT}
            ${DIR_NVCODEC}
        )
        target_link_libraries(${TEST_NAME} PRIVATE
            Eigen3::Eigen
            Ceres::ceres
            glfw
            GLEW::GLEW
            OpenGL::GL
            PkgConfig::FFMPEG
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nppc
            CUDA::nppicc
            CUDA::nppidei
            CUDA::nppig
            CUDA::nppial
            nvcuvid
            ${RED_CBLAS_LIB}
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
        if(HAS_ONNXRUNTIME)
            target_compile_definitions(${TEST_NAME} PRIVATE RED_HAS_ONNXRUNTIME)
            target_include_directories(${TEST_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")
            target_link_libraries(${TEST_NAME} PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
        endif()
        if(HAS_TENSORRT_HN)
            target_compile_definitions(${TEST_NAME} PRIVATE RED_HAS_TENSORRT_HN)
            target_include_directories(${TEST_NAME} PRIVATE "${TENSORRT_INCLUDE_DIR}")
            target_link_libraries(${TEST_NAME} PRIVATE
                "${TENSORRT_LIB}"
                "-Wl,--no-as-needed"
                "${TENSORRT_PLUGIN_LIB}"
                "-Wl,--as-needed")
        endif()
        if(HAS_MUJOCO)
            target_compile_definitions(${TEST_NAME} PRIVATE RED_HAS_MUJOCO)
            target_include_directories(${TEST_NAME} PRIVATE "${MUJOCO_DIR}/include" "${MUJOCO_DIR}/include/mujoco")
            target_link_libraries(${TEST_NAME} PRIVATE "${MUJOCO_DIR}/lib/libmujoco.so")
        endif()
        set_target_properties(${TEST_NAME} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            BUILD_RPATH "${RED_RPATH_BUILD}"
            INSTALL_RPATH "${RED_RPATH_INSTALL}"
        )
        # See note on the red target: DT_RPATH so the shared FFmpeg's
        # transitive libswresample.so.3 resolves at runtime.
        target_link_options(${TEST_NAME} PRIVATE "-Wl,--disable-new-dtags")
    endforeach()

endif()



