cmake_minimum_required(VERSION 3.18...3.31)

# Build the NVDEC + CUDA-GL interop path (Windows/Linux; ignored on macOS,
# which uses VideoToolbox + Metal).
#
# ON  -- default, unchanged behavior: NVDEC decode, CUDA NV12->RGB, CUDA-GL
#        interop upload. Falls back to FFmpeg at RUNTIME when no usable GPU is
#        found (see src/decode_backend.h).
# OFF -- no CUDA language, no .cu sources, and critically no libcuda/libnvcuvid
#        link. Those are DT_NEEDED, so a default build will not even load on a
#        machine with no NVIDIA driver -- the loader fails before main() and the
#        runtime probe never gets to run. This is the build for such machines.
#
#   cmake -S . -B release -DRED_ENABLE_CUDA=OFF
option(RED_ENABLE_CUDA "Build the NVDEC + CUDA-GL interop path" ON)

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

    # Where the macOS dependencies live (Eigen, FFmpeg, GLFW, Ceres,
    # turbojpeg). No package manager is required -- any prefix CMake can search
    # works. Point at one explicitly with either:
    #     -DRED_DEPS_PREFIX=/your/prefix
    #     -DCMAKE_PREFIX_PATH=/your/prefix
    # Otherwise the usual roots are probed in order: Homebrew on Apple Silicon,
    # Homebrew on Intel (or a manual /usr/local install), then MacPorts.
    if(DEFINED HOMEBREW_PREFIX AND NOT DEFINED RED_DEPS_PREFIX)
        set(RED_DEPS_PREFIX "${HOMEBREW_PREFIX}")  # accepted for compatibility
    endif()
    if(NOT DEFINED RED_DEPS_PREFIX)
        foreach(_prefix "/opt/homebrew" "/usr/local" "/opt/local")
            if(IS_DIRECTORY "${_prefix}/include")
                set(RED_DEPS_PREFIX "${_prefix}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT DEFINED RED_DEPS_PREFIX)
        message(FATAL_ERROR
            "No dependency prefix found. Install the dependencies listed in "
            "README.md, or pass -DRED_DEPS_PREFIX=/your/prefix.")
    endif()
    message(STATUS "Using dependency prefix: ${RED_DEPS_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${RED_DEPS_PREFIX}")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)

    find_package(Eigen3 REQUIRED)
    find_package(glfw3 REQUIRED)
    pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)
    find_package(Ceres REQUIRED)



    set(DIR_IMGUI "${CMAKE_SOURCE_DIR}/lib/imgui")
    set(DIR_IMPLOT "${CMAKE_SOURCE_DIR}/lib/implot")

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
        ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
        ${DIR_IMGUI}/backends/imgui_impl_metal.mm
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)

    add_executable(red
        ${SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
    )

    target_include_directories(red PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${FFMPEG_INCLUDE_DIRS}
        ${TURBOJPEG_INCLUDE_DIRS}
        ${RED_DEPS_PREFIX}/include
        ${RED_DEPS_PREFIX}/include/eigen3
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
        "-framework Accelerate"
        "-framework CoreVideo"
        # Explicit: red.cpp calls CGEventSourceFlagsState() via
        # mac_modifier_fix.h. It previously resolved transitively through
        # Cocoa, which is fragile to rely on.
        "-framework CoreGraphics"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    # Removed: Vulkan::Vulkan



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
        ${FFMPEG_INCLUDE_DIRS}
        ${RED_DEPS_PREFIX}/include
        ${RED_DEPS_PREFIX}/include/eigen3
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
        tests/test_annotation.cpp
    )
    target_include_directories(test_annotation PRIVATE
        src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${FFMPEG_INCLUDE_DIRS}
        ${RED_DEPS_PREFIX}/include
        ${RED_DEPS_PREFIX}/include/eigen3
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
        "-framework Accelerate"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    set_target_properties(test_annotation PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )



    # Decoder backend conformance test: same deps as test_gui
    add_executable(test_decoder
        ${TEST_GUI_SRC_FILES}
        ${MM_SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        tests/test_decoder.cpp
    )
    target_include_directories(test_decoder PRIVATE
        src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_FILEBROWSER}
        lib/nvcodec
        lib/IconFontCppHeaders
        ${FFMPEG_INCLUDE_DIRS}
        ${HOMEBREW_PREFIX}/include
        ${HOMEBREW_PREFIX}/include/eigen3
    )
    target_link_directories(test_decoder PRIVATE ${FFMPEG_LIBRARY_DIRS} ${TURBOJPEG_LIBRARY_DIRS})
    target_link_libraries(test_decoder PRIVATE
        Eigen3::Eigen
        glfw
        "-framework Metal"
        "-framework QuartzCore"
        "-framework VideoToolbox"
        "-framework CoreMedia"
        "-framework CoreFoundation"
        "-framework CoreVideo"
        "-framework Accelerate"
        "-framework Cocoa"
        ${FFMPEG_LIBRARIES}
        ${TURBOJPEG_LIBRARIES}
        Ceres::ceres
        pthread
    )
    set_target_properties(test_decoder PROPERTIES
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



    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )



    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
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
        ${RED_DEPS_PREFIX}/include
        ${RED_DEPS_PREFIX}/include/eigen3
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

    set_source_files_properties(
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )


    set_source_files_properties(
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        src/vt_async_decoder.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )


    set_source_files_properties(
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    set_source_files_properties(
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )




elseif(WIN32)
    # Windows build with NVIDIA CUDA support (RTX 3090)
    if(RED_ENABLE_CUDA)
        project(red LANGUAGES CXX CUDA)
        # CUDA architecture: 86 for RTX 3090 (Ampere)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    else()
        project(red LANGUAGES CXX)
    endif()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # No vcpkg auto-detection here: CMAKE_TOOLCHAIN_FILE is only read during
    # the project() call above, so setting it at this point does nothing. It
    # has to arrive on the command line -- build.bat passes it.

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

    # CUDA source files. Empty without RED_ENABLE_CUDA, and NvDecoder.cpp goes
    # with them -- it is pure NVDEC and will not compile without cuviddec.
    if(RED_ENABLE_CUDA)
        set(CUDA_SRC_FILES
            src/create_image_cuda.cu
            src/ColorSpace.cu
            src/kernel.cu
        )
        set(RED_CUDA_LIBS
            "${CUDA_DIR}/lib/x64/cudart.lib"
            "${CUDA_DIR}/lib/x64/cuda.lib"
            "${CUDA_DIR}/lib/x64/nppc.lib"
            "${CUDA_DIR}/lib/x64/nppicc.lib"
            "${CUDA_DIR}/lib/x64/nppidei.lib"
            "${CUDA_DIR}/lib/x64/nppig.lib"
            "${CUDA_DIR}/lib/x64/nppial.lib"
            "${DIR_NVCODEC}/x64/nvcuvid.lib"
            "${DIR_NVCODEC}/x64/nvencodeapi.lib"
            # Required by /DELAYLOAD below.
            delayimp.lib
        )
        # nvcuda.dll and nvcuvid.dll ship with the NVIDIA driver, not with the
        # app. Delay-loading them means a machine with no driver still reaches
        # main(), where decode_backend() probes and falls back to FFmpeg --
        # without this the loader fails first and the app vanishes with no
        # message at all.
        #
        # nvEncodeAPI64.dll is deliberately absent: nvencodeapi.lib is on the
        # link line but no NVENC symbol is referenced anywhere in src/, so no
        # import is emitted and naming it here would only earn an LNK4199.
        set(RED_DELAYLOAD_OPTS
            /DELAYLOAD:nvcuda.dll
            /DELAYLOAD:nvcuvid.dll
        )
    else()
        set(CUDA_SRC_FILES "")
        set(RED_CUDA_LIBS "")
        set(RED_DELAYLOAD_OPTS "")
        list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/NvDecoder.cpp")
    endif()

    # Main executable
    add_executable(red
        ${SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${CUDA_SRC_FILES}
    )



    target_compile_definitions(red PRIVATE NOMINMAX _USE_MATH_DEFINES)
    if(NOT RED_ENABLE_CUDA)
        target_compile_definitions(red PRIVATE RED_NO_CUDA)
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
        ${DIR_FILEBROWSER}
        ${DIR_ICONFONT}
        ${DIR_NVCODEC}
        ${CUDA_DIR}/include
        ${FFMPEG_INCLUDE_DIRS}
    )
    if(TURBOJPEG_INCLUDE)
        target_include_directories(red PRIVATE ${TURBOJPEG_INCLUDE})
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
        # CUDA + bundled NVDEC (empty without RED_ENABLE_CUDA)
        ${RED_CUDA_LIBS}
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



    # Windows system libraries
    target_link_libraries(red PRIVATE
        shell32
        ole32
    )




    # vcpkg automatically copies dependent DLLs via its toolchain

    set_target_properties(red PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # ── Test targets (same deps as red, replacing red.cpp) ──────────────────
    set(TEST_GUI_SRC_FILES ${SRC_FILES})
    list(REMOVE_ITEM TEST_GUI_SRC_FILES "${CMAKE_SOURCE_DIR}/src/red.cpp")

    foreach(TEST_NAME test_gui test_annotation test_decoder)
        add_executable(${TEST_NAME}
            ${TEST_GUI_SRC_FILES}
            ${IMGUI_SRC}
            ${IMPLOT_SRC}
            ${CUDA_SRC_FILES}
            tests/${TEST_NAME}.cpp
        )
        target_compile_definitions(${TEST_NAME} PRIVATE NOMINMAX _USE_MATH_DEFINES)
        if(NOT RED_ENABLE_CUDA)
            target_compile_definitions(${TEST_NAME} PRIVATE RED_NO_CUDA)
        endif()
        target_compile_options(${TEST_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj /EHsc>)
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${DIR_IMGUI}
            ${DIR_IMGUI}/backends
            ${DIR_IMPLOT}
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
            ${RED_CUDA_LIBS}
            shell32
            ole32
        )
        target_link_options(${TEST_NAME} PRIVATE ${RED_DELAYLOAD_OPTS})
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
        set_target_properties(${TEST_NAME} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endforeach()

    # Windows default 1MB stack is too small for deep recursive parsing
    # (complex rodent/fly models) and STB image static initializers.
    # Use 64MB stack for all targets.
    target_link_options(red PRIVATE /STACK:67108864 ${RED_DELAYLOAD_OPTS})
    target_link_options(test_gui PRIVATE /STACK:67108864)
    target_link_options(test_annotation PRIVATE /STACK:67108864)
    target_link_options(test_decoder PRIVATE /STACK:67108864)

else()
    # ────────────────────────────────────────────────────────────────────────
    # Linux build with NVIDIA CUDA + NVDEC
    #
    # Coexists with the orange app (same machine): uses the same CUDA 12.2
    # at /usr/local/cuda, the same custom FFmpeg at $HOME/nvidia/ffmpeg, and
    # the same driver 535.x. Orange's toolchain is UNTOUCHED — red just
    # links read-only against those shared installs.
    # No OpenCV, no LibTorch.
    # ────────────────────────────────────────────────────────────────────────
    if(RED_ENABLE_CUDA)
        project(red LANGUAGES CXX CUDA)
        # CUDA arch: 86 = A16 (Ampere), 89 = RTX 4000 Ada. Override with
        # -DCMAKE_CUDA_ARCHITECTURES=... on the cmake command line if needed.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 86 89)
        endif()
    else()
        project(red LANGUAGES CXX)
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
    if(RED_ENABLE_CUDA)
        find_package(CUDAToolkit REQUIRED)
    endif()
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
    set(DIR_FILEBROWSER "${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog")
    set(DIR_ICONFONT "${CMAKE_SOURCE_DIR}/lib/IconFontCppHeaders")
    set(DIR_NVCODEC "${CMAKE_SOURCE_DIR}/lib/nvcodec")

    # Source files (mirrors Windows — uses patched GLFW backend from src/)
    file(GLOB SRC_FILES src/*.cpp)

    file(GLOB IMGUI_SRC
        ${DIR_IMGUI}/*.cpp
        ${DIR_IMGUI}/backends/imgui_impl_glfw.cpp
        ${DIR_IMGUI}/backends/imgui_impl_opengl3.cpp
        ${DIR_IMGUI}/misc/cpp/imgui_stdlib.cpp
        ${DIR_FILEBROWSER}/ImGuiFileDialog.cpp
    )
    file(GLOB IMPLOT_SRC ${DIR_IMPLOT}/*.cpp)

    # CUDA sources and the libcuda/libnvcuvid link. Empty without
    # RED_ENABLE_CUDA -- that is the whole point of the option: those two are
    # DT_NEEDED entries, and a binary carrying them will not load at all on a
    # machine with no NVIDIA driver. NvDecoder.cpp goes with them; it is pure
    # NVDEC and will not compile without cuviddec.
    if(RED_ENABLE_CUDA)
        set(CUDA_SRC_FILES
            src/create_image_cuda.cu
            src/ColorSpace.cu
            src/kernel.cu
        )
        set(RED_CUDA_LIBS
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
        )
    else()
        set(CUDA_SRC_FILES "")
        set(RED_CUDA_LIBS "")
        list(REMOVE_ITEM SRC_FILES "${CMAKE_SOURCE_DIR}/src/NvDecoder.cpp")
    endif()

    add_executable(red
        ${SRC_FILES}
        ${IMGUI_SRC}
        ${IMPLOT_SRC}
        ${CUDA_SRC_FILES}
    )




    if(RED_ENABLE_CUDA)
        target_compile_definitions(red PRIVATE)
    else()
        target_compile_definitions(red PRIVATE RED_NO_CUDA)
    endif()

    target_include_directories(red PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${DIR_IMGUI}
        ${DIR_IMGUI}/backends
        ${DIR_IMPLOT}
        ${DIR_FILEBROWSER}
        ${DIR_ICONFONT}
        ${DIR_NVCODEC}
    )

    # Link libraries
    target_link_libraries(red PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        glfw
        GLEW::GLEW
        OpenGL::GL
        PkgConfig::FFMPEG
        ${RED_CUDA_LIBS}
        # cblas: used by descriptor_matcher.h on non-Windows/non-Mac
        ${RED_CBLAS_LIB}
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )


    # RPATH: isolate bundled ML libs from system / orange's toolchain.
    # $ORIGIN so the installed binary finds them relative to itself; build
    # rpath points at the absolute bundle paths for running from build/.
    set(RED_RPATH_BUILD "")
    set(RED_RPATH_INSTALL "$ORIGIN")

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

    foreach(TEST_NAME test_gui test_annotation test_decoder)
        add_executable(${TEST_NAME}
            ${TEST_GUI_SRC_FILES}
            ${IMGUI_SRC}
            ${IMPLOT_SRC}
            ${CUDA_SRC_FILES}
            tests/${TEST_NAME}.cpp
        )
        if(RED_ENABLE_CUDA)
            target_compile_definitions(${TEST_NAME} PRIVATE)
        else()
            target_compile_definitions(${TEST_NAME} PRIVATE RED_NO_CUDA)
        endif()
        target_include_directories(${TEST_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${DIR_IMGUI}
            ${DIR_IMGUI}/backends
            ${DIR_IMPLOT}
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
            ${RED_CUDA_LIBS}
            ${RED_CBLAS_LIB}
            Threads::Threads
            ${CMAKE_DL_LIBS}
        )
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


