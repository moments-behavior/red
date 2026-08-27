@echo off
rem Windows build. Discovers the toolchain rather than hardcoding paths, so it
rem runs on any machine with VS 2022 (or Build Tools), the CUDA Toolkit, and
rem vcpkg. Output goes to release\red.exe.
rem
rem   build.bat                      release build
rem   build.bat > log.txt 2>&1       ...capturing output
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem --- Visual Studio: vswhere ships with the VS Installer (2017+) ---------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Install Visual Studio 2022 or Build Tools.
    exit /b 1
)
set "VSPATH="
set "VSREQ=Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires %VSREQ% -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo ERROR: no Visual Studio installation with the C++ toolset was found.
    exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed.
    exit /b 1
)

rem --- Prefer the CMake and Ninja that ship with Visual Studio ------------
rem VS bundles CMake 3.31. Standalone CMake 4.x dropped the MSVC 19.44
rem feature tables this toolchain needs, so put the bundled one first.
set "VSCMAKE=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake"
if exist "%VSCMAKE%\CMake\bin\cmake.exe" set "PATH=%VSCMAKE%\CMake\bin;%VSCMAKE%\Ninja;%PATH%"

rem --- CUDA: the installer sets CUDA_PATH; CMakeLists.txt reads it -------
if not defined CUDA_PATH (
    echo ERROR: CUDA_PATH is not set. Install the CUDA Toolkit ^(12.x^), or set
    echo        it manually, e.g. set CUDA_PATH=C:\Program Files\NVIDIA GPU
    echo        Computing Toolkit\CUDA\v12.6
    exit /b 1
)
set "PATH=%CUDA_PATH%\bin;%PATH%"

rem --- vcpkg: every Windows dependency comes from here ------------------
rem Looked for as VCPKG_ROOT, then the vcpkg on PATH, then the usual clone
rem location. This is a hard requirement: without the toolchain CMake happily
rem falls back to system-wide packages and builds against whatever unrelated
rem Ceres or Eigen happens to be in Program Files, which fails much later and
rem much less clearly.
if not defined VCPKG_ROOT (
    for /f "usebackq tokens=*" %%i in (`where vcpkg 2^>nul`) do (
        if not defined VCPKG_ROOT set "VCPKG_ROOT=%%~dpi"
    )
    if defined VCPKG_ROOT if "!VCPKG_ROOT:~-1!"=="\" set "VCPKG_ROOT=!VCPKG_ROOT:~0,-1!"
)
if not defined VCPKG_ROOT (
    if exist "%USERPROFILE%\vcpkg\scripts\buildsystems\vcpkg.cmake" set "VCPKG_ROOT=%USERPROFILE%\vcpkg"
)
if not defined VCPKG_ROOT (
    echo ERROR: vcpkg not found. Install it, then either put vcpkg on PATH or
    echo        set VCPKG_ROOT, e.g.  set VCPKG_ROOT=C:\vcpkg
    exit /b 1
)
if not exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
    echo ERROR: "%VCPKG_ROOT%" is not a vcpkg root
    echo        ^(no scripts\buildsystems\vcpkg.cmake inside^).
    exit /b 1
)
rem --- FFmpeg: a prebuilt shared build, not vcpkg -----------------------
rem vcpkg's ffmpeg port compiles all of FFmpeg under MSVC and needs a working
rem msys2 + nasm; it fails often and slowly. A prebuilt drop from
rem https://github.com/BtbN/FFmpeg-Builds (win64-lgpl-shared) just works.
rem Point FFMPEG_ROOT at it, or unpack it to C:\ffmpeg.
if not defined FFMPEG_ROOT if exist "C:\ffmpeg\include\libavcodec\avcodec.h" set "FFMPEG_ROOT=C:\ffmpeg"
if not defined FFMPEG_ROOT (
    echo ERROR: FFmpeg not found. Download a win64 shared build from
    echo        https://github.com/BtbN/FFmpeg-Builds/releases ^(lgpl-shared^),
    echo        unpack it to C:\ffmpeg, or set FFMPEG_ROOT to where it lives.
    exit /b 1
)

echo Using vcpkg:  %VCPKG_ROOT%
echo Using CUDA:   %CUDA_PATH%
echo Using FFmpeg: %FFMPEG_ROOT%
set "TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT:\=/%/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_PREFIX_PATH=%FFMPEG_ROOT:\=/%"

cmake -G Ninja -B release -DCMAKE_BUILD_TYPE=Release %TOOLCHAIN% %*
if errorlevel 1 exit /b 1
cmake --build release
if errorlevel 1 exit /b 1

echo.
echo Build complete: release\red.exe
echo Run it with FFmpeg's DLLs on PATH:
echo     set PATH=%FFMPEG_ROOT%\bin;%%PATH%%
echo     release\red.exe
