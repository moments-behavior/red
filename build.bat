@echo off
rem Windows build. Discovers the toolchain rather than hardcoding paths, so it
rem runs on any machine with VS 2022 (or Build Tools), the CUDA Toolkit, and
rem vcpkg. Output goes to build_win\red.exe.
rem
rem   build.bat                      release build
rem   build.bat > log.txt 2>&1       ...capturing output
setlocal
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

rem --- vcpkg: only needed if it is not at %USERPROFILE%\vcpkg, which
rem     CMakeLists.txt already falls back to on its own ----------------------
set "TOOLCHAIN="
if defined VCPKG_ROOT set "TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT:\=/%/scripts/buildsystems/vcpkg.cmake"

cmake -G Ninja -B build_win -DCMAKE_BUILD_TYPE=Release %TOOLCHAIN%
if errorlevel 1 exit /b 1
cmake --build build_win
if errorlevel 1 exit /b 1

echo.
echo Build complete: build_win\red.exe
