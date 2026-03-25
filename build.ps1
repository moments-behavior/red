$ErrorActionPreference = "Continue"

# Get VS environment
$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# Run vcvars64.bat and capture the environment
$envOutput = cmd /C "`"$vsDevCmd`" >nul 2>&1 && set"
foreach ($line in $envOutput) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Set CUDA env — must use v12.6 (driver 560.94 only supports CUDA <= 12.6)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:CudaToolkitDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\"
$env:PATH = "$env:CUDA_PATH\bin;C:\Users\johnsonr\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;C:\Program Files\CMake\bin;$env:PATH"

Set-Location "C:\Users\johnsonr\src\red"

# Configure
cmake -G Ninja -B build_win `
    -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" `
    -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" `
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" `
    -DCMAKE_TOOLCHAIN_FILE="C:/Users/johnsonr/vcpkg/scripts/buildsystems/vcpkg.cmake"

# Build
cmake --build build_win
