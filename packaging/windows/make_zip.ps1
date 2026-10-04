# Package a built red.exe into a self-contained folder and a zip to download.
#
#     build.bat -DRED_ENABLE_CUDA=OFF
#     powershell -ExecutionPolicy Bypass -File packaging\windows\make_zip.ps1
#
# Lays out
#     dist\red\bin\red.exe  + every DLL it needs
#     dist\red\fonts\       default_imgui_layout.ini
# (red looks for fonts and the default layout one folder above its exe), then
# zips it. The DLLs come from three places: vcpkg's, which the build already
# copied next to red.exe; FFmpeg's, from FFMPEG_ROOT\bin (or C:\ffmpeg\bin, as
# build.bat finds it); and the Microsoft C++ runtime from Visual Studio's
# redist folder, so users need not install the VC++ Redistributable. Every
# DLL any file in bin\ imports is then checked with dumpbin: it must be in
# bin\ or part of Windows, or the script fails and says which.
#
# Options: -Build <dir> (default release), -Out <dir> (default dist).

param(
    [string]$Build = "release",
    [string]$Out = "dist"
)
$ErrorActionPreference = "Stop"

# Run an external program with its stderr discarded. Windows PowerShell 5.1
# turns a native command's stderr into a terminating error under
# ErrorActionPreference=Stop, even when redirected -- so git's "no tag here",
# or a line-ending warning, would end the script. Callers check
# $LASTEXITCODE where the outcome matters.
function Quiet([scriptblock]$Run) {
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try { & $Run 2>$null } finally { $ErrorActionPreference = $eap }
}

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BuildDir = Join-Path $Repo $Build
$Exe = Join-Path $BuildDir "red.exe"
if (-not (Test-Path $Exe)) { throw "$Exe not found -- build first: build.bat -DRED_ENABLE_CUDA=OFF" }

# --- Version: a release tag when exactly at one, else branch-commit ---------
$Version = Quiet { git -C $Repo describe --tags --exact-match }
if ($LASTEXITCODE -ne 0) { $Version = $null }
if (-not $Version) {
    $branch = (Quiet { git -C $Repo rev-parse --abbrev-ref HEAD }).Trim()
    $commit = (Quiet { git -C $Repo rev-parse --short HEAD }).Trim()
    $Version = "$branch-$commit"
}
$Version = $Version.Trim().Replace("/", "-")
Quiet { git -C $Repo diff --quiet HEAD -- } | Out-Null
if ($LASTEXITCODE -ne 0) { $Version = "$Version-dirty" }

# --- Visual Studio: dumpbin and the C++ runtime -----------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found -- is Visual Studio 2022 installed?" }
$VS = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $VS) { throw "No Visual Studio with the C++ toolset found." }
$dumpbin = Get-ChildItem (Join-Path $VS "VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe") |
    Sort-Object FullName | Select-Object -Last 1
if (-not $dumpbin) { throw "dumpbin.exe not found under $VS" }
$crt = Get-ChildItem (Join-Path $VS "VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT") -Directory |
    Sort-Object FullName | Select-Object -Last 1
if (-not $crt) { throw "C++ runtime (VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT) not found under $VS" }

# --- FFmpeg: where build.bat found it ---------------------------------------
$FFmpeg = $env:FFMPEG_ROOT
if (-not $FFmpeg -and (Test-Path "C:\ffmpeg\bin")) { $FFmpeg = "C:\ffmpeg" }
if (-not $FFmpeg -or -not (Test-Path (Join-Path $FFmpeg "bin"))) {
    throw "FFmpeg not found: set FFMPEG_ROOT, or unpack it to C:\ffmpeg (as for build.bat)"
}

Write-Host "red $Version"
Write-Host "  Visual Studio: $VS"
Write-Host "  C++ runtime:   $($crt.FullName)"
Write-Host "  FFmpeg:        $FFmpeg"

# --- Lay out dist\red --------------------------------------------------------
$OutDir = Join-Path $Repo $Out
$Stage = Join-Path $OutDir "red"
$Bin = Join-Path $Stage "bin"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Bin | Out-Null

Copy-Item $Exe $Bin
Copy-Item (Join-Path $BuildDir "*.dll") $Bin            # vcpkg's, copied by the build
Copy-Item (Join-Path $FFmpeg "bin\*.dll") $Bin
Copy-Item (Join-Path $crt.FullName "*.dll") $Bin
Copy-Item (Join-Path $Repo "fonts") $Stage -Recurse
Copy-Item (Join-Path $Repo "default_imgui_layout.ini") $Stage

# --- Every import must be in bin\ or part of Windows ------------------------
# Windows' own DLLs live in System32; api-ms-win-* / ext-ms-* are API sets the
# loader maps to them and exist as no file of that name.
$sys32 = Join-Path $env:WINDIR "System32"
$missing = @()
foreach ($f in Get-ChildItem $Bin -Include *.exe, *.dll -Recurse) {
    $lines = Quiet { & $dumpbin.FullName /nologo /dependents $f.FullName }
    foreach ($line in $lines) {
        $name = $line.Trim()
        if ($name -notmatch '^[\w\.\-]+\.dll$') { continue }
        if ($name -match '^(api|ext)-ms-') { continue }
        if (Test-Path (Join-Path $Bin $name)) { continue }
        if (Test-Path (Join-Path $sys32 $name)) { continue }
        $missing += "$($f.Name) -> $name"
    }
}
if ($missing.Count -gt 0) {
    throw ("Not in bin\ and not part of Windows:`n  " + (($missing | Sort-Object -Unique) -join "`n  "))
}

# --- Zip: one build in dist\ at a time ---------------------------------------
Get-ChildItem $OutDir -Filter "red-*-windows-x64.zip" -ErrorAction SilentlyContinue | Remove-Item -Force
$Zip = Join-Path $OutDir "red-$Version-windows-x64.zip"
Compress-Archive -Path $Stage -DestinationPath $Zip
$dlls = (Get-ChildItem $Bin -Filter *.dll).Count
$mb = [math]::Round(((Get-ChildItem $Stage -Recurse | Measure-Object Length -Sum).Sum) / 1MB)
Write-Host "done: $Stage ($dlls DLLs, $mb MB)"
Write-Host "      $Zip ($([math]::Round((Get-Item $Zip).Length / 1MB)) MB)"
Write-Host "      run: red\bin\red.exe"
