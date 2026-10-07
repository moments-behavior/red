# Package a built red.exe into a self-contained folder and a zip to download.
#
#     build.bat -DRED_ENABLE_CUDA=OFF
#     powershell -ExecutionPolicy Bypass -File packaging\windows\make_zip.ps1
#
# Lays out
#     dist\red\bin\red.exe  + every DLL it needs
#     dist\red\fonts\       default_imgui_layout.ini
#     dist\red\licenses\    Red's and every bundled library's licence
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

# --- Licences ------------------------------------------------------------------
# As packaging/licenses/collect_common.sh lays them out on macOS and Linux:
# Red's, the README, the fonts' and vendored code's, the licence texts. Then
# third_party\: each vcpkg port a DLL comes from (its copyright file, and the
# port files saying which source it was built from), FFmpeg's, and the C++
# runtime's.
$LicSrc = Join-Path $Repo "packaging\licenses"
$Lic = Join-Path $Stage "licenses"
$Vend = Join-Path $Lic "vendored"
$Third = Join-Path $Lic "third_party"
New-Item -ItemType Directory -Force -Path $Vend, $Third | Out-Null
Copy-Item (Join-Path $Repo "LICENSE") (Join-Path $Lic "LICENSE.txt")
foreach ($f in "README.txt", "fonts.txt", "GPL-3.0.txt", "Apache-2.0.txt", "OFL-1.1.txt") {
    Copy-Item (Join-Path $LicSrc $f) $Lic
}
Copy-Item (Join-Path $LicSrc "vendored\*.txt") $Vend
foreach ($line in Get-Content (Join-Path $LicSrc "vendored.txt")) {
    if ($line -match '^\s*(#|$)') { continue }
    $name, $file = -split $line
    $src = Join-Path $Repo $file
    if (-not (Test-Path $src)) { throw "licence file missing: $file" }
    Copy-Item $src (Join-Path $Vend "$name.txt")
}

# vcpkg: VCPKG_ROOT, else the vcpkg on PATH, else %USERPROFILE%\vcpkg -- as
# build.bat looks -- and of those the one whose install lists the DLLs.
$roots = @($env:VCPKG_ROOT)
$onPath = Get-Command vcpkg -ErrorAction SilentlyContinue
if ($onPath) { $roots += Split-Path $onPath.Source }
$roots += Join-Path $env:USERPROFILE "vcpkg"
$ffDlls = @(Get-ChildItem (Join-Path $FFmpeg "bin\*.dll") | ForEach-Object { $_.Name })
$vcpkgDlls = @(Get-ChildItem (Join-Path $BuildDir "*.dll") | ForEach-Object { $_.Name } |
    Where-Object { $ffDlls -notcontains $_ })
$Vcpkg = $null
foreach ($r in $roots | Where-Object { $_ }) {
    if (Test-Path (Join-Path $r "installed\vcpkg\info")) { $Vcpkg = $r; break }
}
if ($vcpkgDlls.Count -gt 0 -and -not $Vcpkg) { throw "vcpkg not found (set VCPKG_ROOT): needed for the DLLs' licences" }

$sources = @(
    "Shared libraries in bin\: name, version, where its source is. Licences:",
    "the folder of the same name; for vcpkg ports also the port files, which",
    "give the exact source and how it was built.",
    ""
)
$ports = @{}
foreach ($dll in $vcpkgDlls) {
    $hit = Select-String -Path (Join-Path $Vcpkg "installed\vcpkg\info\*.list") `
        -Pattern ("/bin/" + [regex]::Escape($dll) + '$') -List | Select-Object -First 1
    if (-not $hit) { throw "no vcpkg port installs $dll -- no licence to collect" }
    $ports[(Split-Path $hit.Path -Leaf)] = $true
}
foreach ($list in $ports.Keys | Sort-Object) {
    # <port>_<version>_<triplet>.list; the version may itself hold "_".
    $parts = [IO.Path]::GetFileNameWithoutExtension($list).Split("_")
    $port = $parts[0]
    $triplet = $parts[-1]
    $ver = ($parts[1..($parts.Count - 2)]) -join "_"
    $dst = Join-Path $Third $port
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $copyright = Join-Path $Vcpkg "installed\$triplet\share\$port\copyright"
    if (-not (Test-Path $copyright)) { throw "no $copyright" }
    Copy-Item $copyright $dst
    foreach ($pf in "vcpkg.json", "portfile.cmake") {
        $p = Join-Path $Vcpkg "ports\$port\$pf"
        if (Test-Path $p) { Copy-Item $p (Join-Path $dst "vcpkg-$pf") }
    }
    $sources += "{0,-24} {1,-16} vcpkg port, https://github.com/microsoft/vcpkg/tree/master/ports/{0}" -f $port, $ver
}

# FFmpeg: the licence files at the top of its folder, and its version.
$ffDst = Join-Path $Third "ffmpeg"
New-Item -ItemType Directory -Force -Path $ffDst | Out-Null
$ffLic = @(Get-ChildItem $FFmpeg -File | Where-Object { $_.Name -match '^(licen[cs]e|copying|readme)' })
if ($ffLic.Count -eq 0) { throw "no licence file in $FFmpeg (a BtbN build has LICENSE.txt)" }
$ffLic | Copy-Item -Destination $ffDst
$ffVer = (Split-Path $FFmpeg -Leaf)
$ffExe = Join-Path $FFmpeg "bin\ffmpeg.exe"
if (Test-Path $ffExe) { $ffVer = ((Quiet { & $ffExe -version }) | Select-Object -First 1) }
$sources += "ffmpeg                   $ffVer"
$sources += "                         https://ffmpeg.org/download.html (source: the version above),"
$sources += "                         built by https://github.com/BtbN/FFmpeg-Builds"
$sources += "Microsoft C++ runtime    $(Split-Path $crt.FullName -Leaf)  redistributable files of Visual Studio"
Set-Content -Path (Join-Path $Third "SOURCES.txt") -Value $sources

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

# --- Installer: Inno Setup, when it is installed ------------------------------
# Same dist\red folder, wrapped by packaging\windows\red.iss into a setup.exe
# with a Start menu entry and an uninstaller. Optional: without Inno Setup the
# zip above is the whole release.
$iscc = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")   # winget, per user
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    Write-Host "      (no installer: Inno Setup 6 not found -- winget install JRSoftware.InnoSetup)"
    return
}
Get-ChildItem $OutDir -Filter "red-*-windows-x64-setup.exe" -ErrorAction SilentlyContinue | Remove-Item -Force
$iss = Join-Path $Repo "packaging\windows\red.iss"
$isccOut = Quiet { & $iscc /Q "/DAppVersion=$Version" "/DStageDir=$Stage" "/DOutDir=$OutDir" $iss }
if ($LASTEXITCODE -ne 0) {
    # Run it again visibly so its error reaches the console.
    & $iscc "/DAppVersion=$Version" "/DStageDir=$Stage" "/DOutDir=$OutDir" $iss
    throw "Inno Setup failed (exit $LASTEXITCODE)"
}
$Setup = Join-Path $OutDir "red-$Version-windows-x64-setup.exe"
Write-Host "      $Setup ($([math]::Round((Get-Item $Setup).Length / 1MB)) MB)"
