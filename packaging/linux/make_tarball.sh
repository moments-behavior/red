#!/usr/bin/env bash
# Package a built red into a self-contained folder and a tarball to download.
#
#     ./build.sh -DRED_ENABLE_CUDA=OFF && packaging/linux/make_tarball.sh [build_dir] [out_dir]
#
# Lays out
#     red/bin/red          rpath $ORIGIN/../lib
#     red/lib/             every shared library red needs, minus the system's
#     red/fonts/  default_imgui_layout.ini  icon.png
# (red looks for fonts and the default layout one folder above its binary),
# checks it, and writes <out_dir>/red-<version>-linux-x64.tar.gz (default
# out_dir: dist).
#
# Build on the OLDEST Linux it should run on (Ubuntu 22.04 for now): a binary
# built against a newer glibc will not start on an older one, the reverse is
# fine. CUDA off, so it runs without an NVIDIA driver.
#
# Left out of lib/, as an AppImage's excludelist does: the C library and the
# loader (each system has its own, and mixing them crashes), libstdc++/libgcc
# (a newer system's are compatible with code built on the oldest), and the
# graphics stack -- OpenGL, X11/xcb, Wayland, libdrm -- which has to match the
# user's GPU driver. Everything else is copied.
#
# red is linked with DT_RPATH, which -- unlike DT_RUNPATH -- also applies to
# the libraries it loads, so setting it on the binary alone is enough.
# Needs patchelf (apt install patchelf).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$REPO/${1:-release}"
OUT="$REPO/${2:-dist}"
EXE="$BUILD/red"
[ -x "$EXE" ] || { echo "$EXE not found -- build first: ./build.sh -DRED_ENABLE_CUDA=OFF" >&2; exit 1; }
command -v patchelf > /dev/null || { echo "patchelf not found: sudo apt install patchelf" >&2; exit 1; }

# A release tag when exactly at one, else <branch>-<commit>, as the other
# platforms' packaging and Help > About name it.
VERSION="$(git -C "$REPO" describe --tags --exact-match 2> /dev/null ||
           echo "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)-$(git -C "$REPO" rev-parse --short HEAD)")"
VERSION="${VERSION//\//-}"
git -C "$REPO" diff --quiet HEAD -- || VERSION="$VERSION-dirty"

# Libraries every Linux system provides itself (shell globs on the soname).
is_system() {
    case "$1" in
    linux-vdso.so*|ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|\
    librt.so*|libresolv.so*|libutil.so*|libnsl.so*|libanl.so*|\
    libstdc++.so*|libgcc_s.so*|\
    libGL.so*|libGLX*.so*|libEGL.so*|libOpenGL.so*|libGLdispatch.so*|\
    libGLU.so*|libdrm.so*|libgbm.so*|libvulkan.so*|\
    libX11*.so*|libXext.so*|libXrandr.so*|libXrender.so*|libXi.so*|\
    libXcursor.so*|libXinerama.so*|libXfixes.so*|libXxf86vm.so*|libXau.so*|\
    libXdmcp.so*|libxcb*.so*|libwayland-*.so*|libxkbcommon*.so*|\
    libasound.so*|libz.so*)
        return 0 ;;
    esac
    return 1
}

STAGE="$OUT/red"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/lib"
cp "$EXE" "$STAGE/bin/red"
cp -R "$REPO/fonts" "$STAGE/fonts"
cp "$REPO/default_imgui_layout.ini" "$REPO/icon.png" "$STAGE/"

echo "red $VERSION: collecting libraries..."
# ldd resolves the whole tree at once. A "not found" here means the build box
# itself cannot run red -- stop rather than ship it.
missing="$(ldd "$EXE" | awk '/=> not found/ { print $1 }')"
[ -z "$missing" ] || { echo "ldd cannot resolve on this machine: $missing" >&2; exit 1; }
n=0
: > "$OUT/.sources"
while read -r soname path; do
    is_system "$soname" && continue
    cp -L "$path" "$STAGE/lib/$soname"
    dirname "$(realpath "$path")" >> "$OUT/.sources"
    n=$((n + 1))
done < <(ldd "$EXE" | awk '$2 == "=>" && $3 ~ /^\// { print $1, $3 }')

# Where they came from. A release should carry the distribution's own
# packages; a library from a conda environment, ~/ or another private build
# was compiled for that setup and is not what the build box's OS provides.
echo "  copied from:"
sort "$OUT/.sources" | uniq -c | sed 's/^/    /'
foreign="$(sort -u "$OUT/.sources" | grep -vE '^(/usr)?/lib(64)?(/|$)|^/usr/local/lib(/|$)' || true)"
rm -f "$OUT/.sources"
if [ -n "$foreign" ]; then
    echo "WARNING: libraries from outside the system's folders:" >&2
    echo "$foreign" | sed 's/^/    /' >&2
    echo "  (a conda environment? run 'conda deactivate' and rebuild from a clean shell)" >&2
fi

# DT_RPATH (--force-rpath), not RUNPATH: it must reach the bundled libraries'
# own dependencies too.
patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "$STAGE/bin/red"

# Check: with the bundle in place, everything resolves, and every non-system
# library comes from lib/, not from wherever the build box keeps it.
bad=0
while read -r soname arrow path rest; do
    [ "$arrow" = "=>" ] || continue
    if [ "$path" = "not" ]; then
        echo "  unresolved: $soname" >&2; bad=1; continue
    fi
    is_system "$soname" && continue
    # ldd prints .../bin/../lib/x.so; compare the real path.
    case "$(realpath -m "$path")" in
    "$(realpath "$STAGE")/lib/"*) ;;
    *) echo "  not from the bundle: $soname -> $path" >&2; bad=1 ;;
    esac
done < <(ldd "$STAGE/bin/red")
[ "$bad" = 0 ] || { echo "the bundle is incomplete" >&2; exit 1; }
[ -z "$foreign" ] || { echo "not packaging libraries from outside the system's folders (see above)" >&2; exit 1; }

# One build in dist/ at a time.
rm -f "$OUT"/red-*-linux-x64.tar.gz
TAR="$OUT/red-$VERSION-linux-x64.tar.gz"
tar -C "$OUT" -czf "$TAR" red
echo "done: $STAGE ($n libraries, $(du -sm "$STAGE" | cut -f1) MB)"
echo "      $TAR ($(du -m "$TAR" | cut -f1) MB)"
echo "      glibc of this build box: $(ldd --version | head -1 | awk '{ print $NF }') -- needs that or newer"
echo "      run: red/bin/red"
