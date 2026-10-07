#!/usr/bin/env bash
# Turn the tarball's folder (dist/red, from make_tarball.sh) into an AppImage.
#
#     packaging/linux/make_appimage.sh [out_dir]       (default: dist)
#
# Run by build_release.sh inside the Ubuntu 22.04 image, which carries
# appimagetool and the static type-2 runtime in /opt/appimage. The static
# runtime has FUSE built in, so the AppImage runs on stock Ubuntu 22.04/24.04
# without libfuse2 -- the reason not to use the classic runtime.
#
# AppDir layout: usr/bin/red keeps its rpath $ORIGIN/../lib -> usr/lib, and
# red looks for fonts/ and default_imgui_layout.ini one folder above its binary
# -- usr/ -- so the tarball's tree moves in unchanged.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/${1:-dist}"
STAGE="$OUT/red"
TOOLS=/opt/appimage
[ -x "$STAGE/bin/red" ] || { echo "$STAGE/bin/red not found -- run make_tarball.sh first" >&2; exit 1; }
[ -x "$TOOLS/appimagetool" ] ||
    { echo "$TOOLS/appimagetool not found -- run this through build_release.sh" >&2; exit 1; }

# Named as make_tarball.sh names the tarball.
VERSION="$(git -C "$REPO" describe --tags --exact-match 2> /dev/null ||
           echo "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)-$(git -C "$REPO" rev-parse --short HEAD)")"
VERSION="${VERSION//\//-}"
git -C "$REPO" diff --quiet HEAD -- || VERSION="$VERSION-dirty"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
APPDIR="$WORK/red.AppDir"
mkdir -p "$APPDIR/usr"
cp -a "$STAGE/bin" "$STAGE/lib" "$STAGE/fonts" "$STAGE/licenses" "$STAGE/default_imgui_layout.ini" "$APPDIR/usr/"
cp "$REPO/icon.png" "$APPDIR/red.png"
cp "$REPO/icon.png" "$APPDIR/.DirIcon"

cat > "$APPDIR/red.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Red
Comment=Multi-camera video labeling
Exec=red
Icon=red
Terminal=false
Categories=Science;Video;
EOF

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/red" "$@"
EOF
chmod +x "$APPDIR/AppRun"

rm -f "$OUT"/red-*-linux-x86_64.AppImage
IMG="$OUT/red-$VERSION-linux-x86_64.AppImage"
# appimagetool is itself an AppImage; extract-and-run, as a container has no FUSE.
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 \
    "$TOOLS/appimagetool" --runtime-file "$TOOLS/runtime-x86_64" --no-appstream \
    "$APPDIR" "$IMG"
chmod +x "$IMG"
echo "done: $IMG ($(du -m "$IMG" | cut -f1) MB)"
echo "      run: chmod +x $(basename "$IMG") && ./$(basename "$IMG")"
