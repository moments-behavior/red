#!/usr/bin/env bash
# Package a built red into a self-contained Red.app and a zip to download.
#
#     ./build.sh && packaging/macos/make_app.sh [build_dir] [out_dir]
#
# Takes <build_dir>/red (default: release), copies every non-system dylib it
# needs -- FFmpeg, Arrow, Ceres and what they pull in -- into
# Red.app/Contents/Frameworks, rewrites the references so they load from there
# instead of /opt/homebrew, signs it all ad hoc, checks nothing still points
# outside the app, and zips it into <out_dir> (default: dist).
#
# The app is arm64 only and needs the macOS version the Homebrew bottles were
# built for (read from them into Info.plist). It is signed ad hoc, not with a
# Developer ID, so another Mac refuses the first launch; then System Settings >
# Privacy & Security > Open Anyway (right-click > Open no longer bypasses this
# since macOS 15), or: xattr -dr com.apple.quarantine Red.app
#
# Written for the bash 3.2 macOS ships: no associative arrays.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$REPO/${1:-release}"
OUT="$REPO/${2:-dist}"
EXE="$BUILD/red"
[ -x "$EXE" ] || { echo "$EXE not found -- build first (./build.sh)" >&2; exit 1; }

# A release tag when building exactly at one (v1.2.0); otherwise the branch
# and commit (multianimal-2b84530). `git describe` alone named builds after
# the nearest tag anywhere in history, e.g. fetch_paper-snapshot-231-g2b84530.
VERSION="$(git -C "$REPO" describe --tags --exact-match 2> /dev/null ||
           echo "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)-$(git -C "$REPO" rev-parse --short HEAD)")"
VERSION="${VERSION//\//-}"   # a branch like feature/x must not make a path
git -C "$REPO" diff --quiet HEAD -- || VERSION="$VERSION-dirty"
APP="$OUT/Red.app"   # what Finder and the Dock show; the binary inside stays red
rm -rf "$OUT/red.app"   # the name builds before Red.app used
CONTENTS="$APP/Contents"
FW="$CONTENTS/Frameworks"
RES="$CONTENTS/Resources"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

is_system() { case "$1" in /usr/lib/*|/System/*) return 0;; esac; return 1; }

# Install names a Mach-O links, without its own id.
deps() {
    local id
    id="$(otool -D "$1" | sed -n 2p)"
    otool -L "$1" | tail -n +2 | sed -E 's/^[[:space:]]+//; s/ \(compat.*//' |
        { grep -vxF "${id:-//none//}" || true; }
}

rpaths() { otool -l "$1" | awk '/cmd LC_RPATH/ { getline; getline; print $2 }'; }

# install_name_tool, without its warning that the edit voids the signature --
# true of every file here, and everything is re-signed at the end.
int() { install_name_tool "$@" 2> >(grep -v 'invalidate the code signature' >&2); }

# The real file an install name refers to, seen from the Mach-O $2.
resolve() {
    local name="$1" from="$2" here rest r
    here="$(dirname "$from")"
    case "$name" in
    @loader_path/*)
        [ -e "$here/${name#@loader_path/}" ] && { realpath "$here/${name#@loader_path/}"; return; } ;;
    @rpath/*)
        rest="${name#@rpath/}"
        while IFS= read -r r; do
            r="${r//@loader_path/$here}"
            [ -e "$r/$rest" ] && { realpath "$r/$rest"; return; }
        done < <(rpaths "$from")
        # libgfortran's @rpath siblings (libquadmath, libgcc_s) sit beside it.
        [ -e "$here/$rest" ] && { realpath "$here/$rest"; return; } ;;
    *)
        [ -e "$name" ] && { realpath "$name"; return; } ;;
    esac
    echo "cannot resolve $name (needed by $from)" >&2
    return 1
}

# install_name_tool args leaving $2 as the only rpath of $1.
rpath_args() {
    local r had=0
    while IFS= read -r r; do
        if [ "$r" = "$2" ]; then had=1; else printf '%s\n' -delete_rpath "$r"; fi
    done < <(rpaths "$1")
    [ "$had" = 1 ] || printf '%s\n' -add_rpath "$2"
}

# -change args pointing every non-system reference of $1 at @rpath/<name>.
change_args() {
    local n p
    while IFS= read -r n; do
        is_system "$n" && continue
        p="$(resolve "$n" "$1")"
        printf '%s\n' -change "$n" "@rpath/$(basename "$p")"
    done < <(deps "$1")
}

echo "red $VERSION: collecting libraries..."
# Breadth-first over the dependency tree; $WORK/libs holds real paths seen.
: > "$WORK/libs"
echo "$(realpath "$EXE")" > "$WORK/queue"
while [ -s "$WORK/queue" ]; do
    f="$(head -1 "$WORK/queue")"
    sed -i '' 1d "$WORK/queue"
    while IFS= read -r n; do
        is_system "$n" && continue
        p="$(resolve "$n" "$f")"
        if ! grep -qxF "$p" "$WORK/libs"; then
            echo "$p" >> "$WORK/libs"
            echo "$p" >> "$WORK/queue"
        fi
    done < <(deps "$f")
done
dup="$(while IFS= read -r p; do basename "$p"; done < "$WORK/libs" | sort | uniq -d)"
[ -z "$dup" ] || { echo "two different dylibs share a name: $dup" >&2; exit 1; }
nlibs="$(wc -l < "$WORK/libs" | tr -d ' ')"

rm -rf "$APP"
mkdir -p "$CONTENTS/MacOS" "$FW" "$RES"
cp "$EXE" "$CONTENTS/MacOS/red"
# Data files go in Contents/Resources -- codesign rejects them anywhere else
# in Contents/ -- where red_resource_dir() (gx_helper.h) looks for them.
cp -R "$REPO/fonts" "$RES/fonts"
cp "$REPO/default_imgui_layout.ini" "$RES/"

# App icon from the repo's icon.png (512 px): every size macOS asks for, in
# an .icns. No 512@2x -- that is 1024 px, and blowing 512 up would only blur.
ICONSET="$WORK/red.iconset"
mkdir -p "$ICONSET"
for px in 16 32 128 256 512; do
    sips -z $px $px "$REPO/icon.png" --out "$ICONSET/icon_${px}x${px}.png" > /dev/null
    [ $px = 512 ] && continue
    sips -z $((px * 2)) $((px * 2)) "$REPO/icon.png" \
        --out "$ICONSET/icon_${px}x${px}@2x.png" > /dev/null
done
iconutil -c icns "$ICONSET" -o "$RES/red.icns"

echo "  $nlibs dylibs; rewriting install names..."
while IFS= read -r real; do
    base="$(basename "$real")"
    dst="$FW/$base"
    cp "$real" "$dst"
    chmod 755 "$dst"
    # References are resolved against the ORIGINAL, whose rpaths still point
    # where its dependencies are; the copy only gets the rewritten names.
    args=()
    while IFS= read -r a; do args+=("$a"); done < <(change_args "$real"; rpath_args "$dst" "@loader_path")
    int -id "@rpath/$base" "$dst"
    [ ${#args[@]} -eq 0 ] || int "${args[@]}" "$dst"
done < "$WORK/libs"

args=()
while IFS= read -r a; do args+=("$a"); done < <(change_args "$EXE"; rpath_args "$CONTENTS/MacOS/red" "@executable_path/../Frameworks")
int "${args[@]}" "$CONTENTS/MacOS/red"

# The newest macOS any piece was built for is what the app needs.
MINOS="$( { echo "$EXE"; cat "$WORK/libs"; } | while IFS= read -r f; do
              otool -l "$f" | awk '/ minos / { print $2 }'
          done | sort -t. -k1,1n -k2,2n | tail -1)"

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>               <string>Red</string>
    <key>CFBundleDisplayName</key>        <string>Red</string>
    <key>CFBundleIdentifier</key>         <string>org.moments-behavior.red</string>
    <key>CFBundleExecutable</key>         <string>red</string>
    <key>CFBundleIconFile</key>           <string>red</string>
    <key>CFBundlePackageType</key>        <string>APPL</string>
    <key>CFBundleShortVersionString</key> <string>$VERSION</string>
    <key>CFBundleVersion</key>            <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>     <string>$MINOS</string>
    <key>NSHighResolutionCapable</key>    <true/>
</dict>
</plist>
EOF

echo "  signing (ad hoc)..."
for f in "$FW"/*.dylib "$CONTENTS/MacOS/red"; do
    codesign --force --sign - "$f" 2> /dev/null
done
codesign --force --sign - "$APP" 2> /dev/null
codesign --verify --deep --strict "$APP"

# Nothing may still point outside the app.
bad=0
for f in "$CONTENTS/MacOS/red" "$FW"/*.dylib; do
    while IFS= read -r n; do
        is_system "$n" && continue
        case "$n" in
        @rpath/*) [ -e "$FW/${n#@rpath/}" ] && continue ;;
        esac
        echo "  still linked outside the app: $(basename "$f") -> $n" >&2
        bad=1
    done < <(deps "$f")
done
[ "$bad" = 0 ] || exit 1

# Finder caches an app's icon by path, and this script replaces Red.app at the
# same path -- so a build after an icon change kept showing the old (or no)
# icon. Have Launch Services read this one afresh.
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$APP" 2> /dev/null || true

# One build in dist/ at a time: Red.app is replaced above, so the zips of
# earlier builds -- named after other commits -- go too rather than pile up.
ZIP="$OUT/red-$VERSION-macos$MINOS-arm64.zip"
rm -f "$OUT"/red-*-macos*-arm64.zip
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
echo "done: $APP ($(du -sm "$APP" | cut -f1) MB)"
echo "      $ZIP ($(du -m "$ZIP" | cut -f1) MB)"
echo "      needs macOS $MINOS+, Apple Silicon"
