#!/usr/bin/env bash
set -euo pipefail

APP_NAME="red"

# --- Paths (run from project root) ---
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_SRC="$SRC_DIR/release/$APP_NAME"
FONTS_SRC="$SRC_DIR/fonts"
APP_ICON="$SRC_DIR/icon.png"   # launcher icon (any size OK)

# Optional install prefix (default: ~/.local)
PREFIX="${1:-"$HOME/.local"}"

# XDG dirs
XDG_DATA_HOME_DEFAULT="$HOME/.local/share"
XDG_DATA_HOME="${XDG_DATA_HOME:-$XDG_DATA_HOME_DEFAULT}"

# Targets
BINDIR="$PREFIX/bin"
APPDIR="$PREFIX/opt/$APP_NAME"
DESKTOP_DIR="$XDG_DATA_HOME/applications"
MIME_PKGS_DIR="$XDG_DATA_HOME/mime/packages"

# MIME
MIME_TYPE="application/x-red-project"
MIME_XML="$MIME_PKGS_DIR/red-project.xml"
GLOB_PATTERN="*.redproj"

echo "[*] Installing $APP_NAME ..."

# --- Sanity checks ---
[[ -f "$BIN_SRC" ]]  || { echo "ERROR: $BIN_SRC not found"; exit 1; }
[[ -f "$APP_ICON" ]] || { echo "ERROR: $APP_ICON not found"; exit 1; }

# --- Create dirs ---
mkdir -p "$APPDIR" "$BINDIR" "$DESKTOP_DIR" "$MIME_PKGS_DIR"

# --- Install binary (+fonts if present) ---
install -m 755 "$BIN_SRC" "$APPDIR/$APP_NAME"
if [[ -d "$FONTS_SRC" ]]; then
  mkdir -p "$APPDIR/fonts"
  cp -a "$FONTS_SRC"/. "$APPDIR/fonts"/
fi

# --- App launcher icon ---
install -m 644 "$APP_ICON" "$APPDIR/$APP_NAME.png"

# --- Wrapper ---
WRAPPER="$BINDIR/$APP_NAME"
cat > "$WRAPPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../opt/red" && pwd)"
cd "$APPDIR"
exec "$APPDIR/red" "$@"
EOF
chmod +x "$WRAPPER"

# --- .desktop entry ---
DESKTOP_FILE="$DESKTOP_DIR/$APP_NAME.desktop"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Red
GenericName=Project Editor
Comment=Open Red project files
TryExec=$WRAPPER
Exec=$WRAPPER %f
Icon=$APPDIR/$APP_NAME.png
Terminal=false
Categories=Graphics;Utility;
MimeType=$MIME_TYPE;
# StartupWMClass=red
EOF

# --- MIME definition (no file icon) ---
cat > "$MIME_XML" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="$MIME_TYPE">
    <comment>Red project file</comment>
    <glob pattern="$GLOB_PATTERN"/>
  </mime-type>
</mime-info>
EOF

# --- Refresh caches ---
command -v update-mime-database >/dev/null && update-mime-database "$XDG_DATA_HOME/mime" || true
command -v update-desktop-database >/dev/null && update-desktop-database "$DESKTOP_DIR" || true

# --- Make Red the default handler for .redproj ---
command -v xdg-mime >/dev/null && xdg-mime default "$(basename "$DESKTOP_FILE")" "$MIME_TYPE" || true

echo
echo "✔ Install complete."
echo "➤ Run: $APP_NAME"
echo "   (Make sure '$BINDIR' is in your PATH. If not, add this to your shell rc:)"
echo "     export PATH=\"\$HOME/.local/bin:\$PATH\""
