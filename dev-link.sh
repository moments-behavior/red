#!/usr/bin/env bash
#
# dev-link.sh — wire every way of launching red to the live build output.
#
# Makes the desktop shortcut, the `red` terminal command, and ./release/red all
# resolve to the SAME binary (~/src/red/release/red), so a plain rebuild
# (`cmake --build release -j`) updates every launcher at once. No copy, no drift.
#
# Run this ONCE now, and again only if something breaks the link — notably
# ./install.sh, which replaces the symlink with a standalone *copy* (that's
# install.sh's job: real installs for other machines). For day-to-day dev you
# just rebuild; you do not need install.sh.
#
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="$SRC_DIR/release/red"

# ~/.local/bin/red is used by BOTH the desktop shortcut (Exec=) and the terminal
# `red` command (it's on PATH). Point it straight at the build output.
LAUNCHER="$HOME/.local/bin/red"
# Internal path the old wrapper used; keep it consistent in case anything refers
# to it, but it is no longer on the launch path.
OPT_BIN="$HOME/.local/opt/red/bin/red"

if [[ ! -x "$BUILD_BIN" ]]; then
  echo "ERROR: $BUILD_BIN not found."
  echo "       Build first:  cmake -S \"$SRC_DIR\" -B \"$SRC_DIR/release\" && cmake --build \"$SRC_DIR/release\" -j"
  exit 1
fi

mkdir -p "$(dirname "$LAUNCHER")" "$(dirname "$OPT_BIN")"
ln -sfn "$BUILD_BIN" "$LAUNCHER"
ln -sfn "$BUILD_BIN" "$OPT_BIN"

echo "[*] Linked launchers to live build:"
printf '    %-28s -> %s\n' "$LAUNCHER" "$(readlink -f "$LAUNCHER")"
printf '    %-28s -> %s\n' "$OPT_BIN"  "$(readlink -f "$OPT_BIN")"

# Verify all three entry points resolve to the same inode.
TARGET_REAL="$(readlink -f "$BUILD_BIN")"
ok=1
for p in "$BUILD_BIN" "$LAUNCHER" "$(command -v red || true)"; do
  [[ -n "$p" ]] || continue
  r="$(readlink -f "$p")"
  if [[ "$r" != "$TARGET_REAL" ]]; then
    echo "    MISMATCH: $p -> $r"
    ok=0
  fi
done

echo
if [[ "$ok" == 1 ]]; then
  echo "[✔] Desktop shortcut, \`red\`, and ./release/red all launch:"
  echo "    $TARGET_REAL"
  echo "    Rebuild any time with:  cmake --build \"$SRC_DIR/release\" -j"
else
  echo "[!] Some launcher does not resolve to the build output (see MISMATCH above)."
  echo "    If \`red\` mismatched, ensure ~/.local/bin precedes other red copies on PATH."
  exit 1
fi
