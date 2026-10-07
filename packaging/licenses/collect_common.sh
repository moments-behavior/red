#!/usr/bin/env bash
# Lay out a package's licenses/ folder: Red's own licence, the README that
# explains the folder, the fonts' and vendored code's licences, and the full
# licence texts. Each platform's packaging then adds third_party/ for the
# shared libraries it bundles. (make_zip.ps1 does the same on Windows.)
#
#     packaging/licenses/collect_common.sh <dest>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO/packaging/licenses"
DEST="$1"

mkdir -p "$DEST/vendored" "$DEST/third_party"
cp "$REPO/LICENSE" "$DEST/LICENSE.txt"
cp "$HERE/README.txt" "$HERE/fonts.txt" \
   "$HERE/GPL-3.0.txt" "$HERE/Apache-2.0.txt" "$HERE/OFL-1.1.txt" "$DEST/"
cp "$HERE"/vendored/*.txt "$DEST/vendored/"
# A missing file means a submodule changed its layout: stop rather than ship
# without its notice.
grep -vE '^(#|$)' "$HERE/vendored.txt" | while read -r name file; do
    [ -f "$REPO/$file" ] || { echo "licence file missing: $file" >&2; exit 1; }
    cp "$REPO/$file" "$DEST/vendored/$name.txt"
done
