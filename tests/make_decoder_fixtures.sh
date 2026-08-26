#!/usr/bin/env bash
# Generate the synthetic videos test_decoder checks against, and run the
# matrix. Each frame is a flat colour encoding its own display-order index --
# R = (N%8)*32, G = (N/8)*32 -- so a frame carrying the wrong image fails even
# when the frame number attached to it looks right. Blue is constant per file
# and is what catches a red/blue channel swap.
#
#   tests/make_decoder_fixtures.sh [outdir]     # default: build/decoder_fixtures
set -euo pipefail

OUT="${1:-build/decoder_fixtures}"
BIN="${TEST_DECODER:-./release/test_decoder}"
mkdir -p "$OUT"

gen() { # name gop blue extra-x264-params...
    local name=$1 gop=$2 blue=$3; shift 3
    ffmpeg -hide_banner -loglevel error -f lavfi \
        -i "nullsrc=s=320x240:r=30:d=2" \
        -vf "format=gbrp,geq=r='mod(N\,8)*32':g='floor(N/8)*32':b='$blue',format=yuv420p" \
        -c:v libx264 -preset slow -crf 10 -g "$gop" -keyint_min "$gop" \
        "$@" -y "$OUT/$name.mp4"
    echo "  $name.mp4  $(ffprobe -v error -select_streams v \
        -show_entries frame=pict_type -of csv=p=0 "$OUT/$name.mp4" \
        | tr -d '\n,' | head -c 40)..."
}

echo "generating fixtures in $OUT"
# No B-frames: the shape a hardware rig encoder actually produces.
gen simple 10 0   -x264-params "bframes=0:scenecut=0"
# B-pyramid, fixed GOP: forces the decoder to reorder, which is where frame
# accounting after a seek goes wrong.
gen bframes 10 160 -x264-params "bframes=3:b-adapt=0:b-pyramid=normal:scenecut=0"

[ -x "$BIN" ] || { echo "no test_decoder at $BIN (set TEST_DECODER=)"; exit 1; }

echo
fail=0
for backend in sw hw; do
    for f in "simple:0" "bframes:160"; do
        name=${f%%:*}; blue=${f##*:}
        printf '%-9s %-3s  ' "$name" "$backend"
        if out=$(RED_DECODE_BACKEND=$backend "$BIN" "$OUT/$name.mp4" \
                    --expect-blue "$blue" 2>&1); then
            echo "$out" | grep -E '^[0-9]+ passed'
        else
            echo "$out" | grep -E '^[0-9]+ passed|FAIL' | head -5
            fail=1
        fi
    done
done
# Thread count changes libavcodec's output latency, which the post-seek frame
# accounting depends on; 1 is the case where the keyframe comes back instantly.
for t in 1 2 8; do
    printf 'threads=%-2s sw   ' "$t"
    RED_DECODE_BACKEND=sw RED_SW_DECODE_THREADS=$t "$BIN" "$OUT/simple.mp4" \
        --expect-blue 0 2>&1 | grep -E '^[0-9]+ passed' || fail=1
done
exit $fail
