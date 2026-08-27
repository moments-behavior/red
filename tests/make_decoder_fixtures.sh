#!/usr/bin/env bash
# Generate the synthetic videos test_decoder checks against, and run the
# matrix. Each frame is a flat colour encoding its own display-order index --
# R = (N%8)*32, G = (N/8)*32 -- so a frame carrying the wrong image fails even
# when the frame number attached to it looks right. Blue is constant per file
# and is what catches a red/blue channel swap.
#
#   tests/make_decoder_fixtures.sh [outdir]     # default: build/decoder_fixtures
#
# Env: FFMPEG=/path/to/ffmpeg      pick a specific binary
#      RED_FIXTURE_ENCODER=...     force libx264 or h264_nvenc
#      TEST_DECODER=...            default ./release/test_decoder
set -uo pipefail

OUT="${1:-build/decoder_fixtures}"
BIN="${TEST_DECODER:-./release/test_decoder}"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Find an ffmpeg that can encode H.264. The custom NVDEC build some rigs use is
# decode-focused and ships without libx264, so search rather than assume -- and
# prefer libx264, whose GOP structure can be pinned exactly.
# ---------------------------------------------------------------------------
have_enc() { "$1" -hide_banner -encoders 2>/dev/null | grep -q "[ ]$2[ ]"; }

FF=""; ENC="${RED_FIXTURE_ENCODER:-}"
for cand in "${FFMPEG:-}" ffmpeg /usr/bin/ffmpeg /usr/local/bin/ffmpeg \
            "$HOME/nvidia/ffmpeg/build/bin/ffmpeg"; do
    [ -n "$cand" ] || continue
    command -v "$cand" >/dev/null 2>&1 || continue
    if [ -n "$ENC" ]; then
        have_enc "$cand" "$ENC" && { FF=$cand; break; }; continue
    fi
    have_enc "$cand" libx264 && { FF=$cand; ENC=libx264; break; }
    [ -z "$FF" ] && have_enc "$cand" h264_nvenc && { FF=$cand; ENC=h264_nvenc; }
done

if [ -z "$FF" ]; then
    echo "No ffmpeg with an H.264 encoder found." >&2
    echo "Checked: \$FFMPEG, ffmpeg, /usr/bin, /usr/local/bin, ~/nvidia/ffmpeg/build/bin" >&2
    echo "Install one (apt install ffmpeg) or set FFMPEG=/path/to/ffmpeg." >&2
    exit 2
fi
echo "ffmpeg: $FF  (encoder: $ENC)"
FP="$(dirname "$FF")/ffprobe"; command -v "$FP" >/dev/null 2>&1 || FP=ffprobe

# ---------------------------------------------------------------------------
# Encoder-specific knobs. Both fixtures need a STRICTLY regular GOP: red's
# FindClosestKeyFrameFNI assumes evenly spaced keyframes, so a stray scene-cut
# keyframe makes every backend miss its seek target and buries the real signal.
# ---------------------------------------------------------------------------
enc_args() { # gop bframes
    local gop=$1 bf=$2
    case "$ENC" in
      libx264)
        local p="scenecut=0"
        if [ "$bf" -gt 0 ]; then p="bframes=$bf:b-adapt=0:b-pyramid=normal:$p"
        else p="bframes=0:$p"; fi
        printf -- '-c:v libx264 -preset slow -crf 10 -g %s -keyint_min %s -x264-params %s' \
            "$gop" "$gop" "$p" ;;
      h264_nvenc)
        # -no-scenecut needs lookahead off; -forced-idr pins the interval.
        local ref=disabled; [ "$bf" -gt 0 ] && ref=middle
        printf -- '-c:v h264_nvenc -preset p7 -tune lossless -rc constqp -qp 10 -g %s -bf %s -b_ref_mode %s -rc-lookahead 0 -no-scenecut 1 -forced-idr 1' \
            "$gop" "$bf" "$ref" ;;
      *) echo "unsupported encoder $ENC" >&2; exit 2 ;;
    esac
}

gen() { # name gop blue bframes
    local name=$1 gop=$2 blue=$3 bf=$4
    # shellcheck disable=SC2046  # enc_args intentionally word-splits
    "$FF" -hide_banner -loglevel error -f lavfi -i "nullsrc=s=320x240:r=30:d=2" \
        -vf "format=gbrp,geq=r='mod(N\,8)*32':g='floor(N/8)*32':b='$blue',format=yuv420p" \
        $(enc_args "$gop" "$bf") -y "$OUT/$name.mp4" || return 1

    local types
    types=$("$FP" -v error -select_streams v -show_entries frame=pict_type \
            -of csv=p=0 "$OUT/$name.mp4" | tr -d '\r,' | tr -d '\n')
    echo "  $name.mp4  ${types:0:40}..."

    # A fixture with irregular keyframes tests the wrong thing -- say so rather
    # than letting it surface as mysterious seek failures.
    local i idx=0 bad=0
    for ((i = 0; i < ${#types}; i++)); do
        if [ "${types:i:1}" = "I" ]; then
            [ $((i % gop)) -ne 0 ] && bad=1
        elif [ $((i % gop)) -eq 0 ]; then bad=1; fi
    done
    if [ "$bad" -ne 0 ]; then
        echo "  WARNING: $name.mp4 keyframes are not every $gop frames." >&2
        echo "  This encoder ignored the fixed-GOP request; seek results will" >&2
        echo "  be misleading. Try FFMPEG=<an ffmpeg with libx264>." >&2
        return 1
    fi
    [ "$bf" -gt 0 ] && [[ "$types" != *B* ]] && \
        echo "  NOTE: no B-frames produced; the reordering case is untested." >&2
    return 0
}

echo "generating fixtures in $OUT"
genfail=0
# No B-frames: the shape a hardware rig encoder actually produces.
gen simple  10 0   0 || genfail=1
# B-pyramid: forces the decoder to reorder, which is where post-seek frame
# accounting goes wrong.
gen bframes 10 160 3 || genfail=1
[ "$genfail" -eq 0 ] || echo "(fixture warnings above -- results may be unreliable)" >&2

[ -x "$BIN" ] || { echo "no test_decoder at $BIN (set TEST_DECODER=)" >&2; exit 1; }

echo
fail=0
for backend in sw hw; do
    for f in "simple:0" "bframes:160"; do
        name=${f%%:*}; blue=${f##*:}
        printf '%-9s %-3s  ' "$name" "$backend"
        out=$(RED_DECODE_BACKEND=$backend "$BIN" "$OUT/$name.mp4" \
                --expect-blue "$blue" 2>&1) || fail=1
        echo "$out" | grep -E '^[0-9]+ passed' || echo "(no result)"
        echo "$out" | grep -E '^FAIL' | head -4 | sed 's/^/           /'
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
