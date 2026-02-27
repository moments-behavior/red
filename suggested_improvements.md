# Suggested Performance Improvements

Ranked by expected impact. The pipeline as of the Metal port runs at 1.0×
real-time for a 3-camera, 180 fps, 3208×2200 dataset. These are the remaining
opportunities.

---

## High Impact

### 1. Re-encode source videos as HEVC (H.265)

The single biggest remaining win. Apple Silicon has dedicated HEVC decode
hardware that runs faster and consumes less power than H.264. HEVC also
achieves similar quality at roughly half the bitrate, meaning less data to
read from disk and fewer bits for the decoder to process.

A simple FFmpeg pass over the archive is sufficient:

```bash
ffmpeg -i input.mp4 -c:v hevc_videotoolbox -q:v 60 output.mp4
```

This would benefit every future labeling session without any code changes.

### 2. Color space accuracy (BT.601 → BT.709) ✓ implemented

The NV12→RGBA compute shader in `metal_context.mm` currently uses BT.601
full-range coefficients:

```metal
float r = clamp(y + 1.402f   * uv.y,                   0.0f, 1.0f);
float g = clamp(y - 0.34414f * uv.x - 0.71414f * uv.y, 0.0f, 1.0f);
float b = clamp(y + 1.772f   * uv.x,                   0.0f, 1.0f);
```

BT.601 is correct for standard-definition content. Many modern machine vision
cameras output BT.709 (HD color space). For the current near-grayscale rat
videos this makes no visible difference, but for color video the greens and
reds will be subtly wrong. Worth verifying the camera specs and updating the
coefficients if needed.

BT.709 full-range coefficients for reference:
```metal
float r = clamp(y + 1.5748f  * uv.y,                   0.0f, 1.0f);
float g = clamp(y - 0.1873f  * uv.x - 0.4681f * uv.y,  0.0f, 1.0f);
float b = clamp(y + 1.8556f  * uv.x,                   0.0f, 1.0f);
```

---

## Medium Impact

### 3. Periodic `CVMetalTextureCacheFlush` ✓ implemented

`CVMetalTextureCache` holds internal references to textures imported from
`CVPixelBuffer`. Currently `CVMetalTextureCacheFlush` is only called in
`-dealloc` (on shutdown). Over long labeling sessions with many seeks and
buffer recycles, stale cache entries can accumulate and consume memory.

Fix: call `CVMetalTextureCacheFlush(_texCache, 0)` once per second or on
every seek in `metal_context.mm`. This is a one-liner addition to
`metal_begin_frame` or the seek path.

### 4. GPU profiling with Xcode Metal System Trace ✓ pending (requires macOS 15.6+ / full Xcode)

Now that the pipeline is GPU-side, the next bottleneck (if any) will only be
visible in GPU timelines. The CPU profiling done with `sample` during the
Vulkan phase is no longer sufficient.

**How to profile:**
1. Open Xcode → Instruments → Metal System Trace
2. Attach to a running `red` process during playback
3. Look for per-encoder GPU time, drawable wait time, and whether the NV12
   compute pass or the ImGui render pass is the limiting factor

This will reveal whether there is remaining headroom on the GPU side.

---

## Low Impact / Code Hygiene

### 5. `dispatchThreads` instead of `dispatchThreadgroups` ✓ implemented

The current compute dispatch in `metal_context.mm`:

```objc
MTLSize tg   = MTLSizeMake(16, 16, 1);
MTLSize grid = MTLSizeMake((w + 15) / 16, (h + 15) / 16, 1);
[enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
```

This requires a manual bounds check at the top of the shader:

```metal
if (gid.x >= out_rgba.get_width() || gid.y >= out_rgba.get_height()) return;
```

On Apple Silicon, `dispatchThreads:threadsPerThreadgroup:` handles
non-power-of-two frame sizes automatically and the bounds check can be
removed. Minor GPU improvement; mainly cleaner code.

```objc
MTLSize tg     = MTLSizeMake(16, 16, 1);
MTLSize pixels = MTLSizeMake(w, h, 1);
[enc dispatchThreads:pixels threadsPerThreadgroup:tg];
```

### 6. Increase `REORDER_DEPTH` for high-B-frame content ✓ implemented (4 → 8)

`REORDER_DEPTH = 4` in `vt_async_decoder.h` is sufficient for most H.264
with 1–2 consecutive B-frames. Some encoders use longer B-frame sequences
(up to 8). If out-of-order frame artifacts appear on a new video source,
increasing this constant to 8 is the fix.

---

## Already Optimal

- **Seek parallelism** — `seek_all_cameras` triggers all cameras simultaneously
  before waiting; seeks already run in parallel across decoder threads.
- **Display buffer depth** — 64 frames (~355 ms at 180 fps) gives the decoder
  ample lookahead.
- **Texture format** — output textures are BGRA8Unorm, matching both the VT
  output and the CAMetalLayer pixel format; no swizzle at any stage.
- **Command buffer structure** — compute and render passes share one
  `MTLCommandBuffer` per frame; fewer command buffers means less driver
  overhead.
