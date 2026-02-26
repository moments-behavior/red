# macOS Apple Silicon Port: Native Metal Rendering and Async VideoToolbox Decode

## Overview

This document covers the second major round of macOS performance work, building
directly on the Vulkan backend described in `Vulkan_upgrade_info.md`. At the
end of that effort the pipeline was:

```
VideoToolbox (sync, via FFmpeg avcodec) → CVPixelBuffer
    → vImage CPU NV12→RGBA  (~0.88× real-time; bottleneck: VT XPC IPC 93%)
    → memcpy to Vulkan staging buffer
    → vkCmdCopyBufferToImage → VkImage
    → MoltenVK → Metal → GPU display
```

Three overlapping bottlenecks remained:

1. **MoltenVK translation overhead** — every Vulkan call is translated to Metal
   at runtime by MoltenVK.
2. **CPU color conversion** — vImage is fast but still CPU-bound; the GPU sits
   idle during conversion.
3. **Synchronous VideoToolbox IPC** — decoder threads blocked ~93% of the time
   in `mach_msg2_trap` waiting for the VideoToolbox XPC daemon to return decoded
   frames.

All three are eliminated in this update. The changes were implemented in three
self-contained phases, each committed separately and each leaving the application
fully functional. The Linux/NVIDIA build is untouched throughout.

---

## Phase 1 — Native Metal Rendering (replaces Vulkan/MoltenVK)

**Commit:** `4a4e491 replace Vulkan/MoltenVK with native Metal + async VideoToolbox decode`

### What changed

| Component | Vulkan path | Metal path |
|-----------|------------|------------|
| Rendering API | Vulkan + MoltenVK translation | Native Metal (`CAMetalLayer`) |
| ImGui backend | `imgui_impl_vulkan` | `imgui_impl_metal` |
| Swap chain | `VkSwapchainKHR` | `CAMetalLayer.nextDrawable` |
| Camera textures | `VkImage` + `VkDescriptorSet` | `id<MTLTexture>` |
| CPU upload | `memcpy` → staging `VkBuffer` → `vkCmdCopyBufferToImage` | `[tex replaceRegion:withBytes:]` (zero-copy on Apple Silicon) |

### New files

- **`src/metal_context.h`** — C++ interface (no Objective-C types exposed).
  Declares all Metal init, frame, and texture functions so C++ translation
  units can call them without including any Apple headers.
- **`src/metal_context.mm`** — Objective-C++ implementation. Contains an
  internal `@interface MetalCtx : NSObject` that owns all Metal objects under
  ARC: `MTLDevice`, `MTLCommandQueue`, `CAMetalLayer`, per-camera `MTLTexture`
  array, and a `CVMetalTextureCacheRef` (managed manually since it is a CF type,
  not an ObjC object).

### How the Metal frame loop works

```objc
// Init (once)
NSWindow *nswin = glfwGetCocoaWindow(glfw_window);
CAMetalLayer *layer = [CAMetalLayer layer];
layer.device = MTLCreateSystemDefaultDevice();
layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
layer.contentsScale = nswin.screen.backingScaleFactor;  // Retina fix — see below
nswin.contentView.layer = layer;
nswin.contentView.wantsLayer = YES;

// Per frame
id<CAMetalDrawable> drawable = [layer nextDrawable];
MTLRenderPassDescriptor *rpd = ...;
rpd.colorAttachments[0].texture = drawable.texture;
id<MTLCommandBuffer> cmd = [queue commandBuffer];
ImGui_ImplMetal_NewFrame(rpd);          // must precede ImGui::NewFrame()
// ... ImGui::NewFrame / Render / EndFrame ...
id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cmd, enc);
[enc endEncoding];
[cmd presentDrawable:drawable];
[cmd commit];
```

### CPU texture upload on Apple Silicon

On Apple Silicon, all memory is unified — CPU and GPU share the same physical
DRAM. `MTLTexture` created with `MTLStorageModeShared` is directly CPU-writable
with no staging buffer:

```objc
[tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
       mipmapLevel:0
         withBytes:rgba_data
       bytesPerRow:w * 4];
```

This replaces the Vulkan path: `memcpy` → staging `VkBuffer` → fence →
`vkCmdCopyBufferToImage` → barrier. The Metal path is a single call with no
GPU synchronization required.

### Retina display fix

`CAMetalLayer.contentsScale` defaults to `1.0`. On a Retina display GLFW
reports a 2× framebuffer size (`glfwGetFramebufferSize`), so Dear ImGui sets
`io.DisplayFramebufferScale = (2, 2)` and renders fonts at 2× physical pixels.
With `contentsScale = 1.0` the Metal drawable is only 1× size, so ImGui's 2×
font atlas renders into a 1× drawable — every UI element appears at double its
intended logical size. The fix is one line:

```objc
layer.contentsScale = nswin.screen.backingScaleFactor;
```

### CMake changes

- `project(red LANGUAGES CXX OBJCXX)` — enables Objective-C++ compilation
- Removed `find_package(Vulkan)`, `Vulkan::Vulkan`, `GLFW_INCLUDE_VULKAN`
- Added `file(GLOB MM_SRC_FILES src/*.mm)` and `-fobjc-arc` per-file property
- Replaced `imgui_impl_vulkan.cpp` with `imgui_impl_metal.mm`
- Removed `-framework Accelerate` (vImage no longer used)
- Added `-framework Cocoa` (for `glfwGetCocoaWindow`)
- `vulkan_context.cpp` excluded from the source glob

---

## Phase 2 — GPU NV12→RGBA via Metal Compute Shader

### Motivation

The vImage conversion from Phase 3 of the Vulkan work ran on the CPU using
Grand Central Dispatch threads. While fast, it still required:

- Locking the `CVPixelBuffer` for CPU access
- Running NV12→RGBA on CPU cores shared with the decoder threads
- Copying the RGBA result into a staging buffer for GPU upload

VideoToolbox returns frames as `IOSurface`-backed `CVPixelBuffer` objects.
An `IOSurface` can be imported directly as a Metal texture with zero CPU
involvement using `CVMetalTextureCacheCreateTextureFromImage`.

### New pipeline

```
VideoToolbox → CVPixelBuffer (IOSurface-backed, no CPU access needed)
    → CVMetalTextureCache → MTLTexture (Y plane, R8Unorm)
    → CVMetalTextureCache → MTLTexture (UV plane, RG8Unorm)
    → Metal compute shader → MTLTexture (RGBA, output for ImGui display)
```

No CPU color conversion. No staging buffer. No `memcpy`.

### Metal compute shader

The NV12→RGBA kernel is embedded as a compile-time string in
`metal_context.mm` and compiled at runtime with `newLibraryWithSource:`:

```metal
kernel void nv12_to_rgba(
    texture2d<float, access::read>  y_plane  [[texture(0)]],
    texture2d<float, access::read>  uv_plane [[texture(1)]],
    texture2d<float, access::write> out_rgba [[texture(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= out_rgba.get_width() || gid.y >= out_rgba.get_height()) return;
    float  y  = y_plane.read(gid).r;
    float2 uv = uv_plane.read(gid / 2).rg - 0.5f;
    // BT.601 full-range coefficients
    float r = clamp(y + 1.402f   * uv.y,                   0.0f, 1.0f);
    float g = clamp(y - 0.34414f * uv.x - 0.71414f * uv.y, 0.0f, 1.0f);
    float b = clamp(y + 1.772f   * uv.x,                   0.0f, 1.0f);
    out_rgba.write(float4(r, g, b, 1.0f), gid);
}
```

Dispatched as 16×16 threadgroups covering the full frame:

```objc
MTLSize tg   = MTLSizeMake(16, 16, 1);
MTLSize grid = MTLSizeMake((w + 15) / 16, (h + 15) / 16, 1);
[enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
```

The compute encoder is recorded onto the same `MTLCommandBuffer` as the
ImGui render pass, so no additional synchronization is needed.

### CVPixelBuffer ownership

- The decoder thread calls `CFRetain` when storing a `CVPixelBufferRef` into
  a `PictureBuffer` slot and sets `available_to_write = false`.
- The main thread calls `metal_upload_pixelbuf`, which imports the buffer as
  Metal textures and dispatches the compute shader.
- When the main thread advances past the slot (frame advance loop), it calls
  `CFRelease` on the stored `CVPixelBufferRef` and sets
  `available_to_write = true`, returning the slot to the decoder.
- VideoToolbox reuses the underlying `IOSurface` only after the last
  `CFRelease`, so the GPU cannot read a buffer that VT has reclaimed.

### `PictureBuffer` change

```cpp
struct PictureBuffer {
    unsigned char   *frame;         // CPU RGBA buffer (Linux/image-loader path)
    int              frame_number;
    bool             available_to_write;
#ifdef __APPLE__
    CVPixelBufferRef pixel_buffer;  // macOS video path: IOSurface-backed VT output
#endif
};
```

---

## Phase 3 — Async VideoToolbox Decode

### Motivation

In both the original FFmpeg path and the vImage-accelerated path, the
decoder thread called `avcodec_receive_frame`, which internally calls
`VTDecompressionSessionDecodeFrame` and **blocks until VT returns the decoded
frame**. VideoToolbox runs the actual hardware decode in a separate XPC
process (`com.apple.videoprocessing.avconference`). Every frame required a
synchronous round-trip through the Mach IPC system, causing decoder threads
to spend ~93% of their time in `mach_msg2_trap`.

VideoToolbox's native API supports fully asynchronous decode with an output
callback. The decoder thread can submit a compressed packet and immediately
loop back to demux and submit the next one, while the hardware decoder works
in parallel and delivers decoded frames via the callback.

### New files

- **`src/vt_async_decoder.h`** — C++ class interface, usable from `.cpp`
  files without any Objective-C headers.
- **`src/vt_async_decoder.mm`** — Objective-C++ implementation.

### VTAsyncDecoder design

```cpp
class VTAsyncDecoder {
public:
    bool init(const uint8_t *extradata, int extradata_size, AVCodecID codec_id);
    void submit(const uint8_t *data, size_t size,
                int64_t pts, int64_t dts, double timebase_sec, bool is_keyframe);
    void submit_blocking(/* same args */);  // async submit + FinishDelayedFrames (for seek)
    CVPixelBufferRef pop_next();    // respects reorder depth; returns null if not ready
    CVPixelBufferRef drain_one();   // ignores reorder depth (used during seek)
    void flush();
    void destroy();
};
```

**Format description.** `VTDecompressionSessionCreate` requires a
`CMVideoFormatDescriptionRef` describing the codec parameters. For H.264, this
is created from the SPS and PPS NAL units extracted from the AVCC extradata
stored in the container (`FFmpegDemuxer::GetExtradata()`). For HEVC, parameter
sets are extracted from the HVCC configuration record.

**Annex-B → AVCC conversion.** `FFmpegDemuxer` applies the
`h264_mp4toannexb` / `hevc_mp4toannexb` bitstream filter and outputs packets
with Annex-B start-code prefixes (`00 00 00 01`). VideoToolbox requires AVCC
format (4-byte big-endian NAL length prefix). `annexb_to_avcc()` scans for
start codes, identifies NAL boundaries, and outputs length-prefixed NAL units.

**Async output callback.** VT calls the callback on an internal thread when
a frame is decoded:

```objc
static void output_callback(void *ctx, void *, OSStatus status,
                            VTDecodeInfoFlags, CVImageBufferRef image_buffer,
                            CMTime pts, CMTime) {
    if (status != noErr || !image_buffer) return;
    VTAsyncDecoder *self = (VTAsyncDecoder *)ctx;
    CVPixelBufferRef pb = (CVPixelBufferRef)image_buffer;
    CFRetain(pb);
    std::lock_guard<std::mutex> lk(self->mutex_);
    self->queue_.push({pb, pts});
}
```

**B-frame reorder queue.** H.264 video with B-frames can return decoded frames
out of presentation order. Frames are inserted into a `std::priority_queue`
min-heap sorted by PTS. `pop_next()` only emits a frame once the queue depth
exceeds `REORDER_DEPTH = 4`, ensuring that any out-of-order frames have
arrived before the earliest PTS is emitted.

**Seek path.** After a seek, the existing VT session is flushed and
destroyed, then recreated. `submit_blocking()` is used — it submits a packet
asynchronously and immediately calls `VTDecompressionSessionFinishDelayedFrames`,
which blocks until the output callback has fired. `drain_one()` retrieves the
decoded frame without waiting for reorder depth. This gives seek the
synchronous, frame-accurate behavior it requires while reusing the async
infrastructure.

### Decoder loop (simplified)

```cpp
while (!stop) {
    if (seek requested) {
        vt_dec.flush();
        demuxer->Seek(...);
        vt_dec.destroy();
        vt_dec.init(extradata, extradata_size, codec_id);
        // forward-decode to exact target frame using submit_blocking + drain_one
        continue;
    }

    demuxer->Demux(packet, size, pktinfo);
    vt_dec.submit(packet, size, pktinfo.pts, pktinfo.dts, timebase, ...);

    while (CVPixelBufferRef pb = vt_dec.pop_next()) {
        // wait for free display_buffer slot, store pb (retained)
    }
}
```

---

## Bug Fixes

### AVCC NAL truncation (`kVTVideoDecoderBadDataErr`)

`annexb_to_avcc()` used a while loop to scan for the end of each NAL unit:

```cpp
while (j + 2 < size) {   // exits when j >= size - 2
    if (next_start_code_at_j) break;
    j++;
}
// old code: only extended j to size if j == data_begin (empty NAL check)
```

The loop condition `j + 2 < size` causes the scan to stop 2 bytes before the
end of the buffer. For the final NAL unit in a packet, `j` would be `size - 2`
when the loop exited — not `size`. The last 2 bytes of every packet were
silently dropped. VideoToolbox received malformed AVCC data and returned
`kVTVideoDecoderBadDataErr` (-12909) on every frame.

**Fix:** Track whether the loop exited by finding a start code or by reaching
the end of the buffer:

```cpp
bool found_next = false;
while (j + 2 < size) {
    if (next_start_code_at_j) { found_next = true; break; }
    j++;
}
if (!found_next)
    j = size;  // NAL extends to the true end of the packet
```

---

## Performance Results

Dataset: 3-camera, 180 fps, 3208×2200 resolution.

| Pipeline | Playback speed |
|----------|---------------|
| FFmpeg + sws_scale + OpenGL | ~0.30× real-time |
| FFmpeg + vImage + Vulkan/MoltenVK | ~0.88× real-time |
| Async VT + Metal compute + native Metal | **1.0× real-time** |

The decoder threads no longer appear in `mach_msg2_trap`. The CPU is now
idle enough during playback that the main render thread hits vsync on every
frame, and the "Current Speed" indicator holds at 1.0×.
