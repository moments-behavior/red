# macOS Apple Silicon Port: Vulkan Backend and YUV→RGBA Conversion

## Overview

RED was originally written for Linux with NVIDIA GPUs, using NVDEC for hardware
video decode, CUDA for NV12→RGB color conversion, and OpenGL for rendering.
Porting to macOS Apple Silicon required replacing all three components. The port
was done in three phases, each protected behind `#ifdef __APPLE__` guards so the
Linux/NVIDIA path is completely untouched.

---

## Phase 1 — Initial macOS Port (FFmpeg + VideoToolbox + OpenGL)

**Commit:** `a7f371e port to macOS Apple Silicon`

### What changed

| Component | Linux original | macOS replacement |
|-----------|---------------|-------------------|
| Video decode | NVIDIA NVDEC via `NvDecoder` | FFmpeg `avcodec` with VideoToolbox hwaccel (`AV_HWDEVICE_TYPE_VIDEOTOOLBOX`) |
| Color conversion | CUDA kernel (NV12→RGB on GPU) | `av_hwframe_transfer_data` + `sws_scale` (CPU) |
| Frame buffer | CUDA device memory, PBO | `malloc` CPU buffer |
| Rendering | OpenGL + CUDA PBO upload | OpenGL `glTexSubImage2D` |

### How the decode pipeline worked

1. FFmpeg demuxes compressed packets from the video file.
2. `avcodec_send_packet` / `avcodec_receive_frame` submits work to the VideoToolbox
   XPC service, which runs the hardware H.264/HEVC decoder in a separate process.
3. The decoded frame is returned as a `CVPixelBuffer` in NV12 (YCbCr 4:2:0) format,
   accessible through `hw_frame->data[3]`.
4. `av_hwframe_transfer_data` copies the NV12 data from the VideoToolbox-managed
   buffer into a CPU `AVFrame`.
5. `sws_scale` (libswscale) converts NV12 → RGBA on the CPU, single-threaded.
6. The RGBA buffer is uploaded to an OpenGL texture with `glTexSubImage2D`.

### Performance

~0.30× real-time at 180 fps for a 3208×2200 3-camera dataset. This was well below
the target of 1× real-time, with `sws_scale` being the primary bottleneck.

---

## Phase 2 — Vulkan Rendering Backend

**Commit:** `8d33169 add Vulkan rendering backend for macOS`

### Motivation

macOS deprecated OpenGL in 2018 and it is removed from future OS versions.
Apple Silicon Macs run OpenGL through a compatibility layer over Metal, which
adds latency and limits performance. Vulkan on macOS is provided by
**MoltenVK**, which translates Vulkan calls to native Metal API calls with
minimal overhead.

### New files

- **`src/vulkan_context.h`** — All Vulkan state and operations in one header
  (single translation unit). Contains:
  - `VulkanContext` struct: instance, physical device, logical device, queue,
    swapchain, command pool, sync primitives, per-camera textures
  - `VulkanTexture` struct: `VkImage` + `VkImageView` + persistently-mapped
    staging `VkBuffer` + `VkDescriptorSet` used as `ImTextureID`
  - Functions: `vk_init`, `vk_begin_frame`, `vk_end_frame`, `vk_upload_texture`,
    `vk_allocate_textures`, `vk_cleanup`
- **`src/vulkan_context.cpp`** — Defines the `g_vk` global pointer.

### Key design decisions

**`VK_KHR_dynamic_rendering`** — Used instead of explicit `VkRenderPass` /
`VkFramebuffer` objects. This matches how Dear ImGui's Vulkan backend works
and significantly reduces boilerplate.

**1 frame in flight** — Simplifies synchronization. The fence
`in_flight_fences[0]` is waited on at the start of every frame, guaranteeing
the previous frame's command buffer has completed before recording a new one.

**Persistently mapped staging buffers** — Each camera texture has a
`VkBuffer` allocated with `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` that is permanently `vkMapMemory`'d.
When a new frame arrives, the RGBA data is `memcpy`'d into
`tex.staging_mapped`, and a `vkCmdCopyBufferToImage` command copies it to the
`VkImage` in the same command buffer that renders the Dear ImGui frame. No
per-frame map/unmap overhead.

**Shared `VkSampler`** — One sampler is created at init time and shared by
all camera textures, since all cameras use identical linear filtering settings.

**`vk_begin_frame` before `ImGui::NewFrame`** — The Vulkan command buffer
must be open when `vk_upload_texture` records copy commands, so `vk_begin_frame`
(which calls `vkBeginCommandBuffer`) is called before `ImGui::NewFrame`.

**GLFW_INCLUDE_VULKAN compile definition applied globally** — `imgui_impl_glfw.h`
includes `<GLFW/glfw3.h>` early in its header guard, before any code in the
including translation unit can set `GLFW_INCLUDE_VULKAN`. The definition must
therefore be set globally via CMake `target_compile_definitions` rather than as
a per-file `#define`.

### CMake changes

- Added `find_package(Vulkan REQUIRED)` and `Vulkan::Vulkan`
- Added MoltenVK: `-framework Metal -framework QuartzCore`
- Replaced `imgui_impl_opengl3.cpp` with `imgui_impl_vulkan.cpp`
- Removed GLEW / OpenGL framework

---

## Phase 3 — vImage Color Conversion (0.30× → 0.88×)

**Commit:** `4f174c4 improve macOS playback performance`

### What changed

The `sws_scale` CPU conversion was replaced with Apple's **vImage** framework
(part of the Accelerate umbrella framework), operating directly on the
VideoToolbox `CVPixelBuffer` without any intermediate copy.

### Old path (sws_scale)

```
VideoToolbox → CVPixelBuffer
    → av_hwframe_transfer_data  (DMA: GPU-managed → system RAM)
    → sws_scale                 (single-threaded SIMD NV12→RGBA)
    → malloc CPU buffer
    → memcpy to staging buffer
    → vkCmdCopyBufferToImage
```

### New path (vImage)

```
VideoToolbox → CVPixelBuffer
    → CVPixelBufferLockBaseAddress  (CPU access to existing buffer, no copy)
    → vImageConvert_420Yp8_CbCr8ToARGB8888  (SIMD + tiled multithreading)
    → malloc CPU buffer
    → memcpy to staging buffer
    → vkCmdCopyBufferToImage
```

### Why this is so much faster

**1. Eliminated `av_hwframe_transfer_data`.**
VideoToolbox allocates `CVPixelBuffer` in memory that is already CPU-accessible
(IOSurface-backed). `av_hwframe_transfer_data` was copying the entire frame
(~3208×2200×1.5 bytes ≈ 10 MB per camera) into a separate `AVFrame` buffer
before `sws_scale` could touch it. With `CVPixelBufferLockBaseAddress` we
read the data in-place — zero extra copy.

**2. vImage uses SIMD and automatic internal tiling.**
`vImageConvert_420Yp8_CbCr8ToARGB8888` is Apple's hand-optimized
implementation of YCbCr→RGBA conversion. On Apple Silicon it uses NEON
vector intrinsics and, with `kvImageNoFlags`, splits the image into tiles and
dispatches them across all available CPU cores via Grand Central Dispatch
automatically. `sws_scale` is single-threaded. With three decoder threads each
converting a ~3208×2200 frame, the GCD thread pool serves all three
simultaneously and saturates the CPU cores.

**3. Direct CVPixelBuffer plane access.**
`CVPixelBufferGetBaseAddressOfPlane` gives pointers to the Y and CbCr planes
separately with their actual hardware strides, which vImage consumes directly.
There is no pixel format negotiation or intermediate allocation.

**Measured result:** 0.30× → 0.88× real-time for a 180 fps, 3-camera,
3208×2200 dataset. The remaining 12% gap is VideoToolbox IPC latency:
profiling with the macOS `sample` tool showed decoder threads spending ~93%
of their time blocked in `mach_msg2_trap` waiting for the VideoToolbox XPC
daemon to return decoded frames. This is fundamental to how FFmpeg wraps
VideoToolbox and cannot be improved without a deeper architectural change
(see Future Improvements).

### Other fixes in this commit

- **Directory argument:** When launched with a directory path (e.g.
  `./red /path/to/project_dir`), the program now automatically resolves the
  sibling `.redproj` file instead of crashing with a JSON parse error (macOS
  `std::ifstream` silently "opens" a directory but reads zero bytes).
- **Font and ini paths:** `imgui.ini` and font files are now resolved using
  absolute paths derived from `argv[0]`, independent of the working directory.
- **Decoder thread QoS:** `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`
  is set at decoder thread startup to ensure the scheduler always places them
  on performance (P) cores.

---

## Profiling Results

Tool: macOS `sample` utility, 5-second capture during playback.

| Thread | Time in | % |
|--------|---------|---|
| Main (render) | `CAMetalLayer nextDrawable` vsync wait | ~47% |
| Decoder (per camera) | `mach_msg2_trap` (VideoToolbox XPC IPC) | ~93% |
| vImage GCD pool (~10 threads) | `vImageConvert_420Yp8_CbCr8ToARGB8888` | active |

The bottleneck is VideoToolbox IPC, not CPU conversion. The main thread's
47% vsync wait is expected and harmless (display is not CPU-bound).

---

## Future Performance Improvements

### 1. Async VideoToolbox decode (high impact)

The 93% IPC wait comes from FFmpeg calling `VTDecompressionSessionDecodeFrame`
synchronously: it submits a frame to VideoToolbox and then blocks until the
decoded `CVPixelBuffer` is returned. The VideoToolbox API supports fully
asynchronous decode with an output callback, but FFmpeg's AVCodec wrapper
does not expose this.

**Fix:** Replace the FFmpeg `avcodec` path with direct VideoToolbox API calls
(`VTDecompressionSessionCreate` / `VTDecompressionSessionDecodeFrame` with
`kVTDecodeFrame_EnableAsynchronousDecompression`). The decoder thread would
submit compressed packets and receive decoded frames via a callback, allowing
the CPU to prepare the next packet while the hardware decoder is working.
This would likely push playback speed from 0.88× to 1.5–2×+ real-time.

### 2. GPU YUV→RGBA via Metal compute (high impact)

The vImage conversion still runs on the CPU and requires a `memcpy` into the
Vulkan staging buffer. VideoToolbox returns frames as `CVPixelBuffer` backed
by `IOSurface`. An `IOSurface` can be imported directly as a Metal texture
using `CVMetalTextureCacheCreateTextureFromImage`, making the decoded frame
available on the GPU without any CPU copy.

A Metal compute shader could then convert NV12→RGBA in-place on the GPU,
writing directly into a Metal texture. That texture could then be shared with
Vulkan via `VK_EXT_metal_objects`, eliminating both the vImage conversion and
the staging buffer `memcpy`.

### 3. Switch from Vulkan/MoltenVK to native Metal rendering (moderate impact)

MoltenVK translates every Vulkan API call to Metal at runtime, adding a thin
but non-zero overhead. Replacing `imgui_impl_vulkan` with `imgui_impl_metal`
and driving Metal directly would reduce this overhead and enable tighter
integration with `IOSurface`-backed textures (see above).

### 4. H.265/HEVC re-encoding of source videos (moderate impact)

Apple Silicon's video encoder/decoder hardware has dedicated HEVC accelerators.
HEVC achieves similar visual quality at roughly half the bitrate of H.264, which
means smaller I/O, fewer bits to send to the decoder, and potentially higher
sustained decode throughput. Re-encoding the behavioral video archive as HEVC
before labeling sessions would benefit all future use.

### 5. Reduce staging buffer copies (low impact)

The current path is: vImage writes RGBA into a `malloc` buffer, then
`memcpy` copies it into the persistently-mapped Vulkan staging buffer. If
vImage wrote directly into the staging buffer, the intermediate `malloc`
and `memcpy` would be eliminated. An experiment was conducted (the "zero-copy
slot buffer" approach) but showed a regression (0.88× → 0.83×) because
pinning large HOST_COHERENT allocations (5.1 GB total across all buffer slots)
caused TLB pressure exceeding the copy savings. A more targeted approach —
one staging buffer per camera rather than per slot — avoids this and should
yield a small improvement.
