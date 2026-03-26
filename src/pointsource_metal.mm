#ifdef __APPLE__

#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreVideo/CVMetalTextureCache.h>

#include "pointsource_metal.h"
#include <stdio.h>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// Metal shader source (4 compute kernels embedded as string)
// ---------------------------------------------------------------------------

static const char *kLaserShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Kernel 1: Threshold green channel → binary mask (R8Uint)
kernel void threshold_green(
    texture2d<float, access::read>  src   [[texture(0)]],
    texture2d<uint, access::write>  mask  [[texture(1)]],
    constant int &green_threshold         [[buffer(0)]],
    constant int &green_dominance         [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) return;

    float4 c = src.read(gid); // BGRA as float [0,1]
    // Metal reads BGRA8Unorm as (r=R, g=G, b=B, a=A) — normalized RGBA order
    int r = int(c.r * 255.0);
    int g = int(c.g * 255.0);
    int b = int(c.b * 255.0);

    int gt = green_threshold;
    int gd = green_dominance;

    bool pass = (g > gt) && (g >= r) && (g >= b) &&
                ((g > r + gd && g > b + gd) || (g > 200 && g >= r && g >= b));

    mask.write(uint4(pass ? 1u : 0u, 0, 0, 0), gid);
}

// Kernel 2: Erode 3×3 — pixel stays 1 only if all 9 neighbors are 1
kernel void erode_3x3(
    texture2d<uint, access::read>   src  [[texture(0)]],
    texture2d<uint, access::write>  dst  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = src.get_width();
    uint h = src.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Border pixels forced to 0
    if (gid.x == 0 || gid.y == 0 || gid.x == w - 1 || gid.y == h - 1) {
        dst.write(uint4(0, 0, 0, 0), gid);
        return;
    }

    bool all_on = true;
    for (int dy = -1; dy <= 1 && all_on; dy++) {
        for (int dx = -1; dx <= 1 && all_on; dx++) {
            if (src.read(uint2(gid.x + dx, gid.y + dy)).r == 0u)
                all_on = false;
        }
    }
    dst.write(uint4(all_on ? 1u : 0u, 0, 0, 0), gid);
}

// Kernel 3: Dilate 3×3 — pixel becomes 1 if any of 9 neighbors is 1
kernel void dilate_3x3(
    texture2d<uint, access::read>   src  [[texture(0)]],
    texture2d<uint, access::write>  dst  [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = src.get_width();
    uint h = src.get_height();
    if (gid.x >= w || gid.y >= h) return;

    // Border pixels stay 0 (matching CPU behavior: erode/dilate skip y=0,h-1,x=0,w-1)
    if (gid.x == 0 || gid.y == 0 || gid.x == w - 1 || gid.y == h - 1) {
        dst.write(uint4(0, 0, 0, 0), gid);
        return;
    }

    bool any_on = false;
    for (int dy = -1; dy <= 1 && !any_on; dy++) {
        for (int dx = -1; dx <= 1 && !any_on; dx++) {
            if (src.read(uint2(gid.x + dx, gid.y + dy)).r != 0u)
                any_on = true;
        }
    }
    dst.write(uint4(any_on ? 1u : 0u, 0, 0, 0), gid);
}

// Kernel 4: Reduce centroid — atomically accumulate count, sum_gx, sum_gy, sum_g
// Also track bounding box via atomic min/max for multi-blob rejection.
// result[0] = count, result[1] = sum_gx, result[2] = sum_gy, result[3] = sum_g
// result[4] = min_x, result[5] = min_y (init to 0xFFFFFFFF)
// result[6] = max_x, result[7] = max_y (init to 0)
kernel void reduce_centroid(
    texture2d<uint, access::read>   mask  [[texture(0)]],
    texture2d<float, access::read>  src   [[texture(1)]],
    device atomic_uint *result            [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = mask.get_width();
    uint h = mask.get_height();
    if (gid.x >= w || gid.y >= h) return;

    if (mask.read(gid).r == 0u) return;

    float4 c = src.read(gid);
    uint g = uint(c.g * 255.0);

    atomic_fetch_add_explicit(&result[0], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&result[1], g * gid.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&result[2], g * gid.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&result[3], g, memory_order_relaxed);

    // Bounding box
    atomic_fetch_min_explicit(&result[4], gid.x, memory_order_relaxed);
    atomic_fetch_min_explicit(&result[5], gid.y, memory_order_relaxed);
    atomic_fetch_max_explicit(&result[6], gid.x, memory_order_relaxed);
    atomic_fetch_max_explicit(&result[7], gid.y, memory_order_relaxed);
}
// Kernel 5: Colorize visualization — writes RGBA to shared buffer
// Also counts total mask pixels via atomic counter.
kernel void colorize_viz(
    texture2d<uint, access::read>  mask    [[texture(0)]],
    texture2d<uint, access::read>  dilated [[texture(1)]],
    device uint8_t *rgba_out              [[buffer(0)]],
    device atomic_uint *mask_count        [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint w = mask.get_width();
    uint h = mask.get_height();
    if (gid.x >= w || gid.y >= h) return;

    uint idx = gid.y * w + gid.x;
    uint base = idx * 4;

    bool m = mask.read(gid).r != 0u;
    bool d = dilated.read(gid).r != 0u;

    if (m) atomic_fetch_add_explicit(mask_count, 1u, memory_order_relaxed);

    if (d) {
        // Green placeholder — CPU will recolor per-blob classification
        rgba_out[base]   = 0;   rgba_out[base+1] = 255;
        rgba_out[base+2] = 0;   rgba_out[base+3] = 255;
    } else if (m) {
        // Gray — threshold passed but filtered by erode/dilate
        rgba_out[base]   = 60;  rgba_out[base+1] = 60;
        rgba_out[base+2] = 60;  rgba_out[base+3] = 255;
    } else {
        // Black background
        rgba_out[base]   = 0;   rgba_out[base+1] = 0;
        rgba_out[base+2] = 0;   rgba_out[base+3] = 255;
    }
}
)METAL";


// ---------------------------------------------------------------------------
// PointSourceMetalContext — holds all Metal state
// ---------------------------------------------------------------------------

struct PointSourceMetalContext {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLComputePipelineState> pso_threshold;
    id<MTLComputePipelineState> pso_erode;
    id<MTLComputePipelineState> pso_dilate;
    id<MTLComputePipelineState> pso_reduce;
    id<MTLComputePipelineState> pso_colorize;
    CVMetalTextureCacheRef      texCache;

    // Pre-allocated resources reused across calls (avoid per-frame allocation)
    int alloc_width = 0, alloc_height = 0;
    id<MTLTexture> maskTex;
    id<MTLTexture> erodedTex;
    id<MTLTexture> dilatedTex;
    id<MTLBuffer>  rgbaBuf;       // viz path: w*h*4 RGBA
    id<MTLBuffer>  maskCountBuf;  // viz path: atomic mask counter
    id<MTLBuffer>  threshBuf;     // viz path: green_threshold param
    id<MTLBuffer>  domBuf;        // viz path: green_dominance param
};


PointSourceMetalHandle pointsource_metal_create() {
    @autoreleasepool {
        auto *ctx = new PointSourceMetalContext{};

        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "[LaserMetal] No Metal device found\n");
            delete ctx;
            return nullptr;
        }

        ctx->queue = [ctx->device newCommandQueue];

        // Compile shader library from source
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:kLaserShaderSource];
        id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src
                                                       options:nil
                                                         error:&error];
        if (!lib) {
            fprintf(stderr, "[LaserMetal] Shader compile error: %s\n",
                    error.localizedDescription.UTF8String);
            delete ctx;
            return nullptr;
        }

        // Create pipeline states
        auto make_pso = [&](const char *name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                  [NSString stringWithUTF8String:name]];
            if (!fn) {
                fprintf(stderr, "[LaserMetal] Function '%s' not found\n", name);
                return nil;
            }
            NSError *e = nil;
            id<MTLComputePipelineState> pso =
                [ctx->device newComputePipelineStateWithFunction:fn error:&e];
            if (!pso) {
                fprintf(stderr, "[LaserMetal] PSO '%s' error: %s\n",
                        name, e.localizedDescription.UTF8String);
            }
            return pso;
        };

        ctx->pso_threshold = make_pso("threshold_green");
        ctx->pso_erode     = make_pso("erode_3x3");
        ctx->pso_dilate    = make_pso("dilate_3x3");
        ctx->pso_reduce    = make_pso("reduce_centroid");
        ctx->pso_colorize  = make_pso("colorize_viz");

        if (!ctx->pso_threshold || !ctx->pso_erode ||
            !ctx->pso_dilate || !ctx->pso_reduce || !ctx->pso_colorize) {
            delete ctx;
            return nullptr;
        }

        // Create texture cache for zero-copy CVPixelBuffer import
        CVReturn r = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL,
                                               ctx->device, NULL,
                                               &ctx->texCache);
        if (r != kCVReturnSuccess) {
            fprintf(stderr, "[LaserMetal] CVMetalTextureCacheCreate failed: %d\n", r);
            delete ctx;
            return nullptr;
        }

        return ctx;
    }
}

// ── Shared GPU pipeline: threshold → erode → dilate → colorize readback ──
// Returns the RGBA buffer with green=(0,255,0) for dilated pixels, ready for BFS.
// Caller must CFRelease cvTex when done.
struct MetalPipelineResult {
    uint8_t *rgba = nullptr;     // pointer into maskBuf contents
    id<MTLBuffer> maskBuf;       // retains the RGBA data
    CVMetalTextureRef cvTex = nullptr;
    int width = 0, height = 0;
    bool ok = false;
};

static MetalPipelineResult run_metal_green_pipeline(
    PointSourceMetalHandle ctx, CVPixelBufferRef pixel_buffer,
    int green_threshold, int green_dominance) {

    MetalPipelineResult r;
    if (!ctx || !pixel_buffer) return r;

    r.width  = (int)CVPixelBufferGetWidth(pixel_buffer);
    r.height = (int)CVPixelBufferGetHeight(pixel_buffer);

    CVReturn rv = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, ctx->texCache, pixel_buffer, NULL,
        MTLPixelFormatBGRA8Unorm, r.width, r.height, 0, &r.cvTex);
    if (rv != kCVReturnSuccess || !r.cvTex) return r;
    id<MTLTexture> srcTex = CVMetalTextureGetTexture(r.cvTex);

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Uint
                                                          width:r.width height:r.height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;

    id<MTLTexture> maskTex    = [ctx->device newTextureWithDescriptor:desc];
    id<MTLTexture> erodedTex  = [ctx->device newTextureWithDescriptor:desc];
    id<MTLTexture> dilatedTex = [ctx->device newTextureWithDescriptor:desc];

    int npixels = r.width * r.height;
    r.maskBuf = [ctx->device newBufferWithLength:npixels * 4
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> dummyCount =
        [ctx->device newBufferWithLength:sizeof(uint32_t)
                                 options:MTLResourceStorageModeShared];
    *(uint32_t *)dummyCount.contents = 0;

    MTLSize gridSize = MTLSizeMake(r.width, r.height, 1);
    MTLSize tgSize   = MTLSizeMake(16, 16, 1);

    id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];

    // Threshold
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_threshold];
        [enc setTexture:srcTex  atIndex:0];
        [enc setTexture:maskTex atIndex:1];
        [enc setBytes:&green_threshold length:sizeof(int) atIndex:0];
        [enc setBytes:&green_dominance length:sizeof(int) atIndex:1];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
    }
    // Erode
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_erode];
        [enc setTexture:maskTex   atIndex:0];
        [enc setTexture:erodedTex atIndex:1];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
    }
    // Dilate
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_dilate];
        [enc setTexture:erodedTex  atIndex:0];
        [enc setTexture:dilatedTex atIndex:1];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
    }
    // Colorize (writes RGBA with green=(0,255,0) for dilated pixels)
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_colorize];
        [enc setTexture:maskTex    atIndex:0];
        [enc setTexture:dilatedTex atIndex:1];
        [enc setBuffer:r.maskBuf  offset:0 atIndex:0];
        [enc setBuffer:dummyCount offset:0 atIndex:1];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
    }

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    r.rgba = (uint8_t *)r.maskBuf.contents;
    r.ok = true;
    return r;
}

// ── Shared BFS: find all valid-sized connected components in the RGBA buffer ──
// Green pixels (0,255,0) are dilated; BFS marks them visited by setting R=1.
// Returns blob centroids using source pixel buffer for intensity weighting.
static std::vector<PointSourceMetalBlob> bfs_find_blobs(
    uint8_t *rgba, int width, int height,
    const uint8_t *src_bgra, int src_stride,
    int min_blob_pixels, int max_blob_pixels) {

    std::vector<PointSourceMetalBlob> blobs;
    std::vector<int> queue;
    queue.reserve(1024);

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            int base = idx * 4;
            if (rgba[base] != 0 || rgba[base+1] != 255 || rgba[base+2] != 0)
                continue;

            // BFS flood-fill
            queue.clear();
            queue.push_back(idx);
            rgba[base] = 1; // mark visited

            for (size_t qi = 0; qi < queue.size(); qi++) {
                int ci = queue[qi];
                int cx_pos = ci % width, cy_pos = ci / width;
                const int neighbors[4] = {
                    (cx_pos > 0)          ? ci - 1     : -1,
                    (cx_pos < width - 1)  ? ci + 1     : -1,
                    (cy_pos > 0)          ? ci - width  : -1,
                    (cy_pos < height - 1) ? ci + width  : -1
                };
                for (int ni : neighbors) {
                    if (ni < 0) continue;
                    int nb = ni * 4;
                    if (rgba[nb] == 0 && rgba[nb+1] == 255 && rgba[nb+2] == 0) {
                        rgba[nb] = 1;
                        queue.push_back(ni);
                    }
                }
            }

            int blob_size = (int)queue.size();
            if (blob_size < min_blob_pixels || blob_size > max_blob_pixels)
                continue;

            // Compute intensity-weighted centroid
            double sx = 0, sy = 0, sw = 0;
            for (int pi : queue) {
                int px = pi % width, py = pi / width;
                double g = src_bgra[py * src_stride + px * 4 + 1]; // green (BGRA)
                sx += px * g; sy += py * g; sw += g;
            }
            if (sw > 0)
                blobs.push_back({sx / sw, sy / sw, blob_size});
        }
    }
    return blobs;
}

PointSourceMetalSpot pointsource_metal_detect(PointSourceMetalHandle ctx,
                                   CVPixelBufferRef pixel_buffer,
                                   int green_threshold,
                                   int green_dominance,
                                   int min_blob_pixels,
                                   int max_blob_pixels,
                                   bool smart_blob) {
    PointSourceMetalSpot result = {0, 0, 0, false};
    if (!ctx || !pixel_buffer) return result;

    @autoreleasepool {
        auto pipeline = run_metal_green_pipeline(ctx, pixel_buffer,
                                                  green_threshold, green_dominance);
        if (!pipeline.ok) return result;

        // Use shared BFS to find all blobs, then pick based on mode
        CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
        const uint8_t *src_bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pixel_buffer);
        int src_stride = (int)CVPixelBufferGetBytesPerRow(pixel_buffer);

        auto blobs = bfs_find_blobs(pipeline.rgba, pipeline.width, pipeline.height,
                                     src_bgra, src_stride,
                                     min_blob_pixels, max_blob_pixels);
        CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
        CFRelease(pipeline.cvTex);

        if (!blobs.empty()) {
            if (blobs.size() == 1 || smart_blob) {
                // Pick the largest blob
                auto *best = &blobs[0];
                for (auto &b : blobs)
                    if (b.pixel_count > best->pixel_count) best = &b;
                result.cx = best->cx;
                result.cy = best->cy;
                result.pixel_count = best->pixel_count;
                result.found = true;
            }
            // else: multiple blobs without smart_blob → reject (result.found stays false)
        }
    }

    return result;
}

// Smart Blob v3: return ALL valid-sized blobs for deferred resolution.
std::vector<PointSourceMetalBlob> pointsource_metal_detect_all(
    PointSourceMetalHandle ctx,
    CVPixelBufferRef pixel_buffer,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels) {

    std::vector<PointSourceMetalBlob> blobs;
    if (!ctx || !pixel_buffer) return blobs;

    @autoreleasepool {
        auto pipeline = run_metal_green_pipeline(ctx, pixel_buffer,
                                                  green_threshold, green_dominance);
        if (!pipeline.ok) return blobs;

        CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
        const uint8_t *src_bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(pixel_buffer);
        int src_stride = (int)CVPixelBufferGetBytesPerRow(pixel_buffer);

        blobs = bfs_find_blobs(pipeline.rgba, pipeline.width, pipeline.height,
                                src_bgra, src_stride,
                                min_blob_pixels, max_blob_pixels);
        CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
        CFRelease(pipeline.cvTex);
    }
    return blobs;
}


PointSourceMetalVizResult pointsource_metal_detect_viz(
    PointSourceMetalHandle ctx,
    CVPixelBufferRef pixel_buffer,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels,
    uint8_t *rgba_out) {

    PointSourceMetalVizResult result = {};
    if (!ctx || !pixel_buffer || !rgba_out) return result;

    @autoreleasepool {
        int width  = (int)CVPixelBufferGetWidth(pixel_buffer);
        int height = (int)CVPixelBufferGetHeight(pixel_buffer);
        int npixels = width * height;

        // Zero-copy: wrap CVPixelBuffer as MTLTexture via IOSurface
        CVMetalTextureRef cvTex = NULL;
        CVReturn rv = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, ctx->texCache, pixel_buffer, NULL,
            MTLPixelFormatBGRA8Unorm, width, height, 0, &cvTex);
        if (rv != kCVReturnSuccess || !cvTex) return result;
        id<MTLTexture> srcTex = CVMetalTextureGetTexture(cvTex);

        // Pre-allocate / reuse intermediate textures and buffers
        if (ctx->alloc_width != width || ctx->alloc_height != height) {
            MTLTextureDescriptor *maskDesc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Uint
                                                                  width:width
                                                                 height:height
                                                              mipmapped:NO];
            maskDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            maskDesc.storageMode = MTLStorageModePrivate;

            ctx->maskTex    = [ctx->device newTextureWithDescriptor:maskDesc];
            ctx->erodedTex  = [ctx->device newTextureWithDescriptor:maskDesc];
            ctx->dilatedTex = [ctx->device newTextureWithDescriptor:maskDesc];
            ctx->rgbaBuf    = [ctx->device newBufferWithLength:npixels * 4
                                                       options:MTLResourceStorageModeShared];
            ctx->alloc_width = width;
            ctx->alloc_height = height;
        }
        if (!ctx->maskCountBuf) {
            ctx->maskCountBuf = [ctx->device newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
            ctx->threshBuf    = [ctx->device newBufferWithLength:sizeof(int)
                                                         options:MTLResourceStorageModeShared];
            ctx->domBuf       = [ctx->device newBufferWithLength:sizeof(int)
                                                         options:MTLResourceStorageModeShared];
        }
        *(uint32_t *)ctx->maskCountBuf.contents = 0;
        *(int *)ctx->threshBuf.contents = green_threshold;
        *(int *)ctx->domBuf.contents = green_dominance;

        MTLSize tgSize = MTLSizeMake(16, 16, 1);
        MTLSize gridSize = MTLSizeMake(width, height, 1);

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];

        // Pass 1: threshold_green
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->pso_threshold];
            [enc setTexture:srcTex        atIndex:0];
            [enc setTexture:ctx->maskTex  atIndex:1];
            [enc setBuffer:ctx->threshBuf offset:0 atIndex:0];
            [enc setBuffer:ctx->domBuf    offset:0 atIndex:1];
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
        }

        // Pass 2: erode_3x3
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->pso_erode];
            [enc setTexture:ctx->maskTex   atIndex:0];
            [enc setTexture:ctx->erodedTex atIndex:1];
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
        }

        // Pass 3: dilate_3x3
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->pso_dilate];
            [enc setTexture:ctx->erodedTex  atIndex:0];
            [enc setTexture:ctx->dilatedTex atIndex:1];
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
        }

        // Pass 4: colorize_viz (writes RGBA + counts mask pixels)
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:ctx->pso_colorize];
            [enc setTexture:ctx->maskTex    atIndex:0];
            [enc setTexture:ctx->dilatedTex atIndex:1];
            [enc setBuffer:ctx->rgbaBuf      offset:0 atIndex:0];
            [enc setBuffer:ctx->maskCountBuf offset:0 atIndex:1];
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        result.total_mask_pixels = *(uint32_t *)ctx->maskCountBuf.contents;

        // --- CPU: BFS connected components on dilated (green) pixels ---
        // The colorize_viz kernel wrote green (0,255,0,255) for dilated pixels.
        // We scan the shared RGBA buffer, BFS each green blob, classify by size,
        // and recolor in-place. Dilated pixels are typically ~100-500, so this
        // is negligible. We avoid a full-frame labels array by using the RGBA
        // color itself as a visited marker — recolored pixels are no longer
        // (R=0,G=255,B=0), so subsequent scan skips them.
        uint8_t *rgba = (uint8_t *)ctx->rgbaBuf.contents;
        int num_blobs = 0;
        std::vector<int> queue;
        queue.reserve(1024);

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                int base = idx * 4;
                // Check for unvisited green (dilated) pixel
                if (rgba[base] != 0 || rgba[base+1] != 255 || rgba[base+2] != 0)
                    continue;

                // BFS flood-fill this connected component
                queue.clear();
                queue.push_back(idx);
                rgba[base] = 1; // mark visited (R=1 distinguishes from unvisited green)

                for (size_t qi = 0; qi < queue.size(); qi++) {
                    int ci = queue[qi];
                    int cx_pos = ci % width, cy_pos = ci / width;
                    const int neighbors[4] = {
                        (cx_pos > 0)          ? ci - 1     : -1,
                        (cx_pos < width - 1)  ? ci + 1     : -1,
                        (cy_pos > 0)          ? ci - width  : -1,
                        (cy_pos < height - 1) ? ci + width  : -1
                    };
                    for (int ni : neighbors) {
                        if (ni < 0) continue;
                        int nb = ni * 4;
                        if (rgba[nb] == 0 && rgba[nb+1] == 255 && rgba[nb+2] == 0) {
                            rgba[nb] = 1; // mark visited
                            queue.push_back(ni);
                        }
                    }
                }

                int blob_size = (int)queue.size();

                // Classify blob and assign final color
                uint8_t cr, cg, cb;
                if (blob_size >= min_blob_pixels && blob_size <= max_blob_pixels) {
                    cr = 0; cg = 255; cb = 0;   // green — valid
                    num_blobs++;
                } else if (blob_size < min_blob_pixels) {
                    cr = 200; cg = 200; cb = 0;  // yellow — too small
                } else {
                    cr = 255; cg = 50; cb = 50;  // red — too large
                }

                for (int pi : queue) {
                    int pb = pi * 4;
                    rgba[pb] = cr; rgba[pb+1] = cg; rgba[pb+2] = cb; rgba[pb+3] = 255;
                }
            }
        }

        result.num_blobs = num_blobs;
        memcpy(rgba_out, rgba, npixels * 4);

        CFRelease(cvTex);
    }

    return result;
}


void pointsource_metal_destroy(PointSourceMetalHandle ctx) {
    if (!ctx) return;
    if (ctx->texCache) {
        CVMetalTextureCacheFlush(ctx->texCache, 0);
        CFRelease(ctx->texCache);
    }
    delete ctx;
}

#endif // __APPLE__
