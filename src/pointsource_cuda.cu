#if defined(_WIN32) && defined(USE_CUDA_POINTSOURCE)

#include "pointsource_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// CUDA kernels — exact ports of Metal compute shaders
// ---------------------------------------------------------------------------

// Kernel 1: Threshold green channel -> binary mask (uint8)
// Input: BGRA pixels (B=0, G=1, R=2, A=3 per pixel)
// Output: mask (1 byte per pixel, 0 or 1)
__global__ void threshold_green_kernel(
    const uint8_t *__restrict__ src, uint8_t *__restrict__ mask,
    int width, int height, int stride,
    int green_threshold, int green_dominance)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t *pixel = src + y * stride + x * 4;
    int b = pixel[0];
    int g = pixel[1];
    int r = pixel[2];

    int gt = green_threshold;
    int gd = green_dominance;

    bool pass = (g > gt) && (g >= r) && (g >= b) &&
                ((g > r + gd && g > b + gd) || (g > 200 && g >= r && g >= b));

    mask[y * width + x] = pass ? 1 : 0;
}

// Kernel 2: Erode 3x3 — pixel stays 1 only if all 9 neighbors are 1
// Uses shared memory with 1-pixel halo for coalesced reads.
__global__ void erode_3x3_kernel(
    const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
    int width, int height)
{
    // Shared memory tile: blockDim + 2 pixels of halo on each side
    extern __shared__ uint8_t smem[];
    int tile_w = blockDim.x + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Load center pixel into shared memory (offset by halo)
    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    // Clamp source coordinates for halo loads
    auto load = [&](int gx, int gy) -> uint8_t {
        if (gx < 0 || gx >= width || gy < 0 || gy >= height) return 0;
        return src[gy * width + gx];
    };

    smem[sy * tile_w + sx] = load(x, y);

    // Load halo pixels
    if (threadIdx.x == 0)
        smem[sy * tile_w + 0] = load(x - 1, y);
    if (threadIdx.x == blockDim.x - 1 || x == width - 1)
        smem[sy * tile_w + sx + 1] = load(x + 1, y);
    if (threadIdx.y == 0)
        smem[0 * tile_w + sx] = load(x, y - 1);
    if (threadIdx.y == blockDim.y - 1 || y == height - 1)
        smem[(sy + 1) * tile_w + sx] = load(x, y + 1);

    // Corner halos
    if (threadIdx.x == 0 && threadIdx.y == 0)
        smem[0] = load(x - 1, y - 1);
    if ((threadIdx.x == blockDim.x - 1 || x == width - 1) && threadIdx.y == 0)
        smem[sx + 1] = load(x + 1, y - 1);
    if (threadIdx.x == 0 && (threadIdx.y == blockDim.y - 1 || y == height - 1))
        smem[(sy + 1) * tile_w] = load(x - 1, y + 1);
    if ((threadIdx.x == blockDim.x - 1 || x == width - 1) &&
        (threadIdx.y == blockDim.y - 1 || y == height - 1))
        smem[(sy + 1) * tile_w + sx + 1] = load(x + 1, y + 1);

    __syncthreads();

    if (x >= width || y >= height) return;

    // Border pixels forced to 0 (matching Metal behavior)
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        dst[y * width + x] = 0;
        return;
    }

    bool all_on = true;
    for (int dy = -1; dy <= 1 && all_on; dy++) {
        for (int dx = -1; dx <= 1 && all_on; dx++) {
            if (smem[(sy + dy) * tile_w + (sx + dx)] == 0)
                all_on = false;
        }
    }
    dst[y * width + x] = all_on ? 1 : 0;
}

// Kernel 3: Dilate 3x3 — pixel becomes 1 if any of 9 neighbors is 1
// Uses shared memory with 1-pixel halo for coalesced reads.
__global__ void dilate_3x3_kernel(
    const uint8_t *__restrict__ src, uint8_t *__restrict__ dst,
    int width, int height)
{
    extern __shared__ uint8_t smem[];
    int tile_w = blockDim.x + 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    auto load = [&](int gx, int gy) -> uint8_t {
        if (gx < 0 || gx >= width || gy < 0 || gy >= height) return 0;
        return src[gy * width + gx];
    };

    smem[sy * tile_w + sx] = load(x, y);

    if (threadIdx.x == 0)
        smem[sy * tile_w + 0] = load(x - 1, y);
    if (threadIdx.x == blockDim.x - 1 || x == width - 1)
        smem[sy * tile_w + sx + 1] = load(x + 1, y);
    if (threadIdx.y == 0)
        smem[0 * tile_w + sx] = load(x, y - 1);
    if (threadIdx.y == blockDim.y - 1 || y == height - 1)
        smem[(sy + 1) * tile_w + sx] = load(x, y + 1);

    if (threadIdx.x == 0 && threadIdx.y == 0)
        smem[0] = load(x - 1, y - 1);
    if ((threadIdx.x == blockDim.x - 1 || x == width - 1) && threadIdx.y == 0)
        smem[sx + 1] = load(x + 1, y - 1);
    if (threadIdx.x == 0 && (threadIdx.y == blockDim.y - 1 || y == height - 1))
        smem[(sy + 1) * tile_w] = load(x - 1, y + 1);
    if ((threadIdx.x == blockDim.x - 1 || x == width - 1) &&
        (threadIdx.y == blockDim.y - 1 || y == height - 1))
        smem[(sy + 1) * tile_w + sx + 1] = load(x + 1, y + 1);

    __syncthreads();

    if (x >= width || y >= height) return;

    // Border pixels stay 0 (matching Metal/CPU behavior)
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        dst[y * width + x] = 0;
        return;
    }

    bool any_on = false;
    for (int dy = -1; dy <= 1 && !any_on; dy++) {
        for (int dx = -1; dx <= 1 && !any_on; dx++) {
            if (smem[(sy + dy) * tile_w + (sx + dx)] != 0)
                any_on = true;
        }
    }
    dst[y * width + x] = any_on ? 1 : 0;
}

// Kernel 4: Reduce centroid — atomically accumulate count, sum_gx, sum_gy, sum_g
// and bounding box via atomicMin/atomicMax.
// result[0]=count, result[1]=sum_gx, result[2]=sum_gy, result[3]=sum_g
// result[4]=min_x, result[5]=min_y (init to 0xFFFFFFFF)
// result[6]=max_x, result[7]=max_y (init to 0)
__global__ void reduce_centroid_kernel(
    const uint8_t *__restrict__ mask,
    const uint8_t *__restrict__ src,
    unsigned int *__restrict__ result,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    if (mask[y * width + x] == 0) return;

    // Read green channel from BGRA source (G is at offset 1)
    unsigned int g = src[y * stride + x * 4 + 1];

    atomicAdd(&result[0], 1u);
    atomicAdd(&result[1], g * (unsigned int)x);
    atomicAdd(&result[2], g * (unsigned int)y);
    atomicAdd(&result[3], g);

    // Bounding box
    atomicMin(&result[4], (unsigned int)x);
    atomicMin(&result[5], (unsigned int)y);
    atomicMax(&result[6], (unsigned int)x);
    atomicMax(&result[7], (unsigned int)y);
}

// Kernel 5: Colorize visualization — writes RGBA to output buffer
// Also counts total mask (threshold) pixels via atomic counter.
__global__ void colorize_viz_kernel(
    const uint8_t *__restrict__ mask,
    const uint8_t *__restrict__ dilated,
    uint8_t *__restrict__ rgba_out,
    unsigned int *__restrict__ mask_count,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int base = idx * 4;

    bool m = mask[idx] != 0;
    bool d = dilated[idx] != 0;

    if (m) atomicAdd(mask_count, 1u);

    if (d) {
        // Green placeholder — CPU will recolor per-blob classification
        rgba_out[base]     = 0;
        rgba_out[base + 1] = 255;
        rgba_out[base + 2] = 0;
        rgba_out[base + 3] = 255;
    } else if (m) {
        // Gray — threshold passed but filtered by erode/dilate
        rgba_out[base]     = 60;
        rgba_out[base + 1] = 60;
        rgba_out[base + 2] = 60;
        rgba_out[base + 3] = 255;
    } else {
        // Black background
        rgba_out[base]     = 0;
        rgba_out[base + 1] = 0;
        rgba_out[base + 2] = 0;
        rgba_out[base + 3] = 255;
    }
}


// ---------------------------------------------------------------------------
// PointSourceCudaContext — holds all CUDA state
// ---------------------------------------------------------------------------

struct PointSourceCudaContext {
    cudaStream_t stream = nullptr;

    // Pre-allocated GPU buffers (resized when image dimensions change)
    int alloc_width = 0, alloc_height = 0, alloc_stride = 0;
    uint8_t *d_src = nullptr;       // BGRA source image
    uint8_t *d_mask = nullptr;      // threshold mask (1 byte/pixel)
    uint8_t *d_eroded = nullptr;    // eroded mask
    uint8_t *d_dilated = nullptr;   // dilated mask
    unsigned int *d_result = nullptr; // 8 x uint32 reduction buffer
    uint8_t *d_rgba = nullptr;      // viz output (w*h*4)
    unsigned int *d_mask_count = nullptr; // viz atomic mask counter

    // Host-side readback buffer for reduction results
    unsigned int h_result[8] = {};
};

// Ensure GPU buffers are allocated for the given dimensions.
static void ensure_alloc(PointSourceCudaContext *ctx,
                          int width, int height, int stride) {
    if (ctx->alloc_width == width && ctx->alloc_height == height &&
        ctx->alloc_stride == stride)
        return;

    // Free old allocations
    if (ctx->d_src)        cudaFree(ctx->d_src);
    if (ctx->d_mask)       cudaFree(ctx->d_mask);
    if (ctx->d_eroded)     cudaFree(ctx->d_eroded);
    if (ctx->d_dilated)    cudaFree(ctx->d_dilated);
    if (ctx->d_result)     cudaFree(ctx->d_result);
    if (ctx->d_rgba)       cudaFree(ctx->d_rgba);
    if (ctx->d_mask_count) cudaFree(ctx->d_mask_count);

    int npixels = width * height;
    cudaMalloc(&ctx->d_src, (size_t)height * stride);
    cudaMalloc(&ctx->d_mask, npixels);
    cudaMalloc(&ctx->d_eroded, npixels);
    cudaMalloc(&ctx->d_dilated, npixels);
    cudaMalloc(&ctx->d_result, 8 * sizeof(unsigned int));
    cudaMalloc(&ctx->d_rgba, npixels * 4);
    cudaMalloc(&ctx->d_mask_count, sizeof(unsigned int));

    ctx->alloc_width = width;
    ctx->alloc_height = height;
    ctx->alloc_stride = stride;
}


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

PointSourceCudaHandle pointsource_cuda_create() {
    // Check for CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[PointSourceCUDA] No CUDA device found\n");
        return nullptr;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[PointSourceCUDA] Using device: %s\n", prop.name);

    auto *ctx = new PointSourceCudaContext{};

    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[PointSourceCUDA] Failed to create stream: %s\n",
                cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }

    return ctx;
}


PointSourceCudaSpot pointsource_cuda_detect(PointSourceCudaHandle ctx,
                                            const uint8_t *bgra_data,
                                            int width, int height, int stride,
                                            int green_threshold,
                                            int green_dominance,
                                            int min_blob_pixels,
                                            int max_blob_pixels,
                                            bool smart_blob) {
    PointSourceCudaSpot result = {0, 0, 0, false};
    if (!ctx || !bgra_data) return result;

    ensure_alloc(ctx, width, height, stride);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    size_t smem_size = (block.x + 2) * (block.y + 2) * sizeof(uint8_t);

    // Upload source image
    cudaMemcpyAsync(ctx->d_src, bgra_data, (size_t)height * stride,
                    cudaMemcpyHostToDevice, ctx->stream);

    // Pass 1: threshold_green
    threshold_green_kernel<<<grid, block, 0, ctx->stream>>>(
        ctx->d_src, ctx->d_mask, width, height, stride,
        green_threshold, green_dominance);

    // Pass 2: erode_3x3
    erode_3x3_kernel<<<grid, block, smem_size, ctx->stream>>>(
        ctx->d_mask, ctx->d_eroded, width, height);

    // Pass 3: dilate_3x3
    dilate_3x3_kernel<<<grid, block, smem_size, ctx->stream>>>(
        ctx->d_eroded, ctx->d_dilated, width, height);

    // Pass 4: reduce_centroid
    // Initialize result buffer: count/sum=0, min=0xFFFFFFFF, max=0
    unsigned int init_vals[8] = {0, 0, 0, 0, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0};
    cudaMemcpyAsync(ctx->d_result, init_vals, 8 * sizeof(unsigned int),
                    cudaMemcpyHostToDevice, ctx->stream);

    reduce_centroid_kernel<<<grid, block, 0, ctx->stream>>>(
        ctx->d_dilated, ctx->d_src, ctx->d_result, width, height, stride);

    // Read back reduction results
    cudaMemcpyAsync(ctx->h_result, ctx->d_result, 8 * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    unsigned int count  = ctx->h_result[0];
    unsigned int sum_gx = ctx->h_result[1];
    unsigned int sum_gy = ctx->h_result[2];
    unsigned int sum_g  = ctx->h_result[3];
    unsigned int min_x  = ctx->h_result[4];
    unsigned int min_y  = ctx->h_result[5];
    unsigned int max_x  = ctx->h_result[6];
    unsigned int max_y  = ctx->h_result[7];

    // Always report dilated pixel count
    result.pixel_count = (int)count;

    // Check blob size
    if ((int)count >= min_blob_pixels && (int)count <= max_blob_pixels && sum_g > 0) {
        if (!smart_blob) {
            // Default path: compactness check to reject multi-blob frames.
            // A single compact blob fills >25% of its bbox (circle ~79%).
            uint64_t bbox_w = (uint64_t)(max_x - min_x + 1);
            uint64_t bbox_h = (uint64_t)(max_y - min_y + 1);
            uint64_t bbox_area = bbox_w * bbox_h;
            if (bbox_area <= 4u * (uint64_t)count) {
                result.cx = (double)sum_gx / (double)sum_g;
                result.cy = (double)sum_gy / (double)sum_g;
                result.pixel_count = (int)count;
                result.found = true;
            }
        } else {
            // Smart Blob: run BFS connected components on the dilated mask
            // to find individual blobs, then pick the largest valid one.
            // Download dilated mask and source for CPU BFS.
            int npixels = width * height;
            std::vector<uint8_t> h_dilated(npixels);
            cudaMemcpyAsync(h_dilated.data(), ctx->d_dilated, npixels,
                            cudaMemcpyDeviceToHost, ctx->stream);
            cudaStreamSynchronize(ctx->stream);

            int best_blob_size = 0;
            double best_cx = 0, best_cy = 0;
            int best_count = 0;
            std::vector<int> queue;
            queue.reserve(1024);
            std::vector<bool> visited(npixels, false);

            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    int idx = y * width + x;
                    if (!h_dilated[idx] || visited[idx])
                        continue;

                    // BFS flood-fill
                    queue.clear();
                    queue.push_back(idx);
                    visited[idx] = true;

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
                            if (h_dilated[ni] && !visited[ni]) {
                                visited[ni] = true;
                                queue.push_back(ni);
                            }
                        }
                    }

                    int blob_size = (int)queue.size();
                    if (blob_size < min_blob_pixels || blob_size > max_blob_pixels)
                        continue;

                    // This blob is valid-sized — compute intensity-weighted centroid
                    if (blob_size > best_blob_size) {
                        double sx = 0, sy = 0, sw = 0;
                        for (int pi : queue) {
                            int px = pi % width;
                            int py = pi / width;
                            // Green channel from BGRA source (G at offset 1)
                            double g = bgra_data[py * stride + px * 4 + 1];
                            sx += px * g;
                            sy += py * g;
                            sw += g;
                        }
                        if (sw > 0) {
                            best_blob_size = blob_size;
                            best_cx = sx / sw;
                            best_cy = sy / sw;
                            best_count = blob_size;
                        }
                    }
                }
            }

            if (best_blob_size > 0) {
                result.cx = best_cx;
                result.cy = best_cy;
                result.pixel_count = best_count;
                result.found = true;
            }
        }
    }

    return result;
}


PointSourceCudaVizResult pointsource_cuda_detect_viz(
    PointSourceCudaHandle ctx,
    const uint8_t *bgra_data,
    int width, int height, int stride,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels,
    uint8_t *rgba_out) {

    PointSourceCudaVizResult result = {};
    if (!ctx || !bgra_data || !rgba_out) return result;

    int npixels = width * height;
    ensure_alloc(ctx, width, height, stride);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    size_t smem_size = (block.x + 2) * (block.y + 2) * sizeof(uint8_t);

    // Upload source image
    cudaMemcpyAsync(ctx->d_src, bgra_data, (size_t)height * stride,
                    cudaMemcpyHostToDevice, ctx->stream);

    // Pass 1: threshold_green
    threshold_green_kernel<<<grid, block, 0, ctx->stream>>>(
        ctx->d_src, ctx->d_mask, width, height, stride,
        green_threshold, green_dominance);

    // Pass 2: erode_3x3
    erode_3x3_kernel<<<grid, block, smem_size, ctx->stream>>>(
        ctx->d_mask, ctx->d_eroded, width, height);

    // Pass 3: dilate_3x3
    dilate_3x3_kernel<<<grid, block, smem_size, ctx->stream>>>(
        ctx->d_eroded, ctx->d_dilated, width, height);

    // Pass 4: colorize_viz (writes RGBA + counts mask pixels)
    unsigned int zero = 0;
    cudaMemcpyAsync(ctx->d_mask_count, &zero, sizeof(unsigned int),
                    cudaMemcpyHostToDevice, ctx->stream);

    colorize_viz_kernel<<<grid, block, 0, ctx->stream>>>(
        ctx->d_mask, ctx->d_dilated, ctx->d_rgba, ctx->d_mask_count,
        width, height);

    // Download RGBA result and mask count
    cudaMemcpyAsync(rgba_out, ctx->d_rgba, npixels * 4,
                    cudaMemcpyDeviceToHost, ctx->stream);
    unsigned int mask_count = 0;
    cudaMemcpyAsync(&mask_count, ctx->d_mask_count, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    result.total_mask_pixels = (int)mask_count;

    // --- CPU: BFS connected components on dilated (green) pixels ---
    // The colorize_viz kernel wrote green (0,255,0,255) for dilated pixels.
    // Scan the RGBA buffer, BFS each green blob, classify by size,
    // and recolor in-place. Recolored pixels are no longer (R=0,G=255,B=0),
    // so subsequent scan skips them.
    int num_blobs = 0;
    std::vector<int> queue;
    queue.reserve(1024);

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            int base = idx * 4;
            // Check for unvisited green (dilated) pixel
            if (rgba_out[base] != 0 || rgba_out[base+1] != 255 || rgba_out[base+2] != 0)
                continue;

            // BFS flood-fill this connected component
            queue.clear();
            queue.push_back(idx);
            rgba_out[base] = 1; // mark visited (R=1 distinguishes from unvisited green)

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
                    if (rgba_out[nb] == 0 && rgba_out[nb+1] == 255 && rgba_out[nb+2] == 0) {
                        rgba_out[nb] = 1; // mark visited
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
                rgba_out[pb] = cr; rgba_out[pb+1] = cg;
                rgba_out[pb+2] = cb; rgba_out[pb+3] = 255;
            }
        }
    }

    result.num_blobs = num_blobs;
    return result;
}


void pointsource_cuda_destroy(PointSourceCudaHandle ctx) {
    if (!ctx) return;

    if (ctx->d_src)        cudaFree(ctx->d_src);
    if (ctx->d_mask)       cudaFree(ctx->d_mask);
    if (ctx->d_eroded)     cudaFree(ctx->d_eroded);
    if (ctx->d_dilated)    cudaFree(ctx->d_dilated);
    if (ctx->d_result)     cudaFree(ctx->d_result);
    if (ctx->d_rgba)       cudaFree(ctx->d_rgba);
    if (ctx->d_mask_count) cudaFree(ctx->d_mask_count);

    if (ctx->stream) cudaStreamDestroy(ctx->stream);

    delete ctx;
}

#endif // _WIN32 && USE_CUDA_POINTSOURCE
