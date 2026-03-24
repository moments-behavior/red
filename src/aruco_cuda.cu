#ifdef _WIN32

#include "aruco_cuda.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <chrono>

// Maximum number of simultaneous threshold passes per call
static constexpr int kMaxPasses = 4;

// ---------------------------------------------------------------------------
// CUDA Kernels — separable box filter adaptive threshold
// ---------------------------------------------------------------------------
// Three-kernel approach matching the Metal implementation:
//   Kernel 1 (horizontal_box_sum): one thread per row, sliding window
//   Kernel 2 (vertical_sum_threshold): one thread per column, sliding window
//   Kernel 3 (downsample_binary_3x): 2D, one thread per output pixel, OR 3x3 block

// Kernel 1: BGRA -> grayscale conversion.
// One thread per pixel. Matches Metal luminance coefficients exactly.
__global__ void bgra_to_gray_kernel(
    const uint8_t *bgra, uint8_t *gray,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint8_t *px = bgra + y * stride + x * 4;
    // BGRA order: B=px[0], G=px[1], R=px[2], A=px[3]
    // Luminance: 0.299*R + 0.587*G + 0.114*B (same as Metal shader)
    float lum = 0.299f * px[2] + 0.587f * px[1] + 0.114f * px[0];
    lum = fminf(fmaxf(lum, 0.0f), 255.0f);
    gray[y * width + x] = (uint8_t)lum;
}

// Kernel 2: Horizontal running sum.
// One thread per row. Each thread slides a window across its row,
// maintaining a running sum. Output: h_sum[y][x] = sum of gray pixels
// in the horizontal window centered at (x, y).
__global__ void horizontal_box_sum_kernel(
    const uint8_t *gray, int32_t *h_sum,
    int width, int height, int hw)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    int w = width;
    int row_off = y * w;

    // Initialize: sum of gray[y][0 .. min(hw, w-1)]
    int sum = 0;
    int init_end = min(hw, w - 1);
    for (int x = 0; x <= init_end; x++) {
        sum += gray[row_off + x];
    }
    h_sum[row_off] = sum;

    // Slide window right across the row
    for (int x = 1; x < w; x++) {
        int add_x = x + hw;
        int rem_x = x - hw - 1;
        if (add_x < w)  sum += gray[row_off + add_x];
        if (rem_x >= 0) sum -= gray[row_off + rem_x];
        h_sum[row_off + x] = sum;
    }
}

// Kernel 3: Vertical running sum + threshold.
// One thread per column. Each thread slides a window down its column,
// accumulating h_sum values. At each pixel, computes box mean and thresholds.
__global__ void vertical_sum_threshold_kernel(
    const uint8_t *gray, const int32_t *h_sum, uint8_t *binary,
    int width, int height, int hw, int C_param)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int w = width;
    int h = height;

    // Horizontal extent for this column (constant across all rows)
    int h_extent = min(x + hw, w - 1) - max(x - hw, 0) + 1;

    // Initialize: vertical sum of h_sum[0..min(hw, h-1)][x]
    int vsum = 0;
    int init_end = min(hw, h - 1);
    for (int y = 0; y <= init_end; y++) {
        vsum += h_sum[y * w + x];
    }

    // Process each row
    for (int y = 0; y < h; y++) {
        // Vertical extent at this row
        int v_extent = min(y + hw, h - 1) - max(y - hw, 0) + 1;
        int count = h_extent * v_extent;

        int val = gray[y * w + x];
        // Threshold: val * count > vsum - C * count
        // Equivalent to: val > (vsum / count) - C  (local mean minus C)
        binary[y * w + x] =
            (val * count > vsum - C_param * count) ? (uint8_t)255 : (uint8_t)0;

        // Update sliding window for next row
        int add_y = y + hw + 1;
        int rem_y = y - hw;
        if (add_y < h) vsum += h_sum[add_y * w + x];
        if (rem_y >= 0) vsum -= h_sum[rem_y * w + x];
    }
}

// Kernel 4: 3x downsample binary image.
// One thread per output pixel. Reads a 3x3 block from the full-res binary,
// ORs them: if any pixel in the block is 255, output is 255.
__global__ void downsample_binary_3x_kernel(
    const uint8_t *binary, uint8_t *ds_out,
    int src_w, int src_h, int dst_w, int dst_h)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dst_w || oy >= dst_h) return;

    int sx = ox * 3;
    int sy = oy * 3;

    uint8_t val = 0;
    for (int dy = 0; dy < 3 && (sy + dy) < src_h; dy++) {
        int row = (sy + dy) * src_w + sx;
        for (int dx = 0; dx < 3 && (sx + dx) < src_w; dx++) {
            val |= binary[row + dx];
        }
    }
    ds_out[oy * dst_w + ox] = val;
}

// ---------------------------------------------------------------------------
// ArucoCudaContext
// ---------------------------------------------------------------------------

struct ArucoCudaContext {
    // Pre-allocated device buffers (reused across calls, protected by mutex)
    std::mutex mtx;
    int alloc_w = 0, alloc_h = 0;
    int ds_w = 0, ds_h = 0;         // downsampled dimensions (w/3, h/3)
    uint8_t  *d_gray = nullptr;      // w*h uint8 (device — CPU uploads)
    int32_t  *d_hsum = nullptr;      // w*h int32 (intermediate horizontal sums)
    uint8_t  *d_binary = nullptr;    // w*h uint8 (device only — full-res binary)
    uint8_t  *d_ds[kMaxPasses] = {}; // (w/3)*(h/3) uint8 per pass (device — CPU downloads)

    cudaStream_t stream = nullptr;

    // Timing stats
    int call_count = 0;
    double total_ms = 0;
};

static void free_cuda_buffers(ArucoCudaContext *ctx) {
    if (ctx->d_gray)   { cudaFree(ctx->d_gray);   ctx->d_gray = nullptr; }
    if (ctx->d_hsum)   { cudaFree(ctx->d_hsum);   ctx->d_hsum = nullptr; }
    if (ctx->d_binary) { cudaFree(ctx->d_binary); ctx->d_binary = nullptr; }
    for (int p = 0; p < kMaxPasses; p++) {
        if (ctx->d_ds[p]) { cudaFree(ctx->d_ds[p]); ctx->d_ds[p] = nullptr; }
    }
    ctx->alloc_w = 0;
    ctx->alloc_h = 0;
}

ArucoCudaHandle aruco_cuda_create() {
    // Check for CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "[ArucoCuda] No CUDA device found (err=%d, count=%d)\n",
                (int)err, device_count);
        return nullptr;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    auto *ctx = new ArucoCudaContext{};

    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ArucoCuda] cudaStreamCreate failed: %s\n",
                cudaGetErrorString(err));
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "[ArucoCuda] Initialized on %s (compute %d.%d, %.0f MB)\n",
            prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / 1e6);
    return ctx;
}

static bool ensure_buffers(ArucoCudaContext *ctx, int w, int h) {
    if (ctx->alloc_w == w && ctx->alloc_h == h) return true;

    free_cuda_buffers(ctx);

    size_t gray_size = (size_t)w * h;
    size_t hsum_size = (size_t)w * h * sizeof(int32_t);
    int dw = w / 3, dh = h / 3;
    size_t ds_size = (size_t)dw * dh;

    cudaError_t err;
    err = cudaMalloc(&ctx->d_gray, gray_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_hsum, hsum_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_binary, gray_size);
    if (err != cudaSuccess) goto fail;
    for (int p = 0; p < kMaxPasses; p++) {
        err = cudaMalloc(&ctx->d_ds[p], ds_size);
        if (err != cudaSuccess) goto fail;
    }

    ctx->alloc_w = w;
    ctx->alloc_h = h;
    ctx->ds_w = dw;
    ctx->ds_h = dh;

    fprintf(stderr, "[ArucoCuda] Allocated buffers for %dx%d -> %dx%d "
            "(gray=%.1fMB, fullres=%.1fMB, ds=%.1fKB x %d)\n",
            w, h, dw, dh, gray_size / 1e6, gray_size / 1e6, ds_size / 1e3, kMaxPasses);
    return true;

fail:
    fprintf(stderr, "[ArucoCuda] cudaMalloc failed: %s\n", cudaGetErrorString(err));
    free_cuda_buffers(ctx);
    return false;
}

void aruco_cuda_threshold_batch(
    void *ctx_void,
    const uint8_t *gray, int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs)
{
    auto *ctx = (ArucoCudaContext *)ctx_void;
    if (!ctx || !gray || num_passes <= 0) return;
    if (num_passes > kMaxPasses) num_passes = kMaxPasses;

    // Serialize GPU access (same pattern as Metal)
    std::lock_guard<std::mutex> lock(ctx->mtx);

    auto t_start = std::chrono::steady_clock::now();

    size_t gray_size = (size_t)w * h;
    int dw = w / 3, dh = h / 3;
    size_t ds_size = (size_t)dw * dh;

    // Re-allocate buffers only when image dimensions change
    if (!ensure_buffers(ctx, w, h)) return;

    // Upload grayscale image to GPU
    cudaMemcpyAsync(ctx->d_gray, gray, gray_size,
                    cudaMemcpyHostToDevice, ctx->stream);

    // Process each threshold pass
    for (int p = 0; p < num_passes; p++) {
        int hw = window_sizes[p] / 2;

        // Kernel 1: Horizontal running sum (one thread per row)
        {
            int threads = 256;
            int blocks = (h + threads - 1) / threads;
            horizontal_box_sum_kernel<<<blocks, threads, 0, ctx->stream>>>(
                ctx->d_gray, ctx->d_hsum, w, h, hw);
        }

        // Kernel 2: Vertical running sum + threshold (one thread per column)
        {
            int threads = 256;
            int blocks = (w + threads - 1) / threads;
            vertical_sum_threshold_kernel<<<blocks, threads, 0, ctx->stream>>>(
                ctx->d_gray, ctx->d_hsum, ctx->d_binary, w, h, hw, C);
        }

        // Kernel 3: 3x downsample (2D dispatch)
        {
            dim3 threads(16, 16);
            dim3 blocks((dw + threads.x - 1) / threads.x,
                        (dh + threads.y - 1) / threads.y);
            downsample_binary_3x_kernel<<<blocks, threads, 0, ctx->stream>>>(
                ctx->d_binary, ctx->d_ds[p], w, h, dw, dh);
        }
    }

    // Download downsampled results to caller's output buffers
    for (int p = 0; p < num_passes; p++) {
        cudaMemcpyAsync(binary_outputs[p], ctx->d_ds[p], ds_size,
                        cudaMemcpyDeviceToHost, ctx->stream);
    }

    // Wait for all GPU work + transfers to complete
    cudaStreamSynchronize(ctx->stream);

    auto t_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    ctx->total_ms += ms;
    ctx->call_count++;

    if (ctx->call_count % 100 == 0) {
        fprintf(stderr, "[ArucoCuda] %d calls, avg %.2f ms/call\n",
                ctx->call_count, ctx->total_ms / ctx->call_count);
    }
}

void aruco_cuda_destroy(ArucoCudaHandle ctx) {
    if (!ctx) return;
    if (ctx->call_count > 0) {
        fprintf(stderr, "[ArucoCuda] Final: %d calls, avg %.2f ms/call (%.1fs total)\n",
                ctx->call_count, ctx->total_ms / ctx->call_count,
                ctx->total_ms / 1000.0);
    }
    free_cuda_buffers(ctx);
    if (ctx->stream) {
        cudaStreamDestroy(ctx->stream);
        ctx->stream = nullptr;
    }
    delete ctx;
}

#endif // _WIN32
