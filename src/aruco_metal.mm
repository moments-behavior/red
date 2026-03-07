#ifdef __APPLE__

#import <Metal/Metal.h>
#include "aruco_metal.h"
#include <cstring>
#include <cstdio>
#include <mutex>
#include <chrono>

// ---------------------------------------------------------------------------
// Metal shader source — separable box filter adaptive threshold
// ---------------------------------------------------------------------------
// Two-kernel approach that avoids transferring the 56MB integral image:
//   Kernel 1 (horizontal_box_sum): one thread per row, sliding window
//   Kernel 2 (vertical_sum_threshold): one thread per column, sliding window
// Total GPU data transfer: 7MB gray in + 7MB binary out per pass.

static const char *kArucoShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Kernel 1: Horizontal running sum.
// One thread per row. Each thread slides a window across its row,
// maintaining a running sum. Output: h_sum[y][x] = sum of gray pixels
// in the horizontal window centered at (x, y).
kernel void horizontal_box_sum(
    device const uint8_t *gray    [[buffer(0)]],
    device int32_t       *h_sum   [[buffer(1)]],
    constant uint        &width   [[buffer(2)]],
    constant uint        &height  [[buffer(3)]],
    constant int         &hw      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint y = tid;
    if (y >= height) return;

    int w = int(width);
    int row_off = int(y) * w;

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

// Kernel 2: Vertical running sum + threshold.
// One thread per column. Each thread slides a window down its column,
// accumulating h_sum values. At each pixel, computes box mean and thresholds.
kernel void vertical_sum_threshold(
    device const uint8_t *gray    [[buffer(0)]],
    device const int32_t *h_sum   [[buffer(1)]],
    device uint8_t       *binary  [[buffer(2)]],
    constant uint        &width   [[buffer(3)]],
    constant uint        &height  [[buffer(4)]],
    constant int         &hw      [[buffer(5)]],
    constant int         &C_param [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint x = tid;
    if (x >= width) return;

    int w = int(width);
    int h = int(height);

    // Horizontal extent for this column (constant across all rows)
    int h_extent = min(int(x) + hw, w - 1) - max(int(x) - hw, 0) + 1;

    // Initialize: vertical sum of h_sum[0..min(hw, h-1)][x]
    int vsum = 0;
    int init_end = min(hw, h - 1);
    for (int y = 0; y <= init_end; y++) {
        vsum += h_sum[y * w + int(x)];
    }

    // Process each row
    for (int y = 0; y < h; y++) {
        // Vertical extent at this row
        int v_extent = min(y + hw, h - 1) - max(y - hw, 0) + 1;
        int count = h_extent * v_extent;

        int val = gray[y * w + int(x)];
        // Threshold: val * count > vsum - C * count
        // Equivalent to: val > (vsum / count) - C  (local mean minus C)
        binary[y * w + int(x)] =
            (val * count > vsum - C_param * count) ? (uint8_t)255 : (uint8_t)0;

        // Update sliding window for next row
        int add_y = y + hw + 1;
        int rem_y = y - hw;
        if (add_y < h) vsum += h_sum[add_y * w + int(x)];
        if (rem_y >= 0) vsum -= h_sum[rem_y * w + int(x)];
    }
}
)METAL";

// Maximum number of simultaneous threshold passes per call
static constexpr int kMaxPasses = 4;

// ---------------------------------------------------------------------------
// ArucoMetalContext
// ---------------------------------------------------------------------------

struct ArucoMetalContext {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pso_horizontal;
    id<MTLComputePipelineState> pso_vertical;

    // Pre-allocated buffers (reused across calls, protected by mutex)
    std::mutex mtx;
    int alloc_w = 0, alloc_h = 0;
    id<MTLBuffer> grayBuf;           // w*h uint8
    id<MTLBuffer> hSumBuf;           // w*h int32 (intermediate horizontal sums)
    id<MTLBuffer> outBufs[kMaxPasses]; // w*h uint8 per pass

    // Timing stats
    int call_count = 0;
    double total_ms = 0;
};


ArucoMetalHandle aruco_metal_create() {
    @autoreleasepool {
        auto *ctx = new ArucoMetalContext{};

        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "[ArucoMetal] No Metal device found\n");
            delete ctx;
            return nullptr;
        }

        ctx->queue = [ctx->device newCommandQueue];

        // Compile shader from source
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:kArucoShaderSource];
        id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src
                                                       options:nil
                                                         error:&error];
        if (!lib) {
            fprintf(stderr, "[ArucoMetal] Shader compile error: %s\n",
                    error.localizedDescription.UTF8String);
            delete ctx;
            return nullptr;
        }

        auto make_pso = [&](const char *name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:
                                  [NSString stringWithUTF8String:name]];
            if (!fn) {
                fprintf(stderr, "[ArucoMetal] Function '%s' not found\n", name);
                return nil;
            }
            NSError *e = nil;
            id<MTLComputePipelineState> pso =
                [ctx->device newComputePipelineStateWithFunction:fn error:&e];
            if (!pso) {
                fprintf(stderr, "[ArucoMetal] PSO '%s' error: %s\n",
                        name, e.localizedDescription.UTF8String);
            }
            return pso;
        };

        ctx->pso_horizontal = make_pso("horizontal_box_sum");
        ctx->pso_vertical   = make_pso("vertical_sum_threshold");

        if (!ctx->pso_horizontal || !ctx->pso_vertical) {
            delete ctx;
            return nullptr;
        }

        fprintf(stderr, "[ArucoMetal] Initialized (separable box filter)\n");
        return ctx;
    }
}


void aruco_metal_threshold_batch(
    void *ctx_void,
    const uint8_t *gray, int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs)
{
    auto *ctx = (ArucoMetalContext *)ctx_void;
    if (!ctx || !gray || num_passes <= 0) return;
    if (num_passes > kMaxPasses) num_passes = kMaxPasses;

    // Serialize GPU access — GPU work is ~0.6ms per call, so serialization
    // overhead is minimal vs the 16-thread CPU contour finding that runs
    // in parallel while waiting.
    std::lock_guard<std::mutex> lock(ctx->mtx);

    @autoreleasepool {
        auto t_start = std::chrono::steady_clock::now();

        size_t gray_size = (size_t)w * h;
        size_t hsum_size = (size_t)w * h * sizeof(int32_t);

        // Re-allocate buffers only when image dimensions change
        if (ctx->alloc_w != w || ctx->alloc_h != h) {
            ctx->grayBuf = [ctx->device newBufferWithLength:gray_size
                                                    options:MTLResourceStorageModeShared];
            ctx->hSumBuf = [ctx->device newBufferWithLength:hsum_size
                                                    options:MTLResourceStorageModeShared];
            for (int p = 0; p < kMaxPasses; p++) {
                ctx->outBufs[p] = [ctx->device newBufferWithLength:gray_size
                                                            options:MTLResourceStorageModeShared];
            }
            ctx->alloc_w = w;
            ctx->alloc_h = h;
            fprintf(stderr, "[ArucoMetal] Allocated buffers for %dx%d "
                    "(gray=%.1fMB, hsum=%.1fMB)\n",
                    w, h, gray_size / 1e6, hsum_size / 1e6);
        }

        // Copy only the 7MB gray image (vs 56MB integral image before!)
        memcpy(ctx->grayBuf.contents, gray, gray_size);

        // Small parameter buffers (trivial size, create per call)
        uint32_t width_val = (uint32_t)w;
        uint32_t height_val = (uint32_t)h;
        int32_t C_val = (int32_t)C;

        id<MTLBuffer> widthBuf  = [ctx->device newBufferWithBytes:&width_val
                                                            length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> heightBuf = [ctx->device newBufferWithBytes:&height_val
                                                            length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf      = [ctx->device newBufferWithBytes:&C_val
                                                            length:sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];

        // Encode all passes into one command buffer
        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];

        for (int p = 0; p < num_passes; p++) {
            int32_t hw_val = window_sizes[p] / 2;
            id<MTLBuffer> hwBuf = [ctx->device newBufferWithBytes:&hw_val
                                                            length:sizeof(int32_t)
                                                           options:MTLResourceStorageModeShared];

            // Pass 1: Horizontal running sum (one thread per row)
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:ctx->pso_horizontal];
                [enc setBuffer:ctx->grayBuf offset:0 atIndex:0];
                [enc setBuffer:ctx->hSumBuf offset:0 atIndex:1];
                [enc setBuffer:widthBuf     offset:0 atIndex:2];
                [enc setBuffer:heightBuf    offset:0 atIndex:3];
                [enc setBuffer:hwBuf        offset:0 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(h, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(
                        MIN((NSUInteger)h, ctx->pso_horizontal.maxTotalThreadsPerThreadgroup), 1, 1)];
                [enc endEncoding];
            }

            // Pass 2: Vertical running sum + threshold (one thread per column)
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
                [enc setComputePipelineState:ctx->pso_vertical];
                [enc setBuffer:ctx->grayBuf    offset:0 atIndex:0];
                [enc setBuffer:ctx->hSumBuf    offset:0 atIndex:1];
                [enc setBuffer:ctx->outBufs[p] offset:0 atIndex:2];
                [enc setBuffer:widthBuf        offset:0 atIndex:3];
                [enc setBuffer:heightBuf       offset:0 atIndex:4];
                [enc setBuffer:hwBuf           offset:0 atIndex:5];
                [enc setBuffer:cBuf            offset:0 atIndex:6];
                [enc dispatchThreads:MTLSizeMake(w, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(
                        MIN((NSUInteger)w, ctx->pso_vertical.maxTotalThreadsPerThreadgroup), 1, 1)];
                [enc endEncoding];
            }
        }

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Copy results to caller's output buffers
        for (int p = 0; p < num_passes; p++) {
            memcpy(binary_outputs[p], ctx->outBufs[p].contents, gray_size);
        }

        auto t_end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ctx->total_ms += ms;
        ctx->call_count++;

        if (ctx->call_count % 100 == 0) {
            fprintf(stderr, "[ArucoMetal] %d calls, avg %.2f ms/call\n",
                    ctx->call_count, ctx->total_ms / ctx->call_count);
        }
    }
}


void aruco_metal_destroy(ArucoMetalHandle ctx) {
    if (!ctx) return;
    if (ctx->call_count > 0) {
        fprintf(stderr, "[ArucoMetal] Final: %d calls, avg %.2f ms/call (%.1fs total)\n",
                ctx->call_count, ctx->total_ms / ctx->call_count,
                ctx->total_ms / 1000.0);
    }
    delete ctx;
}

#endif // __APPLE__
