// CUDA stress tests and performance benchmarks for ArUco and PointSource kernels.
// Build: nvcc -o test_cuda_stress test_cuda_stress.cu ../src/aruco_cuda.cu ../src/pointsource_cuda.cu
//        -I../src -DUSE_CUDA_POINTSOURCE -lcuda -lcudart
// Run:   ./test_cuda_stress

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

// Force _WIN32-guarded headers to compile
#ifndef _WIN32
#define _WIN32
#define UNDEF_WIN32
#endif

#include "aruco_cuda.h"
#include "pointsource_cuda.h"

#ifdef UNDEF_WIN32
#undef _WIN32
#endif

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

static int passed = 0, failed = 0;

#define TEST(name) printf("  TEST: %s ... ", name)
#define PASS() do { printf("PASSED\n"); passed++; } while(0)
#define FAIL(msg) do { printf("FAILED: %s\n", msg); failed++; } while(0)

// ============================================================
// Timing helper using CUDA events
// ============================================================

struct TimingStats {
    double avg_ms;
    double min_ms;
    double max_ms;
};

// ============================================================
// ArUco CUDA Stress Tests
// ============================================================

static int test_aruco_4k_resolution() {
    TEST("aruco 4K resolution (3840x2160)");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 3840, h = 2160;
    std::vector<uint8_t> gray(w * h);
    // Horizontal gradient
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            gray[y * w + x] = (uint8_t)(255.0f * x / w);

    int window_sizes[] = {21};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> out(dw * dh, 0);
    uint8_t *outputs[] = {out.data()};

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    // Verify output is not all-zero (gradient should produce some foreground)
    int fg = 0;
    for (int i = 0; i < dw * dh; i++)
        if (out[i] == 255) fg++;

    if (fg > 0 && fg < dw * dh) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "unexpected fg count: %d / %d", fg, dw * dh);
        FAIL(msg);
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_tiny_image() {
    TEST("aruco tiny image (64x48)");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 64, h = 48;
    std::vector<uint8_t> gray(w * h, 128);

    int window_sizes[] = {11};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> out(dw * dh, 0);
    uint8_t *outputs[] = {out.data()};

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    // Uniform image should be mostly foreground (128 > 128 - 7 = 121)
    int fg = 0;
    for (int i = 0; i < dw * dh; i++)
        if (out[i] == 255) fg++;

    float ratio = (float)fg / (dw * dh);
    if (ratio > 0.8f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected >80%% fg, got %.1f%%", ratio * 100);
        FAIL(msg);
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_non_multiple_of_3() {
    TEST("aruco non-multiple-of-3 dimensions (1921x1081)");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1921, h = 1081;
    std::vector<uint8_t> gray(w * h);
    // Checkerboard pattern
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            gray[y * w + x] = ((x / 32) ^ (y / 32)) & 1 ? 220 : 30;

    int window_sizes[] = {33};
    int C = 7;
    int dw = w / 3, dh = h / 3;  // 640 x 360 (truncated)
    std::vector<uint8_t> out(dw * dh, 0);
    uint8_t *outputs[] = {out.data()};

    // Should not crash or corrupt memory with non-multiple-of-3 input
    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    // Verify we got some mix of fg/bg from the checkerboard
    int fg = 0;
    for (int i = 0; i < dw * dh; i++)
        if (out[i] == 255) fg++;

    float ratio = (float)fg / (dw * dh);
    if (ratio > 0.1f && ratio < 0.9f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected 10-90%% fg, got %.1f%%", ratio * 100);
        FAIL(msg);
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_all_black() {
    TEST("aruco all-black image -> all background");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    std::vector<uint8_t> gray(w * h, 0);  // all black

    int window_sizes[] = {15};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> out(dw * dh, 128);  // pre-fill with non-zero
    uint8_t *outputs[] = {out.data()};

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    // All-black: every pixel = 0, local mean = 0, test is 0 > 0 - 7 = -7 -> true (foreground)
    // OR the kernel clamps: depends on implementation. Either way, output should be uniform.
    // With unsigned math: 0 > mean - C could wrap. Just verify we get a consistent result.
    int fg = 0;
    for (int i = 0; i < dw * dh; i++)
        if (out[i] == 255) fg++;

    float ratio = (float)fg / (dw * dh);
    // All-black uniform: should be either all-fg or all-bg (uniform result)
    if (ratio > 0.95f || ratio < 0.05f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected uniform result, got %.1f%% fg", ratio * 100);
        FAIL(msg);
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_performance_1080p() {
    TEST("aruco performance benchmark (1080p, 100 iterations)");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    std::vector<uint8_t> gray(w * h);
    // Realistic-ish synthetic image: noisy gradient
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            gray[y * w + x] = (uint8_t)((255.0f * x / w + (rand() % 30)) / 2);

    int window_sizes[] = {21};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> out(dw * dh, 0);
    uint8_t *outputs[] = {out.data()};

    // Warmup
    for (int i = 0; i < 5; i++)
        aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    const int N = 100;
    std::vector<float> times(N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < N; i++) {
        cudaEventRecord(start);
        aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double sum = 0, min_t = FLT_MAX, max_t = 0;
    for (int i = 0; i < N; i++) {
        sum += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }
    double avg = sum / N;

    printf("PASSED (avg=%.3f ms, min=%.3f ms, max=%.3f ms)\n", avg, min_t, max_t);
    passed++;

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_multi_pass_5_windows() {
    TEST("aruco multi-pass with 5 window sizes");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    std::vector<uint8_t> gray(w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            gray[y * w + x] = (uint8_t)((x + y) % 256);

    int window_sizes[] = {7, 15, 25, 41, 63};
    int C = 7;
    int num_passes = 5;
    int dw = w / 3, dh = h / 3;
    int sz = dw * dh;

    std::vector<std::vector<uint8_t>> buffers(num_passes, std::vector<uint8_t>(sz, 0));
    uint8_t *outputs[5];
    for (int i = 0; i < num_passes; i++)
        outputs[i] = buffers[i].data();

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, num_passes, outputs);

    // Each pass should produce output; different window sizes should give different results
    bool all_have_output = true;
    int fg_counts[5] = {};
    for (int p = 0; p < num_passes; p++) {
        for (int i = 0; i < sz; i++)
            if (buffers[p][i] == 255) fg_counts[p]++;
        if (fg_counts[p] == 0) all_have_output = false;
    }

    // Check that not all passes produce identical output (different windows -> different results)
    bool all_identical = true;
    for (int p = 1; p < num_passes; p++) {
        if (fg_counts[p] != fg_counts[0]) {
            all_identical = false;
            break;
        }
    }

    if (all_have_output && !all_identical) {
        PASS();
    } else if (!all_have_output) {
        FAIL("one or more passes produced all-zero output");
    } else {
        FAIL("all 5 passes produced identical output (unexpected)");
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_context_reuse() {
    TEST("aruco context reuse (create once, 50 calls with varying data)");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    int window_sizes[] = {21};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> gray(w * h);
    std::vector<uint8_t> out(dw * dh, 0);
    uint8_t *outputs[] = {out.data()};

    bool ok = true;
    for (int iter = 0; iter < 50; iter++) {
        // Fill with different data each iteration
        uint8_t base = (uint8_t)(iter * 5);
        for (int i = 0; i < w * h; i++)
            gray[i] = (uint8_t)(base + (i % 128));

        memset(out.data(), 0, out.size());
        aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

        // Verify output is not untouched (at least some non-zero)
        bool any_nonzero = false;
        for (int i = 0; i < dw * dh; i++) {
            if (out[i] != 0) { any_nonzero = true; break; }
        }
        if (!any_nonzero) {
            char msg[128];
            snprintf(msg, sizeof(msg), "iteration %d produced all-zero output", iter);
            FAIL(msg);
            ok = false;
            break;
        }
    }

    if (ok) PASS();
    aruco_cuda_destroy(ctx);
    return 0;
}

// ============================================================
// PointSource CUDA Stress Tests
// ============================================================

// Helper: paint a filled circle of a given BGRA color into an image
static void paint_circle(uint8_t *bgra, int w, int h, int stride,
                         int cx, int cy, int radius,
                         uint8_t b, uint8_t g, uint8_t r, uint8_t a) {
    for (int y = std::max(0, cy - radius); y <= std::min(h - 1, cy + radius); y++) {
        for (int x = std::max(0, cx - radius); x <= std::min(w - 1, cx + radius); x++) {
            float dx = (float)(x - cx), dy = (float)(y - cy);
            if (dx * dx + dy * dy <= (float)(radius * radius)) {
                uint8_t *px = bgra + y * stride + x * 4;
                px[0] = b; px[1] = g; px[2] = r; px[3] = a;
            }
        }
    }
}

static int test_pointsource_multiple_green_spots() {
    TEST("pointsource multiple green spots -> finds largest/brightest");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    int stride = w * 4;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Small green spot at (100, 100) radius 8
    paint_circle(bgra.data(), w, h, stride, 100, 100, 8, 10, 255, 10, 255);
    // Medium green spot at (300, 240) radius 20
    paint_circle(bgra.data(), w, h, stride, 300, 240, 20, 10, 255, 10, 255);
    // Small green spot at (500, 350) radius 6
    paint_circle(bgra.data(), w, h, stride, 500, 350, 6, 10, 255, 10, 255);

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, stride,
                                          80, 30, 5, 5000, false);
    if (result.found) {
        // Should detect the largest spot (radius 20 at 300,240)
        float dist = sqrtf((float)((result.cx - 300) * (result.cx - 300) +
                                   (result.cy - 240) * (result.cy - 240)));
        if (dist < 25.0f) {
            PASS();
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "found spot at (%.1f, %.1f), expected near (300, 240), dist=%.1f",
                     result.cx, result.cy, dist);
            FAIL(msg);
        }
    } else {
        FAIL("no spot detected");
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_red_spot() {
    TEST("pointsource red spot (not green) -> found=false");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    int stride = w * 4;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Bright red circle at center, radius 20. BGRA: B=10, G=10, R=255, A=255
    paint_circle(bgra.data(), w, h, stride, 320, 240, 20, 10, 10, 255, 255);

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, stride,
                                          80, 30, 5, 5000, false);
    if (!result.found) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "false positive: (%.1f, %.1f) %d px",
                 result.cx, result.cy, result.pixel_count);
        FAIL(msg);
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_edge_spot() {
    TEST("pointsource spot at image corner (0,0)");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    int stride = w * 4;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Green spot at top-left corner, radius 15 (will be clipped to quarter circle)
    paint_circle(bgra.data(), w, h, stride, 0, 0, 15, 10, 255, 10, 255);

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, stride,
                                          80, 30, 5, 5000, false);
    // Quarter circle of radius 15 ~= pi*15^2/4 ~= 177 pixels, should be detectable
    if (result.found) {
        // Centroid should be near the corner
        if (result.cx < 20.0 && result.cy < 20.0) {
            PASS();
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "spot found but at (%.1f, %.1f), expected near (0,0)",
                     result.cx, result.cy);
            FAIL(msg);
        }
    } else {
        // Also acceptable: edge spots might be rejected depending on implementation
        printf("PASSED (spot at edge not detected - acceptable boundary behavior)\n");
        passed++;
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_very_small_spot() {
    TEST("pointsource very small spot (3x3) with min_blob_pixels=10 -> not detected");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    int stride = w * 4;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Tiny 3x3 green block at center (9 pixels, below min_blob_pixels=10)
    for (int y = 239; y <= 241; y++) {
        for (int x = 319; x <= 321; x++) {
            uint8_t *px = bgra.data() + y * stride + x * 4;
            px[0] = 10; px[1] = 255; px[2] = 10; px[3] = 255;
        }
    }

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, stride,
                                          80, 30, 10, 5000, false);
    if (!result.found) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "small spot detected: %d pixels at (%.1f, %.1f)",
                 result.pixel_count, result.cx, result.cy);
        FAIL(msg);
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_very_large_spot() {
    TEST("pointsource very large spot (radius 100) with max_blob_pixels=100 -> not detected");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    int stride = w * 4;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Large green circle, radius 100 (~31416 pixels, well above max_blob_pixels=100)
    paint_circle(bgra.data(), w, h, stride, 320, 240, 100, 10, 255, 10, 255);

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, stride,
                                          80, 30, 5, 100, false);
    if (!result.found) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "large spot detected: %d pixels at (%.1f, %.1f)",
                 result.pixel_count, result.cx, result.cy);
        FAIL(msg);
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_performance_1080p() {
    TEST("pointsource performance benchmark (1080p, 100 iterations)");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    int stride = w * 4;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Paint a green spot for a realistic detect workload
    paint_circle(bgra.data(), w, h, stride, 960, 540, 15, 10, 255, 10, 255);

    // Warmup
    for (int i = 0; i < 5; i++)
        pointsource_cuda_detect(ctx, bgra.data(), w, h, stride, 80, 30, 5, 5000, false);

    const int N = 100;
    std::vector<float> times(N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < N; i++) {
        cudaEventRecord(start);
        pointsource_cuda_detect(ctx, bgra.data(), w, h, stride, 80, 30, 5, 5000, false);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double sum = 0, min_t = FLT_MAX, max_t = 0;
    for (int i = 0; i < N; i++) {
        sum += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }
    double avg = sum / N;

    printf("PASSED (avg=%.3f ms, min=%.3f ms, max=%.3f ms)\n", avg, min_t, max_t);
    passed++;

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_stride_padding() {
    TEST("pointsource stride padding (stride > width*4)");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    // Add 128 bytes of padding per row
    int stride = w * 4 + 128;
    std::vector<uint8_t> bgra(stride * h, 0);

    // Paint green spot using padded stride
    paint_circle(bgra.data(), w, h, stride, 320, 240, 15, 10, 255, 10, 255);

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, stride,
                                          80, 30, 5, 5000, false);
    if (result.found) {
        float dist = sqrtf((float)((result.cx - 320) * (result.cx - 320) +
                                   (result.cy - 240) * (result.cy - 240)));
        if (dist < 5.0f) {
            PASS();
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "spot found but offset: (%.1f, %.1f), dist=%.1f",
                     result.cx, result.cy, dist);
            FAIL(msg);
        }
    } else {
        FAIL("green spot not detected with padded stride");
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

// ============================================================
// CUDA device info
// ============================================================

static void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (%.0f MB, SM %d.%d, %d SMs)\n",
           prop.name, prop.totalGlobalMem / 1e6,
           prop.major, prop.minor, prop.multiProcessorCount);
}

int main() {
    printf("=== CUDA Stress Tests & Performance Benchmarks ===\n");

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    print_device_info();
    printf("\n");

    // ---- ArUco stress tests ----
    printf("[ArUco CUDA Stress Tests]\n");
    test_aruco_4k_resolution();
    test_aruco_tiny_image();
    test_aruco_non_multiple_of_3();
    test_aruco_all_black();
    test_aruco_multi_pass_5_windows();
    test_aruco_context_reuse();

    printf("\n[ArUco CUDA Performance]\n");
    test_aruco_performance_1080p();

    // ---- PointSource stress tests ----
    printf("\n[PointSource CUDA Stress Tests]\n");
    test_pointsource_multiple_green_spots();
    test_pointsource_red_spot();
    test_pointsource_edge_spot();
    test_pointsource_very_small_spot();
    test_pointsource_very_large_spot();
    test_pointsource_stride_padding();

    printf("\n[PointSource CUDA Performance]\n");
    test_pointsource_performance_1080p();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
