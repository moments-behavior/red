// Test CUDA kernels (ArUco thresholding + PointSource detection) with synthetic data.
// Build: nvcc -o test_cuda_kernels test_cuda_kernels.cu ../src/aruco_cuda.cu ../src/pointsource_cuda.cu
//        -I../src -DUSE_CUDA_POINTSOURCE -lcuda -lcudart
// Run:   ./test_cuda_kernels

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
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
// ArUco CUDA tests
// ============================================================

static int test_aruco_create_destroy() {
    TEST("aruco_cuda_create/destroy");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create returned nullptr"); return 0; }
    aruco_cuda_destroy(ctx);
    PASS();
    return 0;
}

static int test_aruco_uniform_white() {
    TEST("aruco uniform white image → all foreground");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    std::vector<uint8_t> gray(w * h, 200); // uniform brightness

    int window_sizes[] = {15};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> out(dw * dh, 0);
    uint8_t *outputs[] = {out.data()};

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    // Uniform image: every pixel equals its local mean, so pixel > mean - C
    // should be true for most pixels (200 > 200 - 7 = 193 → foreground=255)
    int fg_count = 0;
    for (int i = 0; i < dw * dh; i++) {
        if (out[i] == 255) fg_count++;
    }
    float fg_ratio = (float)fg_count / (dw * dh);
    if (fg_ratio > 0.95f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected >95%% foreground, got %.1f%%", fg_ratio * 100);
        FAIL(msg);
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_checkerboard() {
    TEST("aruco checkerboard → mixed output");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 960, h = 540;
    std::vector<uint8_t> gray(w * h);
    // 32-pixel checkerboard
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int bx = (x / 32) & 1, by = (y / 32) & 1;
            gray[y * w + x] = (bx ^ by) ? 240 : 20;
        }
    }

    int window_sizes[] = {33}; // just larger than squares
    int C = 7;
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> out(dw * dh, 128);
    uint8_t *outputs[] = {out.data()};

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 1, outputs);

    // Should produce mix of fg and bg
    int fg = 0;
    for (int i = 0; i < dw * dh; i++) {
        if (out[i] == 255) fg++;
    }
    float ratio = (float)fg / (dw * dh);
    if (ratio > 0.2f && ratio < 0.8f) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected 20-80%% foreground, got %.1f%%", ratio * 100);
        FAIL(msg);
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

static int test_aruco_multi_pass() {
    TEST("aruco multi-pass (3 window sizes)");
    auto ctx = aruco_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 1920, h = 1080;
    std::vector<uint8_t> gray(w * h);
    // Gradient
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            gray[y * w + x] = (uint8_t)(255.0f * x / w);

    int window_sizes[] = {11, 25, 51};
    int C = 7;
    int dw = w / 3, dh = h / 3;
    int sz = dw * dh;
    std::vector<uint8_t> out0(sz, 0), out1(sz, 0), out2(sz, 0);
    uint8_t *outputs[] = {out0.data(), out1.data(), out2.data()};

    aruco_cuda_threshold_batch(ctx, gray.data(), w, h, window_sizes, C, 3, outputs);

    // All three should produce output (not all zeros)
    bool any_nonzero[3] = {};
    for (int i = 0; i < sz; i++) {
        if (out0[i]) any_nonzero[0] = true;
        if (out1[i]) any_nonzero[1] = true;
        if (out2[i]) any_nonzero[2] = true;
    }
    if (any_nonzero[0] && any_nonzero[1] && any_nonzero[2]) {
        PASS();
    } else {
        FAIL("one or more passes produced all-zero output");
    }

    aruco_cuda_destroy(ctx);
    return 0;
}

// ============================================================
// PointSource CUDA tests
// ============================================================

static int test_pointsource_create_destroy() {
    TEST("pointsource_cuda_create/destroy");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create returned nullptr"); return 0; }
    pointsource_cuda_destroy(ctx);
    PASS();
    return 0;
}

static int test_pointsource_no_spot() {
    TEST("pointsource detect on black image → no spot");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    std::vector<uint8_t> bgra(w * h * 4, 0); // all black

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, w * 4,
                                          80, 30, 5, 5000, false);
    if (!result.found) {
        PASS();
    } else {
        char msg[128];
        snprintf(msg, sizeof(msg), "false positive at (%.1f, %.1f) with %d pixels",
                 result.cx, result.cy, result.pixel_count);
        FAIL(msg);
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_green_spot() {
    TEST("pointsource detect bright green spot → found");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    std::vector<uint8_t> bgra(w * h * 4, 0);

    // Paint a bright green circle at center (320, 240), radius 15
    int cx = 320, cy = 240, r = 15;
    int pixel_count = 0;
    for (int y = cy - r; y <= cy + r; y++) {
        for (int x = cx - r; x <= cx + r; x++) {
            if (x >= 0 && x < w && y >= 0 && y < h) {
                float dx = (float)(x - cx), dy = (float)(y - cy);
                if (dx * dx + dy * dy <= r * r) {
                    int idx = (y * w + x) * 4;
                    bgra[idx + 0] = 20;  // B
                    bgra[idx + 1] = 255; // G
                    bgra[idx + 2] = 20;  // R
                    bgra[idx + 3] = 255; // A
                    pixel_count++;
                }
            }
        }
    }

    auto result = pointsource_cuda_detect(ctx, bgra.data(), w, h, w * 4,
                                          80, 30, 5, 5000, false);
    if (result.found) {
        float dist = sqrtf((float)((result.cx - cx) * (result.cx - cx) +
                                   (result.cy - cy) * (result.cy - cy)));
        if (dist < 5.0f) {
            PASS();
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "spot found but too far: (%.1f, %.1f), dist=%.1f",
                     result.cx, result.cy, dist);
            FAIL(msg);
        }
    } else {
        FAIL("green spot not detected");
    }

    pointsource_cuda_destroy(ctx);
    return 0;
}

static int test_pointsource_viz() {
    TEST("pointsource viz produces non-zero output");
    auto ctx = pointsource_cuda_create();
    if (!ctx) { FAIL("create failed"); return 0; }

    const int w = 640, h = 480;
    std::vector<uint8_t> bgra(w * h * 4, 0);

    // Paint green spot
    for (int y = 230; y < 250; y++)
        for (int x = 310; x < 330; x++) {
            int idx = (y * w + x) * 4;
            bgra[idx + 0] = 10; bgra[idx + 1] = 250;
            bgra[idx + 2] = 10; bgra[idx + 3] = 255;
        }

    std::vector<uint8_t> rgba_out(w * h * 4, 0);
    auto vr = pointsource_cuda_detect_viz(ctx, bgra.data(), w, h, w * 4,
                                          80, 30, 5, 5000, rgba_out.data());

    int nonzero = 0;
    for (int i = 0; i < w * h * 4; i++)
        if (rgba_out[i] != 0) nonzero++;

    if (nonzero > 0) {
        PASS();
    } else {
        FAIL("viz output was all zeros");
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
    printf("=== CUDA Kernel Tests ===\n");

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    print_device_info();
    printf("\n");

    printf("[ArUco CUDA Thresholding]\n");
    test_aruco_create_destroy();
    test_aruco_uniform_white();
    test_aruco_checkerboard();
    test_aruco_multi_pass();

    printf("\n[PointSource CUDA Detection]\n");
    test_pointsource_create_destroy();
    test_pointsource_no_spot();
    test_pointsource_green_spot();
    test_pointsource_viz();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
