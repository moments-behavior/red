// test_detect_timing.cpp — Benchmark per-stage timing of ChArUco detection
// on a single image to find the actual bottleneck.
#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;

int main() {
    // Load one test image
    const char *img_path =
        "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/images/2002486_1.jpg";

    auto t0 = Clock::now();
    int w = 0, h = 0, channels = 0;
    unsigned char *pixels = stbi_load(img_path, &w, &h, &channels, 1);
    auto t1 = Clock::now();

    if (!pixels) {
        std::cerr << "Failed to load image\n";
        return 1;
    }
    printf("Image: %dx%d, load: %.1f ms\n", w, h,
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Set up board parameters (5x5 ChArUco, DICT_4X4_50)
    aruco_detect::CharucoBoard board;
    board.squares_x = 5;
    board.squares_y = 5;
    board.square_length = 50.0f;
    board.marker_length = 37.5f;
    board.dictionary_id = 0;
    auto dict = aruco_detect::getDictionary(0);

    // ─── CPU path timing ───
    printf("\n=== CPU path ===\n");
    {
        auto ta = Clock::now();
        aruco_detect::detail::IntegralImage integral;
        integral.compute(pixels, w, h);
        auto tb = Clock::now();
        printf("  Integral image: %.1f ms\n",
               std::chrono::duration<double, std::milli>(tb - ta).count());

        // Single threshold pass
        std::vector<uint8_t> bin;
        int win = std::max(3, std::min(w, h) / 40);
        if (win % 2 == 0) win++;
        auto tc = Clock::now();
        aruco_detect::detail::adaptiveThreshold(pixels, w, h, bin, integral, win, 7);
        auto td = Clock::now();
        printf("  Threshold (win=%d): %.1f ms\n", win,
               std::chrono::duration<double, std::milli>(td - tc).count());

        // Contour finding
        auto te = Clock::now();
        auto contours = aruco_detect::detail::findContours(bin, w, h, nullptr, std::max(w,h)*2);
        auto tf = Clock::now();
        printf("  Contours: %.1f ms (%zu contours)\n",
               std::chrono::duration<double, std::milli>(tf - te).count(), contours.size());
    }

    // Full detection (CPU)
    {
        auto ta = Clock::now();
        auto result = aruco_detect::detectCharucoBoard(pixels, w, h, board, dict);
        auto tb = Clock::now();
        printf("  Full detectCharucoBoard: %.1f ms (%zu corners)\n",
               std::chrono::duration<double, std::milli>(tb - ta).count(),
               result.corners.size());

        // Subpixel refinement
        auto tc = Clock::now();
        aruco_detect::cornerSubPix(pixels, w, h, result.corners, 5, 30, 0.01f);
        auto td = Clock::now();
        printf("  cornerSubPix: %.1f ms\n",
               std::chrono::duration<double, std::milli>(td - tc).count());

        printf("  TOTAL CPU: %.1f ms\n",
               std::chrono::duration<double, std::milli>(td - ta).count());
    }

#ifdef __APPLE__
    // ─── GPU path timing ───
    printf("\n=== GPU path ===\n");
    auto metal = aruco_metal_create();
    if (metal) {
        // Warm up
        {
            auto result = aruco_detect::detectCharucoBoard(
                pixels, w, h, board, dict,
                aruco_metal_threshold_batch, metal);
        }

        auto ta = Clock::now();
        auto result = aruco_detect::detectCharucoBoard(
            pixels, w, h, board, dict,
            aruco_metal_threshold_batch, metal);
        auto tb = Clock::now();
        printf("  Full detectCharucoBoard (GPU): %.1f ms (%zu corners)\n",
               std::chrono::duration<double, std::milli>(tb - ta).count(),
               result.corners.size());

        auto tc = Clock::now();
        aruco_detect::cornerSubPix(pixels, w, h, result.corners, 5, 30, 0.01f);
        auto td = Clock::now();
        printf("  cornerSubPix: %.1f ms\n",
               std::chrono::duration<double, std::milli>(td - tc).count());

        printf("  TOTAL GPU: %.1f ms\n",
               std::chrono::duration<double, std::milli>(td - ta).count());

        // Run 10 iterations to get stable average
        printf("\n  10-iteration average:\n");
        auto start = Clock::now();
        for (int i = 0; i < 10; i++) {
            auto r = aruco_detect::detectCharucoBoard(
                pixels, w, h, board, dict,
                aruco_metal_threshold_batch, metal);
            aruco_detect::cornerSubPix(pixels, w, h, r.corners, 5, 30, 0.01f);
        }
        auto end = Clock::now();
        double avg = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
        printf("  Avg per image: %.1f ms\n", avg);

        aruco_metal_destroy(metal);
    }
#endif

    stbi_image_free(pixels);
    return 0;
}
