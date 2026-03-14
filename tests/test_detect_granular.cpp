// test_detect_granular.cpp — Granular per-step timing of ChArUco GPU detection
// Instruments every sub-step to find the real bottleneck.
#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

// Replicate GPU-path internals step by step with individual timers
struct StepTimers {
    double gpu_thresh_ms = 0;
    double downsample_ms = 0;
    double contour_find_ms = 0;
    double contour_scale_ms = 0;
    double quad_process_ms = 0; // approxPolyDP + filter + subpix + readBits + match
    int num_contours = 0;
    int num_quads = 0;
    int num_markers = 0;
};

StepTimers detectMarkersGranular(
    const uint8_t *gray, int w, int h,
    const aruco_detect::ArUcoDictionary &dict,
    aruco_detect::GpuThresholdFunc gpu_thresh, void *gpu_ctx) {

    StepTimers t;

    // --- Step 1: GPU threshold + fused 3x downsample ---
    int small_win = std::max(3, std::min(w, h) / 40);
    if (small_win % 2 == 0) small_win++;
    int large_win = std::max(w, h) / 10;
    if (large_win % 2 == 0) large_win++;
    large_win = std::min(large_win, 255);

    std::vector<int> window_sizes = {small_win};
    if (large_win > small_win + 10)
        window_sizes.push_back(large_win);
    int num_passes = (int)window_sizes.size();
    int C = 7;

    // GPU now returns (w/3)×(h/3) downsampled binary directly
    constexpr int ds_factor = 3;
    int dw = w / 3, dh = h / 3;
    std::vector<std::vector<uint8_t>> ds_images(num_passes);
    std::vector<uint8_t *> ds_ptrs(num_passes);
    for (int p = 0; p < num_passes; p++) {
        ds_images[p].resize((size_t)dw * dh);
        ds_ptrs[p] = ds_images[p].data();
    }

    auto t0 = Clock::now();
    gpu_thresh(gpu_ctx, gray, w, h,
               window_sizes.data(), C, num_passes,
               ds_ptrs.data());
    auto t1 = Clock::now();
    t.gpu_thresh_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Step 2: Downsample is now fused into GPU kernel — no CPU work needed
    t.downsample_ms = 0;

    // --- Step 3: findContours on GPU-downsampled images ---
    int max_contour_len = 800;
    std::vector<std::vector<std::vector<Eigen::Vector2i>>> all_pass_contours(num_passes);

    auto t4 = Clock::now();
    for (int p = 0; p < num_passes; p++) {
        all_pass_contours[p] = aruco_detect::detail::findContours(
            ds_images[p], dw, dh,
            nullptr, max_contour_len / ds_factor);
    }
    auto t5 = Clock::now();
    t.contour_find_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // --- Step 4: Scale contour points back to full resolution ---
    std::vector<std::vector<Eigen::Vector2i>> all_contours;
    auto t6 = Clock::now();
    for (int p = 0; p < num_passes; p++) {
        for (auto &contour : all_pass_contours[p]) {
            for (auto &pt : contour) {
                pt.x() *= ds_factor;
                pt.y() *= ds_factor;
            }
            all_contours.push_back(std::move(contour));
        }
    }
    auto t7 = Clock::now();
    t.contour_scale_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
    t.num_contours = (int)all_contours.size();

    // --- Step 5: Quad processing (approxPolyDP, filtering, subpix, read bits, match) ---
    auto t8 = Clock::now();
    int quads = 0, markers = 0;
    for (const auto &contour : all_contours) {
        std::vector<Eigen::Vector2f> fcontour(contour.size());
        for (size_t i = 0; i < contour.size(); i++)
            fcontour[i] = contour[i].cast<float>();

        float peri = aruco_detect::detail::contourPerimeter(fcontour);
        float epsilon = 0.05f * peri;
        auto poly = aruco_detect::detail::approxPolyDP(fcontour, epsilon);

        if (poly.size() != 4) continue;
        if (!aruco_detect::detail::isConvex(poly)) continue;

        float area = std::abs(aruco_detect::detail::polygonArea(poly));
        if (area < 100.0f) continue;

        float min_side = 1e9f, max_side = 0.0f;
        for (int i = 0; i < 4; i++) {
            float side = (poly[(i + 1) % 4] - poly[i]).norm();
            min_side = std::min(min_side, side);
            max_side = std::max(max_side, side);
        }
        if (min_side < 10.0f) continue;
        if (max_side > 4.0f * min_side) continue;

        quads++;

        std::array<Eigen::Vector2f, 4> quad = {poly[0], poly[1], poly[2], poly[3]};
        aruco_detect::detail::orderQuadCorners(quad);

        // Subpixel refinement
        {
            std::vector<Eigen::Vector2f> qv(quad.begin(), quad.end());
            aruco_detect::cornerSubPix(gray, w, h, qv, 5, 30, 0.01f);
            for (int i = 0; i < 4; i++) quad[i] = qv[i];
        }

        // Winding
        float signed_area =
            (quad[1].x() - quad[0].x()) * (quad[2].y() - quad[0].y()) -
            (quad[2].x() - quad[0].x()) * (quad[1].y() - quad[0].y());
        if (signed_area < 0) std::swap(quad[1], quad[3]);

        // Read bits
        uint64_t bits = 0;
        if (!aruco_detect::detail::readMarkerBits(gray, w, h, quad, dict.marker_bits, bits))
            continue;

        // Dictionary matching
        int best_id = -1, best_hamming = dict.marker_bits * dict.marker_bits + 1;
        uint64_t rotated = bits;
        for (int rot = 0; rot < 4; rot++) {
            for (int mid = 0; mid < dict.num_markers; mid++) {
                int hd = aruco_detect::detail::hammingDistance(rotated, dict.patterns[mid]);
                if (hd <= dict.max_correction_bits && hd < best_hamming) {
                    best_hamming = hd;
                    best_id = mid;
                }
            }
            if (rot < 3)
                rotated = aruco_detect::detail::rotateBits90CW(rotated, dict.marker_bits);
        }
        if (best_id >= 0) markers++;
    }
    auto t9 = Clock::now();
    t.quad_process_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
    t.num_quads = quads;
    t.num_markers = markers;

    return t;
}

int main(int argc, char **argv) {
    const char *img_dir =
        "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/images";
    const char *single_img =
        "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/images/2002486_1.jpg";

    auto dict = aruco_detect::getDictionary(0);
    aruco_detect::CharucoBoard board;
    board.squares_x = 5;
    board.squares_y = 5;
    board.square_length = 50.0f;
    board.marker_length = 37.5f;
    board.dictionary_id = 0;

#ifdef __APPLE__
    auto metal = aruco_metal_create();
    if (!metal) {
        std::cerr << "Failed to create Metal context\n";
        return 1;
    }

    // ─── Part 1: Single image granular timing (10 iterations) ───
    printf("=== SINGLE IMAGE GRANULAR TIMING (10 iterations) ===\n");
    {
        int w, h, ch;
        unsigned char *pixels = stbi_load(single_img, &w, &h, &ch, 1);
        if (!pixels) { std::cerr << "Failed to load\n"; return 1; }
        printf("Image: %dx%d\n\n", w, h);

        // Warmup
        {
            auto r = aruco_detect::detectCharucoBoard(pixels, w, h, board, dict,
                aruco_metal_threshold_batch, metal);
        }

        StepTimers avg = {};
        int N = 10;
        for (int i = 0; i < N; i++) {
            auto t = detectMarkersGranular(pixels, w, h, dict,
                aruco_metal_threshold_batch, metal);
            avg.gpu_thresh_ms += t.gpu_thresh_ms;
            avg.downsample_ms += t.downsample_ms;
            avg.contour_find_ms += t.contour_find_ms;
            avg.contour_scale_ms += t.contour_scale_ms;
            avg.quad_process_ms += t.quad_process_ms;
            avg.num_contours += t.num_contours;
            avg.num_quads += t.num_quads;
            avg.num_markers += t.num_markers;
        }
        printf("  GPU threshold:     %6.1f ms\n", avg.gpu_thresh_ms / N);
        printf("  3x downsample:     %6.1f ms  (2 passes)\n", avg.downsample_ms / N);
        printf("  findContours:      %6.1f ms  (%d contours avg)\n",
               avg.contour_find_ms / N, avg.num_contours / N);
        printf("  Scale contours:    %6.1f ms\n", avg.contour_scale_ms / N);
        printf("  Quad processing:   %6.1f ms  (%d quads, %d markers avg)\n",
               avg.quad_process_ms / N, avg.num_quads / N, avg.num_markers / N);
        double total = (avg.gpu_thresh_ms + avg.downsample_ms + avg.contour_find_ms +
                        avg.contour_scale_ms + avg.quad_process_ms) / N;
        printf("  ─────────────────────────\n");
        printf("  Sum of steps:      %6.1f ms\n", total);

        // Compare with full detectCharucoBoard
        auto ta = Clock::now();
        for (int i = 0; i < N; i++) {
            auto r = aruco_detect::detectCharucoBoard(pixels, w, h, board, dict,
                aruco_metal_threshold_batch, metal);
        }
        auto tb = Clock::now();
        double full_avg = std::chrono::duration<double, std::milli>(tb - ta).count() / N;
        printf("  Full detect:       %6.1f ms\n", full_avg);
        printf("  Unaccounted:       %6.1f ms\n\n", full_avg - total);

        stbi_image_free(pixels);
    }

    // ─── Part 2: Additional single-image analysis ───
    printf("=== STEP 2 BREAKDOWN: GPU-FUSED DOWNSAMPLE ===\n");
    {
        int w, h, ch;
        unsigned char *pixels = stbi_load(single_img, &w, &h, &ch, 1);

        // GPU now returns downsampled binary directly
        int dw = w / 3, dh = h / 3;
        std::vector<uint8_t> ds_binary(dw * dh);
        uint8_t *bp = ds_binary.data();
        int win = std::max(3, std::min(w, h) / 40);
        if (win % 2 == 0) win++;
        aruco_metal_threshold_batch(metal, pixels, w, h, &win, 7, 1, &bp);

        printf("  GPU threshold+downsample (%dx%d → %dx%d): fused on GPU\n",
               w, h, dw, dh);

        // Count foreground pixels in downsampled output
        int fg_small = 0;
        for (auto v : ds_binary) if (v) fg_small++;
        printf("  Downsampled: %d foreground / %d total (%.1f%%)\n\n",
               fg_small, dw*dh, 100.0 * fg_small / (dw*dh));

        // Time findContours alone on downsampled
        auto t2 = Clock::now();
        int total_contours = 0;
        for (int i = 0; i < 100; i++) {
            auto c = aruco_detect::detail::findContours(ds_binary, dw, dh, nullptr, 800/3);
            total_contours = (int)c.size();
        }
        auto t3 = Clock::now();
        printf("  findContours on %dx%d: %.2f ms (%d contours) (avg of 100)\n",
               dw, dh,
               std::chrono::duration<double, std::milli>(t3 - t2).count() / 100.0,
               total_contours);

        // Time findContours WITHOUT max_contour_length limit
        auto t4 = Clock::now();
        int total_contours2 = 0;
        for (int i = 0; i < 10; i++) {
            auto c = aruco_detect::detail::findContours(ds_binary, dw, dh, nullptr, 0);
            total_contours2 = (int)c.size();
        }
        auto t5 = Clock::now();
        printf("  findContours NO limit on %dx%d: %.2f ms (%d contours) (avg of 10)\n\n",
               dw, dh,
               std::chrono::duration<double, std::milli>(t5 - t4).count() / 10.0,
               total_contours2);

        stbi_image_free(pixels);
    }

    // ─── Part 3: Image loading comparison ───
    printf("=== IMAGE LOADING ===\n");
    {
        // stbi_load timing for 10 images
        auto t0 = Clock::now();
        int N = 10;
        for (int i = 1; i <= N; i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/2002486_%d.jpg", img_dir, i);
            int w, h, ch;
            unsigned char *pixels = stbi_load(path, &w, &h, &ch, 1);
            if (pixels) stbi_image_free(pixels);
        }
        auto t1 = Clock::now();
        printf("  stbi_load (grayscale): %.1f ms/image (avg of %d)\n",
               std::chrono::duration<double, std::milli>(t1 - t0).count() / N, N);

        // stbi_load RGB then manual grayscale
        auto t2 = Clock::now();
        for (int i = 1; i <= N; i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/2002486_%d.jpg", img_dir, i);
            int w, h, ch;
            unsigned char *pixels = stbi_load(path, &w, &h, &ch, 3);
            if (pixels) stbi_image_free(pixels);
        }
        auto t3 = Clock::now();
        printf("  stbi_load (RGB):       %.1f ms/image (avg of %d)\n",
               std::chrono::duration<double, std::milli>(t3 - t2).count() / N, N);
    }

    // ─── Part 4: Scan ALL images, count detections vs failures ───
    printf("\n=== FULL DATASET SCAN ===\n");
    {
        std::vector<std::string> paths;
        for (auto &entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".jpg")
                paths.push_back(entry.path().string());
        }
        std::sort(paths.begin(), paths.end());
        printf("  Total images: %zu\n", paths.size());

        int detected = 0, failed = 0;
        double total_load_ms = 0, total_detect_ms = 0;
        double total_load_detected = 0, total_detect_detected = 0;
        double total_load_failed = 0, total_detect_failed = 0;

        StepTimers sum_detected = {}, sum_failed = {};
        int n_detected_granular = 0, n_failed_granular = 0;

        for (size_t i = 0; i < paths.size(); i++) {
            auto tl0 = Clock::now();
            int w, h, ch;
            unsigned char *pixels = stbi_load(paths[i].c_str(), &w, &h, &ch, 1);
            auto tl1 = Clock::now();
            double load_ms = std::chrono::duration<double, std::milli>(tl1 - tl0).count();

            if (!pixels) continue;

            auto td0 = Clock::now();
            auto result = aruco_detect::detectCharucoBoard(
                pixels, w, h, board, dict,
                aruco_metal_threshold_batch, metal);
            auto td1 = Clock::now();
            double det_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();

            total_load_ms += load_ms;
            total_detect_ms += det_ms;

            if (result.corners.size() >= 4) {
                detected++;
                total_load_detected += load_ms;
                total_detect_detected += det_ms;
            } else {
                failed++;
                total_load_failed += load_ms;
                total_detect_failed += det_ms;
            }

            // Granular timing on a sample: every 50th image
            if (i % 50 == 0) {
                auto gt = detectMarkersGranular(pixels, w, h, dict,
                    aruco_metal_threshold_batch, metal);
                if (result.corners.size() >= 4) {
                    sum_detected.gpu_thresh_ms += gt.gpu_thresh_ms;
                    sum_detected.downsample_ms += gt.downsample_ms;
                    sum_detected.contour_find_ms += gt.contour_find_ms;
                    sum_detected.contour_scale_ms += gt.contour_scale_ms;
                    sum_detected.quad_process_ms += gt.quad_process_ms;
                    sum_detected.num_contours += gt.num_contours;
                    n_detected_granular++;
                } else {
                    sum_failed.gpu_thresh_ms += gt.gpu_thresh_ms;
                    sum_failed.downsample_ms += gt.downsample_ms;
                    sum_failed.contour_find_ms += gt.contour_find_ms;
                    sum_failed.contour_scale_ms += gt.contour_scale_ms;
                    sum_failed.quad_process_ms += gt.quad_process_ms;
                    sum_failed.num_contours += gt.num_contours;
                    n_failed_granular++;
                }
            }

            stbi_image_free(pixels);
        }

        printf("\n  Detection results:\n");
        printf("    Board found (>=4 corners): %d (%.0f%%)\n",
               detected, 100.0 * detected / (detected + failed));
        printf("    No board:                  %d (%.0f%%)\n",
               failed, 100.0 * failed / (detected + failed));

        printf("\n  Average times:\n");
        printf("    Load:      %.1f ms/img\n", total_load_ms / (detected + failed));
        printf("    Detect:    %.1f ms/img\n", total_detect_ms / (detected + failed));
        printf("    Total:     %.1f ms/img (%.1f img/s)\n",
               (total_load_ms + total_detect_ms) / (detected + failed),
               1000.0 * (detected + failed) / (total_load_ms + total_detect_ms));

        if (detected > 0) {
            printf("\n    Board FOUND images:\n");
            printf("      Load:   %.1f ms/img\n", total_load_detected / detected);
            printf("      Detect: %.1f ms/img\n", total_detect_detected / detected);
        }
        if (failed > 0) {
            printf("\n    NO BOARD images:\n");
            printf("      Load:   %.1f ms/img\n", total_load_failed / failed);
            printf("      Detect: %.1f ms/img\n", total_detect_failed / failed);
        }

        // Print granular averages
        if (n_detected_granular > 0) {
            int n = n_detected_granular;
            printf("\n  Granular avg (board found, %d samples):\n", n);
            printf("    GPU thresh:   %6.1f ms\n", sum_detected.gpu_thresh_ms / n);
            printf("    Downsample:   %6.1f ms\n", sum_detected.downsample_ms / n);
            printf("    findContours: %6.1f ms (%d contours avg)\n",
                   sum_detected.contour_find_ms / n, sum_detected.num_contours / n);
            printf("    Scale:        %6.1f ms\n", sum_detected.contour_scale_ms / n);
            printf("    Quad proc:    %6.1f ms\n", sum_detected.quad_process_ms / n);
        }
        if (n_failed_granular > 0) {
            int n = n_failed_granular;
            printf("\n  Granular avg (no board, %d samples):\n", n);
            printf("    GPU thresh:   %6.1f ms\n", sum_failed.gpu_thresh_ms / n);
            printf("    Downsample:   %6.1f ms\n", sum_failed.downsample_ms / n);
            printf("    findContours: %6.1f ms (%d contours avg)\n",
                   sum_failed.contour_find_ms / n, sum_failed.num_contours / n);
            printf("    Scale:        %6.1f ms\n", sum_failed.contour_scale_ms / n);
            printf("    Quad proc:    %6.1f ms\n", sum_failed.quad_process_ms / n);
        }

        printf("\n  Wall clock total: %.1f s load + %.1f s detect = %.1f s\n",
               total_load_ms / 1000.0, total_detect_ms / 1000.0,
               (total_load_ms + total_detect_ms) / 1000.0);
    }

    aruco_metal_destroy(metal);
#else
    printf("macOS only (needs Metal GPU threshold)\n");
#endif
    return 0;
}
