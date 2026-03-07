// test_contour_drill.cpp — Deep drill into findContours behavior
// Why does max_contour_length=266 take 64ms but unlimited takes 11ms?
#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;

int main() {
    const char *img_path =
        "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/images/2002486_1.jpg";

#ifdef __APPLE__
    auto metal = aruco_metal_create();
    int w, h, ch;
    unsigned char *pixels = stbi_load(img_path, &w, &h, &ch, 1);
    if (!pixels) { std::cerr << "Failed\n"; return 1; }

    // GPU threshold + fused 3x downsample
    int dw = w / 3, dh = h / 3;
    std::vector<uint8_t> small(dw * dh);
    uint8_t *bp = small.data();
    int win = std::max(3, std::min(w, h) / 40);
    if (win % 2 == 0) win++;
    aruco_metal_threshold_batch(metal, pixels, w, h, &win, 7, 1, &bp);
    printf("Downsampled: %dx%d\n", dw, dh);

    // Count foreground
    int fg = 0;
    for (auto v : small) if (v) fg++;
    printf("Foreground: %d / %d (%.1f%%)\n\n", fg, dw*dh, 100.0*fg/(dw*dh));

    // Test different max_contour_length values
    int limits[] = {0, 100, 200, 266, 400, 600, 800, 1000, 2000, 5000};
    for (int lim : limits) {
        auto t0 = Clock::now();
        int nc = 0;
        int iters = 50;
        for (int i = 0; i < iters; i++) {
            auto c = aruco_detect::detail::findContours(small, dw, dh, nullptr, lim);
            nc = (int)c.size();
        }
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        printf("  max_contour_length=%5d: %6.2f ms  (%d contours)\n", lim, ms, nc);
    }

    // Key question: when max_contour_length is set, contours that exceed
    // it are DISCARDED but their pixels are already traced (visited). When
    // it's 0, max_steps = w*h, meaning all contours are kept.
    // The issue: with a small limit, big contours get aborted partway
    // through tracing. The visited flags for the traced portion are set,
    // but the REST of the contour boundary pixels are NOT visited.
    // This means they get re-found as new start pixels, causing
    // the same large contour to be re-traced many times!

    printf("\n--- Contour size histogram (no limit) ---\n");
    {
        auto contours = aruco_detect::detail::findContours(small, dw, dh, nullptr, 0);
        // Histogram by size buckets
        int buckets[] = {0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 100000};
        int counts[9] = {};
        int max_size = 0;
        for (auto &c : contours) {
            int sz = (int)c.size();
            if (sz > max_size) max_size = sz;
            for (int b = 0; b < 9; b++) {
                if (sz >= buckets[b] && sz < buckets[b+1]) {
                    counts[b]++;
                    break;
                }
            }
        }
        for (int b = 0; b < 9; b++) {
            if (counts[b] > 0)
                printf("  [%5d-%5d): %d contours\n", buckets[b], buckets[b+1], counts[b]);
        }
        printf("  Total: %zu contours, max size: %d\n", contours.size(), max_size);

        // Show the 5 largest
        std::vector<int> sizes;
        for (auto &c : contours) sizes.push_back((int)c.size());
        std::sort(sizes.rbegin(), sizes.rend());
        printf("  Top 5 sizes:");
        for (int i = 0; i < std::min(5, (int)sizes.size()); i++)
            printf(" %d", sizes[i]);
        printf("\n");
    }

    printf("\n--- With max_contour_length=266 ---\n");
    {
        auto contours = aruco_detect::detail::findContours(small, dw, dh, nullptr, 266);
        int max_size = 0;
        for (auto &c : contours) {
            int sz = (int)c.size();
            if (sz > max_size) max_size = sz;
        }
        printf("  Total: %zu contours, max size: %d\n", contours.size(), max_size);

        std::vector<int> sizes;
        for (auto &c : contours) sizes.push_back((int)c.size());
        std::sort(sizes.rbegin(), sizes.rend());
        printf("  Top 5 sizes:");
        for (int i = 0; i < std::min(5, (int)sizes.size()); i++)
            printf(" %d", sizes[i]);
        printf("\n");
    }

    // --- Now test: what if we DON'T discard large contours but still trace them? ---
    // The fix: when a contour exceeds the limit, DON'T add it to results
    // but still mark all its pixels as visited so it isn't re-traced.
    // We can approximate this by running without limit, then filtering.
    printf("\n--- Find all, then filter ---\n");
    {
        auto t0 = Clock::now();
        int nc = 0;
        for (int i = 0; i < 50; i++) {
            auto contours = aruco_detect::detail::findContours(small, dw, dh, nullptr, 0);
            // Filter out contours > 800/3
            contours.erase(
                std::remove_if(contours.begin(), contours.end(),
                    [](const auto &c) { return (int)c.size() > 800/3; }),
                contours.end());
            nc = (int)contours.size();
        }
        auto t1 = Clock::now();
        printf("  Find all + filter: %.2f ms  (%d contours)\n",
               std::chrono::duration<double, std::milli>(t1 - t0).count() / 50, nc);
    }

    stbi_image_free(pixels);
    aruco_metal_destroy(metal);
#endif
    return 0;
}
