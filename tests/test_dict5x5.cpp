// Quick test: detect ChArUco board with DICT_5X5_100 on rat calibration images
#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // Get dictionary 5 = DICT_5X5_100
    auto dict = aruco_detect::getDictionary(5);
    printf("Dictionary: id=5, markers=%d, bits=%d, max_correction=%d, valid=%d\n",
           dict.num_markers, dict.marker_bits, dict.max_correction_bits, dict.valid());
    if (!dict.valid()) {
        printf("ERROR: dictionary not valid!\n");
        return 1;
    }

    // Set up ChArUco board (5x5, 80mm squares, 60mm markers)
    aruco_detect::CharucoBoard board;
    board.squares_x = 5;
    board.squares_y = 5;
    board.square_length = 80.0f;
    board.marker_length = 60.0f;
    board.dictionary_id = 5;

    // Create Metal context for GPU threshold
#ifdef __APPLE__
    auto metal_ctx = aruco_metal_create();
    aruco_detect::GpuThresholdFunc gpu_fn = metal_ctx ? aruco_metal_threshold_batch : nullptr;
#else
    void *metal_ctx = nullptr;
    aruco_detect::GpuThresholdFunc gpu_fn = nullptr;
#endif

    std::string img_dir = "/Users/johnsonr/datasets/rat/calibration_images/2025_08_14_09_23_31/";

    // Test a few images from different cameras
    std::vector<std::string> test_images = {
        "2002479_0.jpg", "2002479_3.jpg",
        "2002480_0.jpg", "2002480_3.jpg",
        "2002484_0.jpg", "2002484_3.jpg",
        "2002490_0.jpg", "2002490_3.jpg",
    };

    int total_markers = 0, total_corners = 0, images_with_board = 0;

    for (const auto &fname : test_images) {
        std::string path = img_dir + fname;
        if (!fs::exists(path)) {
            printf("  %s: NOT FOUND\n", fname.c_str());
            continue;
        }

        // Load as grayscale using calibration_pipeline's stb path
        int w, h, ch;
        uint8_t *data = stbi_load(path.c_str(), &w, &h, &ch, 1);
        if (!data) {
            printf("  %s: LOAD FAILED\n", fname.c_str());
            continue;
        }

        // Detect markers
        auto markers = aruco_detect::detectMarkers(data, w, h, dict, gpu_fn, metal_ctx);

        // Detect ChArUco board
        auto charuco = aruco_detect::detectCharucoBoard(data, w, h, board, dict, gpu_fn, metal_ctx);

        printf("  %s (%dx%d): %d markers, %d charuco corners\n",
               fname.c_str(), w, h, (int)markers.size(), (int)charuco.ids.size());

        for (const auto &m : markers)
            printf("    marker id=%d hamming=%d\n", m.id, m.hamming_distance);

        total_markers += (int)markers.size();
        total_corners += (int)charuco.ids.size();
        if (!charuco.ids.empty()) images_with_board++;

        free(data);
    }

    printf("\nSummary: %d images tested, %d with board detections\n",
           (int)test_images.size(), images_with_board);
    printf("  Total markers: %d, total charuco corners: %d\n",
           total_markers, total_corners);

#ifdef __APPLE__
    if (metal_ctx) aruco_metal_destroy(metal_ctx);
#endif
    return (images_with_board > 0) ? 0 : 1;
}
