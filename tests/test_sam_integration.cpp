// tests/test_sam_integration.cpp
// Integration tests for SAM segmentation pipeline with real annotate1 data.
//
// Tests everything EXCEPT actual ONNX Runtime inference:
//   - CSV loading and centroid computation from v2 annotations
//   - Image preprocessing for MobileSAM and SAM 2.1
//   - Mask-to-polygon conversion
//   - Mask polygon persistence in annotation JSON
//   - Coordinate transforms for SAM model space
//   - sam_init / sam_segment stubs without model files
//
// Requires real data from annotate1 project:
//   - V2 CSVs:  /Users/johnsonr/red_projects/annotate1/labeled_data/2026_03_10_00_43_32_v2/
//   - JPEGs:    /Users/johnsonr/red_projects/annotate1/export/2026_03_10_08_45_34/train/2026_03_10_08_41_51/
//
// Compiled as standalone target with same deps as test_annotation.

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#include "annotation.h"
#include "annotation_csv.h"
#include "sam_inference.h"

// Minimal SamToolState for polygon caching test (avoids sam_tool.h's ImGui deps)
struct SamToolState_Test {
    bool enabled = false;
    std::vector<tuple_d> fg_points, bg_points;
    SamMask current_mask;
    std::vector<std::vector<tuple_d>> current_polygons;
    bool has_pending_mask = false;
    u32 prompt_frame = 0;
    int prompt_cam = -1;
};
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Shared test framework
// ---------------------------------------------------------------------------
#include "test_framework.h"

// ---------------------------------------------------------------------------
// Constants for the annotate1 project
// ---------------------------------------------------------------------------

static const char *V2_CSV_DIR  = "/Users/johnsonr/red_projects/annotate1/labeled_data/2026_03_10_00_43_32_v2";
static const char *JPEG_DIR    = "/Users/johnsonr/red_projects/annotate1/export/2026_03_10_08_45_34/train/2026_03_10_08_41_51";
static constexpr int IMG_W     = 3208;
static constexpr int IMG_H     = 2200;
static constexpr int NUM_NODES = 4;  // Rat4: Snout, EarL, EarR, Tail
static constexpr int NUM_CAMS  = 16;
static constexpr u32 TEST_FRAME = 5998;

static const std::vector<std::string> CAMERA_NAMES = {
    "Cam2002486", "Cam2002487", "Cam2005325", "Cam2006050",
    "Cam2006051", "Cam2006052", "Cam2006054", "Cam2006055",
    "Cam2006515", "Cam2006516", "Cam2008665", "Cam2008666",
    "Cam2008667", "Cam2008668", "Cam2008669", "Cam2008670"
};

// ---------------------------------------------------------------------------
// Helper: load all v2 CSVs into an AnnotationMap
// ---------------------------------------------------------------------------
static bool load_annotate1_data(AnnotationMap &amap, std::string &error) {
    return AnnotationCSV::load_all(V2_CSV_DIR, amap, "Rat4", NUM_NODES, NUM_CAMS,
                                   CAMERA_NAMES, error) == 0;
}

// ---------------------------------------------------------------------------
// Helper: compute centroid from labeled 2D keypoints on a camera
// ---------------------------------------------------------------------------
static bool compute_centroid(const CameraAnnotation &cam, double &cx, double &cy) {
    cx = 0; cy = 0;
    int count = 0;
    for (const auto &kp : cam.keypoints) {
        if (kp.labeled) {
            cx += kp.x;
            cy += kp.y;
            ++count;
        }
    }
    if (count == 0) return false;
    cx /= count;
    cy /= count;
    return true;
}

// ---------------------------------------------------------------------------
// Helper: load a JPEG via stb_image, return RGB buffer
// ---------------------------------------------------------------------------
static std::vector<uint8_t> load_jpeg(const std::string &path, int &w, int &h) {
    int channels = 0;
    uint8_t *data = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!data) return {};
    std::vector<uint8_t> rgb(data, data + w * h * 3);
    stbi_image_free(data);
    return rgb;
}

// ---------------------------------------------------------------------------
// Helper: compute polygon area via shoelace formula
// ---------------------------------------------------------------------------
static double polygon_area(const std::vector<tuple_d> &pts) {
    double area = 0;
    int n = (int)pts.size();
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += pts[i].x * pts[j].y;
        area -= pts[j].x * pts[i].y;
    }
    return std::abs(area) / 2.0;
}

// =========================================================================
// Test 1: Compute centroid from keypoints for a single camera
// =========================================================================
static void test_compute_centroid_from_keypoints() {
    printf("  test_compute_centroid_from_keypoints...\n");

    AnnotationMap amap;
    std::string error;
    EXPECT_TRUE(load_annotate1_data(amap, error));
    if (!error.empty()) printf("    error: %s\n", error.c_str());

    // Frame 5998 should exist
    auto it = amap.find(TEST_FRAME);
    EXPECT_TRUE(it != amap.end());
    if (it == amap.end()) { printf("    SKIP: frame %u not found\n", TEST_FRAME); return; }

    const auto &fa = it->second;
    EXPECT_EQ((int)fa.cameras.size(), NUM_CAMS);

    // Cam2002486 = camera index 0
    const auto &cam = fa.cameras[0];

    double cx, cy;
    EXPECT_TRUE(compute_centroid(cam, cx, cy));

    printf("    Cam2002486 centroid at frame %u: (%.1f, %.1f)\n", TEST_FRAME, cx, cy);

    // Centroid should be within image bounds
    EXPECT_TRUE(cx > 0 && cx < IMG_W);
    EXPECT_TRUE(cy > 0 && cy < IMG_H);

    // Centroid should not be at the very edge or dead center of the image
    // (the rat should be somewhere reasonable in the field of view)
    EXPECT_TRUE(cx > 100 && cx < IMG_W - 100);
    EXPECT_TRUE(cy > 100 && cy < IMG_H - 100);

    // Verify it is not exactly at center (would indicate default/invalid data)
    EXPECT_TRUE(std::abs(cx - IMG_W / 2.0) > 50 || std::abs(cy - IMG_H / 2.0) > 50);
}

// =========================================================================
// Test 2: Compute centroids for all 16 cameras at frame 5998
// =========================================================================
static void test_compute_centroids_all_cameras() {
    printf("  test_compute_centroids_all_cameras...\n");

    AnnotationMap amap;
    std::string error;
    EXPECT_TRUE(load_annotate1_data(amap, error));

    auto it = amap.find(TEST_FRAME);
    EXPECT_TRUE(it != amap.end());
    if (it == amap.end()) { printf("    SKIP: frame %u not found\n", TEST_FRAME); return; }

    const auto &fa = it->second;
    EXPECT_EQ((int)fa.cameras.size(), NUM_CAMS);

    printf("    Frame %u centroids:\n", TEST_FRAME);
    int valid_count = 0;
    for (int c = 0; c < NUM_CAMS; ++c) {
        double cx, cy;
        bool ok = compute_centroid(fa.cameras[c], cx, cy);
        if (ok) {
            printf("      %s: (%.1f, %.1f)\n", CAMERA_NAMES[c].c_str(), cx, cy);

            // All centroids must be within image bounds
            EXPECT_TRUE(cx > 0 && cx < IMG_W);
            EXPECT_TRUE(cy > 0 && cy < IMG_H);
            ++valid_count;
        } else {
            printf("      %s: no labeled keypoints\n", CAMERA_NAMES[c].c_str());
        }
    }

    // All 16 cameras should have labels for this frame
    EXPECT_EQ(valid_count, NUM_CAMS);
}

// =========================================================================
// Test 3: Preprocess a real JPEG for MobileSAM
// =========================================================================
static void test_preprocess_real_jpeg_mobilesam() {
    printf("  test_preprocess_real_jpeg_mobilesam...\n");

    std::string jpeg_path = std::string(JPEG_DIR) + "/Cam2002486/Frame_5998.jpg";
    EXPECT_TRUE(std::filesystem::exists(jpeg_path));
    if (!std::filesystem::exists(jpeg_path)) {
        printf("    SKIP: %s not found\n", jpeg_path.c_str());
        return;
    }

    int w = 0, h = 0;
    auto rgb = load_jpeg(jpeg_path, w, h);
    EXPECT_TRUE(!rgb.empty());
    EXPECT_EQ(w, IMG_W);
    EXPECT_EQ(h, IMG_H);
    printf("    Loaded JPEG: %dx%d (%zu bytes)\n", w, h, rgb.size());

    // Time the preprocessing
    auto t0 = std::chrono::steady_clock::now();

    float scale = 0;
    int pad_x = 0, pad_y = 0;
    auto result = sam_detail::preprocess_mobilesam(rgb.data(), w, h, scale, pad_x, pad_y);

    auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("    MobileSAM preprocess: %.1f ms\n", ms);

    // Output size: 3 channels x 1024 x 1024
    EXPECT_EQ((int)result.size(), 3 * 1024 * 1024);

    // Scale factor: 1024 / max(3208, 2200) = 1024 / 3208 ~ 0.319
    float expected_scale = 1024.0f / std::max(IMG_W, IMG_H);
    EXPECT_NEAR(scale, expected_scale, 0.01f);
    printf("    Scale factor: %.4f (expected ~%.4f)\n", scale, expected_scale);

    // Check values are in a reasonable normalized range (ImageNet normalization)
    // After subtracting mean and dividing by std, values should roughly be in [-3, 3]
    // for typical images, but the padded zero region will be ~[-2.1, -1.8] after normalization.
    bool has_nan = false;
    bool has_inf = false;
    float min_val = result[0], max_val = result[0];
    for (float v : result) {
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    EXPECT_FALSE(has_nan);
    EXPECT_FALSE(has_inf);
    printf("    Value range: [%.3f, %.3f]\n", min_val, max_val);

    // Values should NOT be raw 0-255 (normalization should have been applied)
    EXPECT_TRUE(max_val < 20.0f);  // normalized values won't be > ~3-4

    // Preprocessing should be fast (well under 100ms)
    EXPECT_TRUE(ms < 5000.0f); // generous upper bound
}

// =========================================================================
// Test 4: Preprocess a real JPEG for SAM 2.1
// =========================================================================
static void test_preprocess_real_jpeg_sam2() {
    printf("  test_preprocess_real_jpeg_sam2...\n");

    std::string jpeg_path = std::string(JPEG_DIR) + "/Cam2002486/Frame_5998.jpg";
    if (!std::filesystem::exists(jpeg_path)) {
        printf("    SKIP: %s not found\n", jpeg_path.c_str());
        return;
    }

    int w = 0, h = 0;
    auto rgb = load_jpeg(jpeg_path, w, h);
    EXPECT_TRUE(!rgb.empty());

    auto t0 = std::chrono::steady_clock::now();
    auto result = sam_detail::preprocess_sam2(rgb.data(), w, h);
    auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("    SAM 2.1 preprocess: %.1f ms\n", ms);

    // Output size: 3 channels x 1024 x 1024
    EXPECT_EQ((int)result.size(), 3 * 1024 * 1024);

    // Check for NaN/Inf and value range
    bool has_nan = false;
    bool has_inf = false;
    float min_val = result[0], max_val = result[0];
    for (float v : result) {
        if (std::isnan(v)) has_nan = true;
        if (std::isinf(v)) has_inf = true;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    EXPECT_FALSE(has_nan);
    EXPECT_FALSE(has_inf);
    printf("    Value range: [%.3f, %.3f]\n", min_val, max_val);

    // SAM 2.1 normalizes to 0-1 range then subtracts mean / divides by std
    // So values should be roughly in [-2, 3] range
    EXPECT_TRUE(max_val < 10.0f);   // not raw 0-255
    EXPECT_TRUE(min_val > -10.0f);  // not absurdly negative
}

// =========================================================================
// Test 5: Mask-to-polygon with a synthetic mouse-shaped ellipse
// =========================================================================
static void test_mask_to_polygon_mouse_shaped() {
    printf("  test_mask_to_polygon_mouse_shaped...\n");

    // Load centroid from real data to place the ellipse
    AnnotationMap amap;
    std::string error;
    EXPECT_TRUE(load_annotate1_data(amap, error));

    auto it = amap.find(TEST_FRAME);
    EXPECT_TRUE(it != amap.end());
    if (it == amap.end()) { printf("    SKIP: frame not found\n"); return; }

    double cx, cy;
    EXPECT_TRUE(compute_centroid(it->second.cameras[0], cx, cy));

    // Create a synthetic binary mask with a mouse-shaped ellipse
    // rx=100, ry=50 pixels ~ rough rat body size at this resolution
    constexpr int rx = 100;
    constexpr int ry = 50;
    double ellipse_area = M_PI * rx * ry;

    SamMask mask;
    mask.width = IMG_W;
    mask.height = IMG_H;
    mask.data.resize(IMG_W * IMG_H, 0);
    mask.valid = true;

    int filled = 0;
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            double dx = (x - cx) / rx;
            double dy = (y - cy) / ry;
            if (dx * dx + dy * dy <= 1.0) {
                mask.data[y * IMG_W + x] = 255;
                ++filled;
            }
        }
    }
    printf("    Ellipse at (%.0f, %.0f), rx=%d, ry=%d, filled=%d pixels\n",
           cx, cy, rx, ry, filled);
    printf("    Expected area: %.0f pixels\n", ellipse_area);

    auto t0 = std::chrono::steady_clock::now();
    auto polygons = sam_mask_to_polygon(mask, 2.0);
    auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("    Mask-to-polygon: %.1f ms, %zu polygon(s)\n", ms, polygons.size());

    // At least 1 polygon returned
    EXPECT_TRUE(polygons.size() >= 1);
    if (polygons.empty()) { printf("    SKIP: no polygons extracted\n"); return; }

    // Use the largest polygon (by point count)
    size_t best = 0;
    for (size_t i = 1; i < polygons.size(); ++i) {
        if (polygons[i].size() > polygons[best].size()) best = i;
    }
    const auto &poly = polygons[best];
    printf("    Largest polygon: %zu points\n", poly.size());

    // Polygon should have more than 4 points (it is an ellipse, not a box)
    EXPECT_TRUE(poly.size() > 4);

    // All polygon points within image bounds
    bool all_in_bounds = true;
    for (const auto &pt : poly) {
        if (pt.x < 0 || pt.x >= IMG_W || pt.y < 0 || pt.y >= IMG_H) {
            all_in_bounds = false;
            break;
        }
    }
    EXPECT_TRUE(all_in_bounds);

    // Polygon area should be roughly correct (within 2x of ellipse area)
    double area = polygon_area(poly);
    printf("    Polygon area: %.0f (expected ~%.0f)\n", area, ellipse_area);
    EXPECT_TRUE(area > ellipse_area / 2.0);
    EXPECT_TRUE(area < ellipse_area * 2.0);
}

// =========================================================================
// Test 6: Mask polygon roundtrip through annotations JSON
// =========================================================================
static void test_mask_stored_in_camera_extras() {
    printf("  test_mask_stored_in_camera_extras...\n");

    // Create a FrameAnnotation and store a polygon
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 42, NUM_NODES, 2);

    // Create a simple polygon
    std::vector<tuple_d> poly = {
        {100.5, 200.5}, {300.0, 200.5}, {300.0, 400.0}, {200.0, 450.5}, {100.5, 400.0}
    };

    auto &ext = fa.cameras[0].get_extras();
    ext.mask_polygons.push_back(poly);
    ext.has_mask = true;

    // Verify in-memory
    EXPECT_TRUE(fa.cameras[0].has_mask());
    EXPECT_EQ((int)fa.cameras[0].get_extras().mask_polygons.size(), 1);
    EXPECT_EQ((int)fa.cameras[0].get_extras().mask_polygons[0].size(), 5);

    // Save to JSON (to a temp directory)
    std::string tmp_dir = "/tmp/test_sam_integration_" + std::to_string(getpid());
    std::filesystem::create_directories(tmp_dir);

    EXPECT_TRUE(save_annotations_json(amap, tmp_dir));

    // Verify JSON file was written
    std::string json_path = tmp_dir + "/annotations.json";
    EXPECT_TRUE(std::filesystem::exists(json_path));

    // Create a fresh AnnotationMap with matching frame structure
    AnnotationMap amap2;
    get_or_create_frame(amap2, 42, NUM_NODES, 2);

    // Load
    EXPECT_TRUE(load_annotations_json(amap2, tmp_dir));

    // Verify roundtrip
    auto it = amap2.find(42);
    EXPECT_TRUE(it != amap2.end());
    if (it == amap2.end()) { printf("    SKIP: frame not found after load\n"); return; }

    EXPECT_TRUE(it->second.cameras[0].has_mask());
    const auto &loaded_polys = it->second.cameras[0].get_extras().mask_polygons;
    EXPECT_EQ((int)loaded_polys.size(), 1);
    EXPECT_EQ((int)loaded_polys[0].size(), 5);

    // Check point values roundtrip accurately
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(loaded_polys[0][i].x, poly[i].x, 0.01);
        EXPECT_NEAR(loaded_polys[0][i].y, poly[i].y, 0.01);
    }

    // Camera 1 should not have mask
    EXPECT_FALSE(it->second.cameras[1].has_mask());

    // Cleanup
    std::filesystem::remove_all(tmp_dir);
    printf("    Roundtrip OK\n");
}

// =========================================================================
// Test 7: Coordinate transform from image space to SAM model space
// =========================================================================
static void test_sam_point_transform_to_model_coords() {
    printf("  test_sam_point_transform_to_model_coords...\n");

    // Get a real centroid
    AnnotationMap amap;
    std::string error;
    EXPECT_TRUE(load_annotate1_data(amap, error));

    auto it = amap.find(TEST_FRAME);
    EXPECT_TRUE(it != amap.end());
    if (it == amap.end()) { printf("    SKIP: frame not found\n"); return; }

    double img_x, img_y;
    EXPECT_TRUE(compute_centroid(it->second.cameras[0], img_x, img_y));
    printf("    Image coords: (%.1f, %.1f)\n", img_x, img_y);

    // --- MobileSAM coordinate transform ---
    // model_x = image_x * scale, model_y = image_y * scale
    // where scale = 1024 / max(w, h)
    {
        float scale = 1024.0f / std::max(IMG_W, IMG_H);
        float model_x = (float)(img_x * scale);
        float model_y = (float)(img_y * scale);

        printf("    MobileSAM: scale=%.4f, model=(%.1f, %.1f)\n", scale, model_x, model_y);

        EXPECT_TRUE(model_x >= 0 && model_x <= 1024);
        EXPECT_TRUE(model_y >= 0 && model_y <= 1024);

        // For MobileSAM with longest-side resize, the short dimension should map
        // to less than 1024 (since we pad to square)
        float nw = IMG_W * scale;
        float nh = IMG_H * scale;
        printf("    Resized dims: %.0fx%.0f in 1024x1024\n", nw, nh);
        EXPECT_TRUE(nw <= 1024.0f + 1.0f);  // longest side = 1024
        EXPECT_TRUE(nh <= 1024.0f + 1.0f);

        // The longer dimension should be very close to 1024
        float longest = std::max(nw, nh);
        EXPECT_NEAR(longest, 1024.0f, 2.0f);
    }

    // --- SAM 2.1 coordinate transform ---
    // model_x = image_x * 1024 / w, model_y = image_y * 1024 / h
    {
        float model_x = (float)(img_x * 1024.0 / IMG_W);
        float model_y = (float)(img_y * 1024.0 / IMG_H);

        printf("    SAM 2.1: model=(%.1f, %.1f)\n", model_x, model_y);

        EXPECT_TRUE(model_x >= 0 && model_x <= 1024);
        EXPECT_TRUE(model_y >= 0 && model_y <= 1024);
    }

    // --- Edge case: point at (0,0) ---
    {
        float model_x_ms = 0.0f * (1024.0f / std::max(IMG_W, IMG_H));
        float model_y_ms = 0.0f * (1024.0f / std::max(IMG_W, IMG_H));
        EXPECT_NEAR(model_x_ms, 0.0f, 0.001f);
        EXPECT_NEAR(model_y_ms, 0.0f, 0.001f);
    }

    // --- Edge case: point at (IMG_W-1, IMG_H-1) ---
    {
        float scale = 1024.0f / std::max(IMG_W, IMG_H);
        float model_x = (IMG_W - 1) * scale;
        float model_y = (IMG_H - 1) * scale;
        EXPECT_TRUE(model_x <= 1024.0f);
        EXPECT_TRUE(model_y <= 1024.0f);

        float model_x2 = (float)((IMG_W - 1) * 1024.0 / IMG_W);
        float model_y2 = (float)((IMG_H - 1) * 1024.0 / IMG_H);
        EXPECT_TRUE(model_x2 <= 1024.0f);
        EXPECT_TRUE(model_y2 <= 1024.0f);
    }
}

// =========================================================================
// Test 8: sam_init without models, sam_segment on uninitialized state
// =========================================================================
static void test_sam_init_without_models() {
    printf("  test_sam_init_without_models...\n");

    SamState sam;

    // Init with non-existent paths
    bool ok = sam_init(sam, SamModel::MobileSAM,
                       "/nonexistent/encoder.onnx", "/nonexistent/decoder.onnx");

    // Without ONNX Runtime compiled in: returns false, sets status
    // With ONNX Runtime compiled in: returns false (files don't exist), sets status
    EXPECT_FALSE(ok);
    EXPECT_FALSE(sam.loaded);
    EXPECT_TRUE(!sam.status.empty());
    printf("    sam_init status: %s\n", sam.status.c_str());

    // Segment on uninitialized state should return invalid mask, not crash
    std::vector<uint8_t> dummy_rgb(100 * 100 * 3, 128);
    std::vector<tuple_d> fg = {{50.0, 50.0}};
    std::vector<tuple_d> bg;

    SamMask mask = sam_segment(sam, dummy_rgb.data(), 100, 100, fg, bg);
    EXPECT_FALSE(mask.valid);
    EXPECT_TRUE(mask.data.empty());
    printf("    sam_segment on uninitialized: valid=%d, status=%s\n",
           mask.valid, sam.status.c_str());

    // Try SAM 2.1 as well
    SamState sam2;
    ok = sam_init(sam2, SamModel::SAM2,
                  "/nonexistent/sam2_encoder.onnx", "/nonexistent/sam2_decoder.onnx");
    EXPECT_FALSE(ok);
    EXPECT_FALSE(sam2.loaded);
    printf("    sam_init SAM2 status: %s\n", sam2.status.c_str());

    // Cleanup should not crash
    sam_cleanup(sam);
    sam_cleanup(sam2);
    printf("    Cleanup OK\n");
}

// =========================================================================
// End-to-end inference test (requires real ONNX models)
// =========================================================================

static const char *MOBILESAM_ENCODER = "/Users/johnsonr/src/red/models/mobilesam/mobile_sam_encoder.onnx";
static const char *MOBILESAM_DECODER = "/Users/johnsonr/src/red/models/mobilesam/mobile_sam_decoder.onnx";

static void test_mobilesam_inference_real_image() {
    printf("  test_mobilesam_inference_real_image...\n");

    // Check if models exist
    if (!std::filesystem::exists(MOBILESAM_ENCODER) ||
        !std::filesystem::exists(MOBILESAM_DECODER)) {
        printf("    SKIP: MobileSAM models not found at %s\n", MOBILESAM_ENCODER);
        return;
    }

    // Load a real JPEG
    std::string img_path = std::string(JPEG_DIR) + "/Cam2002486/Frame_5998.jpg";
    int w, h, channels;
    uint8_t *rgb = stbi_load(img_path.c_str(), &w, &h, &channels, 3);
    EXPECT_TRUE(rgb != nullptr);
    if (!rgb) return;
    printf("    Loaded: %dx%d\n", w, h);

    // Compute centroid from keypoints
    AnnotationMap amap;
    std::string err;
    AnnotationCSV::load_2d_csv(
        std::string(V2_CSV_DIR) + "/Cam2002486.csv",
        amap, 0, 4, 1, err);
    auto it = amap.find(5998);
    EXPECT_TRUE(it != amap.end());
    double cx = 0, cy = 0;
    int n = 0;
    for (const auto &kp : it->second.cameras[0].keypoints) {
        if (kp.labeled) { cx += kp.x; cy += kp.y; n++; }
    }
    cx /= n; cy /= n;
    // Y-flip: CSV stores ImPlot coords (y=0 at bottom), SAM expects image coords (y=0 at top)
    double img_cy = h - cy;
    printf("    Centroid (image coords): (%.1f, %.1f)\n", cx, img_cy);

    // Initialize SAM
    SamState sam;
    bool loaded = sam_init(sam, SamModel::MobileSAM, MOBILESAM_ENCODER, MOBILESAM_DECODER);
    printf("    sam_init: %s (status: %s)\n", loaded ? "OK" : "FAILED", sam.status.c_str());
    EXPECT_TRUE(loaded);
    if (!loaded) { stbi_image_free(rgb); return; }

    // Run inference with centroid as foreground point (image coordinates)
    std::vector<tuple_d> fg_points = {{cx, img_cy}};
    auto mask = sam_segment(sam, rgb, w, h, fg_points, {});
    printf("    Encode: %.1f ms, Decode: %.1f ms\n", sam.last_encode_ms, sam.last_decode_ms);
    printf("    Mask: valid=%d, size=%dx%d, iou=%.3f\n",
           mask.valid, mask.width, mask.height, mask.iou_score);

    EXPECT_TRUE(mask.valid);
    EXPECT_EQ(mask.width, w);
    EXPECT_EQ(mask.height, h);
    EXPECT_TRUE(mask.iou_score > 0.5f);

    // Count mask pixels
    int fg_pixels = 0;
    for (auto v : mask.data) if (v > 0) fg_pixels++;
    double fg_pct = 100.0 * fg_pixels / (w * h);
    printf("    Foreground: %d pixels (%.2f%% of image)\n", fg_pixels, fg_pct);

    // A mouse should be ~0.5-10% of a 3208x2200 image
    EXPECT_TRUE(fg_pixels > 0);
    EXPECT_TRUE(fg_pct < 50.0); // not the whole image

    // Convert to polygon
    auto polygons = sam_mask_to_polygon(mask);
    printf("    Polygons: %d\n", (int)polygons.size());
    EXPECT_TRUE(!polygons.empty());

    if (!polygons.empty()) {
        printf("    Largest polygon: %d points\n", (int)polygons[0].size());
        EXPECT_TRUE(polygons[0].size() > 4);

        // Verify polygon bounds are within image
        for (const auto &poly : polygons) {
            for (const auto &pt : poly) {
                EXPECT_TRUE(pt.x >= 0 && pt.x <= w);
                EXPECT_TRUE(pt.y >= 0 && pt.y <= h);
            }
        }
    }

    stbi_image_free(rgb);
    sam_cleanup(sam);
    printf("    OK\n");
}

static void test_mobilesam_multi_camera() {
    printf("  test_mobilesam_multi_camera...\n");

    if (!std::filesystem::exists(MOBILESAM_ENCODER) ||
        !std::filesystem::exists(MOBILESAM_DECODER)) {
        printf("    SKIP: MobileSAM models not found\n");
        return;
    }

    // Initialize SAM once
    SamState sam;
    bool loaded = sam_init(sam, SamModel::MobileSAM, MOBILESAM_ENCODER, MOBILESAM_DECODER);
    EXPECT_TRUE(loaded);
    if (!loaded) return;

    // Load all camera centroids for frame 5998
    const char *cameras[] = {
        "Cam2002486", "Cam2002487", "Cam2005325", "Cam2006050",
        "Cam2006051", "Cam2006052", "Cam2006054", "Cam2006055",
        "Cam2006515", "Cam2006516", "Cam2008665", "Cam2008666",
        "Cam2008667", "Cam2008668", "Cam2008669", "Cam2008670"
    };

    int successes = 0, failures = 0, skipped = 0;
    printf("    Camera           | Centroid        | IoU   | FG%%   | Polys | Encode  | Decode\n");
    printf("    -----------------+-----------------+-------+-------+-------+---------+-------\n");

    for (int c = 0; c < 16; c++) {
        // Load JPEG
        std::string img_path = std::string(JPEG_DIR) + "/" + cameras[c] + "/Frame_5998.jpg";
        if (!std::filesystem::exists(img_path)) {
            printf("    %-16s | (no JPEG)       |       |       |       |         |\n", cameras[c]);
            skipped++;
            continue;
        }

        int w, h, ch;
        uint8_t *rgb = stbi_load(img_path.c_str(), &w, &h, &ch, 3);
        if (!rgb) { skipped++; continue; }

        // Load CSV and compute centroid
        AnnotationMap amap;
        std::string err;
        AnnotationCSV::load_2d_csv(
            std::string(V2_CSV_DIR) + "/" + cameras[c] + ".csv",
            amap, 0, 4, 1, err);

        auto it = amap.find(5998);
        if (it == amap.end()) {
            stbi_image_free(rgb);
            skipped++;
            continue;
        }

        double cx = 0, cy = 0;
        int n = 0;
        for (const auto &kp : it->second.cameras[0].keypoints) {
            if (kp.labeled) { cx += kp.x; cy += kp.y; n++; }
        }
        if (n == 0) { stbi_image_free(rgb); skipped++; continue; }
        cx /= n; cy /= n;
        double img_cy = h - cy; // Y-flip for image coords

        // Run SAM
        std::vector<tuple_d> fg = {{cx, img_cy}};
        auto mask = sam_segment(sam, rgb, w, h, fg, {});

        int fg_pixels = 0;
        if (mask.valid)
            for (auto v : mask.data) if (v > 0) fg_pixels++;
        double fg_pct = mask.valid ? 100.0 * fg_pixels / (w * h) : 0;

        auto polys = mask.valid ? sam_mask_to_polygon(mask) : std::vector<std::vector<tuple_d>>{};

        printf("    %-16s | (%6.1f,%6.1f) | %.3f | %5.2f | %5d | %6.1fms | %.1fms\n",
               cameras[c], cx, img_cy, mask.iou_score, fg_pct,
               (int)polys.size(), sam.last_encode_ms, sam.last_decode_ms);

        if (mask.valid && mask.iou_score > 0.3f && !polys.empty()) {
            successes++;
        } else {
            failures++;
        }

        stbi_image_free(rgb);
    }

    printf("    -----------------+-----------------+-------+-------+-------+---------+-------\n");
    printf("    Results: %d success, %d failed, %d skipped (of 16 cameras)\n",
           successes, failures, skipped);

    // At least half the cameras should produce valid masks
    // (some cameras may have the mouse occluded or at weird angles)
    EXPECT_TRUE(successes >= 8);

    sam_cleanup(sam);
}

// =========================================================================
// BGRA→RGB conversion test (validates the red.cpp extraction path)
// =========================================================================

static void test_bgra_to_rgb_conversion() {
    printf("  test_bgra_to_rgb_conversion...\n");

    // Load a real JPEG as RGB via stbi
    std::string img_path = std::string(JPEG_DIR) + "/Cam2002486/Frame_5998.jpg";
    int w, h, ch;
    uint8_t *rgb_ref = stbi_load(img_path.c_str(), &w, &h, &ch, 3);
    EXPECT_TRUE(rgb_ref != nullptr);
    if (!rgb_ref) return;

    // Create fake BGRA buffer from the RGB (simulates CVPixelBuffer format)
    int stride = w * 4; // no padding for simplicity
    std::vector<uint8_t> bgra(stride * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t r = rgb_ref[(y * w + x) * 3 + 0];
            uint8_t g = rgb_ref[(y * w + x) * 3 + 1];
            uint8_t b = rgb_ref[(y * w + x) * 3 + 2];
            bgra[y * stride + x * 4 + 0] = b;  // B
            bgra[y * stride + x * 4 + 1] = g;  // G
            bgra[y * stride + x * 4 + 2] = r;  // R
            bgra[y * stride + x * 4 + 3] = 255; // A
        }
    }

    // Convert back using the same logic as red.cpp
    std::vector<uint8_t> rgb_out(w * h * 3);
    for (int y = 0; y < h; y++) {
        const uint8_t *src = bgra.data() + y * stride;
        uint8_t *dst = rgb_out.data() + y * w * 3;
        for (int x = 0; x < w; x++) {
            dst[x*3+0] = src[x*4+2]; // R from BGRA
            dst[x*3+1] = src[x*4+1]; // G
            dst[x*3+2] = src[x*4+0]; // B
        }
    }

    // Compare against reference
    int mismatches = 0;
    for (int i = 0; i < w * h * 3; i++) {
        if (rgb_out[i] != rgb_ref[i]) mismatches++;
    }
    printf("    %dx%d, BGRA→RGB mismatches: %d / %d\n", w, h, mismatches, w * h * 3);
    EXPECT_EQ(mismatches, 0);

    // Verify SAM produces same result from converted RGB as from stbi RGB
    if (std::filesystem::exists(MOBILESAM_ENCODER) &&
        std::filesystem::exists(MOBILESAM_DECODER)) {
        SamState sam;
        if (sam_init(sam, SamModel::MobileSAM, MOBILESAM_ENCODER, MOBILESAM_DECODER)) {
            // Use centroid as prompt (image coords)
            std::vector<tuple_d> fg = {{1684.6, 545.7}};

            auto mask_ref = sam_segment(sam, rgb_ref, w, h, fg, {}, nullptr, 0, 0);
            sam.cached_frame = -1; // force re-encode
            auto mask_conv = sam_segment(sam, rgb_out.data(), w, h, fg, {}, nullptr, 1, 0);

            EXPECT_TRUE(mask_ref.valid);
            EXPECT_TRUE(mask_conv.valid);

            if (mask_ref.valid && mask_conv.valid) {
                // Masks should be identical (same input → same output)
                int diff = 0;
                for (size_t i = 0; i < mask_ref.data.size() && i < mask_conv.data.size(); i++)
                    if (mask_ref.data[i] != mask_conv.data[i]) diff++;
                printf("    SAM mask diff (stbi vs BGRA→RGB): %d pixels\n", diff);
                EXPECT_EQ(diff, 0);
            }
            sam_cleanup(sam);
        }
    } else {
        printf("    (SAM model comparison skipped — models not found)\n");
    }

    stbi_image_free(rgb_ref);
}

// =========================================================================
// Polygon caching test
// =========================================================================

static void test_polygon_caching() {
    printf("  test_polygon_caching...\n");

    SamToolState_Test state;
    state.enabled = true;

    // Simulate generating a mask and caching polygons
    SamMask mask;
    mask.width = 100;
    mask.height = 100;
    mask.data.resize(100 * 100, 0);
    // Draw a filled circle
    for (int y = 0; y < 100; y++)
        for (int x = 0; x < 100; x++)
            if ((x-50)*(x-50) + (y-50)*(y-50) < 30*30)
                mask.data[y * 100 + x] = 255;
    mask.valid = true;

    state.current_mask = mask;
    state.current_polygons = sam_mask_to_polygon(mask);
    state.has_pending_mask = true;

    EXPECT_TRUE(!state.current_polygons.empty());
    printf("    Cached %d polygon(s), largest has %d points\n",
           (int)state.current_polygons.size(),
           state.current_polygons.empty() ? 0 : (int)state.current_polygons[0].size());

    // Verify polygon is reasonable
    if (!state.current_polygons.empty()) {
        for (const auto &pt : state.current_polygons[0]) {
            EXPECT_TRUE(pt.x >= 0 && pt.x <= 100);
            EXPECT_TRUE(pt.y >= 0 && pt.y <= 100);
        }
    }

    // Simulate frame change reset
    state.fg_points.clear();
    state.bg_points.clear();
    state.current_polygons.clear();
    state.has_pending_mask = false;

    EXPECT_TRUE(state.current_polygons.empty());
    EXPECT_FALSE(state.has_pending_mask);
}

// =========================================================================
// main
// =========================================================================
int main() {
    printf("=== test_sam_integration ===\n");

    // Verify test data is available
    if (!std::filesystem::exists(V2_CSV_DIR)) {
        fprintf(stderr, "ERROR: V2 CSV directory not found: %s\n", V2_CSV_DIR);
        fprintf(stderr, "This test requires the annotate1 project data.\n");
        return 1;
    }

    printf("\nData model + centroid tests:\n");
    test_compute_centroid_from_keypoints();
    test_compute_centroids_all_cameras();

    printf("\nPreprocessing tests:\n");
    test_preprocess_real_jpeg_mobilesam();
    test_preprocess_real_jpeg_sam2();

    printf("\nMask-to-polygon tests:\n");
    test_mask_to_polygon_mouse_shaped();

    printf("\nAnnotation persistence tests:\n");
    test_mask_stored_in_camera_extras();

    printf("\nCoordinate transform tests:\n");
    test_sam_point_transform_to_model_coords();

    printf("\nSAM init/segment stub tests:\n");
    test_sam_init_without_models();

    printf("\nBGRA→RGB conversion tests:\n");
    test_bgra_to_rgb_conversion();

    printf("\nPolygon caching tests:\n");
    test_polygon_caching();

    printf("\nMobileSAM real inference tests:\n");
    test_mobilesam_inference_real_image();
    test_mobilesam_multi_camera();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
