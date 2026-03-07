// tests/test_gui.cpp
// Unit tests for pure math functions in gui.h.
// Compiled as the test_gui target (same deps as red, replacing red.cpp).

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "camera.h"
#include "global.h"
#include "gui.h"
#include "gui/calibration_tool_window.h"
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT_TRUE(expr)                                                      \
    do {                                                                       \
        if (expr) {                                                            \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: expected true: %s\n", __FILE__,    \
                    __LINE__, #expr);                                          \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_NEAR(a, b, eps)                                                 \
    do {                                                                       \
        float _a = (float)(a), _b = (float)(b), _e = (float)(eps);           \
        float _diff = fabsf(_a - _b);                                         \
        if (_diff <= _e) {                                                     \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: |%s - %s| = %g > %g\n",           \
                    __FILE__, __LINE__, #a, #b, (double)_diff, (double)_e);   \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// is_point_near
// ---------------------------------------------------------------------------

static void test_is_point_near() {
    // Same point: always near
    EXPECT_TRUE(is_point_near({0, 0}, {0, 0}));

    // Within default threshold (15): dist=10, dist_sq=100 < 225
    EXPECT_TRUE(is_point_near({0, 0}, {10, 0}));

    // Exactly at threshold: dist=15, dist_sq=225, condition is strict <, so false
    EXPECT_FALSE(is_point_near({0, 0}, {15, 0}));

    // Beyond threshold
    EXPECT_FALSE(is_point_near({0, 0}, {20, 0}));

    // Custom threshold: 3-4-5 triangle, dist=5
    EXPECT_TRUE(is_point_near({0, 0}, {3, 4}, 6.0f));   // 25 < 36
    EXPECT_FALSE(is_point_near({0, 0}, {3, 4}, 4.0f));  // 25 > 16
}

// ---------------------------------------------------------------------------
// is_point_near_line_segment
// ---------------------------------------------------------------------------

static void test_is_point_near_line_segment() {
    // Point on the midpoint of the segment
    EXPECT_TRUE(is_point_near_line_segment({5, 0}, {0, 0}, {10, 0}));

    // Point at the start endpoint
    EXPECT_TRUE(is_point_near_line_segment({0, 0}, {0, 0}, {10, 0}));

    // Point at the end endpoint
    EXPECT_TRUE(is_point_near_line_segment({10, 0}, {0, 0}, {10, 0}));

    // Perpendicular from midpoint, within default threshold (5): dist=4
    EXPECT_TRUE(is_point_near_line_segment({5, 4}, {0, 0}, {10, 0}));  // 16 <= 25

    // Perpendicular from midpoint, beyond threshold: dist=6
    EXPECT_FALSE(is_point_near_line_segment({5, 6}, {0, 0}, {10, 0})); // 36 > 25

    // Beyond segment end — clamped to endpoint (10,0), dist > 5
    EXPECT_FALSE(is_point_near_line_segment({20, 10}, {0, 0}, {10, 0}));

    // Degenerate segment (zero length): always false
    EXPECT_FALSE(is_point_near_line_segment({3, 3}, {3, 3}, {3, 3}));
}

// ---------------------------------------------------------------------------
// get_obb_corners — axis-aligned box
// ---------------------------------------------------------------------------

static void test_get_obb_corners_axis_aligned() {
    OrientedBoundingBox obb = {};
    obb.center   = {0, 0};
    obb.width    = 20.0f;
    obb.height   = 10.0f;
    obb.rotation = 0.0f;
    obb.state    = OBBComplete;

    ImVec2 corners[4];
    get_obb_corners(&obb, corners);

    // half_w=10, half_h=5, rotation=0 → corners are (±10, ±5)
    EXPECT_NEAR(corners[0].x, -10.0f, 1e-4f);
    EXPECT_NEAR(corners[0].y,  -5.0f, 1e-4f);
    EXPECT_NEAR(corners[1].x,  10.0f, 1e-4f);
    EXPECT_NEAR(corners[1].y,  -5.0f, 1e-4f);
    EXPECT_NEAR(corners[2].x,  10.0f, 1e-4f);
    EXPECT_NEAR(corners[2].y,   5.0f, 1e-4f);
    EXPECT_NEAR(corners[3].x, -10.0f, 1e-4f);
    EXPECT_NEAR(corners[3].y,   5.0f, 1e-4f);
}

static void test_get_obb_corners_incomplete_state() {
    OrientedBoundingBox obb = {};
    obb.state = OBBFirstAxisPoint;
    ImVec2 corners[4];
    get_obb_corners(&obb, corners);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(corners[i].x, 0.0f, 1e-6f);
        EXPECT_NEAR(corners[i].y, 0.0f, 1e-6f);
    }
}

// ---------------------------------------------------------------------------
// get_obb_corners — 90° rotated box
//
// center=(5,5), width=10, height=4, rotation=π/2
// cos(π/2)≈0, sin(π/2)=1
// local BL(-5,-2) → rotated (2,-5) → world (7,0)
// local BR( 5,-2) → rotated (2, 5) → world (7,10)
// local TR( 5, 2) → rotated(-2, 5) → world (3,10)
// local TL(-5, 2) → rotated(-2,-5) → world (3, 0)
// ---------------------------------------------------------------------------

static void test_get_obb_corners_rotated() {
    const float PI_2 = (float)(M_PI / 2.0);
    OrientedBoundingBox obb = {};
    obb.center   = {5, 5};
    obb.width    = 10.0f;
    obb.height   =  4.0f;
    obb.rotation = PI_2;
    obb.state    = OBBComplete;

    ImVec2 corners[4];
    get_obb_corners(&obb, corners);

    EXPECT_NEAR(corners[0].x,  7.0f, 1e-3f);
    EXPECT_NEAR(corners[0].y,  0.0f, 1e-3f);
    EXPECT_NEAR(corners[1].x,  7.0f, 1e-3f);
    EXPECT_NEAR(corners[1].y, 10.0f, 1e-3f);
    EXPECT_NEAR(corners[2].x,  3.0f, 1e-3f);
    EXPECT_NEAR(corners[2].y, 10.0f, 1e-3f);
    EXPECT_NEAR(corners[3].x,  3.0f, 1e-3f);
    EXPECT_NEAR(corners[3].y,  0.0f, 1e-3f);
}

// ---------------------------------------------------------------------------
// set_obb_from_corners / get_obb_corners roundtrip
// ---------------------------------------------------------------------------

static void test_obb_corners_roundtrip() {
    OrientedBoundingBox obb = {};
    obb.center   = {0, 0};
    obb.width    = 20.0f;
    obb.height   = 10.0f;
    obb.rotation = 0.0f;
    obb.state    = OBBComplete;

    ImVec2 corners[4];
    get_obb_corners(&obb, corners);

    OrientedBoundingBox obb2 = {};
    set_obb_from_corners(&obb2, corners, 0);

    EXPECT_NEAR(obb2.center.x,   0.0f, 1e-3f);
    EXPECT_NEAR(obb2.center.y,   0.0f, 1e-3f);
    EXPECT_NEAR(obb2.width,     20.0f, 1e-3f);
    EXPECT_NEAR(obb2.height,    10.0f, 1e-3f);
    EXPECT_NEAR(obb2.rotation,   0.0f, 1e-3f);
    EXPECT_TRUE(obb2.state == OBBComplete);
    EXPECT_TRUE(obb2.class_id == 0);
}

// ---------------------------------------------------------------------------
// is_point_inside_obb
// ---------------------------------------------------------------------------

static void test_is_point_inside_obb() {
    OrientedBoundingBox obb = {};
    obb.center   = {0, 0};
    obb.width    = 20.0f;
    obb.height   = 10.0f;
    obb.rotation = 0.0f;
    obb.state    = OBBComplete;

    // Center and interior points
    EXPECT_TRUE(is_point_inside_obb({0, 0}, obb));
    EXPECT_TRUE(is_point_inside_obb({5, 3}, obb));

    // Outside on each axis
    EXPECT_FALSE(is_point_inside_obb({15, 0}, obb));
    EXPECT_FALSE(is_point_inside_obb({0, 8}, obb));

    // Incomplete state: always false
    obb.state = OBBFirstAxisPoint;
    EXPECT_FALSE(is_point_inside_obb({0, 0}, obb));
}

// ---------------------------------------------------------------------------
// calculate_obb_properties
// ---------------------------------------------------------------------------

static void test_calculate_obb_properties() {
    // Horizontal rectangle: edge (0,0)→(10,0), third point at (5,4)
    // edge_vector=(10,0), perp_vector=(0,10)
    // height = |to_mouse · perp_unit| = |(5,4)·(0,1)| = 4
    // vertices: v1=(0,0), v2=(10,0), v3=(10,4), v4=(0,4)
    // center=(5,2), width=10, height=4, rotation=atan2(0,10)=0
    OrientedBoundingBox obb = {};
    obb.axis_point1  = {0, 0};
    obb.axis_point2  = {10, 0};
    obb.corner_point = {5, 4};
    obb.state        = OBBThirdPoint;

    calculate_obb_properties(&obb);

    EXPECT_NEAR(obb.width,    10.0f, 1e-3f);
    EXPECT_NEAR(obb.height,    4.0f, 1e-3f);
    EXPECT_NEAR(obb.rotation,  0.0f, 1e-3f);
    EXPECT_NEAR(obb.center.x,  5.0f, 1e-3f);
    EXPECT_NEAR(obb.center.y,  2.0f, 1e-3f);

    // Insufficient state: properties should not change
    OrientedBoundingBox obb2 = {};
    obb2.state = OBBFirstAxisPoint;
    obb2.width = 99.0f;
    calculate_obb_properties(&obb2);
    EXPECT_NEAR(obb2.width, 99.0f, 1e-3f);
}

// ---------------------------------------------------------------------------
// current_date_time
// Format: YYYY_MM_DD_HH_MM_SS  (19 chars, underscores at 4,7,10,13,16)
// ---------------------------------------------------------------------------

static void test_current_date_time() {
    std::string dt = current_date_time();

    EXPECT_TRUE(dt.length() == 19);

    // Underscores at expected positions
    EXPECT_TRUE(dt[4]  == '_');
    EXPECT_TRUE(dt[7]  == '_');
    EXPECT_TRUE(dt[10] == '_');
    EXPECT_TRUE(dt[13] == '_');
    EXPECT_TRUE(dt[16] == '_');

    // All other characters are digits
    for (int i = 0; i < 19; i++) {
        if (i == 4 || i == 7 || i == 10 || i == 13 || i == 16)
            continue;
        EXPECT_TRUE(isdigit((unsigned char)dt[i]));
    }
}

// ---------------------------------------------------------------------------
// CalibrationToolState: default initialization
// ---------------------------------------------------------------------------

static void test_calib_state_defaults() {
    CalibrationToolState s;
    EXPECT_FALSE(s.show);
    EXPECT_FALSE(s.project_loaded);
    EXPECT_FALSE(s.dock_pending);
    EXPECT_TRUE(s.show_create_dialog);
    EXPECT_FALSE(s.config_loaded);
    EXPECT_FALSE(s.images_loaded);
    EXPECT_TRUE(s.status.empty());
    EXPECT_FALSE(s.running);
    EXPECT_FALSE(s.done);

    // Laser defaults
    EXPECT_FALSE(s.laser_ready);
    EXPECT_FALSE(s.laser_load_pending);
    EXPECT_TRUE(s.laser_total_frames == 0);
    EXPECT_FALSE(s.laser_running);
    EXPECT_FALSE(s.laser_done);
    EXPECT_TRUE(s.laser_status.empty());
    EXPECT_FALSE(s.laser_show_detection);
    EXPECT_FALSE(s.laser_focus_window);
    EXPECT_TRUE(s.laser_progress != nullptr);

    // LaserVizState defaults
    EXPECT_TRUE(s.laser_viz.ready.empty());
    EXPECT_TRUE(s.laser_viz.pending.empty());
    EXPECT_FALSE(s.laser_viz.computing.load());
    EXPECT_TRUE(s.laser_viz.last_green_th == -1);
}

// ---------------------------------------------------------------------------
// CalibrationToolState: reset-on-close logic
// ---------------------------------------------------------------------------

static void test_calib_state_reset_on_close() {
    // Simulate a loaded project, then close
    CalibrationToolState s;
    s.show = true;
    s.project_loaded = true;
    s.show_create_dialog = false;
    s.config_loaded = true;
    s.images_loaded = true;
    s.done = true;
    s.status = "Calibration complete!";
    s.laser_ready = true;
    s.laser_done = true;
    s.laser_status = "Laser done";
    s.laser_show_detection = true;

    // Simulate closing (this is what DrawCalibrationToolWindow does when !state.show)
    s.show = false;
    // Inline the reset logic from the draw function
    s.project_loaded = false;
    s.show_create_dialog = true;
    s.config_loaded = false;
    s.images_loaded = false;
    s.done = false;
    s.status.clear();
    s.laser_ready = false;
    s.laser_done = false;
    s.laser_status.clear();
    s.laser_show_detection = false;
    if (s.laser_viz.worker.joinable())
        s.laser_viz.worker.join();
    s.laser_viz.ready.clear();
    s.laser_viz.pending.clear();

    // Verify all state is reset
    EXPECT_FALSE(s.show);
    EXPECT_FALSE(s.project_loaded);
    EXPECT_TRUE(s.show_create_dialog);
    EXPECT_FALSE(s.config_loaded);
    EXPECT_FALSE(s.images_loaded);
    EXPECT_FALSE(s.done);
    EXPECT_TRUE(s.status.empty());
    EXPECT_FALSE(s.laser_ready);
    EXPECT_FALSE(s.laser_done);
    EXPECT_TRUE(s.laser_status.empty());
    EXPECT_FALSE(s.laser_show_detection);
    EXPECT_TRUE(s.laser_viz.ready.empty());
    EXPECT_TRUE(s.laser_viz.pending.empty());
}

// ---------------------------------------------------------------------------
// CalibrationToolCallbacks: all fields callable
// ---------------------------------------------------------------------------

static void test_calib_callbacks_callable() {
    CalibrationToolCallbacks cb;

    // All callbacks should be empty/null by default
    EXPECT_TRUE(!cb.load_images);
    EXPECT_TRUE(!cb.load_videos);
    EXPECT_TRUE(!cb.unload_media);
    EXPECT_TRUE(!cb.copy_default_layout);
    EXPECT_TRUE(!cb.switch_ini);
    EXPECT_TRUE(!cb.print_metadata);

    // Wire up with no-op lambdas and verify they're callable
    int call_count = 0;
    cb.load_images = [&](std::map<std::string, std::string> &) { call_count++; };
    cb.load_videos = [&]() { call_count++; };
    cb.unload_media = [&]() { call_count++; };
    cb.copy_default_layout = [&](const std::string &) { call_count++; };
    cb.switch_ini = [&](const std::string &) { call_count++; };
    cb.print_metadata = [&]() { call_count++; };

    std::map<std::string, std::string> dummy_files;
    cb.load_images(dummy_files);
    cb.load_videos();
    cb.unload_media();
    cb.copy_default_layout("/tmp/test");
    cb.switch_ini("/tmp/test");
    cb.print_metadata();

    EXPECT_TRUE(call_count == 6);
}

// ---------------------------------------------------------------------------
// CalibrationToolState: project initialization pattern
// ---------------------------------------------------------------------------

static void test_calib_state_project_init() {
    CalibrationToolState s;

    // Simulate the initialization pattern from red.cpp main()
    std::string default_root = "/tmp/test_root";
    std::string default_media = "/tmp/test_media";
    s.project.project_root_path = default_root;
    s.project.config_file = default_media;

    EXPECT_TRUE(s.project.project_root_path == default_root);
    EXPECT_TRUE(s.project.config_file == default_media);
    EXPECT_TRUE(s.project.project_name.empty());
    EXPECT_TRUE(s.project.project_path.empty());
}

// ---------------------------------------------------------------------------
// LaserVizState: double-buffer lifecycle
// ---------------------------------------------------------------------------

static void test_laser_viz_double_buffer() {
    LaserVizState lv;

    // Simulate producing results
    std::vector<LaserVizState::CamResult> results(4);
    for (int i = 0; i < 4; i++) {
        results[i].frame_num = 42;
        results[i].num_blobs = (i == 0) ? 1 : 0;
        results[i].total_mask_pixels = i * 10;
    }

    // Simulate background thread completing
    lv.pending = std::move(results);
    EXPECT_TRUE(lv.ready.empty());
    EXPECT_TRUE(lv.pending.size() == 4);

    // Simulate main thread consuming
    lv.ready = std::move(lv.pending);
    lv.pending.clear();
    EXPECT_TRUE(lv.ready.size() == 4);
    EXPECT_TRUE(lv.pending.empty());
    EXPECT_TRUE(lv.ready[0].num_blobs == 1);
    EXPECT_TRUE(lv.ready[0].frame_num == 42);
    EXPECT_TRUE(lv.ready[1].num_blobs == 0);
    EXPECT_TRUE(lv.ready[2].total_mask_pixels == 20);

    // Mark as uploaded
    for (auto &cr : lv.ready) cr.uploaded = false;
    lv.ready[0].uploaded = true;
    EXPECT_TRUE(lv.ready[0].uploaded);
    EXPECT_FALSE(lv.ready[1].uploaded);
}

// ---------------------------------------------------------------------------
// LaserVizState: params-changed detection
// ---------------------------------------------------------------------------

static void test_laser_viz_params_changed() {
    LaserVizState lv;
    lv.last_green_th = 100;
    lv.last_green_dom = 30;
    lv.last_min_blob = 5;
    lv.last_max_blob = 500;

    // Same params: no change
    auto params_changed = [&](int gt, int gd, int minb, int maxb) {
        return gt != lv.last_green_th || gd != lv.last_green_dom ||
               minb != lv.last_min_blob || maxb != lv.last_max_blob;
    };

    EXPECT_FALSE(params_changed(100, 30, 5, 500));
    EXPECT_TRUE(params_changed(101, 30, 5, 500));  // green_threshold changed
    EXPECT_TRUE(params_changed(100, 31, 5, 500));  // green_dominance changed
    EXPECT_TRUE(params_changed(100, 30, 6, 500));  // min_blob changed
    EXPECT_TRUE(params_changed(100, 30, 5, 501));  // max_blob changed
}

// ---------------------------------------------------------------------------
// CalibrationToolState: menu open patterns
// ---------------------------------------------------------------------------

static void test_calib_menu_open_patterns() {
    CalibrationToolState s;

    // "Create Calibration Project" menu item
    s.show = true;
    s.show_create_dialog = true;
    EXPECT_TRUE(s.show);
    EXPECT_TRUE(s.show_create_dialog);
    EXPECT_FALSE(s.project_loaded);

    // After project creation succeeds
    s.project_loaded = true;
    s.dock_pending = true;
    s.show_create_dialog = false;
    EXPECT_TRUE(s.project_loaded);
    EXPECT_TRUE(s.dock_pending);
    EXPECT_FALSE(s.show_create_dialog);

    // After docking completes
    s.dock_pending = false;
    EXPECT_FALSE(s.dock_pending);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    test_is_point_near();
    test_is_point_near_line_segment();
    test_get_obb_corners_axis_aligned();
    test_get_obb_corners_incomplete_state();
    test_get_obb_corners_rotated();
    test_obb_corners_roundtrip();
    test_is_point_inside_obb();
    test_calculate_obb_properties();
    test_current_date_time();

    // Calibration tool extraction tests
    test_calib_state_defaults();
    test_calib_state_reset_on_close();
    test_calib_callbacks_callable();
    test_calib_state_project_init();
    test_laser_viz_double_buffer();
    test_laser_viz_params_changed();
    test_calib_menu_open_patterns();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
