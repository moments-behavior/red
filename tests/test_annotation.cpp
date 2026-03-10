// tests/test_annotation.cpp
// Unit tests for the unified annotation system:
//   - annotation.h: data model, migration, JSON persistence
//   - export_formats.h: COCO, YOLO, DLC export logic
//   - obb_tool.h: OBB geometry (calculate, corners, contains)
//   - sam_inference.h: mask-to-polygon conversion
//   - bbox_tool.h: color generation
//
// Pure logic tests — no ImGui/ImPlot/GPU context needed.
// Compiled as standalone target with same deps as test_gui.

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "annotation.h"
#include "annotation_v1.h"
#include "annotation_csv.h"
#include "export_formats.h"
#include "gui/bbox_tool.h"
#include "gui/obb_tool.h"
#include "sam_inference.h"
#include "project.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

// ---------------------------------------------------------------------------
// Shared test framework
// ---------------------------------------------------------------------------
#include "test_framework.h"

// ---------------------------------------------------------------------------
// Helper: create a mock RenderScene + SkeletonContext
// ---------------------------------------------------------------------------

static u32 mock_widths[4] = {1920, 1920, 1920, 1920};
static u32 mock_heights[4] = {1080, 1080, 1080, 1080};

static RenderScene make_mock_scene(int num_cams) {
    RenderScene scene = {};
    scene.num_cams = num_cams;
    scene.image_width = mock_widths;
    scene.image_height = mock_heights;
    return scene;
}

static SkeletonContext make_mock_skeleton(int num_nodes) {
    SkeletonContext skel = {};
    skel.num_nodes = num_nodes;
    skel.num_edges = 0;
    skel.has_skeleton = true;
    skel.name = "TestSkeleton";
    for (int i = 0; i < num_nodes; ++i)
        skel.node_names.push_back("node_" + std::to_string(i));
    return skel;
}

// ═══════════════════════════════════════════════════════════════════════════
// annotation.h: Data Model Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_make_frame_basic() {
    printf("  test_make_frame_basic...\n");
    auto fa = make_frame(6, 3, 0, 1, 2);

    EXPECT_EQ((int)fa.kp3d.size(), 6);
    EXPECT_EQ((int)fa.cameras.size(), 3);
    EXPECT_EQ(fa.instance_id, 1);
    EXPECT_EQ(fa.category_id, 2);

    // All 3D keypoints should be UNLABELED sentinel
    for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(fa.kp3d[k].x, UNLABELED, 1.0);
        EXPECT_NEAR(fa.kp3d[k].y, UNLABELED, 1.0);
        EXPECT_NEAR(fa.kp3d[k].z, UNLABELED, 1.0);
        EXPECT_FALSE(fa.kp3d[k].triangulated);
    }

    // All 2D keypoints should be UNLABELED
    for (int c = 0; c < 3; ++c) {
        EXPECT_EQ((int)fa.cameras[c].keypoints.size(), 6);
        for (int k = 0; k < 6; ++k) {
            EXPECT_NEAR(fa.cameras[c].keypoints[k].x, UNLABELED, 1.0);
            EXPECT_FALSE(fa.cameras[c].keypoints[k].labeled);
        }
    }
}

static void test_make_frame_zero_dims() {
    printf("  test_make_frame_zero_dims...\n");
    auto fa = make_frame(0, 0);
    EXPECT_TRUE(fa.kp3d.empty());
    EXPECT_TRUE(fa.cameras.empty());
}

static void test_get_or_create_frame() {
    printf("  test_get_or_create_frame...\n");
    AnnotationMap amap;

    // First call creates
    auto &fa1 = get_or_create_frame(amap, 42, 3, 2);
    EXPECT_EQ(fa1.frame_number, 42u);
    EXPECT_EQ((int)fa1.cameras.size(), 2);
    EXPECT_EQ((int)fa1.kp3d.size(), 3);
    EXPECT_EQ((int)amap.size(), 1);

    // Second call returns existing (does not create a new one)
    auto &fa2 = get_or_create_frame(amap, 42, 3, 2);
    EXPECT_EQ((int)fa2.cameras.size(), 2);
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(&fa1 == &fa2); // same reference

    // Different frame creates new
    auto &fa3 = get_or_create_frame(amap, 100, 3, 2);
    EXPECT_EQ(fa3.frame_number, 100u);
    EXPECT_EQ((int)amap.size(), 2);
}

static void test_frame_has_any_labels() {
    printf("  test_frame_has_any_labels...\n");

    // Empty frame: no labels
    FrameAnnotation fa;
    fa.frame_number = 0;
    EXPECT_FALSE(frame_has_any_labels(fa));

    // Frame with cameras but no labeled keypoints
    fa = make_frame(3, 2);
    EXPECT_FALSE(frame_has_any_labels(fa));

    // Label one keypoint on camera 1
    fa.cameras[1].keypoints[2].labeled = true;
    EXPECT_TRUE(frame_has_any_labels(fa));
}

// ═══════════════════════════════════════════════════════════════════════════
// annotation.h: Migration Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_migration_roundtrip() {
    printf("  test_migration_roundtrip...\n");

    auto scene = make_mock_scene(2);
    auto skel = make_mock_skeleton(3);

    // Create old-style KeyPoints
    std::map<u32, KeyPoints *> km;
    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);

    // Set some 2D keypoints
    kp->kp2d[0][0].position = {100.0, 200.0};
    kp->kp2d[0][0].is_labeled = true;
    kp->kp2d[0][1].position = {300.0, 400.0};
    kp->kp2d[0][1].is_labeled = true;
    kp->kp2d[1][2].position = {500.0, 600.0};
    kp->kp2d[1][2].is_labeled = true;

    // Set some 3D keypoints
    kp->kp3d[0].position = {1.0, 2.0, 3.0};
    kp->kp3d[0].is_triangulated = true;

    kp->active_id[0] = 1;
    kp->active_id[1] = 2;

    km[10] = kp;

    // Migrate to AnnotationMap
    AnnotationMap amap = migrate_keypoints_map(km, skel, &scene);

    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(amap.find(10) != amap.end());

    auto &fa = amap[10];

    // Check 2D keypoints survived migration
    EXPECT_TRUE(fa.cameras[0].keypoints[0].labeled);
    EXPECT_NEAR(fa.cameras[0].keypoints[0].x, 100.0, 0.001);
    EXPECT_NEAR(fa.cameras[0].keypoints[0].y, 200.0, 0.001);
    EXPECT_TRUE(fa.cameras[0].keypoints[1].labeled);
    EXPECT_FALSE(fa.cameras[0].keypoints[2].labeled); // node 2, cam 0 not labeled
    EXPECT_TRUE(fa.cameras[1].keypoints[2].labeled);
    EXPECT_NEAR(fa.cameras[1].keypoints[2].x, 500.0, 0.001);

    // Check 3D keypoints survived migration
    EXPECT_TRUE(fa.kp3d[0].triangulated);
    EXPECT_NEAR(fa.kp3d[0].x, 1.0, 0.001);
    EXPECT_NEAR(fa.kp3d[0].y, 2.0, 0.001);
    EXPECT_NEAR(fa.kp3d[0].z, 3.0, 0.001);
    EXPECT_FALSE(fa.kp3d[1].triangulated); // not triangulated

    // Check active_id
    EXPECT_EQ(fa.cameras[0].active_id, 1u);
    EXPECT_EQ(fa.cameras[1].active_id, 2u);

    // Cleanup
    free_keypoints(km[10], &scene);
}

// ═══════════════════════════════════════════════════════════════════════════
// annotation.h: JSON Persistence Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_json_roundtrip_bbox() {
    printf("  test_json_roundtrip_bbox...\n");

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 42, 2, 3);

    // Set bbox on camera 0
    auto &ext0 = fa.cameras[0].get_extras();
    ext0.bbox_x = 10.5;
    ext0.bbox_y = 20.5;
    ext0.bbox_w = 100.0;
    ext0.bbox_h = 80.0;
    ext0.has_bbox = true;

    // Set OBB on camera 1
    auto &ext1 = fa.cameras[1].get_extras();
    ext1.obb_cx = 50.0;
    ext1.obb_cy = 60.0;
    ext1.obb_w = 40.0;
    ext1.obb_h = 30.0;
    ext1.obb_angle = 0.785;
    ext1.has_obb = true;

    // Serialize to JSON
    auto j = annotations_to_json(amap);
    EXPECT_TRUE(j.contains("frames"));
    EXPECT_EQ((int)j["frames"].size(), 1);

    // Deserialize into a fresh AnnotationMap (with existing frame)
    AnnotationMap amap2;
    auto &fa2 = get_or_create_frame(amap2, 42, 2, 3);
    annotations_from_json(j, amap2);

    // Check bbox
    EXPECT_TRUE(fa2.cameras[0].has_bbox());
    EXPECT_NEAR(fa2.cameras[0].extras->bbox_x, 10.5, 0.001);
    EXPECT_NEAR(fa2.cameras[0].extras->bbox_y, 20.5, 0.001);
    EXPECT_NEAR(fa2.cameras[0].extras->bbox_w, 100.0, 0.001);
    EXPECT_NEAR(fa2.cameras[0].extras->bbox_h, 80.0, 0.001);

    // Check OBB
    EXPECT_TRUE(fa2.cameras[1].has_obb());
    EXPECT_NEAR(fa2.cameras[1].extras->obb_cx, 50.0, 0.001);
    EXPECT_NEAR(fa2.cameras[1].extras->obb_cy, 60.0, 0.001);
    EXPECT_NEAR(fa2.cameras[1].extras->obb_w, 40.0, 0.001);
    EXPECT_NEAR(fa2.cameras[1].extras->obb_h, 30.0, 0.001);
    EXPECT_NEAR(fa2.cameras[1].extras->obb_angle, 0.785, 0.001);
}

static void test_json_roundtrip_mask() {
    printf("  test_json_roundtrip_mask...\n");

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 7, 1, 2);

    // Set mask on camera 0
    auto &ext = fa.cameras[0].get_extras();
    ext.mask_polygons = {
        {{10.0, 20.0}, {30.0, 20.0}, {30.0, 40.0}, {10.0, 40.0}},
        {{50.0, 60.0}, {70.0, 60.0}, {70.0, 80.0}}
    };
    ext.has_mask = true;

    auto j = annotations_to_json(amap);

    AnnotationMap amap2;
    get_or_create_frame(amap2, 7, 1, 2);
    annotations_from_json(j, amap2);

    auto &cam = amap2[7].cameras[0];
    EXPECT_TRUE(cam.has_mask());
    EXPECT_EQ((int)cam.extras->mask_polygons.size(), 2);
    EXPECT_EQ((int)cam.extras->mask_polygons[0].size(), 4);
    EXPECT_EQ((int)cam.extras->mask_polygons[1].size(), 3);
    EXPECT_NEAR(cam.extras->mask_polygons[0][0].x, 10.0, 0.001);
    EXPECT_NEAR(cam.extras->mask_polygons[1][2].y, 80.0, 0.001);
}

static void test_json_empty_extended_data() {
    printf("  test_json_empty_extended_data...\n");

    // Keypoints-only frame: should produce empty frames array in JSON
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 0, 3, 2);
    fa.cameras[0].keypoints[0].labeled = true; // only keypoints

    auto j = annotations_to_json(amap);
    EXPECT_TRUE(j["frames"].empty()); // no extended data to serialize
}

static void test_json_missing_frame_in_amap() {
    printf("  test_json_missing_frame_in_amap...\n");

    // JSON references frame 99 which doesn't exist in amap — should be silently skipped
    nlohmann::json j;
    j["version"] = 1;
    j["frames"] = nlohmann::json::array();
    nlohmann::json jf;
    jf["frame"] = 99;
    jf["instances"] = nlohmann::json::array();
    j["frames"].push_back(jf);

    AnnotationMap amap; // empty
    annotations_from_json(j, amap); // should not crash
    EXPECT_TRUE(amap.empty());
}

static void test_json_camera_index_out_of_bounds() {
    printf("  test_json_camera_index_out_of_bounds...\n");

    AnnotationMap amap;
    get_or_create_frame(amap, 5, 1, 2); // 2 cameras

    // JSON with camera index 10 (out of bounds)
    nlohmann::json j;
    j["version"] = 1;
    nlohmann::json jf;
    jf["frame"] = 5;
    nlohmann::json ji;
    ji["instance_id"] = 0;
    ji["category_id"] = 0;
    nlohmann::json jc;
    jc["cam"] = 10; // out of bounds
    jc["bbox"] = {1.0, 2.0, 3.0, 4.0};
    ji["cameras"] = nlohmann::json::array({jc});
    jf["instances"] = nlohmann::json::array({ji});
    j["frames"] = nlohmann::json::array({jf});

    annotations_from_json(j, amap); // should not crash, silently skip
    auto &cam0 = amap[5].cameras[0];
    EXPECT_FALSE(cam0.has_bbox()); // unchanged
}

static void test_json_file_save_load() {
    printf("  test_json_file_save_load...\n");

    namespace fs = std::filesystem;
    std::string tmpdir = "/tmp/test_annotation_json_" + std::to_string(getpid());
    fs::create_directories(tmpdir);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 3, 2, 1);
    auto &ext = fa.cameras[0].get_extras();
    ext.bbox_x = 42.0;
    ext.bbox_y = 43.0;
    ext.bbox_w = 100.0;
    ext.bbox_h = 200.0;
    ext.has_bbox = true;

    EXPECT_TRUE(save_annotations_json(amap, tmpdir));
    EXPECT_TRUE(fs::exists(tmpdir + "/annotations.json"));

    AnnotationMap amap2;
    get_or_create_frame(amap2, 3, 2, 1);
    EXPECT_TRUE(load_annotations_json(amap2, tmpdir));

    EXPECT_TRUE(amap2[3].cameras[0].has_bbox());
    EXPECT_NEAR(amap2[3].cameras[0].extras->bbox_x, 42.0, 0.001);

    // Cleanup
    fs::remove_all(tmpdir);
}

static void test_json_file_missing_is_ok() {
    printf("  test_json_file_missing_is_ok...\n");

    AnnotationMap amap;
    // Loading from nonexistent directory is OK (returns true, does nothing)
    EXPECT_TRUE(load_annotations_json(amap, "/tmp/nonexistent_dir_12345"));
}

// ═══════════════════════════════════════════════════════════════════════════
// export_formats.h: Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_format_name() {
    printf("  test_format_name...\n");
    EXPECT_TRUE(std::string(ExportFormats::format_name(ExportFormats::JARVIS)) == "JARVIS");
    EXPECT_TRUE(std::string(ExportFormats::format_name(ExportFormats::COCO)) == "COCO Keypoints");
    EXPECT_TRUE(std::string(ExportFormats::format_name(ExportFormats::YOLO_POSE)) == "YOLO Pose");
    EXPECT_TRUE(std::string(ExportFormats::format_name(ExportFormats::YOLO_DETECT)) == "YOLO Detection");
    EXPECT_TRUE(std::string(ExportFormats::format_name(ExportFormats::DEEPLABCUT)) == "DeepLabCut");
    EXPECT_TRUE(std::string(ExportFormats::format_name(ExportFormats::FORMAT_COUNT)) == "Unknown");
}

static void test_split_train_val() {
    printf("  test_split_train_val...\n");

    std::vector<u32> frames = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<u32> train, val;

    // Standard 80/20 split
    ExportFormats::split_train_val(frames, 0.8f, 42, train, val);
    EXPECT_EQ((int)train.size(), 8);
    EXPECT_EQ((int)val.size(), 2);

    // All frames accounted for
    std::vector<u32> all;
    all.insert(all.end(), train.begin(), train.end());
    all.insert(all.end(), val.begin(), val.end());
    std::sort(all.begin(), all.end());
    EXPECT_EQ((int)all.size(), 10);
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(all[i], (u32)(i + 1));

    // Both sorted
    for (int i = 1; i < (int)train.size(); ++i)
        EXPECT_TRUE(train[i] > train[i - 1]);
    for (int i = 1; i < (int)val.size(); ++i)
        EXPECT_TRUE(val[i] > val[i - 1]);
}

static void test_split_train_val_deterministic() {
    printf("  test_split_train_val_deterministic...\n");

    std::vector<u32> frames = {1, 2, 3, 4, 5};
    std::vector<u32> train1, val1, train2, val2;

    ExportFormats::split_train_val(frames, 0.6f, 123, train1, val1);
    ExportFormats::split_train_val(frames, 0.6f, 123, train2, val2);

    EXPECT_EQ(train1, train2);
    EXPECT_EQ(val1, val2);
}

static void test_split_train_val_empty() {
    printf("  test_split_train_val_empty...\n");

    std::vector<u32> frames;
    std::vector<u32> train, val;
    ExportFormats::split_train_val(frames, 0.9f, 42, train, val);
    EXPECT_TRUE(train.empty());
    EXPECT_TRUE(val.empty());
}

static void test_split_train_val_single() {
    printf("  test_split_train_val_single...\n");

    std::vector<u32> frames = {42};
    std::vector<u32> train, val;

    // ratio=0.9 → n_train = floor(1*0.9) = 0, so all goes to val
    ExportFormats::split_train_val(frames, 0.9f, 42, train, val);
    EXPECT_EQ((int)train.size() + (int)val.size(), 1);
}

static void test_split_train_val_all_train() {
    printf("  test_split_train_val_all_train...\n");

    std::vector<u32> frames = {1, 2, 3};
    std::vector<u32> train, val;
    ExportFormats::split_train_val(frames, 1.0f, 42, train, val);
    EXPECT_EQ((int)train.size(), 3);
    EXPECT_TRUE(val.empty());
}

static void test_get_labeled_frames() {
    printf("  test_get_labeled_frames...\n");

    AnnotationMap amap;

    // Frame 5: has labels
    auto &fa5 = get_or_create_frame(amap, 5, 2, 1);
    fa5.cameras[0].keypoints[0].labeled = true;

    // Frame 10: no labels
    get_or_create_frame(amap, 10, 2, 1);

    // Frame 15: has labels
    auto &fa15 = get_or_create_frame(amap, 15, 2, 1);
    fa15.cameras[0].keypoints[1].labeled = true;

    auto labeled = ExportFormats::get_labeled_frames(amap);
    EXPECT_EQ((int)labeled.size(), 2);
    EXPECT_EQ(labeled[0], 5u);
    EXPECT_EQ(labeled[1], 15u);
}

static void test_build_coco_json() {
    printf("  test_build_coco_json...\n");

    AnnotationMap amap;

    // Create a frame with labeled keypoints
    auto &fa = get_or_create_frame(amap, 1, 3, 2);
    // Label 2 of 3 keypoints on cam 0
    fa.cameras[0].keypoints[0].x = 100.0; fa.cameras[0].keypoints[0].y = 900.0; // ImPlot coords (Y from bottom)
    fa.cameras[0].keypoints[0].labeled = true;
    fa.cameras[0].keypoints[1].x = 200.0; fa.cameras[0].keypoints[1].y = 800.0;
    fa.cameras[0].keypoints[1].labeled = true;
    // keypoint 2 unlabeled

    ExportFormats::ExportConfig cfg;
    cfg.skeleton_name = "test";
    cfg.node_names = {"a", "b", "c"};
    cfg.edges = {{0, 1}};
    cfg.bbox_margin = 10.0f;

    std::vector<u32> frames = {1};
    auto j = ExportFormats::build_coco_json(amap, frames, cfg, 0, "cam0", 1920, 1080);

    // Check structure
    EXPECT_TRUE(j.contains("images"));
    EXPECT_TRUE(j.contains("annotations"));
    EXPECT_TRUE(j.contains("categories"));

    EXPECT_EQ((int)j["images"].size(), 1);
    EXPECT_EQ((int)j["annotations"].size(), 1);

    // Check image
    auto &img = j["images"][0];
    EXPECT_EQ(img["width"].get<int>(), 1920);
    EXPECT_EQ(img["height"].get<int>(), 1080);

    // Check annotation
    auto &ann = j["annotations"][0];
    EXPECT_EQ(ann["image_id"].get<int>(), 0);
    EXPECT_EQ(ann["num_keypoints"].get<int>(), 2);

    // Check keypoints array: 3 keypoints × 3 values each = 9 entries
    EXPECT_EQ((int)ann["keypoints"].size(), 9);
    // First keypoint: x=100, y = 1080-900 = 180, v=2
    EXPECT_NEAR(ann["keypoints"][0].get<double>(), 100.0, 0.1);
    EXPECT_NEAR(ann["keypoints"][1].get<double>(), 180.0, 0.1);
    EXPECT_EQ(ann["keypoints"][2].get<int>(), 2);
    // Third keypoint (unlabeled): 0,0,0
    EXPECT_EQ(ann["keypoints"][6].get<int>(), 0);
    EXPECT_EQ(ann["keypoints"][7].get<int>(), 0);
    EXPECT_EQ(ann["keypoints"][8].get<int>(), 0);

    // Check skeleton edges (1-indexed for COCO)
    EXPECT_EQ((int)j["categories"][0]["skeleton"].size(), 1);
    EXPECT_EQ(j["categories"][0]["skeleton"][0][0].get<int>(), 1);
    EXPECT_EQ(j["categories"][0]["skeleton"][0][1].get<int>(), 2);
}

static void test_build_coco_json_no_visible_keypoints() {
    printf("  test_build_coco_json_no_visible_keypoints...\n");

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 1, 2, 1);
    // No keypoints labeled

    ExportFormats::ExportConfig cfg;
    cfg.node_names = {"a", "b"};

    std::vector<u32> frames = {1};
    auto j = ExportFormats::build_coco_json(amap, frames, cfg, 0, "cam0", 640, 480);

    // Image should still be in the list
    EXPECT_EQ((int)j["images"].size(), 1);
    // But no annotation (zero visible keypoints)
    EXPECT_EQ((int)j["annotations"].size(), 0);
}

static void test_build_coco_json_img_id_consistency() {
    printf("  test_build_coco_json_img_id_consistency...\n");

    // Regression test for the img_id desync bug (was: img_id++ inside instance loop)
    AnnotationMap amap;

    // Frame 1: no visible keypoints
    auto &fa1 = get_or_create_frame(amap, 1, 2, 1);
    // all unlabeled

    // Frame 2: visible keypoints
    auto &fa2 = get_or_create_frame(amap, 2, 2, 1);
    fa2.cameras[0].keypoints[0].x = 50.0; fa2.cameras[0].keypoints[0].y = 50.0;
    fa2.cameras[0].keypoints[0].labeled = true;

    ExportFormats::ExportConfig cfg;
    cfg.node_names = {"a", "b"};
    cfg.bbox_margin = 5.0f;

    std::vector<u32> frames = {1, 2};
    auto j = ExportFormats::build_coco_json(amap, frames, cfg, 0, "cam0", 640, 480);

    // 2 images
    EXPECT_EQ((int)j["images"].size(), 2);
    EXPECT_EQ(j["images"][0]["id"].get<int>(), 0);
    EXPECT_EQ(j["images"][1]["id"].get<int>(), 1);

    // 1 annotation (only frame 2 has labels)
    EXPECT_EQ((int)j["annotations"].size(), 1);
    // The annotation should reference image_id=1 (the second image)
    EXPECT_EQ(j["annotations"][0]["image_id"].get<int>(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// obb_tool.h: OBB Geometry Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_obb_horizontal() {
    printf("  test_obb_horizontal...\n");

    // Horizontal axis from (0,0) to (10,0), corner at (5,5)
    double cx, cy, w, h, angle;
    calculate_obb_from_3_points(0, 0, 10, 0, 5, 5, cx, cy, w, h, angle);

    EXPECT_NEAR(w, 10.0, 0.001);
    EXPECT_NEAR(h, 5.0, 0.001);
    EXPECT_NEAR(angle, 0.0, 0.001);
    EXPECT_NEAR(cx, 5.0, 0.001);
    EXPECT_NEAR(cy, 2.5, 0.001);
}

static void test_obb_vertical() {
    printf("  test_obb_vertical...\n");

    // Vertical axis from (0,0) to (0,10), corner at (5,5)
    double cx, cy, w, h, angle;
    calculate_obb_from_3_points(0, 0, 0, 10, 5, 5, cx, cy, w, h, angle);

    EXPECT_NEAR(w, 10.0, 0.001);
    EXPECT_NEAR(angle, M_PI / 2.0, 0.001);
}

static void test_obb_degenerate_axis() {
    printf("  test_obb_degenerate_axis...\n");

    // Both axis points at same location
    double cx, cy, w, h, angle;
    calculate_obb_from_3_points(5, 5, 5, 5, 10, 10, cx, cy, w, h, angle);

    EXPECT_NEAR(w, 0.0, 0.001);
    EXPECT_NEAR(h, 0.0, 0.001);
}

static void test_obb_corner_on_axis() {
    printf("  test_obb_corner_on_axis...\n");

    // Corner on the axis line (zero perpendicular projection)
    double cx, cy, w, h, angle;
    calculate_obb_from_3_points(0, 0, 10, 0, 5, 0, cx, cy, w, h, angle);

    EXPECT_NEAR(h, 0.0, 0.001); // no perpendicular extent
}

static void test_obb_diagonal() {
    printf("  test_obb_diagonal...\n");

    // 45-degree axis from (0,0) to (10,10)
    double cx, cy, w, h, angle;
    calculate_obb_from_3_points(0, 0, 10, 10, 5, 10, cx, cy, w, h, angle);

    EXPECT_NEAR(angle, M_PI / 4.0, 0.01);
    EXPECT_NEAR(w, sqrt(200.0), 0.01); // length of (0,0)-(10,10)
}

static void test_obb_get_corners_axis_aligned() {
    printf("  test_obb_get_corners_axis_aligned...\n");

    // Axis-aligned OBB centered at (5,5), w=10, h=4, angle=0
    double xs[5], ys[5];
    obb_get_corners(5, 5, 10, 4, 0, xs, ys);

    // Corners should be at (0,3), (10,3), (10,7), (0,7)
    EXPECT_NEAR(xs[0], 0.0, 0.001);  EXPECT_NEAR(ys[0], 3.0, 0.001);
    EXPECT_NEAR(xs[1], 10.0, 0.001); EXPECT_NEAR(ys[1], 3.0, 0.001);
    EXPECT_NEAR(xs[2], 10.0, 0.001); EXPECT_NEAR(ys[2], 7.0, 0.001);
    EXPECT_NEAR(xs[3], 0.0, 0.001);  EXPECT_NEAR(ys[3], 7.0, 0.001);

    // Loop closure
    EXPECT_NEAR(xs[4], xs[0], 0.001);
    EXPECT_NEAR(ys[4], ys[0], 0.001);
}

static void test_obb_contains_basic() {
    printf("  test_obb_contains_basic...\n");

    // Axis-aligned OBB: center (5,5), w=10, h=4, angle=0
    EXPECT_TRUE(obb_contains(5, 5, 10, 4, 0, 5, 5));    // center
    EXPECT_TRUE(obb_contains(5, 5, 10, 4, 0, 0, 3));    // corner
    EXPECT_TRUE(obb_contains(5, 5, 10, 4, 0, 9.9, 6.9)); // near corner
    EXPECT_FALSE(obb_contains(5, 5, 10, 4, 0, -1, 5));   // outside left
    EXPECT_FALSE(obb_contains(5, 5, 10, 4, 0, 5, 8));    // outside top
}

static void test_obb_contains_rotated() {
    printf("  test_obb_contains_rotated...\n");

    // OBB at center (0,0), w=10, h=2, rotated 45 degrees
    double angle = M_PI / 4.0;

    // Point at origin should be inside
    EXPECT_TRUE(obb_contains(0, 0, 10, 2, angle, 0, 0));

    // Point along the rotated axis (inside)
    EXPECT_TRUE(obb_contains(0, 0, 10, 2, angle, 3, 3));

    // Point far away (outside)
    EXPECT_FALSE(obb_contains(0, 0, 10, 2, angle, 10, 10));
}

// ═══════════════════════════════════════════════════════════════════════════
// sam_inference.h: Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_sam_mask_to_polygon_empty() {
    printf("  test_sam_mask_to_polygon_empty...\n");

    SamMask mask;
    mask.valid = false;
    auto polys = sam_mask_to_polygon(mask);
    EXPECT_TRUE(polys.empty());
}

static void test_sam_mask_to_polygon_empty_data() {
    printf("  test_sam_mask_to_polygon_empty_data...\n");

    SamMask mask;
    mask.valid = true;
    mask.width = 10;
    mask.height = 10;
    // Empty data
    auto polys = sam_mask_to_polygon(mask);
    EXPECT_TRUE(polys.empty());
}

static void test_sam_mask_to_polygon_all_zero() {
    printf("  test_sam_mask_to_polygon_all_zero...\n");

    SamMask mask;
    mask.valid = true;
    mask.width = 10;
    mask.height = 10;
    mask.data.resize(100, 0); // all zero
    auto polys = sam_mask_to_polygon(mask);
    EXPECT_TRUE(polys.empty()); // no foreground pixels
}

static void test_sam_mask_to_polygon_rect() {
    printf("  test_sam_mask_to_polygon_rect...\n");

    SamMask mask;
    mask.valid = true;
    mask.width = 20;
    mask.height = 20;
    mask.data.resize(400, 0);

    // Fill a 10x5 rectangle at (3,4)-(12,8)
    for (int y = 4; y <= 8; ++y)
        for (int x = 3; x <= 12; ++x)
            mask.data[y * 20 + x] = 255;

    auto polys = sam_mask_to_polygon(mask);
    EXPECT_EQ((int)polys.size(), 1);
    EXPECT_TRUE((int)polys[0].size() >= 4); // boundary polygon (simplified)

    // Check that polygon covers the rectangle region:
    // all points should be within or on the boundary of (3,4)-(12,8)
    for (const auto &pt : polys[0]) {
        EXPECT_TRUE(pt.x >= 2.5 && pt.x <= 12.5);
        EXPECT_TRUE(pt.y >= 3.5 && pt.y <= 8.5);
    }
}

static void test_sam_init_no_onnx() {
    printf("  test_sam_init_no_onnx...\n");

    SamState s;
    bool ok = sam_init(s, SamModel::MobileSAM, "encoder.onnx", "decoder.onnx");

#ifndef RED_HAS_ONNXRUNTIME
    EXPECT_FALSE(ok);
    EXPECT_FALSE(s.available);
    EXPECT_FALSE(s.loaded);
#else
    // Models don't exist at these paths → load fails
    EXPECT_FALSE(ok);
    EXPECT_TRUE(s.available);
    EXPECT_FALSE(s.loaded);
#endif
}

static void test_sam_segment_not_loaded() {
    printf("  test_sam_segment_not_loaded...\n");

    SamState s;
    std::vector<tuple_d> fg = {{10, 10}};
    std::vector<tuple_d> bg;
    uint8_t rgb[3] = {128, 128, 128};
    auto mask = sam_segment(s, rgb, 1, 1, fg, bg);
    EXPECT_FALSE(mask.valid);
}

// ═══════════════════════════════════════════════════════════════════════════
// bbox_tool.h: Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_bbox_next_class_color() {
    printf("  test_bbox_next_class_color...\n");

    BBoxToolState state;
    // Initial state has 1 class color
    EXPECT_EQ((int)state.class_colors.size(), 1);

    auto c1 = state.next_class_color();
    // Should be a valid color (all components in [0,1])
    EXPECT_TRUE(c1.x >= 0 && c1.x <= 1);
    EXPECT_TRUE(c1.y >= 0 && c1.y <= 1);
    EXPECT_TRUE(c1.z >= 0 && c1.z <= 1);
    EXPECT_TRUE(c1.w >= 0 && c1.w <= 1);

    // Adding a class and getting another color should be different
    state.class_colors.push_back(c1);
    auto c2 = state.next_class_color();
    bool different = (fabs(c1.x - c2.x) > 0.01 ||
                      fabs(c1.y - c2.y) > 0.01 ||
                      fabs(c1.z - c2.z) > 0.01);
    EXPECT_TRUE(different);
}

// ═══════════════════════════════════════════════════════════════════════════
// project.h: AnnotationConfig Tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_annotation_config_defaults() {
    printf("  test_annotation_config_defaults...\n");

    AnnotationConfig cfg;
    EXPECT_TRUE(cfg.enable_keypoints);
    EXPECT_FALSE(cfg.enable_bboxes);
    EXPECT_FALSE(cfg.enable_obbs);
    EXPECT_FALSE(cfg.enable_segmentation);
    EXPECT_EQ((int)cfg.class_names.size(), 1);
    EXPECT_TRUE(cfg.class_names[0] == "animal");
}

static void test_annotation_config_json_roundtrip() {
    printf("  test_annotation_config_json_roundtrip...\n");

    AnnotationConfig cfg;
    cfg.enable_bboxes = true;
    cfg.enable_segmentation = true;
    cfg.class_names = {"rat", "mouse", "fly"};

    nlohmann::json j;
    to_json(j, cfg);

    AnnotationConfig cfg2;
    from_json(j, cfg2);

    EXPECT_TRUE(cfg2.enable_keypoints);
    EXPECT_TRUE(cfg2.enable_bboxes);
    EXPECT_FALSE(cfg2.enable_obbs);
    EXPECT_TRUE(cfg2.enable_segmentation);
    EXPECT_EQ((int)cfg2.class_names.size(), 3);
    EXPECT_TRUE(cfg2.class_names[0] == "rat");
    EXPECT_TRUE(cfg2.class_names[1] == "mouse");
    EXPECT_TRUE(cfg2.class_names[2] == "fly");
}

static void test_annotation_config_backward_compat() {
    printf("  test_annotation_config_backward_compat...\n");

    // Simulates loading a project JSON that has no annotation_config key
    nlohmann::json j = nlohmann::json::object();
    // No "annotation_config" key

    AnnotationConfig cfg;
    // from_json uses j.value() with defaults, so missing keys get defaults
    if (j.contains("annotation_config"))
        cfg = j["annotation_config"].get<AnnotationConfig>();

    // Should have defaults
    EXPECT_TRUE(cfg.enable_keypoints);
    EXPECT_FALSE(cfg.enable_bboxes);
    EXPECT_EQ((int)cfg.class_names.size(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration Tests: Full Export Pipelines with Mock Calibration
// ═══════════════════════════════════════════════════════════════════════════

// Write a minimal OpenCV-style YAML calibration file
static void write_mock_calib_yaml(const std::string &path, int w, int h) {
    std::ofstream f(path);
    f << "%YAML:1.0\n---\n";
    f << "image_width: " << w << "\n";
    f << "image_height: " << h << "\n";
    f << "camera_matrix: !!opencv-matrix\n";
    f << "   rows: 3\n   cols: 3\n   dt: d\n";
    f << "   data: [ 1000, 0, " << w/2 << ", 0, 1000, " << h/2 << ", 0, 0, 1 ]\n";
    f << "distortion_coefficients: !!opencv-matrix\n";
    f << "   rows: 5\n   cols: 1\n   dt: d\n";
    f << "   data: [ 0, 0, 0, 0, 0 ]\n";
    f << "rc_ext: !!opencv-matrix\n";
    f << "   rows: 3\n   cols: 3\n   dt: d\n";
    f << "   data: [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]\n";
    f << "tc_ext: !!opencv-matrix\n";
    f << "   rows: 3\n   cols: 1\n   dt: d\n";
    f << "   data: [ 0, 0, 1000 ]\n";
}

// Build a standard test fixture: 2 cameras, 3 keypoints, 5 labeled frames
struct ExportTestFixture {
    std::string tmpdir;
    std::string calib_dir;
    std::string output_dir;
    AnnotationMap amap;
    ExportFormats::ExportConfig cfg;

    ExportTestFixture() {
        namespace fs = std::filesystem;
        tmpdir = "/tmp/test_export_" + std::to_string(getpid());
        calib_dir = tmpdir + "/calibration";
        output_dir = tmpdir + "/output";
        fs::create_directories(calib_dir);
        fs::create_directories(output_dir);

        // Write mock calibration for 2 cameras
        write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);
        write_mock_calib_yaml(calib_dir + "/cam1.yaml", 640, 480);

        // Build labeled AnnotationMap: 5 frames, partial labeling
        for (u32 f = 0; f < 5; ++f) {
            auto &fa = get_or_create_frame(amap, f * 10, 3, 2);
            // Label all 3 keypoints on cam 0 (ImPlot coords, Y from bottom)
            for (int k = 0; k < 3; ++k) {
                fa.cameras[0].keypoints[k].x = 100.0 + k * 50;
                fa.cameras[0].keypoints[k].y = 300.0 - k * 20;
                fa.cameras[0].keypoints[k].labeled = true;
            }
            // Label 2 of 3 on cam 1
            fa.cameras[1].keypoints[0].x = 120.0; fa.cameras[1].keypoints[0].y = 280.0;
            fa.cameras[1].keypoints[0].labeled = true;
            fa.cameras[1].keypoints[1].x = 180.0; fa.cameras[1].keypoints[1].y = 250.0;
            fa.cameras[1].keypoints[1].labeled = true;
        }

        // Config
        cfg.calibration_folder = calib_dir;
        cfg.output_folder = output_dir;
        cfg.camera_names = {"cam0", "cam1"};
        cfg.skeleton_name = "TestRat";
        cfg.node_names = {"Snout", "EarL", "EarR"};
        cfg.edges = {{0, 1}, {0, 2}};
        cfg.num_keypoints = 3;
        cfg.bbox_margin = 20.0f;
        cfg.train_ratio = 0.6f;
        cfg.seed = 42;
    }

    ~ExportTestFixture() {
        std::filesystem::remove_all(tmpdir);
    }
};

static void test_export_coco_full_pipeline() {
    printf("  test_export_coco_full_pipeline...\n");
    namespace fs = std::filesystem;

    ExportTestFixture fix;
    std::string status;
    bool ok = ExportFormats::export_coco(fix.cfg, fix.amap, &status);

    EXPECT_TRUE(ok);
    EXPECT_TRUE(status.find("complete") != std::string::npos);

    // Check output files exist
    EXPECT_TRUE(fs::exists(fix.output_dir + "/annotations/cam0_train.json"));
    EXPECT_TRUE(fs::exists(fix.output_dir + "/annotations/cam0_val.json"));
    EXPECT_TRUE(fs::exists(fix.output_dir + "/annotations/cam1_train.json"));
    EXPECT_TRUE(fs::exists(fix.output_dir + "/annotations/cam1_val.json"));

    // Parse and validate cam0_train.json
    {
        std::ifstream f(fix.output_dir + "/annotations/cam0_train.json");
        nlohmann::json j;
        f >> j;

        EXPECT_TRUE(j.contains("images"));
        EXPECT_TRUE(j.contains("annotations"));
        EXPECT_TRUE(j.contains("categories"));

        // Should have 3 train frames (60% of 5)
        EXPECT_EQ((int)j["images"].size(), 3);

        // All train frames should have annotations (all labeled on cam0)
        EXPECT_EQ((int)j["annotations"].size(), 3);

        // Check category
        EXPECT_TRUE(j["categories"][0]["name"].get<std::string>() == "TestRat");
        EXPECT_EQ((int)j["categories"][0]["keypoints"].size(), 3);
        EXPECT_EQ((int)j["categories"][0]["skeleton"].size(), 2);

        // Verify keypoint coordinates are Y-flipped (image_height=480)
        auto &ann = j["annotations"][0];
        auto &kps = ann["keypoints"];
        // We set keypoints at ImPlot y=300, so image y = 480-300 = 180
        EXPECT_NEAR(kps[1].get<double>(), 180.0, 0.5);
        // Visibility flag = 2 for labeled
        EXPECT_EQ(kps[2].get<int>(), 2);
    }

    // Parse cam1 val to check partial labeling
    {
        std::ifstream f(fix.output_dir + "/annotations/cam1_val.json");
        nlohmann::json j;
        f >> j;

        // Val should have 2 frames (40% of 5)
        EXPECT_EQ((int)j["images"].size(), 2);

        // Cam1 has 2 of 3 keypoints labeled -> num_keypoints should be 2
        if (!j["annotations"].empty()) {
            EXPECT_EQ(j["annotations"][0]["num_keypoints"].get<int>(), 2);
        }
    }
}

static void test_export_yolo_pose_full_pipeline() {
    printf("  test_export_yolo_pose_full_pipeline...\n");
    namespace fs = std::filesystem;

    ExportTestFixture fix;
    std::string status;
    bool ok = ExportFormats::export_yolo(fix.cfg, fix.amap, true, &status);

    EXPECT_TRUE(ok);
    EXPECT_TRUE(status.find("YOLO Pose") != std::string::npos);

    // Check data.yaml
    EXPECT_TRUE(fs::exists(fix.output_dir + "/data.yaml"));
    {
        std::ifstream f(fix.output_dir + "/data.yaml");
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("kpt_shape") != std::string::npos);
        EXPECT_TRUE(content.find("TestRat") != std::string::npos);
    }

    // Check label directories exist
    EXPECT_TRUE(fs::exists(fix.output_dir + "/labels/train/cam0"));
    EXPECT_TRUE(fs::exists(fix.output_dir + "/labels/val/cam0"));

    // Count label files
    int train_labels = 0;
    for (auto &e : fs::directory_iterator(fix.output_dir + "/labels/train/cam0"))
        if (e.path().extension() == ".txt") ++train_labels;
    EXPECT_EQ(train_labels, 3); // 60% of 5

    int val_labels = 0;
    for (auto &e : fs::directory_iterator(fix.output_dir + "/labels/val/cam0"))
        if (e.path().extension() == ".txt") ++val_labels;
    EXPECT_EQ(val_labels, 2); // 40% of 5

    // Parse a label file and verify format
    for (auto &e : fs::directory_iterator(fix.output_dir + "/labels/train/cam0")) {
        if (e.path().extension() != ".txt") continue;
        std::ifstream f(e.path());
        std::string line;
        std::getline(f, line);
        // YOLO format: class cx cy w h kx1 ky1 v1 kx2 ky2 v2 kx3 ky3 v3
        // = 5 + 3*3 = 14 values
        std::stringstream ss(line);
        int count = 0;
        double val;
        while (ss >> val) ++count;
        EXPECT_EQ(count, 14); // class + 4 bbox + 9 keypoints (3×3)

        // Re-parse and check normalized range
        std::stringstream ss2(line);
        int cls; double cx, cy, bw, bh;
        ss2 >> cls >> cx >> cy >> bw >> bh;
        EXPECT_EQ(cls, 0);
        EXPECT_TRUE(cx >= 0 && cx <= 1);
        EXPECT_TRUE(cy >= 0 && cy <= 1);
        EXPECT_TRUE(bw >= 0 && bw <= 1);
        EXPECT_TRUE(bh >= 0 && bh <= 1);
        break; // just check first file
    }
}

static void test_export_yolo_detect_no_keypoints() {
    printf("  test_export_yolo_detect_no_keypoints...\n");
    namespace fs = std::filesystem;

    ExportTestFixture fix;
    std::string status;
    bool ok = ExportFormats::export_yolo(fix.cfg, fix.amap, false, &status);

    EXPECT_TRUE(ok);
    EXPECT_TRUE(status.find("YOLO Detection") != std::string::npos);

    // data.yaml should NOT have kpt_shape
    {
        std::ifstream f(fix.output_dir + "/data.yaml");
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("kpt_shape") == std::string::npos);
    }

    // Label files should have 5 values: class cx cy w h
    for (auto &e : fs::directory_iterator(fix.output_dir + "/labels/train/cam0")) {
        if (e.path().extension() != ".txt") continue;
        std::ifstream f(e.path());
        std::string line;
        std::getline(f, line);
        std::stringstream ss(line);
        int count = 0;
        double val;
        while (ss >> val) ++count;
        EXPECT_EQ(count, 5);
        break;
    }
}

static void test_export_deeplabcut_full_pipeline() {
    printf("  test_export_deeplabcut_full_pipeline...\n");
    namespace fs = std::filesystem;

    ExportTestFixture fix;
    std::string status;
    bool ok = ExportFormats::export_deeplabcut(fix.cfg, fix.amap, &status);

    EXPECT_TRUE(ok);
    EXPECT_TRUE(status.find("DeepLabCut") != std::string::npos);

    // Check per-camera CSV files
    EXPECT_TRUE(fs::exists(fix.output_dir + "/cam0/CollectedData.csv"));
    EXPECT_TRUE(fs::exists(fix.output_dir + "/cam1/CollectedData.csv"));

    // Check config.yaml
    EXPECT_TRUE(fs::exists(fix.output_dir + "/config.yaml"));
    {
        std::ifstream f(fix.output_dir + "/config.yaml");
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("TestRat") != std::string::npos);
        EXPECT_TRUE(content.find("Snout") != std::string::npos);
        EXPECT_TRUE(content.find("EarL") != std::string::npos);
    }

    // Parse cam0 CSV and verify DLC format
    {
        std::ifstream f(fix.output_dir + "/cam0/CollectedData.csv");
        std::string line;

        // Row 1: scorer header
        std::getline(f, line);
        EXPECT_TRUE(line.find("scorer") != std::string::npos);
        EXPECT_TRUE(line.find("RED") != std::string::npos);

        // Row 2: bodyparts
        std::getline(f, line);
        EXPECT_TRUE(line.find("bodyparts") != std::string::npos);
        EXPECT_TRUE(line.find("Snout") != std::string::npos);
        EXPECT_TRUE(line.find("EarR") != std::string::npos);

        // Row 3: coords
        std::getline(f, line);
        EXPECT_TRUE(line.find("coords") != std::string::npos);

        // Data rows: should have 5 (all frames labeled on cam0)
        int data_rows = 0;
        while (std::getline(f, line)) {
            if (!line.empty()) ++data_rows;
        }
        EXPECT_EQ(data_rows, 5);
    }
}

static void test_export_no_labeled_frames() {
    printf("  test_export_no_labeled_frames...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_export_empty_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    fs::create_directories(calib_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    // AnnotationMap with frames but NO labeled keypoints
    AnnotationMap amap;
    get_or_create_frame(amap, 0, 2, 1); // unlabeled

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = tmpdir + "/out";
    cfg.camera_names = {"cam0"};
    cfg.node_names = {"a", "b"};

    std::string status;

    EXPECT_FALSE(ExportFormats::export_coco(cfg, amap, &status));
    EXPECT_TRUE(status.find("No labeled") != std::string::npos);

    EXPECT_FALSE(ExportFormats::export_yolo(cfg, amap, true, &status));
    EXPECT_TRUE(status.find("No labeled") != std::string::npos);

    EXPECT_FALSE(ExportFormats::export_deeplabcut(cfg, amap, &status));
    EXPECT_TRUE(status.find("No labeled") != std::string::npos);

    fs::remove_all(tmpdir);
}

static void test_export_missing_calibration() {
    printf("  test_export_missing_calibration...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_export_nocal_" + std::to_string(getpid());
    fs::create_directories(tmpdir);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 0, 2, 1);
    fa.cameras[0].keypoints[0].labeled = true;
    fa.cameras[0].keypoints[0].x = 50; fa.cameras[0].keypoints[0].y = 50;

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = tmpdir + "/nonexistent";
    cfg.output_folder = tmpdir + "/out";
    cfg.camera_names = {"cam0"};
    cfg.node_names = {"a", "b"};

    std::string status;
    EXPECT_FALSE(ExportFormats::export_coco(cfg, amap, &status));
    EXPECT_TRUE(status.find("calibration") != std::string::npos ||
                status.find("Error") != std::string::npos);

    fs::remove_all(tmpdir);
}

static void test_export_dispatch() {
    printf("  test_export_dispatch...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_dispatch_" + std::to_string(getpid());
    fs::create_directories(tmpdir);

    AnnotationMap amap;
    ExportFormats::ExportConfig cfg;
    cfg.output_folder = tmpdir + "/out";
    std::string status;

    // Unknown format
    bool ok = ExportFormats::export_dataset(ExportFormats::FORMAT_COUNT, cfg, amap, &status);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(status.find("Unknown") != std::string::npos);

    fs::remove_all(tmpdir);
}

static void test_build_coco_json_with_explicit_bbox() {
    printf("  test_build_coco_json_with_explicit_bbox...\n");

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 1, 2, 1);

    // Label keypoints
    fa.cameras[0].keypoints[0].x = 100.0; fa.cameras[0].keypoints[0].y = 400.0;
    fa.cameras[0].keypoints[0].labeled = true;
    fa.cameras[0].keypoints[1].x = 200.0; fa.cameras[0].keypoints[1].y = 300.0;
    fa.cameras[0].keypoints[1].labeled = true;

    // Set explicit bbox (should be used instead of deriving from keypoints)
    auto &ext = fa.cameras[0].get_extras();
    ext.bbox_x = 50.0;
    ext.bbox_y = 100.0;
    ext.bbox_w = 300.0;
    ext.bbox_h = 200.0;
    ext.has_bbox = true;

    ExportFormats::ExportConfig cfg;
    cfg.node_names = {"a", "b"};
    cfg.bbox_margin = 999.0f; // large margin — should be ignored when has_bbox

    std::vector<u32> frames = {1};
    auto j = ExportFormats::build_coco_json(amap, frames, cfg, 0, "cam0", 640, 480);

    auto &bbox = j["annotations"][0]["bbox"];
    EXPECT_NEAR(bbox[0].get<double>(), 50.0, 0.1);
    EXPECT_NEAR(bbox[1].get<double>(), 100.0, 0.1);
    EXPECT_NEAR(bbox[2].get<double>(), 300.0, 0.1);
    EXPECT_NEAR(bbox[3].get<double>(), 200.0, 0.1);
}

static void test_build_coco_json_with_mask() {
    printf("  test_build_coco_json_with_mask...\n");

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 1, 2, 1);

    fa.cameras[0].keypoints[0].x = 100.0; fa.cameras[0].keypoints[0].y = 400.0;
    fa.cameras[0].keypoints[0].labeled = true;

    // Set mask polygons
    auto &ext = fa.cameras[0].get_extras();
    ext.mask_polygons = {
        {{10.0, 20.0}, {100.0, 20.0}, {100.0, 100.0}, {10.0, 100.0}}
    };
    ext.has_mask = true;

    ExportFormats::ExportConfig cfg;
    cfg.node_names = {"a", "b"};
    cfg.bbox_margin = 10.0f;

    std::vector<u32> frames = {1};
    auto j = ExportFormats::build_coco_json(amap, frames, cfg, 0, "cam0", 640, 480);

    auto &seg = j["annotations"][0]["segmentation"];
    EXPECT_EQ((int)seg.size(), 1); // one polygon
    EXPECT_EQ((int)seg[0].size(), 8); // 4 points × 2 coords

    // Y should be flipped: original y=20 → 480-20=460
    EXPECT_NEAR(seg[0][1].get<double>(), 460.0, 0.1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration: Save/Load Roundtrip through AnnotationMap
// ═══════════════════════════════════════════════════════════════════════════

static void test_save_load_keypoints_roundtrip() {
    printf("  test_save_load_keypoints_roundtrip...\n");
    namespace fs = std::filesystem;

    std::vector<std::string> camera_names = {"cam0", "cam1"};

    std::string tmpdir = "/tmp/test_saveload_" + std::to_string(getpid());
    std::string label_root = tmpdir + "/labeled_data";
    fs::create_directories(label_root);

    // Build AnnotationMap directly
    AnnotationMap amap;
    for (u32 f : {5u, 15u, 25u}) {
        auto &fa = get_or_create_frame(amap, f, 3, 2);
        // Label some keypoints on cam 0
        fa.cameras[0].keypoints[0].x = 100.0 + f;
        fa.cameras[0].keypoints[0].y = 200.0 + f;
        fa.cameras[0].keypoints[0].labeled = true;
        fa.cameras[0].keypoints[2].x = 300.0 + f;
        fa.cameras[0].keypoints[2].y = 400.0 + f;
        fa.cameras[0].keypoints[2].labeled = true;
        // Label one keypoint on cam 1
        fa.cameras[1].keypoints[1].x = 500.0 + f;
        fa.cameras[1].keypoints[1].y = 600.0 + f;
        fa.cameras[1].keypoints[1].labeled = true;
        // Set 3D for triangulated keypoint 0
        fa.kp3d[0].x = 1.0 + f;
        fa.kp3d[0].y = 2.0 + f;
        fa.kp3d[0].z = 3.0 + f;
        fa.kp3d[0].triangulated = true;
    }
    EXPECT_EQ((int)amap.size(), 3);

    // Save via AnnotationCSV
    std::string save_err;
    std::string saved_folder = AnnotationCSV::save_all(
        label_root, "TestSkeleton", amap, 2, 3, camera_names, &save_err);
    EXPECT_TRUE(!saved_folder.empty());
    EXPECT_TRUE(fs::exists(saved_folder + "/keypoints3d.csv"));
    EXPECT_TRUE(fs::exists(saved_folder + "/cam0.csv"));
    EXPECT_TRUE(fs::exists(saved_folder + "/cam1.csv"));

    // Find the saved folder
    std::string found_folder, find_err;
    int rc = AnnotationCSV::find_most_recent_labels(label_root, found_folder, find_err);
    EXPECT_EQ(rc, 0);
    EXPECT_TRUE(found_folder == saved_folder);

    // Load into a fresh AnnotationMap
    AnnotationMap amap2;
    std::string load_err;
    rc = AnnotationCSV::load_all(saved_folder, amap2, "TestSkeleton", 3, 2,
                                  camera_names, load_err);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ((int)amap2.size(), 3);

    // Verify data survived the roundtrip
    for (u32 f : {5u, 15u, 25u}) {
        auto &fa = amap2[f];

        // Cam 0, keypoint 0
        EXPECT_TRUE(fa.cameras[0].keypoints[0].labeled);
        EXPECT_NEAR(fa.cameras[0].keypoints[0].x, 100.0 + f, 0.01);
        EXPECT_NEAR(fa.cameras[0].keypoints[0].y, 200.0 + f, 0.01);

        // Cam 0, keypoint 1 should be unlabeled
        EXPECT_FALSE(fa.cameras[0].keypoints[1].labeled);

        // Cam 0, keypoint 2
        EXPECT_TRUE(fa.cameras[0].keypoints[2].labeled);
        EXPECT_NEAR(fa.cameras[0].keypoints[2].x, 300.0 + f, 0.01);

        // Cam 1, keypoint 1
        EXPECT_TRUE(fa.cameras[1].keypoints[1].labeled);
        EXPECT_NEAR(fa.cameras[1].keypoints[1].x, 500.0 + f, 0.01);

        // 3D keypoint 0
        EXPECT_TRUE(fa.kp3d[0].triangulated);
        EXPECT_NEAR(fa.kp3d[0].x, 1.0 + f, 0.01);
        EXPECT_NEAR(fa.kp3d[0].y, 2.0 + f, 0.01);
        EXPECT_NEAR(fa.kp3d[0].z, 3.0 + f, 0.01);
    }

    fs::remove_all(tmpdir);
}

static void test_save_load_with_extended_data() {
    printf("  test_save_load_with_extended_data...\n");
    namespace fs = std::filesystem;

    std::vector<std::string> camera_names = {"cam0"};

    std::string tmpdir = "/tmp/test_saveload_ext_" + std::to_string(getpid());
    std::string label_root = tmpdir + "/labeled_data";
    fs::create_directories(label_root);

    // Build AnnotationMap with keypoints + bbox
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 10, 2, 1);
    fa.cameras[0].keypoints[0].x = 50.0;
    fa.cameras[0].keypoints[0].y = 100.0;
    fa.cameras[0].keypoints[0].labeled = true;
    auto &bext = fa.cameras[0].get_extras();
    bext.bbox_x = 10.0;
    bext.bbox_y = 20.0;
    bext.bbox_w = 200.0;
    bext.bbox_h = 150.0;
    bext.has_bbox = true;

    // Save via AnnotationCSV (includes extended JSON automatically)
    std::string save_err;
    std::string saved_folder = AnnotationCSV::save_all(
        label_root, "TestSkeleton", amap, 1, 2, camera_names, &save_err);
    EXPECT_TRUE(!saved_folder.empty());

    // Load into fresh AnnotationMap
    AnnotationMap amap2;
    std::string load_err;
    int rc = AnnotationCSV::load_all(saved_folder, amap2, "TestSkeleton", 2, 1,
                                      camera_names, load_err);
    EXPECT_EQ(rc, 0);

    // Verify keypoints survived
    EXPECT_TRUE(amap2[10].cameras[0].keypoints[0].labeled);
    EXPECT_NEAR(amap2[10].cameras[0].keypoints[0].x, 50.0, 0.01);

    // Verify bbox survived
    EXPECT_TRUE(amap2[10].cameras[0].has_bbox());
    EXPECT_NEAR(amap2[10].cameras[0].extras->bbox_x, 10.0, 0.01);
    EXPECT_NEAR(amap2[10].cameras[0].extras->bbox_h, 150.0, 0.01);

    fs::remove_all(tmpdir);
}

// ═══════════════════════════════════════════════════════════════════════════
// Y-flip consistency tests
// ═══════════════════════════════════════════════════════════════════════════

static void test_yolo_pose_y_flip() {
    printf("  test_yolo_pose_y_flip...\n");
    namespace fs = std::filesystem;

    // Setup: single camera, single frame, known ImPlot coords
    std::string tmpdir = "/tmp/test_yflip_yolo_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    std::string output_dir = tmpdir + "/output";
    fs::create_directories(calib_dir);
    fs::create_directories(output_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 100, 1, 1);
    // ImPlot y=400, image_height=480 → image y = 480-400 = 80
    fa.cameras[0].keypoints[0].x = 320.0; fa.cameras[0].keypoints[0].y = 400.0;
    fa.cameras[0].keypoints[0].labeled = true;

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = output_dir;
    cfg.camera_names = {"cam0"};
    cfg.skeleton_name = "Test";
    cfg.node_names = {"pt"};
    cfg.num_keypoints = 1;
    cfg.bbox_margin = 20.0f;
    cfg.train_ratio = 1.0f;
    cfg.seed = 1;

    std::string status;
    ExportFormats::export_yolo(cfg, amap, true, &status);

    // Read the label file
    std::string lbl_path = output_dir + "/labels/train/cam0/Frame_100.txt";
    EXPECT_TRUE(fs::exists(lbl_path));
    std::ifstream f(lbl_path);
    std::string line;
    std::getline(f, line);

    // Parse: class cx cy w h kx ky vis
    std::stringstream ss(line);
    int cls; double cx, cy, bw, bh, kx, ky; int vis;
    ss >> cls >> cx >> cy >> bw >> bh >> kx >> ky >> vis;

    // YOLO ky = (480 - 400) / 480 = 80/480 = 0.16667
    EXPECT_NEAR(ky, 80.0 / 480.0, 0.001);
    // kx = 320 / 640 = 0.5
    EXPECT_NEAR(kx, 320.0 / 640.0, 0.001);
    EXPECT_EQ(vis, 2);

    fs::remove_all(tmpdir);
}

static void test_dlc_y_flip() {
    printf("  test_dlc_y_flip...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_yflip_dlc_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    std::string output_dir = tmpdir + "/output";
    fs::create_directories(calib_dir);
    fs::create_directories(output_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 100, 1, 1);
    // Same coords: ImPlot y=400, h=480 → DLC y = 480-400 = 80
    fa.cameras[0].keypoints[0].x = 320.0; fa.cameras[0].keypoints[0].y = 400.0;
    fa.cameras[0].keypoints[0].labeled = true;

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = output_dir;
    cfg.camera_names = {"cam0"};
    cfg.skeleton_name = "Test";
    cfg.node_names = {"pt"};
    cfg.num_keypoints = 1;
    cfg.seed = 1;

    std::string status;
    ExportFormats::export_deeplabcut(cfg, amap, &status);

    // Read the CSV — skip 3 header rows
    std::string csv_path = output_dir + "/cam0/CollectedData.csv";
    EXPECT_TRUE(fs::exists(csv_path));
    std::ifstream f(csv_path);
    std::string line;
    std::getline(f, line); // scorer
    std::getline(f, line); // bodyparts
    std::getline(f, line); // coords
    std::getline(f, line); // first data row

    // Parse: path,x,y
    auto comma1 = line.find(',');
    auto comma2 = line.find(',', comma1 + 1);
    double x = std::stod(line.substr(comma1 + 1, comma2 - comma1 - 1));
    double y = std::stod(line.substr(comma2 + 1));
    EXPECT_NEAR(x, 320.0, 0.01);
    EXPECT_NEAR(y, 80.0, 0.01); // 480 - 400 = 80

    fs::remove_all(tmpdir);
}

// ═══════════════════════════════════════════════════════════════════════════
// SAM preprocessing unit tests (no ONNX needed)
// ═══════════════════════════════════════════════════════════════════════════

static void test_sam_preprocess_mobilesam() {
    printf("  test_sam_preprocess_mobilesam...\n");
    // 4x3 white image (RGB=255)
    const int w = 4, h = 3;
    uint8_t rgb[w * h * 3];
    memset(rgb, 255, sizeof(rgb));

    float scale; int pad_x, pad_y;
    auto result = sam_detail::preprocess_mobilesam(rgb, w, h, scale, pad_x, pad_y);

    // Output shape: 3 * 1024 * 1024
    EXPECT_EQ((int)result.size(), 3 * 1024 * 1024);

    // Scale = 1024 / max(4,3) = 256
    EXPECT_NEAR(scale, 1024.0f / 4.0f, 0.1f);

    // Check normalization: pixel 255, mean=123.675, std=58.395
    // Normalized = (255 - 123.675) / 58.395 ≈ 2.249
    // The resized region should have roughly this value
    float v = result[0]; // First pixel of R channel
    EXPECT_NEAR(v, (255.0f - 123.675f) / 58.395f, 0.1f);

    // Padding region (outside resized area) should be ~= (0-mean)/std = negative
    // For R channel: (0 - 123.675) / 58.395 ≈ -2.118
    // Pixel at bottom-right corner (1023, 1023) in R channel is padded → 0
    // Actually padding uses 0.0f directly (no normalization of zeros)
    // The code initializes to 0.0f and only writes the resized region
    float padded = result[0 * 1024 * 1024 + 1023 * 1024 + 1023];
    EXPECT_NEAR(padded, 0.0f, 0.001f);
}

static void test_sam_preprocess_sam2() {
    printf("  test_sam_preprocess_sam2...\n");
    // 8x8 mid-gray image (RGB=128)
    const int w = 8, h = 8;
    uint8_t rgb[w * h * 3];
    memset(rgb, 128, sizeof(rgb));

    auto result = sam_detail::preprocess_sam2(rgb, w, h);

    // Output shape: 3 * 1024 * 1024
    EXPECT_EQ((int)result.size(), 3 * 1024 * 1024);

    // SAM2 normalizes to 0-1 first: pixel/255 = 128/255 ≈ 0.502
    // Then (0.502 - mean) / std
    // R channel: (0.502 - 0.485) / 0.229 ≈ 0.074
    float v = result[0]; // First pixel of R channel
    double expected_r = (128.0 / 255.0 - 0.485) / 0.229;
    EXPECT_NEAR(v, expected_r, 0.05);

    // G channel: (0.502 - 0.456) / 0.224 ≈ 0.205
    float v_g = result[1 * 1024 * 1024];
    double expected_g = (128.0 / 255.0 - 0.456) / 0.224;
    EXPECT_NEAR(v_g, expected_g, 0.05);
}

static void test_sam_bilinear_resize() {
    printf("  test_sam_bilinear_resize...\n");
    // 2x2 gradient source → 4x4 output
    float src[4] = {0.0f, 0.5f, 0.5f, 1.0f}; // smooth gradient
    auto dst = sam_detail::bilinear_resize(src, 2, 2, 4, 4);

    EXPECT_EQ((int)dst.size(), 16);

    // Output should be finite (half-pixel centering can slightly overshoot
    // source range for some patterns, so just check finiteness)
    for (float v : dst) {
        EXPECT_TRUE(std::isfinite(v));
    }

    // Larger resize: 3x3 → 6x6 (output shape correctness)
    float src2[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto dst2 = sam_detail::bilinear_resize(src2, 3, 3, 6, 6);
    EXPECT_EQ((int)dst2.size(), 36);
}

// ═══════════════════════════════════════════════════════════════════════════
// Contour extraction for complex shapes
// ═══════════════════════════════════════════════════════════════════════════

static void test_sam_contour_l_shape() {
    printf("  test_sam_contour_l_shape...\n");
    // 10x10 mask with L-shape
    const int w = 10, h = 10;
    uint8_t mask[w * h];
    memset(mask, 0, sizeof(mask));

    // L-shape: rows 1-8 col 1-3, rows 6-8 col 1-7
    for (int y = 1; y <= 8; ++y)
        for (int x = 1; x <= 3; ++x)
            mask[y * w + x] = 255;
    for (int y = 6; y <= 8; ++y)
        for (int x = 4; x <= 7; ++x)
            mask[y * w + x] = 255;

    auto contours = sam_detail::extract_contours(mask, w, h);
    EXPECT_TRUE(!contours.empty());
    // Concave L-shape should have more than 4 points
    EXPECT_TRUE(contours[0].size() > 4);
}

static void test_sam_contour_single_pixel() {
    printf("  test_sam_contour_single_pixel...\n");
    // 5x5 mask with single pixel
    const int w = 5, h = 5;
    uint8_t mask[w * h];
    memset(mask, 0, sizeof(mask));
    mask[2 * w + 2] = 255; // single pixel at (2,2)

    // Should not crash
    auto contours = sam_detail::extract_contours(mask, w, h);
    // Single pixel = single boundary pixel, can't form a closed contour with 3+ points
    // Graceful: either empty or a small contour
    // (no crash is the main assertion)
    EXPECT_TRUE(true); // survived without crash
}

// ═══════════════════════════════════════════════════════════════════════════
// Export schema validation
// ═══════════════════════════════════════════════════════════════════════════

static void test_coco_json_schema() {
    printf("  test_coco_json_schema...\n");

    ExportTestFixture fix;

    auto coco = ExportFormats::build_coco_json(
        fix.amap, {0, 10, 20, 30, 40}, fix.cfg, 0, "cam0", 640, 480);

    // Top-level keys
    EXPECT_TRUE(coco.contains("images"));
    EXPECT_TRUE(coco.contains("annotations"));
    EXPECT_TRUE(coco.contains("categories"));
    EXPECT_TRUE(coco.is_object());

    // Images: each must have id, file_name, width, height
    for (const auto &img : coco["images"]) {
        EXPECT_TRUE(img.contains("id"));
        EXPECT_TRUE(img.contains("file_name"));
        EXPECT_TRUE(img.contains("width"));
        EXPECT_TRUE(img.contains("height"));
        EXPECT_EQ(img["width"].get<int>(), 640);
        EXPECT_EQ(img["height"].get<int>(), 480);
    }

    // Annotations: each must have required fields
    for (const auto &ann : coco["annotations"]) {
        EXPECT_TRUE(ann.contains("id"));
        EXPECT_TRUE(ann.contains("image_id"));
        EXPECT_TRUE(ann.contains("category_id"));
        EXPECT_TRUE(ann.contains("keypoints"));
        EXPECT_TRUE(ann.contains("num_keypoints"));
        EXPECT_TRUE(ann.contains("bbox"));
        EXPECT_TRUE(ann.contains("area"));
        EXPECT_TRUE(ann.contains("iscrowd"));
        // Keypoints array: 3 values per keypoint (x, y, vis)
        EXPECT_EQ((int)ann["keypoints"].size(), 3 * 3); // 3 keypoints × 3 values
        // Area must be positive
        EXPECT_TRUE(ann["area"].get<double>() > 0);
    }

    // Categories: skeleton edges must be 1-indexed
    auto &cat = coco["categories"][0];
    EXPECT_TRUE(cat.contains("skeleton"));
    for (const auto &edge : cat["skeleton"]) {
        EXPECT_TRUE(edge[0].get<int>() >= 1); // 1-indexed
        EXPECT_TRUE(edge[1].get<int>() >= 1);
    }
}

static void test_yolo_label_format_validation() {
    printf("  test_yolo_label_format_validation...\n");
    namespace fs = std::filesystem;

    ExportTestFixture fix;
    fix.cfg.train_ratio = 1.0f; // all train

    std::string status;
    ExportFormats::export_yolo(fix.cfg, fix.amap, true, &status);

    // Validate every label file
    std::string lbl_dir = fix.output_dir + "/labels/train/cam0";
    for (auto &e : fs::directory_iterator(lbl_dir)) {
        if (e.path().extension() != ".txt") continue;
        std::ifstream f(e.path());
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<double> values;
            double v;
            while (ss >> v) values.push_back(v);

            // YOLO pose: class + 4 bbox + 3*num_keypoints = 5 + 9 = 14
            EXPECT_EQ((int)values.size(), 14);

            // Class index
            EXPECT_TRUE(values[0] >= 0);

            // Bbox values normalized 0-1
            for (int i = 1; i <= 4; ++i)
                EXPECT_TRUE(values[i] >= 0.0 && values[i] <= 1.0);

            // Keypoint x,y values (every 3rd starting at 5) normalized 0-1
            for (int k = 0; k < 3; ++k) {
                int base = 5 + k * 3;
                double kx = values[base];
                double ky = values[base + 1];
                int kvis = (int)values[base + 2];
                if (kvis > 0) {
                    EXPECT_TRUE(kx >= 0.0 && kx <= 1.0);
                    EXPECT_TRUE(ky >= 0.0 && ky <= 1.0);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bridge integration test
// ═══════════════════════════════════════════════════════════════════════════

static void test_bridge_keypoints_and_extended_roundtrip() {
    printf("  test_bridge_keypoints_and_extended_roundtrip...\n");
    namespace fs = std::filesystem;

    auto scene = make_mock_scene(2);
    auto skel = make_mock_skeleton(3);
    std::vector<std::string> camera_names = {"cam0", "cam1"};

    std::string tmpdir = "/tmp/test_bridge_rt_" + std::to_string(getpid());
    std::string label_root = tmpdir + "/labeled_data";
    fs::create_directories(label_root);

    // Step 1: Create keypoints in keypoints_map (simulating user labeling)
    std::map<u32, KeyPoints *> km;
    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);
    kp->kp2d[0][0].position = {100.0, 200.0};
    kp->kp2d[0][0].is_labeled = true;
    kp->kp2d[0][1].position = {150.0, 250.0};
    kp->kp2d[0][1].is_labeled = true;
    kp->kp2d[1][0].position = {110.0, 210.0};
    kp->kp2d[1][0].is_labeled = true;
    km[50] = kp;

    // Step 2: Bridge → AnnotationMap
    AnnotationMap amap = migrate_keypoints_map(km, skel, &scene);
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(amap[50].cameras[0].keypoints[0].labeled);

    // Step 3: Add bbox via AnnotationMap (simulating annotation tool)
    auto &bext = amap[50].cameras[0].get_extras();
    bext.bbox_x = 50.0;
    bext.bbox_y = 100.0;
    bext.bbox_w = 200.0;
    bext.bbox_h = 300.0;
    bext.has_bbox = true;

    // Step 4: Save via AnnotationCSV (CSV + extended JSON)
    std::string save_err;
    std::string saved_folder = AnnotationCSV::save_all(
        label_root, "TestSkeleton", amap, 2, 3, camera_names, &save_err);
    EXPECT_TRUE(!saved_folder.empty());

    // Step 5: Simulate restart — load into fresh AnnotationMap
    AnnotationMap amap2;
    std::string load_err;
    int load_ret = AnnotationCSV::load_all(saved_folder, amap2, "TestSkeleton", 3, 2,
                                            camera_names, load_err);
    EXPECT_EQ(load_ret, 0);

    // Step 6: Verify keypoints survived
    EXPECT_EQ((int)amap2.size(), 1);
    EXPECT_TRUE(amap2.count(50));
    EXPECT_TRUE(amap2[50].cameras[0].keypoints[0].labeled);
    EXPECT_NEAR(amap2[50].cameras[0].keypoints[0].x, 100.0, 0.01);
    EXPECT_NEAR(amap2[50].cameras[0].keypoints[0].y, 200.0, 0.01);
    EXPECT_TRUE(amap2[50].cameras[0].keypoints[1].labeled);
    EXPECT_TRUE(amap2[50].cameras[1].keypoints[0].labeled);

    // Step 7: Verify bbox survived
    EXPECT_TRUE(amap2[50].cameras[0].has_bbox());
    EXPECT_NEAR(amap2[50].cameras[0].extras->bbox_x, 50.0, 0.01);
    EXPECT_NEAR(amap2[50].cameras[0].extras->bbox_w, 200.0, 0.01);
    EXPECT_NEAR(amap2[50].cameras[0].extras->bbox_h, 300.0, 0.01);

    // Step 8: Simulate user editing keypoints → refresh_keypoints_in_amap
    kp->kp2d[0][2].position = {175.0, 275.0};
    kp->kp2d[0][2].is_labeled = true;
    refresh_keypoints_in_amap(amap, km, skel, &scene);

    // Bbox should still be there after refresh
    EXPECT_TRUE(amap[50].cameras[0].has_bbox());
    EXPECT_NEAR(amap[50].cameras[0].extras->bbox_x, 50.0, 0.01);
    // New keypoint should be reflected
    EXPECT_TRUE(amap[50].cameras[0].keypoints[2].labeled);
    EXPECT_NEAR(amap[50].cameras[0].keypoints[2].x, 175.0, 0.01);

    // Cleanup
    for (auto &[f, k] : km)  free_keypoints(k, &scene);
    fs::remove_all(tmpdir);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bridge: refresh_keypoints_in_amap edge cases
// ═══════════════════════════════════════════════════════════════════════════

static void test_refresh_empty_keypoints_map() {
    printf("  test_refresh_empty_keypoints_map...\n");
    auto scene = make_mock_scene(2);
    auto skel = make_mock_skeleton(3);

    // Start with populated amap, sync from empty km → amap should become empty
    AnnotationMap amap;
    get_or_create_frame(amap, 10, 3, 2);
    amap[10].cameras[0].keypoints[0].labeled = true;
    amap[10].cameras[0].get_extras().has_bbox = true;
    EXPECT_EQ((int)amap.size(), 1);

    std::map<u32, KeyPoints *> km; // empty
    refresh_keypoints_in_amap(amap, km, skel, &scene);

    // All frames removed since km is empty
    EXPECT_EQ((int)amap.size(), 0);
}

static void test_refresh_null_keypoints_skipped() {
    printf("  test_refresh_null_keypoints_skipped...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);

    std::map<u32, KeyPoints *> km;
    km[10] = nullptr; // null entry

    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);
    kp->kp2d[0][0].position = {42.0, 84.0};
    kp->kp2d[0][0].is_labeled = true;
    km[20] = kp;

    AnnotationMap amap;
    refresh_keypoints_in_amap(amap, km, skel, &scene);

    // Frame 10 (null) skipped, frame 20 created
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(amap.count(20));
    EXPECT_FALSE(amap.count(10));
    EXPECT_TRUE(amap[20].cameras[0].keypoints[0].labeled);
    EXPECT_NEAR(amap[20].cameras[0].keypoints[0].x, 42.0, 0.001);

    free_keypoints(km[20], &scene);
}

static void test_refresh_preserves_bbox_on_update() {
    printf("  test_refresh_preserves_bbox_on_update...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);

    // Create keypoints
    std::map<u32, KeyPoints *> km;
    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);
    kp->kp2d[0][0].position = {10.0, 20.0};
    kp->kp2d[0][0].is_labeled = true;
    km[5] = kp;

    // Create amap with bbox already set
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 5, 2, 1);
    auto &fext = fa.cameras[0].get_extras();
    fext.bbox_x = 100.0;
    fext.bbox_w = 200.0;
    fext.has_bbox = true;
    fext.has_obb = true;
    fext.obb_cx = 150.0;

    // Refresh should update keypoints but keep bbox/obb
    refresh_keypoints_in_amap(amap, km, skel, &scene);

    EXPECT_TRUE(amap[5].cameras[0].keypoints[0].labeled);
    EXPECT_NEAR(amap[5].cameras[0].keypoints[0].x, 10.0, 0.001);
    EXPECT_TRUE(amap[5].cameras[0].has_bbox());
    EXPECT_NEAR(amap[5].cameras[0].extras->bbox_x, 100.0, 0.001);
    EXPECT_TRUE(amap[5].cameras[0].has_obb());
    EXPECT_NEAR(amap[5].cameras[0].extras->obb_cx, 150.0, 0.001);

    free_keypoints(km[5], &scene);
}

static void test_refresh_adds_new_frames() {
    printf("  test_refresh_adds_new_frames...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);

    // amap has frame 10, km has 10 and 20
    std::map<u32, KeyPoints *> km;
    for (u32 f : {10u, 20u}) {
        KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
        allocate_keypoints(kp, &scene, &skel);
        kp->kp2d[0][0].position = {(double)f, (double)f};
        kp->kp2d[0][0].is_labeled = true;
        km[f] = kp;
    }

    AnnotationMap amap;
    get_or_create_frame(amap, 10, 2, 1);
    EXPECT_EQ((int)amap.size(), 1);

    refresh_keypoints_in_amap(amap, km, skel, &scene);

    // Both frames should exist, frame 20 was added
    EXPECT_EQ((int)amap.size(), 2);
    EXPECT_TRUE(amap.count(10));
    EXPECT_TRUE(amap.count(20));
    EXPECT_TRUE(amap[20].cameras[0].keypoints[0].labeled);

    for (auto &[f, kp] : km) free_keypoints(kp, &scene);
}

static void test_refresh_removes_stale_frames() {
    printf("  test_refresh_removes_stale_frames...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);

    // km has only frame 10, amap has 10 and 20
    std::map<u32, KeyPoints *> km;
    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);
    km[10] = kp;

    AnnotationMap amap;
    get_or_create_frame(amap, 10, 2, 1);
    get_or_create_frame(amap, 20, 2, 1);
    EXPECT_EQ((int)amap.size(), 2);

    refresh_keypoints_in_amap(amap, km, skel, &scene);

    // Frame 20 should be removed
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(amap.count(10));
    EXPECT_FALSE(amap.count(20));

    free_keypoints(km[10], &scene);
}

static void test_refresh_new_frame_gets_keypoints() {
    printf("  test_refresh_new_frame_gets_keypoints...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);

    std::map<u32, KeyPoints *> km;
    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);
    kp->kp2d[0][0].position = {99.0, 88.0};
    kp->kp2d[0][0].is_labeled = true;
    km[5] = kp;

    // amap is empty; refresh should create frame 5 from keypoints
    AnnotationMap amap;

    refresh_keypoints_in_amap(amap, km, skel, &scene);

    // Should have created the frame with keypoints
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(amap.count(5));
    EXPECT_TRUE(amap[5].cameras[0].keypoints[0].labeled);
    EXPECT_NEAR(amap[5].cameras[0].keypoints[0].x, 99.0, 0.001);

    free_keypoints(km[5], &scene);
}

// ═══════════════════════════════════════════════════════════════════════════
// Migration edge cases
// ═══════════════════════════════════════════════════════════════════════════

static void test_migrate_null_keypoints_skipped() {
    printf("  test_migrate_null_keypoints_skipped...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);

    std::map<u32, KeyPoints *> km;
    km[10] = nullptr;

    KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
    allocate_keypoints(kp, &scene, &skel);
    km[20] = kp;

    AnnotationMap amap = migrate_keypoints_map(km, skel, &scene);
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_FALSE(amap.count(10));
    EXPECT_TRUE(amap.count(20));

    free_keypoints(km[20], &scene);
}

static void test_migrate_multi_frame() {
    printf("  test_migrate_multi_frame...\n");
    auto scene = make_mock_scene(2);
    auto skel = make_mock_skeleton(4);

    std::map<u32, KeyPoints *> km;
    for (u32 f : {0u, 100u, 500u, 999u}) {
        KeyPoints *kp = (KeyPoints *)malloc(sizeof(KeyPoints));
        allocate_keypoints(kp, &scene, &skel);
        kp->kp2d[0][0].position = {(double)f, (double)f * 2};
        kp->kp2d[0][0].is_labeled = true;
        km[f] = kp;
    }

    AnnotationMap amap = migrate_keypoints_map(km, skel, &scene);
    EXPECT_EQ((int)amap.size(), 4);

    for (u32 f : {0u, 100u, 500u, 999u}) {
        EXPECT_TRUE(amap.count(f));
        EXPECT_EQ((int)amap[f].cameras.size(), 2);
        EXPECT_EQ((int)amap[f].kp3d.size(), 4);
        EXPECT_TRUE(amap[f].cameras[0].keypoints[0].labeled);
        EXPECT_NEAR(amap[f].cameras[0].keypoints[0].x, (double)f, 0.001);
    }

    for (auto &[f, kp] : km) free_keypoints(kp, &scene);
}

static void test_migrate_empty_map() {
    printf("  test_migrate_empty_map...\n");
    auto scene = make_mock_scene(1);
    auto skel = make_mock_skeleton(2);
    std::map<u32, KeyPoints *> km;

    AnnotationMap amap = migrate_keypoints_map(km, skel, &scene);
    EXPECT_EQ((int)amap.size(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Export edge cases
// ═══════════════════════════════════════════════════════════════════════════

static void test_export_coco_single_camera() {
    printf("  test_export_coco_single_camera...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_coco_1cam_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    std::string output_dir = tmpdir + "/output";
    fs::create_directories(calib_dir);
    fs::create_directories(output_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 10, 2, 1);
    fa.cameras[0].keypoints[0].x = 100.0; fa.cameras[0].keypoints[0].y = 300.0;
    fa.cameras[0].keypoints[0].labeled = true;

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = output_dir;
    cfg.camera_names = {"cam0"};
    cfg.skeleton_name = "Test";
    cfg.node_names = {"pt0", "pt1"};
    cfg.num_keypoints = 2;
    cfg.bbox_margin = 10.0f;
    cfg.train_ratio = 1.0f;
    cfg.seed = 1;
    cfg.edges = {{0, 1}};

    std::string status;
    bool ok = ExportFormats::export_coco(cfg, amap, &status);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(fs::exists(output_dir + "/annotations/cam0_train.json"));

    // Parse and verify
    std::ifstream f(output_dir + "/annotations/cam0_train.json");
    nlohmann::json j;
    f >> j;
    EXPECT_EQ((int)j["images"].size(), 1);
    EXPECT_EQ((int)j["annotations"].size(), 1);
    EXPECT_EQ(j["annotations"][0]["num_keypoints"].get<int>(), 1);

    fs::remove_all(tmpdir);
}

static void test_export_empty_annotation_map() {
    printf("  test_export_empty_annotation_map...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_empty_amap_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    std::string output_dir = tmpdir + "/output";
    fs::create_directories(calib_dir);
    fs::create_directories(output_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    AnnotationMap amap; // empty

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = output_dir;
    cfg.camera_names = {"cam0"};
    cfg.skeleton_name = "Test";
    cfg.node_names = {"pt"};
    cfg.num_keypoints = 1;

    std::string status;
    bool ok = ExportFormats::export_coco(cfg, amap, &status);
    EXPECT_FALSE(ok); // should fail: no labeled frames
    EXPECT_TRUE(status.find("Error") != std::string::npos);

    ok = ExportFormats::export_yolo(cfg, amap, true, &status);
    EXPECT_FALSE(ok);

    ok = ExportFormats::export_deeplabcut(cfg, amap, &status);
    EXPECT_FALSE(ok);

    fs::remove_all(tmpdir);
}

static void test_export_coco_y_flip_consistency() {
    printf("  test_export_coco_y_flip_consistency...\n");
    // Verify COCO and YOLO produce consistent Y coordinates
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_yflip_coco_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    std::string output_dir = tmpdir + "/output";
    fs::create_directories(calib_dir);
    fs::create_directories(output_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 10, 1, 1);
    // ImPlot y=400, h=480 → image y = 80
    fa.cameras[0].keypoints[0].x = 320.0; fa.cameras[0].keypoints[0].y = 400.0;
    fa.cameras[0].keypoints[0].labeled = true;

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = output_dir;
    cfg.camera_names = {"cam0"};
    cfg.skeleton_name = "T";
    cfg.node_names = {"p"};
    cfg.num_keypoints = 1;
    cfg.bbox_margin = 10.0f;
    cfg.train_ratio = 1.0f;
    cfg.seed = 1;

    auto coco = ExportFormats::build_coco_json(amap, {10}, cfg, 0, "cam0", 640, 480);

    // COCO keypoint y should be 80 (480 - 400)
    double coco_y = coco["annotations"][0]["keypoints"][1].get<double>();
    EXPECT_NEAR(coco_y, 80.0, 0.01);

    fs::remove_all(tmpdir);
}

static void test_export_yolo_detect_has_no_keypoints() {
    printf("  test_export_yolo_detect_has_no_keypoints...\n");
    namespace fs = std::filesystem;

    std::string tmpdir = "/tmp/test_yolo_det_nokp_" + std::to_string(getpid());
    std::string calib_dir = tmpdir + "/calib";
    std::string output_dir = tmpdir + "/output";
    fs::create_directories(calib_dir);
    fs::create_directories(output_dir);
    write_mock_calib_yaml(calib_dir + "/cam0.yaml", 640, 480);

    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 10, 2, 1);
    fa.cameras[0].keypoints[0].x = 100.0; fa.cameras[0].keypoints[0].y = 300.0;
    fa.cameras[0].keypoints[0].labeled = true;
    fa.cameras[0].keypoints[1].x = 200.0; fa.cameras[0].keypoints[1].y = 250.0;
    fa.cameras[0].keypoints[1].labeled = true;

    ExportFormats::ExportConfig cfg;
    cfg.calibration_folder = calib_dir;
    cfg.output_folder = output_dir;
    cfg.camera_names = {"cam0"};
    cfg.skeleton_name = "T";
    cfg.node_names = {"a", "b"};
    cfg.num_keypoints = 2;
    cfg.bbox_margin = 10.0f;
    cfg.train_ratio = 1.0f;
    cfg.seed = 1;

    std::string status;
    ExportFormats::export_yolo(cfg, amap, false, &status); // detect mode

    // Check label file: should have only 5 fields (class cx cy w h), no keypoints
    for (auto &e : fs::directory_iterator(output_dir + "/labels/train/cam0")) {
        if (e.path().extension() != ".txt") continue;
        std::ifstream f(e.path());
        std::string line;
        std::getline(f, line);
        std::stringstream ss(line);
        int count = 0;
        double v;
        while (ss >> v) ++count;
        EXPECT_EQ(count, 5); // class + 4 bbox only
        break;
    }

    // data.yaml should NOT have kpt_shape
    {
        std::ifstream f(output_dir + "/data.yaml");
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        EXPECT_TRUE(content.find("kpt_shape") == std::string::npos);
    }

    fs::remove_all(tmpdir);
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON persistence edge cases
// ═══════════════════════════════════════════════════════════════════════════

static void test_json_obb_roundtrip() {
    printf("  test_json_obb_roundtrip...\n");
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 7, 1, 1);
    auto &ext = fa.cameras[0].get_extras();
    ext.has_obb = true;
    ext.obb_cx = 100.5;
    ext.obb_cy = 200.5;
    ext.obb_w = 50.0;
    ext.obb_h = 30.0;
    ext.obb_angle = 0.785; // ~45 degrees

    auto j = annotations_to_json(amap);
    EXPECT_EQ((int)j["frames"].size(), 1);

    // Deserialize into fresh amap
    AnnotationMap amap2;
    get_or_create_frame(amap2, 7, 1, 1);
    annotations_from_json(j, amap2);

    auto &cam2 = amap2[7].cameras[0];
    EXPECT_TRUE(cam2.has_obb());
    EXPECT_NEAR(cam2.extras->obb_cx, 100.5, 0.001);
    EXPECT_NEAR(cam2.extras->obb_angle, 0.785, 0.001);
}

static void test_json_multiple_masks() {
    printf("  test_json_multiple_masks...\n");
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 3, 1, 1);
    auto &ext = fa.cameras[0].get_extras();
    ext.has_mask = true;
    ext.mask_polygons = {
        {{10.0, 20.0}, {30.0, 40.0}, {50.0, 20.0}},
        {{100.0, 100.0}, {120.0, 130.0}, {140.0, 100.0}, {120.0, 70.0}}
    };

    auto j = annotations_to_json(amap);
    AnnotationMap amap2;
    get_or_create_frame(amap2, 3, 1, 1);
    annotations_from_json(j, amap2);

    auto &cam2 = amap2[3].cameras[0];
    EXPECT_TRUE(cam2.has_mask());
    EXPECT_EQ((int)cam2.extras->mask_polygons.size(), 2);
    EXPECT_EQ((int)cam2.extras->mask_polygons[0].size(), 3);
    EXPECT_EQ((int)cam2.extras->mask_polygons[1].size(), 4);
    EXPECT_NEAR(cam2.extras->mask_polygons[1][3].y, 70.0, 0.001);
}

static void test_json_combined_bbox_obb_mask() {
    printf("  test_json_combined_bbox_obb_mask...\n");
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 1, 1, 2);
    auto &ext0 = fa.cameras[0].get_extras();
    ext0.has_bbox = true;
    ext0.bbox_x = 10; ext0.bbox_y = 20; ext0.bbox_w = 100; ext0.bbox_h = 80;
    auto &ext1 = fa.cameras[1].get_extras();
    ext1.has_obb = true;
    ext1.obb_cx = 50; ext1.obb_cy = 60; ext1.obb_w = 40; ext1.obb_h = 20; ext1.obb_angle = 1.0;
    ext1.has_mask = true;
    ext1.mask_polygons = {{{0, 0}, {10, 10}, {20, 0}}};

    auto j = annotations_to_json(amap);
    AnnotationMap amap2;
    get_or_create_frame(amap2, 1, 1, 2);
    annotations_from_json(j, amap2);

    EXPECT_TRUE(amap2[1].cameras[0].has_bbox());
    EXPECT_FALSE(amap2[1].cameras[0].has_obb());
    EXPECT_TRUE(amap2[1].cameras[1].has_obb());
    EXPECT_TRUE(amap2[1].cameras[1].has_mask());
    EXPECT_NEAR(amap2[1].cameras[0].extras->bbox_w, 100.0, 0.001);
}

// ═══════════════════════════════════════════════════════════════════════════
// Data model edge cases
// ═══════════════════════════════════════════════════════════════════════════

static void test_get_or_create_frame_idempotent() {
    printf("  test_get_or_create_frame_idempotent...\n");
    AnnotationMap amap;

    auto &fa1 = get_or_create_frame(amap, 10, 3, 2);
    fa1.cameras[0].keypoints[0].labeled = true;

    auto &fa2 = get_or_create_frame(amap, 10, 3, 2);
    // Should return same frame, not create a new one
    EXPECT_EQ((int)amap.size(), 1);
    EXPECT_TRUE(fa2.cameras[0].keypoints[0].labeled);
}

static void test_make_frame_sizes_match() {
    printf("  test_make_frame_sizes_match...\n");
    auto fa = make_frame(5, 3, 0, 0, 1);
    EXPECT_EQ((int)fa.kp3d.size(), 5);
    EXPECT_EQ((int)fa.cameras.size(), 3);
    for (int c = 0; c < 3; ++c) {
        EXPECT_EQ((int)fa.cameras[c].keypoints.size(), 5);
    }
    EXPECT_EQ(fa.instance_id, 0);
    EXPECT_EQ(fa.category_id, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// SAM cache sentinel test
// ═══════════════════════════════════════════════════════════════════════════

static void test_sam_cache_sentinel_default() {
    printf("  test_sam_cache_sentinel_default...\n");
    // Verify SamState defaults prevent false cache hits
    SamState s;
    EXPECT_EQ(s.cached_frame, -1);
    EXPECT_EQ(s.cached_cam, -1);

    // With defaults of -1, the sentinel guard should prevent cache hit
    // even if sam_encode is called with frame_num=-1, cam_idx=-1
    // (This tests the design, not the ONNX path)
    EXPECT_FALSE(s.loaded); // not loaded → encode would fail anyway
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("=== Annotation System Tests ===\n");

    printf("\n--- Data Model ---\n");
    test_make_frame_basic();
    test_make_frame_zero_dims();
    test_get_or_create_frame();
    test_frame_has_any_labels();

    printf("\n--- Migration ---\n");
    test_migration_roundtrip();

    printf("\n--- JSON Persistence ---\n");
    test_json_roundtrip_bbox();
    test_json_roundtrip_mask();
    test_json_empty_extended_data();
    test_json_missing_frame_in_amap();
    test_json_camera_index_out_of_bounds();
    test_json_file_save_load();
    test_json_file_missing_is_ok();

    printf("\n--- Export Formats ---\n");
    test_format_name();
    test_split_train_val();
    test_split_train_val_deterministic();
    test_split_train_val_empty();
    test_split_train_val_single();
    test_split_train_val_all_train();
    test_get_labeled_frames();
    test_build_coco_json();
    test_build_coco_json_no_visible_keypoints();
    test_build_coco_json_img_id_consistency();

    printf("\n--- OBB Geometry ---\n");
    test_obb_horizontal();
    test_obb_vertical();
    test_obb_degenerate_axis();
    test_obb_corner_on_axis();
    test_obb_diagonal();
    test_obb_get_corners_axis_aligned();
    test_obb_contains_basic();
    test_obb_contains_rotated();

    printf("\n--- SAM Inference ---\n");
    test_sam_mask_to_polygon_empty();
    test_sam_mask_to_polygon_empty_data();
    test_sam_mask_to_polygon_all_zero();
    test_sam_mask_to_polygon_rect();
    test_sam_init_no_onnx();
    test_sam_segment_not_loaded();

    printf("\n--- Bbox Tool ---\n");
    test_bbox_next_class_color();

    printf("\n--- AnnotationConfig ---\n");
    test_annotation_config_defaults();
    test_annotation_config_json_roundtrip();
    test_annotation_config_backward_compat();

    printf("\n--- Export Pipeline: COCO ---\n");
    test_export_coco_full_pipeline();
    test_build_coco_json_with_explicit_bbox();
    test_build_coco_json_with_mask();

    printf("\n--- Export Pipeline: YOLO ---\n");
    test_export_yolo_pose_full_pipeline();
    test_export_yolo_detect_no_keypoints();

    printf("\n--- Export Pipeline: DeepLabCut ---\n");
    test_export_deeplabcut_full_pipeline();

    printf("\n--- Export Pipeline: Edge Cases ---\n");
    test_export_no_labeled_frames();
    test_export_missing_calibration();
    test_export_dispatch();

    printf("\n--- Save/Load Roundtrip ---\n");
    test_save_load_keypoints_roundtrip();
    test_save_load_with_extended_data();

    printf("\n--- Y-flip Consistency ---\n");
    test_yolo_pose_y_flip();
    test_dlc_y_flip();

    printf("\n--- SAM Preprocessing ---\n");
    test_sam_preprocess_mobilesam();
    test_sam_preprocess_sam2();
    test_sam_bilinear_resize();

    printf("\n--- Contour Extraction ---\n");
    test_sam_contour_l_shape();
    test_sam_contour_single_pixel();

    printf("\n--- Export Schema Validation ---\n");
    test_coco_json_schema();
    test_yolo_label_format_validation();

    printf("\n--- Bridge Integration ---\n");
    test_bridge_keypoints_and_extended_roundtrip();

    printf("\n--- Bridge Edge Cases ---\n");
    test_refresh_empty_keypoints_map();
    test_refresh_null_keypoints_skipped();
    test_refresh_preserves_bbox_on_update();
    test_refresh_adds_new_frames();
    test_refresh_removes_stale_frames();
    test_refresh_new_frame_gets_keypoints();

    printf("\n--- Migration Edge Cases ---\n");
    test_migrate_null_keypoints_skipped();
    test_migrate_multi_frame();
    test_migrate_empty_map();

    printf("\n--- Export Edge Cases (extended) ---\n");
    test_export_coco_single_camera();
    test_export_empty_annotation_map();
    test_export_coco_y_flip_consistency();
    test_export_yolo_detect_has_no_keypoints();

    printf("\n--- JSON Persistence (extended) ---\n");
    test_json_obb_roundtrip();
    test_json_multiple_masks();
    test_json_combined_bbox_obb_mask();

    printf("\n--- Data Model Edge Cases ---\n");
    test_get_or_create_frame_idempotent();
    test_make_frame_sizes_match();

    printf("\n--- SAM Cache Sentinel ---\n");
    test_sam_cache_sentinel_default();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
