// test_jarvis_golden.cpp — Golden-file comparison test for JARVIS export
//
// Loads annotate1 v2 label data into an AnnotationMap, calls
// generate_annotation_json_from_amap(), and compares the output against
// the known-good export at annotate1/export/2026_03_10_08_45_34/.
//
// This validates that the AnnotationMap-based JARVIS export produces
// identical annotation data to the old CSV-based path.

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "annotation.h"
#include "annotation_csv.h"
#include "jarvis_export.h"
#include "json.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "test_framework.h"

// Paths to real annotate1 project data
static const char *V2_CSV_DIR = "/Users/johnsonr/red_projects/annotate1/labeled_data/2026_03_10_00_43_32_v2";
static const char *CALIB_DIR = "/Users/johnsonr/red_projects/calib1/aruco_calibration/2026_03_07_15_20_21";
static const char *OLD_TRAIN_JSON = "/Users/johnsonr/red_projects/annotate1/export/2026_03_10_08_45_34/annotations/instances_train.json";
static const char *OLD_VAL_JSON = "/Users/johnsonr/red_projects/annotate1/export/2026_03_10_08_45_34/annotations/instances_val.json";

static const char *CAMERA_NAMES[] = {
    "Cam2002486", "Cam2002487", "Cam2005325", "Cam2006050",
    "Cam2006051", "Cam2006052", "Cam2006054", "Cam2006055",
    "Cam2006515", "Cam2006516", "Cam2008665", "Cam2008666",
    "Cam2008667", "Cam2008668", "Cam2008669", "Cam2008670"
};
static const int NUM_CAMERAS = 16;
static const int NUM_KEYPOINTS = 4;

static void test_jarvis_golden_comparison() {
    printf("  test_jarvis_golden_comparison...\n");
    namespace fs = std::filesystem;

    // 1. Load annotate1 v2 data into AnnotationMap
    AnnotationMap amap;
    std::string err;
    std::vector<std::string> cam_names(CAMERA_NAMES, CAMERA_NAMES + NUM_CAMERAS);
    int ret = AnnotationCSV::load_all(V2_CSV_DIR, amap, "Rat4",
                                       NUM_KEYPOINTS, NUM_CAMERAS, cam_names, err);
    EXPECT_EQ(ret, 0);
    printf("    Loaded %d frames from v2 CSVs\n", (int)amap.size());

    // 2. Read image dimensions from calibration (same as JARVIS exporter does)
    std::map<std::string, int> image_width, image_height;
    for (const auto &cam : cam_names) {
        std::string path = std::string(CALIB_DIR) + "/" + cam + ".yaml";
        try {
            auto yaml = opencv_yaml::read(path);
            image_width[cam] = yaml.getInt("image_width");
            image_height[cam] = yaml.getInt("image_height");
        } catch (...) {
            fprintf(stderr, "    ERROR: Cannot read calibration: %s\n", path.c_str());
            EXPECT_TRUE(false);
            return;
        }
    }
    printf("    Image dims: %dx%d\n", image_width[cam_names[0]], image_height[cam_names[0]]);

    // 3. Get valid frames (fully triangulated) — same filter as new exporter
    std::vector<int> valid_frames;
    for (const auto &[fid, fa] : amap)
        if (frame_is_fully_triangulated(fa, NUM_KEYPOINTS))
            valid_frames.push_back((int)fid);
    std::sort(valid_frames.begin(), valid_frames.end());
    printf("    Fully triangulated frames: %d\n", (int)valid_frames.size());

    // 4. Load old export for comparison
    nlohmann::json old_train, old_val;
    {
        std::ifstream f(OLD_TRAIN_JSON);
        EXPECT_TRUE(f.is_open());
        if (!f.is_open()) return;
        f >> old_train;
    }
    {
        std::ifstream f(OLD_VAL_JSON);
        EXPECT_TRUE(f.is_open());
        if (!f.is_open()) return;
        f >> old_val;
    }

    int old_total_ann = (int)old_train["annotations"].size() + (int)old_val["annotations"].size();
    int old_total_img = (int)old_train["images"].size() + (int)old_val["images"].size();
    printf("    Old export: %d images, %d annotations\n", old_total_img, old_total_ann);

    // 5. Build ExportConfig matching the old export
    JarvisExport::ExportConfig config;
    config.camera_names = cam_names;
    config.skeleton_name = "Rat4";
    config.node_names = {"Snout", "EarL", "EarR", "Tail"};
    config.edges = {{0,1}, {0,2}, {1,3}, {2,3}};
    config.num_keypoints = NUM_KEYPOINTS;
    config.margin_pixel = 50.0f;
    config.train_ratio = 0.9f;
    config.seed = 42;

    // 6. Use the same train/val split as the old export (same seed + ratio)
    std::vector<int> train_frames, val_frames;
    JarvisExport::split_frames(valid_frames, config.train_ratio, config.seed,
                                train_frames, val_frames);
    printf("    Split: %d train, %d val (seed=%d, ratio=%.1f)\n",
           (int)train_frames.size(), (int)val_frames.size(),
           config.seed, config.train_ratio);

    // The old export used a different label folder timestamp as trial_name.
    // Extract it from the old export's image file_name.
    std::string old_trial;
    if (!old_train["images"].empty()) {
        std::string fn = old_train["images"][0]["file_name"].get<std::string>();
        old_trial = fn.substr(0, fn.find('/'));
    }
    printf("    Old trial_name: %s\n", old_trial.c_str());

    // Use the v2 folder name as trial_name for new export
    std::string new_trial = fs::path(V2_CSV_DIR).filename().string();
    printf("    New trial_name: %s\n", new_trial.c_str());

    // 7. Generate new JARVIS JSON from AnnotationMap
    int new_train_complete = 0, new_train_incomplete = 0;
    auto new_train = JarvisExport::generate_annotation_json_from_amap(
        new_trial, train_frames, amap, config, image_width, image_height,
        &new_train_complete, &new_train_incomplete);

    int new_val_complete = 0, new_val_incomplete = 0;
    auto new_val = JarvisExport::generate_annotation_json_from_amap(
        new_trial, val_frames, amap, config, image_width, image_height,
        &new_val_complete, &new_val_incomplete);

    int new_total_ann = (int)new_train["annotations"].size() + (int)new_val["annotations"].size();
    int new_total_img = (int)new_train["images"].size() + (int)new_val["images"].size();
    printf("    New export: %d images, %d annotations\n", new_total_img, new_total_ann);
    printf("    New train: %d complete, %d incomplete\n", new_train_complete, new_train_incomplete);

    // 8. Compare structure
    EXPECT_TRUE(new_train.contains("images"));
    EXPECT_TRUE(new_train.contains("annotations"));
    EXPECT_TRUE(new_train.contains("categories"));
    EXPECT_TRUE(new_train.contains("keypoint_names"));
    EXPECT_TRUE(new_train.contains("skeleton"));
    EXPECT_TRUE(new_train.contains("calibrations"));
    EXPECT_TRUE(new_train.contains("framesets"));

    // The old and new exports may have different frame counts if the v2 data
    // has more labeled frames than the old export's source data.
    // What matters is that for frames present in BOTH, the annotation values match.

    // 9. Build lookup: old annotations by (file_name without trial prefix)
    // This handles the trial_name difference between old and new exports.
    auto strip_trial = [](const std::string &fn) -> std::string {
        auto pos = fn.find('/');
        return (pos != std::string::npos) ? fn.substr(pos + 1) : fn;
    };

    // Build old image_id → file_name and annotation_id → annotation maps
    std::map<std::string, nlohmann::json> old_ann_by_file;
    {
        std::map<int, std::string> old_img_id_to_file;
        auto merge = [&](const nlohmann::json &j) {
            for (const auto &img : j["images"])
                old_img_id_to_file[img["id"].get<int>()] = strip_trial(img["file_name"].get<std::string>());
            for (const auto &ann : j["annotations"]) {
                int img_id = ann["image_id"].get<int>();
                auto it = old_img_id_to_file.find(img_id);
                if (it != old_img_id_to_file.end())
                    old_ann_by_file[it->second] = ann;
            }
        };
        merge(old_train);
        merge(old_val);
    }

    // Build new annotation lookup the same way
    std::map<std::string, nlohmann::json> new_ann_by_file;
    {
        std::map<int, std::string> new_img_id_to_file;
        auto merge = [&](const nlohmann::json &j) {
            for (const auto &img : j["images"])
                new_img_id_to_file[img["id"].get<int>()] = strip_trial(img["file_name"].get<std::string>());
            for (const auto &ann : j["annotations"]) {
                int img_id = ann["image_id"].get<int>();
                auto it = new_img_id_to_file.find(img_id);
                if (it != new_img_id_to_file.end())
                    new_ann_by_file[it->second] = ann;
            }
        };
        merge(new_train);
        merge(new_val);
    }

    printf("    Old annotations by file: %d\n", (int)old_ann_by_file.size());
    printf("    New annotations by file: %d\n", (int)new_ann_by_file.size());

    // 10. Compare annotation values for all files present in both
    int compared = 0, kp_matches = 0, bbox_matches = 0, mismatches = 0;
    for (const auto &[file, old_ann] : old_ann_by_file) {
        auto it = new_ann_by_file.find(file);
        if (it == new_ann_by_file.end()) continue; // frame only in old export
        const auto &new_ann = it->second;
        compared++;

        // Compare keypoints (int values, should be identical)
        auto old_kps = old_ann["keypoints"];
        auto new_kps = new_ann["keypoints"];
        bool kp_ok = (old_kps.size() == new_kps.size());
        if (kp_ok) {
            for (size_t i = 0; i < old_kps.size(); ++i) {
                if (old_kps[i].get<int>() != new_kps[i].get<int>()) {
                    kp_ok = false;
                    break;
                }
            }
        }
        if (kp_ok) kp_matches++;
        else {
            mismatches++;
            if (mismatches <= 3)
                printf("    MISMATCH %s: old_kps=%s new_kps=%s\n",
                       file.c_str(),
                       old_kps.dump().substr(0, 60).c_str(),
                       new_kps.dump().substr(0, 60).c_str());
        }

        // Compare bbox (floating point, allow small tolerance)
        auto old_bbox = old_ann["bbox"];
        auto new_bbox = new_ann["bbox"];
        bool bbox_ok = (old_bbox.size() == new_bbox.size());
        if (bbox_ok) {
            for (size_t i = 0; i < old_bbox.size(); ++i) {
                if (fabs(old_bbox[i].get<double>() - new_bbox[i].get<double>()) > 0.01) {
                    bbox_ok = false;
                    break;
                }
            }
        }
        if (bbox_ok) bbox_matches++;

        // category_id should be 1
        EXPECT_EQ(new_ann["category_id"].get<int>(), 1);
        EXPECT_EQ(new_ann["num_keypoints"].get<int>(), NUM_KEYPOINTS);
    }

    printf("    Compared: %d annotations\n", compared);
    printf("    Keypoint matches: %d/%d\n", kp_matches, compared);
    printf("    Bbox matches: %d/%d\n", bbox_matches, compared);
    printf("    Mismatches: %d\n", mismatches);

    EXPECT_TRUE(compared > 0);
    EXPECT_EQ(mismatches, 0);
    EXPECT_EQ(kp_matches, compared);
    EXPECT_EQ(bbox_matches, compared);

    // 11. Verify JARVIS-specific top-level fields
    EXPECT_EQ(new_train["keypoint_names"].size(), (size_t)NUM_KEYPOINTS);
    EXPECT_EQ(new_train["skeleton"].size(), config.edges.size());
    EXPECT_TRUE(new_train["calibrations"].contains(new_trial));
    EXPECT_TRUE(new_train["categories"][0]["name"].get<std::string>() == "Rat4");
    EXPECT_EQ(new_train["categories"][0]["num_keypoints"].get<int>(), NUM_KEYPOINTS);

    printf("    OK\n");
}

int main() {
    printf("=== JARVIS Golden File Comparison Test ===\n\n");

    namespace fs = std::filesystem;
    if (!fs::exists(V2_CSV_DIR)) {
        fprintf(stderr, "ERROR: V2 CSV directory not found: %s\n", V2_CSV_DIR);
        return 1;
    }
    if (!fs::exists(OLD_TRAIN_JSON)) {
        fprintf(stderr, "ERROR: Old export not found: %s\n", OLD_TRAIN_JSON);
        return 1;
    }

    test_jarvis_golden_comparison();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
