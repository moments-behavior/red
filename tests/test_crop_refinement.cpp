// Synthetic end-to-end test for the two-stage cropped-sensor calibration:
//   1. posts triangulation + posts_3d.csv round-trip
//   2. crop transform (exact principal-point shift, per-camera dims)
//   3. stage-2 refinement recovery of injected principal-point / focal drift
//
// No video/GUI dependencies — everything is generated in memory.


#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>

#include "crop_calibration.h"
#include "cropped_refinement.h"
#include "orange_config.h"
#include "calibration_pipeline.h"
#include "red_math.h"

namespace fs = std::filesystem;
using CalibrationPipeline::CameraPose;

static int g_checks = 0;

#define REQUIRE(cond, msg)                                                       \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL(line %d): %s\n", __LINE__, msg);             \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static const int NUM_CAMS = 8;
static const int NUM_POSTS = 12;
static const int FULL_W = 2048, FULL_H = 1536;
static const double FX = 8000.0, CX = 1024.0, CY = 768.0;

// Cameras on a ring, radius 1200 mm, height 1200 mm, looking at the origin.
static std::vector<CameraPose> make_rig() {
    std::vector<CameraPose> poses(NUM_CAMS);
    for (int i = 0; i < NUM_CAMS; i++) {
        double th = 2.0 * M_PI * i / NUM_CAMS;
        Eigen::Vector3d C(1200.0 * std::cos(th), 1200.0 * std::sin(th), 1200.0);
        Eigen::Vector3d fwd = (-C).normalized();
        Eigen::Vector3d up(0, 0, 1);
        Eigen::Vector3d x = fwd.cross(up).normalized();
        Eigen::Vector3d y = fwd.cross(x).normalized();
        CameraPose p;
        p.R.row(0) = x.transpose();
        p.R.row(1) = y.transpose();
        p.R.row(2) = fwd.transpose();
        p.t = -p.R * C;
        p.K = Eigen::Matrix3d::Identity();
        p.K(0, 0) = FX;
        p.K(1, 1) = FX;
        p.K(0, 2) = CX;
        p.K(1, 2) = CY;
        p.dist << -0.05, 0, 0, 0, 0;
        poses[i] = p;
    }
    return poses;
}

// Posts in a shallow volume around the origin.
static std::vector<Eigen::Vector3d> make_posts() {
    std::vector<Eigen::Vector3d> posts;
    for (int i = 0; i < NUM_POSTS; i++) {
        double a = 2.0 * M_PI * i / NUM_POSTS;
        double r = 10.0 + 15.0 * (i % 3);
        posts.push_back(Eigen::Vector3d(r * std::cos(a), r * std::sin(a),
                                        2.0 * (i % 5)));
    }
    return posts;
}

static std::map<std::string, std::map<int, Eigen::Vector2d>> project_all(
    const std::vector<CameraPose> &poses,
    const std::vector<std::string> &names,
    const std::vector<Eigen::Vector3d> &posts, double noise_std = 0.0,
    uint32_t seed = 1) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, noise_std);
    std::map<std::string, std::map<int, Eigen::Vector2d>> lm;
    for (int c = 0; c < (int)poses.size(); c++)
        for (int p = 0; p < (int)posts.size(); p++) {
            auto px = red_math::projectPointR(posts[p], poses[c].R, poses[c].t,
                                              poses[c].K, poses[c].dist);
            if (noise_std > 0)
                px += Eigen::Vector2d(noise(rng), noise(rng));
            lm[names[c]][p] = px;
        }
    return lm;
}

static CameraPose read_pose(const std::string &yaml_path, int *w = nullptr,
                            int *h = nullptr) {
    auto yf = opencv_yaml::read(yaml_path);
    CameraPose p;
    p.K = yf.getMatrix("camera_matrix").block<3, 3>(0, 0);
    Eigen::MatrixXd dm = yf.getMatrix("distortion_coefficients");
    for (int j = 0; j < 5; j++) p.dist(j) = dm(j, 0);
    p.R = yf.getMatrix("rc_ext").block<3, 3>(0, 0);
    Eigen::MatrixXd tm = yf.getMatrix("tc_ext");
    p.t = Eigen::Vector3d(tm(0, 0), tm(1, 0), tm(2, 0));
    if (w) *w = yf.getInt("image_width");
    if (h) *h = yf.getInt("image_height");
    return p;
}

int main() {
    fs::path root = fs::temp_directory_path() / "red_test_crop_refinement";
    fs::remove_all(root);
    fs::create_directories(root);

    auto poses = make_rig();
    auto posts = make_posts();
    std::vector<std::string> names;
    for (int i = 0; i < NUM_CAMS; i++) names.push_back(std::to_string(1000 + i));

    // ── Stage 1: write full-frame calibration ──────────────────────────────
    std::string stage1 = (root / "stage1").string();
    std::string status;
    REQUIRE(CalibrationPipeline::write_calibration(
              poses, names, stage1, std::vector<int>(NUM_CAMS, FULL_W),
              std::vector<int>(NUM_CAMS, FULL_H), &status),
          "stage1 write_calibration");

    // ── 1. Posts triangulation + CSV round-trip ────────────────────────────
    {
        auto lm = project_all(poses, names, posts);
        std::map<int, Eigen::Vector3d> tri;
        auto reports = CropCalibration::triangulate_posts_with_report(
            names, lm, poses, 2.0, tri);
        REQUIRE((int)reports.size() == NUM_POSTS, "one report per post");
        for (const auto &r : reports) {
            REQUIRE(r.n_views == NUM_CAMS, "post seen by all cameras");
            REQUIRE(r.accepted, "noise-free post accepted");
            REQUIRE(r.mean_reproj < 0.01, "noise-free reproj tiny");
            REQUIRE((tri.at(r.id) - posts[r.id]).norm() < 1e-3,
                  "triangulated post matches ground truth");
        }

        // Round-trip with a missing post (row of NaNs keeps ids aligned)
        std::string csv = (root / "posts_3d.csv").string();
        std::map<int, Eigen::Vector3d> partial = tri;
        partial.erase(7);
        REQUIRE(CropCalibration::write_posts_3d_csv(csv, NUM_POSTS, partial),
              "write posts csv");
        auto back = CropCalibration::read_posts_3d_csv(csv);
        REQUIRE((int)back.size() == NUM_POSTS - 1, "NaN row skipped on read");
        REQUIRE(back.count(7) == 0, "missing post stays missing");
        REQUIRE((back.at(9) - tri.at(9)).norm() < 1e-4,
              "row index == post id preserved across the gap");
    }

    // ── 2. Crop transform ───────────────────────────────────────────────────
    CropCalibration::CropSpec spec;
    for (int i = 0; i < NUM_CAMS; i++) {
        CropCalibration::CameraCrop c;
        c.serial = names[i];
        c.width = 640 + 16 * (i % 2);   // heterogeneous crop sizes
        c.height = 480 + 16 * (i % 2);
        c.offset_x = (int)CX - c.width / 2 + 8 * i;
        c.offset_y = (int)CY - c.height / 2 + 4 * i;
        spec.cameras.push_back(c);
    }
    std::string cropped = (root / "cropped").string();
    {
        auto cr = CropCalibration::apply_crop_to_calibration(stage1, spec,
                                                             cropped);
        REQUIRE(cr.success, cr.error.c_str());
        REQUIRE(cr.warnings.empty(), "no warnings for in-bounds crops");

        for (int i = 0; i < NUM_CAMS; i++) {
            int w = 0, h = 0;
            auto stage1_pose =
                read_pose(stage1 + "/Cam" + names[i] + ".yaml");
            auto p = read_pose(cropped + "/Cam" + names[i] + ".yaml", &w, &h);
            REQUIRE(w == spec.cameras[i].width && h == spec.cameras[i].height,
                  "per-camera crop dims written (WP1 regression)");
            REQUIRE(std::abs(p.K(0, 2) -
                           (stage1_pose.K(0, 2) - spec.cameras[i].offset_x)) <
                      1e-9,
                  "cx shifted exactly");
            REQUIRE(std::abs(p.K(1, 2) -
                           (stage1_pose.K(1, 2) - spec.cameras[i].offset_y)) <
                      1e-9,
                  "cy shifted exactly");
            REQUIRE((p.R - stage1_pose.R).norm() < 1e-12, "R unchanged");
            REQUIRE((p.t - stage1_pose.t).norm() < 1e-12, "t unchanged");
            REQUIRE((p.dist - stage1_pose.dist).norm() < 1e-12,
                  "distortion unchanged");
            REQUIRE(std::abs(p.K(0, 0) - stage1_pose.K(0, 0)) < 1e-12,
                  "focal unchanged");
        }

        CropCalibration::CropSpec back;
        std::string err;
        REQUIRE(CropCalibration::load_crop_spec(cropped + "/crop_info.json",
                                              back, &err),
              "crop_info.json readable");
        REQUIRE((int)back.cameras.size() == NUM_CAMS &&
                  back.cameras[3].offset_x == spec.cameras[3].offset_x &&
                  back.cameras[5].height == spec.cameras[5].height,
              "crop spec round-trips");
        REQUIRE(!back.source_calibration.empty() && !back.timestamp.empty(),
              "provenance recorded");
    }

    // Cropped ground-truth poses (in memory) + posts_3d.csv for refinement
    std::vector<CameraPose> cropped_true = poses;
    for (int i = 0; i < NUM_CAMS; i++) {
        cropped_true[i].K(0, 2) -= spec.cameras[i].offset_x;
        cropped_true[i].K(1, 2) -= spec.cameras[i].offset_y;
    }
    std::string posts_csv = (root / "posts_for_refine.csv").string();
    {
        std::map<int, Eigen::Vector3d> tri;
        for (int p = 0; p < NUM_POSTS; p++) tri[p] = posts[p];
        REQUIRE(CropCalibration::write_posts_3d_csv(posts_csv, NUM_POSTS, tri),
              "write refine posts csv");
    }

    const double dcx[NUM_CAMS] = {2.5, -3.0, 1.5, -2.25, 3.0, -1.75, 2.0, -2.5};
    const double dcy[NUM_CAMS] = {-1.5, 2.0, -3.0, 1.25, -2.5, 3.0, -1.0, 2.25};

    // ── 3a. Principal-point drift recovery, noise-free ──────────────────────
    {
        auto drifted = cropped_true;
        for (int i = 0; i < NUM_CAMS; i++) {
            drifted[i].K(0, 2) += dcx[i];
            drifted[i].K(1, 2) += dcy[i];
        }
        auto lm = project_all(drifted, names, posts);

        CroppedRefinement::RefineConfig cfg;
        cfg.calibration_folder = cropped;
        cfg.posts_3d_csv = posts_csv;
        cfg.output_folder = (root / "refined_pp").string();
        cfg.camera_names = names;
        cfg.free_focal = false;
        auto res = CroppedRefinement::run_cropped_refinement(cfg, lm);
        REQUIRE(res.success, res.error.c_str());
        REQUIRE(res.obs_dropped == 0, "no outliers in clean data");
        REQUIRE(res.mean_before > 1.0, "injected drift visible before");
        REQUIRE(res.mean_after < 0.05, "drift absorbed after");
        for (int i = 0; i < NUM_CAMS; i++) {
            REQUIRE(std::abs(res.per_camera[i].d_cx - dcx[i]) < 0.05,
                  "cx drift recovered");
            REQUIRE(std::abs(res.per_camera[i].d_cy - dcy[i]) < 0.05,
                  "cy drift recovered");
            REQUIRE(std::abs(res.per_camera[i].d_f) < 1e-9,
                  "principal-only mode leaves focal untouched");
        }
        // Extrinsics and distortion must be byte-identical to the input
        for (int i = 0; i < NUM_CAMS; i++) {
            auto in = read_pose(cropped + "/Cam" + names[i] + ".yaml");
            auto out =
                read_pose(cfg.output_folder + "/Cam" + names[i] + ".yaml");
            REQUIRE((in.R - out.R).norm() == 0.0, "refined R identical");
            REQUIRE((in.t - out.t).norm() == 0.0, "refined t identical");
            REQUIRE((in.dist - out.dist).norm() == 0.0,
                  "refined distortion identical");
        }
        REQUIRE(fs::exists(cfg.output_folder + "/refine_report.json"),
              "refine report written");
        REQUIRE(fs::exists(cfg.output_folder + "/crop_info.json"),
              "crop provenance carried forward");
    }

    // ── 3b. Principal + focal recovery, noise-free ──────────────────────────
    {
        auto drifted = cropped_true;
        for (int i = 0; i < NUM_CAMS; i++) {
            drifted[i].K(0, 2) += dcx[i];
            drifted[i].K(1, 2) += dcy[i];
            double scale = (i % 2) ? 1.003 : 0.997;  // ±0.3% focal change
            drifted[i].K(0, 0) *= scale;
            drifted[i].K(1, 1) *= scale;
        }
        auto lm = project_all(drifted, names, posts);

        CroppedRefinement::RefineConfig cfg;
        cfg.calibration_folder = cropped;
        cfg.posts_3d_csv = posts_csv;
        cfg.output_folder = (root / "refined_focal").string();
        cfg.camera_names = names;
        cfg.free_focal = true;
        cfg.prior_focal_weight = 1e-4;  // let the data speak in this test
        auto res = CroppedRefinement::run_cropped_refinement(cfg, lm);
        REQUIRE(res.success, res.error.c_str());
        REQUIRE(res.mean_after < 0.05, "focal+principal drift absorbed");
        for (int i = 0; i < NUM_CAMS; i++) {
            double df_true = FX * ((i % 2) ? 0.003 : -0.003);
            REQUIRE(std::abs(res.per_camera[i].d_f - df_true) < 0.5,
                  "focal drift recovered");
            REQUIRE(std::abs(res.per_camera[i].d_cx - dcx[i]) < 0.05,
                  "cx recovered alongside focal");
        }
        // fy/fx ratio preserved
        auto out = read_pose(cfg.output_folder + "/Cam" + names[0] + ".yaml");
        REQUIRE(std::abs(out.K(1, 1) / out.K(0, 0) - 1.0) < 1e-9,
              "fx=fy tie preserved");
    }

    // ── 3c. Noisy observations + holdout ────────────────────────────────────
    {
        auto drifted = cropped_true;
        for (int i = 0; i < NUM_CAMS; i++) {
            drifted[i].K(0, 2) += dcx[i];
            drifted[i].K(1, 2) += dcy[i];
        }
        auto lm = project_all(drifted, names, posts, 0.2, 7);

        CroppedRefinement::RefineConfig cfg;
        cfg.calibration_folder = cropped;
        cfg.posts_3d_csv = posts_csv;
        cfg.output_folder = (root / "refined_noisy").string();
        cfg.camera_names = names;
        cfg.free_focal = false;
        cfg.holdout_fraction = 0.25;
        auto res = CroppedRefinement::run_cropped_refinement(cfg, lm);
        REQUIRE(res.success, res.error.c_str());
        REQUIRE(res.holdout_obs > 0, "holdout split active");
        for (int i = 0; i < NUM_CAMS; i++) {
            REQUIRE(std::abs(res.per_camera[i].d_cx - dcx[i]) < 0.5,
                  "cx recovered under noise");
            REQUIRE(std::abs(res.per_camera[i].d_cy - dcy[i]) < 0.5,
                  "cy recovered under noise");
        }
        REQUIRE(res.mean_after < 0.5, "train reproj at noise floor");
        REQUIRE(res.holdout_after < 1.0, "holdout reproj sane (no overfit)");
    }

    // ── 4. Outlier click rejection ───────────────────────────────────────────
    {
        auto lm = project_all(cropped_true, names, posts);
        lm[names[2]][4] += Eigen::Vector2d(80.0, -60.0);  // a badly clicked post

        CroppedRefinement::RefineConfig cfg;
        cfg.calibration_folder = cropped;
        cfg.posts_3d_csv = posts_csv;
        cfg.output_folder = (root / "refined_outlier").string();
        cfg.camera_names = names;
        auto res = CroppedRefinement::run_cropped_refinement(cfg, lm);
        REQUIRE(res.success, res.error.c_str());
        REQUIRE(res.obs_dropped == 1, "bad click dropped by initial-reproj gate");
        REQUIRE(res.mean_after < 0.05, "clean data unaffected by dropped click");
    }

    // ── 5. shift_landmarks_to_crop: exactness + out-of-crop drop ────────────
    {
        auto full_lm = project_all(poses, names, posts);
        std::vector<std::string> dropped;
        auto crop_lm = CropCalibration::shift_landmarks_to_crop(
            full_lm, spec, dropped);

        // Shifted full-frame observation == direct projection through the
        // cropped calibration (the crop transform is exact).
        int compared = 0;
        for (int i = 0; i < NUM_CAMS; i++) {
            auto it = crop_lm.find(names[i]);
            if (it == crop_lm.end()) continue;
            for (const auto &[pid, px] : it->second) {
                auto direct = red_math::projectPointR(
                    posts[pid], cropped_true[i].R, cropped_true[i].t,
                    cropped_true[i].K, cropped_true[i].dist);
                REQUIRE((px - direct).norm() < 1e-9,
                        "shifted obs == direct cropped projection");
                REQUIRE(px.x() >= 0 && px.y() >= 0 &&
                            px.x() < spec.cameras[i].width &&
                            px.y() < spec.cameras[i].height,
                        "shifted obs inside crop bounds");
                compared++;
            }
        }
        REQUIRE(compared > NUM_CAMS * NUM_POSTS / 2,
                "most observations survive the shift");
        // In-crop posts + dropped == total
        int kept = 0;
        for (const auto &[s, pts] : crop_lm) kept += (int)pts.size();
        REQUIRE(kept + (int)dropped.size() == NUM_CAMS * NUM_POSTS,
                "kept + dropped == total observations");

        // Force an out-of-crop observation and check it is dropped+reported
        auto lm2 = full_lm;
        lm2[names[0]][0] = Eigen::Vector2d(1.0, 1.0);  // far from any crop
        std::vector<std::string> dropped2;
        auto crop_lm2 =
            CropCalibration::shift_landmarks_to_crop(lm2, spec, dropped2);
        REQUIRE(crop_lm2[names[0]].count(0) == 0, "outside obs dropped");
        REQUIRE((int)dropped2.size() == (int)dropped.size() + 1,
                "outside obs reported");
    }

    // ── 6. Orange config export (in-place update, keys preserved) ───────────
    {
        fs::path odir = root / "orange_configs";
        fs::create_directories(odir);
        // Fake orange configs with the 13 keys orange requires — only for
        // the first 7 cameras, so the last one triggers a missing warning.
        for (int i = 0; i < NUM_CAMS - 1; i++) {
            nlohmann::json j = {
                {"name", "Cam" + std::to_string(i)},
                {"width", FULL_W},
                {"height", FULL_H},
                {"frame_rate", 30},
                {"gain", 1500 + i},
                {"iris", 0},
                {"focus", 345},
                {"exposure", 2500},
                {"pixel_format", "BayerRG8"},
                {"gpu_id", 0},
                {"color_temp", "CT_3000K"},
                {"gpu_direct", false},
                {"color", true}};
            std::ofstream f(odir / (names[i] + ".json"));
            f << j.dump(4) << "\n";
        }

        // 16-aligned spec for the export
        CropCalibration::CropSpec ospec;
        for (int i = 0; i < NUM_CAMS; i++) {
            CropCalibration::CameraCrop c;
            c.serial = names[i];
            c.width = 640;
            c.height = 480;
            c.offset_x = 704 + 16 * i;
            c.offset_y = 528;
            ospec.cameras.push_back(c);
        }
        auto er = OrangeConfig::update_orange_configs(
            odir.string(), ospec, std::vector<int>(NUM_CAMS, FULL_W),
            std::vector<int>(NUM_CAMS, FULL_H));
        REQUIRE(er.success, er.error.c_str());
        REQUIRE(er.updated == NUM_CAMS - 1, "seven configs updated");
        bool missing_warned = false;
        for (const auto &w : er.warnings)
            if (w.find(names[NUM_CAMS - 1]) != std::string::npos)
                missing_warned = true;
        REQUIRE(missing_warned, "missing config warned");

        // Round-trip: ROI updated, every other key preserved verbatim
        {
            std::ifstream f(odir / (names[2] + ".json"));
            nlohmann::json j;
            f >> j;
            REQUIRE(j["width"] == 640 && j["height"] == 480,
                    "orange width/height updated");
            REQUIRE(j["offsetx"] == 704 + 16 * 2 && j["offsety"] == 528,
                    "orange offsetx/offsety written");
            REQUIRE(j["gain"] == 1502 && j["exposure"] == 2500 &&
                        j["pixel_format"] == "BayerRG8" &&
                        j["color_temp"] == "CT_3000K" &&
                        j["gpu_direct"] == false && j["color"] == true &&
                        j["frame_rate"] == 30 && j["focus"] == 345 &&
                        j["iris"] == 0 && j["gpu_id"] == 0 &&
                        j["name"] == "Cam2",
                    "all other orange keys preserved");
        }

        // Non-multiple-of-16 warning
        ospec.cameras[0].offset_x = 705;
        auto er2 =
            OrangeConfig::update_orange_configs(odir.string(), ospec);
        REQUIRE(er2.success, er2.error.c_str());
        bool align_warned = false;
        for (const auto &w : er2.warnings)
            if (w.find("multiple of 16") != std::string::npos)
                align_warned = true;
        REQUIRE(align_warned, "misaligned ROI warned");
    }

    printf("test_crop_refinement: all %d checks passed\n", g_checks);
    fs::remove_all(root);
    return 0;
}
