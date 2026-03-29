// test_stac_comparison.cpp — Compare IK_dm_control vs STAC-calibrated IK
//
// Loads the rodent model, reads 3D keypoints from tiny_project, runs:
//   1. Baseline IK (IK_dm_control) on all frames
//   2. STAC calibration (3 rounds, up to 200 frames)
//   3. IK with STAC offsets on the same frames
// Prints a comparison table of per-frame residual statistics.

#ifdef RED_HAS_MUJOCO

#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "mujoco_stac.h"
#include "skeleton.h"
#include "annotation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

// ── Parse one line of keypoints3d.csv (v2 columnar format) ──────────────
// Format: frame,x0,y0,z0,c0,x1,y1,z1,c1,...,x24,y24,z24,c24
// Each keypoint has x, y, z, confidence (confidence may be empty).
static bool parse_kp3d_v2(const std::string &line, int &frame_num,
                          std::vector<Keypoint3D> &kp3d, int num_nodes) {
    kp3d.resize(num_nodes);
    for (auto &k : kp3d) {
        k.x = UNLABELED; k.y = UNLABELED; k.z = UNLABELED;
        k.triangulated = false; k.confidence = 0.0f;
    }

    std::istringstream ss(line);
    std::string token;

    // First field: frame number
    if (!std::getline(ss, token, ',')) return false;
    try { frame_num = std::stoi(token); }
    catch (...) { return false; }

    // Remaining fields: x,y,z,c per keypoint
    for (int i = 0; i < num_nodes; i++) {
        std::string sx, sy, sz, sc;
        if (!std::getline(ss, sx, ',')) break;
        if (!std::getline(ss, sy, ',')) break;
        if (!std::getline(ss, sz, ',')) break;
        if (!std::getline(ss, sc, ',')) break;  // confidence (may be empty)

        if (sx.empty() || sy.empty() || sz.empty()) continue;
        try {
            double x = std::stod(sx), y = std::stod(sy), z = std::stod(sz);
            if (std::abs(x) < 1e6 && std::abs(y) < 1e6 && std::abs(z) < 1e6) {
                kp3d[i].x = x; kp3d[i].y = y; kp3d[i].z = z;
                kp3d[i].triangulated = true;
                kp3d[i].confidence = sc.empty() ? 1.0f : (float)std::stod(sc);
            }
        } catch (...) { continue; }
    }
    return true;
}

// ── Statistics helper ───────────────────────────────────────────────────
struct Stats {
    double mean, median, min_val, max_val, stddev;
    int count;
};

static Stats compute_stats(std::vector<double> &vals) {
    Stats s{};
    s.count = (int)vals.size();
    if (s.count == 0) { return s; }

    std::sort(vals.begin(), vals.end());
    s.min_val = vals.front();
    s.max_val = vals.back();
    s.median = (s.count % 2 == 1)
        ? vals[s.count / 2]
        : 0.5 * (vals[s.count / 2 - 1] + vals[s.count / 2]);

    double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    s.mean = sum / s.count;

    double sq_sum = 0.0;
    for (double v : vals) sq_sum += (v - s.mean) * (v - s.mean);
    s.stddev = std::sqrt(sq_sum / s.count);

    return s;
}

// ── Main ────────────────────────────────────────────────────────────────
int main() {
    printf("=== STAC Comparison Test: IK_dm_control vs IK+STAC ===\n\n");

    // --- Setup skeleton and model ---
    SkeletonContext skeleton;
    skeleton_initialize("Rat24Target", &skeleton, Rat24Target);
    printf("Skeleton: %s (%d nodes)\n", skeleton.name.c_str(), skeleton.num_nodes);

    MujocoContext mj;
    bool ok = mj.load("models/rodent/rodent_no_collision.xml", skeleton);
    if (!ok) {
        printf("FATAL: Failed to load MuJoCo model: %s\n", mj.load_error.c_str());
        return 1;
    }
    printf("Model loaded: %d sites mapped out of %d nodes\n\n", mj.mapped_count, skeleton.num_nodes);

    // --- Load keypoints from CSV ---
    std::string csv_path = "/Users/johnsonr/datasets/rat/tiny_project/"
                           "labeled_data/2025_11_04_06_17_13_v2/keypoints3d.csv";
    std::ifstream csv(csv_path);
    if (!csv.is_open()) {
        printf("FATAL: Could not open CSV: %s\n", csv_path.c_str());
        return 1;
    }

    int num_nodes = skeleton.num_nodes; // 25 for Rat24Target

    // Skip header lines (lines starting with '#' or containing column names)
    std::string line;
    while (std::getline(csv, line)) {
        if (line.empty() || line[0] == '#' || line.substr(0, 5) == "frame") continue;
        break; // first data line
    }

    // Parse all frames
    struct FrameEntry {
        int frame_num;
        std::vector<Keypoint3D> kp3d;
        int active_count;
    };
    std::vector<FrameEntry> frames;

    // Process first data line we already read
    auto process_line = [&](const std::string &l) {
        FrameEntry fe;
        if (!parse_kp3d_v2(l, fe.frame_num, fe.kp3d, num_nodes)) return;
        fe.active_count = 0;
        for (int i = 0; i < num_nodes; i++)
            if (fe.kp3d[i].triangulated) fe.active_count++;
        if (fe.active_count >= 4) // need at least 4 for meaningful IK
            frames.push_back(std::move(fe));
    };

    process_line(line);
    while (std::getline(csv, line)) {
        if (line.empty() || line[0] == '#') continue;
        process_line(line);
    }
    csv.close();

    printf("Loaded %d frames from CSV (with >= 4 triangulated keypoints)\n", (int)frames.size());
    if (frames.empty()) {
        printf("FATAL: No usable frames\n");
        return 1;
    }

    // Report keypoint coverage
    {
        int min_kp = 9999, max_kp = 0;
        double sum_kp = 0;
        for (auto &f : frames) {
            min_kp = std::min(min_kp, f.active_count);
            max_kp = std::max(max_kp, f.active_count);
            sum_kp += f.active_count;
        }
        printf("Keypoint coverage: min=%d, max=%d, avg=%.1f\n\n",
               min_kp, max_kp, sum_kp / frames.size());
    }

    int nq = (int)mj.model->nq;

    // =====================================================================
    // Phase 1: Baseline IK (IK_dm_control) on all frames
    // =====================================================================
    printf("--- Phase 1: Baseline IK (IK_dm_control) ---\n");
    auto t1_start = std::chrono::high_resolution_clock::now();

    MujocoIKState ik_baseline;
    ik_baseline.max_iterations = 5000;
    mj.scale_factor = 0.0f; // auto-detect (mm -> m)

    // Reset to default pose
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    std::vector<double> baseline_residuals;
    baseline_residuals.reserve(frames.size());
    double baseline_total_time = 0;

    for (size_t i = 0; i < frames.size(); i++) {
        auto &f = frames[i];
        mujoco_ik_solve(mj, ik_baseline, f.kp3d.data(), num_nodes, f.frame_num);
        baseline_residuals.push_back(ik_baseline.final_residual * 1000.0); // mm
        baseline_total_time += ik_baseline.solve_time_ms;
    }

    auto t1_end = std::chrono::high_resolution_clock::now();
    double phase1_sec = std::chrono::duration<double>(t1_end - t1_start).count();
    Stats baseline_stats = compute_stats(baseline_residuals);

    printf("  %d frames solved in %.1f s (%.1f ms/frame avg)\n",
           (int)frames.size(), phase1_sec, baseline_total_time / frames.size());
    printf("  Residuals (mm): mean=%.2f, median=%.2f, min=%.2f, max=%.2f, std=%.2f\n\n",
           baseline_stats.mean, baseline_stats.median,
           baseline_stats.min_val, baseline_stats.max_val, baseline_stats.stddev);

    // =====================================================================
    // Phase 2: STAC calibration
    // =====================================================================
    printf("--- Phase 2: STAC Calibration (3 rounds) ---\n");

    // Build AnnotationMap for STAC (it expects this format)
    AnnotationMap amap;
    int max_stac_frames = std::min((int)frames.size(), 200);
    for (int i = 0; i < max_stac_frames; i++) {
        auto &f = frames[i];
        FrameAnnotation fa;
        fa.frame_number = (u32)f.frame_num;
        fa.kp3d = f.kp3d;
        amap[(u32)f.frame_num] = std::move(fa);
    }

    // Reset model to original state (clear any warm-start effects)
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);
    mj.scale_factor = 0.0f; // auto-detect

    StacState stac;
    stac.n_iters = 3;
    stac.n_sample_frames = max_stac_frames;
    stac.q_max_iters = 300;
    stac.m_max_iters = 500;
    stac.m_lr = 5e-4;
    stac.m_momentum = 0.9;
    stac.m_reg_coef = 0.1;

    MujocoIKState ik_stac;
    ik_stac.max_iterations = 5000;

    auto t2_start = std::chrono::high_resolution_clock::now();
    bool stac_ok = stac_calibrate(mj, stac, ik_stac, amap, num_nodes);
    auto t2_end = std::chrono::high_resolution_clock::now();
    double phase2_sec = std::chrono::duration<double>(t2_end - t2_start).count();

    if (!stac_ok) {
        printf("  STAC calibration FAILED\n");
        return 1;
    }

    printf("  STAC completed in %.1f s using %d frames\n", phase2_sec, stac.frames_used);
    printf("  STAC internal residual: %.2f mm -> %.2f mm\n", stac.pre_residual, stac.post_residual);

    // Report offset magnitudes
    {
        double max_offset = 0, sum_offset = 0;
        int n_offsets = 0;
        for (int i = 0; i < (int)stac.site_offsets.size() / 3; i++) {
            double ox = stac.site_offsets[3*i+0];
            double oy = stac.site_offsets[3*i+1];
            double oz = stac.site_offsets[3*i+2];
            double mag = std::sqrt(ox*ox + oy*oy + oz*oz);
            if (mag > 1e-10) {
                sum_offset += mag;
                max_offset = std::max(max_offset, mag);
                n_offsets++;
            }
        }
        if (n_offsets > 0)
            printf("  Site offsets: %d non-zero, mean=%.4f m, max=%.4f m\n\n",
                   n_offsets, sum_offset / n_offsets, max_offset);
    }

    // =====================================================================
    // Phase 3: IK with STAC offsets on all frames
    // =====================================================================
    printf("--- Phase 3: IK with STAC offsets ---\n");
    auto t3_start = std::chrono::high_resolution_clock::now();

    // Offsets are already applied to model by stac_calibrate
    MujocoIKState ik_post;
    ik_post.max_iterations = 5000;
    mj.scale_factor = 0.0f; // auto-detect

    // Reset to default pose
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    std::vector<double> stac_residuals;
    stac_residuals.reserve(frames.size());
    double stac_total_time = 0;

    for (size_t i = 0; i < frames.size(); i++) {
        auto &f = frames[i];
        mujoco_ik_solve(mj, ik_post, f.kp3d.data(), num_nodes, f.frame_num);
        stac_residuals.push_back(ik_post.final_residual * 1000.0); // mm
        stac_total_time += ik_post.solve_time_ms;
    }

    auto t3_end = std::chrono::high_resolution_clock::now();
    double phase3_sec = std::chrono::duration<double>(t3_end - t3_start).count();
    Stats stac_stats = compute_stats(stac_residuals);

    printf("  %d frames solved in %.1f s (%.1f ms/frame avg)\n",
           (int)frames.size(), phase3_sec, stac_total_time / frames.size());
    printf("  Residuals (mm): mean=%.2f, median=%.2f, min=%.2f, max=%.2f, std=%.2f\n\n",
           stac_stats.mean, stac_stats.median,
           stac_stats.min_val, stac_stats.max_val, stac_stats.stddev);

    // =====================================================================
    // Comparison table
    // =====================================================================
    printf("=======================================================================\n");
    printf("                  IK_dm_control vs IK+STAC Comparison\n");
    printf("=======================================================================\n");
    printf("  %-12s  %12s  %12s  %12s\n", "Metric", "IK_dm_control", "IK+STAC", "Improvement");
    printf("  %-12s  %12s  %12s  %12s\n", "--------", "-------------", "-------", "-----------");
    printf("  %-12s  %10.2f mm  %10.2f mm  %+10.2f mm\n",
           "Mean", baseline_stats.mean, stac_stats.mean,
           baseline_stats.mean - stac_stats.mean);
    printf("  %-12s  %10.2f mm  %10.2f mm  %+10.2f mm\n",
           "Median", baseline_stats.median, stac_stats.median,
           baseline_stats.median - stac_stats.median);
    printf("  %-12s  %10.2f mm  %10.2f mm  %+10.2f mm\n",
           "Min", baseline_stats.min_val, stac_stats.min_val,
           baseline_stats.min_val - stac_stats.min_val);
    printf("  %-12s  %10.2f mm  %10.2f mm  %+10.2f mm\n",
           "Max", baseline_stats.max_val, stac_stats.max_val,
           baseline_stats.max_val - stac_stats.max_val);
    printf("  %-12s  %10.2f mm  %10.2f mm\n",
           "Std", baseline_stats.stddev, stac_stats.stddev);
    printf("  %-12s  %10.1f ms  %10.1f ms\n",
           "Avg time", baseline_total_time / frames.size(),
           stac_total_time / frames.size());
    printf("=======================================================================\n");

    double improvement_pct = 100.0 * (baseline_stats.mean - stac_stats.mean) / baseline_stats.mean;
    printf("  STAC calibration time: %.1f s\n", phase2_sec);
    printf("  Mean residual improvement: %.1f%%\n", improvement_pct);

    if (stac_stats.mean < baseline_stats.mean)
        printf("  Result: STAC IMPROVES over baseline\n");
    else
        printf("  Result: STAC does NOT improve over baseline\n");

    printf("=======================================================================\n");

    // Clean up STAC offsets
    stac_reset(mj, stac);

    return 0;
}

#else // !RED_HAS_MUJOCO

#include <stdio.h>
int main() {
    printf("MuJoCo not available — skipping STAC comparison test\n");
    return 0;
}

#endif
