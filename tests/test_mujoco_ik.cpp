// test_mujoco_ik.cpp — Comprehensive MuJoCo IK integration tests
//
// Tests:
// 1. Model loading and site mapping (24/24)
// 2. Free joint was added via mjSpec
// 3. Round-trip IK convergence (synthetic)
// 4. Joint limits are respected
// 5. Root position converges to target centroid
// 6. Real data IK with centroid alignment
// 7. Warm-start improves consecutive frame performance
// 8. Sparse keypoints (partial visibility)
// 9. Zero keypoints (graceful failure)
// 10. Scale factor affects target positions

#ifdef RED_HAS_MUJOCO

#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "skeleton.h"
#include "annotation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <cassert>

static int g_pass = 0, g_fail = 0, g_warn = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

#define WARN(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_warn++; printf("  WARN: %s\n", msg); } \
} while(0)

// Parse one line of keypoints3d.csv (Rat24Target format)
static bool parse_kp3d_line(const std::string &line, int &frame_num,
                            std::vector<Keypoint3D> &kp3d, int num_nodes) {
    kp3d.resize(num_nodes);
    for (auto &k : kp3d) { k.x = UNLABELED; k.y = UNLABELED; k.z = UNLABELED; k.triangulated = false; }
    std::istringstream ss(line);
    std::string token;
    if (!std::getline(ss, token, ',')) return false;
    frame_num = std::stoi(token);
    while (std::getline(ss, token, ',')) {
        int idx = std::stoi(token);
        std::string sx, sy, sz;
        if (!std::getline(ss, sx, ',')) break;
        if (!std::getline(ss, sy, ',')) break;
        if (!std::getline(ss, sz, ',')) break;
        if (idx < 0 || idx >= num_nodes) continue;
        double x = std::stod(sx), y = std::stod(sy), z = std::stod(sz);
        if (std::abs(x) < 1e6 && std::abs(y) < 1e6 && std::abs(z) < 1e6) {
            kp3d[idx].x = x; kp3d[idx].y = y; kp3d[idx].z = z;
            kp3d[idx].triangulated = true; kp3d[idx].confidence = 1.0f;
        }
    }
    return true;
}

int main() {
    printf("=== MuJoCo IK Comprehensive Test Suite ===\n\n");

    SkeletonContext skeleton;
    skeleton_initialize("Rat24", &skeleton, Rat24);

    MujocoContext mj;
    bool ok = mj.load("models/rodent/rodent_no_collision.xml", skeleton);

    // ---------------------------------------------------------------
    printf("--- Test 1: Model loading and site mapping ---\n");
    CHECK(ok, "Model loaded successfully");
    CHECK(mj.loaded, "mj.loaded == true");
    CHECK(mj.model != nullptr, "mj.model != nullptr");
    CHECK(mj.data != nullptr, "mj.data != nullptr");
    CHECK(mj.mapped_count == 24, "24/24 sites mapped");
    // Verify every skeleton node has a valid mapping
    bool all_mapped = true;
    for (int i = 0; i < 24; i++) {
        if (mj.skeleton_to_site[i] < 0) {
            printf("  FAIL: node %d (%s) unmapped\n", i, skeleton.node_names[i].c_str());
            all_mapped = false;
        }
    }
    CHECK(all_mapped, "Every skeleton node has a site mapping");
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 2: Free joint added via mjSpec ---\n");
    CHECK(mj.has_free_joint, "has_free_joint flag set");
    // Find the free joint in the compiled model
    bool found_free = false;
    int free_jnt_idx = -1;
    for (int j = 0; j < (int)mj.model->njnt; j++) {
        if (mj.model->jnt_type[j] == mjJNT_FREE) {
            found_free = true;
            free_jnt_idx = j;
            break;
        }
    }
    CHECK(found_free, "Free joint exists in compiled model");
    int nq_expected = 61 + 7; // original 61 hinge DOFs + 7 free joint qpos
    int nv_expected = 61 + 6; // original 61 hinge DOFs + 6 free joint DOFs
    CHECK((int)mj.model->nq == nq_expected,
          ("nq == " + std::to_string(nq_expected) + " (got " + std::to_string((int)mj.model->nq) + ")").c_str());
    CHECK((int)mj.model->nv == nv_expected,
          ("nv == " + std::to_string(nv_expected) + " (got " + std::to_string((int)mj.model->nv) + ")").c_str());
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 3: Round-trip IK (synthetic) ---\n");
    // Get default pose site positions as targets
    std::copy(mj.model->qpos0, mj.model->qpos0 + (int)mj.model->nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    std::vector<Keypoint3D> target_kp(24);
    for (int n = 0; n < 24; n++) {
        int s = mj.skeleton_to_site[n];
        if (s >= 0) {
            target_kp[n].x = mj.data->site_xpos[3*s+0];
            target_kp[n].y = mj.data->site_xpos[3*s+1];
            target_kp[n].z = mj.data->site_xpos[3*s+2];
            target_kp[n].triangulated = true;
        }
    }

    // Perturb qpos (skip free joint DOFs at start)
    int nq = (int)mj.model->nq;
    int free_offset = mj.has_free_joint ? 7 : 0;
    for (int i = free_offset; i < nq; i++)
        mj.data->qpos[i] += 0.03 * sin(i * 1.7);
    mj_forward(mj.model, mj.data);

    MujocoIKState ik;
    ik.max_iterations = 200;
    ik.damping = 1e-4;
    bool conv = mujoco_ik_solve(mj, ik, target_kp.data(), 24, 0);
    CHECK(conv, "Round-trip converged");
    CHECK(ik.final_residual < 0.001,
          ("Residual < 1mm (got " + std::to_string(ik.final_residual * 1000) + " mm)").c_str());
    CHECK(ik.iterations_used < 100,
          ("Converged in < 100 iters (got " + std::to_string(ik.iterations_used) + ")").c_str());
    printf("  Info: %.4f mm residual, %d iters, %.1f ms\n",
           ik.final_residual * 1000, ik.iterations_used, ik.solve_time_ms);
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 4: Joint limits respected ---\n");
    // After IK solve, check all limited hinge joints are within range
    bool limits_ok = true;
    int violations = 0;
    for (int j = 0; j < (int)mj.model->njnt; j++) {
        if (!mj.model->jnt_limited[j]) continue;
        int type = mj.model->jnt_type[j];
        if (type != mjJNT_HINGE && type != mjJNT_SLIDE) continue;
        int qa = (int)mj.model->jnt_qposadr[j];
        double lo = mj.model->jnt_range[2*j];
        double hi = mj.model->jnt_range[2*j+1];
        double val = mj.data->qpos[qa];
        if (val < lo - 1e-6 || val > hi + 1e-6) {
            limits_ok = false;
            violations++;
            if (violations <= 3) {
                const char *name = mj_id2name(mj.model, mjOBJ_JOINT, j);
                printf("  VIOLATION: %s = %.4f, range [%.4f, %.4f]\n",
                       name ? name : "?", val, lo, hi);
            }
        }
    }
    CHECK(limits_ok, ("All joint limits respected (" + std::to_string(violations) + " violations)").c_str());
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 5: Root position converges to target centroid ---\n");
    // Create targets offset at (1.0, 0.5, 0.2) from default
    mujoco_ik_reset(ik);
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    std::vector<Keypoint3D> offset_kp(24);
    double shift[3] = {1.0, 0.5, 0.2};
    for (int n = 0; n < 24; n++) {
        int s = mj.skeleton_to_site[n];
        if (s >= 0) {
            offset_kp[n].x = mj.data->site_xpos[3*s+0] + shift[0];
            offset_kp[n].y = mj.data->site_xpos[3*s+1] + shift[1];
            offset_kp[n].z = mj.data->site_xpos[3*s+2] + shift[2];
            offset_kp[n].triangulated = true;
        }
    }

    ik.max_iterations = 100;
    ik.damping = 1e-3;
    conv = mujoco_ik_solve(mj, ik, offset_kp.data(), 24, 100);

    // Check that root moved close to the offset
    if (free_jnt_idx >= 0) {
        int qa = (int)mj.model->jnt_qposadr[free_jnt_idx];
        double rx = mj.data->qpos[qa+0], ry = mj.data->qpos[qa+1], rz = mj.data->qpos[qa+2];
        double root_dist = sqrt((rx-shift[0])*(rx-shift[0]) + (ry-shift[1])*(ry-shift[1]) + (rz-shift[2])*(rz-shift[2]));
        CHECK(root_dist < 0.1,
              ("Root within 100mm of target centroid (dist=" + std::to_string(root_dist*1000) + " mm)").c_str());
        printf("  Info: root at (%.3f, %.3f, %.3f), target centroid ~(%.1f, %.1f, %.1f)\n",
               rx, ry, rz, shift[0], shift[1], shift[2]);
    }
    CHECK(ik.final_residual < 0.01,
          ("Offset residual < 10mm (got " + std::to_string(ik.final_residual*1000) + " mm)").c_str());
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 6: Real data IK ---\n");
    std::string csv_path = "/Users/johnsonr/datasets/rat/sessions/2025_09_03_15_18_21/"
                           "trials/2025_10_18_17_06_27/red/Rat24Target/"
                           "labeled_data/2025_11_04_06_17_13/keypoints3d.csv";
    std::ifstream csv(csv_path);
    if (!csv.is_open()) {
        printf("  SKIP: Could not open real data CSV\n\n");
    } else {
        std::string header;
        std::getline(csv, header);

        // Reset state
        std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
        mj_forward(mj.model, mj.data);
        mj.scale_factor = 1.0f;

        MujocoIKState real_ik;
        real_ik.max_iterations = 50;
        real_ik.damping = 1e-3;

        int num_frames = 0;
        double total_residual = 0, total_time = 0;
        double min_res = 1e9, max_res = 0;
        double first_res = 0;

        std::string line;
        while (std::getline(csv, line) && num_frames < 50) {
            int fn;
            std::vector<Keypoint3D> kp3d;
            if (!parse_kp3d_line(line, fn, kp3d, 24)) continue;
            int tri = 0;
            for (auto &k : kp3d) if (k.triangulated) tri++;
            if (tri < 10) continue;
            // Convert mm to meters
            for (auto &k : kp3d) if (k.triangulated) { k.x /= 1000; k.y /= 1000; k.z /= 1000; }

            mujoco_ik_solve(mj, real_ik, kp3d.data(), 24, fn);
            if (num_frames == 0) first_res = real_ik.final_residual;
            num_frames++;
            total_residual += real_ik.final_residual;
            total_time += real_ik.solve_time_ms;
            if (real_ik.final_residual < min_res) min_res = real_ik.final_residual;
            if (real_ik.final_residual > max_res) max_res = real_ik.final_residual;
        }
        csv.close();

        double avg_res = total_residual / num_frames;
        double avg_time = total_time / num_frames;
        printf("  %d frames: avg=%.1fmm, min=%.1fmm, max=%.1fmm, avg_time=%.0fms\n",
               num_frames, avg_res*1000, min_res*1000, max_res*1000, avg_time);

        CHECK(avg_res < 0.1, ("Avg residual < 100mm (got " + std::to_string(avg_res*1000) + " mm)").c_str());
        CHECK(avg_time < 200, ("Avg time < 200ms (got " + std::to_string(avg_time) + " ms)").c_str());
        WARN(avg_res < 0.05, ("Avg residual < 50mm (got " + std::to_string(avg_res*1000) + " mm)").c_str());
        printf("\n");
    }

    // ---------------------------------------------------------------
    printf("--- Test 7: Warm-start improves consecutive frames ---\n");
    // Use offset targets so cold-start has real work to do
    mujoco_ik_reset(ik);
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    // Create targets that require non-trivial solve (offset + perturbed)
    std::vector<Keypoint3D> hard_kp(24);
    for (int n = 0; n < 24; n++) {
        int s = mj.skeleton_to_site[n];
        if (s >= 0) {
            hard_kp[n].x = mj.data->site_xpos[3*s+0] + 0.5 + 0.01 * sin(n);
            hard_kp[n].y = mj.data->site_xpos[3*s+1] + 0.3;
            hard_kp[n].z = mj.data->site_xpos[3*s+2] + 0.1 + 0.005 * cos(n);
            hard_kp[n].triangulated = true;
        }
    }

    // Cold start
    ik.max_iterations = 100;
    ik.damping = 1e-3;
    mujoco_ik_solve(mj, ik, hard_kp.data(), 24, 200);
    int cold_iters = ik.iterations_used;
    double cold_time = ik.solve_time_ms;

    // Warm start with slightly changed targets (simulating next frame)
    std::vector<Keypoint3D> hard_kp2 = hard_kp;
    for (auto &k : hard_kp2) if (k.triangulated) { k.x += 0.001; k.y += 0.0005; }
    mujoco_ik_solve(mj, ik, hard_kp2.data(), 24, 201);
    int warm_iters = ik.iterations_used;
    double warm_time = ik.solve_time_ms;

    printf("  Cold: %d iters, %.1f ms | Warm: %d iters, %.1f ms\n",
           cold_iters, cold_time, warm_iters, warm_time);
    CHECK(warm_iters <= cold_iters, "Warm-start uses <= iterations than cold-start");
    CHECK(ik.has_warm_start, "Warm-start state preserved");
    CHECK(ik.prev_frame == 201, "prev_frame updated to 201");
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 8: Sparse keypoints (partial visibility) ---\n");
    mujoco_ik_reset(ik);
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    // Only 6 keypoints visible (head + spine)
    std::vector<Keypoint3D> sparse_kp(24);
    int visible[] = {0, 1, 2, 3, 4, 5}; // Snout, EarL, EarR, Neck, SpineL, TailBase
    for (int i : visible) {
        int s = mj.skeleton_to_site[i];
        if (s >= 0) {
            sparse_kp[i].x = mj.data->site_xpos[3*s+0] + 0.01;
            sparse_kp[i].y = mj.data->site_xpos[3*s+1];
            sparse_kp[i].z = mj.data->site_xpos[3*s+2];
            sparse_kp[i].triangulated = true;
        }
    }

    ik.max_iterations = 100;
    ik.damping = 1e-3;
    conv = mujoco_ik_solve(mj, ik, sparse_kp.data(), 24, 300);
    CHECK(ik.active_sites == 6, ("Active sites == 6 (got " + std::to_string(ik.active_sites) + ")").c_str());
    CHECK(ik.final_residual < 0.05, ("Sparse residual < 50mm (got " + std::to_string(ik.final_residual*1000) + " mm)").c_str());
    printf("  Info: %d sites, residual %.1f mm, %d iters\n",
           ik.active_sites, ik.final_residual*1000, ik.iterations_used);
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 9: Zero keypoints (graceful failure) ---\n");
    std::vector<Keypoint3D> empty_kp(24); // all untriangulated
    mujoco_ik_reset(ik);
    bool result = mujoco_ik_solve(mj, ik, empty_kp.data(), 24, 400);
    CHECK(!result, "Returns false with zero keypoints");
    CHECK(ik.active_sites == 0, "Active sites == 0");
    CHECK(ik.iterations_used == 0, "Zero iterations performed");
    printf("\n");

    // ---------------------------------------------------------------
    printf("--- Test 10: Scale factor affects targets ---\n");
    mujoco_ik_reset(ik);
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    // Solve with scale=1.0 (default)
    mj.scale_factor = 1.0f;
    ik.max_iterations = 50;
    mujoco_ik_solve(mj, ik, target_kp.data(), 24, 500);
    double res_1x = ik.final_residual;

    // Solve with scale=2.0 (targets doubled — should be much worse)
    mujoco_ik_reset(ik);
    std::copy(mj.model->qpos0, mj.model->qpos0 + nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);
    mj.scale_factor = 2.0f;
    mujoco_ik_solve(mj, ik, target_kp.data(), 24, 501);
    double res_2x = ik.final_residual;

    mj.scale_factor = 1.0f; // restore
    printf("  Scale 1.0: %.4f mm | Scale 2.0: %.4f mm\n", res_1x*1000, res_2x*1000);
    CHECK(res_2x > res_1x, "Scale 2.0 produces higher residual than 1.0");
    CHECK(res_1x < 0.001, "Scale 1.0 on default pose converges to ~0");
    printf("\n");

    // ---------------------------------------------------------------
    printf("==============================\n");
    printf("Results: %d passed, %d failed, %d warnings\n", g_pass, g_fail, g_warn);
    printf("==============================\n");
    return g_fail > 0 ? 1 : 0;
}

#else // !RED_HAS_MUJOCO

#include <stdio.h>
int main() {
    printf("MuJoCo not available — skipping IK tests\n");
    return 0;
}

#endif
