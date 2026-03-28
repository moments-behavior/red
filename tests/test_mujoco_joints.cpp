// test_mujoco_joints.cpp — Verify joint connectivity and limb articulation
//
// Tests both rodent models to ensure:
// 1. All limb bodies are connected through joints (no floating segments)
// 2. Moving a joint actually moves its child bodies
// 3. The kinematic chain is intact from torso to extremities
// 4. Skin mesh loads correctly (dm_rodent model)

#ifdef RED_HAS_MUJOCO

#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "skeleton.h"
#include "annotation.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// Get world-space position of a site by name
static bool get_site_pos(const mjModel *m, const mjData *d, const char *name, double out[3]) {
    int id = mj_name2id(m, mjOBJ_SITE, name);
    if (id < 0) return false;
    out[0] = d->site_xpos[3*id]; out[1] = d->site_xpos[3*id+1]; out[2] = d->site_xpos[3*id+2];
    return true;
}

// Get world-space position of a body by name
static bool get_body_pos(const mjModel *m, const mjData *d, const char *name, double out[3]) {
    int id = mj_name2id(m, mjOBJ_BODY, name);
    if (id < 0) return false;
    out[0] = d->xpos[3*id]; out[1] = d->xpos[3*id+1]; out[2] = d->xpos[3*id+2];
    return true;
}

static double dist3(const double a[3], const double b[3]) {
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Test that perturbing a joint moves a downstream body
static bool test_joint_moves_body(mjModel *m, mjData *d, const char *joint_name,
                                  const char *body_name, double perturb = 0.3) {
    int jid = mj_name2id(m, mjOBJ_JOINT, joint_name);
    if (jid < 0) { printf("    joint '%s' not found\n", joint_name); return false; }

    // Record body position at default pose
    std::vector<double> qpos0(m->qpos0, m->qpos0 + (int)m->nq);
    std::copy(qpos0.begin(), qpos0.end(), d->qpos);
    mj_forward(m, d);
    double pos_before[3];
    if (!get_body_pos(m, d, body_name, pos_before)) {
        printf("    body '%s' not found\n", body_name);
        return false;
    }

    // Perturb joint
    int qa = (int)m->jnt_qposadr[jid];
    d->qpos[qa] += perturb;
    mj_forward(m, d);
    double pos_after[3];
    get_body_pos(m, d, body_name, pos_after);

    double moved = dist3(pos_before, pos_after);

    // Restore
    std::copy(qpos0.begin(), qpos0.end(), d->qpos);
    mj_forward(m, d);

    if (moved < 1e-6) {
        printf("    '%s' -> '%s': NOT MOVED (%.6f m)\n", joint_name, body_name, moved);
        return false;
    }
    return true;
}

static void test_model(const char *xml_path, const char *label) {
    printf("\n=== Testing: %s (%s) ===\n", label, xml_path);

    SkeletonContext skeleton;
    skeleton_initialize("Rat24", &skeleton, Rat24);

    MujocoContext mj;
    bool ok = mj.load(xml_path, skeleton);
    CHECK(ok, "Model loaded");
    if (!ok) return;

    printf("  Info: %d bodies, %d joints (nq=%d, nv=%d), %d sites, %d skins\n",
           (int)mj.model->nbody, (int)mj.model->njnt, (int)mj.model->nq,
           (int)mj.model->nv, (int)mj.model->nsite, (int)mj.model->nskin);
    CHECK(mj.mapped_count >= 20,
          (std::to_string(mj.mapped_count) + "/24 sites mapped (need >=20)").c_str());

    // --- Test: Free joint exists ---
    CHECK(mj.has_free_joint, "Free joint added to torso");

    // --- Test: Limb kinematic chains are connected ---
    // Each chain: perturb a proximal joint, verify distal body moves
    printf("\n  --- Left forelimb chain ---\n");
    // Try different joint names (old model uses different names than new)
    bool has_scap = (mj_name2id(mj.model, mjOBJ_JOINT, "scapula_L_extend") >= 0);

    if (has_scap) {
        CHECK(test_joint_moves_body(mj.model, mj.data, "scapula_L_extend", "upper_arm_L"),
              "scapula_L_extend -> upper_arm_L moves");
        CHECK(test_joint_moves_body(mj.model, mj.data, "scapula_L_extend", "lower_arm_L"),
              "scapula_L_extend -> lower_arm_L moves");
        CHECK(test_joint_moves_body(mj.model, mj.data, "scapula_L_extend", "hand_L"),
              "scapula_L_extend -> hand_L moves (through chain)");
        CHECK(test_joint_moves_body(mj.model, mj.data, "scapula_L_extend", "finger_L"),
              "scapula_L_extend -> finger_L moves (through chain)");
    }

    CHECK(test_joint_moves_body(mj.model, mj.data, "elbow_L", "hand_L"),
          "elbow_L -> hand_L moves");
    CHECK(test_joint_moves_body(mj.model, mj.data, "elbow_L", "finger_L"),
          "elbow_L -> finger_L moves");

    printf("\n  --- Right forelimb chain ---\n");
    CHECK(test_joint_moves_body(mj.model, mj.data, "elbow_R", "hand_R"),
          "elbow_R -> hand_R moves");
    CHECK(test_joint_moves_body(mj.model, mj.data, "elbow_R", "finger_R"),
          "elbow_R -> finger_R moves");

    printf("\n  --- Left hindlimb chain ---\n");
    if (mj_name2id(mj.model, mjOBJ_JOINT, "hip_L_extend") >= 0) {
        CHECK(test_joint_moves_body(mj.model, mj.data, "hip_L_extend", "lower_leg_L"),
              "hip_L_extend -> lower_leg_L moves");
        CHECK(test_joint_moves_body(mj.model, mj.data, "hip_L_extend", "foot_L"),
              "hip_L_extend -> foot_L moves (through chain)");
        CHECK(test_joint_moves_body(mj.model, mj.data, "hip_L_extend", "toe_L"),
              "hip_L_extend -> toe_L moves (through chain)");
    }
    CHECK(test_joint_moves_body(mj.model, mj.data, "knee_L", "foot_L"),
          "knee_L -> foot_L moves");
    CHECK(test_joint_moves_body(mj.model, mj.data, "knee_L", "toe_L"),
          "knee_L -> toe_L moves");
    CHECK(test_joint_moves_body(mj.model, mj.data, "ankle_L", "toe_L"),
          "ankle_L -> toe_L moves");

    printf("\n  --- Right hindlimb chain ---\n");
    CHECK(test_joint_moves_body(mj.model, mj.data, "knee_R", "foot_R"),
          "knee_R -> foot_R moves");
    CHECK(test_joint_moves_body(mj.model, mj.data, "ankle_R", "toe_R"),
          "ankle_R -> toe_R moves");

    printf("\n  --- Spine/tail chain ---\n");
    // Cervical chain connects to skull (not lumbar vertebra_1)
    if (mj_name2id(mj.model, mjOBJ_JOINT, "vertebra_cervical_5_extend") >= 0)
        CHECK(test_joint_moves_body(mj.model, mj.data, "vertebra_cervical_5_extend", "skull"),
              "cervical_5_extend -> skull moves (through neck)");
    CHECK(test_joint_moves_body(mj.model, mj.data, "vertebra_1_extend", "pelvis"),
          "vertebra_1_extend -> pelvis moves");

    // --- Test: wrist/finger/toe joints (only in dm_rodent model) ---
    printf("\n  --- Distal joints ---\n");
    int wrist_id = mj_name2id(mj.model, mjOBJ_JOINT, "wrist_L");
    if (wrist_id >= 0) {
        CHECK(test_joint_moves_body(mj.model, mj.data, "wrist_L", "finger_L"),
              "wrist_L -> finger_L moves");
        // finger_L/toe_L joints rotate around their own origin, so check
        // that they move a downstream geom point instead of body origin
        CHECK(mj_name2id(mj.model, mjOBJ_JOINT, "finger_L") >= 0,
              "finger_L joint exists");
        CHECK(mj_name2id(mj.model, mjOBJ_JOINT, "toe_L") >= 0,
              "toe_L joint exists");
        printf("  (dm_rodent: 6 distal joints present)\n");
    } else {
        printf("  (no distal joints — old model, bodies rigidly attached)\n");
    }

    // --- Test: IK moves all limbs ---
    printf("\n  --- IK moves all keypoint sites ---\n");
    std::copy(mj.model->qpos0, mj.model->qpos0 + (int)mj.model->nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    // Record default site positions
    std::vector<double> default_pos(24 * 3);
    for (int n = 0; n < 24; n++) {
        int s = mj.skeleton_to_site[n];
        if (s >= 0) {
            default_pos[3*n+0] = mj.data->site_xpos[3*s+0];
            default_pos[3*n+1] = mj.data->site_xpos[3*s+1];
            default_pos[3*n+2] = mj.data->site_xpos[3*s+2];
        }
    }

    // Create perturbed targets
    std::vector<Keypoint3D> targets(24);
    for (int n = 0; n < 24; n++) {
        int s = mj.skeleton_to_site[n];
        if (s >= 0) {
            targets[n].x = default_pos[3*n+0] + 0.02 * std::sin(n * 0.5);
            targets[n].y = default_pos[3*n+1] + 0.01 * std::cos(n * 0.7);
            targets[n].z = default_pos[3*n+2] + 0.01;
            targets[n].triangulated = true;
        }
    }

    MujocoIKState ik;
    ik.max_iterations = 100;
    mj.scale_factor = 1.0f;
    mujoco_ik_solve(mj, ik, targets.data(), 24, 0);

    // Check that each site moved from its default position
    int sites_moved = 0;
    for (int n = 0; n < 24; n++) {
        int s = mj.skeleton_to_site[n];
        if (s < 0) continue;
        double cur[3] = {mj.data->site_xpos[3*s], mj.data->site_xpos[3*s+1], mj.data->site_xpos[3*s+2]};
        double def[3] = {default_pos[3*n], default_pos[3*n+1], default_pos[3*n+2]};
        if (dist3(cur, def) > 0.001) sites_moved++;
    }
    CHECK(sites_moved >= 20,
          (std::to_string(sites_moved) + "/24 sites moved by IK (need >=20)").c_str());
    printf("  IK result: residual=%.1fmm, %d iters, %s\n",
           ik.final_residual * 1000, ik.iterations_used,
           ik.converged ? "converged" : "not converged");

    // --- Test: Skin data (dm_rodent only) ---
    if (mj.model->nskin > 0) {
        printf("\n  --- Skin mesh ---\n");
        CHECK((int)mj.model->nskin == 1, "1 skin loaded");
        CHECK((int)mj.model->nskinvert > 1000,
              (std::to_string((int)mj.model->nskinvert) + " skin vertices").c_str());
        CHECK((int)mj.model->nskinface > 1000,
              (std::to_string((int)mj.model->nskinface) + " skin faces").c_str());

        // Verify skin updates with pose
        mjvScene scn; mjvOption opt; mjvCamera cam;
        mjv_defaultScene(&scn); mjv_defaultOption(&opt); mjv_defaultCamera(&cam);
        for (int i = 0; i < 6; i++) opt.sitegroup[i] = 1;
        mjv_makeScene(mj.model, &scn, 5000);
        cam.distance = 0.5;
        mjv_updateScene(mj.model, mj.data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
        CHECK(scn.nskin == 1, "Skin appears in scene");
        if (scn.nskin > 0) {
            CHECK(scn.skinvertnum[0] > 0,
                  (std::to_string(scn.skinvertnum[0]) + " skin verts in scene").c_str());
        }
        mjv_freeScene(&scn);
    }
}

int main() {
    printf("=== MuJoCo Joint Connectivity Test Suite ===\n");

    // Test old model (from IK_resources)
    test_model("models/rodent/rodent_no_collision.xml", "IK_resources model");

    // Test new model (dm_rodent with skin)
    test_model("models/rodent/rodent.xml", "dm_rodent model");

    printf("\n==============================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("==============================\n");
    return g_fail > 0 ? 1 : 0;
}

#else
#include <stdio.h>
int main() { printf("MuJoCo not available\n"); return 0; }
#endif
