// test_mujoco_geometry.cpp — Verify joint connectivity is maintained after IK
//
// For each parent-child body pair connected by a joint, verify that the
// child body origin remains at the expected position relative to the parent
// (i.e., the joint is not "broken" — the kinematic chain is intact).
// This checks both before and after IK solving with real data.

#ifdef RED_HAS_MUJOCO

#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "skeleton.h"
#include "annotation.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

static double dist3(const double *a, const double *b) {
    double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// For each geom, compute the world-space endpoints of capsules.
// A capsule geom has: pos (center), mat (3x3 rotation), size[0]=radius, size[1]=half-length.
// The two endpoints are: pos ± half_length * z_axis (column 2 of mat).
struct CapsuleEndpoints {
    std::string name;
    double p1[3], p2[3]; // endpoints
    double radius;
    int bodyid;
};

static std::vector<CapsuleEndpoints> get_capsule_endpoints(const mjModel *m, const mjData *d) {
    std::vector<CapsuleEndpoints> caps;
    for (int g = 0; g < (int)m->ngeom; g++) {
        if (m->geom_type[g] != mjGEOM_CAPSULE) continue;
        CapsuleEndpoints c;
        const char *name = mj_id2name(m, mjOBJ_GEOM, g);
        c.name = name ? name : ("geom" + std::to_string(g));
        c.bodyid = m->geom_bodyid[g];
        c.radius = m->geom_size[3*g + 0];
        double half_len = m->geom_size[3*g + 1];

        // World-space geom position and orientation
        double *pos = d->geom_xpos + 3*g;
        double *mat = d->geom_xmat + 9*g; // 3x3 row-major

        // Z-axis of the geom frame (column 2 of rotation matrix)
        double zx = mat[2], zy = mat[5], zz = mat[8];

        c.p1[0] = pos[0] + half_len * zx; c.p1[1] = pos[1] + half_len * zy; c.p1[2] = pos[2] + half_len * zz;
        c.p2[0] = pos[0] - half_len * zx; c.p2[1] = pos[1] - half_len * zy; c.p2[2] = pos[2] - half_len * zz;
        caps.push_back(c);
    }
    return caps;
}

// Check that adjacent bodies in a kinematic chain have overlapping
// or very close capsule endpoints. If a joint "breaks", the child
// body's capsules will be far from the parent's capsule endpoints.
static void check_limb_chain(const mjModel *m, const mjData *d,
                             const char *label,
                             const std::vector<const char*> &body_names) {
    printf("  Chain: %s\n", label);

    for (size_t i = 0; i + 1 < body_names.size(); i++) {
        int parent_id = mj_name2id(m, mjOBJ_BODY, body_names[i]);
        int child_id = mj_name2id(m, mjOBJ_BODY, body_names[i+1]);
        if (parent_id < 0 || child_id < 0) {
            printf("    SKIP: %s or %s not found\n", body_names[i], body_names[i+1]);
            continue;
        }

        // Get world position of both body origins
        double parent_pos[3] = {d->xpos[3*parent_id], d->xpos[3*parent_id+1], d->xpos[3*parent_id+2]};
        double child_pos[3] = {d->xpos[3*child_id], d->xpos[3*child_id+1], d->xpos[3*child_id+2]};

        // The child body origin should be close to the parent body
        // (within the parent's capsule extent + some tolerance).
        // Find the closest capsule endpoint of the parent to the child origin.
        double min_dist = 1e9;
        auto caps = get_capsule_endpoints(m, d);
        for (size_t ci = 0; ci < caps.size(); ci++) {
            if (caps[ci].bodyid != parent_id) continue;
            double d1 = dist3(caps[ci].p1, child_pos);
            double d2 = dist3(caps[ci].p2, child_pos);
            min_dist = std::min(min_dist, std::min(d1, d2));
        }

        // Also check body-to-body distance directly
        double body_dist = dist3(parent_pos, child_pos);

        // A connected joint should keep bodies within ~50mm of each other
        // (the largest bone in the rat is ~40mm)
        bool connected = (body_dist < 0.06); // 60mm tolerance
        std::string msg = std::string(body_names[i]) + " -> " + body_names[i+1] +
                          " (dist=" + std::to_string(body_dist*1000).substr(0,5) + "mm)";
        CHECK(connected, msg.c_str());
        if (!connected) {
            printf("    BROKEN: %s -> %s: body dist=%.1fmm\n",
                   body_names[i], body_names[i+1], body_dist*1000);
        }
    }
}

int main() {
    printf("=== Joint Geometry Integrity Test ===\n\n");

    SkeletonContext skeleton;
    skeleton_initialize("Rat24", &skeleton, Rat24);

    MujocoContext mj;
    mj.load("models/rodent/rodent_no_collision.xml", skeleton);

    // Define kinematic chains to check
    std::vector<const char*> left_arm = {"scapula_L", "upper_arm_L", "lower_arm_L", "hand_L", "finger_L"};
    std::vector<const char*> right_arm = {"scapula_R", "upper_arm_R", "lower_arm_R", "hand_R", "finger_R"};
    std::vector<const char*> left_leg = {"pelvis", "upper_leg_L", "lower_leg_L", "foot_L", "toe_L"};
    std::vector<const char*> right_leg = {"pelvis", "upper_leg_R", "lower_leg_R", "foot_R", "toe_R"};
    std::vector<const char*> spine = {"torso", "vertebra_1", "vertebra_2", "vertebra_3",
                                      "vertebra_4", "vertebra_5", "vertebra_6", "pelvis"};

    // --- Test at default pose ---
    printf("--- Default pose ---\n");
    std::copy(mj.model->qpos0, mj.model->qpos0 + (int)mj.model->nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    check_limb_chain(mj.model, mj.data, "Left arm", left_arm);
    check_limb_chain(mj.model, mj.data, "Right arm", right_arm);
    check_limb_chain(mj.model, mj.data, "Left leg", left_leg);
    check_limb_chain(mj.model, mj.data, "Right leg", right_leg);
    check_limb_chain(mj.model, mj.data, "Spine", spine);

    int default_pass = g_pass, default_fail = g_fail;
    printf("  Default pose: %d passed, %d failed\n\n", default_pass, default_fail);

    // --- Test after IK with real data ---
    printf("--- After IK on real data (5 frames) ---\n");

    std::string csv = "/Users/johnsonr/datasets/rat/mujoco_test/"
                      "labeled_data/2025_11_04_06_17_13/keypoints3d.csv";
    std::ifstream f(csv);
    std::string hdr, line;
    std::getline(f, hdr);

    MujocoIKState ik;
    ik.max_iterations = 200;
    mj.scale_factor = 0.0f;

    int frames_tested = 0;
    while (std::getline(f, line) && frames_tested < 5) {
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        int fn = std::stoi(tok);

        std::vector<Keypoint3D> kp(24);
        while (std::getline(ss, tok, ',')) {
            int idx = std::stoi(tok);
            std::string a,b,c;
            std::getline(ss,a,','); std::getline(ss,b,','); std::getline(ss,c,',');
            if (idx<0||idx>=24) continue;
            double x=std::stod(a),y=std::stod(b),z=std::stod(c);
            if (std::abs(x)<1e6) kp[idx]={x,y,z,true,1.0f};
        }

        mujoco_ik_solve(mj, ik, kp.data(), 24, fn);
        printf("\n  Frame %d (residual=%.1fmm, %d iters):\n", fn,
               ik.final_residual*1000, ik.iterations_used);

        check_limb_chain(mj.model, mj.data, "Left arm", left_arm);
        check_limb_chain(mj.model, mj.data, "Right arm", right_arm);
        check_limb_chain(mj.model, mj.data, "Left leg", left_leg);
        check_limb_chain(mj.model, mj.data, "Right leg", right_leg);
        check_limb_chain(mj.model, mj.data, "Spine", spine);

        frames_tested++;
    }

    printf("\n==============================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("==============================\n");
    return g_fail > 0 ? 1 : 0;
}

#else
#include <stdio.h>
int main() { printf("MuJoCo not available\n"); return 0; }
#endif
