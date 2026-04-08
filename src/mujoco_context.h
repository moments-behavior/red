#pragma once
// mujoco_context.h — MuJoCo model loading + site-to-skeleton mapping
//
// Optional dependency: compile with -DRED_HAS_MUJOCO to enable.
// Without MuJoCo, provides empty stubs so downstream code compiles cleanly.

#include "skeleton.h"
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

#ifdef RED_HAS_MUJOCO
#include <mujoco.h>
#endif

struct MujocoContext {
    bool loaded = false;
    std::string model_path;
    std::string load_error;

    // Bidirectional site <-> skeleton mapping.
    // site_to_skeleton[mj_site_idx] = skeleton_node_idx (-1 if unmapped)
    // skeleton_to_site[skeleton_node_idx] = mj_site_idx (-1 if unmapped)
    std::vector<int> site_to_skeleton;
    std::vector<int> skeleton_to_site;
    int mapped_count = 0;

#ifdef RED_HAS_MUJOCO
    mjModel  *model = nullptr;
    mjData   *data  = nullptr;
    mjvScene  scene = {};  // zero-init prevents uninitialized read in destructor
    mjvOption opt = {};
#endif

    // Global scale factor: scales the target keypoints to match the model.
    // 0 = auto-detect from data (recommended). Positive values override.
    float scale_factor = 0.0f;

    // Whether a free joint was added to the root body
    bool has_free_joint = false;

    // Uniform model scale (1.0 = no scaling). Applied to body positions,
    // geom sizes, site positions, joint positions, and mesh vertices.
    // Mass/inertia are recomputed from density (spec path) or scaled
    // by s^3/s^5 (compiled model path). Set before or after loading.
    float model_scale = 1.0f;

    // Apply uniform scale to the compiled mjModel.
    // Matches Mason's scale_body: scales positions, geom sizes, meshes.
    // Mass scales by s^3, inertia by s^5 (uniform density assumption).
    void apply_model_scale(float s) {
#ifdef RED_HAS_MUJOCO
        if (!model || std::abs(s - 1.0f) < 1e-6f) return;
        float s3 = s * s * s, s5 = s3 * s * s;
        for (int i = 0; i < model->nbody; i++) {
            for (int c = 0; c < 3; c++) {
                model->body_pos[3*i+c] *= s;
                model->body_ipos[3*i+c] *= s;
                model->body_inertia[3*i+c] *= s5;
            }
            model->body_mass[i] *= s3;
        }
        for (int i = 0; i < model->njnt; i++)
            for (int c = 0; c < 3; c++)
                model->jnt_pos[3*i+c] *= s;
        for (int i = 0; i < model->ngeom; i++) {
            for (int c = 0; c < 3; c++) {
                model->geom_pos[3*i+c] *= s;
                model->geom_size[3*i+c] *= s;
            }
            model->geom_rbound[i] *= s;
            for (int c = 0; c < 6; c++)
                model->geom_aabb[6*i+c] *= s;
        }
        for (int i = 0; i < model->nsite; i++)
            for (int c = 0; c < 3; c++) {
                model->site_pos[3*i+c] *= s;
                model->site_size[3*i+c] *= s;
            }
        for (int i = 0; i < model->nmeshvert; i++)
            for (int c = 0; c < 3; c++)
                model->mesh_vert[3*i+c] *= s;
        // Update statistics
        for (int c = 0; c < 3; c++) model->stat.center[c] *= s;
        model->stat.extent *= s;
        model->stat.meansize *= s;
        model->stat.meanmass *= s3;
        model->stat.meaninertia *= s5;
        // Recompute derived constants and forward kinematics
        if (data) {
            mj_setConst(model, data);
            mj_forward(model, data);
        }
        model_scale *= s;  // cumulative: s is relative, model_scale is absolute
        std::cout << "[MuJoCo] Applied model scale: " << s << std::endl;
#endif
    }

    // Load a MuJoCo XML body model and build the site↔skeleton mapping.
    // Programmatically adds a free joint to the torso via mjSpec so the
    // IK solver can optimize root position and orientation.
    // Returns true on success.
    bool load(const std::string &xml_path_in, const SkeletonContext &skeleton) {
#ifdef RED_HAS_MUJOCO
        // Convert to absolute path — MuJoCo needs this to resolve
        // relative asset references (meshes, skins) on all platforms
        std::string xml_path = xml_path_in;
        {
            std::filesystem::path p(xml_path_in);
            if (p.is_relative()) {
                auto abs = std::filesystem::absolute(p);
                if (std::filesystem::exists(abs))
                    xml_path = abs.string();
            }
        }
        unload();

        // Support pre-compiled MJB files (binary, sites already injected)
        if (xml_path.size() > 4 &&
            xml_path.substr(xml_path.size() - 4) == ".mjb") {
            model = mj_loadModel(xml_path.c_str(), nullptr);
            if (!model) {
                load_error = "Failed to load MJB: " + xml_path;
                return false;
            }
            has_free_joint = false;
            for (int j = 0; j < model->njnt; j++) {
                if (model->jnt_type[j] == mjJNT_FREE) { has_free_joint = true; break; }
            }
            data = mj_makeData(model);
            if (!data) {
                load_error = "mj_makeData failed";
                mj_deleteModel(model); model = nullptr;
                return false;
            }
            // Initialize visualization
            mjv_defaultScene(&scene);
            mjv_defaultOption(&opt);
            for (int i = 0; i < mjNGROUP; i++) opt.sitegroup[i] = 1;
            mjv_makeScene(model, &scene, 2000);
            // Skip XML path — go directly to mapping
            return finalize_load(xml_path, skeleton);
        }

        char error_buf[1024] = {0};

        // Parse XML into mjSpec for programmatic editing
        mjSpec *spec = mj_parseXML(xml_path.c_str(), nullptr, error_buf, sizeof(error_buf));
        if (!spec) {
            load_error = std::string("MuJoCo XML parse failed: ") + error_buf;
            std::cerr << load_error << std::endl;
            return false;
        }

        // Add a free joint to the root body so the IK solver can
        // optimize root position and orientation (6 extra DOFs).
        // Rodent models use "torso", fly models use "thorax".
        has_free_joint = false;
        // Check if a free joint already exists (e.g., fruitfly v2 has one)
        if (mjs_findElement(spec, mjOBJ_JOINT, "free")) {
            has_free_joint = true;
            std::cout << "[MuJoCo] Model already has free joint" << std::endl;
        } else {
            for (const char *root_name : {"torso", "thorax"}) {
                mjsBody *root_body = mjs_findBody(spec, root_name);
                if (root_body) {
                    mjs_addFreeJoint(root_body);
                    has_free_joint = true;
                    std::cout << "[MuJoCo] Added free joint to '" << root_name
                              << "' body" << std::endl;
                    break;
                }
            }
        }
        if (!has_free_joint) {
            std::cerr << "[MuJoCo] WARNING: no 'torso' or 'thorax' body found — "
                         "root position will be fixed" << std::endl;
        }

        // Add missing keypoint sites for models that lack them.
        // We detect the model type (rodent vs fly) by checking for
        // characteristic body names, then inject the appropriate sites.
        struct SiteDef {
            const char *name; const char *kpsite; const char *body; double pos[3];
        };

        // Detect model type: fly models have "thorax" + "head" + "wing_left",
        // rodent models have "skull" + "torso".
        bool is_fly_model = (mjs_findBody(spec, "thorax") &&
                             mjs_findBody(spec, "head") &&
                             mjs_findBody(spec, "wing_left"));

        // ---- Rodent (rat24) site definitions ----
        // The rodent_no_collision.xml already has all 24 *_kpsite sites;
        // the dm_rodent rodent.xml does not. We check for both the plain
        // name ("nose") and the _kpsite variant ("nose_0_kpsite") before
        // adding, to avoid duplicates.
        static const SiteDef rodent_kp_sites[] = {
            {"nose",     "nose_0_kpsite",       "skull",        { 0.043700,  0.000000, -0.004600}},
            {"ear_L",    "ear_L_1_kpsite",      "skull",        {-0.012650,  0.014720,  0.014950}},
            {"ear_R",    "ear_R_2_kpsite",      "skull",        {-0.012650, -0.014720,  0.014950}},
            {"neck",     "neck_3_kpsite",       "torso",        { 0.028750,  0.000000,  0.023000}},
            {"spineL",   "spineL_4_kpsite",     "vertebra_5",   { 0.000000,  0.000000,  0.008050}},
            {"tailbase", "tailbase_5_kpsite",   "vertebra_C4",  { 0.000000,  0.000000,  0.012650}},
            {"hand_L",   "hand_L_9_kpsite",     "finger_L",     { 0.000000,  0.000000,  0.000000}},
            {"hand_R",   "hand_R_13_kpsite",    "finger_R",     { 0.000000,  0.000000,  0.000000}},
            {"foot_L",   "foot_L_16_kpsite",    "toe_L",        { 0.000000,  0.000000,  0.000000}},
            {"foot_R",   "foot_R_19_kpsite",    "toe_R",        { 0.000000,  0.000000,  0.000000}},
            {"tailtip",  "tailtip_20_kpsite",   "vertebra_C30", { 0.000000,  0.000000,  0.000000}},
            {"tailmid",  "tailmid_21_kpsite",   "vertebra_C17", { 0.000000,  0.000000,  0.000000}},
            {"tail1Q",   "tail1Q_22_kpsite",    "vertebra_C11", { 0.000000,  0.000000,  0.000000}},
            {"tail3Q",   "tail3Q_23_kpsite",    "vertebra_C23", { 0.000000,  0.000000,  0.000000}},
        };

        // ---- Fruit fly (fly50) site definitions ----
        // 50 keypoint sites matching the RED Fly50 skeleton, derived from
        // Mason Kamb's fly-body-tuning add_keypoint_sites() function.
        // The 6 TaTip sites use the tarsal_claw fromto endpoint positions
        // extracted from the fruitfly v2.1 XML model.
        //  idx  RED skeleton name    MuJoCo site name     parent body               local pos (x, y, z)
        //  ---  ------------------   ------------------   -----------------------   ---------------------
        //   0   Antenna_Base         Antenna_Base         head                      ( 0.0,      0.038,    0.012   )
        //   1   EyeL                 EyeL                 head                      (-0.0245,   0.0135,   0.0285  )
        //   2   EyeR                 EyeR                 head                      ( 0.0245,   0.0135,   0.0285  )
        //   3   Scutellum            Scutellum            thorax                    (-0.049,    0.0,      0.04    )
        //   4   Abd_A4               Abd_A4               abdomen_3                 ( 0.0,      0.0335,   0.021   )
        //   5   Abd_tip              Abd_tip              abdomen_7                 ( 0.0,      0.0395,  -0.001   )
        //   6   WingL_base           WingL_base           thorax                    (-0.0095,   0.045,    0.0175  )
        //   7   WingL_V12            WingL_V12            wing_left                 (-0.0072,  -0.2125,   0.0075  )
        //   8   WingL_V13            WingL_V13            wing_left                 ( 0.0221,  -0.2562,  -0.0253  )
        //   9   T1L_ThxCx            T1L_ThxCx            coxa_T1_left              ( 0.0,      0.0,      0.0     )
        //  10   T1L_Tro              T1L_Tro              femur_T1_left             ( 0.0,      0.0,      0.0     )
        //  11   T1L_FeTi             T1L_FeTi             tibia_T1_left             ( 0.0,      0.0,      0.0     )
        //  12   T1L_TiTa             T1L_TiTa             tarsus1_T1_left           ( 0.0,      0.0,      0.0     )
        //  13   T1L_TaT1             T1L_TaT1             tarsus2_T1_left           ( 0.0,      0.0,      0.0     )
        //  14   T1L_TaT3             T1L_TaT3             tarsus4_T1_left           ( 0.0,      0.0,      0.0     )
        //  15   T1L_TaTip            T1L_TaTip            tarsal_claw_T1_left       ( 0.0,      0.0105,   0.0006  )
        //  16   T2L_Tro              T2L_Tro              femur_T2_left             ( 0.0,      0.0,      0.0     )
        //  17   T2L_FeTi             T2L_FeTi             tibia_T2_left             ( 0.0,      0.0,      0.0     )
        //  18   T2L_TiTa             T2L_TiTa             tarsus1_T2_left           ( 0.0,      0.0,      0.0     )
        //  19   T2L_TaT1             T2L_TaT1             tarsus2_T2_left           ( 0.0,      0.0,      0.0     )
        //  20   T2L_TaT3             T2L_TaT3             tarsus4_T2_left           ( 0.0,      0.0,      0.0     )
        //  21   T2L_TaTip            T2L_TaTip            tarsal_claw_T2_left       ( 0.0,      0.0122,   0.0006  )
        //  22   T3L_Tro              T3L_Tro              femur_T3_left             ( 0.0,      0.0,      0.0     )
        //  23   T3L_FeTi             T3L_FeTi             tibia_T3_left             ( 0.0,      0.0,      0.0     )
        //  24   T3L_TiTa             T3L_TiTa             tarsus1_T3_left           ( 0.0,      0.0,      0.0     )
        //  25   T3L_TaT1             T3L_TaT1             tarsus2_T3_left           ( 0.0,      0.0,      0.0     )
        //  26   T3L_TaT3             T3L_TaT3             tarsus4_T3_left           ( 0.0,      0.0,      0.0     )
        //  27   T3L_TaTip            T3L_TaTip            tarsal_claw_T3_left       ( 0.0,      0.0111,   0.0008  )
        //  28   WingR_base           WingR_base           thorax                    (-0.0095,  -0.045,    0.0175  )
        //  29   WingR_V12            WingR_V12            wing_right                ( 0.0072,   0.2125,  -0.0075  )
        //  30   WingR_V13            WingR_V13            wing_right                (-0.0221,   0.2562,   0.0253  )
        //  31   T1R_ThxCx            T1R_ThxCx            coxa_T1_right             ( 0.0,      0.0,      0.0     )
        //  32   T1R_Tro              T1R_Tro              femur_T1_right            ( 0.0,      0.0,      0.0     )
        //  33   T1R_FeTi             T1R_FeTi             tibia_T1_right            ( 0.0,      0.0,      0.0     )
        //  34   T1R_TiTa             T1R_TiTa             tarsus1_T1_right          ( 0.0,      0.0,      0.0     )
        //  35   T1R_TaT1             T1R_TaT1             tarsus2_T1_right          ( 0.0,      0.0,      0.0     )
        //  36   T1R_TaT3             T1R_TaT3             tarsus4_T1_right          ( 0.0,      0.0,      0.0     )
        //  37   T1R_TaTip            T1R_TaTip            tarsal_claw_T1_right      ( 0.0,     -0.0101,  -0.0006  )
        //  38   T2R_Tro              T2R_Tro              femur_T2_right            ( 0.0,      0.0,      0.0     )
        //  39   T2R_FeTi             T2R_FeTi             tibia_T2_right            ( 0.0,      0.0,      0.0     )
        //  40   T2R_TiTa             T2R_TiTa             tarsus1_T2_right          ( 0.0,      0.0,      0.0     )
        //  41   T2R_TaT1             T2R_TaT1             tarsus2_T2_right          ( 0.0,      0.0,      0.0     )
        //  42   T2R_TaT3             T2R_TaT3             tarsus4_T2_right          ( 0.0,      0.0,      0.0     )
        //  43   T2R_TaTip            T2R_TaTip            tarsal_claw_T2_right      ( 0.0,     -0.0118,  -0.0006  )
        //  44   T3R_Tro              T3R_Tro              femur_T3_right            ( 0.0,      0.0,      0.0     )
        //  45   T3R_FeTi             T3R_FeTi             tibia_T3_right            ( 0.0,      0.0,      0.0     )
        //  46   T3R_TiTa             T3R_TiTa             tarsus1_T3_right          ( 0.0,      0.0,      0.0     )
        //  47   T3R_TaT1             T3R_TaT1             tarsus2_T3_right          ( 0.0,      0.0,      0.0     )
        //  48   T3R_TaT3             T3R_TaT3             tarsus4_T3_right          ( 0.0,      0.0,      0.0     )
        //  49   T3R_TaTip            T3R_TaTip            tarsal_claw_T3_right      ( 0.0,     -0.0109,  -0.0008  )
        static const SiteDef fly50_kp_sites[] = {
            // --- Head & body ---
            {"Antenna_Base", nullptr, "head",       { 0.000000,  0.038000,  0.012000}},
            {"EyeL",         nullptr, "head",       {-0.024500,  0.013500,  0.028500}},
            {"EyeR",         nullptr, "head",       { 0.024500,  0.013500,  0.028500}},
            {"Scutellum",    nullptr, "thorax",     {-0.049000,  0.000000,  0.040000}},
            {"Abd_A4",       nullptr, "abdomen_3",  { 0.000000,  0.033500,  0.021000}},
            {"Abd_tip",      nullptr, "abdomen_7",  { 0.000000,  0.039500, -0.001000}},
            // --- Wings ---
            {"WingL_base",   nullptr, "thorax",     {-0.009500,  0.045000,  0.017500}},
            {"WingL_V12",    nullptr, "wing_left",  {-0.007200, -0.212500,  0.007500}},
            {"WingL_V13",    nullptr, "wing_left",  { 0.022100, -0.256200, -0.025300}},
            {"WingR_base",   nullptr, "thorax",     {-0.009500, -0.045000,  0.017500}},
            {"WingR_V12",    nullptr, "wing_right", { 0.007200,  0.212500, -0.007500}},
            {"WingR_V13",    nullptr, "wing_right", {-0.022100,  0.256200,  0.025300}},
            // --- Left T1 leg ---
            {"T1L_ThxCx",   nullptr, "coxa_T1_left",         { 0.000000,  0.000000,  0.000000}},
            {"T1L_Tro",     nullptr, "femur_T1_left",        { 0.000000,  0.000000,  0.000000}},
            {"T1L_FeTi",    nullptr, "tibia_T1_left",        { 0.000000,  0.000000,  0.000000}},
            {"T1L_TiTa",    nullptr, "tarsus1_T1_left",      { 0.000000,  0.000000,  0.000000}},
            {"T1L_TaT1",    nullptr, "tarsus2_T1_left",      { 0.000000,  0.000000,  0.000000}},
            {"T1L_TaT3",    nullptr, "tarsus4_T1_left",      { 0.000000,  0.000000,  0.000000}},
            {"T1L_TaTip",   nullptr, "tarsal_claw_T1_left",  { 0.000000,  0.010500,  0.000600}},
            // --- Left T2 leg ---
            {"T2L_Tro",     nullptr, "femur_T2_left",        { 0.000000,  0.000000,  0.000000}},
            {"T2L_FeTi",    nullptr, "tibia_T2_left",        { 0.000000,  0.000000,  0.000000}},
            {"T2L_TiTa",    nullptr, "tarsus1_T2_left",      { 0.000000,  0.000000,  0.000000}},
            {"T2L_TaT1",    nullptr, "tarsus2_T2_left",      { 0.000000,  0.000000,  0.000000}},
            {"T2L_TaT3",    nullptr, "tarsus4_T2_left",      { 0.000000,  0.000000,  0.000000}},
            {"T2L_TaTip",   nullptr, "tarsal_claw_T2_left",  { 0.000000,  0.012200,  0.000600}},
            // --- Left T3 leg ---
            {"T3L_Tro",     nullptr, "femur_T3_left",        { 0.000000,  0.000000,  0.000000}},
            {"T3L_FeTi",    nullptr, "tibia_T3_left",        { 0.000000,  0.000000,  0.000000}},
            {"T3L_TiTa",    nullptr, "tarsus1_T3_left",      { 0.000000,  0.000000,  0.000000}},
            {"T3L_TaT1",    nullptr, "tarsus2_T3_left",      { 0.000000,  0.000000,  0.000000}},
            {"T3L_TaT3",    nullptr, "tarsus4_T3_left",      { 0.000000,  0.000000,  0.000000}},
            {"T3L_TaTip",   nullptr, "tarsal_claw_T3_left",  { 0.000000,  0.011100,  0.000800}},
            // --- Right T1 leg ---
            {"T1R_ThxCx",   nullptr, "coxa_T1_right",        { 0.000000,  0.000000,  0.000000}},
            {"T1R_Tro",     nullptr, "femur_T1_right",       { 0.000000,  0.000000,  0.000000}},
            {"T1R_FeTi",    nullptr, "tibia_T1_right",       { 0.000000,  0.000000,  0.000000}},
            {"T1R_TiTa",    nullptr, "tarsus1_T1_right",     { 0.000000,  0.000000,  0.000000}},
            {"T1R_TaT1",    nullptr, "tarsus2_T1_right",     { 0.000000,  0.000000,  0.000000}},
            {"T1R_TaT3",    nullptr, "tarsus4_T1_right",     { 0.000000,  0.000000,  0.000000}},
            {"T1R_TaTip",   nullptr, "tarsal_claw_T1_right", { 0.000000, -0.010100, -0.000600}},
            // --- Right T2 leg ---
            {"T2R_Tro",     nullptr, "femur_T2_right",       { 0.000000,  0.000000,  0.000000}},
            {"T2R_FeTi",    nullptr, "tibia_T2_right",       { 0.000000,  0.000000,  0.000000}},
            {"T2R_TiTa",    nullptr, "tarsus1_T2_right",     { 0.000000,  0.000000,  0.000000}},
            {"T2R_TaT1",    nullptr, "tarsus2_T2_right",     { 0.000000,  0.000000,  0.000000}},
            {"T2R_TaT3",    nullptr, "tarsus4_T2_right",     { 0.000000,  0.000000,  0.000000}},
            {"T2R_TaTip",   nullptr, "tarsal_claw_T2_right", { 0.000000, -0.011800, -0.000600}},
            // --- Right T3 leg ---
            {"T3R_Tro",     nullptr, "femur_T3_right",       { 0.000000,  0.000000,  0.000000}},
            {"T3R_FeTi",    nullptr, "tibia_T3_right",       { 0.000000,  0.000000,  0.000000}},
            {"T3R_TiTa",    nullptr, "tarsus1_T3_right",     { 0.000000,  0.000000,  0.000000}},
            {"T3R_TaT1",    nullptr, "tarsus2_T3_right",     { 0.000000,  0.000000,  0.000000}},
            {"T3R_TaT3",    nullptr, "tarsus4_T3_right",     { 0.000000,  0.000000,  0.000000}},
            {"T3R_TaTip",   nullptr, "tarsal_claw_T3_right", { 0.000000, -0.010900, -0.000800}},
        };

        const SiteDef *kp_sites;
        int kp_sites_count;
        if (is_fly_model) {
            kp_sites = fly50_kp_sites;
            kp_sites_count = (int)(sizeof(fly50_kp_sites) / sizeof(fly50_kp_sites[0]));
            std::cout << "[MuJoCo] Detected fruit fly model" << std::endl;
        } else {
            kp_sites = rodent_kp_sites;
            kp_sites_count = (int)(sizeof(rodent_kp_sites) / sizeof(rodent_kp_sites[0]));
            std::cout << "[MuJoCo] Detected rodent model" << std::endl;
        }

        int sites_added = 0, sites_skipped = 0;
        for (int si = 0; si < kp_sites_count; si++) {
            const auto &sd = kp_sites[si];
            // Skip if the site already exists (check both name and kpsite variant)
            if (mjs_findElement(spec, mjOBJ_SITE, sd.name)) {
                sites_skipped++;
                continue;
            }
            if (sd.kpsite && mjs_findElement(spec, mjOBJ_SITE, sd.kpsite)) {
                sites_skipped++;
                continue;
            }
            mjsBody *body = mjs_findBody(spec, sd.body);
            if (body) {
                mjsSite *site = mjs_addSite(body, nullptr);
                if (site) {
                    mjs_setName(site->element, sd.name);
                    site->pos[0] = sd.pos[0];
                    site->pos[1] = sd.pos[1];
                    site->pos[2] = sd.pos[2];
                    sites_added++;
                }
            } else {
                std::cerr << "[MuJoCo] WARNING: body '" << sd.body
                          << "' not found for site '" << sd.name << "'" << std::endl;
            }
        }
        if (sites_added > 0)
            std::cout << "[MuJoCo] Added " << sites_added << " keypoint sites"
                      << " (skipped " << sites_skipped << " already present)" << std::endl;
        else if (sites_skipped > 0)
            std::cout << "[MuJoCo] All " << sites_skipped
                      << " keypoint sites already present in model" << std::endl;

        // Try to load skin mesh if present (e.g., rodent_walker_skin.skn)
        // Only add if the model doesn't already have one.
        {
            namespace fs = std::filesystem;
            fs::path xml_dir = fs::path(xml_path).parent_path();
            fs::path skin_path = xml_dir / "rodent_walker_skin.skn";
            bool has_skin = (mjs_findElement(spec, mjOBJ_SKIN, "skin") != nullptr);
            if (fs::exists(skin_path) && !has_skin) {
                mjsSkin *skin = mjs_addSkin(spec);
                if (skin) {
                    // Use just the filename — MuJoCo resolves relative to XML dir
                    mjs_setString(skin->file, "rodent_walker_skin.skn");
                    mjs_setName(skin->element, "skin");
                    skin->rgba[0] = 0.45f; skin->rgba[1] = 0.75f;
                    skin->rgba[2] = 0.85f; skin->rgba[3] = 0.85f;
                    std::cout << "[MuJoCo] Added skin mesh: "
                              << skin_path.string() << std::endl;
                }
            }
        }

        // Compile the edited spec into mjModel
        model = mj_compile(spec, nullptr);
        if (!model) {
            // Get error details from MuJoCo's global error string
            std::cerr << "[MuJoCo] mj_compile failed after adding "
                      << sites_added << " sites (skipped " << sites_skipped << ")" << std::endl;
            // Try loading with mj_loadXML for a detailed error message
            {
                char err2[1024] = {0};
                mjModel *test = mj_loadXML(xml_path.c_str(), nullptr, err2, sizeof(err2));
                if (test) {
                    std::cerr << "[MuJoCo] Unmodified XML loads OK — issue is in mjSpec modifications" << std::endl;
                    mj_deleteModel(test);
                } else if (err2[0]) {
                    std::cerr << "[MuJoCo] " << err2 << std::endl;
                }
            }
            // If compile failed, it might be due to the skin — retry without it
            std::cerr << "[MuJoCo] mj_compile failed, retrying without skin..."
                      << std::endl;
            // Re-parse and apply only free joint + sites (no skin)
            mj_deleteSpec(spec);
            spec = mj_parseXML(xml_path.c_str(), nullptr, error_buf, sizeof(error_buf));
            if (spec) {
                if (!mjs_findElement(spec, mjOBJ_JOINT, "free")) {
                    for (const char *rn : {"torso", "thorax"}) {
                        mjsBody *t = mjs_findBody(spec, rn);
                        if (t) { mjs_addFreeJoint(t); has_free_joint = true; break; }
                    }
                } else { has_free_joint = true; }
                // Re-inject sites (same loop as above)
                // Re-detect model type for the fresh spec
                bool is_fly_retry = (mjs_findBody(spec, "thorax") &&
                                     mjs_findBody(spec, "head") &&
                                     mjs_findBody(spec, "wing_left"));
                const SiteDef *retry_sites = is_fly_retry ? fly50_kp_sites : rodent_kp_sites;
                int retry_count = is_fly_retry
                    ? (int)(sizeof(fly50_kp_sites) / sizeof(fly50_kp_sites[0]))
                    : (int)(sizeof(rodent_kp_sites) / sizeof(rodent_kp_sites[0]));
                for (int ri = 0; ri < retry_count; ri++) {
                    const auto &sd = retry_sites[ri];
                    if (mjs_findElement(spec, mjOBJ_SITE, sd.name)) continue;
                    if (sd.kpsite && mjs_findElement(spec, mjOBJ_SITE, sd.kpsite)) continue;
                    mjsBody *body = mjs_findBody(spec, sd.body);
                    if (body) {
                        mjsSite *site = mjs_addSite(body, nullptr);
                        if (site) {
                            mjs_setName(site->element, sd.name);
                            site->pos[0] = sd.pos[0];
                            site->pos[1] = sd.pos[1];
                            site->pos[2] = sd.pos[2];
                        }
                    }
                }
                model = mj_compile(spec, nullptr);
            }
        }
        mj_deleteSpec(spec);
        if (!model) {
            load_error = "mj_compile failed (check XML validity)";
            std::cerr << load_error << std::endl;
            return false;
        }

        data = mj_makeData(model);
        if (!data) {
            load_error = "mj_makeData failed";
            mj_deleteModel(model);
            model = nullptr;
            return false;
        }

        // Initialize abstract visualization (no OpenGL needed)
        mjv_defaultScene(&scene);
        mjv_defaultOption(&opt);
        // Enable all site groups so keypoint sites are visible
        for (int i = 0; i < mjNGROUP; i++) opt.sitegroup[i] = 1;
        mjv_makeScene(model, &scene, 2000);

        return finalize_load(xml_path, skeleton);
    }

    // Finalize loading: set site sizes/colors, build skeleton mapping.
    // Called from both XML and MJB loading paths after model/data are ready.
    bool finalize_load(const std::string &path, const SkeletonContext &skeleton) {
        bool is_fly = (mj_name2id(model, mjOBJ_BODY, "thorax") >= 0 &&
                       mj_name2id(model, mjOBJ_BODY, "head") >= 0 &&
                       mj_name2id(model, mjOBJ_BODY, "wing_left") >= 0);

        // Shrink keypoint sites to 50% of default (4mm radius)
        for (int i = 0; i < (int)model->nsite; i++) {
            float r = 0.004f; // 4mm radius
            model->site_size[3 * i + 0] = r;
            model->site_size[3 * i + 1] = r;
            model->site_size[3 * i + 2] = r;
        }

        // Build site name -> MuJoCo site index lookup
        std::unordered_map<std::string, int> mj_site_map;
        for (int i = 0; i < (int)model->nsite; i++) {
            const char *name = mj_id2name(model, mjOBJ_SITE, i);
            if (name) mj_site_map[name] = i;
        }

        // Mapping from RED skeleton node names to MuJoCo site names.
        // We try the _kpsite variant first, then fall back to the plain name.
        // For fly50, the RED skeleton names match site names exactly, so the
        // fallback exact-match path handles them without an explicit table.
        struct MjNameVariants { std::string kpsite; std::string plain; };

        // Rodent mapping: RED uses CamelCase, MuJoCo uses snake_case
        static const std::unordered_map<std::string, MjNameVariants> rodent_skeleton_to_mj = {
            {"Snout",     {"nose_0_kpsite",       "nose"}},
            {"EarL",      {"ear_L_1_kpsite",      "ear_L"}},
            {"EarR",      {"ear_R_2_kpsite",      "ear_R"}},
            {"Neck",      {"neck_3_kpsite",       "neck"}},
            {"SpineL",    {"spineL_4_kpsite",     "spineL"}},
            {"TailBase",  {"tailbase_5_kpsite",   "tailbase"}},
            {"ShoulderL", {"shoulder_L_6_kpsite", "shoulder_L"}},
            {"ElbowL",    {"elbow_L_7_kpsite",    "elbow_L"}},
            {"WristL",    {"wrist_L_8_kpsite",    "wrist_L"}},
            {"HandL",     {"hand_L_9_kpsite",     "hand_L"}},
            {"ShoulderR", {"shoulder_R_10_kpsite", "shoulder_R"}},
            {"ElbowR",    {"elbow_R_11_kpsite",   "elbow_R"}},
            {"WristR",    {"wrist_R_12_kpsite",   "wrist_R"}},
            {"HandR",     {"hand_R_13_kpsite",    "hand_R"}},
            {"KneeL",     {"knee_L_14_kpsite",    "knee_L"}},
            {"AnkleL",    {"ankle_L_15_kpsite",   "ankle_L"}},
            {"FootL",     {"foot_L_16_kpsite",    "foot_L"}},
            {"KneeR",     {"knee_R_17_kpsite",    "knee_R"}},
            {"AnkleR",    {"ankle_R_18_kpsite",   "ankle_R"}},
            {"FootR",     {"foot_R_19_kpsite",    "foot_R"}},
            {"TailTip",   {"tailtip_20_kpsite",   "tailtip"}},
            {"TailMid",   {"tailmid_21_kpsite",   "tailmid"}},
            {"Tail1Q",    {"tail1Q_22_kpsite",    "tail1Q"}},
            {"Tail3Q",    {"tail3Q_23_kpsite",    "tail3Q"}},
        };

        // Fly50: RED skeleton names are identical to MuJoCo site names
        // (Antenna_Base, EyeL, T1L_ThxCx, etc.), so no renaming needed.
        // We use an empty map and rely on the exact-match fallback below.
        static const std::unordered_map<std::string, MjNameVariants> fly50_skeleton_to_mj = {};

        const auto &skeleton_to_mj_name = is_fly ? fly50_skeleton_to_mj : rodent_skeleton_to_mj;

        // Build bidirectional mapping
        int num_nodes = skeleton.num_nodes;
        int num_sites = (int)model->nsite;

        site_to_skeleton.assign(num_sites, -1);
        skeleton_to_site.assign(num_nodes, -1);
        mapped_count = 0;

        for (int n = 0; n < num_nodes; n++) {
            const std::string &node_name = skeleton.node_names[n];
            int site_idx = -1;

            // Try explicit mapping table (kpsite variant first, then plain)
            auto it = skeleton_to_mj_name.find(node_name);
            if (it != skeleton_to_mj_name.end()) {
                auto sit = mj_site_map.find(it->second.kpsite);
                if (sit != mj_site_map.end()) {
                    site_idx = sit->second;
                } else {
                    sit = mj_site_map.find(it->second.plain);
                    if (sit != mj_site_map.end())
                        site_idx = sit->second;
                }
            }

            // Fallback: try exact node name match
            if (site_idx < 0) {
                auto sit = mj_site_map.find(node_name);
                if (sit != mj_site_map.end())
                    site_idx = sit->second;
            }

            if (site_idx >= 0) {
                skeleton_to_site[n] = site_idx;
                site_to_skeleton[site_idx] = n;
                mapped_count++;
            } else {
                std::cerr << "[MuJoCo] WARNING: no site for skeleton node '"
                          << node_name << "'" << std::endl;
            }
        }

        // Color each mapped site to match the skeleton keypoint color
        for (int i = 0; i < num_sites; i++) {
            int node_idx = site_to_skeleton[i];
            if (node_idx >= 0 && node_idx < (int)skeleton.node_colors.size()) {
                const ImVec4 &c = skeleton.node_colors[node_idx];
                model->site_rgba[4 * i + 0] = c.x;
                model->site_rgba[4 * i + 1] = c.y;
                model->site_rgba[4 * i + 2] = c.z;
                model->site_rgba[4 * i + 3] = 1.0f;
            } else {
                // Unmapped sites: dim gray
                model->site_rgba[4 * i + 0] = 0.4f;
                model->site_rgba[4 * i + 1] = 0.4f;
                model->site_rgba[4 * i + 2] = 0.4f;
                model->site_rgba[4 * i + 3] = 0.5f;
            }
        }

        std::cout << "[MuJoCo] Loaded " << path << std::endl;
        std::cout << "[MuJoCo] Model: " << (int)model->nbody << " bodies, "
                  << (int)model->njnt << " joints, " << (int)model->nsite << " sites, "
                  << (int)model->nu << " actuators" << std::endl;
        std::cout << "[MuJoCo] Site mapping: " << mapped_count << "/"
                  << num_nodes << " skeleton nodes matched" << std::endl;

        // Compute default pose FK so site positions are valid
        mj_forward(model, data);

        model_path = path;
        loaded = true;
        load_error.clear();
        return true;
    }
#else
    bool load(const std::string &, const SkeletonContext &) {
        load_error = "MuJoCo not available (compile with RED_HAS_MUJOCO)";
        return false;
    }
    bool finalize_load(const std::string &, const SkeletonContext &) { return false; }
#endif

    void unload() {
#ifdef RED_HAS_MUJOCO
        if (data)  { mj_deleteData(data);   data = nullptr; }
        if (model) {
            if (scene.maxgeom > 0) mjv_freeScene(&scene);
            memset(&scene, 0, sizeof(scene));
            memset(&opt, 0, sizeof(opt));
            mj_deleteModel(model); model = nullptr;
        }
#endif
        loaded = false;
        has_free_joint = false;
        scale_factor = 0.0f;
        model_scale = 1.0f;
        model_path.clear();
        load_error.clear();
        site_to_skeleton.clear();
        skeleton_to_site.clear();
        mapped_count = 0;
    }

    ~MujocoContext() { unload(); }

    // Non-copyable, movable
    MujocoContext() = default;
    MujocoContext(const MujocoContext &) = delete;
    MujocoContext &operator=(const MujocoContext &) = delete;
    MujocoContext(MujocoContext &&o) noexcept { *this = std::move(o); }
    MujocoContext &operator=(MujocoContext &&o) noexcept {
        if (this != &o) {
            unload();
            loaded = o.loaded;
            has_free_joint = o.has_free_joint;
            scale_factor = o.scale_factor;
            model_path = std::move(o.model_path);
            load_error = std::move(o.load_error);
            site_to_skeleton = std::move(o.site_to_skeleton);
            skeleton_to_site = std::move(o.skeleton_to_site);
            mapped_count = o.mapped_count;
#ifdef RED_HAS_MUJOCO
            model = o.model; o.model = nullptr;
            data = o.data;   o.data = nullptr;
            scene = o.scene;
            opt = o.opt;
            memset(&o.scene, 0, sizeof(o.scene));
            memset(&o.opt, 0, sizeof(o.opt));
#endif
            o.loaded = false;
            o.has_free_joint = false;
            o.scale_factor = 0.0f;
        }
        return *this;
    }
};
