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
    mjvScene  scene;
    mjvOption opt;
#endif

    // Global scale factor: scales the target keypoints to match the model.
    // 0 = auto-detect from data (recommended). Positive values override.
    float scale_factor = 0.0f;

    // Whether a free joint was added to the root body
    bool has_free_joint = false;

    // Load a MuJoCo XML body model and build the site↔skeleton mapping.
    // Programmatically adds a free joint to the torso via mjSpec so the
    // IK solver can optimize root position and orientation.
    // Returns true on success.
    bool load(const std::string &xml_path, const SkeletonContext &skeleton) {
#ifdef RED_HAS_MUJOCO
        unload();

        char error_buf[1024] = {0};

        // Parse XML into mjSpec for programmatic editing
        mjSpec *spec = mj_parseXML(xml_path.c_str(), nullptr, error_buf, sizeof(error_buf));
        if (!spec) {
            load_error = std::string("MuJoCo XML parse failed: ") + error_buf;
            std::cerr << load_error << std::endl;
            return false;
        }

        // Add a free joint to the torso body so the IK solver can
        // optimize root position and orientation (6 extra DOFs).
        mjsBody *torso = mjs_findBody(spec, "torso");
        has_free_joint = false;
        if (torso) {
            mjs_addFreeJoint(torso);
            has_free_joint = true;
            std::cout << "[MuJoCo] Added free joint to torso body" << std::endl;
        } else {
            std::cerr << "[MuJoCo] WARNING: no 'torso' body found — "
                         "root position will be fixed" << std::endl;
        }

        // Add missing keypoint sites for models that lack them.
        // The rodent_no_collision.xml already has all 24 *_kpsite sites;
        // the dm_rodent rodent.xml does not. We check for both the plain
        // name ("nose") and the _kpsite variant ("nose_0_kpsite") before
        // adding, to avoid duplicates.
        struct SiteDef {
            const char *name; const char *kpsite; const char *body; double pos[3];
        };
        static const SiteDef kp_sites[] = {
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
        int sites_added = 0, sites_skipped = 0;
        for (const auto &sd : kp_sites) {
            // Skip if either the plain name or the _kpsite variant already exists
            if (mjs_findElement(spec, mjOBJ_SITE, sd.name) ||
                mjs_findElement(spec, mjOBJ_SITE, sd.kpsite)) {
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
            // If compile failed, it might be due to the skin — retry without it
            std::cerr << "[MuJoCo] mj_compile failed, retrying without skin..."
                      << std::endl;
            // Re-parse and apply only free joint + sites (no skin)
            mj_deleteSpec(spec);
            spec = mj_parseXML(xml_path.c_str(), nullptr, error_buf, sizeof(error_buf));
            if (spec) {
                mjsBody *t = mjs_findBody(spec, "torso");
                if (t) { mjs_addFreeJoint(t); has_free_joint = true; }
                // Re-inject sites (same loop as above)
                for (const auto &sd : kp_sites) {
                    if (mjs_findElement(spec, mjOBJ_SITE, sd.name) ||
                        mjs_findElement(spec, mjOBJ_SITE, sd.kpsite)) continue;
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
        // Supports both naming conventions:
        //   - rodent_no_collision.xml: "nose_0_kpsite", "ear_L_1_kpsite", etc.
        //   - older models: "nose", "ear_L", etc.
        // We try the _kpsite variant first, then fall back to the plain name.
        struct MjNameVariants { std::string kpsite; std::string plain; };
        static const std::unordered_map<std::string, MjNameVariants> skeleton_to_mj_name = {
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

        std::cout << "[MuJoCo] Loaded " << xml_path << std::endl;
        std::cout << "[MuJoCo] Model: " << (int)model->nbody << " bodies, "
                  << (int)model->njnt << " joints, " << (int)model->nsite << " sites, "
                  << (int)model->nu << " actuators" << std::endl;
        std::cout << "[MuJoCo] Site mapping: " << mapped_count << "/"
                  << num_nodes << " skeleton nodes matched" << std::endl;

        // Compute default pose FK so site positions are valid
        mj_forward(model, data);

        model_path = xml_path;
        loaded = true;
        load_error.clear();
        return true;
#else
        (void)xml_path; (void)skeleton;
        load_error = "MuJoCo not available (compile with RED_HAS_MUJOCO)";
        return false;
#endif
    }

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
