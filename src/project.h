#pragma once
#include "camera.h"
#include "global.h"
#include "keypoint_colors.h"
#include "project_handler.h"
#include "skeleton.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Annotation capabilities — saved in project JSON
struct AnnotationConfig {
    bool enable_keypoints    = true;  // default on (existing behavior)
    bool enable_bboxes       = false;
    bool enable_obbs         = false;
    bool enable_segmentation = false;
    std::vector<std::string> class_names = {"animal"};
};

inline void to_json(nlohmann::json &j, const AnnotationConfig &a) {
    j = nlohmann::json{
        {"enable_keypoints", a.enable_keypoints},
        {"enable_bboxes", a.enable_bboxes},
        {"enable_obbs", a.enable_obbs},
        {"enable_segmentation", a.enable_segmentation},
        {"class_names", a.class_names}};
}
inline void from_json(const nlohmann::json &j, AnnotationConfig &a) {
    a.enable_keypoints    = j.value("enable_keypoints", true);
    a.enable_bboxes       = j.value("enable_bboxes", false);
    a.enable_obbs         = j.value("enable_obbs", false);
    a.enable_segmentation = j.value("enable_segmentation", false);
    a.class_names         = j.value("class_names", std::vector<std::string>{"animal"});
}

struct ProjectManager {
    bool show_project_window = false;
    std::string project_root_path;
    std::string project_path;
    std::string project_name;
    bool load_skeleton_from_json = false;
    std::string skeleton_file;
    std::string calibration_folder;
    std::string keypoints_root_folder;
    bool plot_keypoints_flag = false;
    std::vector<CameraParams> camera_params;
    std::vector<std::string> camera_names;
    std::string skeleton_name;
    std::string media_folder;
    bool telecentric = false; // true if using telecentric DLT calibration
    bool annotation_2d = false; // 2D-only: single/uncalibrated camera(s), no
                                // calibration / triangulation / 3D view
    // Canonical-timeline desync fix (sync_plan.h). Project-scoped because it
    // changes what a frame index means: with the fix ON, labels/predictions
    // are keyed by canonical trigger slot, OFF by raw mp4 index.
    bool sync_fix_enabled = false;

    // Pump Events (pump_events_core.h). Only the settings are stored, never the
    // dispenses themselves: the log is re-read on open so anything pumpctl
    // appended since is picked up. pump_log_path is an explicit override —
    // empty means "auto-discover in the recording folder", the normal case.
    std::string pump_log_path;
    float pump_offset_ms = 0.0f;
    bool pump_show_pulls = false;

    AnnotationConfig annotation_config; // annotation capabilities

    // JARVIS models imported into this project
    struct JarvisModelEntry {
        std::string name;            // display name
        std::string relative_path;   // e.g., "jarvis_models/mouseJan30"
        int num_joints = 0;
        int center_input_size = 0;
        int keypoint_input_size = 0;
    };
    std::vector<JarvisModelEntry> jarvis_models;
    int active_jarvis_model = -1;  // index into jarvis_models, -1 = none

    // A "model set" runs several JARVIS models over the SAME video (offline
    // batch) and concatenates their keypoints into one merged prediction store.
    // Each model tracks a different set of keypoints (e.g. a big body model plus
    // a small tail model), so each member carries its OWN skeleton — the channel
    // -> node mapping is positional, so the active skeleton must match the model
    // being run. The merged store's keypoints are the members' keypoints in
    // order (member 0's nodes, then member 1's, ...), matched by a synthesized
    // "combined skeleton" the batch switches to on completion.
    struct JarvisModelSetMember {
        int model_index = -1;             // index into jarvis_models
        bool skeleton_from_json = false;  // else a built-in preset
        std::string skeleton_file;        // when skeleton_from_json
        std::string skeleton_name;        // preset name when !skeleton_from_json
    };
    struct JarvisModelSet {
        std::string name;                        // display name, e.g. "body+tail"
        std::vector<JarvisModelSetMember> members;  // ordered = concat order
    };
    std::vector<JarvisModelSet> jarvis_model_sets;
    int active_jarvis_model_set = -1;  // index into jarvis_model_sets, -1 = none
};

inline void to_json(nlohmann::json &j, const ProjectManager::JarvisModelSetMember &m) {
    j = {{"model_index", m.model_index},
         {"skeleton_from_json", m.skeleton_from_json},
         {"skeleton_file", m.skeleton_file},
         {"skeleton_name", m.skeleton_name}};
}
inline void from_json(const nlohmann::json &j, ProjectManager::JarvisModelSetMember &m) {
    m.model_index = j.value("model_index", -1);
    m.skeleton_from_json = j.value("skeleton_from_json", false);
    m.skeleton_file = j.value("skeleton_file", std::string{});
    m.skeleton_name = j.value("skeleton_name", std::string{});
}
inline void to_json(nlohmann::json &j, const ProjectManager::JarvisModelSet &s) {
    j = {{"name", s.name}, {"members", s.members}};
}
inline void from_json(const nlohmann::json &j, ProjectManager::JarvisModelSet &s) {
    s.name = j.value("name", std::string{});
    s.members = j.value("members",
                        std::vector<ProjectManager::JarvisModelSetMember>{});
}

inline void to_json(nlohmann::json &j, const ProjectManager::JarvisModelEntry &m) {
    j = {{"name", m.name}, {"relative_path", m.relative_path},
         {"num_joints", m.num_joints}, {"center_input_size", m.center_input_size},
         {"keypoint_input_size", m.keypoint_input_size}};
}
inline void from_json(const nlohmann::json &j, ProjectManager::JarvisModelEntry &m) {
    m.name = j.value("name", std::string{});
    m.relative_path = j.value("relative_path", std::string{});
    m.num_joints = j.value("num_joints", 0);
    m.center_input_size = j.value("center_input_size", 0);
    m.keypoint_input_size = j.value("keypoint_input_size", 0);
}

inline void to_json(nlohmann::json &j, const ProjectManager &p) {
    j = nlohmann::json{{"project_root_path", p.project_root_path},
                       {"project_path", p.project_path},
                       {"project_name", p.project_name},
                       {"load_skeleton_from_json", p.load_skeleton_from_json},
                       {"skeleton_file", p.skeleton_file},
                       {"calibration_folder", p.calibration_folder},
                       {"keypoints_root_folder", p.keypoints_root_folder},
                       {"plot_keypoints_flag", p.plot_keypoints_flag},
                       {"camera_names", p.camera_names},
                       {"skeleton_name", p.skeleton_name},
                       {"media_folder", p.media_folder},
                       {"telecentric", p.telecentric},
                       {"annotation_2d", p.annotation_2d},
                       {"sync_fix_enabled", p.sync_fix_enabled},
                       {"pump_log_path", p.pump_log_path},
                       {"pump_offset_ms", p.pump_offset_ms},
                       {"pump_show_pulls", p.pump_show_pulls},
                       {"annotation_config", p.annotation_config},
                       {"jarvis_models", p.jarvis_models},
                       {"active_jarvis_model", p.active_jarvis_model},
                       {"jarvis_model_sets", p.jarvis_model_sets},
                       {"active_jarvis_model_set", p.active_jarvis_model_set}};
}

inline void from_json(const nlohmann::json &j, ProjectManager &p) {
    p.project_root_path = j.value("project_root_path", std::string{});
    p.project_path = j.value("project_path", std::string{});
    p.project_name = j.value("project_name", std::string{});
    p.load_skeleton_from_json = j.value("load_skeleton_from_json", false);
    p.skeleton_file = j.value("skeleton_file", std::string{});
    p.calibration_folder = j.value("calibration_folder", std::string{});
    p.keypoints_root_folder = j.value("keypoints_root_folder", std::string{});
    p.plot_keypoints_flag = j.value("plot_keypoints_flag", false);
    p.camera_names = j.value("camera_names", std::vector<std::string>{});
    p.skeleton_name = j.value("skeleton_name", std::string{});
    p.media_folder = j.value("media_folder", std::string{});
    p.telecentric = j.value("telecentric", false);
    p.annotation_2d = j.value("annotation_2d", false);
    p.sync_fix_enabled = j.value("sync_fix_enabled", false);
    p.pump_log_path = j.value("pump_log_path", std::string{});
    p.pump_offset_ms = j.value("pump_offset_ms", 0.0f);
    p.pump_show_pulls = j.value("pump_show_pulls", false);
    if (j.contains("annotation_config"))
        p.annotation_config = j["annotation_config"].get<AnnotationConfig>();
    if (j.contains("jarvis_models"))
        p.jarvis_models = j["jarvis_models"].get<std::vector<ProjectManager::JarvisModelEntry>>();
    p.active_jarvis_model = j.value("active_jarvis_model", -1);
    if (j.contains("jarvis_model_sets"))
        p.jarvis_model_sets =
            j["jarvis_model_sets"].get<std::vector<ProjectManager::JarvisModelSet>>();
    p.active_jarvis_model_set = j.value("active_jarvis_model_set", -1);
}

// True when the project is 2D-only: explicitly flagged, or has no usable
// calibration loaded. Gates triangulation / 3D view / reprojection / 3D export
// and the multi-view completeness metric. (camera_params is empty for an
// uncalibrated project, so old uncalibrated projects are treated 2D-safely.)
inline bool project_is_2d(const ProjectManager &p) {
    return p.annotation_2d || p.camera_params.empty();
}

inline bool save_project_manager_json(const ProjectManager &p,
                                      const std::filesystem::path &file,
                                      std::string *err = nullptr,
                                      int indent = 2,
                                      const ProjectHandlerRegistry *reg = nullptr) {
    try {
        nlohmann::json j = p;
        if (reg)
            project_handlers_save(*reg, j);

        std::error_code ec;
        std::filesystem::create_directories(file.parent_path(), ec);
        if (ec) {
            if (err)
                *err = ec.message();
            return false;
        }

        std::ofstream ofs(file, std::ios::binary);
        if (!ofs) {
            if (err)
                *err = "Failed to open file for writing: " + file.string();
            return false;
        }
        ofs << j.dump(indent);
        return true;
    } catch (const std::exception &e) {
        if (err)
            *err = e.what();
        return false;
    }
}

inline bool load_project_manager_json(ProjectManager *out,
                                      const std::filesystem::path &file,
                                      std::string *err = nullptr,
                                      const ProjectHandlerRegistry *reg = nullptr) {
    if (!out) {
        if (err)
            *err = "Output pointer is null";
        return false;
    }
    try {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) {
            if (err)
                *err = "Failed to open file for reading: " + file.string();
            return false;
        }
        nlohmann::json j;
        ifs >> j;
        *out = j.get<ProjectManager>();
        if (reg)
            project_handlers_load(*reg, j);
        return true;
    } catch (const std::exception &e) {
        if (err)
            *err = e.what();
        return false;
    }
}

// Load/replace the active skeleton per pm.load_skeleton_from_json /
// skeleton_file / skeleton_name, clearing the target SkeletonContext's
// existing node/edge state first. Shared by setup_project() (new project)
// and switch_project_skeleton() (gui/switch_skeleton_window.h, switching an
// already-open project's skeleton mid-session).
inline bool reload_skeleton(ProjectManager &pm, SkeletonContext &skeleton,
                            const std::map<std::string, SkeletonPrimitive> &skeleton_map,
                            std::string *err) {
    skeleton.num_nodes = 0;
    skeleton.num_edges = 0;
    skeleton.name.clear();
    skeleton.has_skeleton = true;
    skeleton.node_colors.clear();
    skeleton.edges.clear();
    skeleton.node_names.clear();

    if (pm.load_skeleton_from_json) {
        try {
            load_skeleton_json(pm.skeleton_file, &skeleton);
        } catch (const std::exception &e) {
            if (err) *err = "Error loading skeleton: " + std::string(e.what());
            return false;
        }
    } else {
        auto it = skeleton_map.find(pm.skeleton_name);
        if (it == skeleton_map.end()) {
            if (err)
                *err = "Unknown skeleton: " + pm.skeleton_name;
            return false;
        }
        skeleton_initialize(it->first.c_str(), &skeleton, it->second);
    }
    return true;
}

// Build a standalone SkeletonContext for one model-set member (preset or JSON).
// Mirrors reload_skeleton but writes into an arbitrary `out` rather than the
// project's live skeleton, so a member's layout can be resolved without
// disturbing the project.
inline bool resolve_member_skeleton(
    const ProjectManager::JarvisModelSetMember &m,
    const std::map<std::string, SkeletonPrimitive> &skeleton_map,
    SkeletonContext &out, std::string *err) {
    out = SkeletonContext{};
    out.has_skeleton = true;
    if (m.skeleton_from_json) {
        try {
            load_skeleton_json(m.skeleton_file, &out);
        } catch (const std::exception &e) {
            if (err) *err = "Error loading skeleton '" + m.skeleton_file +
                            "': " + std::string(e.what());
            return false;
        }
    } else {
        auto it = skeleton_map.find(m.skeleton_name);
        if (it == skeleton_map.end()) {
            if (err) *err = "Unknown skeleton: " + m.skeleton_name;
            return false;
        }
        skeleton_initialize(it->first.c_str(), &out, it->second);
    }
    return true;
}

// Concatenate several skeletons into one. Node names/colors are appended in
// order; edges are shifted by the running node offset so each part's bones keep
// their shape in the combined index space. Node colors are re-derived on load
// from position (see load_skeleton_json), so they need not be preserved here.
inline void build_combined_skeleton(const std::vector<SkeletonContext> &parts,
                                    const std::string &name,
                                    SkeletonContext &out) {
    out = SkeletonContext{};
    out.name = name;
    out.has_skeleton = true;
    int offset = 0;
    for (const auto &p : parts) {
        for (const auto &nm : p.node_names) out.node_names.push_back(nm);
        for (const auto &c : p.node_colors) out.node_colors.push_back(c);
        for (const auto &e : p.edges)
            out.edges.push_back(tuple_i{e.x + offset, e.y + offset});
        offset += p.num_nodes;
    }
    out.num_nodes = offset;
    out.num_edges = (int)out.edges.size();
}

// Serialize a skeleton to the JSON schema load_skeleton_json() reads back
// (has_skeleton, num_nodes, num_edges, node_names, edges). Colors are omitted
// because the loader regenerates them from node position.
inline bool write_skeleton_json(const SkeletonContext &s,
                                const std::string &path, std::string *err) {
    nlohmann::json j;
    j["has_skeleton"] = s.has_skeleton;
    j["num_nodes"] = s.num_nodes;
    j["num_edges"] = s.num_edges;
    j["node_names"] = s.node_names;
    nlohmann::json edges = nlohmann::json::array();
    for (const auto &e : s.edges) edges.push_back({e.x, e.y});
    j["edges"] = edges;
    std::ofstream f(path);
    if (!f) { if (err) *err = "Cannot write skeleton file: " + path; return false; }
    f << j.dump(2) << "\n";
    return true;
}

inline bool setup_project(ProjectManager &pm, SkeletonContext &skeleton,
                   const std::map<std::string, SkeletonPrimitive> &skeleton_map,
                   std::string *err) {
    if (!ensure_dir_exists(pm.project_path, err))
        return false;

    pm.camera_params.clear();
    if (pm.camera_names.size() > 1 && !pm.calibration_folder.empty()) {
        namespace fs = std::filesystem;

        // Auto-detect timestamped subfolder: if the calibration folder
        // doesn't contain YAML/CSV files directly but has a single
        // timestamped subdirectory (YYYY_MM_DD_*), use that instead.
        std::string calib_dir = pm.calibration_folder;
        if (!calib_dir.empty() && fs::is_directory(calib_dir))
        {
            bool has_calib_files = false;
            std::string newest_subdir;
            for (const auto &entry : fs::directory_iterator(calib_dir)) {
                if (entry.is_regular_file()) {
                    auto ext = entry.path().extension().string();
                    if (ext == ".yaml" || ext == ".csv")
                        has_calib_files = true;
                } else if (entry.is_directory()) {
                    auto name = entry.path().filename().string();
                    // Match YYYY_MM_DD pattern (calibration output folders)
                    if (name.size() >= 10 && name[4] == '_' && name[7] == '_') {
                        if (name > newest_subdir)
                            newest_subdir = name;
                    }
                }
            }
            if (!has_calib_files && !newest_subdir.empty()) {
                calib_dir = (fs::path(calib_dir) / newest_subdir).string();
                pm.calibration_folder = calib_dir;
            }
        }

        for (const std::string &cam_name : pm.camera_names) {
            CameraParams cam;
            std::string cam_err;
            bool loaded = false;

            if (pm.telecentric) {
                // Telecentric: load DLT CSV
                fs::path dlt_path =
                    fs::path(calib_dir) /
                    (cam_name + "_dlt.csv");
                loaded = camera_load_params_from_dlt_csv(
                    dlt_path.string(), cam, cam_err);
            } else {
                // Perspective: load YAML
                fs::path yaml_path =
                    fs::path(calib_dir) /
                    (cam_name + ".yaml");
                loaded = camera_load_params_from_yaml(
                    yaml_path.string(), cam, cam_err);
            }

            if (!loaded) {
                pm.camera_params.clear();
                if (err) {
                    *err = "Failed to load camera params: " + cam_name +
                        (cam_err.empty() ? "" : (" (" + cam_err + ")"));
                }
                return false;
            }
            pm.camera_params.push_back(cam);
        }
    }

    if (!reload_skeleton(pm, skeleton, skeleton_map, err))
        return false;

    // Recolor keypoints from the user-selected colormap (single global source
    // of truth). Covers both the JSON and primitive skeleton paths at once.
    apply_keypoint_colormap(skeleton, g_keypoint_colormap);

    pm.keypoints_root_folder =
        (std::filesystem::path(pm.project_path) / "labeled_data").string();
    if (!ensure_dir_exists(pm.keypoints_root_folder, err))
        return false;

    pm.plot_keypoints_flag = true;
    pm.show_project_window = false;
    return true;
}
