#pragma once
#include "camera.h"
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
    AnnotationConfig annotation_config; // annotation capabilities
};

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
                       {"annotation_config", p.annotation_config}};
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
    if (j.contains("annotation_config"))
        p.annotation_config = j["annotation_config"].get<AnnotationConfig>();
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

inline bool setup_project(ProjectManager &pm, SkeletonContext &skeleton,
                   const std::map<std::string, SkeletonPrimitive> &skeleton_map,
                   std::string *err) {
    if (!ensure_dir_exists(pm.project_path, err))
        return false;

    pm.camera_params.clear();
    if (pm.camera_names.size() > 1) {
        for (const std::string &cam_name : pm.camera_names) {
            std::filesystem::path cam_path =
                std::filesystem::path(pm.calibration_folder) /
                (cam_name + ".yaml");

            CameraParams cam;
            std::string cam_err;
            if (!camera_load_params_from_yaml(cam_path.string(), cam,
                                              cam_err)) {
                pm.camera_params.clear();
                if (err) {
                    *err =
                        "Failed to load camera params: " + cam_path.string() +
                        (cam_err.empty() ? "" : (" (" + cam_err + ")"));
                }
                return false;
            }
            pm.camera_params.push_back(cam);
        }
    }

    skeleton.num_nodes = 0;
    skeleton.num_edges = 0;
    skeleton.name.clear();
    skeleton.has_skeleton = true;
    skeleton.node_colors.clear();
    skeleton.edges.clear();
    skeleton.node_names.clear();

    if (pm.load_skeleton_from_json) {
        load_skeleton_json(pm.skeleton_file, &skeleton);
    } else {
        auto it = skeleton_map.find(pm.skeleton_name);
        if (it == skeleton_map.end()) {
            if (err)
                *err = "Unknown skeleton: " + pm.skeleton_name;
            return false;
        }
        skeleton_initialize(it->first.c_str(), &skeleton, it->second);
    }

    pm.keypoints_root_folder =
        (std::filesystem::path(pm.project_path) / "labeled_data").string();
    if (!ensure_dir_exists(pm.keypoints_root_folder, err))
        return false;

    pm.plot_keypoints_flag = true;
    pm.show_project_window = false;
    return true;
}
