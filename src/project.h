#pragma once
#include "camera.h"
#include "global.h"
#include "render.h"
#include "skeleton.h"
#include "utils.h"
#include <ImGuiFileDialog.h>
#include <fstream>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <string>
#include <vector>

struct ProjectManager {
    bool confirm_overwrite = false;
    bool show_project_window = false;
    std::string project_root_path;
    std::string project_path;
    std::string project_name;
    bool load_skeleton_from_json = false;
    std::string project_skeleton_file;
    std::string project_calibration_folder;
    std::string keypoints_root_folder;
    bool plot_keypoints_flag = false;
    std::vector<CameraParams> camera_params;
    std::vector<std::string> camera_names;
    std::string skeleton_name;
    std::string media_dir;
};

inline void to_json(nlohmann::json &j, const ProjectManager &p) {
    j = nlohmann::json{
        {"project_root_path", p.project_root_path},
        {"project_path", p.project_path},
        {"project_name", p.project_name},
        {"load_skeleton_from_json", p.load_skeleton_from_json},
        {"project_skeleton_file", p.project_skeleton_file},
        {"project_calibration_folder", p.project_calibration_folder},
        {"keypoints_root_folder", p.keypoints_root_folder},
        {"plot_keypoints_flag", p.plot_keypoints_flag},
        {"camera_names", p.camera_names},
        {"skeleton_name", p.skeleton_name},
        {"media_dir", p.media_dir}};
}

inline void from_json(const nlohmann::json &j, ProjectManager &p) {
    // Leave show_project_window & camera_params at their defaults
    p.project_root_path = j.value("project_root_path", std::string{});
    p.project_path = j.value("project_path", std::string{});
    p.project_name = j.value("project_name", std::string{});
    p.load_skeleton_from_json = j.value("load_skeleton_from_json", false);
    p.project_skeleton_file = j.value("project_skeleton_file", std::string{});
    p.project_calibration_folder =
        j.value("project_calibration_folder", std::string{});
    p.keypoints_root_folder = j.value("keypoints_root_folder", std::string{});
    p.plot_keypoints_flag = j.value("plot_keypoints_flag", false);
    p.camera_names = j.value("camera_names", std::vector<std::string>{});
    p.skeleton_name = j.value("skeleton_name", std::string{});
    p.media_dir = j.value("media_dir", std::string{});
}

inline bool save_project_manager_json(const ProjectManager &p,
                                      const std::filesystem::path &file,
                                      std::string *err = nullptr,
                                      int indent = 2) {
    try {
        nlohmann::json j = p;

        // Ensure parent directory exists
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
                                      std::string *err = nullptr) {
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
        return true;
    } catch (const std::exception &e) {
        if (err)
            *err = e.what();
        return false;
    }
}

bool setup_project(ProjectManager &pm, SkeletonContext &skeleton,
                   const std::map<std::string, SkeletonPrimitive> &skeleton_map,
                   std::string *err) {
    // 1) Ensure project directory
    if (!ensure_dir_exists(pm.project_path, err))
        return false;

    // 2) Load camera params if needed
    pm.camera_params.clear();
    if (pm.camera_names.size() > 1) {
        for (const std::string &cam_name : pm.camera_names) {
            std::filesystem::path cam_path =
                std::filesystem::path(pm.project_calibration_folder) /
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

    // 3) Reset & init skeleton
    skeleton.num_nodes = 0;
    skeleton.num_edges = 0;
    skeleton.name.clear();
    skeleton.has_bbox = false;
    skeleton.has_skeleton = true;
    skeleton.node_colors.clear();
    skeleton.edges.clear();
    skeleton.node_names.clear();

    if (pm.load_skeleton_from_json) {
        load_skeleton_json(pm.project_skeleton_file,
                           &skeleton); // assume throws/handles its own errors
    } else {
        auto it = skeleton_map.find(pm.skeleton_name);
        if (it == skeleton_map.end()) {
            if (err)
                *err = "Unknown skeleton: " + pm.skeleton_name;
            return false;
        }
        skeleton_initialize(it->first.c_str(), &skeleton, it->second);
    }

    // 4) Ensure keypoints dir
    pm.keypoints_root_folder =
        (std::filesystem::path(pm.project_path) / "labeled_data").string();
    if (!ensure_dir_exists(pm.keypoints_root_folder, err))
        return false;

    // 5) Save config
    std::filesystem::path cfg =
        std::filesystem::path(pm.project_path) / "project.json";
    if (!save_project_manager_json(pm, cfg, err))
        return false;

    // 6) Final flags
    pm.plot_keypoints_flag = true;
    pm.show_project_window = false;
    return true;
}

inline void
DrawProjectWindow(ProjectManager &pm,
                  std::map<std::string, SkeletonPrimitive> &skeleton_map,
                  SkeletonContext &skeleton, std::string &skeleton_dir,
                  bool &show_error, std::string &error_message) {
    if (!pm.show_project_window)
        return;

    // Resizable window (no auto-resize)
    ImGuiWindowFlags win_flags = ImGuiWindowFlags_NoCollapse;
    ImGui::SetNextWindowSize(ImVec2(720, 460), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Create Project", &pm.show_project_window, win_flags)) {

        // error banner
        if (show_error && !error_message.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(1.0f, 0.45f, 0.45f, 1.0f));
            ImGui::TextUnformatted(error_message.c_str());
            ImGui::PopStyleColor();
            ImGui::Separator();
        }

        if (ImGui::BeginTable(
                "projectForm", 3,
                ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX |
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
            ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthFixed,
                                    160.0f);
            ImGui::TableSetupColumn("Field", ImGuiTableColumnFlags_WidthStretch,
                                    1.0f); // stretches wide
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed,
                                    110.0f);

            auto LabelCell = [](const char *t) {
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(t);
            };

            // ---- Project Name ----
            ImGui::TableNextRow();
            LabelCell("Project Name");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##projname", &pm.project_name);
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Project Root Path ----
            ImGui::TableNextRow();
            LabelCell("Project Root Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##rootpath", &pm.project_root_path);
            ImGui::TableSetColumnIndex(2);
            if (ImGui::Button("Browse##project")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = pm.project_root_path;
                cfg.fileName = pm.project_name;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseProjectDir", "Choose Project Directory", nullptr,
                    cfg);
            }

            // ---- Full Path (computed) ----
            {
                std::filesystem::path p =
                    std::filesystem::path(pm.project_root_path) /
                    pm.project_name;
                pm.project_path = p.string();
            }
            const bool path_exists = !pm.project_path.empty() &&
                                     std::filesystem::exists(pm.project_path);
            bool path_nonempty = false;
            if (path_exists) {
                std::error_code ec;
                path_nonempty =
                    !std::filesystem::is_empty(pm.project_path, ec) && !ec;
            }

            ImGui::TableNextRow();
            LabelCell("Full Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled(); // read-only preview
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##fullpath", &pm.project_path);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // Existence warning + overwrite checkbox (full width row)
            if (path_exists) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TableSetBgColor(
                    ImGuiTableBgTarget_RowBg0,
                    ImGui::GetColorU32(ImVec4(0.25f, 0.16f, 0.05f, 0.15f)));
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Warning");

                ImGui::TableSetColumnIndex(1);
                ImGui::PushStyleColor(ImGuiCol_Text,
                                      ImVec4(1.0f, 0.78f, 0.20f, 1.0f));
                if (path_nonempty)
                    ImGui::TextWrapped("Folder exists and is NOT empty: %s",
                                       pm.project_path.c_str());
                else
                    ImGui::TextWrapped("Folder exists: %s",
                                       pm.project_path.c_str());
                ImGui::PopStyleColor();

                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(1);
                ImGui::Checkbox("Overwrite existing folder##confirm_overwrite",
                                &pm.confirm_overwrite);
                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));
            } else {
                pm.confirm_overwrite =
                    false; // reset if target no longer exists
            }

            // ---- Skeleton source toggle ----
            ImGui::TableNextRow();
            LabelCell("Load skeleton from file");
            ImGui::TableSetColumnIndex(1);
            ImGui::Checkbox("##loadskel", &pm.load_skeleton_from_json);
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Skeleton File ----
            ImGui::TableNextRow();
            LabelCell("Skeleton File");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled(!pm.load_skeleton_from_json);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##skelpath", &pm.project_skeleton_file);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::BeginDisabled(!pm.load_skeleton_from_json);
            if (ImGui::Button("Browse##loadprojectskeleton")) {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = skeleton_dir;
                config.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseSkeleton", "Choose Skeleton", ".json", config);
            }
            ImGui::EndDisabled();

            // ---- Skeleton Preset ----
            std::vector<const char *> labels_s;
            labels_s.reserve(skeleton_map.size());
            for (auto &kv : skeleton_map)
                labels_s.push_back(kv.first.c_str());
            static int skeleton_idx = 0;
            if (skeleton_idx >= (int)labels_s.size())
                skeleton_idx = 0;

            ImGui::TableNextRow();
            LabelCell("Select Skeleton");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled(pm.load_skeleton_from_json ||
                                 labels_s.empty());
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::Combo("##skeleton_combo", &skeleton_idx, labels_s.data(),
                         (int)labels_s.size());
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            pm.skeleton_name =
                pm.load_skeleton_from_json
                    ? std::string()
                    : (labels_s.empty() ? std::string()
                                        : std::string(labels_s[skeleton_idx]));

            // ---- Calibration Folder ----
            if (pm.camera_names.size() > 1) {
                ImGui::TableNextRow();
                LabelCell("Calibration Folder");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calibfolder",
                                 &pm.project_calibration_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##loadprojectcalibration")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = pm.project_root_path;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibration", "Choose Calibration Folder",
                        nullptr, cfg);
                }
            }
            ImGui::EndTable();
        }

        ImGui::Separator();

        // Validation (also require overwrite if target exists)
        const bool needed_ok =
            !pm.project_name.empty() && !pm.project_root_path.empty() &&
            (pm.camera_names.size() <= 1 ||
             !pm.project_calibration_folder.empty()) &&
            (!pm.load_skeleton_from_json || !pm.project_skeleton_file.empty());

        const bool allow_create =
            needed_ok &&
            (!std::filesystem::exists(pm.project_path) || pm.confirm_overwrite);

        // Right-align Create button
        float avail = ImGui::GetContentRegionAvail().x;
        const char *create_label = "Create Project##action";
        float w = ImGui::CalcTextSize(create_label).x +
                  ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));

        ImGui::BeginDisabled(!allow_create);
        if (ImGui::Button(create_label)) {
            show_error =
                !setup_project(pm, skeleton, skeleton_map, &error_message);
        }
        ImGui::EndDisabled();
    }
    ImGui::End();
}

inline void
load_videos(std::map<std::string, std::string> &selected_files,
            PlaybackState &ps, ProjectManager &pm,
            std::unordered_map<std::string, bool> &window_was_decoding,
            std::vector<FFmpegDemuxer *> &demuxers, DecoderContext *dc_context,
            render_scene *scene, int label_buffer_size,
            std::vector<std::thread> &decoder_threads,
            std::vector<bool> &is_view_focused) {
    for (const auto &elem : selected_files) {
        std::size_t cam_string_mp4_position = elem.first.find("mp4");
        std::string cam_string =
            elem.first.substr(0, cam_string_mp4_position - 1);
        pm.camera_names.push_back(cam_string);
        std::cout << "camera names: " << cam_string << std::endl;
        window_need_decoding[cam_string].store(true);
        window_was_decoding[cam_string] = true;
        std::map<std::string, std::string> m;
        FFmpegDemuxer *demuxer = new FFmpegDemuxer(elem.second.c_str(), m);
        demuxers.push_back(demuxer);
    }
    std::map<std::string, std::string> m;
    FFmpegDemuxer dummy_dmuxer(selected_files.begin()->second.c_str(), m);
    dc_context->seek_interval =
        (int)dummy_dmuxer.FindKeyFrameInterval(); // get the seek
                                                  // interval
    dc_context->video_fps = dummy_dmuxer.GetFramerate();
    scene->num_cams = selected_files.size();
    scene->image_width = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    scene->image_height = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    for (u32 j = 0; j < scene->num_cams; j++) {
        scene->image_width[j] = demuxers[j]->GetWidth();
        scene->image_height[j] = demuxers[j]->GetHeight();
    }
    render_allocate_scene_memory(scene, label_buffer_size);

    for (int i = 0; i < scene->num_cams; i++) {
        decoder_threads.push_back(std::thread(
            &decoder_process, dc_context, demuxers[i], pm.camera_names[i],
            scene->display_buffer[i], scene->size_of_buffer,
            &scene->seek_context[i], scene->use_cpu_buffer));
        is_view_focused.push_back(false);
    }
    ps.video_loaded = true;
}
