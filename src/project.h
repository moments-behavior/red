#pragma once
#include "camera.h"
#include "global.h"
#include "gui/popup_stack.h"
#include "project_handler.h"
#include "render.h"
#include "skeleton.h"
#include "utils.h"
#include <ImGuiFileDialog.h>
#include <fstream>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <string>
#include <vector>
#ifdef __APPLE__
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#endif

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
                       {"media_folder", p.media_folder}};
}

inline void from_json(const nlohmann::json &j, ProjectManager &p) {
    // Leave show_project_window & camera_params at their defaults
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
}

inline bool save_project_manager_json(const ProjectManager &p,
                                      const std::filesystem::path &file,
                                      std::string *err = nullptr,
                                      int indent = 2,
                                      const ProjectHandlerRegistry *reg = nullptr) {
    try {
        nlohmann::json j = p;

        // Merge registered subsystem sections
        if (reg)
            project_handlers_save(*reg, j);

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

        // Extract registered subsystem sections
        if (reg)
            project_handlers_load(*reg, j);

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
        load_skeleton_json(pm.skeleton_file,
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

    // 6) Final flags
    pm.plot_keypoints_flag = true;
    pm.show_project_window = false;
    return true;
}

inline void
DrawProjectWindow(ProjectManager &pm,
                  std::map<std::string, SkeletonPrimitive> &skeleton_map,
                  SkeletonContext &skeleton, std::string &skeleton_dir,
                  PopupStack &popups) {
    if (!pm.show_project_window)
        return;

    // Resizable window (no auto-resize)
    ImGuiWindowFlags win_flags = ImGuiWindowFlags_NoCollapse;
    ImGui::SetNextWindowSize(ImVec2(720, 460), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Create Project", &pm.show_project_window, win_flags)) {

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

            ImGui::TableNextRow();
            LabelCell("Full Path");
            ImGui::TableSetColumnIndex(1);
            ImGui::BeginDisabled(); // read-only preview
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputText("##fullpath", &pm.project_path);
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(2);
            ImGui::Dummy(ImVec2(1, 1));

            // ---- Skeleton (single row, compact toggle right, value centered)
            // ---- Build preset labels
            std::vector<const char *> labels_s;
            labels_s.reserve(skeleton_map.size());
            for (auto &kv : skeleton_map)
                labels_s.push_back(kv.first.c_str());
            static int skeleton_idx = 0;
            if (skeleton_idx >= (int)labels_s.size())
                skeleton_idx = 0;

            // Mode: 0 = File, 1 = Preset (mirror your boolean)
            int mode = pm.load_skeleton_from_json ? 0 : 1;

            ImGui::TableNextRow();
            LabelCell("Skeleton");

            // -------- Field (center column): the value control --------
            ImGui::TableSetColumnIndex(1);
            {
                if (pm.load_skeleton_from_json) {
                    // Input path + inline Browse in the same (Field) column
                    float avail = ImGui::GetContentRegionAvail().x;
                    const char *btxt = "Browse##browse_skel";
                    float browse_w = ImGui::CalcTextSize(btxt).x +
                                     ImGui::GetStyle().FramePadding.x * 2.0f;
                    float gap = ImGui::GetStyle().ItemInnerSpacing.x;

                    ImGui::PushID("skelfile");
                    ImGui::SetNextItemWidth(
                        ImMax(50.0f, avail - browse_w - gap));
                    ImGui::InputText("##path", &pm.skeleton_file);
                    ImGui::SameLine(0.0f, gap);
                    if (ImGui::Button(btxt)) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = skeleton_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseSkeleton", "Choose Skeleton", ".json",
                            config);
                    }
                    ImGui::PopID();
                } else {
                    // Preset list in the Field column
                    ImGui::BeginDisabled(labels_s.empty());
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::Combo("##skeleton_preset", &skeleton_idx,
                                 labels_s.data(), (int)labels_s.size());
                    ImGui::EndDisabled();
                }
            }

            // -------- Action (right column): compact File/Preset toggle
            // --------
            ImGui::TableSetColumnIndex(2);
            // Keep this small
            ImGui::SetNextItemWidth(90.0f);
            if (ImGui::Combo("##skeleton_mode_small", &mode,
                             "File\0Preset\0")) {
                pm.load_skeleton_from_json = (mode == 0);
                if (pm.load_skeleton_from_json)
                    pm.skeleton_name.clear();
            }

            // Keep pm.skeleton_name in sync with preset selection
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
                ImGui::InputText("##calibfolder", &pm.calibration_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##loadprojectcalibration")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = pm.project_root_path;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibration", "Select Calibration Folder",
                        nullptr, cfg);
                }
            }

            ImGui::EndTable();
        }

        ImGui::Separator();

        // Validation (also require overwrite if target exists)
        const bool needed_ok =
            !pm.project_name.empty() && !pm.project_root_path.empty() &&
            (pm.camera_names.size() <= 1 || !pm.calibration_folder.empty()) &&
            (!pm.load_skeleton_from_json || !pm.skeleton_file.empty());

        // Right-align Create button
        float avail = ImGui::GetContentRegionAvail().x;
        const char *create_label = "Create Project##action";
        float w = ImGui::CalcTextSize(create_label).x +
                  ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));

        ImGui::BeginDisabled(!needed_ok);
        if (ImGui::Button(create_label)) {
            std::string err;
            if (!ensure_dir_exists(pm.project_path, &err))
                popups.pushError(err);

            // Configure the dialog
            IGFD::FileDialogConfig config;
            config.path = pm.project_path;
            config.fileName = "project.redproj"; // prefill name
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_ConfirmOverwrite; // warn on
                                                                  // overwrite
            ImGuiFileDialog::Instance()->OpenDialog(
                "SaveProjectFileDlg", "Save Project File",
                "Red Project{.redproj},All files{.*}", config);
        }
        ImGui::EndDisabled();
    }
    ImGui::End();

    if (ImGuiFileDialog::Instance()->Display("SaveProjectFileDlg", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            // Full path the user chose (may contain or omit extension)
            std::string outPath =
                ImGuiFileDialog::Instance()->GetFilePathName();

            if (!ends_with_ci(outPath, ".redproj")) {
                outPath += ".redproj";
            }

            std::string err;
            if (!setup_project(pm, skeleton, skeleton_map, &err)) {
                popups.pushError(err);
            } else if (!save_project_manager_json(pm, outPath, &err)) {
                popups.pushError(err);
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}

// Tear down existing media (decoder threads, demuxers, scene memory)
// so that load_images or load_videos can be called cleanly.
inline void
unload_media(PlaybackState &ps, ProjectManager &pm,
             std::vector<FFmpegDemuxer *> &demuxers,
             DecoderContext *dc_context,
             RenderScene *scene,
             std::vector<std::thread> &decoder_threads,
             std::vector<bool> &is_view_focused,
             std::unordered_map<std::string, bool> &window_was_decoding) {
    if (!ps.video_loaded)
        return;

    // Signal all decoder/image_loader threads to stop
    dc_context->stop_flag = true;
    for (auto &t : decoder_threads) {
        if (t.joinable())
            t.join();
    }
    decoder_threads.clear();
    dc_context->stop_flag = false;

    // Free demuxers
    for (auto *d : demuxers)
        delete d;
    demuxers.clear();

    // Free scene display buffers
    if (scene->display_buffer) {
        for (u32 j = 0; j < scene->num_cams; j++) {
            if (scene->display_buffer[j]) {
                for (u32 i = 0; i < scene->size_of_buffer; i++) {
#ifdef __APPLE__
                    free(scene->display_buffer[j][i].frame);
                    if (scene->display_buffer[j][i].pixel_buffer) {
                        CVPixelBufferRelease(scene->display_buffer[j][i].pixel_buffer);
                        scene->display_buffer[j][i].pixel_buffer = nullptr;
                    }
#else
                    if (scene->use_cpu_buffer)
                        free(scene->display_buffer[j][i].frame);
                    else
                        cudaFree(scene->display_buffer[j][i].frame);
#endif
                }
                free(scene->display_buffer[j]);
            }
        }
        free(scene->display_buffer);
        scene->display_buffer = nullptr;
    }

    // Free other scene arrays
    free(scene->image_width);
    scene->image_width = nullptr;
    free(scene->image_height);
    scene->image_height = nullptr;
    free(scene->seek_context);
    scene->seek_context = nullptr;
    free(scene->pbo_cuda);
    scene->pbo_cuda = nullptr;
#ifdef __APPLE__
    free(scene->image_descriptor);
    scene->image_descriptor = nullptr;
#endif

    scene->num_cams = 0;
    scene->size_of_buffer = 0;

    // Clear playback and project media state
    is_view_focused.clear();
    pm.camera_names.clear();
    window_need_decoding.clear();
    window_was_decoding.clear();
    ps.video_loaded = false;
    ps.play_video = false;
    ps.to_display_frame_number = 0;
    ps.read_head = 0;
    ps.slider_frame_number = 0;
    ps.just_seeked = false;
    ps.pause_seeked = false;
    dc_context->decoding_flag = false;
    dc_context->total_num_frame = INT_MAX;
    dc_context->estimated_num_frames = 0;

    // Clear stale per-camera decoded frame counters
    latest_decoded_frame.clear();

    // Reset realtime playback (load_images sets false; load_videos expects true)
    ps.realtime_playback = true;
    ps.accumulated_play_time = 0.0;
    ps.last_play_time_start = std::chrono::steady_clock::now();
    ps.pause_selected = 0;
}

inline void
load_images(std::map<std::string, std::string> &selected_files,
            PlaybackState &ps, ProjectManager &pm,
            std::vector<std::string> &imgs_names, RenderScene *scene,
            DecoderContext *dc_context, int label_buffer_size,
            std::vector<std::thread> &decoder_threads,
            std::vector<bool> &is_view_focused,
            std::unordered_map<std::string, bool> &window_was_decoding) {

    std::string file_ext;
    for (const auto &elem : selected_files) {
        std::size_t cam_string_position = elem.first.find("_");
        std::string cam_name = elem.first.substr(0, cam_string_position);
        std::string file_full = elem.first.substr(cam_string_position + 1);

        window_need_decoding[cam_name].store(true);
        window_was_decoding[cam_name] = true;

        // split "123.jpg" → name = "123", ext = "jpg"
        std::size_t dot_pos = file_full.rfind('.');
        std::string file_name = file_full.substr(0, dot_pos);
        file_ext = file_full.substr(dot_pos + 1); // "jpg"

        if (std::find(pm.camera_names.begin(), pm.camera_names.end(),
                      cam_name) == pm.camera_names.end()) {
            pm.camera_names.push_back(cam_name);
        }

        if (std::find(imgs_names.begin(), imgs_names.end(), file_name) ==
            imgs_names.end()) {
            imgs_names.push_back(file_name);
        }
    }

    auto to_number = [](const std::string &s) { return std::stoi(s); };

    std::sort(imgs_names.begin(), imgs_names.end(),
              [&](const std::string &a, const std::string &b) {
                  return to_number(a) < to_number(b);
              });

    dc_context->seek_interval = 1;
    dc_context->video_fps = 1;
    ps.realtime_playback = false;
    scene->num_cams = pm.camera_names.size();
    scene->image_width = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    scene->image_height = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    for (u32 j = 0; j < scene->num_cams; j++) {
        std::string file_name = pm.media_folder + "/" + pm.camera_names[j] +
                                "_" + imgs_names[0] + "." + file_ext;
#ifdef __APPLE__
        int w = 0, h = 0, ch = 0;
        stbi_info(file_name.c_str(), &w, &h, &ch);
        scene->image_width[j] = w;
        scene->image_height[j] = h;
#else
        cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
        scene->image_width[j] = image.cols;
        scene->image_height[j] = image.rows;
#endif
    }
    if (imgs_names.size() < label_buffer_size) {
        label_buffer_size = imgs_names.size();
    }
    render_allocate_scene_memory(scene, label_buffer_size);
    for (int i = 0; i < scene->num_cams; i++) {
        decoder_threads.push_back(
            std::thread(&image_loader, dc_context, imgs_names,
                        scene->display_buffer[i], scene->size_of_buffer,
                        &scene->seek_context[i], scene->use_cpu_buffer,
                        pm.camera_names[i], pm.media_folder, file_ext));
        is_view_focused.push_back(false);
    }
    ps.video_loaded = true;
}

inline void
load_videos(std::map<std::string, std::string> &selected_files,
            PlaybackState &ps, ProjectManager &pm,
            std::unordered_map<std::string, bool> &window_was_decoding,
            std::vector<FFmpegDemuxer *> &demuxers, DecoderContext *dc_context,
            RenderScene *scene, int label_buffer_size,
            std::vector<std::thread> &decoder_threads,
            std::vector<bool> &is_view_focused) {
    if (selected_files.empty()) {
        // order based on pm.camera_names
        for (const auto &cam_string : pm.camera_names) {
            window_need_decoding[cam_string].store(true);
            window_was_decoding[cam_string] = true;
            std::map<std::string, std::string> m;
            std::string media_filename =
                (std::filesystem::path(pm.media_folder) / (cam_string + ".mp4"))
                    .string();
            // std::cout << media_filename << std::endl;
            FFmpegDemuxer *demuxer =
                new FFmpegDemuxer(media_filename.c_str(), m);
            demuxers.push_back(demuxer);
        }
        std::map<std::string, std::string> m;
        std::string media_filename = (std::filesystem::path(pm.media_folder) /
                                      (pm.camera_names[0] + ".mp4"))
                                         .string();
        FFmpegDemuxer dummy_dmuxer(media_filename.c_str(), m);
        dc_context->seek_interval = (int)dummy_dmuxer.FindKeyFrameInterval();
        dc_context->video_fps = dummy_dmuxer.GetFramerate();
    } else {
        // ordered base on selected_files order
        for (const auto &elem : selected_files) {
            std::size_t cam_string_mp4_position = elem.first.find("mp4");
            std::string cam_string =
                elem.first.substr(0, cam_string_mp4_position - 1);
            pm.camera_names.push_back(cam_string);
            // std::cout << "camera names: " << cam_string << std::endl;
            window_need_decoding[cam_string].store(true);
            window_was_decoding[cam_string] = true;
            std::map<std::string, std::string> m;
            FFmpegDemuxer *demuxer = new FFmpegDemuxer(elem.second.c_str(), m);
            demuxers.push_back(demuxer);
        }
        std::map<std::string, std::string> m;
        FFmpegDemuxer dummy_dmuxer(selected_files.begin()->second.c_str(), m);
        dc_context->seek_interval = (int)dummy_dmuxer.FindKeyFrameInterval();
        dc_context->video_fps = dummy_dmuxer.GetFramerate();
    }

    scene->num_cams = demuxers.size();
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
