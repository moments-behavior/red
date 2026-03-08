#pragma once
#include "app_context.h"
#include "gui/gui_keypoints.h"
#include "gui/gui_save_load.h"
#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <ctime>

struct LabelingToolState {
    std::time_t last_saved = static_cast<std::time_t>(-1);
    bool save_requested = false;
};

inline void DrawLabelingToolWindow(
    LabelingToolState &state, AppContext &ctx,
    int current_frame_num, bool keypoints_find) {
    auto &pm = ctx.pm;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    auto &skeleton = ctx.skeleton;
    auto &keypoints_map = ctx.keypoints_map;
    auto &ps = ctx.ps;
    auto &popups = ctx.popups;
    auto &toasts = ctx.toasts;
    bool &input_is_imgs = ctx.input_is_imgs;
    auto &imgs_names = ctx.imgs_names;

    state.save_requested = false;

    if (ImGui::Begin("Labeling Tool")) {
        if (scene->num_cams > 1) {
            bool keypoint_triangulated_all = true;
            if (keypoints_find && scene->num_cams > 1) {
                for (int j = 0; j < skeleton.num_nodes; j++) {
                    if (!keypoints_map.at(current_frame_num)
                             ->kp3d[j]
                             .is_triangulated) {
                        keypoint_triangulated_all = false;
                        break;
                    }
                }
            } else {
                keypoint_triangulated_all = false;
            }
            bool apply_color =
                !keypoint_triangulated_all && keypoints_find;
            if (apply_color) {
                ImGui::PushStyleColor(
                    ImGuiCol_Button,
                    (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f));
                ImGui::PushStyleColor(
                    ImGuiCol_ButtonHovered,
                    (ImVec4)ImColor::HSV(0.8, 0.9f, 0.8f));
                ImGui::PushStyleColor(
                    ImGuiCol_ButtonActive,
                    (ImVec4)ImColor::HSV(0.8, 0.9f, 0.5f));
            }

            ImGui::BeginDisabled(!keypoints_find);
            if (ImGui::Button("Triangulate")) {
                reprojection(keypoints_map.at(current_frame_num),
                             &skeleton, pm.camera_params, scene);
            }
            ImGui::EndDisabled();

            if (apply_color) {
                ImGui::PopStyleColor(3);
            }
        } else {
            ImGui::Text("Please load a skeleton to view keypoints.");
        }

        if (ImGui::Button("Update keypoints working directory")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = pm.project_path;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseKeypointsFolder",
                "Choose keypoints working directory", nullptr, config);
        }
        ImGui::SameLine();
        ImGui::Text("%s", pm.keypoints_root_folder.c_str());

        if (ImGui::Button("Save Labeled Data")) {
            state.save_requested = true;
        }
        if (state.last_saved != static_cast<std::time_t>(-1)) {
            ImGui::SameLine();
            ImGui::Text("Last saved: %s", ctime(&state.last_saved));
        }

        if (ImGui::Button("Load Most Recent Labels")) {
            free_all_keypoints(keypoints_map, scene);
            std::string err;
            std::string most_recent_folder;
            if (find_most_recent_labels(pm.keypoints_root_folder,
                                        most_recent_folder, err)) {
                popups.pushError(err);
            } else {
                if (load_keypoints(most_recent_folder,
                                   keypoints_map, &skeleton, scene,
                                   pm.camera_names, err)) {
                    free_all_keypoints(keypoints_map, scene);
                    popups.pushError(err);
                }
            }
        }

        if (ImGui::Button("Load From Selected")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = pm.keypoints_root_folder;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "LoadFromSelected", "Load from selected", nullptr,
                config);
        }

        ImGui::Separator();

        auto next_labeled_frame_it = keypoints_map.end();
        for (auto it = keypoints_map.upper_bound(current_frame_num);
             it != keypoints_map.end(); ++it) {
            if (has_any_labels(it->second, skeleton, scene)) {
                next_labeled_frame_it = it;
                break;
            }
        }
        if (next_labeled_frame_it == keypoints_map.end()) {
            for (auto it = keypoints_map.begin();
                 it != keypoints_map.upper_bound(current_frame_num);
                 ++it) {
                if (has_any_labels(it->second, skeleton, scene)) {
                    next_labeled_frame_it = it;
                    break;
                }
            }
        }

        auto prev_labeled_frame_it = keypoints_map.end();
        auto lb = keypoints_map.lower_bound(current_frame_num);
        if (lb != keypoints_map.begin()) {
            for (auto it = std::prev(lb);;) {
                if (has_any_labels(it->second, skeleton, scene)) {
                    prev_labeled_frame_it = it;
                    break;
                }
                if (it == keypoints_map.begin())
                    break;
                --it;
            }
        }

        bool has_next_frame =
            (next_labeled_frame_it != keypoints_map.end());
        int next_frame_num =
            has_next_frame ? next_labeled_frame_it->first : -1;
        bool has_prev_frame =
            (prev_labeled_frame_it != keypoints_map.end());
        int previous_frame_num =
            has_prev_frame ? prev_labeled_frame_it->first : -1;

        if (ImGui::BeginTable("frame_nav_table", 2,
                              ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::BeginDisabled(!has_next_frame);
            bool button_pressed =
                ImGui::Button("Jump to next labeled frame");
            ImGui::EndDisabled();

            if (button_pressed && has_next_frame) {
                ps.play_video = false;
                seek_all_cameras(scene, next_frame_num,
                                 dc_context->video_fps, ps, true);
            }

            ImGui::TableSetColumnIndex(1);
            if (has_next_frame)
                ImGui::Text("%d", next_frame_num);
            else
                ImGui::Text("none");

            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::BeginDisabled(!has_prev_frame);
            if (ImGui::Button("Copy from previous labeled frame")) {
                if (keypoints_find) {
                    free_keypoints(keypoints_map[current_frame_num],
                                   scene);
                    keypoints_map.erase(current_frame_num);
                }

                KeyPoints *keypoints =
                    (KeyPoints *)malloc(sizeof(KeyPoints));
                allocate_keypoints(keypoints, scene, &skeleton);
                keypoints_map[current_frame_num] = keypoints;

                KeyPoints *prev = keypoints_map[previous_frame_num];
                KeyPoints *curr = keypoints_map[current_frame_num];
                copy_keypoints(curr, prev, scene, &skeleton);
            }
            ImGui::EndDisabled();

            ImGui::TableSetColumnIndex(1);
            if (has_prev_frame)
                ImGui::Text("%d", previous_frame_num);
            else
                ImGui::Text("none");

            ImGui::EndTable();
        }

        size_t labeled_count = 0;
        for (const auto &[frame_num, keypoints] : keypoints_map) {
            if (has_any_labels(keypoints, skeleton, scene,
                               /*yolo_thresh=*/0.5f)) {
                ++labeled_count;
            }
        }
        ImGui::Text("Total labeled frames : %zu", labeled_count);

    }
    ImGui::End();

    // File dialog handlers (called every frame, inside this draw function)
    if (ImGuiFileDialog::Instance()->Display("ChooseKeypointsFolder",
            ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.keypoints_root_folder =
                ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("LoadFromSelected",
            ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            auto selected_folder =
                ImGuiFileDialog::Instance()->GetCurrentPath();
            free_all_keypoints(keypoints_map, scene);
            std::string err;
            if (load_keypoints(selected_folder, keypoints_map, &skeleton,
                               scene, pm.camera_names, err)) {
                free_all_keypoints(keypoints_map, scene);
                popups.pushError(err);
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Ctrl+S save handling
    if (pm.plot_keypoints_flag &&
        ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_S, false) &&
        !ImGui::GetIO().WantTextInput) {
        state.save_requested = true;
    }

    if (state.save_requested) {
        save_keypoints(keypoints_map, &skeleton,
                       pm.keypoints_root_folder,
                       scene->num_cams, pm.camera_names,
                       &input_is_imgs, imgs_names);
        state.last_saved = time(NULL);
        toasts.pushSuccess("Labels saved");
    }
}
