#pragma once
#include "app_context.h"
#include "gui/gui_keypoints.h"
#include "gui/gui_save_load.h"
#include "IconsForkAwesome.h"
#include "implot.h"
#include "implot_internal.h"
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
        // Find prev/next labeled frames (used by Prev/Next buttons)
        auto next_labeled_it = keypoints_map.end();
        for (auto it = keypoints_map.upper_bound(current_frame_num);
             it != keypoints_map.end(); ++it) {
            if (has_any_labels(it->second, skeleton, scene)) {
                next_labeled_it = it; break;
            }
        }
        if (next_labeled_it == keypoints_map.end()) {
            for (auto it = keypoints_map.begin();
                 it != keypoints_map.upper_bound(current_frame_num); ++it) {
                if (has_any_labels(it->second, skeleton, scene)) {
                    next_labeled_it = it; break;
                }
            }
        }
        auto prev_labeled_it = keypoints_map.end();
        auto lb = keypoints_map.lower_bound(current_frame_num);
        if (lb != keypoints_map.begin()) {
            for (auto it = std::prev(lb);;) {
                if (has_any_labels(it->second, skeleton, scene)) {
                    prev_labeled_it = it; break;
                }
                if (it == keypoints_map.begin()) break;
                --it;
            }
        }
        // Wrap around: search backward from end of map
        if (prev_labeled_it == keypoints_map.end() && !keypoints_map.empty()) {
            for (auto it = std::prev(keypoints_map.end());;) {
                if (it->first <= (u32)current_frame_num) break;
                if (has_any_labels(it->second, skeleton, scene)) {
                    prev_labeled_it = it; break;
                }
                if (it == keypoints_map.begin()) break;
                --it;
            }
        }
        bool has_next = (next_labeled_it != keypoints_map.end());
        bool has_prev = (prev_labeled_it != keypoints_map.end());
        int next_frame = has_next ? (int)next_labeled_it->first : -1;
        int prev_frame = has_prev ? (int)prev_labeled_it->first : -1;

        // === Top row: Save, Triangulate, Prev/Next label ===
        if (ImGui::Button(ICON_FK_FLOPPY_O " Save")) {
            state.save_requested = true;
        }

        if (scene->num_cams > 1) {
            ImGui::SameLine();

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
        }

        // Prev / Jump to Label / Next
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        ImGui::BeginDisabled(!has_prev);
        if (ImGui::Button(ICON_FK_CHEVRON_LEFT " Prev")) {
            ps.play_video = false;
            seek_all_cameras(scene, prev_frame,
                             dc_context->video_fps, ps, true);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "Jump");
        ImGui::SameLine();

        ImGui::BeginDisabled(!has_next);
        if (ImGui::Button("Next " ICON_FK_CHEVRON_RIGHT)) {
            ps.play_video = false;
            seek_all_cameras(scene, next_frame,
                             dc_context->video_fps, ps, true);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        ImGui::BeginDisabled(!has_prev);
        if (ImGui::Button("Copy Prev")) {
            if (keypoints_find) {
                free_keypoints(keypoints_map[current_frame_num], scene);
                keypoints_map.erase(current_frame_num);
            }
            KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
            allocate_keypoints(keypoints, scene, &skeleton);
            keypoints_map[current_frame_num] = keypoints;
            KeyPoints *prev = keypoints_map[prev_frame];
            KeyPoints *curr = keypoints_map[current_frame_num];
            copy_keypoints(curr, prev, scene, &skeleton);
        }
        ImGui::EndDisabled();

        if (state.last_saved != static_cast<std::time_t>(-1)) {
            ImGui::TextDisabled("Last saved: %s", ctime(&state.last_saved));
        }

        ImGui::Separator();

        // === Collect labeled frames (shared by grid + timeline) ===
        struct LabeledFrameInfo { int frame; bool complete; };
        std::vector<LabeledFrameInfo> labeled_frames;
        for (const auto &[fnum, kp] : keypoints_map) {
            if (!has_any_labels(kp, skeleton, scene))
                continue;
            bool complete = true;
            if (skeleton.has_skeleton && scene->num_cams > 1) {
                for (int cam = 0; cam < scene->num_cams && cam < MAX_VIEWS; ++cam)
                    for (int k = 0; k < skeleton.num_nodes; ++k)
                        if (!kp->kp2d[cam][k].is_labeled) { complete = false; goto done; }
                for (int k = 0; k < skeleton.num_nodes; ++k)
                    if (!kp->kp3d[k].is_triangulated) { complete = false; goto done; }
            } else {
                complete = false;
            }
            done:
            labeled_frames.push_back({(int)fnum, complete});
        }

        // === Labeled Frames grid ===
        ImGui::Text("Labeled Frames (%zu)", labeled_frames.size());

        if (!labeled_frames.empty()) {
            const ImVec4 teal(0.2f, 0.7f, 0.7f, 1.0f);
            const ImVec4 green(0.2f, 0.8f, 0.3f, 1.0f);
            const ImU32 teal_u32 = IM_COL32(51, 179, 179, 255);
            const ImU32 green_u32 = IM_COL32(51, 204, 77, 255);
            const ImVec2 cell_size(16, 16);
            const float gap = ImGui::GetStyle().ItemSpacing.y;
            float avail_w = ImGui::GetContentRegionAvail().x;

            for (size_t i = 0; i < labeled_frames.size(); ++i) {
                auto &lf = labeled_frames[i];
                bool is_current = (lf.frame == current_frame_num);

                ImGui::PushID((int)i);
                ImGuiColorEditFlags flags =
                    ImGuiColorEditFlags_NoAlpha |
                    ImGuiColorEditFlags_NoPicker |
                    ImGuiColorEditFlags_NoTooltip |
                    ImGuiColorEditFlags_NoDragDrop;

                ImVec4 color = lf.complete ? green : teal;

                if (is_current) {
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    ImGui::GetWindowDrawList()->AddRect(
                        ImVec2(pos.x - 1, pos.y - 1),
                        ImVec2(pos.x + cell_size.x + 1, pos.y + cell_size.y + 1),
                        IM_COL32(255, 255, 255, 255), 0.0f, 0, 2.0f);
                }

                if (ImGui::ColorButton("##cell", color, flags, cell_size)) {
                    ps.play_video = false;
                    seek_all_cameras(scene, lf.frame,
                                     dc_context->video_fps, ps, true);
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Frame %d", lf.frame);

                ImGui::PopID();

                if (i + 1 < labeled_frames.size()) {
                    float next_x = ImGui::GetItemRectMax().x + gap + cell_size.x;
                    if (next_x < ImGui::GetWindowPos().x + avail_w)
                        ImGui::SameLine(0, gap);
                }
            }

            // === Timeline minimap (ImPlot — scroll zoom, drag pan, box select) ===
            ImGui::Spacing();
            int total_frames = dc_context->estimated_num_frames;
            if (total_frames > 0) {
                // Reserve space for rotated "Timeline" label on the left
                float label_font = ImGui::GetFontSize();
                float label_margin = label_font + 6.0f;
                float timeline_w = ImGui::GetContentRegionAvail().x - label_margin;
                float timeline_h = 60.0f;

                // Draw rotated "Timeline" label on the left
                {
                    ImVec2 label_pos = ImGui::GetCursorScreenPos();
                    float text_w = ImGui::CalcTextSize("Timeline").x;
                    ImVec2 tp(label_pos.x + (label_margin - label_font) * 0.5f,
                              label_pos.y + (timeline_h + text_w) * 0.5f);
                    ImPlot::AddTextVertical(ImGui::GetWindowDrawList(), tp,
                        ImGui::GetColorU32(ImGuiCol_Text), "Timeline");
                    ImGui::Dummy(ImVec2(label_margin, timeline_h));
                    ImGui::SameLine();
                }

                // Split labeled frames into teal/green arrays
                std::vector<double> teal_x, green_x;
                for (auto &lf : labeled_frames) {
                    if (lf.complete)
                        green_x.push_back((double)lf.frame);
                    else
                        teal_x.push_back((double)lf.frame);
                }

                // Double-click resets to full range: override ImPlot default
                // by forcing limits when double-click detected on hovered plot
                static bool reset_pending = false;
                if (reset_pending) {
                    ImPlot::SetNextAxesLimits(0, total_frames, 0, 1);
                    reset_pending = false;
                }

                ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(4, 2));
                ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoTitle |
                                         ImPlotFlags_NoMouseText;
                if (ImPlot::BeginPlot("##timeline", ImVec2(timeline_w, timeline_h), plot_flags)) {
                    ImPlotAxisFlags x_flags = ImPlotAxisFlags_NoLabel;
                    ImPlotAxisFlags y_flags = ImPlotAxisFlags_NoLabel |
                                              ImPlotAxisFlags_NoTickLabels |
                                              ImPlotAxisFlags_NoTickMarks |
                                              ImPlotAxisFlags_NoGridLines |
                                              ImPlotAxisFlags_Lock;
                    ImPlot::SetupAxes("frame number", nullptr, x_flags, y_flags);
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, total_frames, ImPlotCond_Once);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1, ImPlotCond_Always);
                    ImPlot::SetupAxisZoomConstraints(ImAxis_X1, 50, total_frames);

                    // Current frame indicator
                    double cf = (double)current_frame_num;
                    ImPlot::SetNextLineStyle(ImVec4(1, 1, 1, 0.4f), 1.0f);
                    ImPlot::PlotInfLines("##current", &cf, 1);

                    // Teal tick marks (partially labeled)
                    if (!teal_x.empty()) {
                        ImPlot::SetNextLineStyle(teal, 2.0f);
                        ImPlot::PlotInfLines("##teal", teal_x.data(), (int)teal_x.size());
                    }

                    // Green tick marks (fully labeled + triangulated)
                    if (!green_x.empty()) {
                        ImPlot::SetNextLineStyle(green, 2.0f);
                        ImPlot::PlotInfLines("##green", green_x.data(), (int)green_x.size());
                    }

                    // Double-click to reset to full video range
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseDoubleClicked(0))
                        reset_pending = true;

                    // Click on a tick mark to seek
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0)) {
                        ImPlotPoint mp = ImPlot::GetPlotMousePos();
                        ImPlotRect lims = ImPlot::GetPlotLimits();
                        double px_per_frame = timeline_w / (lims.X.Max - lims.X.Min);
                        double tolerance = 5.0 / px_per_frame;
                        int nearest = -1;
                        double nearest_dist = tolerance + 1;
                        for (auto &lf : labeled_frames) {
                            double d = fabs((double)lf.frame - mp.x);
                            if (d < nearest_dist) {
                                nearest_dist = d;
                                nearest = lf.frame;
                            }
                        }
                        if (nearest >= 0 && nearest_dist <= tolerance) {
                            ps.play_video = false;
                            seek_all_cameras(scene, nearest,
                                             dc_context->video_fps, ps, true);
                        }
                    }

                    // Tooltip for nearest tick
                    if (ImPlot::IsPlotHovered()) {
                        ImPlotPoint mp = ImPlot::GetPlotMousePos();
                        ImPlotRect lims = ImPlot::GetPlotLimits();
                        double px_per_frame = timeline_w / (lims.X.Max - lims.X.Min);
                        double tolerance = 5.0 / px_per_frame;
                        for (auto &lf : labeled_frames) {
                            if (fabs((double)lf.frame - mp.x) <= tolerance) {
                                ImGui::SetTooltip("Frame %d", lf.frame);
                                break;
                            }
                        }
                    }

                    ImPlot::EndPlot();
                }
                ImPlot::PopStyleVar();

            }
        }

    }
    ImGui::End();

    // Ctrl+S save handling
    if (pm.plot_keypoints_flag &&
        ImGui::GetIO().KeyCtrl &&
        ImGui::IsKeyPressed(ImGuiKey_S, false) &&
        !ImGui::GetIO().WantTextInput) {
        state.save_requested = true;
    }

    // Toolbar Save button (from main menu bar)
    if (ctx.save_requested) {
        state.save_requested = true;
        ctx.save_requested = false;
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
