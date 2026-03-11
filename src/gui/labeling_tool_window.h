#pragma once
#include "app_context.h"
#include "annotation.h"
#include "annotation_csv.h"
#include "gui/gui_keypoints.h"
#include "IconsForkAwesome.h"
#include "implot.h"
#include "implot_internal.h"
#include <imgui.h>
#include <ctime>

struct LabelingToolState {
    std::time_t last_saved = static_cast<std::time_t>(-1);
    bool save_requested = false;
    bool timeline_reset_pending = false;
};

inline void DrawLabelingToolWindow(
    LabelingToolState &state, AppContext &ctx,
    int current_frame_num, bool keypoints_find) {
    auto &pm = ctx.pm;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    auto &skeleton = ctx.skeleton;
    auto &annotations = ctx.annotations;
    auto &ps = ctx.ps;
    auto &popups = ctx.popups;
    auto &toasts = ctx.toasts;

    state.save_requested = false;

    if (ImGui::Begin("Labeling Tool")) {
        // Helper: find prev/next frame matching a predicate (with wraparound)
        struct PrevNext { int prev = -1; int next = -1; };
        auto find_prev_next = [&](auto predicate) -> PrevNext {
            PrevNext pn;
            // Next: search forward from current, wrap to beginning
            for (auto it = annotations.upper_bound(current_frame_num);
                 it != annotations.end(); ++it)
                if (predicate(it->second)) { pn.next = (int)it->first; break; }
            if (pn.next < 0)
                for (auto it = annotations.begin();
                     it != annotations.upper_bound(current_frame_num); ++it)
                    if (predicate(it->second)) { pn.next = (int)it->first; break; }
            // Prev: search backward from current, wrap to end
            auto lb = annotations.lower_bound(current_frame_num);
            if (lb != annotations.begin())
                for (auto it = std::prev(lb);;) {
                    if (predicate(it->second)) { pn.prev = (int)it->first; break; }
                    if (it == annotations.begin()) break;
                    --it;
                }
            if (pn.prev < 0 && !annotations.empty())
                for (auto it = std::prev(annotations.end());;) {
                    if (it->first <= (u32)current_frame_num) break;
                    if (predicate(it->second)) { pn.prev = (int)it->first; break; }
                    if (it == annotations.begin()) break;
                    --it;
                }
            return pn;
        };

        // Helper: render Prev [Jump] Next buttons. id_suffix for unique widget IDs.
        auto jump_buttons = [&](PrevNext pn, const char *id_suffix) {
            ImGui::BeginDisabled(pn.prev < 0);
            char prev_id[32]; snprintf(prev_id, sizeof(prev_id), ICON_FK_CHEVRON_LEFT " Prev##%s", id_suffix);
            if (ImGui::SmallButton(prev_id)) {
                ps.play_video = false;
                seek_all_cameras(scene, pn.prev, dc_context->video_fps, ps, true);
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("Jump");
            ImGui::SameLine();
            ImGui::BeginDisabled(pn.next < 0);
            char next_id[32]; snprintf(next_id, sizeof(next_id), "Next " ICON_FK_CHEVRON_RIGHT "##%s", id_suffix);
            if (ImGui::SmallButton(next_id)) {
                ps.play_video = false;
                seek_all_cameras(scene, pn.next, dc_context->video_fps, ps, true);
            }
            ImGui::EndDisabled();
        };

        // Find prev/next for keypoints
        auto kp_pn = find_prev_next([](const FrameAnnotation &fa) {
            return frame_has_any_keypoints(fa);
        });
        bool has_next = kp_pn.next >= 0;
        bool has_prev = kp_pn.prev >= 0;
        int next_frame = kp_pn.next;
        int prev_frame = kp_pn.prev;

        // === Top row: Save, Triangulate, Prev/Next label ===
        if (ImGui::Button(ICON_FK_FLOPPY_O " Save")) {
            state.save_requested = true;
        }

        if (scene->num_cams > 1) {
            ImGui::SameLine();

            bool keypoint_triangulated_all = true;
            if (keypoints_find && scene->num_cams > 1) {
                const auto &fa = annotations.at(current_frame_num);
                for (int j = 0; j < skeleton.num_nodes; j++) {
                    if (!fa.kp3d[j].triangulated) {
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
                reprojection(annotations.at(current_frame_num),
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
            // Copy annotations from prev frame into current frame
            const auto &prev_fa = annotations.at(prev_frame);
            FrameAnnotation new_fa = make_frame(skeleton.num_nodes, scene->num_cams, current_frame_num);
            // Copy keypoints from prev frame
            for (int c = 0; c < scene->num_cams && c < (int)prev_fa.cameras.size(); ++c) {
                for (int k = 0; k < skeleton.num_nodes && k < (int)prev_fa.cameras[c].keypoints.size(); ++k) {
                    new_fa.cameras[c].keypoints[k] = prev_fa.cameras[c].keypoints[k];
                }
                new_fa.cameras[c].active_id = prev_fa.cameras[c].active_id;
            }
            for (int k = 0; k < skeleton.num_nodes && k < (int)prev_fa.kp3d.size(); ++k) {
                new_fa.kp3d[k] = prev_fa.kp3d[k];
            }
            annotations[current_frame_num] = std::move(new_fa);
        }
        ImGui::EndDisabled();

        if (state.last_saved != static_cast<std::time_t>(-1)) {
            char time_buf[32];
            struct tm tm_buf;
            localtime_r(&state.last_saved, &tm_buf);
            strftime(time_buf, sizeof(time_buf), "%H:%M:%S", &tm_buf);
            ImGui::TextDisabled("Last saved: %s", time_buf);
        }

        ImGui::Separator();

        // === Collect labeled frames (shared by grid + timeline) ===
        struct LabeledFrameInfo { int frame; bool complete; };
        std::vector<LabeledFrameInfo> labeled_frames;
        for (const auto &[fnum, fa] : annotations) {
            if (!frame_has_any_keypoints(fa))
                continue;
            bool complete = true;
            if (skeleton.has_skeleton && scene->num_cams > 1) {
                for (int cam = 0; cam < scene->num_cams && cam < (int)fa.cameras.size(); ++cam)
                    for (int k = 0; k < skeleton.num_nodes; ++k)
                        if (k >= (int)fa.cameras[cam].keypoints.size() ||
                            !fa.cameras[cam].keypoints[k].labeled) { complete = false; goto done; }
                for (int k = 0; k < skeleton.num_nodes; ++k)
                    if (k >= (int)fa.kp3d.size() ||
                        !fa.kp3d[k].triangulated) { complete = false; goto done; }
            } else {
                complete = false;
            }
            done:
            labeled_frames.push_back({(int)fnum, complete});
        }

        // === Collect SAM mask frames ===
        struct MaskFrameInfo { int frame; };
        std::vector<MaskFrameInfo> mask_frames;
        for (const auto &[fnum, fa] : annotations) {
            if (frame_has_any_masks(fa))
                mask_frames.push_back({(int)fnum});
        }

        // === Collect bounding box frames ===
        struct BBoxFrameInfo { int frame; bool has_bbox; bool has_obb; };
        std::vector<BBoxFrameInfo> bbox_frames;
        for (const auto &[fnum, fa] : annotations) {
            bool any_bbox = false, any_obb = false;
            for (const auto &cam : fa.cameras) {
                if (cam.has_bbox()) any_bbox = true;
                if (cam.has_obb())  any_obb  = true;
            }
            if (any_bbox || any_obb)
                bbox_frames.push_back({(int)fnum, any_bbox, any_obb});
        }

        // === Shared constants (grid + timeline) ===
        const ImVec2 cell_size(16, 16);
        const float gap = ImGui::GetStyle().ItemSpacing.y;
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const ImU32 white = IM_COL32(255, 255, 255, 255);

        // Annotation type colors (shared between grid cells and timeline ticks)
        const ImVec4 color_teal(0.2f, 0.7f, 0.7f, 1.0f);
        const ImVec4 color_green(0.2f, 0.8f, 0.3f, 1.0f);
        const ImVec4 color_orange(0.9f, 0.55f, 0.12f, 1.0f);
        const ImVec4 color_purple(0.63f, 0.35f, 0.86f, 1.0f);
        const ImVec4 color_lilac(0.78f, 0.59f, 1.0f, 1.0f);

        // Grid cell PushID offsets (max ~10k frames per section before collision)
        constexpr int kKpIdOffset   = 0;
        constexpr int kSamIdOffset  = 10000;
        constexpr int kBBoxIdOffset = 20000;

        // Helper: render a clickable grid cell with custom drawing.
        // draw_fn(ImDrawList*, ImVec2 min, ImVec2 max) draws the cell interior.
        // tooltip is shown on hover. Returns true if clicked.
        auto grid_cell = [&](int idx, int frame_num, const char *tooltip_text,
                             auto draw_fn) -> bool {
            bool clicked = false;
            bool is_current = (frame_num == current_frame_num);

            ImGui::PushID(idx);

            // White border for current frame (drawn before button so it's behind)
            if (is_current) {
                ImVec2 pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddRect(
                    ImVec2(pos.x - 1, pos.y - 1),
                    ImVec2(pos.x + cell_size.x + 1, pos.y + cell_size.y + 1),
                    white, 0.0f, 0, 2.0f);
            }

            // Invisible button for click + hover detection
            if (ImGui::InvisibleButton("##cell", cell_size)) {
                ps.play_video = false;
                seek_all_cameras(scene, frame_num,
                                 dc_context->video_fps, ps, true);
                clicked = true;
            }

            // Draw custom shape into the button rect
            ImVec2 rmin = ImGui::GetItemRectMin();
            ImVec2 rmax = ImGui::GetItemRectMax();
            draw_fn(ImGui::GetWindowDrawList(), rmin, rmax);

            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tooltip_text);

            ImGui::PopID();
            return clicked;
        };

        // Helper: wrap to next row or stay on same line
        auto grid_wrap = [&](size_t i, size_t count) {
            if (i + 1 < count) {
                float next_x = ImGui::GetItemRectMax().x + gap + cell_size.x;
                if (next_x < ImGui::GetWindowPos().x + avail_w)
                    ImGui::SameLine(0, gap);
            }
        };

        // ─── Section 1: Keypoint Labels ───
        ImGui::Text("Keypoint Labels (%zu)", labeled_frames.size());

        if (!labeled_frames.empty()) {
            ImU32 teal_u32  = ImGui::ColorConvertFloat4ToU32(color_teal);
            ImU32 green_u32 = ImGui::ColorConvertFloat4ToU32(color_green);

            for (size_t i = 0; i < labeled_frames.size(); ++i) {
                auto &lf = labeled_frames[i];
                ImU32 fill = lf.complete ? green_u32 : teal_u32;
                char tip[64];
                snprintf(tip, sizeof(tip), "Frame %d", lf.frame);

                grid_cell(kKpIdOffset + (int)i, lf.frame, tip,
                    [fill](ImDrawList *dl, ImVec2 mn, ImVec2 mx) {
                        dl->AddRectFilled(mn, mx, fill);
                    });
                grid_wrap(i, labeled_frames.size());
            }
        }

        // ─── Section 2: SAM Labels ───
        if (!mask_frames.empty()) {
            ImGui::Spacing();
            auto mask_pn = find_prev_next([](const FrameAnnotation &fa) {
                return frame_has_any_masks(fa);
            });
            ImGui::Text("SAM Labels (%zu)", mask_frames.size());
            ImGui::SameLine();
            jump_buttons(mask_pn, "sam");

            ImU32 orange_u32 = ImGui::ColorConvertFloat4ToU32(color_orange);

            for (size_t i = 0; i < mask_frames.size(); ++i) {
                auto &mf = mask_frames[i];
                char tip[64];
                snprintf(tip, sizeof(tip), "Frame %d", mf.frame);

                grid_cell(kSamIdOffset + (int)i, mf.frame, tip,
                    [orange_u32](ImDrawList *dl, ImVec2 mn, ImVec2 mx) {
                        float cx = (mn.x + mx.x) * 0.5f;
                        float cy = (mn.y + mx.y) * 0.5f;
                        float r  = (mx.x - mn.x) * 0.5f - 1.0f;
                        dl->AddCircleFilled(ImVec2(cx, cy), r, orange_u32, 16);
                    });
                grid_wrap(i, mask_frames.size());
            }
        }

        // ─── Section 3: Bounding Box Labels ───
        if (!bbox_frames.empty()) {
            ImGui::Spacing();
            auto bbox_pn = find_prev_next([](const FrameAnnotation &fa) {
                for (const auto &cam : fa.cameras)
                    if (cam.has_bbox() || cam.has_obb()) return true;
                return false;
            });
            ImGui::Text("Bounding Box Labels (%zu)", bbox_frames.size());
            ImGui::SameLine();
            jump_buttons(bbox_pn, "bbox");

            ImU32 purple_u32  = ImGui::ColorConvertFloat4ToU32(color_purple);
            ImU32 lilac_u32   = ImGui::ColorConvertFloat4ToU32(color_lilac);

            for (size_t i = 0; i < bbox_frames.size(); ++i) {
                auto &bf = bbox_frames[i];
                char tip[96];
                if (bf.has_bbox && bf.has_obb)
                    snprintf(tip, sizeof(tip), "Frame %d (BBox+OBB)", bf.frame);
                else if (bf.has_obb)
                    snprintf(tip, sizeof(tip), "Frame %d (OBB)", bf.frame);
                else
                    snprintf(tip, sizeof(tip), "Frame %d (BBox)", bf.frame);

                bool has_bb = bf.has_bbox, has_ob = bf.has_obb;
                grid_cell(kBBoxIdOffset + (int)i, bf.frame, tip,
                    [purple_u32, lilac_u32, has_bb, has_ob](ImDrawList *dl, ImVec2 mn, ImVec2 mx) {
                        // BBox: purple square outline (inset 1px for clarity)
                        if (has_bb) {
                            dl->AddRect(
                                ImVec2(mn.x + 1, mn.y + 1),
                                ImVec2(mx.x - 1, mx.y - 1),
                                purple_u32, 0.0f, 0, 1.5f);
                        }
                        // OBB: lighter purple diamond outline
                        if (has_ob) {
                            float cx = (mn.x + mx.x) * 0.5f;
                            float cy = (mn.y + mx.y) * 0.5f;
                            float hx = (mx.x - mn.x) * 0.5f - 1.5f;
                            float hy = (mx.y - mn.y) * 0.5f - 1.5f;
                            ImVec2 pts[4] = {
                                ImVec2(cx, cy - hy),   // top
                                ImVec2(cx + hx, cy),   // right
                                ImVec2(cx, cy + hy),   // bottom
                                ImVec2(cx - hx, cy),   // left
                            };
                            dl->AddPolyline(pts, 4, lilac_u32, ImDrawFlags_Closed, 1.5f);
                        }
                    });
                grid_wrap(i, bbox_frames.size());
            }
        }

        // === Timeline minimap (ImPlot — all annotation types) ===
        ImGui::Spacing();
        int total_frames = dc_context->estimated_num_frames;
        bool has_any_annotations = !labeled_frames.empty() || !mask_frames.empty() || !bbox_frames.empty();
        if (total_frames > 0 && has_any_annotations) {

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

            // Build tick arrays for each annotation type
            std::vector<double> teal_x, green_x, orange_x, purple_x, lilac_x;
            for (auto &lf : labeled_frames) {
                if (lf.complete) green_x.push_back((double)lf.frame);
                else teal_x.push_back((double)lf.frame);
            }
            for (auto &mf : mask_frames)
                orange_x.push_back((double)mf.frame);
            for (auto &bf : bbox_frames) {
                if (bf.has_bbox) purple_x.push_back((double)bf.frame);
                if (bf.has_obb) lilac_x.push_back((double)bf.frame);
            }

            // Collect all annotated frames for click-to-seek
            std::vector<int> all_annotated_frames;
            for (auto &lf : labeled_frames) all_annotated_frames.push_back(lf.frame);
            for (auto &mf : mask_frames) all_annotated_frames.push_back(mf.frame);
            for (auto &bf : bbox_frames) all_annotated_frames.push_back(bf.frame);
            std::sort(all_annotated_frames.begin(), all_annotated_frames.end());
            all_annotated_frames.erase(
                std::unique(all_annotated_frames.begin(), all_annotated_frames.end()),
                all_annotated_frames.end());

            if (state.timeline_reset_pending) {
                ImPlot::SetNextAxesLimits(0, total_frames, 0, 1);
                state.timeline_reset_pending = false;
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

                // Keypoint ticks (teal=partial, green=complete)
                if (!teal_x.empty()) {
                    ImPlot::SetNextLineStyle(color_teal, 2.0f);
                    ImPlot::PlotInfLines("##teal", teal_x.data(), (int)teal_x.size());
                }
                if (!green_x.empty()) {
                    ImPlot::SetNextLineStyle(color_green, 2.0f);
                    ImPlot::PlotInfLines("##green", green_x.data(), (int)green_x.size());
                }

                // SAM mask ticks (orange)
                if (!orange_x.empty()) {
                    ImPlot::SetNextLineStyle(color_orange, 2.0f);
                    ImPlot::PlotInfLines("##sam", orange_x.data(), (int)orange_x.size());
                }

                // BBox ticks (purple)
                if (!purple_x.empty()) {
                    ImPlot::SetNextLineStyle(color_purple, 2.0f);
                    ImPlot::PlotInfLines("##bbox", purple_x.data(), (int)purple_x.size());
                }

                // OBB ticks (lilac)
                if (!lilac_x.empty()) {
                    ImPlot::SetNextLineStyle(color_lilac, 2.0f);
                    ImPlot::PlotInfLines("##obb", lilac_x.data(), (int)lilac_x.size());
                }

                // Double-click to reset to full video range
                if (ImPlot::IsPlotHovered() && ImGui::IsMouseDoubleClicked(0))
                    state.timeline_reset_pending = true;

                // Click to seek + tooltip (combined to avoid duplicate computation)
                if (ImPlot::IsPlotHovered()) {
                    ImPlotPoint mp = ImPlot::GetPlotMousePos();
                    ImPlotRect lims = ImPlot::GetPlotLimits();
                    double px_per_frame = timeline_w / (lims.X.Max - lims.X.Min);
                    double tolerance = 5.0 / px_per_frame;
                    int nearest = -1;
                    double nearest_dist = tolerance + 1;
                    for (int f : all_annotated_frames) {
                        double d = fabs((double)f - mp.x);
                        if (d < nearest_dist) { nearest_dist = d; nearest = f; }
                    }
                    if (nearest >= 0 && nearest_dist <= tolerance) {
                        ImGui::SetTooltip("Frame %d", nearest);
                        if (ImGui::IsMouseClicked(0)) {
                            ps.play_video = false;
                            seek_all_cameras(scene, nearest,
                                             dc_context->video_fps, ps, true);
                        }
                    }
                }

                ImPlot::EndPlot();
            }
            ImPlot::PopStyleVar();
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
        std::string save_err;
        std::string saved_folder = AnnotationCSV::save_all(
            pm.keypoints_root_folder, skeleton.name,
            annotations, scene->num_cams, skeleton.num_nodes,
            pm.camera_names, &save_err);
        if (saved_folder.empty()) {
            toasts.pushError("Save failed: " + save_err);
        } else {
            state.last_saved = time(NULL);
            toasts.pushSuccess("Labels saved");
        }
    }
}
