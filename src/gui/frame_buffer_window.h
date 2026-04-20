#pragma once
#include "app_context.h"
#include "implot_internal.h"

// Draw the Frame Buffer window — shows buffered frames as horizontal selectables
// stacked top-to-bottom (vertical list, text reads left-to-right normally).
// select_corr_head: the buffer index corresponding to the currently selected frame.
inline void DrawFrameBufferWindow(AppContext &ctx, int select_corr_head) {
    auto &ps = ctx.ps;
    auto &scene = *ctx.scene;
    auto &annotations = ctx.annotations;
    auto &skeleton = ctx.skeleton;

    if (!ps.video_loaded)
        return;

    ImGui::SetNextWindowSize(ImVec2(120, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Frame Buffer")) {
        if (ps.play_video) {
            ImGui::BeginDisabled();
            ImGui::TextDisabled("Playing...");
            ImGui::EndDisabled();
        } else {
            // Find a visible camera index
            int visible_idx = 0;
            if (!ps.pause_seeked) {
                for (u32 i = 0; i < scene.num_cams && i < ctx.pm.camera_names.size(); i++) {
                    if (ctx.window_was_decoding[ctx.pm.camera_names[i]]) {
                        visible_idx = (int)i;
                        break;
                    }
                }
            }

            // Vertical scrollable column of frames, one row per buffer slot.
            float scale = 1.10f;
            float font_size = ImGui::GetFontSize() * scale;
            float item_h = font_size + 4.0f;
            float item_w = ImGui::GetContentRegionAvail().x;
            if (item_w < 40.0f) item_w = 40.0f;

            ImGui::SetWindowFontScale(scale);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0f, 2.0f));
            ImGui::BeginChild("##vscroll", ImVec2(0, 0), false,
                              ImGuiWindowFlags_NoScrollWithMouse |
                              ImGuiWindowFlags_NoScrollbar);
            ImDrawList *dl = ImGui::GetWindowDrawList();

            // Auto-scroll to keep the highlighted slot visible. Only trigger
            // when the selection changes so the user can still freely wheel-
            // scroll to a different region without being yanked back.
            static int s_last_selected = -1;
            bool want_scroll_to_selected =
                (ps.pause_selected != s_last_selected);
            s_last_selected = ps.pause_selected;

            for (u32 i = 0; i < scene.size_of_buffer; i++) {
                int buf_idx =
                    (i + ps.read_head) % scene.size_of_buffer;
                int frame_num =
                    scene.display_buffer[visible_idx][buf_idx].frame_number;

                char label[32];
                if (ctx.input_is_imgs)
                    snprintf(label, sizeof(label), "%d:%s",
                             frame_num, ctx.imgs_names[i].c_str());
                else
                    snprintf(label, sizeof(label), "%d", frame_num);

                bool is_selected = (ps.pause_selected == (int)i);

                ImGui::PushID((int)i);
                ImVec2 pos = ImGui::GetCursorScreenPos();
                if (ImGui::Selectable("##fbuf", is_selected, 0,
                                      ImVec2(item_w, item_h))) {
                    if (!is_selected) {
                        ps.pause_selected = (int)i;
                    }
                }
                // Scroll so the highlighted slot stays in view (centered).
                if (is_selected && want_scroll_to_selected) {
                    ImGui::SetScrollHereY(0.5f);
                }

                // Color code the text: green = fully labeled + triangulated,
                // teal = partially labeled, default = unlabeled.
                ImU32 text_col;
                auto ann_it = annotations.find((u32)frame_num);
                if (ann_it != annotations.end() &&
                    frame_has_any_keypoints(ann_it->second)) {
                    bool complete = frame_is_complete(ann_it->second);
                    if (complete && skeleton.has_skeleton && scene.num_cams > 1) {
                        for (int k = 0; k < skeleton.num_nodes; ++k)
                            if (!ann_it->second.kp3d[k].triangulated)
                                complete = false;
                    }
                    text_col = complete
                        ? IM_COL32(51, 204, 77, 255)   // green
                        : IM_COL32(51, 179, 179, 255); // teal
                } else {
                    text_col = is_selected
                        ? ImGui::GetColorU32(ImGuiCol_Text)
                        : ImGui::GetColorU32(ImGuiCol_TextDisabled);
                }
                // Normal left-to-right text, left-padded inside the selectable.
                ImVec2 text_pos(pos.x + 6.0f,
                                pos.y + (item_h - font_size) * 0.5f);
                dl->AddText(text_pos, text_col, label);
                ImGui::PopID();
            }

            // Mouse wheel → vertical scroll (default ImGui behavior; we just
            // don't override it now that the layout is vertical).
            if (ImGui::IsWindowHovered()) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f)
                    ImGui::SetScrollY(ImGui::GetScrollY() - wheel * item_h * 3.0f);
            }
            ImGui::EndChild();
            ImGui::PopStyleVar(2);  // WindowPadding, ItemSpacing
            ImGui::SetWindowFontScale(1.0f);
        }
    }
    ImGui::End();
}
