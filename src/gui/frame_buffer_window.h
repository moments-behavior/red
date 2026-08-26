#pragma once
#include "app_context.h"

// Draw the Frame Buffer window — a vertical list of buffered frames, one row
// per slot, newest-first order matching the read head.
// select_corr_head: the buffer index corresponding to the currently selected frame.
inline void DrawFrameBufferWindow(AppContext &ctx, int select_corr_head) {
    auto &ps = ctx.ps;
    auto &scene = *ctx.scene;
    auto &annotations = ctx.annotations;
    auto &skeleton = ctx.skeleton;

    if (!ps.video_loaded)
        return;

    ImGui::SetNextWindowSize(ImVec2(150, 420), ImGuiCond_FirstUseEver);
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

            // Vertical scrollable list: one full-width row per buffer slot.
            float scale = 1.15f;
            float font_size = ImGui::GetFontSize() * scale;
            float item_h = font_size + 4.0f;

            ImGui::SetWindowFontScale(scale);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
            ImGui::BeginChild("##vscroll", ImVec2(0, 0), false);
            float item_w = ImGui::GetContentRegionAvail().x;
            ImDrawList *dl = ImGui::GetWindowDrawList();

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

                // Color code: green = fully labeled + triangulated,
                // teal = partially labeled, default = unlabeled
                const char *text = label;
                ImU32 text_col;
                auto ann_it = annotations.find((u32)frame_num);
                if (ann_it != annotations.end() &&
                    frame_has_any_keypoints(ann_it->second)) {
                    bool complete = frame_is_complete(ann_it->second);
                    if (complete && skeleton.has_skeleton &&
                        (project_is_2d(ctx.pm) || scene.num_cams > 1)) {
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
                // Desync fix: this slot is a duplicate for a dropped frame
                // on the visible camera — tint it red.
                if (ctx.dc_context->sync_fix_active.load() &&
                    scene.display_buffer[visible_idx][buf_idx].dropped.load())
                    text_col = IM_COL32(230, 80, 80, 255);
                ImVec2 ts = ImGui::CalcTextSize(text);
                ImVec2 text_pos(pos.x + 4.0f, pos.y + (item_h - ts.y) * 0.5f);
                dl->AddText(text_pos, text_col, text);
                ImGui::PopID();
            }

            // Vertical scroll is ImGui's default wheel behaviour, so no custom
            // handler is needed here.
            ImGui::EndChild();
            ImGui::PopStyleVar(2);  // WindowPadding, ItemSpacing
            ImGui::SetWindowFontScale(1.0f);
        }
    }
    ImGui::End();
}
