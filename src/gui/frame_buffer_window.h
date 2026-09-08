#pragma once
#include "app_context.h"
#include "keypoint_colors.h"

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

                // The frame number is the label. For images the source file
                // name is worth appending only when it differs from the frame
                // index -- a tailcycle group names frame 477 "000477", so
                // printing both would just repeat itself.
                //
                // This used to index imgs_names by `i`, the ring-slot position,
                // which named an unrelated file: "477:000017" meant "frame 477
                // sits in slot 17".
                char label[64];
                const std::string *src_name = nullptr;
                if (ctx.input_is_imgs && frame_num >= 0 &&
                    frame_num < (int)ctx.imgs_names.size()) {
                    const std::string &n = ctx.imgs_names[frame_num];
                    // Not stoi: this runs every frame in the render loop, and a
                    // non-numeric name would throw rather than mislabel.
                    long v = 0;
                    bool numeric = !n.empty();
                    for (char c : n) {
                        if (c < '0' || c > '9') { numeric = false; break; }
                        v = v * 10 + (c - '0');
                    }
                    if (!numeric || v != frame_num) src_name = &n;
                }
                if (src_name)
                    snprintf(label, sizeof(label), "%d (%s)", frame_num,
                             src_name->c_str());
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

                // Label state, in the shared vocabulary (keypoint_colors.h),
                // classified by the shared frame_kp_progress so this and the
                // Labeling Tool's timeline cannot disagree about a frame.
                // Unlabelled stays the theme's disabled text: in this list it
                // is the common case, and an accent would light up the whole
                // window.
                const char *text = label;
                ImU32 text_col;
                auto ann_it = annotations.find((u32)frame_num);
                const bool ann_ok = ann_it != annotations.end() &&
                                    !ann_it->second.empty();
                const FrameAnnotation *fa =
                    ann_ok ? &ann_it->second.front() : nullptr;
                KpProgress prog =
                    fa ? frame_kp_progress(*fa, skeleton.num_nodes,
                                           (int)scene.num_cams,
                                           project_is_2d(ctx.pm),
                                           skeleton.has_skeleton)
                       : KpProgress::None;
                const char *state_tip = nullptr;
                if (fa && fa->needs_improvement) {
                    text_col = ImGui::ColorConvertFloat4ToU32(kLabelNeedsFix);
                    state_tip = "needs fixing";
                } else switch (prog) {
                case KpProgress::AllViews:
                    text_col = ImGui::ColorConvertFloat4ToU32(kLabelComplete);
                    state_tip = "all views";
                    break;
                case KpProgress::Triangulated:
                    text_col = ImGui::ColorConvertFloat4ToU32(kLabelTriangulated);
                    state_tip = "triangulated";
                    break;
                case KpProgress::Untriangulated:
                    text_col = ImGui::ColorConvertFloat4ToU32(kLabelUntriangulated);
                    state_tip = "untriangulated";
                    break;
                case KpProgress::None:
                default:
                    text_col = is_selected
                        ? ImGui::GetColorU32(ImGuiCol_Text)
                        : ImGui::GetColorU32(ImGuiCol_TextDisabled);
                    break;
                }

                // Desync fix: this slot holds a duplicate standing in for a
                // frame the visible camera dropped. That is the media being at
                // fault, not the labels, so it gets its own channel -- a red
                // stripe down the left edge -- instead of overwriting the text
                // colour. Recolouring the text lost the label state entirely:
                // a complete, triangulated frame that happened to be a
                // duplicate read as plain red.
                const bool dropped =
                    ctx.dc_context->sync_fix_active.load() &&
                    scene.display_buffer[visible_idx][buf_idx].dropped.load();
                if (dropped)
                    dl->AddRectFilled(
                        ImVec2(pos.x, pos.y),
                        ImVec2(pos.x + 3.0f, pos.y + item_h),
                        ImGui::ColorConvertFloat4ToU32(kFrameDropped));

                if (ImGui::IsItemHovered() && (state_tip || dropped)) {
                    if (state_tip && dropped)
                        ImGui::SetTooltip(
                            "%s \xE2\x80\x94 and a duplicate: a camera dropped "
                            "this frame, so the nearest decoded one is shown.",
                            state_tip);
                    else if (dropped)
                        ImGui::SetTooltip(
                            "A camera dropped this frame; the nearest decoded "
                            "one is shown in its place.");
                    else
                        ImGui::SetTooltip("%s", state_tip);
                }

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
