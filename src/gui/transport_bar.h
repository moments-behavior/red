#pragma once
#include "app_context.h"
#include "IconsForkAwesome.h"
#include "utils.h"
#include <algorithm>
#include <chrono>

struct TransportBarState {
    // Cmd+click on slider: replace slider with InputInt, pause playback.
    bool slider_text_editing = false;
    bool focus_input = false;  // one-shot: give InputInt keyboard focus
    int edit_buf = 0;
};

inline void DrawTransportBar(TransportBarState &state, AppContext &ctx) {
    int &current_frame_num = ctx.current_frame_num;
    if (!ctx.ps.video_loaded) return;

    ImGuiViewport *vp = ImGui::GetMainViewport();
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings;
    if (!ImGui::BeginViewportSideBar("##TransportBar", vp, ImGuiDir_Up,
                                      34.0f, flags)) {
        ImGui::End();
        return;
    }

    auto &ps = ctx.ps;
    auto *dc = ctx.dc_context;
    auto &display = ctx.display;
    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;

    // Vertically center controls in the bar
    float bar_h = ImGui::GetContentRegionAvail().y;
    float frame_h = ImGui::GetFrameHeight();
    float pad_y = (bar_h - frame_h) * 0.5f;
    if (pad_y > 0.0f) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + pad_y);

    // === Bar label ===
    ImVec4 label_col(0.5f, 0.7f, 1.0f, 1.0f);   // light blue (reused below)
    ImVec4 green_col(0.5f, 0.9f, 0.5f, 1.0f);    // light green
    ImGui::SetWindowFontScale(1.15f);
    ImGui::TextColored(green_col, "Media Controls");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::SameLine(0, spacing * 3);

    // === Transport buttons ===
    if (ImGui::Button(ICON_FK_FAST_BACKWARD)) {
        int f = std::max(0, current_frame_num - 10 * dc->seek_interval);
        seek_all_cameras(ctx.scene, f, dc->video_fps, ps, false);
    }
    ImGui::SameLine(0.0f, spacing);
    if (ImGui::Button(ICON_FK_STEP_BACKWARD)) {
        int f = std::max(0, current_frame_num - dc->seek_interval);
        seek_all_cameras(ctx.scene, f, dc->video_fps, ps, false);
    }
    ImGui::SameLine(0.0f, spacing);

    // Play/Pause/Repeat button with colored styling
    if (ps.to_display_frame_number == (dc->total_num_frame - 1)) {
        ImVec4 repeat_normal = ImVec4(0.85f, 0.75f, 0.20f, 1.0f);
        ImVec4 repeat_hover  = ImVec4(0.90f, 0.80f, 0.25f, 1.0f);
        ImVec4 repeat_active = ImVec4(0.80f, 0.70f, 0.18f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Button, repeat_normal);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, repeat_hover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, repeat_active);
        if (ImGui::Button(ICON_FK_REPEAT)) {
            seek_all_cameras(ctx.scene, 0, dc->video_fps, ps, false);
        }
        ImGui::PopStyleColor(3);
    } else {
        ImVec4 normal, hover, active;
        if (ps.play_video) {
            normal = ImVec4(0.8f, 0.3f, 0.3f, 1.0f);
            hover  = ImVec4(0.9f, 0.4f, 0.4f, 1.0f);
            active = ImVec4(0.7f, 0.2f, 0.2f, 1.0f);
        } else {
            normal = ImVec4(0.2f, 0.6f, 0.2f, 1.0f);
            hover  = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
            active = ImVec4(0.3f, 0.75f, 0.3f, 1.0f);
        }
        ImGui::PushStyleColor(ImGuiCol_Button, normal);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
        if (ImGui::Button(ps.play_video ? ICON_FK_PAUSE : ICON_FK_PLAY)) {
            ps.play_video = !ps.play_video;
            if (ps.play_video) {
                ps.pause_seeked = false;
                ps.last_play_time_start = std::chrono::steady_clock::now();
                ps.accumulated_play_time = ps.to_display_frame_number / dc->video_fps;
            } else {
                ps.pause_selected = 0;
            }
        }
        ImGui::PopStyleColor(3);
    }

    ImGui::SameLine(0.0f, spacing);
    if (ImGui::Button(ICON_FK_STEP_FORWARD)) {
        int f = std::min(dc->total_num_frame,
                         current_frame_num + dc->seek_interval);
        seek_all_cameras(ctx.scene, f, dc->video_fps, ps, false);
    }
    ImGui::SameLine(0.0f, spacing);
    if (ImGui::Button(ICON_FK_FAST_FORWARD)) {
        int f = std::min(dc->total_num_frame,
                         current_frame_num + 10 * dc->seek_interval);
        seek_all_cameras(ctx.scene, f, dc->video_fps, ps, false);
    }

    ImGui::SameLine(0, spacing);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0, spacing);

    // === Timeline slider / frame input ===
    // Uses ImGui's built-in Ctrl+click behavior on SliderInt: Ctrl+click opens
    // a TempInput text field on the slider itself. Enter commits, Escape
    // cancels. While the user is typing, SliderInt returns changed=true on
    // every keystroke (ImGui writes the parsed value to the bound variable
    // as they type). Seeking on every keystroke caused 16 decoders to spawn
    // overlapping seeks for each digit typed, exhausting NVDEC surface
    // pools and eventually corrupting display_buffer reads from the main
    // thread. We now only seek when:
    //   - slider is dragged (changed && !is_temp_input), OR
    //   - text input is just committed (was_temp_input && !is_temp_input).
    state.edit_buf = ps.slider_frame_number;
    ImGui::SetNextItemWidth(200.0f);
    ImGuiID slider_id = ImGui::GetID("##timeline");
    bool was_temp_input = ImGui::TempInputIsActive(slider_id);
    bool changed = ImGui::SliderInt(
        "##timeline", &state.edit_buf, 0, dc->estimated_num_frames);
    bool is_temp_input = ImGui::TempInputIsActive(slider_id);

    // Pause playback as soon as the user Ctrl+clicks into text edit mode.
    if (is_temp_input && !was_temp_input && ps.play_video) {
        ps.play_video = false;
        ps.pause_selected = 0;
    }

    bool just_committed_text = was_temp_input && !is_temp_input;
    bool is_drag_change = changed && !is_temp_input && !was_temp_input;
    // Seek ONLY on slider drag or text commit. Never mid-typing.
    if (is_drag_change || just_committed_text) {
        state.edit_buf = std::clamp(state.edit_buf, 0, dc->estimated_num_frames);
        ps.slider_frame_number = state.edit_buf;
        ps.slider_just_changed = true;
        // Accurate seek for typed-in frames (user wants the exact frame);
        // keyframe-granular for drag (interactive scrub).
        seek_all_cameras(ctx.scene, state.edit_buf, dc->video_fps, ps,
                         just_committed_text);
    }
    ps.slider_text_editing = is_temp_input;

    ImGui::SameLine(0, spacing);
    float current_time_sec = state.edit_buf / dc->video_fps;
    float total_time_sec = dc->estimated_num_frames / dc->video_fps;
    std::string current_str = format_time(current_time_sec);
    std::string total_str = format_time(total_time_sec);
    ImGui::Text("%s / %s", current_str.c_str(), total_str.c_str());

    // === Playback + Display controls ===
    ImGui::SameLine(0, spacing * 3);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0, spacing * 3);

    ImGui::TextColored(label_col, "RT");
    ImGui::SameLine(0, spacing);
    ImGui::Checkbox("##realtime", &ps.realtime_playback);

    ImGui::SameLine(0, spacing);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0, spacing);

    ImGui::BeginDisabled(!ps.realtime_playback);
    ImGui::TextColored(label_col, "Set Play Speed");
    ImGui::SameLine(0, spacing);
    char speed_label[16];
    int denom = (int)roundf(1.0f / ps.set_playback_speed);
    if (denom <= 1)
        snprintf(speed_label, sizeof(speed_label), "1x");
    else
        snprintf(speed_label, sizeof(speed_label), "1/%dx", denom);
    ImGui::SetNextItemWidth(80.0f);
    ImGui::SliderFloat("##speed", &ps.set_playback_speed,
                        1.0f / 16.0f, 1.0f, speed_label,
                        ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();

    ImGui::SameLine(0, spacing * 2);
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine(0, spacing * 2);

    // Contrast + Brightness + Reset
#ifdef __APPLE__
    (void)0;
#else
    ImGui::BeginDisabled(ps.play_video);
#endif

    ImGui::TextUnformatted(ICON_FK_ADJUST);
    ImGui::SameLine(0, spacing);
    ImGui::SetNextItemWidth(80.0f);
    ImGui::SliderFloat("##contrast", &display.contrast, 0.0f, 3.0f, "%.2f");

    ImGui::SameLine(0, spacing);
    ImGui::TextUnformatted(ICON_FK_SUN_O);
    ImGui::SameLine(0, spacing);
    ImGui::SetNextItemWidth(80.0f);
    ImGui::SliderInt("##brightness", &display.brightness, -150, 150);

    ImGui::SameLine(0, spacing);
    if (ImGui::Button("Reset##display")) {
        display.contrast = 1.0f;
        display.brightness = 0;
        display.pivot_midgray = true;
    }

#ifndef __APPLE__
    ImGui::EndDisabled();
#endif

    // === Right-aligned status readouts ===
    // Pre-format value strings to compute total width for right-alignment
    char val_fr[16], val_spd[16], val_rr[16];
    snprintf(val_fr,  sizeof(val_fr),  "%.0f fps", dc->video_fps);
    snprintf(val_spd, sizeof(val_spd), "%.2fx",    ps.inst_speed);
    snprintf(val_rr,  sizeof(val_rr),  "%.0f fps", ImGui::GetIO().Framerate);
    const char *lbl_fr = "Recorded FR", *lbl_spd = "Play Speed", *lbl_rr = "Render Rate";
    float gap = spacing * 3;
    float total_w = ImGui::CalcTextSize(lbl_fr).x + spacing + ImGui::CalcTextSize(val_fr).x + gap
                  + ImGui::CalcTextSize(lbl_spd).x + spacing + ImGui::CalcTextSize(val_spd).x + gap
                  + ImGui::CalcTextSize(lbl_rr).x + spacing + ImGui::CalcTextSize(val_rr).x;
    ImGui::SameLine(ImGui::GetWindowWidth() - total_w - 12.0f);

    ImGui::TextColored(label_col, "%s", lbl_fr);
    ImGui::SameLine(0, spacing); ImGui::TextDisabled("%s", val_fr);
    ImGui::SameLine(0, gap);
    ImGui::TextColored(label_col, "%s", lbl_spd);
    ImGui::SameLine(0, spacing); ImGui::TextDisabled("%s", val_spd);
    ImGui::SameLine(0, gap);
    ImGui::TextColored(label_col, "%s", lbl_rr);
    ImGui::SameLine(0, spacing); ImGui::TextDisabled("%s", val_rr);

    ImGui::End();
}
