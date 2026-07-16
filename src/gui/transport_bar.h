#pragma once
#include "app_context.h"
#include "IconsForkAwesome.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <climits>

struct TransportBarState {
    // Cmd+click on slider: replace slider with InputInt, pause playback.
    bool slider_text_editing = false;
    bool focus_input = false;  // one-shot: give InputInt keyboard focus
    int edit_buf = 0;

    // Desync-fix gap ticks under the timeline slider: which of the N bins
    // contain a dropped-frame gap on any camera. Rebuilt when the plan or the
    // Sync-fix state changes (positions differ between the two coordinate
    // systems).
    std::string tick_key;
    std::vector<uint8_t> tick_bins;
};

// Toggle the canonical-timeline desync fix. Pauses, switches the timeline
// between canonical slots and mp4 indices (translating the current position
// through the reference camera's mapping), then re-baselines every decoder
// with an accurate seek — the seek is the epoch boundary at which decoders
// re-sample the mode.
inline void sync_fix_toggle(AppContext &ctx, bool enable) {
    auto *dc = ctx.dc_context;
    const sync_plan::SyncPlan &plan = g_sync_fix.plan;
    if (!plan.usable() || ctx.pm.camera_names.empty() || ctx.demuxers.empty())
        return;
    const sync_plan::SyncCam *cam0 = plan.cam(ctx.pm.camera_names[0]);
    if (!cam0) return;

    ctx.ps.play_video = false;
    ctx.ps.pause_selected = 0;

    int64_t cur = ctx.current_frame_num;
    int64_t target;
    if (enable) {
        int64_t p = std::clamp<int64_t>(cur, 0, cam0->decoded_len - 1);
        target = std::clamp<int64_t>(cam0->slot_of_pos(p), 0,
                                     plan.canonical_len - 1);
        dc->sync_canonical_len = plan.canonical_len;
        dc->total_num_frame = (int)plan.canonical_len;
        dc->estimated_num_frames = (int)plan.canonical_len - 1;
        // Canonical slots are uniform in trigger time — pace playback by the
        // trigger interval.
        dc->video_fps = 1e9 / (double)plan.delta_ns;
        dc->sync_fix_active = true;
    } else {
        int64_t slot = std::clamp<int64_t>(cur, 0, plan.canonical_len - 1);
        target = cam0->seek_pos(slot);
        dc->sync_fix_active = false;
        // Restore native counts/fps from the reference demuxer; decoders
        // repopulate total_num_frame when they hit EOF (as at initial load).
        dc->video_fps = ctx.demuxers[0]->GetFramerate();
        if (ctx.demuxers[0]->GetNumFrames() == 0)
            dc->estimated_num_frames =
                (int)(ctx.demuxers[0]->GetDuration() * dc->video_fps);
        else
            dc->estimated_num_frames = (int)ctx.demuxers[0]->GetNumFrames() - 1;
        dc->total_num_frame = INT_MAX;
    }

    seek_all_cameras(ctx.scene, (int)target, dc->video_fps, ctx.ps, true);
    ctx.current_frame_num = (int)target;
    for (auto &kv : window_need_decoding) kv.second.store(true);

    ctx.pm.sync_fix_enabled = enable;
    if (!ctx.pm.project_path.empty() && !ctx.pm.project_name.empty()) {
        std::string redproj =
            ctx.pm.project_path + "/" + ctx.pm.project_name + ".redproj";
        save_project_manager_json(ctx.pm, redproj);
    }
}

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
                ps.accumulated_play_time =
                    ps.to_display_frame_number / dc->video_fps;
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
    // Cmd+click replaces the slider with an InputInt so the user can type
    // a frame number at their leisure.  Enter = seek, Escape = cancel.
    if (state.slider_text_editing) {
        // --- Text input mode: InputInt replaces slider ---
        ImGui::SetNextItemWidth(200.0f);
        if (state.focus_input) {
            ImGui::SetKeyboardFocusHere();
            state.focus_input = false;
        }
        ImGui::InputInt("##timeline_input", &state.edit_buf, 0, 0);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            // Enter — clamp and seek to the EXACT frame the user typed.
            // seek_accurate = true forward-decodes from the keyframe to the
            // requested frame (same precise path the Labeling Tool uses),
            // instead of snapping to the previous keyframe like scrubbing.
            state.edit_buf = std::clamp(state.edit_buf, 0, dc->estimated_num_frames);
            seek_all_cameras(ctx.scene, state.edit_buf, dc->video_fps, ps, true);
            state.slider_text_editing = false;
            ps.slider_text_editing = false;
        } else if (ImGui::IsItemDeactivated()) {
            // Escape or click-away — cancel
            state.slider_text_editing = false;
            ps.slider_text_editing = false;
        }
    } else {
        // --- Normal mode: slider ---
        state.edit_buf = ps.slider_frame_number;
        ImGui::SetNextItemWidth(200.0f);
        bool changed = ImGui::SliderInt(
            "##timeline", &state.edit_buf, 0, dc->estimated_num_frames);

        if (ImGui::TempInputIsActive(ImGui::GetItemID())) {
            // Cmd+click detected — pause and switch to InputInt next frame
            if (ps.play_video) {
                ps.play_video = false;
                ps.pause_selected = 0;
            }
            state.slider_text_editing = true;
            ps.slider_text_editing = true;
            state.focus_input = true;
        } else if (changed) {
            ps.slider_frame_number = state.edit_buf;
            ps.slider_just_changed = true;
            seek_all_cameras(ctx.scene, state.edit_buf, dc->video_fps, ps, false);
        }

        // Red gap ticks under the slider: where any camera dropped frames.
        // Positions are canonical slots when the Sync fix is ON; when OFF they
        // are mapped through the reference camera's mp4 index (a gap has zero
        // width there — the lost frames simply don't exist in that timeline).
        if (g_sync_fix.plan.usable() &&
            g_sync_fix.plan.status == sync_plan::Status::Reindex &&
            !ctx.input_is_imgs && !ctx.pm.camera_names.empty()) {
            const sync_plan::SyncPlan &plan = g_sync_fix.plan;
            const bool fix_on = dc->sync_fix_active.load();
            constexpr int kTickBins = 200;
            std::string key = plan.source + (fix_on ? "#on" : "#off");
            if (state.tick_key != key) {
                state.tick_key = key;
                state.tick_bins.assign(kTickBins, 0);
                const sync_plan::SyncCam *cam0 =
                    plan.cam(ctx.pm.camera_names[0]);
                const int64_t range =
                    std::max<int64_t>(1, dc->estimated_num_frames + 1);
                for (const auto &kv : plan.cams) {
                    for (const auto &g : kv.second.gaps) {
                        int64_t pos = fix_on || !cam0 ? g.slot
                                                      : cam0->seek_pos(g.slot);
                        int64_t span = fix_on ? g.lost : 1;
                        int b0 = (int)std::clamp<int64_t>(
                            pos * kTickBins / range, 0, kTickBins - 1);
                        int b1 = (int)std::clamp<int64_t>(
                            (pos + span - 1) * kTickBins / range, 0,
                            kTickBins - 1);
                        for (int b = b0; b <= b1; ++b) state.tick_bins[b] = 1;
                    }
                }
            }
            ImVec2 r0 = ImGui::GetItemRectMin();
            ImVec2 r1 = ImGui::GetItemRectMax();
            ImDrawList *tdl = ImGui::GetWindowDrawList();
            for (int b = 0; b < kTickBins; ++b) {
                if (!state.tick_bins[b]) continue;
                float x = r0.x + (r1.x - r0.x) * (b + 0.5f) / kTickBins;
                tdl->AddLine(ImVec2(x, r1.y - 3.0f), ImVec2(x, r1.y - 1.0f),
                             IM_COL32(220, 60, 60, 200), 1.0f);
            }
        }
    }

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

    // === Desync fix (canonical trigger timeline) ===
    // Only shown when a usable sync plan exists for the loaded videos.
    if (g_sync_fix.plan.usable() && !ctx.input_is_imgs) {
        const sync_plan::SyncPlan &plan = g_sync_fix.plan;
        bool clean = plan.status == sync_plan::Status::Clean;

        ImGui::SameLine(0, spacing);
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine(0, spacing);

        ImGui::TextColored(label_col, "Sync");
        ImGui::SameLine(0, spacing);
        bool sync_active = dc->sync_fix_active.load();
        ImGui::BeginDisabled(clean);
        if (ImGui::Checkbox("##syncfix", &sync_active))
            sync_fix_toggle(ctx, sync_active);
        ImGui::EndDisabled();

        ImGui::SameLine(0, spacing);
        ImVec4 status_col = clean ? green_col
                            : plan.status == sync_plan::Status::Trim
                                ? ImVec4(0.9f, 0.9f, 0.4f, 1.0f)
                                : ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
        ImGui::TextColored(status_col, "%s", sync_plan::status_name(plan.status));
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            if (clean) {
                ImGui::TextUnformatted(
                    "Cameras are already aligned — nothing to fix.");
            } else {
                ImGui::Text("Plan source: %s", plan.source.c_str());
                ImGui::Text("Canonical timeline: %lld trigger slots",
                            (long long)plan.canonical_len);
                if (plan.first_drop_slot >= 0)
                    ImGui::Text("First drop at slot %lld",
                                (long long)plan.first_drop_slot);
                ImGui::Separator();
                if (ImGui::BeginTable("##syncplan", 5)) {
                    ImGui::TableSetupColumn("camera");
                    ImGui::TableSetupColumn("start");
                    ImGui::TableSetupColumn("decoded");
                    ImGui::TableSetupColumn("drops");
                    ImGui::TableSetupColumn("lost");
                    ImGui::TableHeadersRow();
                    for (const auto &kv : plan.cams) {
                        int64_t lost = 0;
                        for (const auto &g : kv.second.gaps) lost += g.lost;
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(kv.first.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%lld", (long long)kv.second.start_slot);
                        ImGui::TableNextColumn();
                        ImGui::Text("%lld", (long long)kv.second.decoded_len);
                        ImGui::TableNextColumn();
                        ImGui::Text("%zu", kv.second.gaps.size());
                        ImGui::TableNextColumn();
                        ImGui::Text("%lld", (long long)lost);
                    }
                    ImGui::EndTable();
                }
                ImGui::Separator();
                ImGui::TextUnformatted(
                    sync_active
                        ? "Timeline is canonical trigger slots; dropped "
                          "frames show the held previous frame with a badge.\n"
                          "Note: seeks are frame-accurate in this mode "
                          "(step/fast buttons are slower)."
                        : "Cameras are index-paired and desync after drops. "
                          "Enable to view on the canonical trigger timeline.");
            }
            ImGui::EndTooltip();
        }
    }

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
