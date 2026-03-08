#pragma once
#include "imgui.h"
#include "app_context.h"
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>

struct SettingsState {
    bool show = false;
};

inline void DrawSettingsWindow(SettingsState &state, AppContext &ctx) {
    auto &s = ctx.user_settings;

    drawPanel("Settings", state.show,
        [&]() {
        bool display_changed = false;
        bool playback_changed = false;
        bool other_changed = false;

        // --- Paths ---
        if (ImGui::CollapsingHeader("Paths", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Default Project Root");
            if (ImGui::InputText("##proj_root", &s.default_project_root_path))
                other_changed = true;
            ImGui::SameLine();
            if (ImGui::Button("Browse##proj_root")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = s.default_project_root_path.empty()
                               ? ctx.red_data_dir
                               : s.default_project_root_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "SettingsBrowseProjRoot", "Choose Project Root", nullptr, cfg);
            }

            ImGui::Text("Default Media Root");
            if (ImGui::InputText("##media_root", &s.default_media_root_path))
                other_changed = true;
            ImGui::SameLine();
            if (ImGui::Button("Browse##media_root")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = s.default_media_root_path.empty()
                               ? ctx.red_data_dir
                               : s.default_media_root_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "SettingsBrowseMediaRoot", "Choose Media Root", nullptr, cfg);
            }
        }

        // --- Display ---
        if (ImGui::CollapsingHeader("Display Defaults")) {
            if (ImGui::SliderInt("Brightness", &s.default_brightness, -150, 150))
                display_changed = true;
            if (ImGui::SliderFloat("Contrast", &s.default_contrast, 0.0f, 3.0f, "%.2f"))
                display_changed = true;
            if (ImGui::Checkbox("Pivot Mid-Gray", &s.default_pivot_midgray))
                display_changed = true;
        }

        // --- Playback ---
        if (ImGui::CollapsingHeader("Playback Defaults")) {
            char speed_label[16];
            int denom = (int)roundf(1.0f / s.default_playback_speed);
            if (denom <= 1)
                snprintf(speed_label, sizeof(speed_label), "1x");
            else
                snprintf(speed_label, sizeof(speed_label), "1/%dx", denom);
            if (ImGui::SliderFloat("Playback Speed", &s.default_playback_speed,
                                   1.0f / 16.0f, 1.0f, speed_label,
                                   ImGuiSliderFlags_Logarithmic))
                playback_changed = true;
            if (ImGui::Checkbox("Realtime Playback", &s.default_realtime_playback))
                playback_changed = true;
            if (ImGui::InputInt("Buffer Size", &s.default_buffer_size))
                playback_changed = true;
            // Seek interval is auto-detected per-video from keyframe spacing
            // (shown as "Seek Step" in Navigator for runtime adjustment).
        }

        // --- Export ---
        if (ImGui::CollapsingHeader("JARVIS Export Defaults")) {
            if (ImGui::SliderFloat("Bbox Margin (px)", &s.jarvis_margin, 0.0f, 200.0f))
                other_changed = true;
            if (ImGui::SliderFloat("Train Ratio", &s.jarvis_train_ratio, 0.5f, 0.99f))
                other_changed = true;
            if (ImGui::InputInt("Random Seed", &s.jarvis_seed))
                other_changed = true;
            if (ImGui::SliderInt("JPEG Quality", &s.jarvis_jpeg_quality, 10, 100))
                other_changed = true;
        }

        ImGui::Separator();

        if (ImGui::Button("Save")) {
            save_user_settings(s);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset to Defaults")) {
            UserSettings defaults;
            defaults.default_project_root_path = s.default_project_root_path;
            defaults.default_media_root_path = s.default_media_root_path;
            s = defaults;
            display_changed = playback_changed = other_changed = true;
        }
        // Propagate only the sections that actually changed (no auto-save;
        // user presses "Save" explicitly to persist to disk)
        if (display_changed) {
            ctx.display.brightness = s.default_brightness;
            ctx.display.contrast = s.default_contrast;
            ctx.display.pivot_midgray = s.default_pivot_midgray;
        }
        if (playback_changed) {
            ctx.ps.set_playback_speed = s.default_playback_speed;
            ctx.ps.realtime_playback = s.default_realtime_playback;
            // NOTE: seek_interval is NOT propagated here — it's auto-detected
            // per-video from the actual keyframe interval (media_loader.h).
        }
        },
        [&]() {
        // File dialog handlers
        if (ImGuiFileDialog::Instance()->Display("SettingsBrowseProjRoot",
                ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                s.default_project_root_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                save_user_settings(s);
            }
            ImGuiFileDialog::Instance()->Close();
        }
        if (ImGuiFileDialog::Instance()->Display("SettingsBrowseMediaRoot",
                ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                s.default_media_root_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                save_user_settings(s);
            }
            ImGuiFileDialog::Instance()->Close();
        }
        },
        ImVec2(500, 500));
}
