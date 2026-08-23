#pragma once
#include "imgui.h"
#include "implot.h"
#include "app_context.h"
#include "global.h"
#include "gui/panel.h"
#include "keypoint_colors.h"
#include "gui/label_palette.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>

struct SettingsState {
    bool show = false;
};

inline void DrawSettingsWindow(SettingsState &state, AppContext &ctx) {
    auto &s = ctx.user_settings;

    DrawPanel("Settings", state.show,
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

        // --- Keypoint Colors ---
        if (ImGui::CollapsingHeader("Keypoint Colors")) {
            // Colormap for all keypoints. "Rainbow (HSV)" is the legacy
            // default; the rest are ImPlot's built-in matplotlib/MATLAB maps
            // (Viridis, Plasma, Jet, Spectral, ...). Selecting one recolors
            // the live skeleton immediately; Save persists the choice.
            const char *preview = (s.keypoint_colormap < 0)
                ? "Rainbow (HSV)"
                : ImPlot::GetColormapName(s.keypoint_colormap);
            if (ImGui::BeginCombo("Colormap", preview)) {
                if (ImGui::Selectable("Rainbow (HSV)",
                                      s.keypoint_colormap < 0)) {
                    s.keypoint_colormap = KEYPOINT_COLORMAP_RAINBOW;
                    g_keypoint_colormap = s.keypoint_colormap;
                    apply_keypoint_colormap(ctx.skeleton, g_keypoint_colormap);
                    other_changed = true;
                }
                for (int i = 0; i < ImPlot::GetColormapCount(); i++) {
                    if (ImGui::Selectable(ImPlot::GetColormapName(i),
                                          s.keypoint_colormap == i)) {
                        s.keypoint_colormap = i;
                        g_keypoint_colormap = i;
                        apply_keypoint_colormap(ctx.skeleton,
                                                g_keypoint_colormap);
                        other_changed = true;
                    }
                }
                ImGui::EndCombo();
            }
            // Visual preview bar of the selected colormap.
            if (s.keypoint_colormap >= 0) {
                ImPlot::ColormapButton(
                    ImPlot::GetColormapName(s.keypoint_colormap),
                    ImVec2(-1, 0), s.keypoint_colormap);
            }

            // Active (selected) keypoint highlight color.
            if (s.active_keypoint_color.size() < 3)
                s.active_keypoint_color.resize(3, 1.0f);
            if (ImGui::ColorEdit3("Active Keypoint",
                                  s.active_keypoint_color.data()))
                other_changed = true;
            ImGui::TextDisabled(
                "Applies to all camera views and the Keypoints table.");
        }

        // --- Label Colors ---
        if (ImGui::CollapsingHeader("Label Colors")) {
            ImGui::TextDisabled(
                "Frame-state colors in the Labeling Tool grid & timeline and "
                "the Frame Buffer window.");
            LabelPalette &lp = label_palette();
            for (const auto &role : label_palette_roles()) {
                ImVec4 &c = lp.*role.color;
                if (ImGui::ColorEdit3(role.label, (float *)&c)) {
                    c.w = 1.0f;
                    s.label_colors[role.key] = {c.x, c.y, c.z};
                    other_changed = true;
                }
            }
            if (ImGui::SmallButton("Reset Label Colors")) {
                lp = LabelPalette{};
                s.label_colors.clear();
                other_changed = true;
            }
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
            ImGui::InputInt("Buffer Size", &s.default_buffer_size);
            // No propagation needed — takes effect on next video load
        }

#ifndef __APPLE__
        // --- Hardware (Linux only) ---
        if (ImGui::CollapsingHeader("Hardware")) {
            // Buffer Type drives how display_buffer[].frame is allocated
            // (cudaMalloc vs malloc). That allocation only happens at
            // startup, so changing the value at runtime would just lie to
            // every later code path and crash on the next render. We
            // persist the new choice to user_settings and tell the user
            // to restart; the live ctx.scene->use_cpu_buffer is left alone.
            const char *buf_items[] = {"CPU Buffer", "GPU Buffer"};
            // Combo reflects the PERSISTED setting (what'll apply on next
            // launch), not the live runtime mode — they may differ after a
            // pending change.
            int buf_current = s.use_cpu_buffer ? 0 : 1;
            if (ImGui::Combo("Buffer Type", &buf_current, buf_items, IM_ARRAYSIZE(buf_items))) {
                s.use_cpu_buffer = (buf_current == 0);
                other_changed = true;  // forces save_user_settings below
                if (s.use_cpu_buffer != ctx.scene->use_cpu_buffer) {
                    ctx.popups.pushInfo("Restart Required",
                        "Buffer Type changes take effect after restarting red.\n"
                        "Your choice has been saved; close and reopen red to apply.");
                }
            }
            if (s.use_cpu_buffer != ctx.scene->use_cpu_buffer) {
                ImGui::TextDisabled(
                    "(pending — restart red to apply; currently running in %s)",
                    ctx.scene->use_cpu_buffer ? "CPU Buffer" : "GPU Buffer");
            }
        }
#endif

        // --- Annotation Tools ---
        if (ImGui::CollapsingHeader("Annotation Tools")) {
            auto &ac = ctx.pm.annotation_config;
            ImGui::Checkbox("Keypoints", &ac.enable_keypoints);
            ImGui::Checkbox("Bounding Boxes", &ac.enable_bboxes);
            ImGui::Checkbox("Oriented Bounding Boxes", &ac.enable_obbs);
            ImGui::Checkbox("Segmentation (SAM)", &ac.enable_segmentation);
            ImGui::TextDisabled("Enable tools to show their panels in the Tools menu.");
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
            label_palette() = LabelPalette{};
            g_keypoint_colormap = s.keypoint_colormap;
            apply_keypoint_colormap(ctx.skeleton, g_keypoint_colormap);
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
