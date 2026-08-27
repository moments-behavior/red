#pragma once
#include "imgui.h"
#include "implot.h"
#include "app_context.h"
#include "decode_backend.h"
#include "global.h"
#include "gui/panel.h"
#include "keypoint_colors.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <cmath>

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
            // Applied live to style.FontScaleMain by the main loop; ImGui 1.92+
            // re-rasterises glyphs at the scaled size, so text stays sharp.
            if (ImGui::SliderFloat("UI Text Size", &s.ui_text_scale,
                                   0.7f, 2.0f, "%.2fx")) {
                ImGui::GetStyle().FontScaleMain = s.ui_text_scale;
                other_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##textscale")) {
                s.ui_text_scale = 1.0f;
                ImGui::GetStyle().FontScaleMain = 1.0f;
                other_changed = true;
            }
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

        // --- Playback ---
        if (ImGui::CollapsingHeader("Playback Defaults")) {
            // Same single control as the transport bar: one question ("how
            // fast?") rather than a mode checkbox plus a rate slider.
            struct DefSpeed { const char *label; float speed; bool clock_paced; };
            static const DefSpeed kDefSpeeds[] = {
                {"1x (real time)", 1.0f,        true},
                {"1/2x",           1.0f / 2.0f,  true},
                {"1/4x",           1.0f / 4.0f,  true},
                {"1/8x",           1.0f / 8.0f,  true},
                {"1/16x",          1.0f / 16.0f, true},
                {"Every frame",    1.0f,         false},
            };
            constexpr int kNumDefSpeeds =
                (int)(sizeof(kDefSpeeds) / sizeof(kDefSpeeds[0]));
            int def_idx = kNumDefSpeeds - 1;
            if (s.default_realtime_playback) {
                float best = 1e9f;
                for (int i = 0; i < kNumDefSpeeds - 1; ++i) {
                    float d = fabsf(kDefSpeeds[i].speed - s.default_playback_speed);
                    if (d < best) { best = d; def_idx = i; }
                }
            }
            if (ImGui::BeginCombo("Playback Speed", kDefSpeeds[def_idx].label)) {
                for (int i = 0; i < kNumDefSpeeds; ++i) {
                    if (ImGui::Selectable(kDefSpeeds[i].label, i == def_idx)) {
                        s.default_realtime_playback = kDefSpeeds[i].clock_paced;
                        if (kDefSpeeds[i].clock_paced)
                            s.default_playback_speed = kDefSpeeds[i].speed;
                        playback_changed = true;
                    }
                    if (i == def_idx) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "1x and the fractions play to the wall clock (accurate "
                    "timing, skips frames if decoding lags).\n"
                    "\"Every frame\" shows every decoded frame instead.");
            ImGui::InputInt("Buffer Size", &s.default_buffer_size);
            // No propagation needed — takes effect on next video load
        }

#ifndef __APPLE__
        // --- Hardware (Linux only) ---
        if (ImGui::CollapsingHeader("Hardware")) {
            ImGui::Text("Decode backend: %s", red::decode_backend_name());
            ImGui::TextDisabled("(%s)", red::decode_backend_reason());
            // Software decode writes host memory, so render_allocate_scene_memory
            // forces CPU Buffer regardless of what is persisted. Say so and
            // disable the control rather than leaving a "pending restart" note
            // that a restart would never clear.
            const bool sw_backend = red::decode_backend_is_software();
            if (sw_backend)
                ImGui::TextDisabled(
                    "Software decode delivers frames in host memory;\n"
                    "Buffer Type is fixed to CPU Buffer.");
            ImGui::BeginDisabled(sw_backend);
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
            ImGui::EndDisabled();
            if (!sw_backend && s.use_cpu_buffer != ctx.scene->use_cpu_buffer) {
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
