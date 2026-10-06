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
        bool other_changed = false;

        // --- Paths ---
        ImGui::SeparatorText("Paths");
        {
            // Set: dialogs always start there. Empty: they start in the folder
            // last used (default_project_root, media_browse_dir).
            ImGui::Text("Start project dialogs in");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                ImGui::SetTooltip("Open Project and Save Project always start here.\n"
                                  "Leave empty to start in the folder last used.");
            if (ImGui::InputTextWithHint("##proj_root", "the folder last used", &s.default_project_root_path))
                other_changed = true;
            ImGui::SameLine();
            if (ImGui::Button("Browse##proj_root")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = s.default_project_root_path.empty()
                               ? ctx.default_dir
                               : s.default_project_root_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "SettingsBrowseProjRoot", "Choose Project Root", nullptr, cfg);
            }

            ImGui::Text("Start media dialogs in");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                ImGui::SetTooltip("Open Videos, Open Images and New Project's Media "
                                  "Folder always start here.\nLeave empty to start "
                                  "in the folder last used.");
            if (ImGui::InputTextWithHint("##media_root", "the folder last used", &s.default_media_root_path))
                other_changed = true;
            ImGui::SameLine();
            if (ImGui::Button("Browse##media_root")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = s.default_media_root_path.empty()
                               ? ctx.default_dir
                               : s.default_media_root_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "SettingsBrowseMediaRoot", "Choose Media Root", nullptr, cfg);
            }
        }

        // --- Display ---
        ImGui::SeparatorText("Display");
        {
            // Applied live to style.FontScaleMain by the main loop; ImGui 1.92+
            // re-rasterises glyphs at the scaled size, so text stays sharp.
            if (ImGui::SliderFloat("UI Text Size", &s.ui_text_scale,
                                   kUiTextScaleMin, kUiTextScaleMax, "%.2fx")) {
                ImGui::GetStyle().FontScaleMain = s.ui_text_scale;
                other_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##textscale")) {
                s.ui_text_scale = 1.0f;
                ImGui::GetStyle().FontScaleMain = 1.0f;
                other_changed = true;
            }
            // Brightness / contrast are adjusted live in the transport bar,
            // starting neutral each session. How contrast behaves is a
            // preference, kept here: stretch around mid-gray (darks darker,
            // lights lighter) or scale from black.
            if (ImGui::Checkbox("Contrast pivots on mid-gray", &s.default_pivot_midgray)) {
                ctx.display.pivot_midgray = s.default_pivot_midgray;
                other_changed = true;
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                ImGui::SetTooltip("On: contrast stretches values away from mid-gray, "
                                  "keeping overall brightness.\nOff: it scales from "
                                  "black, so more contrast also brightens.");
        }

        // --- Keypoint Colors ---
        ImGui::SeparatorText("Keypoint Colors");
        {
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
        ImGui::SeparatorText("Playback");
        {
            // Speed is the transport bar's; every session starts at 1x.
            ImGui::InputInt("Buffer Size", &s.default_buffer_size);
            // No propagation needed — takes effect on next video load
        }

#ifndef __APPLE__
        // --- Hardware (Linux only) ---
        ImGui::SeparatorText("Hardware");
        {
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
            other_changed = true;
        }
        // Propagate only the sections that actually changed (no auto-save;
        // user presses "Save" explicitly to persist to disk)
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
        ImVec2(520, 720));   // every section shows, so room for them all
}
