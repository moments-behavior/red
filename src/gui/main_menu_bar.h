#pragma once
#include "app_context.h"
#include "video_files.h"
#include "gui/window_states.h"
#include "IconsForkAwesome.h"
#include "tailcycle_import.h"
#include "gui/tailcycle_open_window.h"
#include "gui/folder_dialog.h"
#include <ImGuiFileDialog.h>
#include <filesystem>

inline void DrawMainMenuBar(AppContext &ctx, WindowStates &win) {
    auto &annot_state      = win.annotation;
    auto &settings_state   = win.settings;
    auto &jarvis_export_state = win.jarvis_export;
    auto &export_state     = win.export_win;
    auto &bbox_state       = win.bbox;
    auto &obb_state        = win.obb;
    auto &triangulation_diag_state = win.triangulation_diag;
    auto &show_help_window = win.show_help;
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto &user_settings = ctx.user_settings;

    if (!ImGui::BeginMainMenuBar())
        return;

    // --- Text menus ---
    // File: getting things in and out. Project: this project's settings.
    // Label: the labelling tools. View: windows to look at. Then Tailcycle and
    // Help. 3D / calibration-dependent entries are disabled for 2D
    // (uncalibrated) projects -- they index camera calibration.
    const bool is_2d = project_is_2d(pm);

    if (ImGui::BeginMenu("File")) {
        // "New / Open Project" for the Welcome screen's Create / Load
        // Annotation Project: the same form and the same dialog.
        if (ImGui::MenuItem("New Project..."))
            annot_state.open();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Create an annotation project: one or more "
                              "cameras; with several, choose\ncalibrated (3D "
                              "triangulation) or not (2D labels only) in the form.");
        if (ImGui::MenuItem("Open Project...")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = pm.project_root_path;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseProject", "Choose Project File", ".redproj",
                config);
        }
        if (ImGui::BeginMenu("Recent Projects",
                             !user_settings.recent_projects.empty())) {
            for (const auto &path : user_settings.recent_projects) {
                std::filesystem::path p(path);
                const std::string shown =
                    p.parent_path().filename().string() + "/" + p.filename().string();
                std::error_code ec;
                if (ImGui::MenuItem(shown.c_str(), nullptr, false,
                                    std::filesystem::exists(path, ec)))
                    win.load_project_request = path;   // the main loop loads it
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                    ImGui::SetTooltip("%s", path.c_str());
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Open Videos...")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 0;
            config.path = media_browse_dir(ctx);
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseMedia", "Choose Media", video_ext_filter(), config);
        }
        if (ImGui::MenuItem("Open Images...")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 0;
            config.path = media_browse_dir(ctx);
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseImages", "Choose Images", image_ext_filter(), config);
        }
        ImGui::Separator();
        // Same action as the toolbar floppy icon and the Labeling Tool's Save
        // button: ctx.save_requested is forwarded to the labeling tool.
        ImGui::BeginDisabled(!pm.plot_keypoints_flag);
        if (ImGui::MenuItem("Save", RED_MOD_KEY "+S"))
            ctx.save_requested = true;
        ImGui::EndDisabled();
        ImGui::Separator();
        if (ImGui::BeginMenu("Export")) {
            if (ImGui::MenuItem("Export Tool..."))
                export_state.show = true;
            ImGui::BeginDisabled(is_2d);
            if (ImGui::MenuItem("JARVIS..."))
                jarvis_export_state.show = true;
            ImGui::EndDisabled();
            // A standalone multi-dataset merge: works with no project open.
            if (ImGui::MenuItem("Group JARVIS..."))
                win.group_export.show = true;
            ImGui::EndMenu();
        }
        ImGui::BeginDisabled(is_2d);
        if (ImGui::MenuItem("Import JARVIS Predictions..."))
            win.jarvis_import.show = true;
        ImGui::EndDisabled();
        ImGui::Separator();
        if (ImGui::MenuItem("Settings..."))
            settings_state.show = true;
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Project")) {
        ImGui::BeginDisabled(pm.project_path.empty());
        if (ImGui::MenuItem("Switch Skeleton...")) {
            win.switch_skeleton.show = true;
            win.switch_skeleton.initialized = false;
        }
        ImGui::EndDisabled();
        // Where this project's per-camera timestamps are, for the desync fix
        // and Frame Drops. A project setting: red does not look for them.
        const bool can_set_timestamps = !pm.project_path.empty() &&
                                        ps.video_loaded && !ctx.input_is_imgs;
        ImGui::BeginDisabled(!can_set_timestamps);
        if (ImGui::MenuItem("Camera Timestamps...")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = !pm.timestamps_folder.empty() ? pm.timestamps_folder
                                                        : pm.media_folder;
            config.flags = ImGuiFileDialogFlags_Modal;
            open_folder_dialog("ChooseProjectTimestamps", "Select Camera Timestamps Folder",
                               timestamps_folder_kind(), config);
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort |
                                 ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (!can_set_timestamps)
                ImGui::SetTooltip(pm.untitled
                                      ? "Save the project first (" RED_MOD_KEY "+S)."
                                      : "Open a video project first.");
            else if (pm.timestamps_folder.empty())
                ImGui::SetTooltip("Choose the folder with the cameras' frame "
                                  "timestamps, for the desync fix\nand Frame "
                                  "Drops. None is set for this project.");
            else
                ImGui::SetTooltip("Timestamps from: %s",
                                  pm.timestamps_folder.c_str());
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Skeleton Creator..."))
            win.skeleton_creator.show = true;
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Draw a skeleton and save it as the .json a "
                              "project loads.");
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Label")) {
        if (ImGui::MenuItem("Bbox Tool"))
            bbox_state.show = true;
        if (ImGui::MenuItem("OBB Tool"))
            obb_state.show = true;
        ImGui::BeginDisabled(is_2d);
        if (ImGui::MenuItem("Midline Tool"))
            win.midline.show = true;
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Pose Stats"))
            win.pose_stats.show = true;
        if (ImGui::MenuItem("Frame Drops"))
            win.frame_drops.show = true;
        ImGui::BeginDisabled(is_2d);
        if (ImGui::MenuItem("Triangulation Diagnostics"))
            triangulation_diag_state.show = true;
        ImGui::EndDisabled();
        ImGui::Separator();
        if (ImGui::BeginMenu("Text Size")) {
            const float cur = user_settings.ui_text_scale;
            if (ImGui::MenuItem("Larger", keys::display(keys::Sc::TextLarger).c_str(),
                                false, cur < kUiTextScaleMax))
                set_ui_text_scale(ctx, cur + kUiTextScaleStep);
            if (ImGui::MenuItem("Smaller", keys::display(keys::Sc::TextSmaller).c_str(),
                                false, cur > kUiTextScaleMin))
                set_ui_text_scale(ctx, cur - kUiTextScaleStep);
            char reset[48];
            snprintf(reset, sizeof(reset), "Reset (now %d%%)", (int)std::lround(cur * 100));
            if (ImGui::MenuItem(reset, keys::display(keys::Sc::TextReset).c_str(),
                                false, cur != 1.0f))
                set_ui_text_scale(ctx, 1.0f);
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }

    // Its own menu rather than one entry in File and a format buried in the
    // Export Tool's combo, with tracktail, the tracker that works with it.
    // The dataset entries need Parquet and are left out without it; tracktail
    // does not.
    if (ImGui::BeginMenu("Tailcycle")) {
        if (TailcycleImport::available()) {
            if (ImGui::MenuItem("Open Dataset...")) {
                run_or_confirm_unsaved(ctx, [&win]() {
                    tailcycle_open_browse(win.tailcycle_open);
                });
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
                ImGui::SetTooltip("Open a tailcycle-dataset session to look at or "
                                  "correct.");
            ImGui::BeginDisabled(pm.project_path.empty() && !ps.video_loaded);
            if (ImGui::MenuItem("Export Dataset...")) {
                export_state.show = true;
                export_state.want_format = (int)ExportFormats::TAILCYCLE;
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort |
                                     ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Write this project out as a tailcycle-dataset. "
                                  "Opens the Export Tool with that format chosen.");
            ImGui::Separator();
        }
        // Needs calibration: it tracks the 3D keypoints.
        ImGui::BeginDisabled(project_is_2d(pm));
        if (ImGui::MenuItem("tracktail")) {
            win.tracktail.show = true;
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort |
                                 ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("Predict the next frames from the current "
                              "frame's 3D keypoints (forward temporal "
                              "tracker).");
        ImGui::EndMenu();
    }

    // Help: last, where both macOS and Windows put it.
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Red Help", keys::display(keys::Sc::ToggleHelp).c_str()))
            show_help_window = true;
        ImGui::Separator();
        if (ImGui::MenuItem("About Red"))
            win.show_about = true;
        if (ImGui::MenuItem("Report an Issue...")) {
            ImGuiPlatformIO &pio = ImGui::GetPlatformIO();
            if (pio.Platform_OpenInShellFn)
                pio.Platform_OpenInShellFn(ImGui::GetCurrentContext(),
                                           "https://github.com/moments-behavior/red/issues");
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Opens red's GitHub issues page. Include the "
                              "version from About Red.");
        ImGui::EndMenu();
    }

    // --- Toolbar icons ---
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);

    // New Project
    if (ImGui::MenuItem(ICON_FK_FILE_O "##toolbar_new")) {
        annot_state.open();
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        ImGui::SetTooltip("New Project");

    // Open Project
    if (ImGui::MenuItem(ICON_FK_FOLDER_OPEN "##toolbar_open")) {
        IGFD::FileDialogConfig config;
        config.countSelectionMax = 1;
        config.path = pm.project_root_path;
        config.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseProject", "Choose Project File", ".redproj",
            config);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        ImGui::SetTooltip("Open Project");

    // Save Labels
    ImGui::BeginDisabled(!pm.plot_keypoints_flag);
    if (ImGui::MenuItem(ICON_FK_FLOPPY_O "##toolbar_save")) {
        ctx.save_requested = true;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        ImGui::SetTooltip("Save (" RED_MOD_KEY "+S)");
    ImGui::EndDisabled();

    // Settings
    if (ImGui::MenuItem(ICON_FK_COG "##toolbar_settings")) {
        settings_state.show = true;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        ImGui::SetTooltip("Settings");

    // --- Right-aligned project name ---
    if (!pm.project_name.empty()) {
        float avail = ImGui::GetContentRegionAvail().x;
        float text_w = ImGui::CalcTextSize(pm.project_name.c_str()).x;
        if (avail > text_w + 8.0f) {
            ImGui::SameLine(ImGui::GetWindowWidth() - text_w - 16.0f);
            ImGui::TextDisabled("%s", pm.project_name.c_str());
        }
    }

    ImGui::EndMainMenuBar();
}
