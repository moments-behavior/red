#pragma once
#include "app_context.h"
#include "gui/window_states.h"
#include "IconsForkAwesome.h"
#include <ImGuiFileDialog.h>

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

    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Open Video(s)")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 0;
            config.path = pm.media_folder;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseMedia", "Choose Media", ".mp4", config);
        }
        if (ImGui::MenuItem("Open Images")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 0;
            config.path = pm.media_folder;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseImages", "Choose Images",
                ".jpg,.tiff,.jpeg,.png", config);
        }
        ImGui::BeginDisabled(!ps.video_loaded);
        if (ImGui::MenuItem("Create Project")) {
            pm.show_project_window = true;
        }
        ImGui::EndDisabled();
        if (ImGui::MenuItem("Load Project")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = pm.project_root_path;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseProject", "Choose Project File", ".redproj",
                config);
        }
        ImGui::BeginDisabled(pm.project_path.empty());
        if (ImGui::MenuItem("Switch Skeleton...")) {
            win.switch_skeleton.show = true;
            win.switch_skeleton.initialized = false;
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        // Save Labels — same action as the toolbar floppy icon and the Labeling
        // Tool's Save button: ctx.save_requested is forwarded to the labeling
        // tool, which writes the per-camera and 3D label CSVs.
        ImGui::BeginDisabled(!pm.plot_keypoints_flag);
        if (ImGui::MenuItem("Save Labels")) {
            ctx.save_requested = true;
        }
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Annotate")) {
        if (ImGui::MenuItem("Create Annotation Project")) {
            annot_state.show = true;
            annot_state.discovered_cameras.clear();
            annot_state.camera_selected.clear();
            annot_state.status.clear();
        }
        if (ImGui::MenuItem("Load Annotation Project")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = pm.project_root_path;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "LoadAnnotProject", "Load Annotation Project",
                "Red Project{.redproj}", cfg);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools")) {
        // 3D / calibration-dependent tools are disabled for 2D (uncalibrated)
        // projects — they index camera calibration and would otherwise crash.
        const bool is_2d = project_is_2d(pm);

        if (ImGui::MenuItem("Export Tool")) {
            export_state.show = true;
        }
        // Standalone multi-dataset merge — works with no project open, so it is
        // deliberately NOT gated by the is_2d / open-project check below.
        if (ImGui::MenuItem("Group JARVIS Export...")) {
            win.group_export.show = true;
        }
        ImGui::Separator();
        ImGui::BeginDisabled(is_2d);
        if (ImGui::MenuItem("JARVIS Export Tool")) {
            jarvis_export_state.show = true;
        }
        if (ImGui::MenuItem("Import JARVIS Predictions")) {
            win.jarvis_import.show = true;
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        if (ImGui::MenuItem("Bbox Tool")) {
            bbox_state.show = true;
        }
        if (ImGui::MenuItem("OBB Tool")) {
            obb_state.show = true;
        }
        ImGui::BeginDisabled(is_2d);
        if (ImGui::MenuItem("Midline Tool")) {
            win.midline.show = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Triangulation Diagnostics")) {
            triangulation_diag_state.show = true;
        }
        ImGui::EndDisabled();
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Settings")) {
            settings_state.show = true;
        }
        if (ImGui::MenuItem("Pose Stats")) {
            win.pose_stats.show = true;
        }
        if (ImGui::MenuItem("Frame Drops")) {
            win.frame_drops.show = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Help")) {
            show_help_window = true;
        }
        ImGui::EndMenu();
    }

    // --- Toolbar icons ---
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);

    // New Project
    ImGui::BeginDisabled(!ps.video_loaded);
    if (ImGui::MenuItem(ICON_FK_FILE_O "##toolbar_new")) {
        pm.show_project_window = true;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        ImGui::SetTooltip("Create Project");
    ImGui::EndDisabled();

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
        ImGui::SetTooltip("Save Labels");
    ImGui::EndDisabled();

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
