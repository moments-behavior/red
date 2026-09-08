#pragma once
#include "app_context.h"
#include "video_files.h"
#include "gui/window_states.h"
#include "IconsForkAwesome.h"
#include "tailcycle_import.h"
#include "gui/tailcycle_open_window.h"
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
                "ChooseMedia", "Choose Media", video_ext_filter(), config);
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
        // Both modes of the one form. The mode is always set explicitly: it
        // used to be left at whatever the last creation put there, so making a
        // 2D project once meant every later "Create Project" quietly opened in
        // 2D mode too.
        auto open_create = [&](bool two_d) {
            annot_state.show = true;
            annot_state.two_d_mode = two_d;
            annot_state.discovered_cameras.clear();
            annot_state.camera_selected.clear();
            annot_state.status.clear();
        };
        // "Annotation" is kept rather than trimmed: it is the kind of project
        // this makes, and calibration projects are expected to come back as a
        // second kind. The welcome screen and the dialog title use the same
        // words.
        if (ImGui::MenuItem("Create Annotation Project"))
            open_create(false);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Calibrated multi-camera project with 3D "
                              "triangulation.");
        if (ImGui::MenuItem("Create 2D Annotation Project"))
            open_create(true);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Single or uncalibrated cameras. No calibration, "
                              "no triangulation.");
        if (ImGui::MenuItem("Load Annotation Project")) {
            IGFD::FileDialogConfig config;
            config.countSelectionMax = 1;
            config.path = pm.project_root_path;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseProject", "Choose Project File", ".redproj",
                config);
        }
        // A tailcycle session is not a .redproj -- it brings its own cameras,
        // calibration and skeleton -- but opening one belongs with the other
        // ways of opening something to work on. Hidden rather than disabled
        // when the build has no Parquet: there is nothing the user could do
        // about it from here.
        if (TailcycleImport::available()) {
            if (ImGui::MenuItem("Open tailcycle Dataset...")) {
                tailcycle_open_browse(win.tailcycle_open);
            }
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
        if (ImGui::MenuItem("Skeleton Creator")) {
            win.skeleton_creator.show = true;
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Draw a skeleton and save it as the .json a "
                              "project loads.");
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
    if (ImGui::MenuItem(ICON_FK_FILE_O "##toolbar_new")) {
        annot_state.show = true;
        annot_state.two_d_mode = false;
        annot_state.discovered_cameras.clear();
        annot_state.camera_selected.clear();
        annot_state.status.clear();
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        ImGui::SetTooltip("Create Project");

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
