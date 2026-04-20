#pragma once
#include "app_context.h"
#include "gui/window_states.h"
#include "IconsForkAwesome.h"
#include <ImGuiFileDialog.h>

inline void DrawMainMenuBar(AppContext &ctx, WindowStates &win) {
    auto &calib_state      = win.calibration;
    auto &annot_state      = win.annotation;
    auto &settings_state   = win.settings;
    auto &jarvis_export_state = win.jarvis_export;
    auto &jarvis_import_state = win.jarvis_import;
    auto &export_state     = win.export_win;
    auto &bbox_state       = win.bbox;
    auto &obb_state        = win.obb;
    auto &sam_tool_state   = win.sam_tool;
    auto &jarvis_predict_state = win.jarvis_predict;
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

    if (ImGui::BeginMenu("Calibrate")) {
        if (ImGui::MenuItem("Create Calibration Project")) {
            calib_state.show = true;
            calib_state.show_create_dialog = true;
            // Reset project fields for a fresh dialog
            calib_state.project = CalibrationTool::CalibProject{};
            calib_state.config = CalibrationTool::CalibConfig{};
            calib_state.config_loaded = false;
            calib_state.calib_aruco_media_info = {};
            calib_state.calib_global_reg_info = {};
        }
        if (ImGui::MenuItem("Load Calibration Project")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path =
                user_settings.default_project_root_path.empty()
                    ? ctx.red_data_dir
                    : user_settings.default_project_root_path;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "LoadCalibProject", "Load Calibration Project",
                "Red Project{.redproj}", cfg);
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools")) {
        if (ImGui::MenuItem("Export Tool")) {
            export_state.show = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("JARVIS Export Tool")) {
            jarvis_export_state.show = true;
        }
        if (ImGui::MenuItem("JARVIS Import Tool")) {
            jarvis_import_state.show = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Bbox Tool")) {
            bbox_state.show = true;
        }
        if (ImGui::MenuItem("OBB Tool")) {
            obb_state.show = true;
        }
        if (ImGui::MenuItem("SAM Assist")) {
            sam_tool_state.show = true;
        }
        if (ImGui::MenuItem("JARVIS Predict")) {
            jarvis_predict_state.show = true;
        }
#ifdef RED_HAS_MUJOCO
        ImGui::Separator();
        if (ImGui::MenuItem("Body Model")) {
            win.body_model.show = true;
        }
#endif
        ImGui::Separator();
        if (ImGui::MenuItem("Bout Inspector")) {
            ctx.bout_state.active = !ctx.bout_state.active;
            if (ctx.bout_state.active)
                ctx.bout_state.filters.dirty = true;
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Settings")) {
            settings_state.show = true;
        }
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
