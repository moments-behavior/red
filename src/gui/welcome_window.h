#pragma once
#include "imgui.h"
#include "app_context.h"
#include "gui/window_states.h"
#include "calibration_tool.h"
#include <filesystem>

// Blender-style welcome/startup screen shown when no project is loaded.
inline void DrawWelcomeWindow(AppContext &ctx, WindowStates &win) {
    // Center on viewport
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(520, 0));  // auto height

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoDocking;

    if (!ImGui::Begin("##Welcome", nullptr, flags)) {
        ImGui::End();
        return;
    }

    // Title
    {
        const char *title = "RED";
        const char *subtitle = "Multi-Camera Keypoint Labeling Tool";
        float title_w = ImGui::CalcTextSize(title).x;
        float sub_w = ImGui::CalcTextSize(subtitle).x;
        float avail = ImGui::GetContentRegionAvail().x;

        ImGui::SetCursorPosX((avail - title_w) * 0.5f);
        ImGui::PushFont(nullptr); // default font, but larger via scale
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "%s", title);
        ImGui::PopFont();

        ImGui::SetCursorPosX((avail - sub_w) * 0.5f);
        ImGui::TextDisabled("%s", subtitle);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    // Quick actions row
    {
        float btn_w = 150.0f;
        float avail = ImGui::GetContentRegionAvail().x;
        float spacing = 10.0f;
        float start_x = (avail - 2 * btn_w - spacing) * 0.5f;

        ImGui::SetCursorPosX(start_x);
        if (ImGui::Button("Open Videos", ImVec2(btn_w, 30))) {
            // Trigger the same action as File > Open Video(s)
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 0;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseVideo", "Select Video(s)",
                "Video{.mp4,.avi,.mkv,.mov}", cfg);
        }
        ImGui::SameLine(0, spacing);
        if (ImGui::Button("Load Project", ImVec2(btn_w, 30))) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseProject", "Load Project",
                "Red Project{.redproj}", cfg);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Calibrate section
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.6f, 1.0f), "Calibrate");
    ImGui::Spacing();

    auto calibButton = [&](const char *label, const char *desc,
                           CalibrationTool::CalibSubtype subtype) {
        ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0.5f));
        if (ImGui::Button(label, ImVec2(-1, 0))) {
            win.calibration.show = true;
            win.calibration.show_create_dialog = true;
            win.calibration.project = CalibrationTool::CalibProject{};
            win.calibration.project.subtype = subtype;
            win.calibration.config = CalibrationTool::CalibConfig{};
            win.calibration.config_loaded = false;
            win.calibration.calib_aruco_media_info = {};
            win.calibration.calib_global_reg_info = {};
            win.calibration.subtype_chosen = true;
            if (subtype == CalibrationTool::CalibSubtype::Telecentric)
                win.calibration.project.camera_model = CalibrationTool::CameraModel::Telecentric;
        }
        ImGui::PopStyleVar();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", desc);
    };

    calibButton("ArUco Calibration",
                "Calibrate cameras from ChArUco board images or videos",
                CalibrationTool::CalibSubtype::ArucoFull);
    calibButton("PointSource Refinement",
                "Refine an existing calibration with light wand data",
                CalibrationTool::CalibSubtype::PointSourceRefinement);
    calibButton("PointSource From Scratch",
                "Calibrate from ChArUco board + light wand (no prior calibration needed)",
                CalibrationTool::CalibSubtype::PointSourceFromScratch);
    calibButton("Telecentric DLT",
                "Calibrate telecentric cameras from known 3D landmarks",
                CalibrationTool::CalibSubtype::Telecentric);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Annotate section
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.6f, 1.0f), "Annotate");
    ImGui::Spacing();

    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0.5f));
    if (ImGui::Button("Create Annotation Project", ImVec2(-1, 0))) {
        win.annotation.show = true;
    }
    if (ImGui::Button("Load Annotation Project", ImVec2(-1, 0))) {
        IGFD::FileDialogConfig cfg;
        cfg.countSelectionMax = 1;
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "LoadAnnotProject", "Load Annotation Project",
            "Red Project{.redproj}", cfg);
    }
    ImGui::PopStyleVar();

    // Recent Projects section
    if (!ctx.user_settings.recent_projects.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Recent Projects");
        ImGui::Spacing();

        for (const auto &path : ctx.user_settings.recent_projects) {
            // Show just the project name + parent folder
            std::filesystem::path p(path);
            std::string display = p.parent_path().filename().string() + "/" + p.filename().string();
            if (!std::filesystem::exists(path)) {
                ImGui::TextDisabled("[missing] %s", display.c_str());
                continue;
            }
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0.5f));
            if (ImGui::Button(display.c_str(), ImVec2(-1, 0))) {
                // Load the project — detect type from JSON
                try {
                    std::ifstream f(path);
                    nlohmann::json j;
                    f >> j;
                    std::string type = j.value("type", std::string{});
                    if (type == "calibration" || type == "laser_calibration") {
                        // Load as calibration project
                        CalibrationTool::CalibProject loaded;
                        std::string err;
                        if (CalibrationTool::load_project(&loaded, path, &err)) {
                            win.calibration.project = loaded;
                            win.calibration.show = true;
                            win.calibration.project_loaded = true;
                            win.calibration.show_create_dialog = false;
                        }
                    } else {
                        // Load as annotation project via file dialog result
                        IGFD::FileDialogConfig cfg;
                        cfg.countSelectionMax = 1;
                        cfg.flags = ImGuiFileDialogFlags_Modal;
                        // Set the path directly — trigger the normal load flow
                        // by opening the ChooseProject dialog pre-pointed
                        cfg.path = p.parent_path().string();
                        cfg.fileName = p.filename().string();
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseProject", "Load Project",
                            "Red Project{.redproj}", cfg);
                    }
                } catch (...) {
                    // Silently skip corrupt files
                }
            }
            ImGui::PopStyleVar();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", path.c_str());
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Help
    float avail = ImGui::GetContentRegionAvail().x;
    float btn_w = 160.0f;
    ImGui::SetCursorPosX((avail - btn_w) * 0.5f);
    if (ImGui::Button("Help & Tutorials", ImVec2(btn_w, 0))) {
        win.show_help = true;
    }

    ImGui::Spacing();
    ImGui::End();
}
