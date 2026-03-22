#pragma once
#include "calib_tool_state.h"
#include "app_context.h"
#include "imgui.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <filesystem>
#include <string>

// Draw the Phase 1 creation dialog when no project is loaded.
// Called only when state.show_create_dialog && !state.project_loaded.
inline void DrawCalibCreateDialog(CalibrationToolState &state, AppContext &ctx,
                                   const CalibrationToolCallbacks &cb) {
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto *dc_context = ctx.dc_context;
    const auto &user_settings = ctx.user_settings;
    const auto &red_data_dir = ctx.red_data_dir;

        ImGui::SetNextWindowSize(ImVec2(720, 440), ImGuiCond_FirstUseEver);
        ImGuiWindowFlags cw_flags = ImGuiWindowFlags_NoCollapse;
        if (ImGui::Begin("Create Calibration Project", &state.show, cw_flags)) {

            // Error banner
            if (!state.status.empty() &&
                state.status.find("Error") != std::string::npos) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                                      ImVec4(1.0f, 0.45f, 0.45f, 1.0f));
                ImGui::TextUnformatted(state.status.c_str());
                ImGui::PopStyleColor();
                ImGui::Separator();
            }

            if (ImGui::BeginTable(
                    "calibCreateForm", 3,
                    ImGuiTableFlags_SizingStretchProp |
                        ImGuiTableFlags_PadOuterX |
                        ImGuiTableFlags_RowBg |
                        ImGuiTableFlags_BordersInnerV)) {
                ImGui::TableSetupColumn("Label",
                                        ImGuiTableColumnFlags_WidthFixed,
                                        180.0f);
                ImGui::TableSetupColumn("Field",
                                        ImGuiTableColumnFlags_WidthStretch,
                                        1.0f);
                ImGui::TableSetupColumn("Action",
                                        ImGuiTableColumnFlags_WidthFixed,
                                        110.0f);

                auto LabelCell = [](const char *t) {
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted(t);
                };

                // ---- Project Name ----
                ImGui::TableNextRow();
                LabelCell("Project Name");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_projname",
                                 &state.project.project_name);
                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));

                // ---- Project Root Path ----
                ImGui::TableNextRow();
                LabelCell("Project Root Path");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_rootpath",
                                 &state.project.project_root_path);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_root")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.path = state.project.project_root_path;
                    cfg.fileName = state.project.project_name;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibRootDir",
                        "Choose Project Root Directory", nullptr, cfg);
                }

                // ---- Full Path (computed, read-only) ----
                {
                    std::filesystem::path p =
                        std::filesystem::path(
                            state.project.project_root_path) /
                        state.project.project_name;
                    state.project.project_path = p.string();
                }
                ImGui::TableNextRow();
                LabelCell("Full Path");
                ImGui::TableSetColumnIndex(1);
                ImGui::BeginDisabled();
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_fullpath",
                                 &state.project.project_path);
                ImGui::EndDisabled();
                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));

                // ---- Camera Model ----
                ImGui::TableNextRow();
                LabelCell("Camera Model");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                {
                    const char *model_items[] = {"Projective", "Telecentric"};
                    int model_idx = static_cast<int>(state.project.camera_model);
                    if (ImGui::Combo("##calib_camera_model", &model_idx,
                                     model_items, IM_ARRAYSIZE(model_items))) {
                        auto new_model = static_cast<CalibrationTool::CameraModel>(model_idx);
                        if (new_model != state.project.camera_model) {
                            state.project.camera_model = new_model;
                            // Clear the other mode's fields
                            if (new_model == CalibrationTool::CameraModel::Telecentric) {
                                state.project.config_file.clear();
                                state.project.aruco_media_folder.clear();
                                state.project.aruco_media_type.clear();
                                state.project.global_reg_media_folder.clear();
                                state.project.global_reg_media_type.clear();
                                state.project.calibration_folder.clear();
                                state.project.camera_names.clear();
                            } else {
                                state.project.landmark_labels_folder.clear();
                                state.project.landmarks_3d_file.clear();
                                state.project.media_folder.clear();
                                state.project.camera_names.clear();
                            }
                        }
                    }
                }
                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));

                if (!state.project.is_telecentric()) {
                // ---- Projective mode fields ----

                // ---- Config File (optional -- empty for laser-only) ----
                ImGui::TableNextRow();
                LabelCell("Config File (optional)");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_configfile",
                                 &state.project.config_file);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_cfg")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    if (!state.project.config_file.empty()) {
                        cfg.path =
                            std::filesystem::path(
                                state.project.config_file)
                                .parent_path()
                                .string();
                    } else if (!user_settings.default_media_root_path
                                    .empty()) {
                        cfg.path =
                            user_settings.default_media_root_path;
                    }
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibConfigCreate",
                        "Choose config.json", ".json", cfg);
                }

                // ---- Aruco Media (images or videos — auto-detected) ----
                ImGui::TableNextRow();
                LabelCell("Aruco Media");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::InputText("##calib_arucomedia",
                                     &state.project.aruco_media_folder)) {
                    // Re-detect on manual edit
                    auto info = CalibrationTool::detect_aruco_media(
                        state.project.aruco_media_folder);
                    state.project.aruco_media_type = info.type;
                    state.calib_aruco_media_info = info;
                }
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_arucomedia")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.aruco_media_folder.empty())
                        cfg.path = state.project.aruco_media_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibArucoMediaFolder",
                        "Select Aruco Media Folder", nullptr, cfg);
                }
                // Show auto-detection result
                if (!state.calib_aruco_media_info.description.empty()) {
                    ImGui::TableNextRow();
                    LabelCell("");
                    ImGui::TableSetColumnIndex(1);
                    if (state.calib_aruco_media_info.type.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                            "No aruco images or videos found in folder");
                    } else {
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "%s", state.calib_aruco_media_info.description.c_str());
                    }
                }

                // ---- Global Registration Media (optional) ----
                ImGui::TableNextRow();
                LabelCell("Global Reg. Media (optional)");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::InputText("##calib_globalreg",
                                     &state.project.global_reg_media_folder)) {
                    auto info = CalibrationTool::detect_aruco_media(
                        state.project.global_reg_media_folder);
                    state.project.global_reg_media_type = info.type;
                    state.calib_global_reg_info = info;
                }
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_globalreg")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.global_reg_media_folder.empty())
                        cfg.path = state.project.global_reg_media_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibGlobalRegFolder",
                        "Select Global Registration Media Folder", nullptr, cfg);
                }
                // Show auto-detection result
                if (!state.calib_global_reg_info.description.empty()) {
                    ImGui::TableNextRow();
                    LabelCell("");
                    ImGui::TableSetColumnIndex(1);
                    if (state.calib_global_reg_info.type.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                            "No images or videos found in folder");
                    } else {
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "%s", state.calib_global_reg_info.description.c_str());
                    }
                }

                // ---- Initialize Calibration YAMLs (optional) ----
                ImGui::TableNextRow();
                LabelCell("Initialize Calibration YAMLs");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_yamlfolder",
                                 &state.project.calibration_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_yaml")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.calibration_folder.empty())
                        cfg.path = state.project.calibration_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibYAMLFolder",
                        "Select Calibration YAMLs Folder", nullptr, cfg);
                }

                // ---- Laser Videos (optional) ----
                ImGui::TableNextRow();
                LabelCell("PointSource Videos");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_vidfolder",
                                 &state.project.media_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_vid")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.media_folder.empty())
                        cfg.path = state.project.media_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibPointSourceFolder",
                        "Select PointSource Videos Folder", nullptr, cfg);
                }

                // ---- Board Setup (when no config file, aruco media detected) ----
                if (state.project.config_file.empty() &&
                    !state.calib_aruco_media_info.type.empty()) {
                    // Separator row
                    ImGui::TableNextRow();
                    LabelCell(""); ImGui::TableSetColumnIndex(1);
                    ImGui::Separator();

                    ImGui::TableNextRow();
                    LabelCell("Board Setup");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                        "No config file — set board parameters:");

                    auto &cs = state.project.charuco_setup;
                    ImGui::TableNextRow();
                    LabelCell("  Squares X");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(120);
                    ImGui::InputInt("##board_w", &cs.w);
                    cs.w = std::max(3, std::min(cs.w, 20));

                    ImGui::TableNextRow();
                    LabelCell("  Squares Y");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(120);
                    ImGui::InputInt("##board_h", &cs.h);
                    cs.h = std::max(3, std::min(cs.h, 20));

                    ImGui::TableNextRow();
                    LabelCell("  Square Size (mm)");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(120);
                    ImGui::InputFloat("##board_sqsize", &cs.square_side_length, 1.0f, 10.0f, "%.1f");
                    cs.square_side_length = std::max(1.0f, cs.square_side_length);

                    ImGui::TableNextRow();
                    LabelCell("  Marker Size (mm)");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(120);
                    ImGui::InputFloat("##board_mksize", &cs.marker_side_length, 1.0f, 10.0f, "%.1f");
                    cs.marker_side_length = std::max(1.0f, cs.marker_side_length);

                    ImGui::TableNextRow();
                    LabelCell("  Dictionary");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(200);
                    {
                        const char *dict_names[] = {
                            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
                            "unused_3",
                            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
                            "unused_7",
                            "DICT_6X6_50", "unused_9", "DICT_6X6_250"};
                        // Map dictionary IDs (0-10) to combo, skipping unused
                        int combo_items[] = {0, 1, 2, 4, 5, 6, 8, 10};
                        const char *combo_labels[] = {
                            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
                            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250",
                            "DICT_6X6_50", "DICT_6X6_250"};
                        int combo_count = 8;
                        int sel = 0;
                        for (int i = 0; i < combo_count; i++)
                            if (combo_items[i] == cs.dictionary) { sel = i; break; }
                        if (ImGui::Combo("##board_dict", &sel, combo_labels, combo_count))
                            cs.dictionary = combo_items[sel];
                    }

                    // Show auto-generated gt_pts summary
                    {
                        int n_pts = (cs.w - 1) * (cs.h - 1);
                        float half_x = (cs.w - 2) * cs.square_side_length / 2.0f;
                        float half_y = (cs.h - 2) * cs.square_side_length / 2.0f;
                        ImGui::TableNextRow();
                        LabelCell("  Ground Truth");
                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "%d pts, center-origin, z=0 (%.0f to %.0f mm)",
                            n_pts, -half_x, half_x);
                    }
                }

                } else {
                // ---- Telecentric mode fields ----

                // ---- Calibration Videos ----
                ImGui::TableNextRow();
                LabelCell("Calibration Videos");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##tele_vidfolder",
                                 &state.project.media_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##tele_vid")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.media_folder.empty())
                        cfg.path = state.project.media_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseTeleVideoFolder",
                        "Select Calibration Videos Folder", nullptr, cfg);
                }

                // ---- 2D Landmark Labels ----
                ImGui::TableNextRow();
                LabelCell("2D Landmark Labels (optional)");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##tele_labelfolder",
                                 &state.project.landmark_labels_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##tele_label")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.landmark_labels_folder.empty())
                        cfg.path = state.project.landmark_labels_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseTeleLabelFolder",
                        "Select 2D Landmark Labels Folder", nullptr, cfg);
                }

                // ---- 3D Landmarks File ----
                ImGui::TableNextRow();
                LabelCell("3D Landmarks File (.csv)");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##tele_3dfile",
                                 &state.project.landmarks_3d_file);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##tele_3d")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    if (!state.project.landmarks_3d_file.empty()) {
                        cfg.path =
                            std::filesystem::path(
                                state.project.landmarks_3d_file)
                                .parent_path()
                                .string();
                    } else if (!user_settings.default_media_root_path.empty()) {
                        cfg.path = user_settings.default_media_root_path;
                    }
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseTele3DFile",
                        "Choose 3D Landmarks CSV", ".csv", cfg);
                }

                } // end telecentric fields

                ImGui::EndTable();
            }

            // Handle file dialogs for Aruco Media, Global Reg, YAML and Laser Video
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseCalibArucoMediaFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.project.aruco_media_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                    auto info = CalibrationTool::detect_aruco_media(
                        state.project.aruco_media_folder);
                    state.project.aruco_media_type = info.type;
                    state.calib_aruco_media_info = info;
                }
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseCalibGlobalRegFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.project.global_reg_media_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                    auto info = CalibrationTool::detect_aruco_media(
                        state.project.global_reg_media_folder);
                    state.project.global_reg_media_type = info.type;
                    state.calib_global_reg_info = info;
                }
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseCalibYAMLFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.calibration_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseCalibPointSourceFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.media_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                ImGuiFileDialog::Instance()->Close();
            }
            // Telecentric file dialog handlers
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseTeleVideoFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.media_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseTeleLabelFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.landmark_labels_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseTele3DFile", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.landmarks_3d_file =
                        ImGuiFileDialog::Instance()->GetFilePathName();
                ImGuiFileDialog::Instance()->Close();
            }

            // Validate and show matched cameras
            if (!state.project.is_telecentric()) {
                // Projective validation
                if (!state.project.media_folder.empty() &&
                    !state.project.calibration_folder.empty()) {
                    auto matched = PointSourceCalibration::validate_cameras(
                        state.project.media_folder,
                        state.project.calibration_folder);
                    state.project.camera_names = matched;

                    if (matched.empty()) {
                        ImGui::TextColored(
                            ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                            "No cameras matched between videos and YAML files");
                    } else {
                        std::string cam_list;
                        for (int i = 0; i < (int)matched.size(); i++) {
                            if (i > 0) cam_list += ", ";
                            cam_list += "Cam" + matched[i];
                        }
                        ImGui::Text("Matched cameras (%d): %s",
                                    (int)matched.size(), cam_list.c_str());
                    }
                } else if (!state.project.calibration_folder.empty() &&
                           state.project.media_folder.empty()) {
                    state.project.camera_names =
                        CalibrationTool::derive_camera_names_from_yaml(
                            state.project.calibration_folder);
                    if (!state.project.camera_names.empty()) {
                        std::string cam_list;
                        for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                            if (i > 0) cam_list += ", ";
                            cam_list += "Cam" + state.project.camera_names[i];
                        }
                        ImGui::Text("YAML cameras (%d): %s",
                                    (int)state.project.camera_names.size(),
                                    cam_list.c_str());
                    }
                }
            } else {
                // Telecentric validation (cached -- only rescan on path change)
                static std::string tc_cached_media;
                static std::string tc_cached_labels;
                static std::string tc_cached_3d;
                static std::vector<std::string> tc_cached_cams;
                static std::map<std::string, int> tc_cached_label_counts;
                static int tc_cached_label_found = 0;
                static bool tc_cached_3d_ok = false;

                // Rescan videos when media_folder changes
                if (tc_cached_media != state.project.media_folder) {
                    tc_cached_media = state.project.media_folder;
                    state.project.camera_names =
                        CalibrationTool::derive_camera_names_from_videos(
                            state.project.media_folder);
                    tc_cached_cams = state.project.camera_names;
                    // Invalidate dependent caches
                    tc_cached_labels.clear();
                    tc_cached_label_counts.clear();
                    tc_cached_label_found = 0;
                }
                if (!state.project.camera_names.empty()) {
                    std::string cam_list;
                    for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                        if (i > 0) cam_list += ", ";
                        cam_list += "Cam" + state.project.camera_names[i];
                    }
                    ImGui::Text("Video cameras (%d): %s",
                                (int)state.project.camera_names.size(),
                                cam_list.c_str());
                } else if (!state.project.media_folder.empty()) {
                    ImGui::TextColored(
                        ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                        "No Cam*.mp4 videos found in folder");
                }

                // Rescan labels when labels folder or cameras change
                if (!state.project.landmark_labels_folder.empty() &&
                    !state.project.camera_names.empty()) {
                    if (tc_cached_labels != state.project.landmark_labels_folder ||
                        tc_cached_cams != state.project.camera_names) {
                        tc_cached_labels = state.project.landmark_labels_folder;
                        tc_cached_cams = state.project.camera_names;
                        tc_cached_label_counts =
                            CalibrationTool::validate_telecentric_labels(
                                tc_cached_labels, tc_cached_cams);
                        tc_cached_label_found = (int)tc_cached_label_counts.size();
                    }
                    int total = (int)state.project.camera_names.size();
                    if (tc_cached_label_found == total) {
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "Labels found for all %d cameras", total);
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                            "Labels found for %d / %d cameras",
                            tc_cached_label_found, total);
                    }
                }

                // Check 3D file when path changes
                if (!state.project.landmarks_3d_file.empty()) {
                    if (tc_cached_3d != state.project.landmarks_3d_file) {
                        tc_cached_3d = state.project.landmarks_3d_file;
                        tc_cached_3d_ok =
                            std::filesystem::is_regular_file(tc_cached_3d);
                    }
                    if (tc_cached_3d_ok) {
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                            "3D landmarks file: OK");
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                            "3D landmarks file: not found");
                    }
                }
            }

            ImGui::Separator();

            // Validation: name + root required, AND at least one calibration source
            bool create_ok = false;
            if (!state.project.is_telecentric()) {
                create_ok =
                    !state.project.project_name.empty() &&
                    !state.project.project_root_path.empty() &&
                    (!state.project.config_file.empty() ||
                     !state.project.calibration_folder.empty() ||
                     !state.project.aruco_media_folder.empty());
            } else {
                // Telecentric: labels folder is optional (can label in-app)
                create_ok =
                    !state.project.project_name.empty() &&
                    !state.project.project_root_path.empty() &&
                    !state.project.media_folder.empty() &&
                    !state.project.landmarks_3d_file.empty() &&
                    !state.project.camera_names.empty();
            }

            // Right-align "Create Project" button
            float avail = ImGui::GetContentRegionAvail().x;
            const char *create_label = "Create Project##calib_create";
            float cw = ImGui::CalcTextSize(create_label).x +
                       ImGui::GetStyle().FramePadding.x * 2.0f;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() +
                                 (avail - cw));

            ImGui::BeginDisabled(!create_ok);
            if (ImGui::Button(create_label)) {
                std::string err;

                // Parse config.json if provided (aruco path)
                if (!state.project.config_file.empty()) {
                    state.config_path = state.project.config_file;
                    if (CalibrationTool::parse_config(
                            state.config_path, state.config, err)) {
                        state.config_loaded = true;
                        state.init_camera_enabled();
                        // Sync board setup to project for persistence
                        state.project.charuco_setup = state.config.charuco_setup;
                        state.images_loaded = false;
                        state.img_done = false;
                        state.vid_done = false;
                        // Populate legacy img_path from config
                        state.project.img_path = state.config.img_path;
                        // If user didn't set aruco_media_folder but config has img_path,
                        // auto-populate from config
                        if (state.project.aruco_media_folder.empty() &&
                            !state.config.img_path.empty()) {
                            state.project.aruco_media_folder = state.config.img_path;
                            auto info = CalibrationTool::detect_aruco_media(
                                state.project.aruco_media_folder);
                            state.project.aruco_media_type = info.type;
                        }
                        // If config has global_registration_video, use as global reg media
                        if (state.project.global_reg_media_folder.empty() &&
                            !state.config.global_registration_video.empty()) {
                            state.project.global_reg_media_folder =
                                state.config.global_registration_video;
                            auto info = CalibrationTool::detect_aruco_media(
                                state.project.global_reg_media_folder);
                            state.project.global_reg_media_type = info.type;
                        }
                        // Populate legacy aruco_video_folder for backward compat
                        if (state.project.aruco_is_video())
                            state.project.aruco_video_folder =
                                state.project.aruco_media_folder;
                    } else {
                        state.config_loaded = false;
                        state.status = "Error parsing config: " + err;
                    }
                }

                // Config-free aruco: synthesize CalibConfig from UI fields
                if (state.project.config_file.empty() &&
                    !state.project.aruco_media_folder.empty() &&
                    !state.calib_aruco_media_info.serials.empty()) {
                    // Derive camera names from detected media
                    state.project.camera_names =
                        state.calib_aruco_media_info.serials;
                    // Build CalibConfig from project fields
                    state.config.cam_ordered =
                        state.calib_aruco_media_info.serials;
                    state.config.charuco_setup = state.project.charuco_setup;
                    state.config.img_path = state.project.aruco_media_folder;
                    state.config_loaded = true;
                    state.init_camera_enabled();
                    state.images_loaded = false;
                    state.img_done = false;
                    state.vid_done = false;
                    // Populate legacy fields
                    if (state.project.aruco_is_video())
                        state.project.aruco_video_folder =
                            state.project.aruco_media_folder;
                    state.project.img_path = state.project.aruco_media_folder;
                }

                // Derive camera names for laser-only path (no config)
                if (state.project.config_file.empty() &&
                    !state.project.calibration_folder.empty() &&
                    state.project.camera_names.empty()) {
                    state.project.camera_names =
                        CalibrationTool::derive_camera_names_from_yaml(
                            state.project.calibration_folder);
                }

                // Set laser output folder if laser inputs present
                if (!state.project.calibration_folder.empty())
                    state.project.pointsource_output_folder =
                        state.project.project_path + "/pointsource_calibration";

                // Only proceed if no parse error
                if (state.status.find("Error") == std::string::npos) {
                    // Create project folder
                    if (!ensure_dir_exists(state.project.project_path,
                                           &err)) {
                        state.status = "Error: " + err;
                    } else {
                        // Copy default layout before switching ini
                        cb.copy_default_layout(
                            state.project.project_path);

                        // Save .redproj
                        std::string proj_file =
                            state.project.project_path + "/" +
                            state.project.project_name + ".redproj";
                        if (!CalibrationTool::save_project(
                                state.project, proj_file, &err)) {
                            state.status =
                                "Error saving project: " + err;
                        } else {
                            // Switch layout ini
                            cb.switch_ini(state.project.project_path);

                            state.project_loaded = true;
                            state.dock_pending = true;
                            state.show_create_dialog = false;

                            // For pure laser projects (no aruco), auto-load videos.
                            // For aruco+laser, defer to "Load PointSource Videos" button.
                            if (state.project.has_laser_input() &&
                                !state.project.has_aruco()) {
                                state.pointsource_config.media_folder =
                                    state.project.media_folder;
                                state.pointsource_config.calibration_folder =
                                    state.project.calibration_folder;
                                state.pointsource_config.camera_names =
                                    state.project.camera_names;
                                state.pointsource_config.output_folder =
                                    state.project.pointsource_output_folder;
                                state.pointsource_ready = true;

                                // Load videos
                                if (!ps.video_loaded) {
                                    pm.media_folder = state.project.media_folder;
                                    pm.camera_names.clear();
                                    for (const auto &cn : state.project.camera_names)
                                        pm.camera_names.push_back("Cam" + cn);
                                    cb.load_videos();
                                    cb.print_metadata();
                                }
                                state.pointsource_total_frames = dc_context->estimated_num_frames;
                            }

                            // Auto-load videos for telecentric (direct, like laser).
                            // Do NOT setup skeleton or labels here -- dock layout
                            // isn't ready yet. User clicks "Start Labeling" or
                            // "Load Videos" which handles skeleton + label import.
                            if (state.project.is_telecentric() &&
                                !ps.video_loaded) {
                                pm.media_folder = state.project.media_folder;
                                pm.camera_names.clear();
                                for (const auto &cn : state.project.camera_names)
                                    pm.camera_names.push_back("Cam" + cn);
                                cb.load_videos();
                                cb.print_metadata();
                                state.tele_videos_loaded = true;
                                // Schedule label import after dock layout stabilizes
                                state.tele_deferred_label_frames = 3;
                            }

                            // Status
                            if (state.project.is_telecentric()) {
                                state.status =
                                    "Project created. Loading videos...";
                            } else if (state.config_loaded) {
                                state.status =
                                    "Project created. Config: " +
                                    std::to_string(
                                        state.config.cam_ordered.size()) +
                                    " cameras";
                            } else {
                                state.status =
                                    "Project created: " +
                                    std::to_string(
                                        state.project.camera_names.size()) +
                                    " cameras (pointsource refinement)";
                            }
                        }
                    }
                }
            }
            ImGui::EndDisabled();
        }
        ImGui::End();
}
