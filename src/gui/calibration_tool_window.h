#pragma once
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "app_context.h"
#include "calibration_tool.h"
#include "calibration_pipeline.h"
#include "laser_calibration.h"
#include "aruco_metal.h"
#include "laser_metal.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <string>
#include <thread>
#include <vector>

// Async detection visualization state (laser spot overlay)
struct LaserVizState {
    // Double-buffered: "ready" results for display, "pending" being computed
    struct CamResult {
        int num_blobs = 0;
        int total_mask_pixels = 0;
        std::vector<uint8_t> rgba;       // w*h*4 RGBA mask
        int frame_num = -1;              // frame this result corresponds to
        bool uploaded = false;           // has this been uploaded to GPU?
    };
    std::vector<CamResult> ready;        // display these
    std::vector<CamResult> pending;      // being computed in background
    std::atomic<bool> computing{false};  // background work in flight
    std::thread worker;

#ifdef __APPLE__
    // Metal context for GPU-accelerated viz (macOS only)
    LaserMetalHandle metal_ctx = nullptr;
#endif

    // Params that triggered the current computation
    int last_green_th = -1, last_green_dom = -1;
    int last_min_blob = -1, last_max_blob = -1;

    ~LaserVizState() {
        if (worker.joinable()) worker.join();
#ifdef __APPLE__
        if (metal_ctx) laser_metal_destroy(metal_ctx);
#endif
    }
};

struct CalibrationToolState {
    // Core project state
    bool show = false;
    CalibrationTool::CalibProject project;
    bool project_loaded = false;
    bool dock_pending = false;
    bool show_create_dialog = true;
    std::string config_path;
    CalibrationTool::CalibConfig config;
    bool config_loaded = false;
    bool images_loaded = false;
    std::string status;

    // Aruco image pipeline async
    bool img_running = false;
    bool img_done = false;
    CalibrationPipeline::CalibrationResult img_result;
    std::future<CalibrationPipeline::CalibrationResult> img_future;

    // Aruco video pipeline async
    bool vid_running = false;
    bool vid_done = false;
    CalibrationPipeline::CalibrationResult vid_result;
    std::future<CalibrationPipeline::CalibrationResult> vid_future;

    // Aruco video state
    bool aruco_videos_loaded = false;
    int aruco_start_frame = 0;
    int aruco_stop_frame = 0;      // 0 = all
    int aruco_frame_step = 10;     // 30fps → 3fps effective
    int aruco_total_frames = 0;
    int aruco_video_count = 0;     // cached number of matched videos

    // Experimental image pipeline async
    bool exp_img_running = false;
    bool exp_img_done = false;
    CalibrationPipeline::CalibrationResult exp_img_result;
    std::future<CalibrationPipeline::CalibrationResult> exp_img_future;

    // Experimental video pipeline async
    bool exp_vid_running = false;
    bool exp_vid_done = false;
    CalibrationPipeline::CalibrationResult exp_vid_result;
    std::future<CalibrationPipeline::CalibrationResult> exp_vid_future;

    // Helper: is any aruco pipeline running?
    bool aruco_running() const {
        return img_running || vid_running || exp_img_running || exp_vid_running;
    }

    // Laser refinement
    bool laser_ready = false;
    LaserCalibration::LaserConfig laser_config;
    int laser_total_frames = 0;
    bool laser_running = false;
    bool laser_done = false;
    std::string laser_status;
    LaserCalibration::LaserResult laser_result;
    std::shared_ptr<LaserCalibration::DetectionProgress> laser_progress =
        std::make_shared<LaserCalibration::DetectionProgress>();
    std::future<LaserCalibration::LaserResult> laser_future;
    bool laser_show_detection = false;
    bool laser_focus_window = false;

    // Laser visualization
    LaserVizState laser_viz;
};

struct CalibrationToolCallbacks {
    // Load calibration images into camera windows
    std::function<void(std::map<std::string, std::string> &files)> load_images;
    // Load laser videos into camera windows
    std::function<void()> load_videos;
    // Unload all media (on close/reset)
    std::function<void()> unload_media;
    // Copy default layout ini to project folder (if not already present)
    std::function<void(const std::string &project_path)> copy_default_layout;
    // Switch imgui_layout.ini to project folder
    std::function<void(const std::string &project_path)> switch_ini;
    // Print video metadata to console
    std::function<void()> print_metadata;
    // Deferred queue for scheduling main-thread work from callbacks
    DeferredQueue *deferred = nullptr;
};

inline void DrawCalibrationToolWindow(
    CalibrationToolState &state, AppContext &ctx,
    const CalibrationToolCallbacks &cb) {
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    const auto &user_settings = ctx.user_settings;
    const auto &red_data_dir = ctx.red_data_dir;
    auto &imgs_names = ctx.imgs_names;
#ifdef __APPLE__
    auto &mac_last_uploaded_frame = ctx.mac_last_uploaded_frame;
#endif
    // Always process file dialogs (even when window is hidden)

    // Calibration: Browse for root directory (creation dialog)
    if (ImGuiFileDialog::Instance()->Display("ChooseCalibRootDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            state.project.project_root_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Calibration: Browse for config.json (creation dialog)
    if (ImGuiFileDialog::Instance()->Display("ChooseCalibConfigCreate", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected =
                ImGuiFileDialog::Instance()->GetFilePathName();
            if (!selected.empty())
                state.project.config_file = selected;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Calibration: Load existing .redproj (unified — handles both aruco and laser projects)
    if (ImGuiFileDialog::Instance()->Display(
            "LoadCalibProject", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string selected =
                ImGuiFileDialog::Instance()->GetFilePathName();
            if (!selected.empty()) {
                CalibrationTool::CalibProject loaded;
                std::string err;
                if (CalibrationTool::load_project(
                        &loaded, selected, &err)) {
                    state.project = loaded;

                    // Switch layout ini to project folder
                    cb.switch_ini(state.project.project_path);

                    // Load aruco config if present
                    if (state.project.has_aruco()) {
                        state.config_path = state.project.config_file;
                        if (CalibrationTool::parse_config(
                                state.config_path, state.config, err)) {
                            state.config_loaded = true;
                            state.images_loaded = false;
                            state.img_done = false;
                            state.vid_done = false;
                        } else {
                            state.config_loaded = false;
                            state.status = "Error parsing config: " + err;
                        }
                    }

                    // Set up laser config if laser inputs present.
                    // For pure laser projects (no aruco), auto-load videos.
                    // For aruco+laser projects, defer to the "Load Laser Videos" button.
                    if (state.project.has_laser_input() &&
                        !state.project.has_aruco()) {
                        state.laser_config.media_folder = state.project.media_folder;
                        state.laser_config.calibration_folder =
                            state.project.calibration_folder;
                        state.laser_config.camera_names = state.project.camera_names;
                        state.laser_config.output_folder =
                            state.project.project_path + "/laser_calibration";
                        state.laser_ready = true;
                        state.laser_focus_window = true;

                        // Load videos into 2x2 grid
                        if (!ps.video_loaded) {
                            pm.media_folder = state.project.media_folder;
                            pm.camera_names.clear();
                            for (const auto &cn : state.project.camera_names)
                                pm.camera_names.push_back("Cam" + cn);
                            cb.load_videos();
                            cb.print_metadata();
                        }
                        state.laser_total_frames = dc_context->estimated_num_frames;
                    }

                    state.project_loaded = true;
                    state.dock_pending = true;
                    state.show_create_dialog = false;
                    state.show = true;

                    // Status message
                    if (state.project.has_aruco() && state.config_loaded) {
                        state.status =
                            "Project loaded: " +
                            std::to_string(state.config.cam_ordered.size()) +
                            " cameras";
                    } else if (state.project.has_laser_input()) {
                        state.status =
                            "Project loaded: " +
                            std::to_string(state.project.camera_names.size()) +
                            " cameras (laser refinement)";
                    }
                } else {
                    state.status = "Error loading project: " + err;
                }
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (!state.show)
        return;

    // ── Calibration Tool: Unified Creation Dialog + Tool Window ──

    // Phase 1: Show creation dialog when no project is loaded
    if (state.show_create_dialog && !state.project_loaded) {
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

                // ── Project Name ──
                ImGui::TableNextRow();
                LabelCell("Project Name");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_projname",
                                 &state.project.project_name);
                ImGui::TableSetColumnIndex(2);
                ImGui::Dummy(ImVec2(1, 1));

                // ── Project Root Path ──
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

                // ── Full Path (computed, read-only) ──
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

                // ── Config File (optional — empty for laser-only) ──
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

                // ── Aruco Videos (optional) ──
                ImGui::TableNextRow();
                LabelCell("Aruco Videos");
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::InputText("##calib_arucovids",
                                 &state.project.aruco_video_folder);
                ImGui::TableSetColumnIndex(2);
                if (ImGui::Button("Browse##calib_arucovid")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    if (!state.project.aruco_video_folder.empty())
                        cfg.path = state.project.aruco_video_folder;
                    else if (!user_settings.default_media_root_path.empty())
                        cfg.path = user_settings.default_media_root_path;
                    else
                        cfg.path = red_data_dir;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseCalibArucoVideoFolder",
                        "Select Aruco Videos Folder", nullptr, cfg);
                }

                // ── Initialize Calibration YAMLs (optional) ──
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

                // ── Laser Videos (optional) ──
                ImGui::TableNextRow();
                LabelCell("Laser Videos");
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
                        "ChooseCalibVideoFolder",
                        "Select Laser Videos Folder", nullptr, cfg);
                }

                ImGui::EndTable();
            }

            // Handle file dialogs for Aruco Videos, YAML and Laser Video folder browsing
            if (ImGuiFileDialog::Instance()->Display(
                    "ChooseCalibArucoVideoFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.aruco_video_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
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
                    "ChooseCalibVideoFolder", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.project.media_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                ImGuiFileDialog::Instance()->Close();
            }

            // Validate and show matched cameras when laser inputs provided
            if (!state.project.media_folder.empty() &&
                !state.project.calibration_folder.empty()) {
                auto matched = LaserCalibration::validate_cameras(
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
                // YAML folder only — derive camera names from filenames
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

            ImGui::Separator();

            // Validation: name + root required, AND at least one calibration source
            const bool create_ok =
                !state.project.project_name.empty() &&
                !state.project.project_root_path.empty() &&
                (!state.project.config_file.empty() ||
                 !state.project.calibration_folder.empty() ||
                 !state.project.aruco_video_folder.empty());

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
                        state.images_loaded = false;
                        state.img_done = false;
                        state.vid_done = false;
                        state.project.img_path = state.config.img_path;
                    } else {
                        state.config_loaded = false;
                        state.status = "Error parsing config: " + err;
                    }
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
                    state.project.laser_output_folder =
                        state.project.project_path + "/laser_calibration";

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
                            // For aruco+laser, defer to "Load Laser Videos" button.
                            if (state.project.has_laser_input() &&
                                !state.project.has_aruco()) {
                                state.laser_config.media_folder =
                                    state.project.media_folder;
                                state.laser_config.calibration_folder =
                                    state.project.calibration_folder;
                                state.laser_config.camera_names =
                                    state.project.camera_names;
                                state.laser_config.output_folder =
                                    state.project.laser_output_folder;
                                state.laser_ready = true;

                                // Load videos
                                if (!ps.video_loaded) {
                                    pm.media_folder = state.project.media_folder;
                                    pm.camera_names.clear();
                                    for (const auto &cn : state.project.camera_names)
                                        pm.camera_names.push_back("Cam" + cn);
                                    cb.load_videos();
                                    cb.print_metadata();
                                }
                                state.laser_total_frames = dc_context->estimated_num_frames;
                            }

                            // Status
                            if (state.config_loaded) {
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
                                    " cameras (laser refinement)";
                            }
                        }
                    }
                }
            }
            ImGui::EndDisabled();
        }
        ImGui::End();

    } else if (state.project_loaded) {
        // Phase 2: Unified Calibration Tool window (project loaded)
        // Clear stale dock-pending flag (no longer auto-docking)
        state.dock_pending = false;
        ImGui::SetNextWindowSize(ImVec2(580, 600), ImGuiCond_FirstUseEver);
        if (state.laser_focus_window) {
            ImGui::SetNextWindowFocus();
            state.laser_focus_window = false;
        }
        if (ImGui::Begin("Calibration Tool", &state.show)) {

            // ── Section 1: Project Info ──
            if (ImGui::CollapsingHeader("Project", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                ImGui::Text("Project: %s",
                            state.project.project_name.c_str());
                ImGui::Text("Path:    %s",
                            state.project.project_path.c_str());
                ImGui::Unindent();
            }
            ImGui::Spacing();

            // ── Section 2: Aruco Calibration (if has_aruco) ──
            if (state.project.has_aruco() && state.config_loaded) {
            if (ImGui::CollapsingHeader("Aruco Calibration", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();
            ImGui::Text("Cameras:      %d",
                        (int)state.config.cam_ordered.size());
            ImGui::Text("Board:        %d x %d  (%.1f mm squares)",
                        state.config.charuco_setup.w,
                        state.config.charuco_setup.h,
                        state.config.charuco_setup.square_side_length);

            // ── Sub-section: Image Calibration ──
            bool has_images = !state.config.img_path.empty();
            if (has_images) {
            if (ImGui::CollapsingHeader("Image Calibration",
                    ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                ImGui::Text("Image Path: %s",
                            state.config.img_path.c_str());

                // Count images (cached)
                static std::string cached_img_path;
                static int cached_img_count = 0;
                static int cached_per_cam = 0;
                if (cached_img_path != state.config.img_path) {
                    cached_img_path = state.config.img_path;
                    auto files =
                        CalibrationTool::discover_images(state.config);
                    cached_img_count = (int)files.size();
                    if (!state.config.cam_ordered.empty()) {
                        cached_per_cam =
                            CalibrationTool::count_images_per_camera(
                                files, state.config.cam_ordered[0]);
                    }
                }
                ImGui::Text("Images: %d total (%d per camera)",
                            cached_img_count, cached_per_cam);

                // Load Images button
                if (!state.images_loaded) {
                    if (ImGui::Button("Load Images")) {
                        auto files =
                            CalibrationTool::discover_images(state.config);
                        if (files.empty()) {
                            state.status =
                                "Error: No matching images found in " +
                                state.config.img_path;
                        } else {
                            pm.media_folder = state.config.img_path;
                            pm.camera_names.clear();
                            imgs_names.clear();
                            cb.load_images(files);
                            state.images_loaded = true;
                            state.status =
                                "Loaded " + std::to_string(files.size()) +
                                " images across " +
                                std::to_string(
                                    state.config.cam_ordered.size()) +
                                " cameras";
                        }
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                       "Images loaded");
                }

                ImGui::Separator();

                // Poll image pipeline future
                if (state.img_running && state.img_future.valid()) {
                    auto fs = state.img_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.img_result = state.img_future.get();
                        state.img_running = false;
                        state.img_done = true;
                        if (state.img_result.success) {
                            state.project.image_output_folder =
                                state.img_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Image calibration complete! Reproj: " +
                                std::to_string(
                                    state.img_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                        } else {
                            state.status =
                                "Error: " + state.img_result.error;
                        }
                    }
                }

                // Calibrate (images) button
                bool img_can_run = state.config_loaded &&
                    !state.config.img_path.empty() &&
                    !state.aruco_running();
                ImGui::BeginDisabled(!img_can_run);
                if (ImGui::Button("Calibrate##img")) {
                    state.img_running = true;
                    state.img_done = false;
                    state.status = "Starting image calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_image_calibration";
                    state.img_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr,
                                nullptr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.img_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.img_done && state.img_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Mean reproj error: %.3f px",
                        state.img_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.image_output_folder.c_str());
                }

                ImGui::Spacing();

                // Poll experimental image pipeline future
                if (state.exp_img_running && state.exp_img_future.valid()) {
                    auto fs = state.exp_img_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.exp_img_result = state.exp_img_future.get();
                        state.exp_img_running = false;
                        state.exp_img_done = true;
                        if (state.exp_img_result.success) {
                            state.project.image_experimental_folder =
                                state.exp_img_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Experimental image calibration complete! Reproj: " +
                                std::to_string(
                                    state.exp_img_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                        } else {
                            state.status =
                                "Error: " + state.exp_img_result.error;
                        }
                    }
                }

                // Experimental (images) button
                ImGui::BeginDisabled(!img_can_run);
                if (ImGui::Button("Experimental##exp_img")) {
                    state.exp_img_running = true;
                    state.exp_img_done = false;
                    state.status = "Starting experimental image calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_image_experimental";
                    state.exp_img_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_experimental_pipeline(
                                config, base, status_ptr,
                                nullptr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_experimental_pipeline(
                                config, base, status_ptr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.exp_img_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.exp_img_done && state.exp_img_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Experimental reproj error: %.3f px",
                        state.exp_img_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.image_experimental_folder.c_str());
                }

                ImGui::Unindent();
            } // end Image Calibration header
            } // end has_images

            // ── Sub-section: Video Calibration ──
            bool has_videos = !state.project.aruco_video_folder.empty();
            if (has_videos) {
            if (ImGui::CollapsingHeader("Video Calibration",
                    ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                // Cache video discovery
                static std::string cached_vid_folder;
                if (cached_vid_folder != state.project.aruco_video_folder) {
                    cached_vid_folder = state.project.aruco_video_folder;
                    auto vids = CalibrationTool::discover_aruco_videos(
                        state.project.aruco_video_folder,
                        state.config.cam_ordered);
                    state.aruco_video_count = (int)vids.size();
                    if (!vids.empty()) {
                        state.aruco_total_frames =
                            CalibrationPipeline::get_video_frame_count(
                                vids.begin()->second);
                    }
                }

                ImGui::Text("Video Path: %s",
                            state.project.aruco_video_folder.c_str());
                ImGui::Text("Videos: %d cameras, ~%d frames each",
                            state.aruco_video_count,
                            state.aruco_total_frames);

                // Load Videos button (display in camera viewports)
                if (!state.aruco_videos_loaded) {
                    ImGui::BeginDisabled(
                        state.aruco_video_count == 0 ||
                        state.aruco_running());
                    if (ImGui::Button("Load Videos##aruco_vid")) {
                        // Defer unload+load to next frame start
                        // (freeing Metal textures mid-frame crashes ImGui)
                        state.status = "Loading aruco videos...";
                        cb.deferred->enqueue([&state, &pm, &ps, &cb,
                                              &imgs_names, dc_context
#ifdef __APPLE__
                                              , &mac_last_uploaded_frame
#endif
                        ]() {
                            try {
                                if (ps.video_loaded)
                                    cb.unload_media();
                                imgs_names.clear();
#ifdef __APPLE__
                                for (size_t ci = 0;
                                     ci < mac_last_uploaded_frame.size(); ci++)
                                    mac_last_uploaded_frame[ci] = -1;
#endif
                                pm.media_folder =
                                    state.project.aruco_video_folder;
                                pm.camera_names.clear();
                                for (const auto &cn :
                                     state.config.cam_ordered)
                                    pm.camera_names.push_back("Cam" + cn);
                                cb.load_videos();
                                cb.print_metadata();
                                state.aruco_total_frames =
                                    dc_context->estimated_num_frames;
                                state.aruco_videos_loaded = true;
                                state.status =
                                    "Loaded " +
                                    std::to_string(
                                        state.config.cam_ordered.size()) +
                                    " aruco videos";
                            } catch (const std::exception &e) {
                                state.status =
                                    std::string("Error loading videos: ") +
                                    e.what();
                            }
                        });
                    }
                    ImGui::EndDisabled();
                } else {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                       "Videos loaded");
                }

                ImGui::Separator();

                int slider_max = state.aruco_total_frames > 0
                    ? state.aruco_total_frames : 100000;
                ImGui::SliderInt("Start Frame##aruco",
                                 &state.aruco_start_frame, 0, slider_max);
                ImGui::SliderInt("Stop Frame (0=all)##aruco",
                                 &state.aruco_stop_frame, 0, slider_max);
                ImGui::SliderInt("Every Nth Frame##aruco",
                                 &state.aruco_frame_step, 1, 100);
                {
                    int eff_stop = state.aruco_stop_frame > 0
                        ? state.aruco_stop_frame
                        : (state.aruco_total_frames > 0
                            ? state.aruco_total_frames : 0);
                    if (eff_stop > state.aruco_start_frame &&
                        state.aruco_frame_step > 0) {
                        int est = (eff_stop - state.aruco_start_frame) /
                                  state.aruco_frame_step;
                        ImGui::Text("~%d frames per camera (%d cameras = %d total)",
                                    est, state.aruco_video_count,
                                    est * state.aruco_video_count);
                    }
                }

                ImGui::Separator();

                // Poll video pipeline future
                if (state.vid_running && state.vid_future.valid()) {
                    auto fs = state.vid_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.vid_result = state.vid_future.get();
                        state.vid_running = false;
                        state.vid_done = true;
                        if (state.vid_result.success) {
                            state.project.video_output_folder =
                                state.vid_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Video calibration complete! Reproj: " +
                                std::to_string(
                                    state.vid_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                        } else {
                            state.status =
                                "Error: " + state.vid_result.error;
                        }
                    }
                }

                // Calibrate (video) button
                bool vid_can_run = state.config_loaded &&
                    state.aruco_video_count > 0 &&
                    !state.aruco_running();
                ImGui::BeginDisabled(!vid_can_run);
                if (ImGui::Button("Calibrate##vid")) {
                    state.vid_running = true;
                    state.vid_done = false;
                    state.status = "Starting video calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_video_calibration";

                    CalibrationPipeline::VideoFrameRange vfr;
                    vfr.video_folder = state.project.aruco_video_folder;
                    vfr.cam_ordered = state.config.cam_ordered;
                    vfr.start_frame = state.aruco_start_frame;
                    vfr.stop_frame = state.aruco_stop_frame;
                    vfr.frame_step = state.aruco_frame_step;

                    state.vid_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status, vfr]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr,
                                &vfr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_full_pipeline(
                                config, base, status_ptr, &vfr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.vid_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.vid_done && state.vid_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Mean reproj error: %.3f px",
                        state.vid_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.video_output_folder.c_str());
                }

                ImGui::Spacing();

                // Poll experimental video pipeline future
                if (state.exp_vid_running && state.exp_vid_future.valid()) {
                    auto fs = state.exp_vid_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fs == std::future_status::ready) {
                        state.exp_vid_result = state.exp_vid_future.get();
                        state.exp_vid_running = false;
                        state.exp_vid_done = true;
                        if (state.exp_vid_result.success) {
                            state.project.video_experimental_folder =
                                state.exp_vid_result.output_folder;
                            std::string pf =
                                state.project.project_path + "/" +
                                state.project.project_name + ".redproj";
                            std::string se;
                            CalibrationTool::save_project(
                                state.project, pf, &se);
                            state.status =
                                "Experimental video calibration complete! Reproj: " +
                                std::to_string(
                                    state.exp_vid_result.mean_reproj_error)
                                    .substr(0, 5) + " px";
                        } else {
                            state.status =
                                "Error: " + state.exp_vid_result.error;
                        }
                    }
                }

                // Experimental (video) button
                ImGui::BeginDisabled(!vid_can_run);
                if (ImGui::Button("Experimental##exp_vid")) {
                    state.exp_vid_running = true;
                    state.exp_vid_done = false;
                    state.status = "Starting experimental video calibration...";
                    std::string base =
                        state.project.project_path +
                        "/aruco_video_experimental";

                    CalibrationPipeline::VideoFrameRange exp_vfr;
                    exp_vfr.video_folder = state.project.aruco_video_folder;
                    exp_vfr.cam_ordered = state.config.cam_ordered;
                    exp_vfr.start_frame = state.aruco_start_frame;
                    exp_vfr.stop_frame = state.aruco_stop_frame;
                    exp_vfr.frame_step = state.aruco_frame_step;

                    state.exp_vid_future = std::async(
                        std::launch::async,
                        [config = state.config, base,
                         status_ptr = &state.status, exp_vfr]() {
#ifdef __APPLE__
                            auto am = aruco_metal_create();
                            aruco_detect::GpuThresholdFunc gfn =
                                am ? aruco_metal_threshold_batch : nullptr;
                            auto r = CalibrationPipeline::run_experimental_pipeline(
                                config, base, status_ptr,
                                &exp_vfr, gfn, am);
                            aruco_metal_destroy(am);
                            return r;
#else
                            return CalibrationPipeline::run_experimental_pipeline(
                                config, base, status_ptr, &exp_vfr);
#endif
                        });
                }
                ImGui::EndDisabled();
                if (state.exp_vid_running) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Running...");
                }
                if (state.exp_vid_done && state.exp_vid_result.success) {
                    ImGui::TextColored(
                        ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                        "Experimental reproj error: %.3f px",
                        state.exp_vid_result.mean_reproj_error);
                    ImGui::Text("Output: %s",
                        state.project.video_experimental_folder.c_str());
                }

                ImGui::Unindent();
            } // end Video Calibration header
            } // end has_videos

            // ── Quality Dashboard (visible after any experimental pipeline completes) ──
            {
                const CalibrationPipeline::CalibrationResult *exp_result = nullptr;
                if (state.exp_vid_done && state.exp_vid_result.success)
                    exp_result = &state.exp_vid_result;
                else if (state.exp_img_done && state.exp_img_result.success)
                    exp_result = &state.exp_img_result;

                if (exp_result && !exp_result->per_camera_metrics.empty()) {
                    if (ImGui::CollapsingHeader("Quality Dashboard",
                            ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();

                        const auto &metrics = exp_result->per_camera_metrics;
                        int nc = (int)metrics.size();

                        // Summary text
                        ImGui::Text("BA rounds: %d  |  Outliers removed: %d  |  "
                                    "Mean reproj: %.3f px",
                                    exp_result->ba_rounds,
                                    exp_result->outliers_removed,
                                    exp_result->mean_reproj_error);
                        ImGui::Spacing();

                        // Prepare data arrays for ImPlot
                        std::vector<double> mean_errs(nc), median_errs(nc);
                        std::vector<double> det_counts(nc);
                        std::vector<const char *> labels(nc);
                        std::vector<std::string> label_strs(nc);
                        for (int i = 0; i < nc; i++) {
                            mean_errs[i] = metrics[i].mean_reproj;
                            median_errs[i] = metrics[i].median_reproj;
                            det_counts[i] = (double)metrics[i].detection_count;
                            label_strs[i] = metrics[i].name;
                            labels[i] = label_strs[i].c_str();
                        }

                        // Bar chart: Per-camera reprojection error (mean + median)
                        if (ImPlot::BeginPlot("Per-Camera Reprojection Error",
                                ImVec2(-1, 200))) {
                            ImPlot::SetupAxes("Camera", "Error (px)");
                            ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, nc,
                                                    labels.data());
                            ImPlot::PlotBars("Mean", mean_errs.data(), nc, 0.3, -0.15);
                            ImPlot::PlotBars("Median", median_errs.data(), nc, 0.3, 0.15);
                            ImPlot::EndPlot();
                        }

                        // Bar chart: Detection count per camera
                        if (ImPlot::BeginPlot("Detections Per Camera",
                                ImVec2(-1, 160))) {
                            ImPlot::SetupAxes("Camera", "Frames");
                            ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, nc,
                                                    labels.data());
                            ImPlot::PlotBars("Detections", det_counts.data(),
                                              nc, 0.5);
                            ImPlot::EndPlot();
                        }

                        // Histogram: reprojection error distribution
                        if (!exp_result->all_reproj_errors.empty()) {
                            // Pre-bin into 50 bins from 0 to max
                            const auto &all_errs = exp_result->all_reproj_errors;
                            double max_err = *std::max_element(
                                all_errs.begin(), all_errs.end());
                            max_err = std::max(max_err, 1.0);
                            int num_bins = 50;
                            double bin_width = max_err / num_bins;
                            std::vector<double> bins(num_bins, 0);
                            std::vector<double> bin_centers(num_bins);
                            for (int i = 0; i < num_bins; i++)
                                bin_centers[i] = (i + 0.5) * bin_width;
                            for (double e : all_errs) {
                                int b = std::min((int)(e / bin_width),
                                                  num_bins - 1);
                                bins[b]++;
                            }

                            if (ImPlot::BeginPlot("Error Distribution",
                                    ImVec2(-1, 160))) {
                                ImPlot::SetupAxes("Error (px)", "Count");
                                ImPlot::PlotBars("Observations", bin_centers.data(),
                                                  bins.data(), num_bins, bin_width);
                                ImPlot::EndPlot();
                            }
                        }

                        // Per-camera detail table
                        if (ImGui::TreeNode("Per-Camera Details")) {
                            if (ImGui::BeginTable("exp_cam_details", 7,
                                    ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_SizingFixedFit)) {
                                ImGui::TableSetupColumn("Camera", 0, 80.0f);
                                ImGui::TableSetupColumn("Dets", 0, 45.0f);
                                ImGui::TableSetupColumn("Obs", 0, 50.0f);
                                ImGui::TableSetupColumn("Mean", 0, 55.0f);
                                ImGui::TableSetupColumn("Median", 0, 55.0f);
                                ImGui::TableSetupColumn("Std", 0, 55.0f);
                                ImGui::TableSetupColumn("Max", 0, 55.0f);
                                ImGui::TableHeadersRow();

                                for (const auto &m : metrics) {
                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::Text("%s", m.name.c_str());
                                    ImGui::TableSetColumnIndex(1);
                                    ImGui::Text("%d", m.detection_count);
                                    ImGui::TableSetColumnIndex(2);
                                    ImGui::Text("%d", m.observation_count);
                                    ImGui::TableSetColumnIndex(3);
                                    ImGui::Text("%.3f", m.mean_reproj);
                                    ImGui::TableSetColumnIndex(4);
                                    ImGui::Text("%.3f", m.median_reproj);
                                    ImGui::TableSetColumnIndex(5);
                                    ImGui::Text("%.3f", m.std_reproj);
                                    ImGui::TableSetColumnIndex(6);
                                    ImGui::Text("%.3f", m.max_reproj);
                                }
                                ImGui::EndTable();
                            }
                            ImGui::TreePop();
                        }

                        ImGui::Unindent();
                    } // end Quality Dashboard
                }
            }

            ImGui::Unindent();
            } // end CollapsingHeader("Aruco Calibration")
            ImGui::Spacing();
            } // end aruco section

            // ── Section 3: Laser Refinement ──
            bool aruco_succeeded =
                (state.img_done && state.img_result.success) ||
                (state.vid_done && state.vid_result.success);
            bool show_laser_section =
                state.project.has_laser_input() || aruco_succeeded;

            // Auto-populate laser calibration_folder from most recent aruco output
            if (aruco_succeeded &&
                state.project.calibration_folder.empty()) {
                // Prefer video output if available, else image output
                std::string aruco_out =
                    !state.project.video_output_folder.empty()
                        ? state.project.video_output_folder
                        : state.project.image_output_folder;
                if (!aruco_out.empty()) {
                    state.project.calibration_folder = aruco_out;
                    state.project.camera_names =
                        CalibrationTool::derive_camera_names_from_yaml(
                            state.project.calibration_folder);
                    state.project.laser_output_folder =
                        state.project.project_path + "/laser_calibration";
                    std::string proj_file =
                        state.project.project_path + "/" +
                        state.project.project_name + ".redproj";
                    std::string save_err;
                    CalibrationTool::save_project(
                        state.project, proj_file, &save_err);
                }
            }

            if (show_laser_section) {
                if (ImGui::CollapsingHeader("Laser Refinement", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();

                // Poll async result
                if (state.laser_running && state.laser_future.valid()) {
                    auto fut_status = state.laser_future.wait_for(
                        std::chrono::milliseconds(0));
                    if (fut_status == std::future_status::ready) {
                        state.laser_result = state.laser_future.get();
                        state.laser_running = false;
                        state.laser_done = true;
                        if (state.laser_result.success) {
                            state.laser_status =
                                "Complete! Reproj: " +
                                std::to_string(
                                    state.laser_result.mean_reproj_before)
                                    .substr(0, 5) +
                                " -> " +
                                std::to_string(
                                    state.laser_result.mean_reproj_after)
                                    .substr(0, 5) +
                                " px. Output: " +
                                state.laser_result.output_folder;
                        } else {
                            state.laser_status =
                                "Error: " + state.laser_result.error;
                        }
                    }
                }

                ImGui::Text("Calibration: %s",
                            state.project.calibration_folder.c_str());

                // Laser Video Folder — text field + Browse + Load button
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 200.0f);
                ImGui::InputText("##laser_vid_path",
                                 &state.project.media_folder);
                ImGui::SameLine();
                if (ImGui::Button("Browse##laser_vid_tool")) {
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
                        "ChooseLaserVideoTool",
                        "Select Laser Video Folder", nullptr, cfg);
                }
                ImGui::SameLine();
                {
                    // Validate cameras only when paths change (avoid per-frame filesystem I/O)
                    static std::string last_media, last_calib;
                    bool paths_changed = (state.project.media_folder != last_media ||
                                          state.project.calibration_folder != last_calib);
                    if (paths_changed &&
                        !state.project.media_folder.empty() &&
                        !state.project.calibration_folder.empty()) {
                        last_media = state.project.media_folder;
                        last_calib = state.project.calibration_folder;
                        state.project.camera_names = LaserCalibration::validate_cameras(
                            state.project.media_folder,
                            state.project.calibration_folder);
                    }
                    bool has_valid_cameras = !state.project.camera_names.empty();

                    ImGui::BeginDisabled(
                        state.project.media_folder.empty() ||
                        !has_valid_cameras || state.laser_running);
                    if (ImGui::Button("Load Laser Videos")) {
                        // Save updated media_folder to .redproj
                        state.project.laser_output_folder =
                            state.project.project_path + "/laser_calibration";
                        std::string proj_file =
                            state.project.project_path + "/" +
                            state.project.project_name + ".redproj";
                        std::string save_err;
                        CalibrationTool::save_project(
                            state.project, proj_file, &save_err);

                        // Set up laser_config
                        state.laser_config.media_folder = state.project.media_folder;
                        state.laser_config.calibration_folder =
                            state.project.calibration_folder;
                        state.laser_config.camera_names = state.project.camera_names;
                        state.laser_config.output_folder =
                            state.project.laser_output_folder;

                        // Defer the actual unload+load to next frame start
                        // (freeing Metal textures mid-frame crashes ImGui rendering)
                        state.laser_status = "Loading laser videos...";
                        cb.deferred->enqueue([&state, &pm, &ps, &cb, &imgs_names,
                                              dc_context
#ifdef __APPLE__
                                              , &mac_last_uploaded_frame
#endif
                        ]() {
                            try {
                                if (ps.video_loaded)
                                    cb.unload_media();
                                imgs_names.clear();
#ifdef __APPLE__
                                for (size_t ci = 0; ci < mac_last_uploaded_frame.size(); ci++)
                                    mac_last_uploaded_frame[ci] = -1;
#endif
                                pm.media_folder = state.project.media_folder;
                                for (const auto &cn : state.project.camera_names)
                                    pm.camera_names.push_back("Cam" + cn);
                                cb.load_videos();
                                cb.print_metadata();
                                state.laser_total_frames = dc_context->estimated_num_frames;
                                state.laser_ready = true;
                                state.laser_status =
                                    "Loaded " +
                                    std::to_string(state.project.camera_names.size()) +
                                    " laser videos";
                            } catch (const std::exception &e) {
                                state.laser_status =
                                    std::string("Error loading videos: ") + e.what();
                            }
                        });
                    }
                    ImGui::EndDisabled();
                }

                // Handle video folder browse dialog
                if (ImGuiFileDialog::Instance()->Display(
                        "ChooseLaserVideoTool", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        state.project.media_folder =
                            ImGuiFileDialog::Instance()->GetCurrentPath();
                    }
                    ImGuiFileDialog::Instance()->Close();
                }

                // Show matched cameras
                if (!state.project.camera_names.empty()) {
                    char cam_header[64];
                    snprintf(cam_header, sizeof(cam_header), "Cameras (%d matched)",
                             (int)state.project.camera_names.size());
                    if (ImGui::CollapsingHeader(cam_header, ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                        if (ImGui::BeginTable("##cam_grid", 4)) {
                            for (int i = 0; i < (int)state.project.camera_names.size(); i++) {
                                ImGui::TableNextColumn();
                                ImGui::Text("Cam%s", state.project.camera_names[i].c_str());
                            }
                            ImGui::EndTable();
                        }
                        ImGui::Unindent();
                    }
                } else if (!state.project.media_folder.empty() &&
                           !state.project.calibration_folder.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                        "No matching videos found — update the video folder path");
                }

                if (state.laser_ready) {
                    // Detection parameters
                    if (ImGui::CollapsingHeader("Detection Parameters")) {
                    ImGui::Indent();
                    ImGui::SliderInt("Green Threshold",
                                     &state.laser_config.green_threshold, 20, 255);
                    ImGui::SliderInt("Green Dominance",
                                     &state.laser_config.green_dominance, 5, 100);
                    ImGui::SliderInt("Min Blob Pixels",
                                     &state.laser_config.min_blob_pixels, 1, 100);
                    ImGui::SliderInt("Max Blob Pixels",
                                     &state.laser_config.max_blob_pixels, 50, 5000);

                    int slider_max = state.laser_total_frames > 0 ? state.laser_total_frames : 100000;
                    ImGui::SliderInt("Start Frame",
                                     &state.laser_config.start_frame, 0, slider_max);
                    ImGui::SliderInt("Stop Frame (0=all)",
                                     &state.laser_config.stop_frame, 0, slider_max);
                    ImGui::SliderInt("Every Nth Frame",
                                     &state.laser_config.frame_step, 1, 100);
                    {
                        int eff_stop = state.laser_config.stop_frame > 0
                            ? state.laser_config.stop_frame
                            : (state.laser_total_frames > 0 ? state.laser_total_frames : 0);
                        if (eff_stop > state.laser_config.start_frame && state.laser_config.frame_step > 0) {
                            int est = (eff_stop - state.laser_config.start_frame) / state.laser_config.frame_step;
                            ImGui::Text("~%d frames per camera", est);
                        } else if (state.laser_config.stop_frame == 0 && state.laser_total_frames == 0) {
                            ImGui::Text("~all frames per camera");
                        }
                    }
                    ImGui::Unindent();
                    } // end Detection Parameters

                    // Filtering parameters
                    if (ImGui::CollapsingHeader("Filtering")) {
                    ImGui::Indent();
                    int max_min_cams =
                        std::max(2, (int)state.laser_config.camera_names.size());
                    ImGui::SliderInt("Min Cameras",
                                     &state.laser_config.min_cameras, 2,
                                     max_min_cams);
                    float reproj_th = (float)state.laser_config.reproj_threshold;
                    if (ImGui::SliderFloat("Reproj Threshold (px)", &reproj_th,
                                           1.0f, 50.0f))
                        state.laser_config.reproj_threshold = reproj_th;
                    ImGui::Unindent();
                    } // end Filtering

                    // BA parameters
                    if (ImGui::CollapsingHeader("Bundle Adjustment")) {
                    ImGui::Indent();
                    float ba_th1 = (float)state.laser_config.ba_outlier_th1;
                    float ba_th2 = (float)state.laser_config.ba_outlier_th2;
                    if (ImGui::SliderFloat("BA Outlier Pass 1 (px)", &ba_th1,
                                           1.0f, 50.0f))
                        state.laser_config.ba_outlier_th1 = ba_th1;
                    if (ImGui::SliderFloat("BA Outlier Pass 2 (px)", &ba_th2,
                                           0.5f, 20.0f))
                        state.laser_config.ba_outlier_th2 = ba_th2;
                    ImGui::SliderInt("BA Max Iterations",
                                     &state.laser_config.ba_max_iter, 10, 200);
                    ImGui::Checkbox("Lock Intrinsics",
                                    &state.laser_config.lock_intrinsics);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Fix focal length and distortion coefficients.\n"
                            "Recommended when laser points lack depth diversity.");
                    ImGui::Unindent();
                    } // end Bundle Adjustment

                    ImGui::Separator();

                    // Run button
                    bool can_run_laser = !state.laser_running &&
                                       !state.laser_config.camera_names.empty();
                    ImGui::BeginDisabled(!can_run_laser);
                    if (ImGui::Button("Run Laser Refinement")) {
                        state.laser_running = true;
                        state.laser_done = false;
                        state.laser_status =
                            "Starting laser calibration pipeline...";
                        state.laser_future = std::async(
                            std::launch::async,
                            [config = state.laser_config,
                             status_ptr = &state.laser_status,
                             prog = state.laser_progress]() {
                                return LaserCalibration::
                                    run_laser_refinement(config, status_ptr,
                                                         prog.get());
                            });
                    }
                    ImGui::EndDisabled();

                    if (state.laser_running) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                           "Running...");
                    }

                    // Progress sub-section (visible during/after detection)
                    if (state.laser_running && !state.laser_progress->cameras.empty()) {
                        if (ImGui::CollapsingHeader("Progress", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                            int eff_stop = state.laser_config.stop_frame > 0
                                ? state.laser_config.stop_frame
                                : (state.laser_total_frames > 0 ? state.laser_total_frames : 0);
                            int eff_start = state.laser_config.start_frame;
                            int eff_step = std::max(1, state.laser_config.frame_step);
                            int prog_total = (eff_stop > eff_start)
                                ? (eff_stop - eff_start + eff_step - 1) / eff_step
                                : 0;
                            int total_done = 0;
                            for (int ci = 0; ci < (int)state.laser_progress->cameras.size(); ci++)
                                if (state.laser_progress->cameras[ci]->done.load(std::memory_order_relaxed))
                                    total_done++;
                            ImGui::Text("Detection: %d / %d cameras complete",
                                        total_done, (int)state.laser_progress->cameras.size());

                            if (ImGui::BeginTable(
                                    "laser_det_progress", 4,
                                    ImGuiTableFlags_RowBg |
                                        ImGuiTableFlags_BordersInnerV)) {
                                ImGui::TableSetupColumn("Camera", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                                ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthStretch);
                                ImGui::TableSetupColumn("Spots", ImGuiTableColumnFlags_WidthFixed, 60.0f);
                                ImGui::TableSetupColumn("Rate", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                                ImGui::TableHeadersRow();

                                for (int ci = 0; ci < (int)state.laser_progress->cameras.size(); ci++) {
                                    auto &cp = state.laser_progress->cameras[ci];
                                    int fr = cp->frames_processed.load(std::memory_order_relaxed);
                                    int sp = cp->spots_detected.load(std::memory_order_relaxed);
                                    bool dn = cp->done.load(std::memory_order_relaxed);
                                    float frac = prog_total > 0 ? (float)fr / prog_total : 0.0f;
                                    if (frac > 1.0f) frac = 1.0f;

                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    if (ci < (int)state.laser_config.camera_names.size())
                                        ImGui::Text("Cam%s", state.laser_config.camera_names[ci].c_str());
                                    ImGui::TableSetColumnIndex(1);
                                    char overlay[64];
                                    if (prog_total > 0)
                                        snprintf(overlay, sizeof(overlay), "%d / %d", fr, prog_total);
                                    else
                                        snprintf(overlay, sizeof(overlay), "%d", fr);
                                    if (dn)
                                        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
                                    ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0), overlay);
                                    if (dn)
                                        ImGui::PopStyleColor();
                                    ImGui::TableSetColumnIndex(2);
                                    ImGui::Text("%d", sp);
                                    ImGui::TableSetColumnIndex(3);
                                    ImGui::Text("%.0f%%", fr > 0 ? 100.0 * sp / fr : 0.0);
                                }
                                ImGui::EndTable();
                            }
                        ImGui::Unindent();
                        } // end Progress
                    }

                    // Results
                    if (state.laser_done && state.laser_result.success) {
                        if (ImGui::CollapsingHeader("Results", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Indent();
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                           "Reprojection error: %.3f -> %.3f px",
                                           state.laser_result.mean_reproj_before,
                                           state.laser_result.mean_reproj_after);
                        double avg_obs = state.laser_result.valid_3d_points > 0
                            ? (double)state.laser_result.total_observations / state.laser_result.valid_3d_points
                            : 0.0;
                        ImGui::Text("3D points: %d | Observations: %d (avg %.1f cameras/point)",
                                    state.laser_result.valid_3d_points,
                                    state.laser_result.total_observations,
                                    avg_obs);
                        if (state.laser_result.ba_outliers_removed > 0)
                            ImGui::Text("BA outliers removed: %d", state.laser_result.ba_outliers_removed);
                        ImGui::Text("Output: %s",
                                    state.laser_result.output_folder.c_str());

                        // Per-camera changes table
                        if (!state.laser_result.camera_changes.empty()) {
                            if (ImGui::TreeNode("Per-camera changes")) {
                                if (ImGui::BeginTable(
                                        "laser_cam_changes", 9,
                                        ImGuiTableFlags_RowBg |
                                            ImGuiTableFlags_BordersInnerV |
                                            ImGuiTableFlags_Resizable |
                                            ImGuiTableFlags_SizingFixedFit)) {
                                    ImGui::TableSetupColumn("Camera", 0, 80.0f);
                                    ImGui::TableSetupColumn("Spots", 0, 45.0f);
                                    ImGui::TableSetupColumn("Obs", 0, 40.0f);
                                    ImGui::TableSetupColumn("dfx", 0, 55.0f);
                                    ImGui::TableSetupColumn("dfy", 0, 55.0f);
                                    ImGui::TableSetupColumn("dcx", 0, 55.0f);
                                    ImGui::TableSetupColumn("dcy", 0, 55.0f);
                                    ImGui::TableSetupColumn("|dt|", 0, 55.0f);
                                    ImGui::TableSetupColumn("drot", 0, 60.0f);
                                    ImGui::TableHeadersRow();

                                    for (const auto &cc : state.laser_result.camera_changes) {
                                        ImGui::TableNextRow();
                                        ImGui::TableSetColumnIndex(0);
                                        ImGui::Text("Cam%s", cc.name.c_str());
                                        ImGui::TableSetColumnIndex(1);
                                        ImGui::Text("%d", cc.detections);
                                        ImGui::TableSetColumnIndex(2);
                                        ImGui::Text("%d", cc.observations);
                                        ImGui::TableSetColumnIndex(3);
                                        ImGui::Text("%+.2f", cc.dfx);
                                        ImGui::TableSetColumnIndex(4);
                                        ImGui::Text("%+.2f", cc.dfy);
                                        ImGui::TableSetColumnIndex(5);
                                        ImGui::Text("%+.2f", cc.dcx);
                                        ImGui::TableSetColumnIndex(6);
                                        ImGui::Text("%+.2f", cc.dcy);
                                        ImGui::TableSetColumnIndex(7);
                                        ImGui::Text("%.3f", cc.dt_norm);
                                        ImGui::TableSetColumnIndex(8);
                                        ImGui::Text("%.4f%s", cc.drot_deg, "\xC2\xB0");
                                    }
                                    ImGui::EndTable();
                                }
                                ImGui::TreePop();
                            }
                        }
                        ImGui::Unindent();
                        } // end CollapsingHeader("Results")
                    }

                    // Laser status
                    if (!state.laser_status.empty()) {
                        ImGui::Separator();
                        if (state.laser_status.find("Error") !=
                            std::string::npos) {
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                                state.laser_status.c_str());
                        } else {
                            ImGui::TextWrapped("%s", state.laser_status.c_str());
                        }
                    }

                    // Detection Processing visualization
                    if (ImGui::CollapsingHeader("Detection Processing")) {
                    ImGui::Indent();
                    bool prev_detection = state.laser_show_detection;
                    ImGui::Checkbox("Enable", &state.laser_show_detection);
                    if (prev_detection && !state.laser_show_detection) {
#ifdef __APPLE__
                        for (int ci = 0; ci < scene->num_cams; ci++)
                            mac_last_uploaded_frame[ci] = -1;
#endif
                        state.laser_viz.ready.clear();
                    }
                    if (state.laser_show_detection && ps.video_loaded) {
                        if (state.laser_viz.computing.load(std::memory_order_relaxed))
                            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                               "  Processing...");
                        int detecting_count = 0;
                        int total_cams = std::min((int)state.laser_viz.ready.size(), (int)scene->num_cams);
                        for (int ci = 0; ci < total_cams; ci++) {
                            auto &cr = state.laser_viz.ready[ci];
                            const char *blob_str =
                                cr.num_blobs == 0  ? "0 blobs" :
                                cr.num_blobs == 1  ? "1 blob (OK)" :
                                cr.num_blobs == -1 ? "invalid" :
                                "multiple blobs";
                            ImVec4 col = cr.num_blobs == 1
                                ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)
                                : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                            ImGui::TextColored(col, "  %s: %s, %d px",
                                ci < (int)pm.camera_names.size()
                                    ? pm.camera_names[ci].c_str() : "?",
                                blob_str, cr.total_mask_pixels);
                            if (cr.num_blobs == 1) detecting_count++;
                        }
                        if (total_cams > 0) {
                            ImVec4 summary_col = (detecting_count == total_cams)
                                ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)
                                : ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                            ImGui::TextColored(summary_col,
                                "  %d/%d cameras detecting", detecting_count, total_cams);
                        }
                        ImGui::TextWrapped(
                            "Green=valid blob, Yellow=too small, "
                            "Red=too large, Gray=filtered by erode/dilate");
                    }
                    ImGui::Unindent();
                    } // end Detection Processing
                } else {
                    // Laser inputs not yet complete — show hint
                    if (state.project.media_folder.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                            "Set Video Folder to enable laser refinement");
                    } else if (state.project.camera_names.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                            "Set Video Folder and click Load Laser Videos");
                    }
                }
                ImGui::Unindent();
                } // end CollapsingHeader("Laser Refinement")
                ImGui::Spacing();
            } // end laser section

            // Status text (general)
            if (!state.status.empty()) {
                ImGui::Separator();
                if (state.status.find("Error") != std::string::npos) {
                    ImGui::TextColored(
                        ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%s",
                        state.status.c_str());
                } else {
                    ImGui::Text("%s", state.status.c_str());
                }
            }
        }
        ImGui::End();
    }

    // Reset state when calibration tool window is closed
    if (!state.show) {
        state.project_loaded = false;
        state.show_create_dialog = true;
        state.config_loaded = false;
        state.images_loaded = false;
        state.img_done = false;
        state.vid_done = false;
        state.exp_img_done = false;
        state.exp_vid_done = false;
        state.status.clear();
        state.laser_ready = false;
        state.laser_done = false;
        state.laser_status.clear();
        state.laser_show_detection = false;
        if (state.laser_viz.worker.joinable())
            state.laser_viz.worker.join();
        state.laser_viz.ready.clear();
        state.laser_viz.pending.clear();
#ifdef __APPLE__
        for (int ci = 0; ci < scene->num_cams; ci++)
            mac_last_uploaded_frame[ci] = -1;
#endif
    }
}
