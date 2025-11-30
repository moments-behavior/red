#include "IconsForkAwesome.h"
#include "camera.h"
#include "filesystem"
#include "global.h"
#include "gui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "live_table.h"
#include "project.h"
#include "render.h"
#include "reprojection_tool.h"
#include "skeleton.h"
#include "utils.h"
#include "yolo_export.h"
#include "yolo_torch.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#include "kernel.cuh"
#include "keypoints_table.h"
#include "reprojection_tool.h"

int main(int argc, char **argv) {
    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    *window =
        (gx_context){.swap_interval = 1, // use vsync
                     .width = 1920,
                     .height = 1080,
                     .render_target_title = (char *)malloc(100), // window title
                     .glsl_version = (char *)malloc(100)};

    render_initialize_target(window);
    RenderScene *scene = (RenderScene *)malloc(sizeof(RenderScene));
    std::string red_data_dir;
    std::string media_root_dir;
    prepare_application_folders(red_data_dir, media_root_dir);
    std::string skeleton_dir = red_data_dir + "/skeleton";
    std::string yolo_model_dir = red_data_dir + "/yolo_model";
    std::vector<std::thread> decoder_threads;
    std::vector<FFmpegDemuxer *> demuxers;

    DecoderContext *dc_context =
        (DecoderContext *)malloc(sizeof(DecoderContext));
    *dc_context = (DecoderContext){.decoding_flag = false,
                                   .stop_flag = false,
                                   .total_num_frame = int(INT_MAX),
                                   .estimated_num_frames = 0,
                                   .gpu_index = 0,
                                   .seek_interval = 250,
                                   .video_fps = 60.0f};

    // gui states, todo: bundle this later
    std::time_t last_saved = static_cast<std::time_t>(-1);
    bool cpu_buffer_toggle = true;
    int current_frame_num = 0;
    int previous_frame_num = -1;
    std::vector<std::string> imgs_names;

    // for labeling
    SkeletonContext skeleton;
    std::map<u32, KeyPoints *> keypoints_map;
    bool keypoints_find = false;
    std::map<std::string, SkeletonPrimitive> skeleton_map = skeleton_get_all();

    // others
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();

    ImPlotStyle &style = ImPlot::GetStyle();
    ImVec4 *colors = style.Colors;

    // Skeleton Creator variables
    bool show_skeleton_creator = false;
    std::vector<SkeletonCreatorNode> creator_nodes;
    std::vector<SkeletonCreatorEdge> creator_edges;
    int next_node_id = 0;
    int selected_node_for_edge = -1;
    std::string skeleton_creator_name = "CustomSkeleton";
    bool skeleton_creator_has_bbox = false;
    std::string skeleton_file_path = ""; // Track currently loaded skeleton file

    // YOLO Export Tool variables
    bool show_yolo_export_tool = false;
    std::string yolo_export_label_dir = media_root_dir + "/labeled_data";
    std::string yolo_export_video_dir = media_root_dir;
    std::string yolo_export_output_dir = media_root_dir + "/export";
    std::string yolo_export_skeleton_file;
    std::string yolo_export_class_names_file;
    std::vector<std::string> yolo_export_cam_names;
    int yolo_export_image_size = 640;
    float yolo_export_train_ratio = 0.7f;
    float yolo_export_val_ratio = 0.2f;
    float yolo_export_test_ratio = 0.1f;
    int yolo_export_seed = 42;
    YoloExportMode yolo_export_mode = YOLO_DETECTION;
    bool yolo_export_in_progress = false;
    std::string yolo_export_status = "";

    // Bounding box class management
    std::vector<std::string> bbox_class_names = {"Class_1"};
    std::vector<ImVec4> bbox_class_colors = {ImVec4(0.3f, 1.0f, 1.0f, 1.0f)};
    int current_bbox_class = 0;
    int current_bbox_id =
        0; // Track the currently selected bbox ID within the class
    bool show_bbox_ids = false; // Toggle for displaying bbox IDs on frame
    std::string new_class_name = "";

    // Helper function to create a new bbox class
    auto create_new_bbox_class = [&]() {
        std::string new_class_name =
            "Class_" + std::to_string(bbox_class_names.size() + 1);
        bbox_class_names.push_back(new_class_name);
        // Generate a unique color for the new class (HSV with different hues)
        float hue = (bbox_class_colors.size() *
                     0.618034f); // Golden ratio for nice color distribution
        while (hue > 1.0f)
            hue -= 1.0f;
        ImVec4 new_color = (ImVec4)ImColor::HSV(hue, 0.8f, 1.0f);
        bbox_class_colors.push_back(new_color);
        current_bbox_class = bbox_class_names.size() - 1;
    };

    // Helper function to cleanup YOLO drag boxes
    auto cleanup_yolo_drag_boxes = [&]() {
        for (int cam_id = 0; cam_id < MAX_VIEWS; cam_id++) {
            for (auto &drag_box : yolo_drag_boxes[cam_id]) {
                if (drag_box.rect) {
                    delete drag_box.rect;
                    drag_box.rect = nullptr;
                }
            }
            yolo_drag_boxes[cam_id].clear();
            yolo_active_bbox_idx[cam_id] = -1;
        }
    };

    std::string background_image_path = "";
    bool background_image_selected = false;
    GLuint background_texture = 0;
    int background_width = 0, background_height = 0;
    colors[ImPlotCol_Crosshairs] = ImVec4(0.3f, 0.10f, 0.64f, 1.00f);

    bool yolo_detection = false;
    int label_buffer_size = 64;
    bool show_help_window = false;
    std::vector<bool> is_view_focused;
    bool input_is_imgs = false;
    bool show_error = false;
    std::string error_message;

    int hovered_bbox_cam = -1;
    int hovered_bbox_idx = -1;
    float hovered_bbox_confidence = 0.0f;
    int hovered_bbox_class = -1;
    int hovered_bbox_id = -1; // Track the ID of the hovered bbox

    // Hovered OBB tracking variables
    int hovered_obb_cam = -1;
    int hovered_obb_idx = -1;
    float hovered_obb_confidence = 0.0f;
    int hovered_obb_class = -1;
    int hovered_obb_id = -1; // Track the ID of the hovered OBB

    bool auto_yolo_labeling = false;
    std::set<int>
        yolo_processed_frames; // Track which frames have been processed
    std::unordered_map<std::string, bool> window_was_decoding;
    double inst_speed = 1.0;
    float set_playback_speed = 1.0f;
    PlaybackState ps;
    LiveTable table;

    int brightness = 0;
    float contrast = 1.0f;     // neutral contrastst
    bool pivot_midgray = true; // typical contrast feel

    // variables for project management
    ProjectManager pm = ProjectManager();
    pm.project_root_path = red_data_dir;
    pm.media_folder = media_root_dir;

    ReprojectionTool rp_tool;

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            const char *path = argv[i];
            ProjectManager loaded;
            if (!load_project_manager_json(&loaded, path, &error_message)) {
                show_error = true;
            } else {
                pm = loaded;
                bool ok =
                    setup_project(pm, skeleton, skeleton_map, &error_message);
                if (ok) {
                    std::map<std::string, std::string> empty_selected_files;
                    load_videos(empty_selected_files, ps, pm,
                                window_was_decoding, demuxers, dc_context,
                                scene, label_buffer_size, decoder_threads,
                                is_view_focused);
                    std::string most_recent_folder;
                    if (!find_most_recent_labels(pm.keypoints_root_folder,
                                                 most_recent_folder,
                                                 error_message)) {
                        if (load_keypoints(most_recent_folder, keypoints_map,
                                           &skeleton, scene, pm.camera_names,
                                           error_message, bbox_class_names)) {
                            free_all_keypoints(keypoints_map, scene);
                            show_error = true;
                        }
                    }
                } else {
                    show_error = true;
                }
            }
        }
    }

    while (!glfwWindowShouldClose(window->render_target)) {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Reset hovered bbox info at start of each frame
        hovered_bbox_cam = -1;
        hovered_bbox_idx = -1;
        hovered_bbox_confidence = 0.0f;
        hovered_bbox_class = -1;
        hovered_bbox_id = -1;

        // Reset hovered OBB info at start of each frame
        hovered_obb_cam = -1;
        hovered_obb_idx = -1;
        hovered_obb_confidence = 0.0f;
        hovered_obb_class = -1;
        hovered_obb_id = -1;

        // --- Update playback time ---
        auto now = std::chrono::steady_clock::now();

        if (ps.play_video) {
            ps.accumulated_play_time +=
                std::chrono::duration<double>(now - ps.last_play_time_start)
                    .count() *
                set_playback_speed;
            ps.last_play_time_start = now;
        }
        double playback_time_now = ps.accumulated_play_time;

        if (ImGui::Begin("File Browser", NULL, ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    ImGui::BeginDisabled(ps.video_loaded);
                    if (ImGui::MenuItem("Open Video(s)")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 0;
                        config.path = pm.media_folder;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseMedia", "Choose Media", ".mp4", config);
                    }
                    ImGui::EndDisabled();
                    ImGui::BeginDisabled(!ps.video_loaded);
                    if (ImGui::MenuItem("Create Project")) {
                        pm.show_project_window = true;
                    }
                    ImGui::EndDisabled();
                    ImGui::BeginDisabled(ps.video_loaded);
                    if (ImGui::MenuItem("Load Project")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = pm.project_root_path;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseProject", "Choose Project File", ".redproj",
                            config);
                    }
                    ImGui::EndDisabled();
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Tools")) {
                    if (ImGui::MenuItem("Skeleton Creator")) {
                        show_skeleton_creator = true;
                    }
                    if (ImGui::MenuItem("YOLO Export Tool")) {
                        show_yolo_export_tool = true;
                    }
                    if (ImGui::MenuItem("Spreadsheet")) {
                        table.is_open = true;
                    }
                    if (ImGui::MenuItem("Reprojection Error")) {
                        rp_tool.show_reprojection_error = true;
                    }

                    ImGui::EndMenu();
                }

                ImGui::EndMenuBar();
            }
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);

            if (!ps.video_loaded) {
                {
                    const char *items[] = {"CPU Buffer", "GPU Buffer"};
                    static int item_current = 0;
                    ImGui::Combo("Buffer Type", &item_current, items,
                                 IM_ARRAYSIZE(items));
                    if (item_current == 0) {
                        scene->use_cpu_buffer = true;
                    } else {
                        scene->use_cpu_buffer = false;
                    }
                }

                ImGui::InputInt("Buffer Size", &label_buffer_size);
            }

            if (ps.video_loaded) {
                ImGui::InputInt("Seek Step", &dc_context->seek_interval, 10,
                                100);
                static int seek_accurate_frame_num = 0;
                ImGui::InputInt("Seek Accurate", &seek_accurate_frame_num, 1,
                                100);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    seek_all_cameras(scene, seek_accurate_frame_num,
                                     dc_context->video_fps, ps, true);
                }

                auto now_wall = std::chrono::steady_clock::now();
                double wall_seconds =
                    std::chrono::duration<double>(now_wall -
                                                  ps.last_wall_time_playspeed)
                        .count();
                int frame_delta =
                    current_frame_num - ps.last_frame_num_playspeed;
                if (wall_seconds > 0.5 && ps.play_video) {
                    inst_speed =
                        frame_delta / (dc_context->video_fps *
                                       wall_seconds); // Real-time normalized
                    ps.last_frame_num_playspeed = current_frame_num;
                    ps.last_wall_time_playspeed = now_wall;
                }

                // === Playback section ===
                ImGui::SeparatorText("Playback");

                // Two-column table: label | control
                if (ImGui::BeginTable("##playback_tbl", 2,
                                      ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_BordersInnerV)) {
                    ImGui::TableSetupColumn(
                        "Label", ImGuiTableColumnFlags_WidthFixed, 170.0f);
                    ImGui::TableSetupColumn("Control",
                                            ImGuiTableColumnFlags_WidthStretch);

                    // Row: FPS (read-only)
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Video FPS");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.1f", dc_context->video_fps);

                    // Row: Playback speed slider
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Set Playback Speed");
                    ImGui::SameLine();
                    HelpMarker("Range: 0.1x to 1.0x; affects render pacing.");

                    ImGui::TableSetColumnIndex(1);
                    ImGui::SliderFloat("##set_playback_speed",
                                       &set_playback_speed, 0.1f, 1.0f,
                                       "%.1fx");

                    // Row: Current speed readout
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Current Speed");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2fx", inst_speed);

                    ImGui::EndTable();
                }

                // Tip (wrapped, subtle)
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 24.0f);
                ImGui::Spacing();
                ImGui::TextDisabled(
                    "Tip: If playback is slower than real-time (< 1.0x), "
                    "collapse camera views to improve speed.");
                ImGui::PopTextWrapPos();

                // === Display section ===
                ImGui::SeparatorText("Display Controls");
                ImGui::BeginDisabled(ps.play_video); // Disable if playing video

                if (ImGui::BeginTable("##display_tbl", 2,
                                      ImGuiTableFlags_SizingStretchProp |
                                          ImGuiTableFlags_BordersInnerV)) {
                    ImGui::TableSetupColumn(
                        "Label", ImGuiTableColumnFlags_WidthFixed, 170.0f);
                    ImGui::TableSetupColumn("Control",
                                            ImGuiTableColumnFlags_WidthStretch);

                    // Contrast
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Contrast (alpha)");
                    ImGui::SameLine();
                    HelpMarker("1.00 = neutral. Increase to boost separation.");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SliderFloat("##contrast", &contrast, 0.0f, 3.0f,
                                       "%.2f");

                    // Brightness
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Brightness (beta)");
                    ImGui::SameLine();
                    HelpMarker("Shift pixel values. 0 = neutral.");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SliderInt("##brightness", &brightness, -150, 150);

                    // Reset row
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("Display Preset");
                    ImGui::TableSetColumnIndex(1);
                    if (ImGui::Button("Reset##display")) {
                        contrast = 1.0f;
                        brightness = 0;
                        pivot_midgray = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(restores neutral)");

                    ImGui::EndTable();
                }
                ImGui::EndDisabled();
            }
        }
        ImGui::End();

        DrawProjectWindow(pm, skeleton_map, skeleton, skeleton_dir, show_error,
                          error_message);

        if (ImGuiFileDialog::Instance()->Display("ChooseProjectDir")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::filesystem::path chosen;

                auto sel = ImGuiFileDialog::Instance()->GetSelection();
                if (!sel.empty()) {
                    chosen = std::filesystem::path(sel.begin()->second);
                    if (std::filesystem::is_regular_file(chosen)) {
                        chosen = chosen.parent_path();
                    }
                } else {
                    chosen = std::filesystem::path(
                        ImGuiFileDialog::Instance()->GetCurrentPath());
                }
                pm.project_root_path = chosen.string();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseCalibration")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::filesystem::path chosen;
                auto sel = ImGuiFileDialog::Instance()->GetSelection();
                if (!sel.empty()) {
                    chosen = std::filesystem::path(sel.begin()->second);
                    if (std::filesystem::is_regular_file(chosen)) {
                        chosen = chosen.parent_path();
                    }
                } else {
                    chosen = std::filesystem::path(
                        ImGuiFileDialog::Instance()->GetCurrentPath());
                }
                pm.calibration_folder = chosen.string();
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseMedia")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto selected_files =
                    ImGuiFileDialog::Instance()->GetSelection();
                pm.media_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
                pm.project_name =
                    dir_difference(pm.media_folder, media_root_dir);
                pm.media_folder = pm.media_folder;
                load_videos(selected_files, ps, pm, window_was_decoding,
                            demuxers, dc_context, scene, label_buffer_size,
                            decoder_threads, is_view_focused);
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseProject")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                const auto sel = ImGuiFileDialog::Instance()->GetSelection();
                // Choose the picked file (single-select assumed)
                std::filesystem::path cfg_path;
                if (!sel.empty()) {
                    cfg_path =
                        std::filesystem::path(sel.begin()->second); // full path
                } else {
                    std::string full =
                        ImGuiFileDialog::Instance()->GetFilePathName(
                            IGFD_ResultMode_KeepInputFile);
                    if (!full.empty())
                        cfg_path = std::filesystem::path(full);
                    else
                        cfg_path = std::filesystem::path(
                            ImGuiFileDialog::Instance()->GetCurrentPath());
                }
                ProjectManager loaded;
                if (!load_project_manager_json(&loaded, cfg_path,
                                               &error_message)) {
                    show_error = true;
                } else {
                    pm = loaded;
                    bool ok = setup_project(pm, skeleton, skeleton_map,
                                            &error_message);
                    if (ok) {
                        std::map<std::string, std::string> empty_selected_files;
                        load_videos(empty_selected_files, ps, pm,
                                    window_was_decoding, demuxers, dc_context,
                                    scene, label_buffer_size, decoder_threads,
                                    is_view_focused);

                        std::string most_recent_folder;
                        if (!find_most_recent_labels(pm.keypoints_root_folder,
                                                     most_recent_folder,
                                                     error_message)) {
                            if (load_keypoints(most_recent_folder,
                                               keypoints_map, &skeleton, scene,
                                               pm.camera_names, error_message,
                                               bbox_class_names)) {
                                free_all_keypoints(keypoints_map, scene);
                                show_error = true;
                            }
                        }

                    } else {
                        show_error = true;
                    }
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseSkeleton")) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                pm.skeleton_file =
                    ImGuiFileDialog::Instance()->GetFilePathName();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        // Handle background image selection for skeleton creator
        if (ImGuiFileDialog::Instance()->Display("ChooseBackgroundImage")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto file_selection =
                    ImGuiFileDialog::Instance()->GetSelection();
                if (!file_selection.empty()) {
                    background_image_path = file_selection.begin()->second;
                    background_image_selected = true;

                    // Load the background image texture
                    if (background_texture != 0) {
                        glDeleteTextures(1, &background_texture);
                        background_texture = 0;
                    }

                    // Load image using stb_image
                    int channels;
                    unsigned char *image_data = stbi_load(
                        background_image_path.c_str(), &background_width,
                        &background_height, &channels, 0);
                    if (image_data) {
                        glGenTextures(1, &background_texture);
                        glBindTexture(GL_TEXTURE_2D, background_texture);

                        GLenum format = GL_RGB;
                        if (channels == 4)
                            format = GL_RGBA;
                        else if (channels == 1)
                            format = GL_RED;

                        glTexImage2D(GL_TEXTURE_2D, 0, format, background_width,
                                     background_height, 0, format,
                                     GL_UNSIGNED_BYTE, image_data);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                                        GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                                        GL_LINEAR);

                        stbi_image_free(image_data);
                    }
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        // Handle skeleton load dialog for editing
        if (ImGuiFileDialog::Instance()->Display("LoadSkeletonForEdit")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string file_path =
                    ImGuiFileDialog::Instance()->GetFilePathName();

                std::ifstream file(file_path);
                if (file.is_open()) {
                    nlohmann::json skeleton_json;
                    file >> skeleton_json;
                    file.close();

                    // Clear existing data
                    creator_nodes.clear();
                    creator_edges.clear();
                    selected_node_for_edge = -1;
                    next_node_id = 0;

                    // Load skeleton data
                    if (skeleton_json.contains("name")) {
                        skeleton_creator_name = skeleton_json["name"];
                    }

                    if (skeleton_json.contains("has_bbox")) {
                        skeleton_creator_has_bbox = skeleton_json["has_bbox"];
                    }

                    if (skeleton_json.contains("node_names") &&
                        skeleton_json.contains("node_positions")) {
                        std::vector<std::string> node_names =
                            skeleton_json["node_names"];
                        std::vector<std::vector<double>> node_positions =
                            skeleton_json["node_positions"];

                        for (size_t i = 0;
                             i < node_names.size() && i < node_positions.size();
                             i++) {
                            if (node_positions[i].size() >= 2) {
                                SkeletonCreatorNode node;
                                node.id = next_node_id++;
                                node.name = node_names[i];
                                node.position = ImPlotPoint(
                                    node_positions[i][0], node_positions[i][1]);
                                node.color = (ImVec4)ImColor::HSV(
                                    node.id / 10.0f, 1.0f, 1.0f);
                                creator_nodes.push_back(node);
                            }
                        }
                    } else if (skeleton_json.contains("node_names")) {
                        // Fallback for skeletons without saved positions
                        std::vector<std::string> node_names =
                            skeleton_json["node_names"];
                        double spacing = 0.8 / (node_names.size() + 1);

                        for (size_t i = 0; i < node_names.size(); i++) {
                            SkeletonCreatorNode node;
                            node.id = next_node_id++;
                            node.name = node_names[i];
                            node.position =
                                ImPlotPoint(0.1 + spacing * (i + 1), 0.5);
                            node.color = (ImVec4)ImColor::HSV(node.id / 10.0f,
                                                              1.0f, 1.0f);
                            creator_nodes.push_back(node);
                        }
                    }

                    if (skeleton_json.contains("edges")) {
                        std::vector<std::vector<int>> edges_array =
                            skeleton_json["edges"];
                        for (const auto &edge : edges_array) {
                            if (edge.size() >= 2 &&
                                edge[0] < creator_nodes.size() &&
                                edge[1] < creator_nodes.size()) {
                                SkeletonCreatorEdge creator_edge;
                                creator_edge.node1_id =
                                    creator_nodes[edge[0]].id;
                                creator_edge.node2_id =
                                    creator_nodes[edge[1]].id;
                                creator_edges.push_back(creator_edge);
                            }
                        }
                    }

                    std::cout << "Skeleton loaded from: " << file_path
                              << " (with " << creator_nodes.size() << " nodes)"
                              << std::endl;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        static int select_corr_head = 0;
        if (ps.video_loaded && (!ps.play_video)) {
            int visible_idx = 0;
            if (!ps.pause_seeked) {
                for (int i = 0; i < scene->num_cams; i++) {
                    if (window_was_decoding[pm.camera_names[i]]) {
                        visible_idx = i;
                        break;
                    }
                }
            }

            ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frames in the buffer")) {

                for (u32 i = 0; i < scene->size_of_buffer; i++) {
                    int seletable_frame_id =
                        (i + ps.read_head) % scene->size_of_buffer;
                    char label[32];
                    if (input_is_imgs) {
                        snprintf(label, sizeof(label), "%d: %s",
                                 scene
                                     ->display_buffer[visible_idx]
                                                     [seletable_frame_id]
                                     .frame_number,
                                 imgs_names[i].c_str());
                    } else {
                        sprintf(label, "Frame %d",
                                scene
                                    ->display_buffer[visible_idx]
                                                    [seletable_frame_id]
                                    .frame_number);
                    }
                    if (ImGui::Selectable(label, ps.pause_selected == i)) {
                        // start from the lowest frame
                        ps.pause_selected = i;
                    }
                }

                if (ImGui::IsKeyPressed(ImGuiKey_Comma, true) &&
                    !io.WantTextInput) {
                    if (ps.pause_selected > 0) {
                        ps.pause_selected--;
                    }
                };

                if (ImGui::IsKeyPressed(ImGuiKey_Period, true) &&
                    !io.WantTextInput) {
                    if (ps.pause_selected < (scene->size_of_buffer - 1)) {
                        ps.pause_selected++;
                    }
                };
            }
            ImGui::End();
            select_corr_head =
                (ps.pause_selected + ps.read_head) % scene->size_of_buffer;
            current_frame_num =
                scene->display_buffer[visible_idx][select_corr_head]
                    .frame_number;

            // Automatic YOLO detection for current frame
            if (auto_yolo_labeling && !yolo_model_path.empty() &&
                skeleton.has_bbox) {
                if (!frameHasYoloDetections(current_frame_num, keypoints_map,
                                            &skeleton) &&
                    yolo_processed_frames.find(current_frame_num) ==
                        yolo_processed_frames.end()) {

                    // Mark frame as processed to avoid duplicate processing
                    yolo_processed_frames.insert(current_frame_num);

                    // Enable YOLO detection flag
                    yolo_detection = true;

                    std::cout << "Auto YOLO: Processing frame "
                              << current_frame_num << std::endl;

                    // Run YOLO inference on all cameras for this frame
                    for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                        if (ps.pause_seeked) {
                            unsigned char *frame_data =
                                scene->display_buffer[cam_id][select_corr_head]
                                    .frame;

                            if (frame_data) {
                                yolo_predictions[cam_id] = runYoloInference(
                                    yolo_model_path, frame_data,
                                    scene->image_width[cam_id],
                                    scene->image_height[cam_id]);

                                yolo_bboxes[cam_id].clear();
                                for (const auto &pred :
                                     yolo_predictions[cam_id]) {
                                    yolo_bboxes[cam_id].emplace_back(pred);
                                }
                            }
                        } else {
                            if (window_was_decoding[pm.camera_names[cam_id]]) {
                                unsigned char *frame_data =
                                    scene
                                        ->display_buffer[cam_id]
                                                        [select_corr_head]
                                        .frame;

                                if (frame_data) {
                                    yolo_predictions[cam_id] = runYoloInference(
                                        yolo_model_path, frame_data,
                                        scene->image_width[cam_id],
                                        scene->image_height[cam_id]);

                                    yolo_bboxes[cam_id].clear();
                                    for (const auto &pred :
                                         yolo_predictions[cam_id]) {
                                        yolo_bboxes[cam_id].emplace_back(pred);
                                    }
                                }
                            }
                        }
                    }

                    // Add YOLO detections to main bounding box system
                    if (!yolo_bboxes.empty() &&
                        std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(),
                                    [](const auto &cam_bboxes) {
                                        return !cam_bboxes.empty();
                                    })) {
                        for (int cam_id = 0; cam_id < scene->num_cams;
                             cam_id++) {
                            if (!yolo_bboxes[cam_id].empty()) {
                                // Ensure keypoints structure exists
                                bool keypoints_find =
                                    keypoints_map.find(current_frame_num) !=
                                    keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       &skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }

                                // Add YOLO detections to main bounding box
                                // system
                                int yolo_bbox_id = 0;
                                for (const auto &yolo_bbox :
                                     yolo_bboxes[cam_id]) {
                                    if (yolo_bbox.is_valid) {
                                        while (yolo_bbox.class_id >=
                                               bbox_class_colors.size()) {
                                            create_new_bbox_class();
                                        }

                                        BoundingBox bbox;

                                        bbox.rect = new ImPlotRect(
                                            yolo_bbox.x_min, yolo_bbox.x_max,
                                            yolo_bbox.y_min, yolo_bbox.y_max);

                                        bbox.state = RectTwoPoints;
                                        bbox.class_id = yolo_bbox.class_id;
                                        bbox.id = yolo_bbox_id++;
                                        bbox.confidence = yolo_bbox.confidence;
                                        bbox.has_bbox_keypoints = false;
                                        bbox.bbox_keypoints2d = nullptr;
                                        bbox.active_kp_id = nullptr;

                                        if (skeleton.has_bbox &&
                                            skeleton.has_skeleton &&
                                            skeleton.num_nodes > 0) {
                                            allocate_bbox_keypoints(
                                                &bbox, scene, &skeleton);
                                        }

                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[cam_id]
                                            .push_back(bbox);
                                    }
                                }

                                std::cout << "Auto YOLO: Added "
                                          << yolo_bboxes[cam_id].size()
                                          << " detections for camera " << cam_id
                                          << ", frame " << current_frame_num
                                          << std::endl;
                            }
                        }
                    }
                }
            }
        }

        // Render a video frame
        if (ps.video_loaded) {
            for (int j = 0; j < scene->num_cams; j++) {
                const std::string &win_name = pm.camera_names[j];

                // layout
                ImGui::SetNextWindowSize(ImVec2(500, 400),
                                         ImGuiCond_FirstUseEver);
                ImVec2 window_pos;

                if (scene->num_cams < 8) {
                    if (j % 2 == 0) {
                        window_pos.y = 200.0;
                        window_pos.x = (j / 2.0) * 500;
                    } else {
                        window_pos.y = 600.0;
                        window_pos.x = (j - 1) / 2.0 * 500;
                    }
                } else {
                    int row = j % 4;
                    float base_x = (row == 0) ? j : (j - 1);
                    float x_group = (base_x / 4.0f) * 500;
                    switch (row) {
                    case 0:
                        window_pos = {x_group, 200.0f};
                        break;
                    case 1:
                        window_pos = {x_group, 600.0f};
                        break;
                    case 2:
                        window_pos = {x_group, 1000.0f};
                        break;
                    case 3:
                        window_pos = {x_group, 1400.0f};
                        break;
                    }
                }

                ImGui::SetNextWindowPos(window_pos, ImGuiCond_FirstUseEver);
                bool is_visible = ImGui::Begin(win_name.c_str());

                if (!window_was_decoding[win_name] && is_visible &&
                    ps.play_video) {
                    // seek if visibility has changed
                    seek_all_cameras(scene, current_frame_num,
                                     dc_context->video_fps, ps, true);
                }

                if (!window_was_decoding[win_name] && is_visible &&
                    !ps.play_video && !ps.pause_seeked) {
                    // seek if visibility has changed
                    seek_all_cameras(scene, current_frame_num,
                                     dc_context->video_fps, ps, true);
                    for (auto &[key, value] : window_need_decoding) {
                        value.store(true);
                    }
                }

                if (ps.play_video) {
                    window_need_decoding[win_name].store(is_visible);
                };

                if (is_visible) {
                    if (ps.play_video) {
                        // if the current frame is ready, upload for
                        // display, otherwise wait for the frame to get
                        // ready while
                        // (scene->display_buffer[j][ps.read_head]
                        //            .frame_number !=
                        //        ps.to_display_frame_number) {
                        //     std::cout
                        //         << win_name << " , read head: " <<
                        //         ps.read_head
                        //         << ", frame_number: "
                        //         << scene->display_buffer[j][ps.read_head]
                        //                .frame_number
                        //         << ", to_display_frame_number: "
                        //         << ps.to_display_frame_number <<
                        //         std::endl;
                        //     std::this_thread::sleep_for(
                        //         std::chrono::milliseconds(1));
                        // }

                        current_frame_num = ps.to_display_frame_number;
                        if (scene->use_cpu_buffer) {
                            // upload_texture(&scene->image_texture[j],
                            // scene->display_buffer[j][read_head].frame,
                            // scene->image_width[j],
                            // scene->image_height[j]);
                            // // 2x slower than pbo copy frame to cuda
                            // buffer
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][ps.read_head].frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyHostToDevice));
                        } else {
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][ps.read_head].frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyDeviceToDevice));
                        }
                    } else {
                        if (scene->use_cpu_buffer) {
                            // upload_texture(&scene->image_texture[j],
                            // scene->display_buffer[j][select_corr_head].frame,
                            // scene->image_width[j],
                            // scene->image_height[j]);
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][select_corr_head]
                                    .frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyHostToDevice));

                            // // brighten in-place on GPU using npp
                            // NppiSize roi = {
                            //     static_cast<int>(scene->image_width[j]),
                            //     static_cast<int>(scene->image_height[j])};
                            // Npp8u *d_img =
                            //     (Npp8u *)scene->pbo_cuda[j].cuda_buffer;
                            // int step = scene->image_width[j] *
                            //            4; // RGBA = 4 bytes per pixel

                            // Npp8u addC[3] = {(Npp8u)brightness,
                            //                  (Npp8u)brightness,
                            //                  (Npp8u)brightness};

                            // nppiAddC_8u_AC4IRSfs(addC, d_img, step, roi,
                            //                      0); // in-place RGBA

                            apply_contrast_brightness_rgba(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->image_width[j], scene->image_height[j],
                                contrast,          // e.g. 1.0f default
                                (float)brightness, // from ImGui slider
                                pivot_midgray, // pivot around mid-gray (128)
                                0);

                        } else {
                            ck(cudaMemcpy(
                                scene->pbo_cuda[j].cuda_buffer,
                                scene->display_buffer[j][select_corr_head]
                                    .frame,
                                scene->image_width[j] * scene->image_height[j] *
                                    4,
                                cudaMemcpyDeviceToDevice));
                        }
                    }
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width[j],
                                                scene->image_height[j]);
                    unbind_pbo();
                    unbind_texture();

                    ImGui::BeginGroup();
                    std::string scene_name = "scene view" + std::to_string(j);
                    ImGui::BeginChild(
                        scene_name.c_str(),
                        ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));
                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    // ImGui::Image((void*)(intptr_t)image_texture[j],
                    // avail_size);
                    //
                    if (pm.plot_keypoints_flag) {
                        if (keypoints_map.find(current_frame_num) ==
                            keypoints_map.end()) {
                            keypoints_find = false;
                        } else {
                            keypoints_find = true;
                        }
                    }

                    if (ImPlot::BeginPlot("##no_plot_name", avail_size,
                                          ImPlotFlags_Equal |
                                              ImPlotAxisFlags_AutoFit |
                                              ImPlotFlags_Crosshairs)) {
                        ImPlot::PlotImage(
                            "##no_image_name",
                            (ImTextureID)(intptr_t)scene->image_texture[j],
                            ImVec2(0, 0),
                            ImVec2(scene->image_width[j],
                                   scene->image_height[j]));

                        if (pm.plot_keypoints_flag) {
                            // plot arena for testing camera parameters
                            // gui_plot_perimeter(&camera_params[j],
                            // scene->image_height[j]); if (scene->num_cams
                            // > 1)
                            // {
                            //     gui_plot_world_coordinates(&camera_params[j],
                            //     j, scene->image_height[j]);
                            // }

                            // labeling
                            if (ImPlot::IsPlotHovered()) {
                                is_view_focused[j] = true;
                                if (ImGui::IsKeyPressed(ImGuiKey_B, false) &&
                                    !io.WantTextInput) {
                                    // create keypoints
                                    if (!keypoints_find) {
                                        // not found
                                        KeyPoints *keypoints =
                                            (KeyPoints *)malloc(
                                                sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene,
                                                           &skeleton);
                                        keypoints_map[current_frame_num] =
                                            keypoints;
                                    }
                                }

                                if (keypoints_find && skeleton.has_skeleton &&
                                    !skeleton.has_bbox) {
                                    u32 *kp = &(keypoints_map[current_frame_num]
                                                    ->active_id[j]);
                                    if (ImGui::IsKeyPressed(ImGuiKey_W,
                                                            false) &&
                                        !io.WantTextInput) {
                                        // labeling sequentially each view
                                        ImPlotPoint mouse =
                                            ImPlot::GetPlotMousePos();
                                        keypoints_map[current_frame_num]
                                            ->kp2d[j][*kp]
                                            .position = {mouse.x, mouse.y};
                                        keypoints_map[current_frame_num]
                                            ->kp2d[j][*kp]
                                            .is_labeled = true;
                                        if (*kp < (skeleton.num_nodes - 1)) {
                                            (*kp)++;
                                        }
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_A, true) &&
                                        !io.WantTextInput) {
                                        if (*kp <= 0) {
                                            *kp = 0;
                                        } else
                                            (*kp)--;
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_D, true) &&
                                        !io.WantTextInput) {
                                        if (*kp >= skeleton.num_nodes - 1) {
                                            *kp = skeleton.num_nodes - 1;
                                        } else
                                            (*kp)++;
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_E,
                                                            false) &&
                                        !io.WantTextInput) {
                                        *kp = skeleton.num_nodes - 1;
                                    }

                                    if (ImGui::IsKeyPressed(ImGuiKey_Q,
                                                            false) &&
                                        !io.WantTextInput) {
                                        *kp = 0;
                                    }

                                    // delete all keypoint on a frame
                                    if (ImGui::IsKeyPressed(ImGuiKey_Backspace,
                                                            false) &&
                                        !io.WantTextInput) {
                                        free_keypoints(
                                            keypoints_map[current_frame_num],
                                            scene);
                                        keypoints_map.erase(current_frame_num);
                                        keypoints_find = false;
                                    }
                                }
                                if (skeleton.has_bbox) {
                                    static bool shift_was_pressed = false;
                                    bool shift_pressed =
                                        ImGui::GetIO().KeyShift;

                                    bool keypoints_find =
                                        keypoints_map.find(current_frame_num) !=
                                        keypoints_map.end();
                                    if (!keypoints_find) {
                                        KeyPoints *keypoints =
                                            (KeyPoints *)malloc(
                                                sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene,
                                                           &skeleton);
                                        keypoints_map[current_frame_num] =
                                            keypoints;
                                    }

                                    if (shift_pressed && !shift_was_pressed) {
                                        ImPlotPoint mouse =
                                            ImPlot::GetPlotMousePos();

                                        // Clamp mouse coordinates to frame
                                        // bounds
                                        double clamped_x = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_width[j],
                                                mouse.x));
                                        double clamped_y = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_height[j],
                                                mouse.y));

                                        // Delete existing bbox with the
                                        // same class_id and id
                                        if (keypoints_map.find(
                                                current_frame_num) !=
                                            keypoints_map.end()) {
                                            auto &bbox_list =
                                                keypoints_map[current_frame_num]
                                                    ->bbox2d_list[j];

                                            // Find and remove bbox with
                                            // same class_id and id
                                            bbox_list.erase(
                                                std::remove_if(
                                                    bbox_list.begin(),
                                                    bbox_list.end(),
                                                    [current_bbox_class,
                                                     current_bbox_id](
                                                        const BoundingBox
                                                            &bbox) {
                                                        return bbox.class_id ==
                                                                   current_bbox_class &&
                                                               bbox.id ==
                                                                   current_bbox_id;
                                                    }),
                                                bbox_list.end());
                                        }

                                        BoundingBox new_bbox;
                                        new_bbox.rect = new ImPlotRect(
                                            clamped_x, clamped_x, clamped_y,
                                            clamped_y);
                                        new_bbox.state = RectOnePoint;
                                        new_bbox.class_id =
                                            current_bbox_class; // Use
                                                                // currently
                                                                // selected
                                                                // class
                                        new_bbox.id =
                                            current_bbox_id; // Set bbox ID
                                        new_bbox.confidence = 1.0f;
                                        new_bbox.has_bbox_keypoints = false;
                                        new_bbox.bbox_keypoints2d = nullptr;
                                        new_bbox.active_kp_id = nullptr;

                                        // Allocate keypoints for this
                                        // bounding box only if skeleton has
                                        // both bbox and skeleton
                                        if (skeleton.has_bbox &&
                                            skeleton.has_skeleton &&
                                            skeleton.num_nodes > 0) {
                                            allocate_bbox_keypoints(
                                                &new_bbox, scene, &skeleton);
                                        }

                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[j]
                                            .push_back(new_bbox);
                                    }

                                    // Only process bbox operations if
                                    // keypoints exist for this frame
                                    if (keypoints_map.find(current_frame_num) !=
                                        keypoints_map.end()) {
                                        for (auto &bbox :
                                             keypoints_map[current_frame_num]
                                                 ->bbox2d_list[j]) {
                                            if (bbox.state == RectOnePoint &&
                                                shift_pressed) {
                                                ImPlotPoint mouse =
                                                    ImPlot::GetPlotMousePos();
                                                // Clamp mouse coordinates
                                                // to frame bounds
                                                double clamped_x = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_width[j],
                                                        mouse.x));
                                                double clamped_y = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_height[j],
                                                        mouse.y));
                                                bbox.rect->X.Max = clamped_x;
                                                bbox.rect->Y.Max = clamped_y;
                                            }

                                            if (bbox.state == RectOnePoint &&
                                                !shift_pressed &&
                                                shift_was_pressed) {
                                                bbox.state = RectTwoPoints;

                                                double x_min =
                                                    std::min(bbox.rect->X.Min,
                                                             bbox.rect->X.Max);
                                                double x_max =
                                                    std::max(bbox.rect->X.Min,
                                                             bbox.rect->X.Max);
                                                double y_min =
                                                    std::min(bbox.rect->Y.Min,
                                                             bbox.rect->Y.Max);
                                                double y_max =
                                                    std::max(bbox.rect->Y.Min,
                                                             bbox.rect->Y.Max);

                                                bbox.rect->X.Min = x_min;
                                                bbox.rect->X.Max = x_max;
                                                bbox.rect->Y.Min = y_min;
                                                bbox.rect->Y.Max = y_max;

                                                // Auto-increment bbox ID
                                                // after finishing drawing
                                                current_bbox_id++;
                                            }
                                        }
                                    }

                                    shift_was_pressed = shift_pressed;
                                }
                            } else {
                                is_view_focused[j] = false;
                            }

                            // Plot bounding boxes (both with and without
                            // keypoints)
                            if (skeleton.has_bbox) {
                                bool keypoints_find =
                                    keypoints_map.find(current_frame_num) !=
                                    keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       &skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }

                                // Determine which bbox is active (under
                                // mouse cursor)
                                ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                int active_bbox_idx = -1;

                                // Check which bbox the mouse is hovering
                                // over
                                for (int bbox_idx = 0;
                                     bbox_idx < keypoints_map[current_frame_num]
                                                    ->bbox2d_list[j]
                                                    .size();
                                     bbox_idx++) {
                                    BoundingBox &bbox =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[j][bbox_idx];
                                    if (bbox.rect &&
                                        bbox.state == RectTwoPoints &&
                                        is_point_in_bbox(mouse.x, mouse.y,
                                                         bbox.rect)) {
                                        active_bbox_idx = bbox_idx;
                                        break;
                                    }
                                }

                                // Update global active bbox tracking for
                                // this camera
                                if (j < user_active_bbox_idx.size()) {
                                    user_active_bbox_idx[j] = active_bbox_idx;
                                }

                                // Plot multiple bounding boxes
                                for (int bbox_idx = 0;
                                     bbox_idx < keypoints_map[current_frame_num]
                                                    ->bbox2d_list[j]
                                                    .size();
                                     bbox_idx++) {
                                    BoundingBox &bbox =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[j][bbox_idx];
                                    if (bbox.rect) {
                                        // Get color based on class_id
                                        ImVec4 bbox_color =
                                            ImVec4(0.5f, 1.0f, 1.0f,
                                                   1.0f); // Default fallback
                                        if (bbox.class_id >= 0 &&
                                            bbox.class_id <
                                                bbox_class_colors.size()) {
                                            bbox_color = bbox_class_colors
                                                [bbox.class_id];
                                        }

                                        // Reduce opacity for inactive
                                        // bboxes
                                        bool is_active_bbox =
                                            (bbox_idx == active_bbox_idx);
                                        if (!is_active_bbox) {
                                            bbox_color.w =
                                                0.6f; // Make inactive
                                                      // bboxes more
                                                      // transparent
                                        }

                                        // Draw completed bounding boxes
                                        if (bbox.state == RectTwoPoints) {
                                            bool bbox_clicked = false,
                                                 bbox_hovered = false,
                                                 bbox_held = false;

                                            // Store previous rect for this
                                            // specific bbox
                                            static std::map<std::pair<int, int>,
                                                            ImPlotRect>
                                                bbox_prev_rects; // frame ->
                                                                 // {camera,
                                                                 // bbox_idx}
                                                                 // ->
                                                                 // prev_rect
                                            auto bbox_key =
                                                std::make_pair(j, bbox_idx);

                                            // Initialize previous rect if
                                            // not exists
                                            if (bbox_prev_rects.find(
                                                    bbox_key) ==
                                                bbox_prev_rects.end()) {
                                                bbox_prev_rects[bbox_key] =
                                                    *bbox.rect;
                                            }

                                            ImPlotRect prev_bbox_rect =
                                                bbox_prev_rects[bbox_key];

                                            // Only allow interaction if
                                            // this is the active bbox or no
                                            // bbox is active
                                            ImPlotDragToolFlags drag_flags =
                                                ImPlotDragToolFlags_None;
                                            if (!is_active_bbox &&
                                                active_bbox_idx != -1) {
                                                drag_flags =
                                                    ImPlotDragToolFlags_NoInputs;
                                            }

                                            bool bbox_modified = MyDragRect(
                                                1000 + bbox_idx,
                                                &bbox.rect->X.Min,
                                                &bbox.rect->Y.Min,
                                                &bbox.rect->X.Max,
                                                &bbox.rect->Y.Max, bbox_color,
                                                drag_flags, &bbox_clicked,
                                                &bbox_hovered, &bbox_held);

                                            // Clamp bbox coordinates to
                                            // frame bounds after any
                                            // modification
                                            if (bbox_modified || bbox_held) {
                                                bbox.rect->X.Min = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_width[j],
                                                        bbox.rect->X.Min));
                                                bbox.rect->Y.Min = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_height[j],
                                                        bbox.rect->Y.Min));
                                                bbox.rect->X.Max = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_width[j],
                                                        bbox.rect->X.Max));
                                                bbox.rect->Y.Max = std::max(
                                                    0.0,
                                                    std::min(
                                                        (double)scene
                                                            ->image_height[j],
                                                        bbox.rect->Y.Max));
                                            }

                                            // Display bbox ID on frame if
                                            // enabled
                                            if (show_bbox_ids) {
                                                // Position text above
                                                // top-right corner of bbox
                                                double text_x =
                                                    bbox.rect->X.Max -
                                                    10.0; // Offset to the
                                                          // left
                                                double text_y =
                                                    bbox.rect->Y.Max -
                                                    10.0; // Offset above
                                                          // the box
                                                ImPlot::PushStyleColor(
                                                    ImPlotCol_InlayText,
                                                    ImVec4(1.0f, 1.0f, 1.0f,
                                                           1.0f));
                                                ImPlot::PlotText(
                                                    std::to_string(bbox.id)
                                                        .c_str(),
                                                    text_x, text_y);
                                            }

                                            if (bbox_clicked || bbox_held ||
                                                bbox_modified) {
                                                active_bbox_idx = bbox_idx;
                                            }

                                            bool is_active_bbox =
                                                (bbox_idx == active_bbox_idx);

                                            // Scale bbox keypoints if bbox
                                            // was resized and this is the
                                            // active bbox
                                            if (bbox_modified &&
                                                bbox.has_bbox_keypoints &&
                                                bbox.bbox_keypoints2d &&
                                                is_active_bbox) {
                                                scale_bbox_keypoints(
                                                    &bbox, scene, &skeleton,
                                                    &prev_bbox_rect, bbox.rect);
                                                bbox_prev_rects[bbox_key] =
                                                    *bbox.rect; // Update
                                                                // stored
                                                                // previous
                                                                // rect
                                            }

                                            // Update previous rect when not
                                            // being dragged
                                            if (!bbox_held) {
                                                bbox_prev_rects[bbox_key] =
                                                    *bbox.rect;
                                            }

                                            // Handle keyboard shortcuts
                                            // when hovering over bounding
                                            // box
                                            if (bbox_hovered) {
                                                // Update hovered bbox info
                                                // for display
                                                hovered_bbox_cam = j;
                                                hovered_bbox_idx = bbox_idx;
                                                hovered_bbox_confidence =
                                                    bbox.confidence;
                                                hovered_bbox_class =
                                                    bbox.class_id;
                                                hovered_bbox_id = bbox.id;

                                                // Delete bounding box from
                                                // current camera when 'T'
                                                // key is pressed while
                                                // hovering
                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_F, false) &&
                                                    !io.WantTextInput) {
                                                    // Clean up bbox
                                                    // keypoints before
                                                    // deletion
                                                    if (bbox.has_bbox_keypoints &&
                                                        bbox.bbox_keypoints2d) {
                                                        free(
                                                            bbox.bbox_keypoints2d);
                                                        bbox.bbox_keypoints2d =
                                                            nullptr;
                                                        free(bbox.active_kp_id);
                                                        bbox.active_kp_id =
                                                            nullptr;
                                                    }
                                                    // Mark for deletion by
                                                    // setting state to
                                                    // RectNull
                                                    delete bbox.rect;
                                                    bbox.rect = nullptr;
                                                    bbox.state = RectNull;
                                                    bbox.has_bbox_keypoints =
                                                        false;
                                                }

                                                // Delete bounding box from
                                                // all cameras when 'O' key
                                                // is pressed while hovering
                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_O, false) &&
                                                    !io.WantTextInput) {
                                                    // Find this bbox's
                                                    // class_id and delete
                                                    // all bboxes with same
                                                    // class from all
                                                    // cameras
                                                    int target_class_id =
                                                        bbox.class_id;
                                                    for (int cam_idx = 0;
                                                         cam_idx <
                                                         scene->num_cams;
                                                         cam_idx++) {
                                                        auto &bbox_list =
                                                            keypoints_map
                                                                [current_frame_num]
                                                                    ->bbox2d_list
                                                                        [cam_idx];
                                                        for (auto &other_bbox :
                                                             bbox_list) {
                                                            if (other_bbox
                                                                        .class_id ==
                                                                    target_class_id &&
                                                                other_bbox
                                                                        .state !=
                                                                    RectNull &&
                                                                other_bbox
                                                                        .rect !=
                                                                    nullptr) {
                                                                // Clean up
                                                                // bbox
                                                                // keypoints
                                                                // before
                                                                // deletion
                                                                if (other_bbox
                                                                        .has_bbox_keypoints &&
                                                                    other_bbox
                                                                        .bbox_keypoints2d) {
                                                                    free(
                                                                        other_bbox
                                                                            .bbox_keypoints2d);
                                                                    other_bbox
                                                                        .bbox_keypoints2d =
                                                                        nullptr;
                                                                    free(
                                                                        other_bbox
                                                                            .active_kp_id);
                                                                    other_bbox
                                                                        .active_kp_id =
                                                                        nullptr;
                                                                }
                                                                delete other_bbox
                                                                    .rect;
                                                                other_bbox
                                                                    .rect =
                                                                    nullptr;
                                                                other_bbox
                                                                    .state =
                                                                    RectNull;
                                                                other_bbox
                                                                    .has_bbox_keypoints =
                                                                    false;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        // Draw bounding boxes being created
                                        // (one point set)
                                        else if (bbox.state == RectOnePoint) {
                                            // Draw a preview rectangle
                                            // while dragging
                                            double xs[5] = {bbox.rect->X.Min,
                                                            bbox.rect->X.Max,
                                                            bbox.rect->X.Max,
                                                            bbox.rect->X.Min,
                                                            bbox.rect->X.Min};
                                            double ys[5] = {bbox.rect->Y.Max,
                                                            bbox.rect->Y.Max,
                                                            bbox.rect->Y.Min,
                                                            bbox.rect->Y.Min,
                                                            bbox.rect->Y.Max};
                                            ImPlot::SetNextLineStyle(bbox_color,
                                                                     2.0f);
                                            ImPlot::PlotLine("##bbox_preview",
                                                             xs, ys, 5);
                                        }

                                        // Plot keypoints within this
                                        // bounding box (only if keypoints
                                        // are enabled and skeleton has
                                        // keypoints)
                                        if (bbox.state == RectTwoPoints &&
                                            keypoints_find &&
                                            skeleton.has_skeleton &&
                                            bbox.has_bbox_keypoints) {
                                            bool is_saved = true;
                                            // Only allow interaction with
                                            // keypoints if this is the
                                            // active bbox
                                            gui_plot_bbox_keypoints(
                                                &bbox, &skeleton, j,
                                                scene->num_cams, is_active_bbox,
                                                is_saved, bbox_idx);

                                            // Handle keypoint labeling with
                                            // W key for bounding box
                                            // keypoints (only on active
                                            // bbox)
                                            if (is_active_bbox &&
                                                ImGui::IsKeyPressed(ImGuiKey_W,
                                                                    false) &&
                                                !io.WantTextInput) {
                                                ImPlotPoint mouse =
                                                    ImPlot::GetPlotMousePos();
                                                if (is_point_in_bbox(
                                                        mouse.x, mouse.y,
                                                        bbox.rect)) {
                                                    u32 active_kp =
                                                        bbox.active_kp_id[j];
                                                    if (active_kp <
                                                        skeleton.num_nodes) {
                                                        bbox.bbox_keypoints2d
                                                            [j][active_kp]
                                                                .position = {
                                                            mouse.x, mouse.y};
                                                        bbox.bbox_keypoints2d
                                                            [j][active_kp]
                                                                .is_labeled =
                                                            true;
                                                        constrain_keypoint_to_bbox(
                                                            &bbox.bbox_keypoints2d
                                                                 [j][active_kp],
                                                            bbox.rect);
                                                        if (active_kp <
                                                            (skeleton
                                                                 .num_nodes -
                                                             1)) {
                                                            bbox.active_kp_id
                                                                [j]++;
                                                        }
                                                    }
                                                }
                                            }

                                            if (is_active_bbox) {
                                                u32 *active_kp =
                                                    &(bbox.active_kp_id[j]);

                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_A, true) &&
                                                    !io.WantTextInput) {
                                                    if (*active_kp <= 0) {
                                                        *active_kp = 0;
                                                    } else {
                                                        (*active_kp)--;
                                                    }
                                                }

                                                if (ImGui::IsKeyPressed(
                                                        ImGuiKey_D, true) &&
                                                    !io.WantTextInput) {
                                                    if (*active_kp >=
                                                        skeleton.num_nodes -
                                                            1) {
                                                        *active_kp =
                                                            skeleton.num_nodes -
                                                            1;
                                                    } else {
                                                        (*active_kp)++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Plot oriented bounding boxes
                            if (skeleton.has_obb) {
                                bool keypoints_find =
                                    keypoints_map.find(current_frame_num) !=
                                    keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       &skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }

                                // ESC key cancels OBB creation
                                if (is_view_focused[j] &&
                                    ImGui::IsKeyPressed(ImGuiKey_Escape,
                                                        false) &&
                                    !io.WantTextInput) {
                                    // Find and cancel any incomplete OBB
                                    for (auto &obb :
                                         keypoints_map[current_frame_num]
                                             ->obb2d_list[j]) {
                                        if (obb.state == OBBFirstAxisPoint ||
                                            obb.state == OBBSecondAxisPoint) {
                                            obb.state = OBBNull;
                                            break;
                                        }
                                    }
                                }

                                // Handle OBB interaction with W key
                                if (is_view_focused[j] &&
                                    ImGui::IsKeyPressed(ImGuiKey_W, false) &&
                                    !io.WantTextInput) {
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();

                                    // Find an OBB to continue or create a
                                    // new one
                                    bool found_incomplete_obb = false;
                                    for (auto &obb :
                                         keypoints_map[current_frame_num]
                                             ->obb2d_list[j]) {
                                        if (obb.state == OBBNull) {
                                            // Start new OBB - place first
                                            // axis point Clamp mouse
                                            // coordinates to frame bounds
                                            double clamped_x = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_width[j],
                                                         (double)mouse.x));
                                            double clamped_y = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_height[j],
                                                         (double)mouse.y));
                                            obb.axis_point1 =
                                                ImVec2(clamped_x, clamped_y);
                                            obb.state = OBBFirstAxisPoint;
                                            obb.class_id =
                                                current_bbox_class; // Use
                                                                    // currently
                                                                    // selected
                                                                    // class
                                            obb.confidence = 1.0f;
                                            found_incomplete_obb = true;
                                            break;
                                        } else if (obb.state ==
                                                   OBBFirstAxisPoint) {
                                            // Place second axis point
                                            // Clamp mouse coordinates to
                                            // frame bounds
                                            double clamped_x = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_width[j],
                                                         (double)mouse.x));
                                            double clamped_y = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_height[j],
                                                         (double)mouse.y));
                                            obb.axis_point2 =
                                                ImVec2(clamped_x, clamped_y);
                                            obb.state = OBBSecondAxisPoint;
                                            found_incomplete_obb = true;
                                            break;
                                        } else if (obb.state ==
                                                   OBBSecondAxisPoint) {
                                            // Place corner point and
                                            // complete the OBB Clamp mouse
                                            // coordinates to frame bounds
                                            double clamped_x = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_width[j],
                                                         (double)mouse.x));
                                            double clamped_y = std::max(
                                                0.0,
                                                std::min((double)scene
                                                             ->image_height[j],
                                                         (double)mouse.y));
                                            obb.corner_point =
                                                ImVec2(clamped_x, clamped_y);
                                            obb.state = OBBThirdPoint;
                                            calculate_obb_properties(&obb);
                                            obb.state = OBBComplete;

                                            // Clear the construction points
                                            // after completion
                                            obb.axis_point1 = ImVec2(0, 0);
                                            obb.axis_point2 = ImVec2(0, 0);
                                            obb.corner_point = ImVec2(0, 0);

                                            // Auto-increment bbox ID after
                                            // finishing drawing OBB
                                            current_bbox_id++;

                                            found_incomplete_obb = true;
                                            break;
                                        }
                                    }

                                    // If no incomplete OBB found, create a
                                    // new one
                                    if (!found_incomplete_obb) {
                                        // Delete existing OBB with the same
                                        // class_id and id
                                        auto &obb_list =
                                            keypoints_map[current_frame_num]
                                                ->obb2d_list[j];
                                        obb_list.erase(
                                            std::remove_if(
                                                obb_list.begin(),
                                                obb_list.end(),
                                                [current_bbox_class,
                                                 current_bbox_id](
                                                    const OrientedBoundingBox
                                                        &obb) {
                                                    return obb.class_id ==
                                                               current_bbox_class &&
                                                           obb.id ==
                                                               current_bbox_id;
                                                }),
                                            obb_list.end());

                                        OrientedBoundingBox new_obb;
                                        // Clamp mouse coordinates to frame
                                        // bounds for new OBB
                                        double clamped_x = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_width[j],
                                                (double)mouse.x));
                                        double clamped_y = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_height[j],
                                                (double)mouse.y));
                                        new_obb.axis_point1 =
                                            ImVec2(clamped_x, clamped_y);
                                        new_obb.axis_point2 = ImVec2(0, 0);
                                        new_obb.corner_point = ImVec2(0, 0);
                                        new_obb.center = ImVec2(0, 0);
                                        new_obb.width = 0;
                                        new_obb.height = 0;
                                        new_obb.rotation = 0;
                                        new_obb.state = OBBFirstAxisPoint;
                                        new_obb.class_id =
                                            current_bbox_class; // Use
                                                                // currently
                                                                // selected
                                                                // class
                                        new_obb.id =
                                            current_bbox_id; // Set OBB ID
                                        new_obb.confidence = 1.0f;
                                        keypoints_map[current_frame_num]
                                            ->obb2d_list[j]
                                            .push_back(new_obb);
                                    }
                                }

                                // Handle OBB manipulation and interaction
                                static bool obb_dragging = false;
                                static size_t dragged_obb_idx = 0;

                                // Handle OBB construction point dragging
                                static bool obb_point_dragging = false;
                                static size_t dragged_point_obb_idx = 0;
                                static int dragged_point_type =
                                    0; // 0 = axis_point1, 1 = axis_point2,
                                       // 2 = corner_point

                                // Handle construction point dragging
                                if (is_view_focused[j] &&
                                    ImPlot::IsPlotHovered()) {
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();
                                    ImVec2 mouse_vec = ImVec2(mouse.x, mouse.y);

                                    // Start dragging construction points
                                    if (ImGui::IsMouseClicked(
                                            ImGuiMouseButton_Left) &&
                                        !obb_point_dragging && !obb_dragging) {
                                        for (size_t obb_idx = 0;
                                             obb_idx <
                                             keypoints_map[current_frame_num]
                                                 ->obb2d_list[j]
                                                 .size();
                                             obb_idx++) {
                                            auto &obb =
                                                keypoints_map[current_frame_num]
                                                    ->obb2d_list[j][obb_idx];

                                            // Only allow dragging for
                                            // incomplete OBBs
                                            if (obb.state >=
                                                    OBBFirstAxisPoint &&
                                                obb.state < OBBComplete) {
                                                // Check if clicking near
                                                // axis_point1
                                                if (obb.state >=
                                                        OBBFirstAxisPoint &&
                                                    is_point_near(
                                                        mouse_vec,
                                                        obb.axis_point1)) {
                                                    obb_point_dragging = true;
                                                    dragged_point_obb_idx =
                                                        obb_idx;
                                                    dragged_point_type = 0;
                                                    break;
                                                }
                                                // Check if clicking near
                                                // axis_point2
                                                if (obb.state >=
                                                        OBBSecondAxisPoint &&
                                                    is_point_near(
                                                        mouse_vec,
                                                        obb.axis_point2)) {
                                                    obb_point_dragging = true;
                                                    dragged_point_obb_idx =
                                                        obb_idx;
                                                    dragged_point_type = 1;
                                                    break;
                                                }
                                                // Check if clicking near
                                                // corner_point
                                                if (obb.state >=
                                                        OBBThirdPoint &&
                                                    is_point_near(
                                                        mouse_vec,
                                                        obb.corner_point)) {
                                                    obb_point_dragging = true;
                                                    dragged_point_obb_idx =
                                                        obb_idx;
                                                    dragged_point_type = 2;
                                                    break;
                                                }
                                            }
                                        }
                                    }

                                    // Continue dragging
                                    if (obb_point_dragging &&
                                        ImGui::IsMouseDragging(
                                            ImGuiMouseButton_Left)) {
                                        auto &obb =
                                            keypoints_map[current_frame_num]
                                                ->obb2d_list
                                                    [j][dragged_point_obb_idx];

                                        // Clamp mouse coordinates to frame
                                        // bounds
                                        double clamped_x = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_width[j],
                                                (double)mouse_vec.x));
                                        double clamped_y = std::max(
                                            0.0,
                                            std::min(
                                                (double)scene->image_height[j],
                                                (double)mouse_vec.y));
                                        ImVec2 clamped_mouse =
                                            ImVec2(clamped_x, clamped_y);

                                        if (dragged_point_type == 0) {
                                            obb.axis_point1 = clamped_mouse;
                                        } else if (dragged_point_type == 1) {
                                            obb.axis_point2 = clamped_mouse;
                                        } else if (dragged_point_type == 2) {
                                            obb.corner_point = clamped_mouse;
                                            // Recalculate OBB properties if
                                            // dragging corner point
                                            if (obb.state == OBBThirdPoint) {
                                                calculate_obb_properties(&obb);
                                            }
                                        }
                                    }

                                    // Stop dragging
                                    if (obb_point_dragging &&
                                        ImGui::IsMouseReleased(
                                            ImGuiMouseButton_Left)) {
                                        obb_point_dragging = false;
                                    }
                                }

                                // Track which OBB is being hovered
                                int current_hovered_obb = -1;

                                // Check for hover when not dragging
                                if (!obb_dragging && !obb_point_dragging &&
                                    ImPlot::IsPlotHovered()) {
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();

                                    // Check if hovering over any OBB
                                    for (size_t obb_idx = 0;
                                         obb_idx <
                                         keypoints_map[current_frame_num]
                                             ->obb2d_list[j]
                                             .size();
                                         obb_idx++) {
                                        auto &obb =
                                            keypoints_map[current_frame_num]
                                                ->obb2d_list[j][obb_idx];

                                        // Check if mouse is inside the OBB
                                        if (is_point_inside_obb(
                                                ImVec2(mouse.x, mouse.y),
                                                obb)) {
                                            current_hovered_obb = obb_idx;
                                            hovered_obb_cam = j;
                                            hovered_obb_idx = obb_idx;
                                            hovered_obb_class = obb.class_id;
                                            hovered_obb_confidence =
                                                obb.confidence;
                                            hovered_obb_id = obb.id;

                                            // Handle key presses for OBB
                                            // manipulation (similar to
                                            // bbox) Delete OBB from current
                                            // camera when 'T' key is
                                            // pressed while hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_T,
                                                                    false) &&
                                                !io.WantTextInput) {
                                                obb.state = OBBNull;
                                            }

                                            // Delete OBB from all cameras
                                            // when 'F' key is pressed while
                                            // hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_F,
                                                                    false) &&
                                                !io.WantTextInput) {
                                                int target_class_id =
                                                    obb.class_id;
                                                for (int cam = 0;
                                                     cam < scene->num_cams;
                                                     cam++) {
                                                    if (keypoints_map.count(
                                                            current_frame_num) >
                                                        0) {
                                                        auto &obb_list =
                                                            keypoints_map
                                                                [current_frame_num]
                                                                    ->obb2d_list
                                                                        [cam];
                                                        for (auto &other_obb :
                                                             obb_list) {
                                                            if (other_obb
                                                                    .class_id ==
                                                                target_class_id) {
                                                                other_obb
                                                                    .state =
                                                                    OBBNull;
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            // Switch OBB class when 'A' or
                                            // 'D' key is pressed while
                                            // hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_A,
                                                                    true) &&
                                                !io.WantTextInput) {
                                                obb.class_id =
                                                    (obb.class_id - 1 +
                                                     bbox_class_names.size()) %
                                                    bbox_class_names.size();
                                            }
                                            if (ImGui::IsKeyPressed(ImGuiKey_D,
                                                                    true) &&
                                                !io.WantTextInput) {
                                                obb.class_id =
                                                    (obb.class_id + 1) %
                                                    bbox_class_names.size();
                                            }

                                            break;
                                        }
                                    }
                                }

                                // Draw all OBBs for this camera
                                ImPlotPoint current_mouse =
                                    ImPlot::GetPlotMousePos();
                                // Clamp mouse coordinates for preview to
                                // frame bounds
                                double clamped_mouse_x = std::max(
                                    0.0, std::min((double)scene->image_width[j],
                                                  current_mouse.x));
                                double clamped_mouse_y = std::max(
                                    0.0,
                                    std::min((double)scene->image_height[j],
                                             current_mouse.y));
                                ImVec2 clamped_preview_mouse =
                                    ImVec2(clamped_mouse_x, clamped_mouse_y);
                                for (size_t obb_idx = 0;
                                     obb_idx < keypoints_map[current_frame_num]
                                                   ->obb2d_list[j]
                                                   .size();
                                     obb_idx++) {
                                    auto &obb = keypoints_map[current_frame_num]
                                                    ->obb2d_list[j][obb_idx];

                                    if (obb.state != OBBNull) {
                                        // Get color based on class_id (same
                                        // system as bboxes)
                                        ImVec4 obb_color =
                                            ImVec4(0.3f, 1.0f, 1.0f,
                                                   1.0f); // Default color
                                        if (obb.class_id >= 0 &&
                                            obb.class_id <
                                                bbox_class_colors.size()) {
                                            obb_color =
                                                bbox_class_colors[obb.class_id];
                                        }

                                        // Highlight when hovering,
                                        // dragging, or during construction
                                        bool is_active =
                                            (obb_dragging &&
                                             dragged_obb_idx == obb_idx) ||
                                            (current_hovered_obb ==
                                             (int)obb_idx) ||
                                            (!obb_dragging &&
                                             obb.state < OBBComplete);

                                        // Show preview when we have at
                                        // least one point and mouse is in
                                        // plot area
                                        bool show_preview =
                                            ((obb.state == OBBFirstAxisPoint ||
                                              obb.state ==
                                                  OBBSecondAxisPoint) &&
                                             ImPlot::IsPlotHovered() &&
                                             !obb_dragging &&
                                             !obb_point_dragging &&
                                             current_hovered_obb == -1);

                                        draw_obb(obb, is_active, obb_color,
                                                 clamped_preview_mouse,
                                                 show_preview);

                                        // Display OBB ID on frame if
                                        // enabled and OBB is complete
                                        if (show_bbox_ids &&
                                            obb.state == OBBComplete) {
                                            // Position text above top-right
                                            // corner of OBB
                                            double text_x =
                                                obb.center.x + obb.width / 2 +
                                                5.0; // Right side of OBB
                                            double text_y =
                                                obb.center.y + obb.height / 2 +
                                                5.0; // Above the OBB
                                            ImPlot::PlotText(
                                                std::to_string(obb.id).c_str(),
                                                text_x, text_y);
                                        }
                                    }
                                }
                            }

                            if (keypoints_find) {
                                // Only plot keypoints if skeleton has
                                // keypoints and we're not in bbox+keypoints
                                // mode
                                if (skeleton.has_skeleton &&
                                    !skeleton.has_bbox) {
                                    gui_plot_keypoints(
                                        keypoints_map.at(current_frame_num),
                                        &skeleton, j, scene->num_cams);
                                }

                                if (skeleton.name == "Rat4Box" ||
                                    skeleton.name == "Rat4Box3Ball") {
                                    gui_plot_bbox_from_keypoints(
                                        keypoints_map.at(current_frame_num),
                                        &skeleton, j, 4, 5);
                                }
                            }
                        }
                        ImPlot::EndPlot();
                    }

                    ImGui::EndChild();

                    float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
                    if (ImGui::Button(ICON_FK_FAST_BACKWARD)) {
                        int clamped_frame =
                            std::max(0, current_frame_num -
                                            10 * dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine(0.0f, spacing);
                    if (ImGui::Button(ICON_FK_STEP_BACKWARD)) {
                        int clamped_frame = std::max(
                            0, current_frame_num - dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine(0.0f, spacing);

                    if (ps.to_display_frame_number ==
                        (dc_context->total_num_frame - 1)) {
                        ImVec4 repeat_normal = ImVec4(1.0f, 1.0f, 0.2f, 1.0f);
                        ImVec4 repeat_hover = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
                        ImVec4 repeat_active = ImVec4(1.0f, 0.9f, 0.1f, 1.0f);
                        ImGui::PushStyleColor(ImGuiCol_Button, repeat_normal);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                              repeat_hover);
                        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                              repeat_active);

                        if (ImGui::Button(ICON_FK_REPEAT)) {
                            // seek to zero
                            seek_all_cameras(scene, 0, dc_context->video_fps,
                                             ps, false);
                        }
                        ImGui::PopStyleColor(3);
                    } else {
                        ImVec4 normal, hover, active;
                        if (ps.play_video) {
                            normal = ImVec4(0.8f, 0.3f, 0.3f, 1.0f);
                            hover = ImVec4(0.9f, 0.4f, 0.4f, 1.0f);
                            active = ImVec4(0.7f, 0.2f, 0.2f, 1.0f);
                        } else {
                            // green
                            normal = ImVec4(0.2f, 0.6f, 0.2f, 1.0f);
                            hover = ImVec4(0.4f, 0.9f, 0.4f, 1.0f);
                            active = ImVec4(0.3f, 0.75f, 0.3f, 1.0f);
                        }
                        ImGui::PushStyleColor(ImGuiCol_Button, normal);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hover);
                        ImGui::PushStyleColor(ImGuiCol_ButtonActive, active);
                        if (ImGui::Button(ps.play_video ? ICON_FK_PAUSE
                                                        : ICON_FK_PLAY)) {
                            ps.play_video = !ps.play_video;
                            if (ps.play_video) {
                                ps.pause_seeked = false;
                                ps.last_play_time_start =
                                    std::chrono::steady_clock::now();
                            } else {
                                ps.pause_selected = 0;
                            }
                        }
                        ImGui::PopStyleColor(3);
                    }

                    ImGui::SameLine(0.0f, spacing);
                    if (ImGui::Button(ICON_FK_STEP_FORWARD)) {
                        int clamped_frame = std::min(
                            dc_context->total_num_frame,
                            current_frame_num + dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine(0.0f, spacing);
                    if (ImGui::Button(ICON_FK_FAST_FORWARD)) {
                        int clamped_frame = std::min(
                            dc_context->total_num_frame,
                            current_frame_num + 10 * dc_context->seek_interval);
                        seek_all_cameras(scene, clamped_frame,
                                         dc_context->video_fps, ps, false);
                    }
                    ImGui::SameLine();
                    ps.slider_just_changed = ImGui::SliderInt(
                        "##frame count", &ps.slider_frame_number, 0,
                        dc_context->estimated_num_frames);
                    ImGui::SameLine();
                    float current_time_sec =
                        ps.slider_frame_number / dc_context->video_fps;
                    float total_time_sec = dc_context->estimated_num_frames /
                                           dc_context->video_fps;

                    std::string current_str = format_time(current_time_sec);
                    std::string total_str = format_time(total_time_sec);
                    ImGui::Text("%s / %s", current_str.c_str(),
                                total_str.c_str());

                    if (ps.slider_just_changed) {
                        // std::cout << "main, seeking: " <<
                        // ps.slider_frame_number
                        //           << std::endl;
                        seek_all_cameras(scene, ps.slider_frame_number,
                                         dc_context->video_fps, ps, false);
                    }

                    ImGui::EndGroup();
                }
                ImGui::End();
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Space, false) &&
                !io.WantTextInput) {
                ps.play_video = !ps.play_video;
                if (ps.play_video) {
                    ps.pause_seeked = false;
                    ps.last_play_time_start = std::chrono::steady_clock::now();
                } else {
                    ps.pause_selected = 0;
                }
            }

            // Bounding box class switching keybinds
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false) && !io.WantTextInput) {
                // Switch to previous class
                if (bbox_class_names.size() > 0) {
                    current_bbox_class =
                        (current_bbox_class - 1 + bbox_class_names.size()) %
                        bbox_class_names.size();
                    current_bbox_id = 0; // Reset bbox ID when switching classes
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_X, false) && !io.WantTextInput) {
                if (bbox_class_names.size() > 0) {
                    // Switch to next class
                    current_bbox_class =
                        (current_bbox_class + 1) % bbox_class_names.size();
                    current_bbox_id = 0; // Reset bbox ID when switching classes
                }
            }

            // Bounding box ID switching keybinds within current class
            if (ImGui::IsKeyPressed(ImGuiKey_C, false) && !io.WantTextInput) {
                // Decrease bbox ID, stop at 0
                if (current_bbox_id > 0) {
                    current_bbox_id--;
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_V, false) && !io.WantTextInput) {
                // Increment bbox ID (no wrap around)
                current_bbox_id++;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_N, false) && !io.WantTextInput) {
                create_new_bbox_class();
                // reset bbox id to 0
                current_bbox_id = 0;
            }

            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false) &&
                !io.WantTextInput) {
                if (ImGui::GetIO().KeyShift) {
                    int clamped_frame = std::max(
                        0, current_frame_num - 10 * dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                } else {
                    int clamped_frame = std::max(
                        0, current_frame_num - dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                }
            }

            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false) &&
                !io.WantTextInput) {
                if (ImGui::GetIO().KeyShift) {
                    int clamped_frame = std::min(
                        dc_context->total_num_frame,
                        current_frame_num + 10 * dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                } else {
                    int clamped_frame =
                        std::min(dc_context->total_num_frame,
                                 current_frame_num + dc_context->seek_interval);
                    seek_all_cameras(scene, clamped_frame,
                                     dc_context->video_fps, ps, false);
                }
            }

            for (const auto &[name, flag] : window_need_decoding) {
                window_was_decoding[name] = flag.load();
            }
        }

        if (pm.plot_keypoints_flag) {
            DrawKeypointsWindow(
                pm, scene, skeleton, keypoints_map, current_frame_num,
                is_view_focused, bbox_class_names, current_bbox_class,
                bbox_class_colors, current_bbox_id, hovered_bbox_cam,
                hovered_bbox_idx, hovered_bbox_id, hovered_bbox_confidence,
                hovered_bbox_class, hovered_obb_cam, hovered_obb_idx,
                hovered_obb_id, hovered_obb_confidence, hovered_obb_class,
                show_bbox_ids, new_class_name);
        }

        if (keypoints_find) {
            DrawReprojectionWindow(keypoints_map[current_frame_num],
                                   pm.camera_names, scene, skeleton, rp_tool);
        }

        if (pm.plot_keypoints_flag) {

            if (ImGui::Begin("Labeling Tool")) {
                if (scene->num_cams > 1) {
                    bool keypoint_triangulated_all = true;
                    if (keypoints_find && scene->num_cams > 1) {
                        for (int j = 0; j < skeleton.num_nodes; j++) {
                            if (!keypoints_map.at(current_frame_num)
                                     ->kp3d[j]
                                     .is_triangulated) {
                                keypoint_triangulated_all = false;
                                break; // small optimization, exit early
                            }
                        }
                    } else {
                        keypoint_triangulated_all = false;
                    }
                    bool apply_color =
                        !keypoint_triangulated_all && keypoints_find;
                    if (apply_color) {
                        ImGui::PushStyleColor(
                            ImGuiCol_Button,
                            (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f));
                        ImGui::PushStyleColor(
                            ImGuiCol_ButtonHovered,
                            (ImVec4)ImColor::HSV(0.8, 0.9f, 0.8f));
                        ImGui::PushStyleColor(
                            ImGuiCol_ButtonActive,
                            (ImVec4)ImColor::HSV(0.8, 0.9f, 0.5f));
                    }

                    ImGui::BeginDisabled(!keypoints_find);
                    if (ImGui::Button("Triangulate")) {
                        reprojection(keypoints_map.at(current_frame_num),
                                     &skeleton, pm.camera_params, scene);
                    }
                    ImGui::EndDisabled();

                    if (apply_color) {
                        ImGui::PopStyleColor(3);
                    }

                    if (keypoints_find) {
                        if (ImGui::IsKeyPressed(ImGuiKey_T,
                                                false) &&
                            !io.WantTextInput) // triangulate
                        {
                            reprojection(keypoints_map.at(current_frame_num),
                                         &skeleton, pm.camera_params, scene);
                        }
                    }
                } else {
                    // Display message when skeleton is not properly loaded
                    ImGui::Text("Please load a skeleton to view keypoints.");
                }

                if (ImGui::Button("Update keypoints working directory")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = pm.project_path;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseKeypointsFolder",
                        "Choose keypoints working directory", nullptr, config);
                }
                ImGui::SameLine();
                ImGui::Text("%s", pm.keypoints_root_folder.c_str());

                if (ImGui::Button("Save Labeled Data") ||
                    (ImGui::GetIO().KeyCtrl &&
                     ImGui::IsKeyPressed(ImGuiKey_S, false) &&
                     !io.WantTextInput)) {

                    // Detect what type of data we have and save accordingly
                    if (skeleton.has_skeleton && !skeleton.has_bbox) {
                        save_keypoints(keypoints_map, &skeleton,
                                       pm.keypoints_root_folder,
                                       scene->num_cams, pm.camera_names,
                                       &input_is_imgs, imgs_names);
                        std::cout << "Saved skeleton keypoints data"
                                  << std::endl;
                    } else if (!skeleton.has_skeleton && skeleton.has_bbox) {
                        save_bboxes(keypoints_map, &skeleton,
                                    pm.keypoints_root_folder, scene->num_cams,
                                    pm.camera_names, &input_is_imgs,
                                    imgs_names);
                        std::cout << "Saved bounding boxes data" << std::endl;
                    } else if (skeleton.has_obb) {
                        save_obb(keypoints_map, &skeleton,
                                 pm.keypoints_root_folder, pm.camera_names,
                                 scene->num_cams, &imgs_names, &input_is_imgs,
                                 bbox_class_names);
                        std::cout << "Saved oriented bounding boxes data"
                                  << std::endl;
                    } else if (skeleton.has_skeleton && skeleton.has_bbox) {
                        bool has_bbox_keypoints = false;
                        for (const auto &[frame_num, keypoints] :
                             keypoints_map) {
                            if (!keypoints)
                                continue;
                            for (int cam_id = 0;
                                 cam_id < scene->num_cams &&
                                 cam_id < MAX_VIEWS && !has_bbox_keypoints;
                                 cam_id++) {
                                for (const auto &bbox :
                                     keypoints->bbox2d_list[cam_id]) {
                                    if (bbox.state == RectTwoPoints &&
                                        bbox.has_bbox_keypoints) {
                                        has_bbox_keypoints = true;
                                        break;
                                    }
                                }
                            }
                            if (has_bbox_keypoints)
                                break;
                        }

                        if (has_bbox_keypoints) {
                            // Save both skeleton keypoints and bbox
                            // keypoints
                            save_keypoints(keypoints_map, &skeleton,
                                           pm.keypoints_root_folder,
                                           scene->num_cams, pm.camera_names,
                                           &input_is_imgs, imgs_names);
                            save_bbox_keypoints(
                                keypoints_map, &skeleton,
                                pm.keypoints_root_folder, scene->num_cams,
                                pm.camera_names, &input_is_imgs, imgs_names);
                            std::cout << "Saved skeleton keypoints and "
                                         "bounding box "
                                         "keypoints data"
                                      << std::endl;
                        } else {
                            save_keypoints(keypoints_map, &skeleton,
                                           pm.keypoints_root_folder,
                                           scene->num_cams, pm.camera_names,
                                           &input_is_imgs, imgs_names);
                            save_bboxes(keypoints_map, &skeleton,
                                        pm.keypoints_root_folder,
                                        scene->num_cams, pm.camera_names,
                                        &input_is_imgs, imgs_names);
                            std::cout << "Saved skeleton keypoints and "
                                         "bounding boxes data"
                                      << std::endl;
                        }
                    } else {
                        save_keypoints(keypoints_map, &skeleton,
                                       pm.keypoints_root_folder,
                                       scene->num_cams, pm.camera_names,
                                       &input_is_imgs, imgs_names);
                        std::cout << "Saved skeleton keypoints data (fallback)"
                                  << std::endl;
                    }

                    last_saved = time(NULL);
                }
                if (last_saved != static_cast<std::time_t>(-1)) {
                    ImGui::SameLine();
                    ImGui::Text("Last saved: %s", ctime(&last_saved));
                }

                static bool load_old_format = false;
                if (ImGui::Button("Load Most Recent Labels")) {
                    free_all_keypoints(keypoints_map, scene);
                    if (load_old_format) {
                        if (load_keypoints_depreciated(keypoints_map, &skeleton,
                                                       pm.keypoints_root_folder,
                                                       scene, pm.camera_names,
                                                       error_message)) {
                            free_all_keypoints(keypoints_map, scene);
                            show_error = true;
                        }

                    } else {
                        std::string most_recent_folder;
                        if (find_most_recent_labels(pm.keypoints_root_folder,
                                                    most_recent_folder,
                                                    error_message)) {
                            show_error = true;
                        } else {
                            if (load_keypoints(most_recent_folder,
                                               keypoints_map, &skeleton, scene,
                                               pm.camera_names, error_message,
                                               bbox_class_names)) {
                                free_all_keypoints(keypoints_map, scene);
                                show_error = true;
                            }
                        }
                    }
                }
                ImGui::SameLine();
                ImGui::Checkbox("Old format", &load_old_format);

                if (ImGui::Button("Load From Selected")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = pm.keypoints_root_folder;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "LoadFromSelected", "Load from selected", nullptr,
                        config);
                }

                ImGui::Separator();

                auto next_labeled_frame_it = keypoints_map.end();

                // Find next labeled frame (forward, wrap)
                for (auto it = keypoints_map.upper_bound(current_frame_num);
                     it != keypoints_map.end(); ++it) {
                    if (has_any_labels(it->second, skeleton, scene)) {
                        next_labeled_frame_it = it;
                        break;
                    }
                }
                if (next_labeled_frame_it == keypoints_map.end()) {
                    for (auto it = keypoints_map.begin();
                         it != keypoints_map.upper_bound(current_frame_num);
                         ++it) {
                        if (has_any_labels(it->second, skeleton, scene)) {
                            next_labeled_frame_it = it;
                            break;
                        }
                    }
                }

                // Find previous labeled frame (no wrap)
                auto prev_labeled_frame_it = keypoints_map.end();
                auto lb = keypoints_map.lower_bound(current_frame_num);
                if (lb != keypoints_map.begin()) {
                    for (auto it = std::prev(lb);;) {
                        if (has_any_labels(it->second, skeleton, scene)) {
                            prev_labeled_frame_it = it;
                            break;
                        }
                        if (it == keypoints_map.begin())
                            break;
                        --it;
                    }
                }

                bool has_next_frame =
                    (next_labeled_frame_it != keypoints_map.end());
                int next_frame_num =
                    has_next_frame ? next_labeled_frame_it->first : -1;
                bool has_prev_frame =
                    (prev_labeled_frame_it != keypoints_map.end());
                int previous_frame_num =
                    has_prev_frame ? prev_labeled_frame_it->first : -1;

                if (ImGui::BeginTable("frame_nav_table", 2,
                                      ImGuiTableFlags_SizingFixedFit)) {
                    ImGui::TableNextRow();

                    // --- Next ---
                    ImGui::TableSetColumnIndex(0);

                    ImGui::BeginDisabled(!has_next_frame);
                    bool button_pressed =
                        ImGui::Button("Jump to next labeled frame");
                    ImGui::EndDisabled();

                    if (button_pressed && has_next_frame) {
                        seek_all_cameras(scene, next_frame_num,
                                         dc_context->video_fps, ps, true);
                    }

                    ImGui::TableSetColumnIndex(1);
                    if (has_next_frame)
                        ImGui::Text("%d", next_frame_num);
                    else
                        ImGui::Text("none");

                    // --- Previous ---
                    ImGui::TableNextRow();

                    ImGui::TableSetColumnIndex(0);
                    ImGui::BeginDisabled(!has_prev_frame);
                    if (ImGui::Button("Copy from previous labeled frame")) {
                        if (keypoints_find) {
                            // delete the keypoints if found
                            free_keypoints(keypoints_map[current_frame_num],
                                           scene);
                            keypoints_map.erase(current_frame_num);
                        }

                        KeyPoints *keypoints =
                            (KeyPoints *)malloc(sizeof(KeyPoints));
                        allocate_keypoints(keypoints, scene, &skeleton);
                        keypoints_map[current_frame_num] = keypoints;

                        KeyPoints *prev = keypoints_map[previous_frame_num];
                        KeyPoints *curr = keypoints_map[current_frame_num];
                        copy_keypoints(curr, prev, scene, &skeleton);
                    }
                    ImGui::EndDisabled();

                    ImGui::TableSetColumnIndex(1);
                    if (has_prev_frame)
                        ImGui::Text("%d", previous_frame_num);
                    else
                        ImGui::Text("none");

                    ImGui::EndTable();
                }

                size_t labeled_count = 0;
                for (const auto &[frame_num, keypoints] : keypoints_map) {
                    if (has_any_labels(keypoints, skeleton, scene,
                                       /*yolo_thresh=*/0.5f)) {
                        ++labeled_count;
                    }
                }
                ImGui::Text("Total labeled frames : %zu", labeled_count);

                // Only show YOLO button when current skeleton has bounding
                // boxes
                if (skeleton.has_bbox) {
                    if (ImGui::Button("Select YOLO")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = yolo_model_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseYoloModel", "Choose YOLO Model",
                            ".torchscript", config);
                    }
                }

                if (ImGuiFileDialog::Instance()->Display("ChooseYoloModel")) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        std::string model_path =
                            ImGuiFileDialog::Instance()->GetFilePathName();
                        if (!model_path.empty()) {
                            yolo_model_path = model_path;
                            std::cout
                                << "Selected YOLO model: " << yolo_model_path
                                << std::endl;
                        }
                    }
                    // close
                    ImGuiFileDialog::Instance()->Close();
                }

                // Show model path and run prediction button (only if model
                // is selected)
                if (!yolo_model_path.empty()) {
                    ImGui::Text("Selected model: %s", yolo_model_path.c_str());

                    // Automatic YOLO labeling checkbox
                    ImGui::Checkbox("Automatic YOLO Labeling",
                                    &auto_yolo_labeling);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Automatically run YOLO detection on current "
                            "and subsequent frames");
                    }

                    // YOLO parameter sliders
                    ImGui::SliderFloat("Confidence Threshold",
                                       &confidence_threshold, 0.01f, 0.99f,
                                       "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Minimum confidence score for detections");
                    }

                    ImGui::SliderFloat("NMS IoU Threshold", &iou_threshold,
                                       0.01f, 0.99f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip(
                            "Non-Maximum Suppression IoU threshold");
                    }

                    if (ImGui::Button("Run YOLO Prediction")) {
                        std::cout << "Running YOLO prediction on frame "
                                  << ps.to_display_frame_number << std::endl;

                        yolo_detection = true;

                        // Clear existing bounding boxes for current frame
                        // before running inference
                        if (skeleton.has_bbox) {
                            bool keypoints_find =
                                keypoints_map.find(current_frame_num) !=
                                keypoints_map.end();
                            if (keypoints_find) {
                                for (int cam_id = 0; cam_id < scene->num_cams;
                                     cam_id++) {
                                    auto &bbox_list =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list[cam_id];
                                    // Clear all bounding boxes for this
                                    // camera
                                    for (auto &bbox : bbox_list) {
                                        if (bbox.rect) {
                                            delete bbox.rect;
                                            bbox.rect = nullptr;
                                        }
                                        if (bbox.has_bbox_keypoints &&
                                            bbox.bbox_keypoints2d) {
                                            free(bbox.bbox_keypoints2d);
                                            bbox.bbox_keypoints2d = nullptr;
                                            free(bbox.active_kp_id);
                                            bbox.active_kp_id = nullptr;
                                        }
                                    }
                                    bbox_list.clear();
                                }
                                std::cout << "Cleared existing bounding "
                                             "boxes for frame "
                                          << current_frame_num << std::endl;
                            }
                        }
                        yolo_processed_frames.insert(current_frame_num);
                        for (int cam_id = 0; cam_id < scene->num_cams;
                             cam_id++) {
                            if (ps.pause_seeked) {
                                unsigned char *frame_data =
                                    scene
                                        ->display_buffer[cam_id]
                                                        [select_corr_head]
                                        .frame;

                                if (frame_data) {
                                    yolo_predictions[cam_id] = runYoloInference(
                                        yolo_model_path, frame_data,
                                        scene->image_width[cam_id],
                                        scene->image_height[cam_id]);

                                    yolo_bboxes[cam_id].clear();
                                    for (const auto &pred :
                                         yolo_predictions[cam_id]) {
                                        yolo_bboxes[cam_id].emplace_back(pred);
                                    }
                                }
                            } else {
                                if (window_was_decoding
                                        [pm.camera_names[cam_id]]) {
                                    unsigned char *frame_data =
                                        scene
                                            ->display_buffer[cam_id]
                                                            [select_corr_head]
                                            .frame;

                                    if (frame_data) {
                                        yolo_predictions[cam_id] =
                                            runYoloInference(
                                                yolo_model_path, frame_data,
                                                scene->image_width[cam_id],
                                                scene->image_height[cam_id]);

                                        yolo_bboxes[cam_id].clear();
                                        for (const auto &pred :
                                             yolo_predictions[cam_id]) {
                                            yolo_bboxes[cam_id].emplace_back(
                                                pred);
                                        }
                                    }
                                }
                            }
                        }

                        if (!yolo_bboxes.empty() &&
                            std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(),
                                        [](const auto &cam_bboxes) {
                                            return !cam_bboxes.empty();
                                        })) {
                            // Convert YOLO detections to main bounding box
                            // system
                            for (int cam_id = 0; cam_id < scene->num_cams;
                                 cam_id++) {
                                if (!yolo_bboxes[cam_id].empty()) {
                                    // Ensure keypoints structure exists
                                    bool keypoints_find =
                                        keypoints_map.find(current_frame_num) !=
                                        keypoints_map.end();
                                    if (!keypoints_find) {
                                        KeyPoints *keypoints =
                                            (KeyPoints *)malloc(
                                                sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene,
                                                           &skeleton);
                                        keypoints_map[current_frame_num] =
                                            keypoints;
                                    }

                                    // Add YOLO detections to main bounding
                                    // box system
                                    int yolo_bbox_id = 0;
                                    for (const auto &yolo_bbox :
                                         yolo_bboxes[cam_id]) {
                                        if (yolo_bbox.is_valid) {
                                            while (yolo_bbox.class_id >=
                                                   bbox_class_colors.size()) {
                                                create_new_bbox_class();
                                            }

                                            BoundingBox bbox;

                                            // Create ImPlotRect from YOLO
                                            // coordinates (no Y-axis
                                            // flipping)
                                            bbox.rect = new ImPlotRect(
                                                yolo_bbox.x_min, // X.Min
                                                yolo_bbox.x_max, // X.Max
                                                yolo_bbox.y_min, // Y.Min
                                                yolo_bbox.y_max  // Y.Max
                                            );

                                            bbox.state = RectTwoPoints;
                                            bbox.class_id = yolo_bbox.class_id;
                                            bbox.id = yolo_bbox_id++;
                                            bbox.confidence =
                                                yolo_bbox.confidence;
                                            bbox.has_bbox_keypoints = false;
                                            bbox.bbox_keypoints2d = nullptr;
                                            bbox.active_kp_id = nullptr;

                                            // Allocate keypoints if
                                            // skeleton supports both bbox
                                            // and skeleton
                                            if (skeleton.has_bbox &&
                                                skeleton.has_skeleton &&
                                                skeleton.num_nodes > 0) {
                                                allocate_bbox_keypoints(
                                                    &bbox, scene, &skeleton);
                                            }

                                            // Add to main bounding box
                                            // system
                                            keypoints_map[current_frame_num]
                                                ->bbox2d_list[cam_id]
                                                .push_back(bbox);
                                        }
                                    }

                                    std::cout << "Added "
                                              << yolo_bboxes[cam_id].size()
                                              << " YOLO detections to main "
                                                 "bounding "
                                                 "box system for camera "
                                              << cam_id << std::endl;
                                }
                            }
                        }
                    }
                }
            }
            ImGui::End();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseKeypointsFolder")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                pm.keypoints_root_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("LoadFromSelected")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto selected_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                free_all_keypoints(keypoints_map, scene);
                if (load_keypoints(selected_folder, keypoints_map, &skeleton,
                                   scene, pm.camera_names, error_message,
                                   bbox_class_names)) {
                    free_all_keypoints(keypoints_map, scene);
                    show_error = true;
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        // YOLO Export Tool Dialog Handlers
        if (ImGuiFileDialog::Instance()->Display("ChooseYoloExportLabelDir")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string selected_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                yolo_export_label_dir = selected_path;
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseYoloExportVideoDir")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string selected_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                yolo_export_video_dir = selected_path;
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseYoloExportOutputDir")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string selected_path =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                yolo_export_output_dir = selected_path;
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display(
                "ChooseYoloExportSkeletonFile")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string selected_file =
                    ImGuiFileDialog::Instance()->GetFilePathName();
                yolo_export_skeleton_file = selected_file;
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseYoloExportClassFile")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string selected_file =
                    ImGuiFileDialog::Instance()->GetFilePathName();
                yolo_export_class_names_file = selected_file;
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_H, false) && !io.WantTextInput) {
            show_help_window = !show_help_window;
        }

        if (show_help_window) {
            if (ImGui::Begin("Help Menu")) {
                ImGui::Text("<Space>: toggle play and pause");
                ImGui::Text("<Left Arrow>    : Seek backward");
                ImGui::Text("<Shift+Left>    : Seek backward (×10)");
                ImGui::Text("<Right Arrow>   : Seek forward");
                ImGui::Text("<Shift+Right>   : Seek forward (×10)");

                ImGui::SeparatorText("When paused");
                ImGui::Text("<,>: previous image in buffer");
                ImGui::Text("<.>: next image in buffer");

                ImGui::SeparatorText("While hovering image");
                ImGui::Text("<b>: create keypoints on frame");
                ImGui::Text("<w>: drop active keypoint");
                ImGui::Text("<a>: active keypoint-- ");
                ImGui::Text("<d>: active keypoint++");
                ImGui::Text("<q>: active keypoint set to first node");
                ImGui::Text("<e>: active keypoint set to last node");
                ImGui::Text("<t>: triangulate");
                ImGui::Text("<Backspace>: delete all keypoints");
                ImGui::Text("<Ctrl+s>: Save labels");

                ImGui::SeparatorText("Bounding box");
                ImGui::Text("<Shift + drag mouse>: draw bbox, drag then "
                            "release to "
                            "finish drawing the bbox");
                ImGui::Text("<f>: delete bounding box from current camera");
                ImGui::Text("<o>: delete all instances of current class");
                ImGui::Text("<z>: switch to previous bbox class");
                ImGui::Text("<x>: switch to next bbox class (creates new "
                            "class if at end)");
                ImGui::Text("<n>: create new bbox class");
                ImGui::Text("<c>: bbox id--");
                ImGui::Text("<v>: bbox id++");

                ImGui::SeparatorText("While hovering keypoints");
                ImGui::Text("<r>: delete active keypoint");
                ImGui::Text("<f>: delete active keypoint on all cameras");
                ImGui::Text("Click keypoint to active it");
            }
            ImGui::End();
        }

        // Skeleton Creator Window
        if (show_skeleton_creator) {
            ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Skeleton Creator", &show_skeleton_creator)) {
                ImGui::SeparatorText("Skeleton Configuration");
                ImGui::InputText("Skeleton Name", &skeleton_creator_name);
                ImGui::Checkbox("Has Bounding Box", &skeleton_creator_has_bbox);

                if (ImGui::Button("Select Background Image")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = std::filesystem::current_path().string();
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseBackgroundImage", "Choose Background Image",
                        ".png,.jpg,.jpeg,.tiff,.bmp,.tga", config);
                }
                ImGui::SameLine();
                if (background_image_selected && background_texture != 0) {
                    std::filesystem::path path(background_image_path);
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                       "Image: %s (%dx%d)",
                                       path.filename().string().c_str(),
                                       background_width, background_height);
                    ImGui::SameLine();
                    if (ImGui::Button("Clear Background")) {
                        if (background_texture != 0) {
                            glDeleteTextures(1, &background_texture);
                            background_texture = 0;
                        }
                        background_image_path = "";
                        background_image_selected = false;
                        background_width = 0;
                        background_height = 0;
                        std::cout << "Background image cleared" << std::endl;
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                                       "No background image");
                }

                ImGui::SeparatorText("Interactive Editor");

                if (ImPlot::BeginPlot("Skeleton Creator", ImVec2(-1, 400),
                                      ImPlotFlags_Equal)) {
                    ImPlot::SetupAxes("", "");
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0,
                                            ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0,
                                            ImGuiCond_Always);

                    ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, 0);
                    ImPlot::SetupAxisTicks(ImAxis_Y1, nullptr, 0);

                    if (background_image_selected && background_texture != 0) {
                        ImPlot::PlotImage(
                            "##background",
                            (ImTextureID)(intptr_t)background_texture,
                            ImPlotPoint(0, 0), ImPlotPoint(1, 1));
                    }

                    if (ImPlot::IsPlotHovered() &&
                        ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                        !ImGui::GetIO().KeyCtrl) {
                        if (selected_node_for_edge < 0) {
                            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
                            creator_nodes.emplace_back(mouse_pos.x, mouse_pos.y,
                                                       next_node_id++);
                        }
                    }

                    if (ImGui::IsKeyPressed(ImGuiKey_Escape) &&
                        !io.WantTextInput) {
                        selected_node_for_edge = -1;
                    }

                    for (const auto &edge : creator_edges) {
                        const SkeletonCreatorNode *node1 = nullptr;
                        const SkeletonCreatorNode *node2 = nullptr;

                        for (const auto &node : creator_nodes) {
                            if (node.id == edge.node1_id)
                                node1 = &node;
                            if (node.id == edge.node2_id)
                                node2 = &node;
                        }

                        if (node1 && node2) {
                            double xs[2] = {node1->position.x,
                                            node2->position.x};
                            double ys[2] = {node1->position.y,
                                            node2->position.y};
                            ImPlot::SetNextLineStyle(
                                ImVec4(0.8f, 0.8f, 0.8f, 1.0f), 2.0f);
                            ImPlot::PlotLine("##edge", xs, ys, 2);
                        }
                    }

                    for (size_t i = 0; i < creator_nodes.size(); i++) {
                        auto &node = creator_nodes[i];

                        bool clicked = false, hovered = false, held = false;
                        ImVec4 node_color = node.color;

                        if (selected_node_for_edge == node.id) {
                            node_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                        }

                        bool modified = ImPlot::DragPoint(
                            node.id, &node.position.x, &node.position.y,
                            node_color, 8.0f, ImPlotDragToolFlags_None,
                            &clicked, &hovered, &held);

                        if (hovered) {
                            ImPlot::PlotText(node.name.c_str(), node.position.x,
                                             node.position.y + 0.03);

                            if (ImGui::IsKeyPressed(ImGuiKey_R, false) &&
                                !io.WantTextInput) {
                                int node_id_to_delete = node.id;

                                creator_nodes.erase(creator_nodes.begin() + i);

                                creator_edges.erase(
                                    std::remove_if(
                                        creator_edges.begin(),
                                        creator_edges.end(),
                                        [node_id_to_delete](
                                            const SkeletonCreatorEdge &edge) {
                                            return edge.node1_id ==
                                                       node_id_to_delete ||
                                                   edge.node2_id ==
                                                       node_id_to_delete;
                                        }),
                                    creator_edges.end());

                                if (selected_node_for_edge ==
                                    node_id_to_delete) {
                                    selected_node_for_edge = -1;
                                }

                                break;
                            }
                        }

                        if (clicked && ImGui::GetIO().KeyCtrl) {
                            if (selected_node_for_edge < 0) {
                                selected_node_for_edge = node.id;
                            } else if (selected_node_for_edge != node.id) {
                                bool edge_exists = false;
                                for (const auto &existing_edge :
                                     creator_edges) {
                                    if ((existing_edge.node1_id ==
                                             selected_node_for_edge &&
                                         existing_edge.node2_id == node.id) ||
                                        (existing_edge.node1_id == node.id &&
                                         existing_edge.node2_id ==
                                             selected_node_for_edge)) {
                                        edge_exists = true;
                                        break;
                                    }
                                }

                                if (!edge_exists) {
                                    creator_edges.emplace_back(
                                        selected_node_for_edge, node.id);
                                } else {
                                    // Delete the existing edge
                                    creator_edges.erase(
                                        std::remove_if(
                                            creator_edges.begin(),
                                            creator_edges.end(),
                                            [selected_node_for_edge,
                                             node_id = node.id](
                                                const SkeletonCreatorEdge
                                                    &edge) {
                                                return (edge.node1_id ==
                                                            selected_node_for_edge &&
                                                        edge.node2_id ==
                                                            node_id) ||
                                                       (edge.node1_id ==
                                                            node_id &&
                                                        edge.node2_id ==
                                                            selected_node_for_edge);
                                            }),
                                        creator_edges.end());
                                }

                                selected_node_for_edge = -1;
                            } else {
                                selected_node_for_edge = -1;
                            }
                        }
                    }

                    ImPlot::EndPlot();
                }

                if (selected_node_for_edge >= 0) {
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                                       "Selected node for edge creation. "
                                       "Ctrl+Click another node to "
                                       "create edge or remove existing "
                                       "edge, or press ESC to cancel.");
                }

                ImGui::SeparatorText("Help");
                ImGui::BulletText(
                    "Left-click in the plot area to add a new node");
                ImGui::BulletText("Drag nodes to reposition them");
                ImGui::BulletText(
                    "Ctrl+Click a node to select it for edge creation");
                ImGui::BulletText(
                    "Ctrl+Click another node to create an edge or remove "
                    "an existing edge between them");
                ImGui::BulletText("Press ESC to cancel edge creation");
                ImGui::BulletText("Press R while hovering a node to delete "
                                  "it and its edges");

                ImGui::SeparatorText("Actions");

                if (ImGui::Button("Clear All")) {
                    creator_nodes.clear();
                    creator_edges.clear();
                    next_node_id = 0;
                    selected_node_for_edge = -1;
                }

                ImGui::SameLine();
                if (ImGui::Button("Load from JSON")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = skeleton_dir;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "LoadSkeletonForEdit", "Load Skeleton", ".json",
                        config);
                }

                ImGui::SameLine();
                if (ImGui::Button("Save to JSON")) {
                    if (!creator_nodes.empty()) {
                        nlohmann::json skeleton_json;
                        skeleton_json["name"] = skeleton_creator_name;
                        skeleton_json["has_skeleton"] = true;
                        skeleton_json["has_bbox"] = skeleton_creator_has_bbox;
                        skeleton_json["num_nodes"] = (int)creator_nodes.size();
                        skeleton_json["num_edges"] = (int)creator_edges.size();

                        std::vector<std::string> node_names;
                        std::vector<std::vector<double>> node_positions;
                        for (const auto &node : creator_nodes) {
                            node_names.push_back(node.name);
                            node_positions.push_back(
                                {node.position.x, node.position.y});
                        }
                        skeleton_json["node_names"] = node_names;
                        skeleton_json["node_positions"] = node_positions;

                        std::vector<std::vector<int>> edges_array;
                        for (const auto &edge : creator_edges) {
                            int idx1 = -1, idx2 = -1;
                            for (size_t i = 0; i < creator_nodes.size(); i++) {
                                if (creator_nodes[i].id == edge.node1_id)
                                    idx1 = (int)i;
                                if (creator_nodes[i].id == edge.node2_id)
                                    idx2 = (int)i;
                            }
                            if (idx1 >= 0 && idx2 >= 0) {
                                edges_array.push_back({idx1, idx2});
                            }
                        }
                        skeleton_json["edges"] = edges_array;

                        std::string filename;
                        filename = skeleton_dir + "/" + skeleton_creator_name +
                                   ".json";

                        std::ofstream file(filename);
                        file << skeleton_json.dump(4);
                        file.close();
                        std::cout << "Skeleton saved to: " << filename
                                  << " (with node positions)" << std::endl;
                    }
                }

                if (!creator_nodes.empty()) {
                    ImGui::SeparatorText("Nodes");
                    if (ImGui::BeginTable("NodeTable", 3,
                                          ImGuiTableFlags_Borders |
                                              ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn(
                            "ID", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                        ImGui::TableSetupColumn(
                            "Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn(
                            "Position", ImGuiTableColumnFlags_WidthFixed,
                            120.0f);
                        ImGui::TableHeadersRow();

                        for (size_t i = 0; i < creator_nodes.size(); i++) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%d", creator_nodes[i].id);

                            ImGui::TableSetColumnIndex(1);
                            ImGui::PushID(i);
                            ImGui::InputText("##name", &creator_nodes[i].name);
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("(%.2f, %.2f)",
                                        creator_nodes[i].position.x,
                                        creator_nodes[i].position.y);
                        }
                        ImGui::EndTable();
                    }
                }
            }
            ImGui::End();
        }

        DrawLiveTable(table, "Spreadsheet", scene, dc_context->video_fps, ps,
                      pm.project_path);

        // YOLO Export Tool Window
        if (show_yolo_export_tool) {
            ImGui::SetNextWindowSize(ImVec2(600, 700), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("YOLO Export Tool", &show_yolo_export_tool)) {
                ImGui::SeparatorText("Export Configuration");

                // Export mode selection
                const char *export_modes[] = {"Detection Dataset",
                                              "Pose Dataset", "OBB Dataset"};
                int current_mode = static_cast<int>(yolo_export_mode);

                auto apply_defaults = [&]() {
                    if (pm.media_folder.empty())
                        return; // guard if config not loaded yet
                    yolo_export_mode =
                        static_cast<YoloExportMode>(current_mode);
                    if (yolo_export_mode == YOLO_DETECTION) {
                        if (pm.project_path.empty()) {
                            yolo_export_output_dir =
                                pm.media_folder + "/yolo_detection_dataset";
                        } else {
                            yolo_export_output_dir =
                                pm.project_path + "/yolo_detection_dataset";
                        }
                    } else if (yolo_export_mode == YOLO_POSE) {
                        if (pm.project_path.empty()) {
                            yolo_export_output_dir =
                                pm.media_folder + "/yolo_pose_dataset";
                        } else {
                            yolo_export_output_dir =
                                pm.project_path + "/yolo_pose_dataset";
                        }
                        if (!skeleton_file_path.empty())
                            yolo_export_skeleton_file = skeleton_file_path;
                    } else {
                        if (pm.project_path.empty()) {
                            yolo_export_output_dir =
                                pm.media_folder + "/yolo_obb_dataset";
                        } else {
                            yolo_export_output_dir =
                                pm.project_path + "/yolo_obb_dataset";
                        }
                    }
                };

                static bool initialized = false;

                // 1) First-time init (once, when media_dir is known)
                if (!initialized && !pm.media_folder.empty()) {
                    apply_defaults();
                    initialized = true;
                }

                // 2) UI: apply when user changes mode
                if (ImGui::Combo("Export Mode", &current_mode, export_modes,
                                 IM_ARRAYSIZE(export_modes))) {
                    apply_defaults();
                }

                ImGui::Separator();
                // Directory and file inputs
                ImGui::Text("Input Directories:");
                ImGui::InputText("Label Directory", &yolo_export_label_dir);
                ImGui::SameLine();
                if (ImGui::Button("Browse##labels")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = yolo_export_label_dir;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseYoloExportLabelDir", "Choose Label Directory",
                        nullptr, config);
                }

                ImGui::InputText("Video Directory", &yolo_export_video_dir);
                ImGui::SameLine();
                if (ImGui::Button("Browse##videos")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = yolo_export_video_dir;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseYoloExportVideoDir", "Choose Video Directory",
                        nullptr, config);
                }

                ImGui::InputText("Output Directory", &yolo_export_output_dir);
                ImGui::SameLine();
                if (ImGui::Button("Browse##output")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = yolo_export_output_dir;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseYoloExportOutputDir", "Choose Output Directory",
                        nullptr, config);
                }

                ImGui::Separator();

                // Configuration files
                if (yolo_export_mode == YOLO_POSE) {
                    ImGui::Text("Skeleton Configuration:");
                    ImGui::InputText("Skeleton File",
                                     &yolo_export_skeleton_file);
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##skeleton")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = std::filesystem::current_path().string() +
                                      "/skeleton";
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseSkeleton", "Choose Skeleton", ".json",
                            config);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Use Current")) {
                        if (!skeleton_file_path.empty()) {
                            yolo_export_skeleton_file = skeleton_file_path;
                        }
                    }
                }

                ImGui::Text("Camera List:");
                for (size_t i = 0; i < yolo_export_cam_names.size(); i++) {
                    ImGui::BulletText("%s", yolo_export_cam_names[i].c_str());
                }

                if (yolo_export_mode == YOLO_DETECTION ||
                    yolo_export_mode == YOLO_OBB) {
                    ImGui::Text("Class Names (Optional):");
                    ImGui::InputText("Class Names File",
                                     &yolo_export_class_names_file);
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##classes")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = yolo_model_dir;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseYoloExportClassFile",
                            "Choose Class Names File", ".txt", config);
                    }
                }

                ImGui::Separator();

                // Export parameters
                ImGui::Text("Export Parameters:");
                ImGui::SliderInt("Image Size", &yolo_export_image_size, 320,
                                 1280, "%d");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Images will be resized to this "
                                      "square size (e.g., 640x640)");
                }

                ImGui::SliderFloat("Train Ratio", &yolo_export_train_ratio,
                                   0.1f, 0.9f, "%.2f");
                ImGui::SliderFloat("Val Ratio", &yolo_export_val_ratio, 0.05f,
                                   0.5f, "%.2f");
                ImGui::SliderFloat("Test Ratio", &yolo_export_test_ratio, 0.05f,
                                   0.5f, "%.2f");

                // Ensure ratios sum to 1.0
                float total_ratio = yolo_export_train_ratio +
                                    yolo_export_val_ratio +
                                    yolo_export_test_ratio;
                if (total_ratio > 0.001f) {
                    ImGui::Text("Total: %.2f", total_ratio);
                    if (total_ratio > 1.001f || total_ratio < 0.999f) {
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f),
                                           "Warning: Ratios should sum to 1.0");
                        if (ImGui::Button("Normalize Ratios")) {
                            yolo_export_train_ratio /= total_ratio;
                            yolo_export_val_ratio /= total_ratio;
                            yolo_export_test_ratio /= total_ratio;
                        }
                    }
                }

                ImGui::InputInt("Random Seed", &yolo_export_seed);
                if (ImGui::Button("Reset Defaults")) {

                    yolo_export_image_size = 640;
                    yolo_export_train_ratio = 0.7f;
                    yolo_export_val_ratio = 0.2f;
                    yolo_export_test_ratio = 0.1f;
                    yolo_export_seed = 42;
                    yolo_export_status = "";
                }

                ImGui::Separator();

                // Export buttons and status
                if (!yolo_export_in_progress) {
                    if (ImGui::Button("Start Export", ImVec2(150, 30))) {
                        // Validate inputs
                        bool valid = true;
                        std::string validation_error;

                        if (yolo_export_label_dir.empty()) {
                            valid = false;
                            validation_error = "Label directory is required";
                        } else if (yolo_export_video_dir.empty()) {
                            valid = false;
                            validation_error = "Video directory is required";
                        } else if (yolo_export_output_dir.empty()) {
                            valid = false;
                            validation_error = "Output directory is required";
                        } else if (yolo_export_mode == YOLO_POSE &&
                                   yolo_export_skeleton_file.empty()) {
                            valid = false;
                            validation_error = "Skeleton file is required "
                                               "for pose datasets";
                        }

                        if (valid) {
                            yolo_export_in_progress = true;
                            yolo_export_status = "Starting export...";

                            // Setup export configuration
                            YoloExport::ExportConfig config;
                            config.label_dir = yolo_export_label_dir;
                            config.video_dir = yolo_export_video_dir;
                            config.output_dir = yolo_export_output_dir;
                            config.cam_names = yolo_export_cam_names;
                            config.skeleton_file = yolo_export_skeleton_file;
                            config.class_names_file =
                                std::string(yolo_export_class_names_file);
                            config.image_size = yolo_export_image_size;
                            config.split.train_ratio = yolo_export_train_ratio;
                            config.split.val_ratio = yolo_export_val_ratio;
                            config.split.test_ratio = yolo_export_test_ratio;
                            config.split.seed = yolo_export_seed;

                            // Run export in background thread
                            std::thread export_thread(
                                [config, yolo_export_mode, &yolo_export_status,
                                 &yolo_export_in_progress]() {
                                    bool success = false;
                                    if (yolo_export_mode == YOLO_DETECTION) {
                                        success = YoloExport::
                                            export_yolo_detection_dataset(
                                                config, &yolo_export_status);
                                    } else if (yolo_export_mode == YOLO_POSE) {
                                        success = YoloExport::
                                            export_yolo_pose_dataset(
                                                config, &yolo_export_status);
                                    } else {
                                        success =
                                            YoloExport::export_yolo_obb_dataset(
                                                config, &yolo_export_status);
                                    }

                                    // Update status (note: this is not
                                    // thread-safe, but for simple status
                                    // updates it should be okay)
                                    if (success) {
                                        yolo_export_status = "Export completed "
                                                             "successfully!";
                                    } else {
                                        yolo_export_status =
                                            "Export failed! Check console "
                                            "for details.";
                                    }
                                    yolo_export_in_progress = false;
                                });
                            export_thread.detach();
                        } else {
                            yolo_export_status = "Error: " + validation_error;
                        }
                    }
                } else {
                    ImGui::Text("Export in progress...");
                    ImGui::SameLine();
                }

                if (!yolo_export_status.empty()) {
                    if (yolo_export_status.find("Error") != std::string::npos ||
                        yolo_export_status.find("failed") !=
                            std::string::npos) {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s",
                                           yolo_export_status.c_str());
                    } else if (yolo_export_status.find("completed") !=
                               std::string::npos) {
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s",
                                           yolo_export_status.c_str());
                    } else {
                        ImGui::Text("%s", yolo_export_status.c_str());
                    }
                }

                ImGui::Separator();

                // Quick setup buttons
                ImGui::Text("Quick Setup:");
                if (ImGui::Button("Use Current Data")) {
                    yolo_export_video_dir = pm.media_folder;
                    yolo_export_output_dir = pm.project_path + "/export";
                    yolo_export_cam_names = pm.camera_names;
                    // Set label dir to current keypoints folder
                    if (!pm.keypoints_root_folder.empty()) {
                        yolo_export_label_dir = pm.keypoints_root_folder;
                    }

                    // Set skeleton file to current one
                    if (!skeleton_file_path.empty()) {
                        yolo_export_skeleton_file = skeleton_file_path;
                    }
                }
            }
            ImGui::End();
        }

        // Make sure this is called before any other popups in the same frame
        if (show_error) {
            ImGui::OpenPopup("Error");
            show_error = false; // Reset the flag so it only opens once
        }

        // Ensures popup is drawn in front of others
        if (ImGui::BeginPopupModal("Error", NULL,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("%s", error_message.c_str());
            ImGui::Separator();

            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window->render_target, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w,
                     clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we
        // save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call
        //  glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window->render_target);

        if (ps.just_seeked) {
            ps.just_seeked = false;
        } else {
            if (dc_context->decoding_flag && ps.play_video) {
                // always round up, mimimum 1
                int frame_to_show = static_cast<int>(
                    std::ceil(playback_time_now * dc_context->video_fps));

                int min_decoded_frame = INT_MAX;
                for (const auto &[cam_name, visible] : window_need_decoding) {
                    if (visible.load()) {
                        int decoded = latest_decoded_frame[cam_name].load();
                        min_decoded_frame =
                            std::min(min_decoded_frame, decoded);
                    }
                }
                frame_to_show = std::min(frame_to_show, min_decoded_frame);

                int frame_delta = frame_to_show - ps.to_display_frame_number;
                if (frame_delta > 0) {
                    // Update frame number
                    ps.to_display_frame_number = frame_to_show;

                    // Mark all intermediate frames as available
                    for (int offset = 0; offset < frame_delta; ++offset) {
                        int index =
                            (ps.read_head + offset) % scene->size_of_buffer;
                        for (int j = 0; j < scene->num_cams; j++) {
                            scene->display_buffer[j][index].available_to_write =
                                true;
                        }
                    }

                    // Advance the read head
                    ps.read_head =
                        (ps.read_head + frame_delta) % scene->size_of_buffer;

                    // Optional: update slider/UI sync
                    ps.slider_frame_number = ps.to_display_frame_number;
                }
            }
        }
    }
    // Cleanup
    cleanup_yolo_drag_boxes(); // Clean up YOLO drag boxes memory
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window->render_target);
    glfwTerminate();

    dc_context->stop_flag = true;
    // wait for threads to join
    for (auto &t : decoder_threads)
        t.join();

    return 0;
}
