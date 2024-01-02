#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Logger.h"
#include "render.h"
#include "implot.h"
#include <iostream>
#include <thread>
#include <imfilebrowser.h>
#include <camera.h>
#include "skeleton.h"
#include "gui.h"
#include "yolo_detection.h"
#include "global.h"
#include "utils.h"


#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#define label_buffer_size 32 

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

std::vector<std::mutex> g_mutexes(MAX_VIEWS);
std::vector<std::condition_variable> g_cvs(MAX_VIEWS);
std::vector<bool> g_ready(MAX_VIEWS);
std::vector<std::vector<cv::Rect>> yolo_boxes(MAX_VIEWS);
std::vector<std::vector<std::string>> yolo_labels(MAX_VIEWS);
std::vector<std::vector<int>> yolo_classid(MAX_VIEWS);
std::vector<unsigned char*> yolo_input_frames_rgba(MAX_VIEWS);

int main(int, char **)
{
    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    *window = (gx_context){
        .swap_interval = 1, // use vsync
        .width = 1920,
        .height = 1080,
        .render_target_title = (char *)malloc(100), // window title
        .glsl_version = (char *)malloc(100)};

    render_initialize_target(window);

    render_scene *scene = (render_scene *)malloc(sizeof(render_scene));

    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
    std::vector<CameraParams> camera_params;
    std::vector<std::thread> decoder_threads;
    DecoderContext *dc_context = (DecoderContext *)malloc(sizeof(DecoderContext));
    *dc_context = (DecoderContext){
        .decoding_flag = false,
        .stop_flag = false,
        .total_num_frame = int(INT_MAX),
        .estimated_num_frames = 0,
        .gpu_index = 0,
        .seek_interval=10};

    // gui states, todo: bundle this later
    int to_display_frame_number = 0;
    int pause_selected = 0;
    int read_head = 0;
    bool play_video = false;
    bool toggle_play_status = false;
    int slider_frame_number = 0;
    bool just_seeked = false;
    bool slider_just_changed = false;
    bool video_loaded = false;
    bool cpu_buffer_toggle = true;
    bool plot_keypoints_flag = false;
    int current_frame_num = 0;
    bool skeleton_chosen = false;

    // for labeling 
    SkeletonContext *skeleton;
    std::map<u32, KeyPoints*> keypoints_map;
    bool keypoints_find = false;
    std::map<std::string, SkeletonPrimitive> skeleton_map;
    
    // others
    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory);
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split (cwd, delimiter);
    std::string start_folder_name = "/home/" + tokenized_path[2] + "/data";

    file_dialog.SetPwd(start_folder_name);
    file_dialog.SetTitle("Select working directory");
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();

    bool yolo_detection = false;
    std::vector<std::thread> yolo_threads;
    yolo_param yolo_setting = yolo_param();
    bool show_world_coordinates = false;

    while (!glfwWindowShouldClose(window->render_target))
    {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("File Browser", NULL, ImGuiWindowFlags_MenuBar))
        {

            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open"))
                    {
                        file_dialog.Open();
                    };
                    ImGui::EndMenu();
                }

                if(video_loaded){
                    if (ImGui::BeginMenu("Skeleton"))
                    {
                        if (!skeleton_chosen) {
                            skeleton = new SkeletonContext;
                            skeleton_map = skeleton_get_all();
                        }

                        for (auto & element : skeleton_map) {
                            
                            if(ImGui::MenuItem(element.first.c_str(), NULL, skeleton->name == element.first, !skeleton_chosen))
                            {
                                // std::string cam_file = root_dir + "/calibration/calibration.csv";
                                for (u32 i = 0; i < scene->num_cams; i++)
                                {
                                    // legacy loading from old formats
                                    // CameraParams cam = camera_load_params_from_csv(cam_file, i);
                                    // camera_params.push_back(cam);
                                    std::string cam_file = root_dir + "/calibration/" + camera_names[i] + ".yaml";
                                    std::cout << cam_file << std::endl;
                                    CameraParams cam = camera_load_params_from_yaml(cam_file);
                                    camera_params.push_back(cam);
                                    skeleton_chosen = true;
                                }
                                skeleton->name = element.first;
                                skeleton_initialize(skeleton, element.second);
                                plot_keypoints_flag = true;
                            }
                        }
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("Detection")) {
                        if (ImGui::MenuItem("YOLOv5")) { 
                            std::string yolov5_onnx = root_dir + "/yolo/v5/best.onnx";
                            std::string yolov5_labelname = root_dir + "/yolo/v5/label.names";
                            read_yolo_labels(yolov5_labelname, &yolo_setting);

                            for (int i = 0; i< scene->num_cams; i++) {
                                yolo_threads.push_back(std::thread(&yolo_process, yolov5_onnx, &yolo_setting, i));
                            }
                            yolo_detection = true;
                        }

                        if (ImGui::MenuItem("YOLOv8Pose")) {
                            std::string engine_file_path = root_dir + "/yolo/v8_pose/rat_pose.engine";
                            for (int i = 0; i< scene->num_cams; i++) {
                                yolo_threads.push_back(std::thread(&yolo_process_v8pose, engine_file_path, i));
                            }
                            yolo_detection = true;

                        }

                        ImGui::EndMenu();
                    }
                }
                ImGui::EndMenuBar();
            }

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

            // if (video_loaded) {
            //     ImGui::Text("Frame number %d ", scene->display_buffer[0][read_head].frame_number);
            //     ImGui::Text("To Display frame number %d ", to_display_frame_number);
            //     ImGui::Text("Readhead %d", read_head);
            // }

            ImGui::Checkbox("CPU Buffer", &cpu_buffer_toggle);
            scene->use_cpu_buffer = cpu_buffer_toggle;
            if (video_loaded && !skeleton_chosen) {
                if (ImGui::InputInt("Seek step (s)", &dc_context->seek_interval, 10, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
                    std::cout << "Seek step: " << dc_context->seek_interval << std::endl;
                }
            }
        }
        ImGui::End();

        file_dialog.Display();

        if (file_dialog.HasSelected())
        {
            root_dir = file_dialog.GetSelected().string();

            // load movies
            std::string movie_dir = root_dir + "/movies";

            for (const auto &entry : std::filesystem::directory_iterator(movie_dir))
            {
                input_file_names.push_back(entry.path().string());
            }

            std::sort(input_file_names.begin(), input_file_names.end());
            scene->image_width = 3208;
            scene->image_height = 2200;
            render_allocate_scene_memory(scene, 3208, 2200, input_file_names.size(), label_buffer_size);

            // multiple threads for decoding for selected videos
            for (u32 i = 0; i < scene->num_cams; i++)
            {
                std::size_t cam_string_position = input_file_names[i].find("Cam");           // position of "Cam" in str
                std::size_t cam_string_mp4_position = input_file_names[i].find("mp4");
                std::size_t length_of_substr = cam_string_mp4_position - cam_string_position - 1;
                std::string cam_string = input_file_names[i].substr(cam_string_position, length_of_substr); // get from "Cam" to the end
                camera_names.push_back(cam_string);
                std::cout << "camera names: " << cam_string << std::endl;
                decoder_threads.push_back(std::thread(&decoder_process, input_file_names[i].c_str(), dc_context, scene->display_buffer[i], scene->size_of_buffer, &scene->seek_context[i], scene->use_cpu_buffer));
            }

            video_loaded = true;
            file_dialog.ClearSelected();
        }

        if (dc_context->decoding_flag && play_video)
        {
            for (u32 j = 0; j < scene->num_cams; j++)
            {
                // if the current frame is ready, upload for display, otherwise wait for the frame to get ready
                while (scene->display_buffer[j][read_head].frame_number != to_display_frame_number)
                {
                    // std::cout << "main wait, " << read_head << ", " << scene->display_buffer[j][read_head].frame_number << ", " << to_display_frame_number << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                // sync yolo detection 
                if (yolo_detection)
                {
                    std::unique_lock<std::mutex> lck(g_mutexes[j]);
                    yolo_input_frames_rgba[j] = scene->display_buffer[j][read_head].frame;
                    g_ready[j] = true;
                    g_cvs[j].notify_one();
                }
                
                // todo: need to use pbo to accelerate this 
                if (scene->use_cpu_buffer) {
                    upload_texture(&scene->image_texture[j], scene->display_buffer[j][read_head].frame, 3208, 2200);
                } else {
                    bind_pbo(&scene->pbo_cuda[j][read_head].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width, scene->image_height); // Needs no arguments because texture and PBO are bound
                    unbind_pbo();
                    unbind_texture();
                }

            }
            current_frame_num = to_display_frame_number;
        }

        // show frames in the buffer if selected
        if (video_loaded && (!play_video))
        {
            static int select_corr_head = 0;
            ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frames in the buffer", NULL, ImGuiWindowFlags_MenuBar))
            {
                {
                    for (u32 i = 0; i < scene->size_of_buffer; i++)
                    {
                        char label[128];
                        sprintf(label, "Buffer %d", i);
                        if (ImGui::Selectable(label, pause_selected == i))
                        {
                            // start from the lowest frame
                            pause_selected = i;
                        }
                    }
                }

                ImGui::Separator();

                if (ImGui::Button(ICON_FK_MINUS) || ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true))
                {
                    if (pause_selected > 0)
                    {
                        pause_selected--;
                    }
                };

                ImGui::SameLine();
                if (ImGui::Button(ICON_FK_PLUS) || ImGui::IsKeyPressed(ImGuiKey_RightBracket, true))
                {
                    if (pause_selected < (scene->size_of_buffer - 1))
                    {
                        pause_selected++;
                    }
                };
            }
            ImGui::Text("Frame number selected: %d", scene->display_buffer[0][select_corr_head].frame_number);
            ImGui::End();

            select_corr_head = (pause_selected + read_head) % scene->size_of_buffer;
            current_frame_num = scene->display_buffer[0][select_corr_head].frame_number;
        
            for (int j = 0; j < scene->num_cams; j++)
            {
                if (scene->use_cpu_buffer) {
                    upload_texture(&scene->image_texture[j], scene->display_buffer[j][select_corr_head].frame, 3208, 2200);
                } else {
                    bind_pbo(&scene->pbo_cuda[j][select_corr_head].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width, scene->image_height); // Needs no arguments because texture and PBO are bound
                    unbind_pbo();
                    unbind_texture();
                }
            }
        }

        if (toggle_play_status && play_video)
        {
            play_video = false;
            toggle_play_status = false;
        }


        if (plot_keypoints_flag) {
            if (keypoints_map.find(current_frame_num)==keypoints_map.end()) {
                keypoints_find = false;
            } else {
                keypoints_find = true;
            }
        }

        // Render a video frame
        if (video_loaded)
        {
            for (int j = 0; j < scene->num_cams; j++)
            {
                // layout
                ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
                ImVec2 window_pos;

                if (scene->num_cams < 8) {
                    if (j % 2 == 0) {
                        window_pos.y = 200.0; window_pos.x = (j/2.0) * 500;}
                    else{
                        window_pos.y = 600.0; window_pos.x = (j-1)/2.0 * 500;}
                } else {
                    if (j % 4 == 0) {
                        window_pos.y = 200.0; window_pos.x = (j/4.0) * 500;}
                    else if (j % 4 == 1) {
                        window_pos.y = 600.0; window_pos.x = (j-1)/4.0 * 500;}
                    else if (j % 4 == 2) {
                        window_pos.y = 1000.0; window_pos.x = (j-1)/4.0 * 500;}
                    else {
                        window_pos.y = 1400.0; window_pos.x = (j-1)/4.0 * 500;}
                }
                
                ImGui::SetNextWindowPos(window_pos, ImGuiCond_FirstUseEver);
                ImGui::Begin(camera_names[j].c_str());
                ImGui::BeginGroup();

                std::string scene_name = "scene view" + std::to_string(j);
                ImGui::BeginChild(scene_name.c_str(), ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below

                ImVec2 avail_size = ImGui::GetContentRegionAvail();
                // ImGui::Image((void*)(intptr_t)image_texture[j], avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size, ImPlotFlags_Equal | ImPlotAxisFlags_AutoFit | ImPlotFlags_Crosshairs))
                {
                    ImPlot::PlotImage("##no_image_name", (void *)(intptr_t)scene->image_texture[j], ImVec2(0, 0), ImVec2(3208, 2200));
                    
                    if (yolo_detection){
                        draw_cv_contours(yolo_boxes.at(j), yolo_labels.at(j), yolo_classid.at(j));
                    }

                    if(plot_keypoints_flag){
                        // plot arena for testing camera parameters 
                        // gui_plot_perimeter(&camera_params[j]);
                        // gui_plot_world_coordinates(&camera_params[j], j);

                        // labeling 
                        if (ImPlot::IsPlotHovered()){
                            
                            if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                                // create keypoints
                                if (!keypoints_find){
                                    // not found
                                    KeyPoints* keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene, skeleton);
                                    keypoints_map[current_frame_num] = keypoints; 
                                }
                            }

                            if (keypoints_find) {

                                u32* kp = &(keypoints_map[current_frame_num]->active_id[j]);
                                if (ImGui::IsKeyPressed(ImGuiKey_W, false)){
                                    // labeling sequentially each view
                                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                    keypoints_map[current_frame_num]->keypoints2d[j][*kp].position = {mouse.x,  mouse.y};
                                    keypoints_map[current_frame_num]->keypoints2d[j][*kp].is_labeled = true;
                                    if(*kp < (skeleton->num_nodes - 1)) {(*kp)++;}
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_3, false)) {
                                    for (int k=0; k<scene->num_cams; k++) {
                                        keypoints_map[current_frame_num]->keypoints2d[k][*kp].position = {1E7,  1E7};
                                        keypoints_map[current_frame_num]->keypoints2d[k][*kp].is_labeled = false;                                        
                                    }
                                }

                                // Use "Q" and "E" keys to scroll through and set active keypoint to label
                                if (ImGui::IsKeyPressed(ImGuiKey_Q, false))
                                {
                                    if (*kp <= 0) {*kp = 0;}
                                    else (*kp)--;
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_E, false))
                                {
                                    if (*kp >= skeleton->num_nodes-1) {*kp = skeleton->num_nodes-1;}
                                    else (*kp)++;
                                }
                                
                                if (ImGui::IsKeyPressed(ImGuiKey_T, false))   // skip to the last keypoint
                                {
                                    *kp = skeleton->num_nodes-1;
                                }
                                
                                if (ImGui::IsKeyPressed(ImGuiKey_D, false))  // delete the active keypoint
                                {
                                    keypoints_map[current_frame_num]->keypoints2d[j][*kp].position.x = 1E7;
                                    keypoints_map[current_frame_num]->keypoints2d[j][*kp].position.y = 1E7;
                                    keypoints_map[current_frame_num]->keypoints2d[j][*kp].is_labeled = false;
                                    std::cout << skeleton->node_names.at(*kp) << " deleted on " << j << std:: endl;
                                }

                                // delete all keypoint, memory leak here, need to handle it cleanly
                                if (ImGui::IsKeyPressed(ImGuiKey_B, false)) 
                                {
                                    std::cout << "keypressed" << std::endl;
                                    KeyPoints* keypoints = nullptr;
                                    keypoints_map.erase(current_frame_num);
                                    keypoints_find = false;
                                }
                            }
                        }

                        if (keypoints_find){
                            gui_plot_keypoints(keypoints_map.at(current_frame_num), skeleton, j);
                            // think more general solution of multiple sets of keypoints 
                            if (skeleton->name == "Rat4Box" || skeleton->name == "Rat4Box3Ball") {
                                gui_plot_bbox_from_keypoints(keypoints_map.at(current_frame_num), skeleton, j, 4, 5);
                            }
                        }
                    }
                    ImPlot::EndPlot();
                }

                ImGui::EndChild();

                if (to_display_frame_number == (dc_context->total_num_frame - 1))
                {
                    if (ImGui::Button(ICON_FK_REPEAT))
                    {
                        // seek to zero
                        for (int i = 0; i < scene->num_cams; i++)
                        {
                            scene->seek_context[i].seek_frame = 0;
                            scene->seek_context[i].use_seek = true;
                            scene->seek_context[i].seek_accurate = false;
                        }

                        for (int i = 0; i < scene->num_cams; i++)
                        {
                            // synchronize seeking
                            while (!(scene->seek_context[i].seek_done))
                            {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                                // std::cout << "Seeking Cam" << i << ", " << scene->seek_context[i].seek_done << std::endl;
                            }
                        }

                        for (int i = 0; i < scene->num_cams; i++)
                        {
                            scene->seek_context[i].seek_done = false;
                        }

                        to_display_frame_number = scene->seek_context[0].seek_frame;
                        read_head = 0;
                        just_seeked = true;
                        slider_frame_number = to_display_frame_number;
                    }
                }
                else
                {
                    if (ImGui::Button(play_video ? ICON_FK_PAUSE : ICON_FK_PLAY)  ||  ImGui::IsKeyPressed(ImGuiKey_Space, true))
                    {
                        play_video = !play_video;
                        if (!play_video) {
                            pause_selected = 0;
                        }
                    }
                }

                ImGui::SameLine();
                // Arrow buttons with Repeater
                float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
                ImGui::PushButtonRepeat(true);
                ImGui::SameLine(0.0f, spacing);
                if (ImGui::Button(ICON_FK_PLUS))
                {
                    // advance_clicks++;
                    play_video = true;
                    toggle_play_status = true;
                }
                ImGui::PopButtonRepeat();
                ImGui::SameLine();

                slider_just_changed = ImGui::SliderInt("##frame count", &slider_frame_number, 0, dc_context->estimated_num_frames);

                if (slider_just_changed)
                {
                    std::cout << "main, seeking: " << slider_frame_number << std::endl;

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        scene->seek_context[i].seek_frame = (uint64_t)slider_frame_number;
                        scene->seek_context[i].use_seek = true;
                        scene->seek_context[i].seek_accurate = false;
                    }

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        // synchronize seeking
                        while (!(scene->seek_context[i].seek_done))
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            // std::cout << "Seeking Cam" << i << ", " << scene->seek_context[i].seek_done << std::endl;
                        }
                    }

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        scene->seek_context[i].seek_done = false;
                    }

                    // std::cout << "Main thread seeking done to frame: " << scene->seek_context[0].seek_frame << std::endl;
                    to_display_frame_number = scene->seek_context[0].seek_frame;
                    read_head = 0;
                    just_seeked = true;
                    slider_frame_number = to_display_frame_number;
                }

                ImGui::EndGroup();
                ImGui::End();
            }
        }

        if (plot_keypoints_flag)
        {
            if (ImGui::Begin("Labeling Tool"))
            {
                for (int i=0; i<scene->num_cams; i++)
                {
                    for (int j=0; j<skeleton->num_nodes; j++)
                    {
                        if (j > 0) ImGui::SameLine();

                        ImGui::PushID(j);

                        if (keypoints_find) {
                            
                            if (keypoints_map.at(current_frame_num)->keypoints2d[i][j].is_labeled)
                            {
                                if (keypoints_map[current_frame_num]->active_id[i] == j) {
                                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f));
                                } else {
                                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(j / (float)skeleton->num_nodes, 0.6f, 0.6f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(j / (float)skeleton->num_nodes, 0.7f, 0.7f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(j / (float)skeleton->num_nodes, 0.8f, 0.8f));
                                }
                            } else {
                                if (keypoints_map[current_frame_num]->active_id[i] == j) {
                                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f));
                                } else {
                                    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(j / (float)skeleton->num_nodes, 0.3f, 0.3f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(j / (float)skeleton->num_nodes, 0.4f, 0.4f));
                                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(j / (float)skeleton->num_nodes, 0.5f, 0.5f));
                                }
                            }
                        } else {
                            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
                        }
                        if (ImGui::Button(skeleton->node_names[j].c_str())) {
                            keypoints_map[current_frame_num]->active_id[i] = j;
                        }
                        ImGui::PopStyleColor(3);
                        ImGui::PopID();
                    }
                }

                if (ImGui::Button("Triangulate"))
                {
                    reprojection(keypoints_map.at(current_frame_num), skeleton, camera_params, scene->num_cams);
                }

                // added by RJ
                if (ImGui::IsKeyPressed(ImGuiKey_2, false))   // triangulate
                {
                    reprojection(keypoints_map.at(current_frame_num), skeleton, camera_params, scene->num_cams);
                }

                if (ImGui::Button("Save Labeled Data"))
                {
                    save_keypoints(keypoints_map, skeleton, root_dir, scene->num_cams, camera_names);
                }

                if (ImGui::Button("Load Labeled Data"))
                {
                    load_keypoints(keypoints_map, skeleton, root_dir, scene, camera_names);
                }

                if (ImGui::Button("Load 2d Keypoints Only"))
                {
                    for (int i=0; i<scene->num_cams; i++) {
                        load_2d_keypoints(keypoints_map, skeleton, root_dir, i, camera_names[i], scene);
                    }
                }
                
                auto upper_it = keypoints_map.upper_bound(current_frame_num); 
                ImGui::Text("Next labeled frame : %d", (*upper_it).first);
                
                static int seek_accurate_frame_num = 0;
                if (ImGui::InputInt("Seek Accurate to: ", &seek_accurate_frame_num, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
                    std::cout << "Seek accurate to: " << seek_accurate_frame_num << std::endl;
                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        scene->seek_context[i].seek_frame = (uint64_t)seek_accurate_frame_num;
                        scene->seek_context[i].use_seek = true;
                        scene->seek_context[i].seek_accurate = true;
                    }

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        while (!(scene->seek_context[i].seek_done))
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        scene->seek_context[i].seek_done = false;
                    }
                    to_display_frame_number = scene->seek_context[0].seek_frame;
                    read_head = 0;
                    just_seeked = true;
                    slider_frame_number = to_display_frame_number;
                }

                ImGui::NewLine();
                ImGui::Text("[ -> Previous image, ] -> Next image");
                ImGui::Text("Space -> Toggle play and pause");
                ImGui::Text("While hovering image...");
                ImGui::Text("C -> Create keypoints on frame");
                ImGui::Text("Q -> Active keypoint++ ");
                ImGui::Text("E -> Active keypoint--");
                ImGui::Text("D -> Delete active keypoint");
                ImGui::Text("T -> Active keypoint set to last node");
                ImGui::Text("W -> Drop active keypoint");
            }
            ImGui::End();
        }


        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window->render_target, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window->render_target);

        if (just_seeked) {
            just_seeked = false;
        } else {
            if (dc_context->decoding_flag && play_video && (to_display_frame_number < (dc_context->total_num_frame - 1)))
            {
                to_display_frame_number++;

                for (int j = 0; j < scene->num_cams; j++)
                {
                    scene->display_buffer[j][read_head].available_to_write = true;
                }

                read_head = (read_head + 1) % scene->size_of_buffer;
                slider_frame_number = to_display_frame_number;
            }
        }
    }

    // Cleanup
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
