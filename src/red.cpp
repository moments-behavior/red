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
#include "skeleton.h"
#include "gui.h"
#include "utils.h"


#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

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
    std::vector<std::thread> decoder_threads;
    std::vector<FFmpegDemuxer*> demuxers; 

    DecoderContext *dc_context = (DecoderContext *)malloc(sizeof(DecoderContext));
    *dc_context = (DecoderContext){
        .decoding_flag = false,
        .stop_flag = false,
        .total_num_frame = int(INT_MAX),
        .estimated_num_frames = 0,
        .gpu_index = 0,
        .seek_interval=250};

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
    int number_of_animals = 5;

    // for labeling 
    SkeletonContext *skeleton;
    std::map<u32, Animals*> keypoints_map;
    bool keypoints_find = false;
    std::map<std::string, SkeletonPrimitive> skeleton_map;
    
    // others
    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory);
    #ifdef _WIN32
        std::string cwd = std::filesystem::current_path().string();
    #else
        std::filesystem::path cwd = std::filesystem::current_path();
    #endif 
    
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split (cwd, delimiter);
    std::string start_folder_name = "/home/" + tokenized_path[2] + "/data";

    file_dialog.SetPwd(start_folder_name);
    file_dialog.SetTitle("Select working directory");
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();

    bool show_world_coordinates = false;
    std::string keypoints_root_folder;
    bool change_keypoints_folder =false;
    int label_buffer_size = 32;
    bool show_help_window = false;
    std::vector<bool> is_view_focused;

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
                                for (u32 i = 0; i < scene->num_cams; i++)
                                {
                                    skeleton_chosen = true;
                                }
                                skeleton->name = element.first;
                                skeleton_initialize(skeleton, element.second);
                                plot_keypoints_flag = true;
                                keypoints_root_folder = root_dir + "/labeled_data/";
                            }
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

            if (!video_loaded) {
                ImGui::Checkbox("CPU Buffer", &cpu_buffer_toggle);
                ImGui::InputInt("Buffer Size", &label_buffer_size, ImGuiInputTextFlags_EnterReturnsTrue);
            }
            scene->use_cpu_buffer = cpu_buffer_toggle;
            if (video_loaded) {
                if (!skeleton_chosen) {
                    ImGui::InputInt("Animals No.", &number_of_animals, 1, 5, ImGuiInputTextFlags_EnterReturnsTrue);
                }

                if (ImGui::InputInt("Seek step", &dc_context->seek_interval, 10, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
                    std::cout << "Seek step: " << dc_context->seek_interval << std::endl;
                }
        
                static int seek_accurate_frame_num = 0;
                if (ImGui::InputInt("Seek Accurate", &seek_accurate_frame_num, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
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
                    pause_selected = 0;
                    slider_frame_number = to_display_frame_number;
                }
            }
        }
        ImGui::End();

        file_dialog.Display();

        if (file_dialog.HasSelected())
        {
            if (change_keypoints_folder) {
                keypoints_root_folder = file_dialog.GetSelected().string();
                change_keypoints_folder = false;
            } else {
                root_dir = file_dialog.GetSelected().string();

                // load movies
                std::string movie_dir = root_dir + "/movies";

                for (const auto &entry : std::filesystem::directory_iterator(movie_dir))
                {
                    input_file_names.push_back(entry.path().string());
                }

                std::sort(input_file_names.begin(), input_file_names.end(), numerical_compare_substr);

                for (u32 i = 0; i < input_file_names.size(); i++)
                {
                    std::map<std::string, std::string> m;
                    FFmpegDemuxer* demuxer = new FFmpegDemuxer(input_file_names[i].c_str(), m);
                    demuxers.push_back(demuxer);
                }

                std::map<std::string, std::string> m;
                FFmpegDemuxer dummy_dmuxer(input_file_names[0].c_str(), m);
                dc_context->seek_interval = (int)dummy_dmuxer.FindKeyFrameInterval(); // get the seek interval

                render_allocate_scene_memory(scene, demuxers, input_file_names.size(), label_buffer_size);
                
                // multiple threads for decoding for selected videos                
                for (u32 i = 0; i < scene->num_cams; i++)
                {
                    std::size_t cam_string_position = input_file_names[i].find("Cam");           // position of "Cam" in str
                    std::size_t cam_string_mp4_position = input_file_names[i].find("mp4");
                    std::size_t length_of_substr = cam_string_mp4_position - cam_string_position - 1;
                    std::string cam_string = input_file_names[i].substr(cam_string_position, length_of_substr); // get from "Cam" to the end
                    camera_names.push_back(cam_string);
                    std::cout << "camera names: " << cam_string << std::endl;
                    decoder_threads.push_back(std::thread(&decoder_process, input_file_names[i].c_str(), dc_context, demuxers[i], scene->display_buffer[i], scene->size_of_buffer, &scene->seek_context[i], scene->use_cpu_buffer));
                    is_view_focused.push_back(false);
                }

                video_loaded = true;
            }
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

                if (scene->use_cpu_buffer) {
                    // upload_texture(&scene->image_texture[j], scene->display_buffer[j][read_head].frame, scene->image_width[j], scene->image_height[j]); // 2x slower than pbo
                    // copy frame to cuda buffer
                    ck(cudaMemcpy(scene->pbo_cuda[j].cuda_buffer, scene->display_buffer[j][read_head].frame, scene->image_width[j] * scene->image_height[j] * 4, cudaMemcpyHostToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width[j], scene->image_height[j]); // Needs no arguments because texture and PBO are bound
                    unbind_pbo();
                    unbind_texture();
                } else {
                    ck(cudaMemcpy(scene->pbo_cuda[j].cuda_buffer, scene->display_buffer[j][read_head].frame, scene->image_width[j] * scene->image_height[j] * 4, cudaMemcpyDeviceToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width[j], scene->image_height[j]); // Needs no arguments because texture and PBO are bound
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
            if (ImGui::Begin("Frames in the buffer"))
            {
                ImGui::Text("Frame number selected: %d", scene->display_buffer[0][select_corr_head].frame_number);
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

                if (ImGui::Button(ICON_FK_MINUS) || ImGui::IsKeyPressed(ImGuiKey_Comma, true))
                {
                    if (pause_selected > 0)
                    {
                        pause_selected--;
                    }
                };

                ImGui::SameLine();
                if (ImGui::Button(ICON_FK_PLUS) || ImGui::IsKeyPressed(ImGuiKey_Period, true))
                {
                    if (pause_selected < (scene->size_of_buffer - 1))
                    {
                        pause_selected++;
                    }
                };
            }
            ImGui::End();

            select_corr_head = (pause_selected + read_head) % scene->size_of_buffer;
            current_frame_num = scene->display_buffer[0][select_corr_head].frame_number;
        
            for (int j = 0; j < scene->num_cams; j++)
            {
                if (scene->use_cpu_buffer) {
                    // upload_texture(&scene->image_texture[j], scene->display_buffer[j][select_corr_head].frame, scene->image_width[j], scene->image_height[j]);
                    ck(cudaMemcpy(scene->pbo_cuda[j].cuda_buffer, scene->display_buffer[j][select_corr_head].frame, scene->image_width[j] * scene->image_height[j] * 4, cudaMemcpyHostToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width[j], scene->image_height[j]); // Needs no arguments because texture and PBO are bound
                    unbind_pbo();
                    unbind_texture();
                } else {
                    ck(cudaMemcpy(scene->pbo_cuda[j].cuda_buffer, scene->display_buffer[j][select_corr_head].frame, scene->image_width[j] * scene->image_height[j] * 4, cudaMemcpyDeviceToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(scene->image_width[j], scene->image_height[j]); // Needs no arguments because texture and PBO are bound
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
                    ImPlot::PlotImage("##no_image_name", (void *)(intptr_t)scene->image_texture[j], ImVec2(0, 0), ImVec2(scene->image_width[j], scene->image_height[j]));

                    if(plot_keypoints_flag)
                    {
                        // labeling 
                        if (ImPlot::IsPlotHovered()) {
                            is_view_focused[j]  = true;

                            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
                                // create keypoints
                                if (!keypoints_find){
                                    // not found
                                    Animals* animals = (Animals *)malloc(sizeof(Animals));
                                    allocate_keypoints(animals, scene, skeleton, number_of_animals);
                                    keypoints_map[current_frame_num] = animals; 
                                }
                            }

                            if (keypoints_find && skeleton->has_skeleton) {
                                Animals* current_frame_data = keypoints_map[current_frame_num];
                                KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                                u32* kp = &frame_keypoints->active_kp_id[j];
                                if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
                                    // labeling sequentially each view
                                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                    frame_keypoints->keypoints2d[j][*kp].position = {mouse.x,  mouse.y};
                                    frame_keypoints->keypoints2d[j][*kp].is_labeled = true;
                                    frame_keypoints->keypoints2d[j][*kp].is_triangulated = false;
                                    if(*kp < (skeleton->num_nodes - 1)) {(*kp)++;}
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_A, true))
                                {
                                    if (*kp <= 0) {*kp = 0;}
                                    else (*kp)--;
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_D, true))
                                {
                                    if (*kp >= skeleton->num_nodes-1) {*kp = skeleton->num_nodes-1;}
                                    else (*kp)++;
                                }
                                
                                if (ImGui::IsKeyPressed(ImGuiKey_E, false))   // skip to the last keypoint
                                {
                                    *kp = skeleton->num_nodes-1;
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_Q, false))   // go to the first keypoint
                                {
                                    *kp = 0;
                                }

                                // delete all keypoint, memory leak here, need to handle it cleanly
                                if (ImGui::IsKeyPressed(ImGuiKey_V, false)) 
                                {
                                    KeyPoints* keypoints = nullptr;
                                    keypoints_map.erase(current_frame_num);
                                    keypoints_find = false;
                                }
                            }  
                            
                            if (keypoints_find && skeleton->has_bbox) {
                                Animals* current_frame_data = keypoints_map[current_frame_num];
                                KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                                if (frame_keypoints->bbox2d->state == RectOnePoint && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                    frame_keypoints->bbox2d->rect->X.Max = mouse.x;
                                    frame_keypoints->bbox2d->rect->Y.Max = mouse.y;
                                }

                                if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
                                    frame_keypoints->bbox2d->state = RectTwoPoints;
                                }

                                if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle, false)) {
                                    if (frame_keypoints->bbox2d->state == RectNull) {
                                        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                        frame_keypoints->bbox2d->rect = new ImPlotRect(mouse.x, mouse.x, mouse.y, mouse.y);
                                        frame_keypoints->bbox2d->state = RectOnePoint;
                                        frame_keypoints->has_labels = true;
                                    }
                                }
                            }

                        } else {
                            is_view_focused[j]  = false;
                        }

                        if (keypoints_find) {
                            Animals* current_frame_data = keypoints_map[current_frame_num];

                            if(skeleton->has_skeleton) {
                                KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                                gui_plot_keypoints(frame_keypoints, skeleton, j, scene->num_cams);
                                // think more general solution of multiple sets of keypoints 
                                if (skeleton->name == "Rat4Box" || skeleton->name == "Rat4Box3Ball") {
                                    gui_plot_bbox_from_keypoints(frame_keypoints, skeleton, j, 4, 5);
                                }
                            }

                            if(skeleton->has_bbox) {
                                // draw all animals
                                for (u32 animal_id=0; animal_id < number_of_animals; animal_id++) {
                                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[animal_id];
                                    ImColor bbox_color = current_frame_data->colors[animal_id];
                                    if (frame_keypoints->bbox2d->state != RectNull) {
                                        ImPlotRect* my_rect = frame_keypoints->bbox2d->rect;
                                        ImPlot::DragRect(0,&my_rect->X.Min,&my_rect->Y.Min,&my_rect->X.Max,&my_rect->Y.Max, bbox_color);
                                    }
                                }

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
            if (keypoints_find) {
        
                Animals* current_frame_data = keypoints_map[current_frame_num];
                    
                if (ImGui::Begin("Keypoints")) {

                    if (ImGui::BeginTable("##Animals", number_of_animals, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
                    {
                        for (int animal_id = 0; animal_id < number_of_animals; animal_id++)
                        {
                            char label[32];
                            sprintf(label, "Ani %d", animal_id);
                            ImGui::TableNextColumn();
                            if(ImGui::Selectable(label, current_frame_data->active_id == animal_id)) {
                                current_frame_data->active_id = animal_id;
                            }
                            if (current_frame_data->keypoints[animal_id].has_labels) {
                                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, current_frame_data->colors[animal_id]);
                            }
                        }
                        ImGui::EndTable();
                    }


                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                    if (skeleton->has_skeleton) {                    
                        const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();
                        {
                            const int rows_count = scene->num_cams;
                            const int columns_count= skeleton->num_nodes+1;

                            static ImGuiTableFlags table_flags = ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_Hideable | ImGuiTableFlags_Resizable | ImGuiTableFlags_HighlightHoveredColumn;

                            if (ImGui::BeginTable("table_angled_headers", columns_count, table_flags, ImVec2(0.0f, TEXT_BASE_HEIGHT * 12)))
                            {
                                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_NoReorder);
                                for (int column = 1; column < columns_count; column++)
                                    ImGui::TableSetupColumn(skeleton->node_names[column-1].c_str(), ImGuiTableColumnFlags_AngledHeader | ImGuiTableColumnFlags_WidthFixed);

                                ImGui::TableAngledHeadersRow(); // Draw angled headers for all columns with the ImGuiTableColumnFlags_AngledHeader flag.
                                ImGui::TableHeadersRow();       // Draw remaining headers and allow access to context-menu and other functions.

                                for (int row = 0; row < rows_count; row++)
                                {
                                    ImGui::PushID(row);
                                    ImGui::TableNextRow();

                                    if (is_view_focused[row] && keypoints_find) {
                                        ImU32 row_bg_color = ImGui::GetColorU32(ImVec4(0.7f, 0.3f, 0.3f, 0.65f)); 
                                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, row_bg_color);
                                    }

                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::AlignTextToFramePadding();
                                    ImGui::Text(camera_names[row].c_str());
                                    for (int column = 1; column < columns_count; column++)
                                        if (ImGui::TableSetColumnIndex(column))
                                        {
                                            if (keypoints_find) {
                                                ImVec4 node_color;
                                                if(frame_keypoints->active_kp_id[row] == column-1)
                                                {
                                                    node_color = (ImVec4)ImColor::HSV(0.8, 0.9f, 0.9f);
                                                }  else {
                                                    if (frame_keypoints->keypoints2d[row][column-1].is_labeled) 
                                                    {
                                                        node_color.w = 0.5;
                                                        node_color.x = skeleton->node_colors[column-1].x;
                                                        node_color.y = skeleton->node_colors[column-1].y;
                                                        node_color.z = skeleton->node_colors[column-1].z;
                                                    }

                                                    if (frame_keypoints->keypoints2d[row][column-1].is_triangulated) 
                                                    {
                                                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "T");
                                                    }    
                                                }
                                                
                                                ImU32 cell_bg_color = ImGui::GetColorU32(node_color);
                                                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                                            }
                                        }
                                    ImGui::PopID();
                                }
                                ImGui::EndTable();
                            }
                        }
                    }
                }
                ImGui::End();
            }
        }

        if (plot_keypoints_flag)
        {
            if (ImGui::Begin("Labeling Tool"))
            {

                if (ImGui::Button("Save Labeled Data"))
                {
                    // save_keypoints(keypoints_map, skeleton, keypoints_root_folder, scene->num_cams, camera_names);
                }

                if (ImGui::Button("Load Labeled Data"))
                {
                    // load_keypoints(keypoints_map, skeleton, keypoints_root_folder, scene, camera_names);
                }

                // TODO: change folder
                ImGui::Text(keypoints_root_folder.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Update keypoints folder")) {
                    change_keypoints_folder = true;
                    file_dialog.Open();
                }

                if (ImGui::Button("Load 2d Keypoints Only"))
                {
                    // for (int i=0; i<scene->num_cams; i++) {
                    //     load_2d_keypoints(keypoints_map, skeleton, keypoints_root_folder, i, camera_names[i], scene);
                    // }
                }

                auto upper_it = keypoints_map.upper_bound(current_frame_num); 
                if (upper_it == keypoints_map.end()) {
                    ImGui::Text("Number of labeled frame : %d", (*upper_it).first);
                } else {
                    ImGui::Text("Next labeled frame : %d", (*upper_it).first);
                }
            }
            ImGui::End();
        }


        if (ImGui::IsKeyPressed(ImGuiKey_H, false))
        {
            show_help_window = !show_help_window;
        }

        if (show_help_window)
        {
            if (ImGui::Begin("Help Menu")) {            
                ImGui::Text("Space -> Toggle play and pause");
                ImGui::Text(", -> Previous image");
                ImGui::Text(". -> Next image");

                ImGui::SeparatorText("While hovering image");
                ImGui::Text("Z -> Create keypoints on frame");
                ImGui::Text("W -> Drop active keypoint");
                ImGui::Text("A -> Active keypoint++ ");
                ImGui::Text("D -> Active keypoint--");
                ImGui::Text("V -> Delete all keypoint");
                ImGui::Text("Q -> Active keypoint set to first node");
                ImGui::Text("E -> Active keypoint set to last node");

                ImGui::SeparatorText("While hovering keypoints");
                ImGui::Text("R -> Delete active keypoint");
                ImGui::Text("F -> Delete active keypoint on all cameras");
                ImGui::Text("Click keypoint to active it");
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
