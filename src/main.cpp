#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Logger.h"
#include "render.h"
#include "implot.h"
#include <iostream> // std::cout
#include <thread>   // std::thread
#include <imfilebrowser.h>
#include <camera.h>
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

    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory);
    file_dialog.SetTitle("Select working directory");

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
        .gpu_index = 0};

    int to_display_frame_number = 0;
    int read_head = 0;

    bool play_video = false;
    bool toggle_play_status = false;

    int slider_frame_number = 0;
    bool just_seeked = false;
    bool slider_just_changed = false;
    bool video_loaded = false;
    bool plot_keypoints_flag = false;
    int current_frame_num = 0;

    ImGuiIO &io = ImGui::GetIO();
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

                    if (video_loaded)
                    {
                        if (ImGui::MenuItem("Label"))
                        {   
                            std::string cam_file = root_dir + "/calibration/calibration.csv";
                            for (u32 i = 0; i < scene->num_cams; i++)
                            {
                                CameraParams cam = camera_load_params_from_csv(cam_file, i);
                                camera_params.push_back(cam);
                            }
                            plot_keypoints_flag = true;
                        };
                    }

                    ImGui::EndMenu();
                }

                ImGui::EndMenuBar();
            }

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            // ImGui::Text("Frame number %d ", display_buffer[0][read_head].frame_number);
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
            render_allocate_scene_memory(scene, 3208, 2200, input_file_names.size(), 32);

            // multiple threads for decoding for selected videos
            for (u32 i = 0; i < scene->num_cams; i++)
            {
                std::size_t cam_string_position = input_file_names[i].find("Cam");           // position of "Cam" in str
                std::string cam_string = input_file_names[i].substr(cam_string_position, 4); // get from "Cam" to the end
                camera_names.push_back(cam_string);
                std::cout << "camera names: " << cam_string << std::endl;
                decoder_threads.push_back(std::thread(&decoder_process, input_file_names[i].c_str(), dc_context, scene->display_buffer[i], scene->size_of_buffer, &scene->seek_context[i]));
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
                    // std::cout << display_buffer[read_head].frame_number << ", " << to_display_frame_number << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                upload_texture(&scene->image_texture[j], scene->display_buffer[j][read_head].frame, 3208, 2200);
            }
            current_frame_num = to_display_frame_number;
        }

        // show frames in the buffer if selected
        if (video_loaded && (!play_video))
        {
            static int selected = 0;
            static int select_corr_head = 0;
            ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frames in the buffer", NULL, ImGuiWindowFlags_MenuBar))
            {
                {
                    for (u32 i = 0; i < scene->size_of_buffer; i++)
                    {
                        char label[128];
                        sprintf(label, "Buffer %d", i);
                        if (ImGui::Selectable(label, selected == i))
                        {
                            // start from the lowest frame
                            selected = i;
                            select_corr_head = (i + read_head) % scene->size_of_buffer;

                            // if not playing the video, then show what's in the buffer
                            if (!play_video)
                            {
                                for (int j = 0; j < scene->num_cams; j++)
                                {
                                    upload_texture(&scene->image_texture[j], scene->display_buffer[j][read_head].frame, 3208, 2200);
                                }
                            }
                        }
                    }
                }

                ImGui::Separator();

                if (ImGui::Button(ICON_FK_MINUS) || ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true))
                {
                    if (selected > 0)
                    {
                        selected--;
                        select_corr_head = (selected + read_head) % scene->size_of_buffer;

                        for (int j = 0; j < scene->num_cams; j++)
                        {
                            upload_texture(&scene->image_texture[j], scene->display_buffer[j][read_head].frame, 3208, 2200);
                        }
                    }
                };

                ImGui::SameLine();
                if (ImGui::Button(ICON_FK_PLUS) || ImGui::IsKeyPressed(ImGuiKey_RightBracket, true))
                {
                    if (selected < (scene->size_of_buffer - 1))
                    {
                        selected++;
                        select_corr_head = (selected + read_head) % scene->size_of_buffer;

                        if (!play_video)
                        {
                            for (u32 j = 0; j < scene->num_cams; j++)
                            {
                                upload_texture(&scene->image_texture[j], scene->display_buffer[j][read_head].frame, 3208, 2200);
                            }
                        }
                    }
                };
            }
            ImGui::Text("Frame number selected: %d", scene->display_buffer[0][select_corr_head].frame_number);

            if (!play_video)
            {
                select_corr_head = (selected + read_head) % scene->size_of_buffer;
                current_frame_num = scene->display_buffer[0][select_corr_head].frame_number;
            }
            ImGui::End();
        }

        if (toggle_play_status && play_video)
        {
            play_video = false;
            toggle_play_status = false;
        }

        // Render a video frame
        if (video_loaded)
        {
             
            for (int j = 0; j < scene->num_cams; j++)
            {
                // layout
                ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
                ImVec2 window_pos;
                if(j % 2 == 0){
                    window_pos.y = 200.0; window_pos.x = (j/2.0) * 500;}
                else{
                    window_pos.y = 600.0; window_pos.x = (j-1)/2.0 * 500;}

                ImGui::SetNextWindowPos(window_pos, ImGuiCond_FirstUseEver);
                ImGui::Begin(camera_names[j].c_str());
                ImGui::BeginGroup();

                std::string scene_name = "scene view" + std::to_string(j);
                ImGui::BeginChild(scene_name.c_str(), ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below

                ImVec2 avail_size = ImGui::GetContentRegionAvail();
                // ImGui::Image((void*)(intptr_t)image_texture[j], avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size))
                {
                    ImPlot::PlotImage("##no_image_name", (void *)(intptr_t)scene->image_texture[j], ImVec2(0, 0), ImVec2(3208, 2200));
                    
                    if(plot_keypoints_flag){
                        // plot arena for testing camera parameters 
                        float arena_x[100]; float arena_y[100]; 
                        camera_arena_projection_points(&camera_params.at(j), arena_x, arena_y, 100);
                        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, ImVec4(1.0, 1.0, 1.0,1.0));
                        ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0,1.0), 3.0);
                        std::string name = "perimeter " + camera_names[j];
                        ImPlot::PlotLine(name.c_str(), arena_x, arena_y, 100);                        
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
                        }

                        for (int i = 0; i < scene->num_cams; i++)
                        {
                            // synchronize seeking
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
                    }
                }
                else
                {
                    if (ImGui::Button(play_video ? ICON_FK_PAUSE : ICON_FK_PLAY))
                    {
                        play_video = !play_video;
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
                    }

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        // synchronize seeking
                        while (!(scene->seek_context[i].seek_done))
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            std::cout << "Seeking Cam" << i << std::endl;
                        }
                    }

                    for (int i = 0; i < scene->num_cams; i++)
                    {
                        scene->seek_context[i].seek_done = false;
                    }

                    to_display_frame_number = scene->seek_context[0].seek_frame;
                    read_head = 0;
                    just_seeked = true;
                }

                ImGui::EndGroup();
                ImGui::End();
            }
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

        if (dc_context->decoding_flag && play_video && (!just_seeked) && (to_display_frame_number < (dc_context->total_num_frame - 1)))
        {
            to_display_frame_number++;

            for (int j = 0; j < scene->num_cams; j++)
            {
                scene->display_buffer[j][read_head].available_to_write = true;
            }

            read_head = (read_head + 1) % scene->size_of_buffer;
            slider_frame_number = to_display_frame_number;
        }

        if (just_seeked)
        {
            just_seeked = false;
            play_video = true;
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
