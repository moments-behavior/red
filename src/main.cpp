#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <GL/glew.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "create_image_cuda.h"
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"
#include "Logger.h"
#include "render.h"
#include "decoder.h"
#include "IconsForkAwesome.h"
#include "implot.h"
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <imfilebrowser.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


int main(int, char**)
{

    window_context *window_c = (window_context *)malloc(sizeof(window_context));
    *window_c = (window_context) {
        .swap_interval = 1, // use vsync
        .width = 1920, 
        .height = 1080,
    };

    initialize_render_target(window_c);


    // ************* Dear Imgui ********************//
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlotContext* implotCtx = ImPlot::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows


    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }


    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_c->render_target, true);
    const char* glsl_version = "#version 130";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load a nice font
    io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 15.0f);
    // merge in icons from Font Awesome
    static const ImWchar icons_ranges[] = { ICON_MIN_FK, ICON_MAX_16_FK, 0 };
    ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF("fonts/forkawesome-webfont.ttf", 15.0f, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid

    // Create a OpenGL texture identifier
    int num_cams;
    GLuint* image_texture;
    PictureBuffer** display_buffer;
    SeekInfo* seek_context;
    
    // Our state
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory);
    file_dialog.SetTitle("Select work directory");

    
    double result = 0.0;
    unsigned long long num_heads = 0;
    unsigned long long num_tails = 0;
    
    
    int size_pic = 3208 * 2200 * 4 *  sizeof(unsigned char);

    // allocate display buffer
    const int size_of_buffer = 32;

    
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;


    std::vector<std::thread> decoder_threads;
    bool* decoding_flag = new bool(false);
    bool* stop_flag = new bool(false);
    int* total_num_frame = new int(INT_MAX);
    int* estimated_num_frames = new int(0);

    int gpu_index = 0;
    int to_display_frame_number = 0;
    int read_head = 0;

    bool play_video = false;
    bool toggle_play_status = false;

    static bool show_app_layout = true;


    int slider_frame_number = 0;
    bool just_seeked = false;

    bool slider_just_changed = false;
    bool video_loaded = false;
    bool plot_keypoints_flag = false;
    int current_frame_num = 0;


    while (!glfwWindowShouldClose(window_c->render_target))
    {
        // todo: increment this draw_id after each ImGui and ImPlot draw request (e.g. with ImPlot::DragPoint)
        int draw_id = 0;
        int* draw_id_ptr = &draw_id;


        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();


        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
       
        if (ImGui::Begin("File Browser",  NULL, ImGuiWindowFlags_MenuBar))
        {

            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open")) { file_dialog.Open(); };

                    if (video_loaded){
                        if (ImGui::MenuItem("Label")) { 
                            
                        };
                    }

                    ImGui::EndMenu();
                }
                
                ImGui::EndMenuBar();
            }

            static float f = 0.0f;
            static int counter = 0;
            ImGui::Text("Flip a coin here!");
            ImGui::SameLine();
            if (ImGui::Button("Flip!")) {
                result = ((double)rand() / (RAND_MAX));
                if (result > 0.5) {
                    num_heads++;
                }
                else {
                    num_tails++;
                }
            }
            if (result > 0.5) {
                ImGui::Text("Heads!");
            }
            else {
                ImGui::Text("Tails!");
            }
            if ((num_heads + num_tails) > 0) {
                ImGui::Text("Proportion heads: %.3f", (float)num_heads / (num_heads + num_tails));
            }
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Frame number %d ", display_buffer[0][read_head].frame_number);             
        }
        ImGui::End();
        
        file_dialog.Display();

        if (file_dialog.HasSelected())
        {
            root_dir = file_dialog.GetSelected().string();

            // load movies 
            std::string movie_dir = root_dir + "/movies";

            for (const auto & entry : std::filesystem::directory_iterator(movie_dir)){
                input_file_names.push_back(entry.path().string());
            }
                
            std::sort(input_file_names.begin(), input_file_names.end());
            num_cams = input_file_names.size();


            image_texture = (GLuint *) malloc(sizeof(GLuint) * num_cams);

            for(int j=0; j<num_cams; j++){
                glGenTextures(1, &image_texture[j]);
                glBindTexture(GL_TEXTURE_2D, image_texture[j]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                // Setup filtering parameters for display
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
            }

            seek_context = (SeekInfo *) malloc(sizeof(SeekInfo) * num_cams);
            for(int j=0; j<num_cams; j++){
                seek_context[j].use_seek=false;
                seek_context[j].seek_frame=0;
                seek_context[j].seek_done=false;
            }

            display_buffer = (PictureBuffer**)malloc(num_cams * sizeof(PictureBuffer*));
            for (int j=0; j<num_cams; j++){
                display_buffer[j] = (PictureBuffer*)malloc(size_of_buffer * sizeof(PictureBuffer));
            }

            for(int j=0; j<num_cams; j++){
                for (int i = 0; i < size_of_buffer; i++) {
                    display_buffer[j][i].frame = (unsigned char*)malloc(size_pic);
                    clear_buffer_with_constant_image(display_buffer[j][i].frame, 3208, 2200);

                    display_buffer[j][i].frame_number = 0;
                    display_buffer[j][i].available_to_write = true;
                }
            }
    


            // multiple threads for decoding for selected videos 
            for(unsigned int i = 0; i < num_cams; i++)
            {
                std::size_t cam_string_position = input_file_names[i].find("Cam");      // position of "Cam" in str
                std::string cam_string = input_file_names[i].substr(cam_string_position, 4);     // get from "Cam" to the end
                camera_names.push_back(cam_string);
                std::cout << "camera names: " << cam_string << std::endl;
                decoder_threads.push_back(std::thread(&decoder_process, input_file_names[i].c_str(), gpu_index, display_buffer[i], decoding_flag, size_of_buffer, stop_flag, &seek_context[i], total_num_frame, estimated_num_frames));
            }

            video_loaded = true;
            file_dialog.ClearSelected();
        }
        
        
        if (*decoding_flag && play_video) {
            for(int j=0; j<num_cams; j++){
                // if the current frame is ready, upload for display, otherwise wait for the frame to get ready 
                while (display_buffer[j][read_head].frame_number != to_display_frame_number) {
                    //std::cout << display_buffer[read_head].frame_number << ", " << to_display_frame_number << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                bind_texture(&image_texture[j]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[j][read_head].frame);
                unbind_texture();
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
                    for (int i = 0; i < size_of_buffer; i++)
                    {
                        char label[128];
                        sprintf(label, "Buffer %d", i);
                        if (ImGui::Selectable(label, selected == i)) {
                            // start from the lowest frame
                            selected = i;
                            select_corr_head = (i + read_head) % size_of_buffer;

                            // if not playing the video, then show what's in the buffer
                            if (!play_video) {
                                for(int j=0; j<num_cams; j++){
                                    bind_texture(&image_texture[j]);
                                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[j][select_corr_head].frame);
                                    unbind_texture();
                                }
                            }
                        }
                    }
                }

                ImGui::Separator();
                
                if (ImGui::Button(ICON_FK_MINUS) || ImGui::IsKeyPressed(ImGuiKey_LeftBracket, true)) {
                    if (selected > 0) {
                        selected--;
                        select_corr_head = (selected + read_head) % size_of_buffer;

                        for(int j=0; j<num_cams; j++){
                            bind_texture(&image_texture[j]);
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[j][select_corr_head].frame);
                            unbind_texture();
                        }
                        
                    }
                };
                
                ImGui::SameLine();
                if (ImGui::Button(ICON_FK_PLUS) || ImGui::IsKeyPressed(ImGuiKey_RightBracket, true)) {
                    if (selected < (size_of_buffer - 1)) {
                        selected++;
                        select_corr_head = (selected + read_head) % size_of_buffer;

                        if (!play_video) {
                            for(int j=0; j<num_cams; j++){
                                bind_texture(&image_texture[j]);
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[j][select_corr_head].frame);
                                unbind_texture();
                            }
                        }
                    }
                };
            }
            ImGui::Text("Frame number selected: %d", display_buffer[0][select_corr_head].frame_number);
            
            if(!play_video){
                select_corr_head = (selected + read_head) % size_of_buffer;
                current_frame_num = display_buffer[0][select_corr_head].frame_number;
            }
            ImGui::End();
        }


        if (toggle_play_status && play_video) {
            play_video = false;
            toggle_play_status = false;
        }


        // Render a video frame
        if (video_loaded){        
            for(int j=0; j<num_cams; j++)
            {
                ImGui::Begin(camera_names[j].c_str());           
                ImGui::BeginGroup();

                std::string scene_name = "scene view" + std::to_string(j);
                ImGui::BeginChild(scene_name.c_str(),  ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below
                ImVec2 avail_size = ImGui::GetContentRegionAvail();
                
                //ImGui::Image((void*)(intptr_t)image_texture[j], avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size)){
                    ImPlot::PlotImage("##no_image_name", (void*)(intptr_t)image_texture[j], ImVec2(0,0), ImVec2(3208, 2200));    
                    ImPlot::EndPlot();
                }

                ImGui::EndChild();

                if (to_display_frame_number == (*total_num_frame - 1)) {
                    if (ImGui::Button(ICON_FK_REPEAT)) {
                        // seek to zero
                        for(int i=0; i<num_cams; i++){
                            seek_context[i].seek_frame = 0;
                            seek_context[i].use_seek = true;
                        }


                        for(int i=0; i<num_cams; i++){
                            // synchronize seeking
                            while (!(seek_context[i].seek_done)) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                        }
                        
                        for(int i=0; i<num_cams; i++){
                            seek_context[i].seek_done = false;
                        }

                        to_display_frame_number = seek_context[0].seek_frame;
                        read_head = 0;
                        just_seeked = true;
                    }
                }
                else {
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
                if (ImGui::Button(ICON_FK_PLUS)) {
                    // advance_clicks++;
                    play_video = true;
                    toggle_play_status = true;
                }
                ImGui::PopButtonRepeat();
                ImGui::SameLine();

                slider_just_changed = ImGui::SliderInt("##frame count", &slider_frame_number, 0, *estimated_num_frames);
                
                if (slider_just_changed){
                    std::cout << "main, seeking: " << slider_frame_number << std::endl;
                    
                    for(int i=0; i<num_cams; i++){
                        seek_context[i].seek_frame = (uint64_t)slider_frame_number;
                        seek_context[i].use_seek = true;
                    }


                    for(int i=0; i<num_cams; i++){
                        // synchronize seeking
                        while (!(seek_context[i].seek_done)) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            std::cout << "Seeking Cam" << i << std::endl;
                        }
                        
                    }
                    
                    for(int i=0; i<num_cams; i++){
                        seek_context[i].seek_done = false;
                    }

                    to_display_frame_number = seek_context[0].seek_frame;
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
        glfwGetFramebufferSize(window_c->render_target, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        
        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window_c->render_target);
 
        
        if(*decoding_flag && play_video && (!just_seeked) && (to_display_frame_number < (*total_num_frame-1))){
            to_display_frame_number++;

            for(int j=0; j<num_cams; j++){
                display_buffer[j][read_head].available_to_write = true;
            }

            read_head = (read_head + 1) % size_of_buffer;
            slider_frame_number = to_display_frame_number;
        }
        
        if (just_seeked) {
            just_seeked = false; play_video = true; 
        }


    }


    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window_c->render_target);
    glfwTerminate();

    *stop_flag = true;
    // wait for threads to join
    for (auto& t : decoder_threads)
        t.join();

    return 0;
}
