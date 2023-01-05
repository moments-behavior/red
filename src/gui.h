#ifndef RED_GUI
#define RED_GUI
#include "render.h"
#include <imfilebrowser.h>

struct ProjectContext{
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};


static void render_project_dialogue(ProjectContext *project, ImGui::FileBrowser *file_dialog)
{
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
}

#endif
