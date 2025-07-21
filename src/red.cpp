#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <algorithm>
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
#include <torch/torch.h>
#include <torch/script.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

struct YoloPrediction {
    float x, y, w, h;
    float confidence;
    int class_id;
};

struct YoloBBox {
    double x_min, y_min, x_max, y_max;
    float confidence;
    int class_id;
    bool is_valid;
    
    YoloBBox() : x_min(0), y_min(0), x_max(0), y_max(0), confidence(0), class_id(-1), is_valid(false) {}
    
    YoloBBox(const YoloPrediction& pred) {
        x_min = pred.x - pred.w / 2.0;
        y_min = pred.y - pred.h / 2.0;
        x_max = pred.x + pred.w / 2.0;
        y_max = pred.y + pred.h / 2.0;
        confidence = pred.confidence;
        class_id = pred.class_id;
        is_valid = true;
    }
};

// Non-Maximum Suppression helper function
float calculateIoU(const YoloPrediction& a, const YoloPrediction& b) {
    float x1_a = a.x - a.w / 2.0f;
    float y1_a = a.y - a.h / 2.0f;
    float x2_a = a.x + a.w / 2.0f;
    float y2_a = a.y + a.h / 2.0f;
    
    float x1_b = b.x - b.w / 2.0f;
    float y1_b = b.y - b.h / 2.0f;
    float x2_b = b.x + b.w / 2.0f;
    float y2_b = b.y + b.h / 2.0f;
    
    float x1_inter = std::max(x1_a, x1_b);
    float y1_inter = std::max(y1_a, y1_b);
    float x2_inter = std::min(x2_a, x2_b);
    float y2_inter = std::min(y2_a, y2_b);
    
    if (x2_inter <= x1_inter || y2_inter <= y1_inter) {
        return 0.0f;
    }
    
    float intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - intersection;
    
    return intersection / union_area;
}

std::vector<YoloPrediction> applyNMS(std::vector<YoloPrediction>& predictions, float iou_threshold = 0.5f) {
    std::sort(predictions.begin(), predictions.end(), 
              [](const YoloPrediction& a, const YoloPrediction& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(predictions.size(), false);
    std::vector<YoloPrediction> result;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(predictions[i]);
        
        for (size_t j = i + 1; j < predictions.size(); ++j) {
            if (suppressed[j]) continue;
            
            if (predictions[i].class_id == predictions[j].class_id) {
                float iou = calculateIoU(predictions[i], predictions[j]);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return result;
}

std::vector<YoloPrediction> runYoloInference(const std::string& model_path, unsigned char* frame_data, int width, int height) {
    std::vector<YoloPrediction> predictions;
    
    try {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file does not exist: " << model_path << std::endl;
            throw std::runtime_error("Model file not found");
        }
        
        if (std::filesystem::file_size(model_path) == 0) {
            std::cerr << "Model file is empty: " << model_path << std::endl;
            throw std::runtime_error("Model file is empty");
        }
        
        std::cout << "Loading YOLO model from: " << model_path << std::endl;
        std::cout << "Model file size: " << std::filesystem::file_size(model_path) << " bytes" << std::endl;
        
        torch::jit::script::Module module;
        try {
            module = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            std::cerr << "Failed to load TorchScript model. This might be caused by:" << std::endl;
            std::cerr << "1. The file is not a valid TorchScript model (.pt format)" << std::endl;
            std::cerr << "2. The model was saved with a different PyTorch version" << std::endl;
            std::cerr << "3. The model file is corrupted" << std::endl;
            std::cerr << "4. The model is in .pth format (state dict) instead of .pt (TorchScript)" << std::endl;
            std::cerr << "PyTorch error: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load TorchScript model");
        }
        
        module.eval();
        
        if (!frame_data) {
            std::cerr << "Frame data is null" << std::endl;
            throw std::runtime_error("Invalid frame data");
        }
        
        std::cout << "Converting frame data to tensor (size: " << width << "x" << height << ")" << std::endl;
        
        torch::Tensor tensor = torch::from_blob(frame_data, {height, width, 4}, torch::kUInt8);
        tensor = tensor.slice(2, 0, 3); 
        tensor = tensor.permute({2, 0, 1}); 
        tensor = tensor.to(torch::kFloat) / 255.0;
        
        tensor = torch::nn::functional::interpolate(tensor.unsqueeze(0), 
            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{640, 640}).mode(torch::kBilinear).align_corners(false));
        
        std::cout << "Running inference..." << std::endl;
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        torch::Tensor output;
        try {
            output = module.forward(inputs).toTensor();
        } catch (const std::exception& e) {
            std::cerr << "Inference failed. The model might expect different input format." << std::endl;
            std::cerr << "Inference error: " << e.what() << std::endl;
            throw std::runtime_error("Inference failed");
        }
        
        std::cout << "Inference completed. Output shape: [";
        for (int i = 0; i < output.dim(); ++i) {
            std::cout << output.size(i);
            if (i < output.dim() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        if (output.dim() == 3 && output.size(1) == 6) {
            output = output.permute({0, 2, 1}); 
        }
        output = output.squeeze(0); 
        
        float conf_threshold = 0.25f;
        float scale_x = static_cast<float>(width) / 640.0f;
        float scale_y = static_cast<float>(height) / 640.0f;
        
        auto output_accessor = output.accessor<float, 2>();
        int num_detections = output.size(0);
        int output_features = output.size(1);
        
        std::cout << "Processing " << num_detections << " detections with " << output_features << " features per detection" << std::endl;
        
        for (int i = 0; i < num_detections; ++i) {
            float confidence, class_score;
            int class_id;
            
            if (output_features == 6) {
                confidence = output_accessor[i][4];
                class_id = static_cast<int>(output_accessor[i][5]);
            } else if (output_features >= 84) {
                float max_score = 0.0f;
                int max_class = -1;
                for (int c = 0; c < (output_features - 4); ++c) {
                    float score = output_accessor[i][4 + c];
                    if (score > max_score) {
                        max_score = score;
                        max_class = c;
                    }
                }
                confidence = max_score;
                class_id = max_class;
            } else {
                std::cerr << "Unexpected output format with " << output_features << " features" << std::endl;
                continue;
            }
            
            if (confidence > conf_threshold) {
                float cx_model = output_accessor[i][0];
                float cy_model = output_accessor[i][1];
                float w_model = output_accessor[i][2];
                float h_model = output_accessor[i][3];
                
                YoloPrediction pred;
                // Scale coordinates back to original image size
                pred.x = cx_model * scale_x;
                pred.y = height - (cy_model * scale_y); 
                pred.w = w_model * scale_x;
                pred.h = h_model * scale_y;
                pred.confidence = confidence;
                pred.class_id = class_id;
                
                predictions.push_back(pred);
            }
        }
        
        std::cout << "Found " << predictions.size() << " raw detections before NMS" << std::endl;
        
        predictions = applyNMS(predictions, 0.45f);  
        
        std::cout << "YOLO inference completed. Found " << predictions.size() << " detections after NMS." << std::endl;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            const auto& pred = predictions[i];
            std::cout << "Final Detection " << i << ": center=(" << pred.x << "," << pred.y 
                     << ") size=(" << pred.w << "," << pred.h << ") conf=" << pred.confidence 
                     << " class=" << pred.class_id << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "PyTorch inference error: " << e.what() << std::endl;
    }
    
    return predictions;
}

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
    bool allow_exit = true;
    std::map<std::string, SkeletonPrimitive> skeleton_map;
    std::time_t last_saved = time(NULL);
    
    // others
    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory);
    ImGui::FileBrowser model_file_dialog;
    #ifdef _WIN32
        std::string cwd = std::filesystem::current_path().string();
    #else
        std::filesystem::path cwd = std::filesystem::current_path();
        std::string delimiter = "/";
        std::vector<std::string> tokenized_path = string_split (cwd, delimiter);
        std::string start_folder_name = "/home/" + tokenized_path[2] + "/data";
        file_dialog.SetPwd(start_folder_name);
        model_file_dialog.SetPwd(start_folder_name);
    #endif 
    
    file_dialog.SetTitle("Select working directory");
    model_file_dialog.SetTitle("Select YOLO Model");
    model_file_dialog.SetTypeFilters({ ".pt", ".pth" });
    
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();

    bool show_world_coordinates = false;
    std::string keypoints_root_folder;
    bool change_keypoints_folder =false;
    std::string yolo_model_path;
    bool model_selected = false;
    std::vector<std::vector<YoloPrediction>> yolo_predictions; // per camera predictions
    std::vector<std::vector<YoloBBox>> yolo_bboxes; 
    std::map<int, std::vector<std::vector<YoloBBox>>> yolo_frame_cache; 
    bool show_yolo_predictions = false;
    bool auto_yolo_labeling = false; 
    int last_yolo_frame = -1; 
    int label_buffer_size = 32;
    bool show_help_window = false;
    std::vector<bool> is_view_focused;

    ImPlotStyle& style              = ImPlot::GetStyle();
    ImVec4* colors                  = style.Colors;
    colors[ImPlotCol_Crosshairs]    = ImVec4(0.3f, 0.10f, 0.64f, 1.00f);

    while (!glfwWindowShouldClose(window->render_target) || !allow_exit) {
        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (glfwWindowShouldClose(window->render_target)) {
            ImGui::OpenPopup("Warning");
        }
        if (ImGui::BeginPopupModal("Warning")) {
            ImGui::Text("You have unsaved changes");
            if (ImGui::Button("Cancel")) {
                glfwSetWindowShouldClose(window->render_target, GLFW_FALSE);
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Close without saving")) {
                allow_exit = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Save before closing")) {
                save_keypoints(keypoints_map, skeleton, keypoints_root_folder, scene->num_cams, camera_names, number_of_animals);
                allow_exit = true;
                last_saved = time(NULL);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        
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
                                skeleton_chosen = true;
                                skeleton = new SkeletonContext;
                                skeleton_initialize(element.first, root_dir, skeleton, element.second);
                                plot_keypoints_flag = true;
                                keypoints_root_folder = root_dir + "/labeled_data/";
                                // create folders
                                std::filesystem::create_directory(keypoints_root_folder);
                                std::filesystem::create_directory(keypoints_root_folder + "/worldKeyPoints");
                                for (u32 i = 0; i < scene->num_cams; i++) {
                                    std::filesystem::create_directory(keypoints_root_folder + "/" + camera_names[i]);
                                }
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
        model_file_dialog.Display();

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

                yolo_predictions.resize(scene->num_cams);
                yolo_bboxes.resize(scene->num_cams);

                video_loaded = true;
            }
            file_dialog.ClearSelected();
        }

        if (model_file_dialog.HasSelected())
        {
            yolo_model_path = model_file_dialog.GetSelected().string();
            model_selected = true;
            std::cout << "Selected YOLO model: " << yolo_model_path << std::endl;
            model_file_dialog.ClearSelected();
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

                static int previous_frame_num = -1;
        if (video_loaded && current_frame_num != previous_frame_num) {
            if (yolo_frame_cache.find(current_frame_num) != yolo_frame_cache.end()) {
                std::cout << "Frame " << current_frame_num << " already processed by YOLO" << std::endl;
            } else {
                for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                    yolo_bboxes[cam_id].clear();
                }
                
                if (auto_yolo_labeling && model_selected) {
                    std::cout << "Running automatic YOLO prediction on frame " << current_frame_num << std::endl;
                    
                    for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                        unsigned char* frame_data = nullptr;
                        if (play_video && dc_context->decoding_flag) {
                            frame_data = scene->display_buffer[cam_id][read_head].frame;
                        } else if (!play_video) {
                            int select_corr_head = (pause_selected + read_head) % scene->size_of_buffer;
                            frame_data = scene->display_buffer[cam_id][select_corr_head].frame;
                        }
                        
                        if (frame_data) {
                            yolo_predictions[cam_id] = runYoloInference(yolo_model_path, frame_data, 
                                                                      scene->image_width[cam_id], scene->image_height[cam_id]);
                            
                            for (const auto& pred : yolo_predictions[cam_id]) {
                                yolo_bboxes[cam_id].emplace_back(pred);
                            }
                            
                            std::cout << "Camera " << cam_id << ": Found " << yolo_predictions[cam_id].size() << " detections" << std::endl;
                        }
                    }
                    
                    if (!yolo_bboxes.empty() && std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(), 
                        [](const auto& cam_bboxes) { return !cam_bboxes.empty(); })) {
                        
                        if (!keypoints_find) {
                            Animals* animals = (Animals *)malloc(sizeof(Animals));
                            allocate_keypoints(animals, scene, skeleton, number_of_animals);
                            keypoints_map[current_frame_num] = animals; 
                            keypoints_find = true;
                        }
                        
                        Animals* current_frame_data = keypoints_map[current_frame_num];
                        
                        std::map<int, int> class_to_animal; // Map class_id to animal_id
                        int next_animal_id = 0;
                        
                        for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                            for (size_t bbox_idx = 0; bbox_idx < yolo_bboxes[cam_id].size(); bbox_idx++) {
                                const auto& yolo_bbox = yolo_bboxes[cam_id][bbox_idx];
                                if (yolo_bbox.is_valid) {
                                    if (class_to_animal.find(yolo_bbox.class_id) == class_to_animal.end()) {
                                        class_to_animal[yolo_bbox.class_id] = next_animal_id % number_of_animals;
                                        next_animal_id++;
                                    }
                                    
                                    int target_animal_id = class_to_animal[yolo_bbox.class_id];
                                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[target_animal_id];
                                    
                                    BoundingBox new_bbox;
                                    new_bbox.rect = new ImPlotRect(yolo_bbox.x_min, yolo_bbox.x_max, yolo_bbox.y_min, yolo_bbox.y_max);
                                    new_bbox.state = RectTwoPoints;
                                    new_bbox.class_id = yolo_bbox.class_id;
                                    new_bbox.confidence = yolo_bbox.confidence;
                                    
                                    frame_keypoints->bbox2d_list[cam_id].push_back(new_bbox);
                                    frame_keypoints->has_labels = true;
                                    allow_exit = false;
                                    
                                    std::cout << "Auto-converted YOLO bbox (class " << yolo_bbox.class_id 
                                              << ") to animal " << target_animal_id << ", camera " << cam_id << std::endl;
                                }
                            }
                        }
                    }
                    
                    yolo_frame_cache[current_frame_num] = std::vector<std::vector<YoloBBox>>(scene->num_cams);
                    for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                        yolo_bboxes[cam_id].clear();
                    }
                }
            }
            previous_frame_num = current_frame_num;
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

                    if (show_yolo_predictions && j < yolo_bboxes.size()) {
                        for (size_t bbox_idx = 0; bbox_idx < yolo_bboxes[j].size(); ++bbox_idx) {
                            auto& bbox = yolo_bboxes[j][bbox_idx];
                            if (bbox.is_valid) {
                                // Check if this YOLO bbox has already been converted to a user bbox
                                bool already_converted = false;
                                if (keypoints_find) {
                                    Animals* current_frame_data = keypoints_map[current_frame_num];
                                    for (u32 animal_id = 0; animal_id < number_of_animals; animal_id++) {
                                        KeyPoints* frame_keypoints = &current_frame_data->keypoints[animal_id];
                                        if (j < frame_keypoints->bbox2d_list.size()) {
                                            for (const auto& user_bbox : frame_keypoints->bbox2d_list[j]) {
                                                if (user_bbox.state != RectNull && user_bbox.rect != nullptr) {
                                                    // Check if this user bbox roughly matches the YOLO bbox
                                                    double tolerance = 10.0; // pixels
                                                    if (std::abs(user_bbox.rect->X.Min - bbox.x_min) < tolerance &&
                                                        std::abs(user_bbox.rect->Y.Min - bbox.y_min) < tolerance &&
                                                        std::abs(user_bbox.rect->X.Max - bbox.x_max) < tolerance &&
                                                        std::abs(user_bbox.rect->Y.Max - bbox.y_max) < tolerance) {
                                                        already_converted = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if (already_converted) break;
                                    }
                                }
                                
                                if (!already_converted) {
                                    int drag_rect_id = 1000 + j * 100 + bbox_idx;
                                    
                                    ImVec4 yolo_color = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
                                    
                                    bool bbox_clicked = false, bbox_hovered = false, bbox_held = false;
                                    bool bbox_modified = ImPlot::DragRect(drag_rect_id, 
                                                                        &bbox.x_min, &bbox.y_min, 
                                                                        &bbox.x_max, &bbox.y_max, 
                                                                        yolo_color, 
                                                                        ImPlotDragToolFlags_None,
                                                                        &bbox_clicked, &bbox_hovered, &bbox_held);
                                    
                                    if (bbox_hovered || bbox_held) {
                                        double center_x = (bbox.x_min + bbox.x_max) / 2.0;
                                        double center_y = (bbox.y_min + bbox.y_max) / 2.0;
                                        std::string info_text = "YOLO Class:" + std::to_string(bbox.class_id) + 
                                                              " Conf:" + std::to_string(bbox.confidence).substr(0, 4);
                                        ImPlot::PlotText(info_text.c_str(), center_x, center_y);
                                    }
                                    
                                    if (bbox_modified) {
                                        yolo_frame_cache[current_frame_num] = yolo_bboxes;
                                        
                                        if (ImGui::IsKeyPressed(ImGuiKey_Y, false)) { 
                                            if (!keypoints_find) {
                                                Animals* animals = (Animals *)malloc(sizeof(Animals));
                                                allocate_keypoints(animals, scene, skeleton, number_of_animals);
                                                keypoints_map[current_frame_num] = animals; 
                                                keypoints_find = true;
                                            }
                                            
                                            Animals* current_frame_data = keypoints_map[current_frame_num];
                                            KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                                            
                                            if (frame_keypoints->bbox2d[j].state == RectNull) {
                                                frame_keypoints->bbox2d[j].rect = new ImPlotRect(bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max);
                                                frame_keypoints->bbox2d[j].state = RectTwoPoints;
                                                frame_keypoints->bbox2d[j].class_id = bbox.class_id;
                                                frame_keypoints->bbox2d[j].confidence = bbox.confidence;
                                                frame_keypoints->has_labels = true;
                                                allow_exit = false;
                                                
                                                std::cout << "Converted YOLO bbox to user bbox for camera " << j << std::endl;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if(plot_keypoints_flag)
                    {
                        // labeling 
                        if (ImPlot::IsPlotHovered()) {
                            is_view_focused[j]  = true;

                            if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                                // create keypoints
                                if (!keypoints_find){
                                    // not found
                                    Animals* animals = (Animals *)malloc(sizeof(Animals));
                                    allocate_keypoints(animals, scene, skeleton, number_of_animals);
                                    keypoints_map[current_frame_num] = animals; 
                                    allow_exit = false;
                                }
                            }

                            if (keypoints_find) {
                                Animals* current_frame_data = keypoints_map[current_frame_num];
                                if (skeleton->has_skeleton) {
                                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                                    u32* kp = &frame_keypoints->active_kp_id[j];
                                    u32* count = &frame_keypoints->counter;
                                    if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
                                        if(!frame_keypoints->keypoints2d[j][*kp].is_labeled) {(*count)++;}
                                        // labeling sequentially each view
                                        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                        frame_keypoints->keypoints2d[j][*kp].position = {mouse.x,  mouse.y};
                                        frame_keypoints->keypoints2d[j][*kp].is_labeled = true;
                                        frame_keypoints->keypoints2d[j][*kp].is_triangulated = false;
                                        if(*kp < (skeleton->num_nodes - 1)) {(*kp)++;}
                                        frame_keypoints->has_labels = true;
                                        allow_exit = false;
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
                                }

                                if (skeleton->has_bbox) {
                                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[current_frame_data->active_id];
                                    if (frame_keypoints->bbox2d[j].state == RectOnePoint && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                                        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                        frame_keypoints->bbox2d[j].rect->X.Max = mouse.x;
                                        frame_keypoints->bbox2d[j].rect->Y.Min = mouse.y;
                                    }

                                    if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
                                        frame_keypoints->bbox2d[j].state = RectTwoPoints;
                                    }

                                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle, false)) {
                                        if (frame_keypoints->bbox2d[j].state == RectNull) {
                                            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                            frame_keypoints->bbox2d[j].rect = new ImPlotRect(mouse.x, mouse.x, mouse.y, mouse.y);
                                            frame_keypoints->bbox2d[j].state = RectOnePoint;
                                            frame_keypoints->has_labels = true;
                                        }
                                        allow_exit = false;
                                    }
                                }

                                if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Delete, false)) 
                                {
                                    delete_all_labels(current_frame_data, scene, skeleton, number_of_animals);
                                    keypoints_map.erase(current_frame_num);
                                    keypoints_find = false;
                                    allow_exit = false;
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_Backspace, false)) 
                                {
                                    reinitalize_keypoint_active_animal(current_frame_data, scene, skeleton);
                                    allow_exit = false;
                                }
                                
                            }  
                        } else {
                            is_view_focused[j]  = false;
                        }

                        if (keypoints_find) {
                            Animals* current_frame_data = keypoints_map[current_frame_num];

                            if(skeleton->has_skeleton) {
                                for (u32 animal_id=0; animal_id < number_of_animals; animal_id++) {
                                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[animal_id];
                                    gui_plot_keypoints(frame_keypoints, skeleton, j, animal_id, scene->num_cams, current_frame_data->active_id == animal_id, allow_exit);
                                }
                            }

                            if(skeleton->has_bbox) {
                                // draw all animals
                                for (u32 animal_id=0; animal_id < number_of_animals; animal_id++) {
                                    KeyPoints* frame_keypoints = &current_frame_data->keypoints[animal_id];
                                    ImColor bbox_color = frame_keypoints->animal_color;
                                    
                                    if (frame_keypoints->bbox2d[j].state != RectNull) {
                                        ImPlotRect* my_rect = frame_keypoints->bbox2d[j].rect;
                                        if (current_frame_data->active_id == animal_id) {
                                            ImPlot::DragRect(0,&my_rect->X.Min,&my_rect->Y.Min,&my_rect->X.Max,&my_rect->Y.Max, bbox_color, ImPlotDragToolFlags_None);
                                        }
                                        ImPlot::DragRect(0,&my_rect->X.Min,&my_rect->Y.Min,&my_rect->X.Max,&my_rect->Y.Max, bbox_color, ImPlotDragToolFlags_NoInputs);
                                    }
                                    
                                    if (j < frame_keypoints->bbox2d_list.size()) {
                                        for (size_t bbox_idx = 0; bbox_idx < frame_keypoints->bbox2d_list[j].size(); ++bbox_idx) {
                                            BoundingBox& bbox = frame_keypoints->bbox2d_list[j][bbox_idx];
                                            if (bbox.state != RectNull && bbox.rect != nullptr) {
                                                int multi_bbox_id = 2000 + animal_id * 1000 + j * 100 + bbox_idx;
                                                
                                                ImVec4 multi_bbox_color = bbox_color;
                                                multi_bbox_color.w = 0.6f; 
                                                
                                                bool bbox_clicked = false, bbox_hovered = false, bbox_held = false;
                                                bool bbox_modified = false;
                                                
                                                if (current_frame_data->active_id == animal_id) {
                                                    bbox_modified = ImPlot::DragRect(multi_bbox_id,
                                                                                   &bbox.rect->X.Min, &bbox.rect->Y.Min,
                                                                                   &bbox.rect->X.Max, &bbox.rect->Y.Max,
                                                                                   multi_bbox_color, ImPlotDragToolFlags_None,
                                                                                   &bbox_clicked, &bbox_hovered, &bbox_held);
                                                } else {
                                                    ImPlot::DragRect(multi_bbox_id,
                                                                   &bbox.rect->X.Min, &bbox.rect->Y.Min,
                                                                   &bbox.rect->X.Max, &bbox.rect->Y.Max,
                                                                   multi_bbox_color, ImPlotDragToolFlags_NoInputs);
                                                }
                                                
                                                if (bbox_hovered || bbox_held) {
                                                    double center_x = (bbox.rect->X.Min + bbox.rect->X.Max) / 2.0;
                                                    double center_y = (bbox.rect->Y.Min + bbox.rect->Y.Max) / 2.0;
                                                    std::string info_text = "Class:" + std::to_string(bbox.class_id) + 
                                                                          " Conf:" + std::to_string(bbox.confidence).substr(0, 4);
                                                    ImPlot::PlotText(info_text.c_str(), center_x, center_y);
                                                }
                                                
                                                if (bbox_modified) {
                                                    allow_exit = false;
                                                }
                                            }
                                        }
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

                if (ImGui::IsKeyPressed(ImGuiKey_Tab, false))
                {
                    if (ImGui::GetIO().KeyShift) {
                        current_frame_data->active_id = (current_frame_data->active_id + number_of_animals - 1) % number_of_animals;
                    } else {
                        current_frame_data->active_id = (current_frame_data->active_id + 1) % number_of_animals;
                    }
                }

                if (ImGui::Begin("Keypoints")) {

                    if (ImGui::BeginTable("##Animals", number_of_animals, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
                    {
                        for (int animal_id = 0; animal_id < number_of_animals; animal_id++)
                        {
                            char label[32];
                            if (skeleton->has_bbox) {
                                sprintf(label, "Ani %d", animal_id);
                            } else {
                                sprintf(label, "Ani %d (%d/%d labeled)", animal_id, current_frame_data->keypoints[animal_id].counter, skeleton->num_nodes);
                                                        }
                            ImGui::TableNextColumn();
                            if(ImGui::Selectable(label, current_frame_data->active_id == animal_id)) {
                                current_frame_data->active_id = animal_id;
                            }
                            if (current_frame_data->keypoints[animal_id].has_labels) {
                                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, (ImColor)current_frame_data->keypoints[animal_id].animal_color);
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
                                        ImU32 row_bg_color = ImGui::GetColorU32(ImVec4(0.7f, 0.3f, 0.65f, 0.65f)); 
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
                                                        node_color = skeleton->node_colors[column-1];
                                                        node_color.w = 0.5;
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

                if (ImGui::Button("Save Labeled Data") || (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S, false))) {
                    save_keypoints(keypoints_map, skeleton, keypoints_root_folder, scene->num_cams, camera_names, number_of_animals);
                    allow_exit = true;
                    last_saved = time(NULL);
                }
                ImGui::SameLine();
                ImGui::Text("Last saved: %s", ctime(&last_saved));

                // if (ImGui::Button("Load Labeled Data"))
                // {
                //     // load_keypoints(keypoints_map, skeleton, keypoints_root_folder, scene, camera_names);
                // }

                // TODO: change folder
                ImGui::Text(keypoints_root_folder.c_str());
                ImGui::SameLine();
                if (ImGui::Button("Update keypoints folder")) {
                    change_keypoints_folder = true;
                    file_dialog.Open();
                }

                if (ImGui::Button("Load 2d Keypoints Only"))
                {
                    for (int i=0; i<scene->num_cams; i++) {
                        load_2d_keypoints(keypoints_map, skeleton, keypoints_root_folder, i, camera_names[i], scene, number_of_animals);
                    }
                }

                auto upper_it = keypoints_map.upper_bound(current_frame_num); 
                if (upper_it == keypoints_map.end()) {
                    upper_it = keypoints_map.begin();
                }
                ImGui::Text("Next labeled frame : %d", (*upper_it).first);
                if (ImGui::Button("Jump to Next Labeled Frame") || ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) {
                    for (int i = 0; i < scene->num_cams; i++) {
                        scene->seek_context[i].seek_frame = (uint64_t)(*upper_it).first;
                        scene->seek_context[i].use_seek = true;
                        scene->seek_context[i].seek_accurate = true;
                    }
                    for (int i = 0; i < scene->num_cams; i++) {
                        while (!(scene->seek_context[i].seek_done)) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                    for (int i = 0; i < scene->num_cams; i++) {
                        scene->seek_context[i].seek_done = false;
                    }
                    to_display_frame_number = scene->seek_context[0].seek_frame;
                    read_head = 0;
                    just_seeked = true;
                    pause_selected = 0;
                    slider_frame_number = to_display_frame_number;
                }
                ImGui::Text("Total labeled frames : %d", keypoints_map.size());
                
                ImGui::SeparatorText("YOLO Model");
                
                if (ImGui::Button("Select YOLO"))
                {
                    model_file_dialog.Open();
                }
                
                if (model_selected)
                {
                    ImGui::Text("Model: %s", std::filesystem::path(yolo_model_path).filename().string().c_str());
                    
                    if (ImGui::Button("Run YOLO Prediction"))
                    {
                        std::cout << "Running YOLO prediction on frame " << current_frame_num << std::endl;
                        
                        for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                            unsigned char* frame_data = nullptr;
                            if (play_video && dc_context->decoding_flag) {
                                frame_data = scene->display_buffer[cam_id][read_head].frame;
                            } else if (!play_video) {
                                int select_corr_head = (pause_selected + read_head) % scene->size_of_buffer;
                                frame_data = scene->display_buffer[cam_id][select_corr_head].frame;
                            }
                            
                            if (frame_data) {
                                yolo_predictions[cam_id] = runYoloInference(yolo_model_path, frame_data, 
                                                                          scene->image_width[cam_id], scene->image_height[cam_id]);
                                
                                yolo_bboxes[cam_id].clear();
                                for (const auto& pred : yolo_predictions[cam_id]) {
                                    yolo_bboxes[cam_id].emplace_back(pred);
                                }
                                
                                std::cout << "Camera " << cam_id << ": Found " << yolo_predictions[cam_id].size() << " detections" << std::endl;
                            }
                        }
                        
                        if (!yolo_bboxes.empty() && std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(), 
                            [](const auto& cam_bboxes) { return !cam_bboxes.empty(); })) {
                            
                            if (!keypoints_find) {
                                Animals* animals = (Animals *)malloc(sizeof(Animals));
                                allocate_keypoints(animals, scene, skeleton, number_of_animals);
                                keypoints_map[current_frame_num] = animals; 
                                keypoints_find = true;
                            }
                            
                            Animals* current_frame_data = keypoints_map[current_frame_num];
                            
                            std::map<int, int> class_to_animal;
                            int next_animal_id = 0;
                            
                            for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                                for (size_t bbox_idx = 0; bbox_idx < yolo_bboxes[cam_id].size(); bbox_idx++) {
                                    const auto& yolo_bbox = yolo_bboxes[cam_id][bbox_idx];
                                    if (yolo_bbox.is_valid) {
                                        if (class_to_animal.find(yolo_bbox.class_id) == class_to_animal.end()) {
                                            class_to_animal[yolo_bbox.class_id] = next_animal_id % number_of_animals;
                                            next_animal_id++;
                                        }
                                        
                                        int target_animal_id = class_to_animal[yolo_bbox.class_id];
                                        KeyPoints* frame_keypoints = &current_frame_data->keypoints[target_animal_id];
                                        
                                        BoundingBox new_bbox;
                                        new_bbox.rect = new ImPlotRect(yolo_bbox.x_min, yolo_bbox.x_max, yolo_bbox.y_min, yolo_bbox.y_max);
                                        new_bbox.state = RectTwoPoints;
                                        new_bbox.class_id = yolo_bbox.class_id;
                                        new_bbox.confidence = yolo_bbox.confidence;
                                        
                                        frame_keypoints->bbox2d_list[cam_id].push_back(new_bbox);
                                        frame_keypoints->has_labels = true;
                                        allow_exit = false;
                                        
                                        std::cout << "Auto-converted YOLO bbox (class " << yolo_bbox.class_id 
                                                  << ") to animal " << target_animal_id << ", camera " << cam_id << std::endl;
                                    }
                                }
                            }
                        }
                    }
                    
                    ImGui::Checkbox("Show YOLO Predictions", &show_yolo_predictions);
                    
                    ImGui::Checkbox("Automatic YOLO Labeling", &auto_yolo_labeling);
                    
                    if (auto_yolo_labeling) {
                        ImGui::SameLine();
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "(Auto mode ON)");
                    }

                }
                else
                {
                    ImGui::Text("No model selected");
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
                ImGui::Text("<Space>: toggle play and pause");
                ImGui::Text("<,>: previous image");
                ImGui::Text("<.>: next image");

                ImGui::Text("<Tab>: circle selecting each animal");
                ImGui::Text("Shift-Tab: select previous animal");

                ImGui::SeparatorText("After selecting an animal, while hovering image");
                ImGui::Text("<c>: create bbox or/and keypoints on frame");
                ImGui::Text("<Backspace>: delete labels of active animal");
                ImGui::Text("CTRL-Delete: delete all labels for all animals");

                ImGui::SeparatorText("Bounding box");
                ImGui::Text("<mouse middle button>: draw bbox, drag then release to finish drawing the bbox");

                ImGui::SeparatorText("Keypoint");
                ImGui::Text("<w>: drop active keypoint");
                ImGui::Text("<a>: active keypoint++ ");
                ImGui::Text("<d>: active keypoint--");
                ImGui::Text("<q>: active keypoint set to first node");
                ImGui::Text("<e>: active keypoint set to last node");
                ImGui::Text("<Right Arrow>: next labeled frame");
                ImGui::Text("CTRL-S: save labels");

                ImGui::Text("While hovering keypoints");
                ImGui::Text("<r>: delete active keypoint");
                ImGui::Text("<f>: delete active keypoint on all cameras");
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
