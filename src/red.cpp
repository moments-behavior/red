#include "Logger.h"
#include "camera.h"
#include "filesystem"
#include "global.h"
#include "gui.h"
#include "gx_helper.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "render.h"
#include "skeleton.h"
#include "utils.h"
#include "yolo_detection.h"
#include <ImGuiFileDialog.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <torch/torch.h>
#include <torch/script.h>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) &&                                 \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

std::vector<std::mutex> g_mutexes(MAX_VIEWS);
std::vector<std::condition_variable> g_cvs(MAX_VIEWS);
std::vector<bool> g_ready(MAX_VIEWS);
std::vector<std::vector<cv::Rect>> yolo_boxes(MAX_VIEWS);
std::vector<std::vector<std::string>> yolo_labels(MAX_VIEWS);
std::vector<std::vector<int>> yolo_classid(MAX_VIEWS);
std::vector<unsigned char *> yolo_input_frames_rgba(MAX_VIEWS);

std::map<int, int> g_yolo_class_map;
std::map<int, int> g_reverse_yolo_class_map;
int next_class_id = 0; 
float confidence_threshold = 0.5f; 
float iou_threshold = 0.4f;

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

    YoloBBox() : x_min(0), y_min(0), x_max(0), y_max(0),
                        confidence(0.0f), class_id(-1), is_valid(false) {}
    
    YoloBBox(const YoloPrediction &pred) {
        x_min = pred.x - pred.w / 2.0;
        y_min = pred.y - pred.h / 2.0;
        x_max = pred.x + pred.w / 2.0;
        y_max = pred.y + pred.h / 2.0;
        confidence = pred.confidence;
        class_id = pred.class_id;
        is_valid = true;
    }
};

// YOLO detection variables (declared after struct definitions)
std::vector<std::vector<YoloBBox>> yolo_bboxes(MAX_VIEWS);
std::vector<std::vector<YoloPrediction>> yolo_predictions(MAX_VIEWS);
std::string yolo_model_path = "";

// YOLO DragRect storage - stores BoundingBox objects for interactive manipulation
std::vector<std::vector<BoundingBox>> yolo_drag_boxes(MAX_VIEWS);
std::vector<int> yolo_active_bbox_idx(MAX_VIEWS, -1); // Track active YOLO bbox per camera

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

std::vector<YoloPrediction> applyNMS(std::vector<YoloPrediction>& predictions, float iou_threshold) {
    predictions.erase(std::remove_if(predictions.begin(), predictions.end(),
        [confidence_threshold](const YoloPrediction& pred) {
            return pred.confidence < confidence_threshold;
        }), predictions.end());
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
        
        if (output.dim() == 3) {
            output = output.permute({0, 2, 1});
        }
        output = output.squeeze(0);  
        
        if (output.dim() != 2) {
            std::cerr << "Unexpected output tensor dimensions: " << output.dim() << std::endl;
            return predictions;
        }
        
        float conf_threshold = 0.25f;
        float scale_x = static_cast<float>(width) / 640.0f;
        float scale_y = static_cast<float>(height) / 640.0f;
        
        auto output_accessor = output.accessor<float, 2>();
        int num_detections = output.size(0);
        int output_features = output.size(1);
        
        std::cout << "Processing " << num_detections << " detections with " << output_features << " features per detection" << std::endl;
        
        for (int i = 0; i < num_detections; ++i) {
            float confidence;
            int class_id;
            
            if (output_features == 6) {
                confidence = output_accessor[i][4];
                class_id = static_cast<int>(output_accessor[i][5]);
            } else if (output_features == 5) {
                confidence = output_accessor[i][4];
                class_id = 0;
            } else if (output_features > 5) {
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
        
        predictions = applyNMS(predictions, iou_threshold); 
        
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

// Helper function to check if current frame already has YOLO detections
bool frameHasYoloDetections(int frame_num, const std::map<u32, KeyPoints*>& keypoints_map, const SkeletonContext* skeleton) {
    if (!skeleton || !skeleton->has_bbox) return false;
    
    auto it = keypoints_map.find(frame_num);
    if (it == keypoints_map.end() || !it->second) return false;
    
    // Check if any camera has bounding boxes with confidence < 1.0 (indicating YOLO detections)
    for (int cam_id = 0; cam_id < MAX_VIEWS && cam_id < it->second->bbox2d_list.size(); cam_id++) {
        const auto& bbox_list = it->second->bbox2d_list[cam_id];
        for (const auto& bbox : bbox_list) {
            if (bbox.confidence < 1.0f && bbox.state == RectTwoPoints) {
                return true; 
            }
        }
    }
    return false;
}


int main(int, char **) {
    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    *window =
        (gx_context){.swap_interval = 1, // use vsync
                     .width = 1920,
                     .height = 1080,
                     .render_target_title = (char *)malloc(100), // window title
                     .glsl_version = (char *)malloc(100)};

    render_initialize_target(window);

    render_scene *scene = (render_scene *)malloc(sizeof(render_scene));

    std::string root_dir;
    std::string skeleton_dir;
    std::vector<std::string> camera_names;
    std::vector<CameraParams> camera_params;
    std::vector<std::thread> decoder_threads;
    std::vector<FFmpegDemuxer *> demuxers;

    DecoderContext *dc_context =
        (DecoderContext *)malloc(sizeof(DecoderContext));
    *dc_context = (DecoderContext){.decoding_flag = false,
                                   .stop_flag = false,
                                   .total_num_frame = int(INT_MAX),
                                   .estimated_num_frames = 0,
                                   .gpu_index = 0,
                                   .seek_interval = 250};

    // gui states, todo: bundle this later
    int to_display_frame_number = 0;
    int pause_selected = 0;
    int read_head = 0;
    bool play_video = false;
    bool toggle_play_status = false;
    int slider_frame_number = 0;
    bool just_seeked = false;
    bool slider_just_changed = false;
    std::time_t last_saved = static_cast<std::time_t>(-1);

    bool video_loaded = false;
    bool cpu_buffer_toggle = true;
    bool plot_keypoints_flag = false;
    int current_frame_num = 0;
    int previous_frame_num = -1;
    bool skeleton_chosen = false;
    std::vector<std::string> imgs_names;

    // for labeling
    SkeletonContext *skeleton;
    std::map<u32, KeyPoints *> keypoints_map;
    bool keypoints_find = false;
    std::map<std::string, SkeletonPrimitive> skeleton_map;

    // others
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split(cwd, delimiter);
    std::string start_folder_name = "/home/" + tokenized_path[2] + "/data";
    start_folder_name = "/nfs/exports/ratlv";
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    ImGuiIO &io = ImGui::GetIO();

    ImPlotStyle &style = ImPlot::GetStyle();
    ImVec4 *colors = style.Colors;
    
    // Skeleton Creator variables
    bool show_skeleton_creator = false;
    struct SkeletonCreatorNode {
        ImPlotPoint position;
        std::string name;
        ImVec4 color;
        int id;
        
        SkeletonCreatorNode() : position(0.5, 0.5), name(""), color(1.0f, 1.0f, 1.0f, 1.0f), id(-1) {}
        SkeletonCreatorNode(double x, double y, int node_id) : position(x, y), id(node_id) {
            name = "Node" + std::to_string(node_id);
            color = (ImVec4)ImColor::HSV(node_id / 10.0f, 1.0f, 1.0f);
        }
    };
    
    struct SkeletonCreatorEdge {
        int node1_id;
        int node2_id;
        
        SkeletonCreatorEdge() : node1_id(-1), node2_id(-1) {}
        SkeletonCreatorEdge(int n1, int n2) : node1_id(n1), node2_id(n2) {}
    };
    
    std::vector<SkeletonCreatorNode> creator_nodes;
    std::vector<SkeletonCreatorEdge> creator_edges;
    int next_node_id = 0;
    int selected_node_for_edge = -1;
    std::string skeleton_creator_name = "CustomSkeleton";
    bool skeleton_creator_has_bbox = false;
    bool skeleton_creator_has_skeleton = true;
    
    // Bounding box class management
    std::vector<std::string> bbox_class_names = {"Class_1"};
    std::vector<ImVec4> bbox_class_colors = {
        ImVec4(0.3f, 1.0f, 1.0f, 1.0f)
    };
    int current_bbox_class = 0;
    static char new_class_name_buffer[64] = "";
    
    // Helper function to create a new bbox class
    auto create_new_bbox_class = [&]() {
        std::string new_class_name = "Class_" + std::to_string(bbox_class_names.size() + 1);
        bbox_class_names.push_back(new_class_name);
        // Generate a unique color for the new class (HSV with different hues)
        float hue = (bbox_class_colors.size() * 0.618034f); // Golden ratio for nice color distribution
        while (hue > 1.0f) hue -= 1.0f;
        ImVec4 new_color = (ImVec4)ImColor::HSV(hue, 0.8f, 1.0f);
        bbox_class_colors.push_back(new_color);
        current_bbox_class = bbox_class_names.size() - 1;
    };
    
    // Helper function to cleanup YOLO drag boxes
    auto cleanup_yolo_drag_boxes = [&]() {
        for (int cam_id = 0; cam_id < MAX_VIEWS; cam_id++) {
            for (auto& drag_box : yolo_drag_boxes[cam_id]) {
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
    std::vector<std::thread> yolo_threads;
    yolo_param yolo_setting = yolo_param();
    std::string keypoints_root_folder;
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
    
    bool auto_yolo_labeling = false;
    std::set<int> yolo_processed_frames; // Track which frames have been processed

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

        if (ImGui::Begin("File Browser", NULL, ImGuiWindowFlags_MenuBar)) {

            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Open")) {
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 0;
                        config.path = "/home/user/data/movies"; // temp default path for test
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChooseMedia", "Choose Media",
                            ".mp4,.tiff,.jpeg,.jpg,.png", config);
                    };
                    ImGui::EndMenu();
                }

                if (video_loaded) {
                    if (ImGui::BeginMenu("Skeleton")) {
                        if (!skeleton_chosen) {
                            skeleton = new SkeletonContext;
                            // Initialize skeleton with safe default values
                            skeleton->num_nodes = 0;
                            skeleton->num_edges = 0;
                            skeleton->name = "";
                            skeleton->has_bbox = false;
                            skeleton->has_skeleton = true;
                            skeleton->node_colors.clear();
                            skeleton->edges.clear();
                            skeleton->node_names.clear();
                            
                            skeleton_map = skeleton_get_all();
                        }

                        for (auto &element : skeleton_map) {

                            if (ImGui::MenuItem(element.first.c_str(), NULL,
                                                skeleton->name == element.first,
                                                !skeleton_chosen)) {
                                if (element.second == SP_LOAD) {
                                    IGFD::FileDialogConfig config;
                                    config.countSelectionMax = 1;
                                    config.path = std::filesystem::current_path().string() + "/skeleton";
                                    config.flags = ImGuiFileDialogFlags_Modal;
                                    ImGuiFileDialog::Instance()->OpenDialog(
                                        "ChooseSkeleton", "Choose Skeleton",
                                        ".json", config);
                                } else {

                                    bool load_calibration = true;
                                    if (scene->num_cams > 1) {
                                        for (u32 i = 0; i < scene->num_cams;
                                             i++) {
                                            std::string cam_file =
                                                root_dir + "/calibration/" +
                                                camera_names[i] + ".yaml";

                                            CameraParams cam;
                                            if (camera_load_params_from_yaml(
                                                    cam_file, cam,
                                                    error_message)) {
                                                camera_params.push_back(cam);
                                            } else {
                                                load_calibration = false;
                                                camera_params.clear();
                                                show_error = true;
                                                break;
                                            }
                                        }
                                    }

                                    if (load_calibration) {
                                        skeleton_initialize(element.first,
                                                            root_dir, skeleton,
                                                            element.second);
                                        plot_keypoints_flag = true;
                                        keypoints_root_folder =
                                            root_dir + "/labeled_data/";
                                        // create folders
                                        std::filesystem::create_directory(
                                            keypoints_root_folder);
                                        skeleton_chosen = true;
                                    }
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }

                    if (ImGui::BeginMenu("Detection")) {
                        if (cpu_buffer_toggle) {
                            if (ImGui::MenuItem("YOLOv5")) {
                                std::string yolov5_onnx =
                                    root_dir + "/yolo/v5/best.onnx";
                                std::string yolov5_labelname =
                                    root_dir + "/yolo/v5/label.names";
                                read_yolo_labels(yolov5_labelname,
                                                 &yolo_setting);

                                for (int i = 0; i < scene->num_cams; i++) {
                                    yolo_threads.push_back(
                                        std::thread(&yolo_process, yolov5_onnx,
                                                    &yolo_setting, i));
                                }
                                yolo_detection = true;
                            }
                        } else {
                            if (ImGui::MenuItem("YOLOv8")) {
                                std::string engine_file_path =
                                    root_dir +
                                    "/yolo/yolorat_bbox/rat_bbox.engine";
                                for (int i = 0; i < scene->num_cams; i++) {
                                    yolo_threads.push_back(std::thread(
                                        &yolo_process_trt, engine_file_path, i,
                                        scene->image_width[i],
                                        scene->image_height[i]));
                                }
                                yolo_detection = true;
                            }

                            if (ImGui::MenuItem("YOLOv8Pose")) {
                                std::string engine_file_path =
                                    root_dir + "/yolo/yolopose/rat_pose.engine";
                                for (int i = 0; i < scene->num_cams; i++) {
                                    yolo_threads.push_back(std::thread(
                                        &yolo_process_v8pose, engine_file_path,
                                        i, scene->image_width[i],
                                        scene->image_height[i]));
                                }
                                yolo_detection = true;
                            }
                        }
                        ImGui::EndMenu();
                    }
                }
                
                if (ImGui::BeginMenu("Tools")) {
                    if (ImGui::MenuItem("Skeleton Creator")) {
                        show_skeleton_creator = true;
                    }
                    ImGui::EndMenu();
                }
                
                ImGui::EndMenuBar();
            }

            // ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
            //             1000.0f / ImGui::GetIO().Framerate,
            //             ImGui::GetIO().Framerate);

            // if (video_loaded) {
            //     ImGui::Text("Frame number %d ",
            //     scene->display_buffer[0][read_head].frame_number);
            //     ImGui::Text("To Display frame number %d ",
            //     to_display_frame_number); ImGui::Text("Readhead %d",
            //     read_head);
            // }
            if (!video_loaded) {
                {
                    const char *items[] = {"CPU Buffer", "GPU Buffer"};
                    static int item_current = 0;
                    ImGui::Combo("Buffer type", &item_current, items,
                                 IM_ARRAYSIZE(items));
                    if (item_current == 0) {
                        scene->use_cpu_buffer = true;
                    } else {
                        scene->use_cpu_buffer = false;
                    }
                }

                ImGui::InputInt("Buffer size", &label_buffer_size);
            }
            if (video_loaded) {
                if (ImGui::InputInt("Seek step", &dc_context->seek_interval, 10,
                                    100)) {
                    std::cout << "Seek step: " << dc_context->seek_interval
                              << std::endl;
                }

                static int seek_accurate_frame_num = 0;
                ImGui::InputInt("Seek Accurate", &seek_accurate_frame_num, 1,
                                100);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    std::cout << "Seek accurate to: " << seek_accurate_frame_num
                              << std::endl;
                    for (int i = 0; i < scene->num_cams; i++) {
                        scene->seek_context[i].seek_frame =
                            (uint64_t)seek_accurate_frame_num;
                        scene->seek_context[i].use_seek = true;
                        scene->seek_context[i].seek_accurate = true;
                    }

                    for (int i = 0; i < scene->num_cams; i++) {
                        while (!(scene->seek_context[i].seek_done)) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(1));
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
            }
        }
        ImGui::End();

        // file explorer display
        if (ImGuiFileDialog::Instance()->Display("ChooseMedia")) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                auto selected_files =
                    ImGuiFileDialog::Instance()->GetSelection();
                root_dir = ImGuiFileDialog::Instance()->GetCurrentPath();
                skeleton_dir = root_dir;

                // check if it is mp4, if it is mp4 files
                auto first_selection =
                    *selected_files.begin(); // Dereferencing iterator
                if (string_ends_with(first_selection.first, ".mp4")) {
                    for (const auto &elem : selected_files) {
                        std::size_t cam_string_mp4_position =
                            elem.first.find("mp4");
                        std::string cam_string =
                            elem.first.substr(0, cam_string_mp4_position - 1);
                        camera_names.push_back(cam_string);
                        std::cout << "camera names: " << cam_string
                                  << std::endl;

                        std::map<std::string, std::string> m;
                        FFmpegDemuxer *demuxer =
                            new FFmpegDemuxer(elem.second.c_str(), m);
                        demuxers.push_back(demuxer);
                    }
                    std::map<std::string, std::string> m;
                    FFmpegDemuxer dummy_dmuxer(
                        selected_files.begin()->second.c_str(), m);
                    dc_context->seek_interval =
                        (int)dummy_dmuxer
                            .FindKeyFrameInterval(); // get the seek interval

                    scene->num_cams = selected_files.size();
                    scene->image_width =
                        (u32 *)malloc(sizeof(u32) * scene->num_cams);
                    scene->image_height =
                        (u32 *)malloc(sizeof(u32) * scene->num_cams);
                    for (u32 j = 0; j < scene->num_cams; j++) {
                        scene->image_width[j] = demuxers[j]->GetWidth();
                        scene->image_height[j] = demuxers[j]->GetHeight();
                    }
                    render_allocate_scene_memory(scene, label_buffer_size);

                    // multiple threads for decoding for selected videos
                    for (int i = 0; i < scene->num_cams; i++) {
                        decoder_threads.push_back(std::thread(
                            &decoder_process, dc_context, demuxers[i],
                            scene->display_buffer[i], scene->size_of_buffer,
                            &scene->seek_context[i], scene->use_cpu_buffer));
                        is_view_focused.push_back(false);
                    }
                    video_loaded = true;
                } else {
                    input_is_imgs = true;
                    for (const auto &elem : selected_files) {
                        std::size_t cam_string_position = elem.first.find("_");
                        std::string cam_name =
                            elem.first.substr(0, cam_string_position);
                        std::string file_name =
                            elem.first.substr(cam_string_position + 1);

                        if (std::find(camera_names.begin(), camera_names.end(),
                                      cam_name) == camera_names.end()) {
                            camera_names.push_back(cam_name);
                        }

                        if (std::find(imgs_names.begin(), imgs_names.end(),
                                      file_name) == imgs_names.end()) {
                            imgs_names.push_back(file_name);
                        }
                    }

                    dc_context->seek_interval = 1;
                    scene->num_cams = camera_names.size();
                    scene->image_width =
                        (u32 *)malloc(sizeof(u32) * scene->num_cams);
                    scene->image_height =
                        (u32 *)malloc(sizeof(u32) * scene->num_cams);
                    for (u32 j = 0; j < scene->num_cams; j++) {
                        std::string file_name = root_dir + "/" +
                                                camera_names[j] + "_" +
                                                imgs_names[0];
                        cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
                        scene->image_width[j] = image.cols;
                        scene->image_height[j] = image.rows;
                    }
                    if (imgs_names.size() < label_buffer_size) {
                        label_buffer_size = imgs_names.size();
                    }
                    render_allocate_scene_memory(scene, label_buffer_size);
                    for (int i = 0; i < scene->num_cams; i++) {
                        decoder_threads.push_back(std::thread(
                            &image_loader, dc_context, imgs_names,
                            scene->display_buffer[i], scene->size_of_buffer,
                            &scene->seek_context[i], scene->use_cpu_buffer,
                            camera_names[i], root_dir));
                        is_view_focused.push_back(false);
                    }
                    video_loaded = true;
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseSkeleton")) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                auto skeleton_file =
                    ImGuiFileDialog::Instance()->GetSelection();

                if (!skeleton_file.empty()) {

                    bool load_calibration = true;
                    if (scene->num_cams > 1) {
                        for (u32 i = 0; i < scene->num_cams; i++) {
                            std::string cam_file = root_dir + "/calibration/" +
                                                   camera_names[i] + ".yaml";

                            CameraParams cam;
                            if (camera_load_params_from_yaml(cam_file, cam,
                                                             error_message)) {
                                camera_params.push_back(cam);
                            } else {
                                load_calibration = false;
                                camera_params.clear();
                                show_error = true;
                                break;
                            }
                        }
                    }

                    if (load_calibration) {
                        skeleton_dir =
                            ImGuiFileDialog::Instance()->GetCurrentPath();
                        skeleton_initialize("", skeleton_file.begin()->second,
                                            skeleton, SP_LOAD);
                        plot_keypoints_flag = true;
                        keypoints_root_folder = root_dir + "/labeled_data/";
                        skeleton_chosen = true;
                    }
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        // Handle background image selection for skeleton creator
        if (ImGuiFileDialog::Instance()->Display("ChooseBackgroundImage")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto file_selection = ImGuiFileDialog::Instance()->GetSelection();
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
                    unsigned char* image_data = stbi_load(background_image_path.c_str(), 
                                                        &background_width, &background_height, &channels, 0);
                    if (image_data) {
                        glGenTextures(1, &background_texture);
                        glBindTexture(GL_TEXTURE_2D, background_texture);
                        
                        GLenum format = GL_RGB;
                        if (channels == 4) format = GL_RGBA;
                        else if (channels == 1) format = GL_RED;
                        
                        glTexImage2D(GL_TEXTURE_2D, 0, format, background_width, background_height, 
                                   0, format, GL_UNSIGNED_BYTE, image_data);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                        
                        stbi_image_free(image_data);
                    }
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        // Handle skeleton load dialog for editing
        if (ImGuiFileDialog::Instance()->Display("LoadSkeletonForEdit")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string file_path = ImGuiFileDialog::Instance()->GetFilePathName();
                
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
                    
                    if (skeleton_json.contains("node_names") && skeleton_json.contains("node_positions")) {
                        std::vector<std::string> node_names = skeleton_json["node_names"];
                        std::vector<std::vector<double>> node_positions = skeleton_json["node_positions"];
                        
                        for (size_t i = 0; i < node_names.size() && i < node_positions.size(); i++) {
                            if (node_positions[i].size() >= 2) {
                                SkeletonCreatorNode node;
                                node.id = next_node_id++;
                                node.name = node_names[i];
                                node.position = ImPlotPoint(node_positions[i][0], node_positions[i][1]);
                                node.color = (ImVec4)ImColor::HSV(node.id / 10.0f, 1.0f, 1.0f);
                                creator_nodes.push_back(node);
                            }
                        }
                    } else if (skeleton_json.contains("node_names")) {
                        // Fallback for skeletons without saved positions
                        std::vector<std::string> node_names = skeleton_json["node_names"];
                        double spacing = 0.8 / (node_names.size() + 1);
                        
                        for (size_t i = 0; i < node_names.size(); i++) {
                            SkeletonCreatorNode node;
                            node.id = next_node_id++;
                            node.name = node_names[i];
                            node.position = ImPlotPoint(0.1 + spacing * (i + 1), 0.5);
                            node.color = (ImVec4)ImColor::HSV(node.id / 10.0f, 1.0f, 1.0f);
                            creator_nodes.push_back(node);
                        }
                    }
                    
                    if (skeleton_json.contains("edges")) {
                        std::vector<std::vector<int>> edges_array = skeleton_json["edges"];
                        for (const auto& edge : edges_array) {
                            if (edge.size() >= 2 && edge[0] < creator_nodes.size() && edge[1] < creator_nodes.size()) {
                                SkeletonCreatorEdge creator_edge;
                                creator_edge.node1_id = creator_nodes[edge[0]].id;
                                creator_edge.node2_id = creator_nodes[edge[1]].id;
                                creator_edges.push_back(creator_edge);
                            }
                        }
                    }
                    
                    std::cout << "Skeleton loaded from: " << file_path << " (with " << creator_nodes.size() << " nodes)" << std::endl;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (dc_context->decoding_flag && play_video) {
            for (u32 j = 0; j < scene->num_cams; j++) {
                // if the current frame is ready, upload for display,
                // otherwise wait for the frame to get ready
                while (scene->display_buffer[j][read_head].frame_number !=
                       to_display_frame_number) {
                    // std::cout << "main wait, " << read_head << ", " <<
                    // scene->display_buffer[j][read_head].frame_number <<
                    // ", "
                    // << to_display_frame_number << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                if (scene->use_cpu_buffer) {
                    // upload_texture(&scene->image_texture[j],
                    // scene->display_buffer[j][read_head].frame,
                    // scene->image_width[j], scene->image_height[j]); // 2x
                    // slower than pbo copy frame to cuda buffer
                    ck(cudaMemcpy(scene->pbo_cuda[j].cuda_buffer,
                                  scene->display_buffer[j][read_head].frame,
                                  scene->image_width[j] *
                                      scene->image_height[j] * 4,
                                  cudaMemcpyHostToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(
                        scene->image_width[j],
                        scene->image_height[j]); // Needs no arguments
                                                 // because texture and PBO
                                                 // are bound
                    unbind_pbo();
                    unbind_texture();
                } else {
                    ck(cudaMemcpy(scene->pbo_cuda[j].cuda_buffer,
                                  scene->display_buffer[j][read_head].frame,
                                  scene->image_width[j] *
                                      scene->image_height[j] * 4,
                                  cudaMemcpyDeviceToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(
                        scene->image_width[j],
                        scene->image_height[j]); // Needs no arguments
                                                 // because texture and PBO
                                                 // are bound
                    unbind_pbo();
                    unbind_texture();
                }

                // sync yolo detection
                if (yolo_detection) {
                    std::unique_lock<std::mutex> lck(g_mutexes[j]);
                    // std::cout << "main_thread: acquire lock" <<
                    // std::endl;
                    yolo_input_frames_rgba[j] = scene->pbo_cuda[j].cuda_buffer;
                    g_ready[j] = true;
                    g_cvs[j].notify_one();
                }
            }
            current_frame_num = to_display_frame_number;
            
            // Automatic YOLO detection for current frame (playing video)
            if (auto_yolo_labeling && !yolo_model_path.empty() && skeleton->has_bbox) {
                if (!frameHasYoloDetections(current_frame_num, keypoints_map, skeleton) && 
                    yolo_processed_frames.find(current_frame_num) == yolo_processed_frames.end()) {
                    
                    // Mark frame as processed to avoid duplicate processing
                    yolo_processed_frames.insert(current_frame_num);
                    
                    // Enable YOLO detection flag
                    yolo_detection = true;
                    
                    std::cout << "Auto YOLO: Processing frame " << current_frame_num << std::endl;
                    
                    // Run YOLO inference on all cameras for this frame
                    for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                        unsigned char* frame_data = scene->display_buffer[cam_id][read_head].frame;
                        
                        if (frame_data) {
                            yolo_predictions[cam_id] = runYoloInference(
                                yolo_model_path, frame_data,
                                scene->image_width[cam_id],
                                scene->image_height[cam_id]);
                                
                            yolo_bboxes[cam_id].clear();
                            for (const auto& pred : yolo_predictions[cam_id]) {
                                yolo_bboxes[cam_id].emplace_back(pred);
                            }
                        }
                    }
                    
                    // Add YOLO detections to main bounding box system
                    if (!yolo_bboxes.empty() && std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(), [](const auto& cam_bboxes) { return !cam_bboxes.empty(); })) {
                        for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                            if (!yolo_bboxes[cam_id].empty()) {
                                // Ensure keypoints structure exists
                                bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene, skeleton);
                                    keypoints_map[current_frame_num] = keypoints;
                                }
                                
                                // Add YOLO detections to main bounding box system
                                for (const auto& yolo_bbox : yolo_bboxes[cam_id]) {
                                    if (yolo_bbox.is_valid) {
                                        BoundingBox bbox;
                                        
                                        bbox.rect = new ImPlotRect(
                                            yolo_bbox.x_min, yolo_bbox.x_max,
                                            yolo_bbox.y_min, yolo_bbox.y_max
                                        );
                                        
                                        bbox.state = RectTwoPoints;
                                        bbox.class_id = yolo_bbox.class_id;
                                        bbox.confidence = yolo_bbox.confidence;
                                        bbox.has_bbox_keypoints = false;
                                        bbox.bbox_keypoints2d = nullptr;
                                        bbox.active_kp_id = nullptr;
                                        
                                        if (skeleton->has_bbox && skeleton->has_skeleton && skeleton->num_nodes > 0) {
                                            allocate_bbox_keypoints(&bbox, scene, skeleton);
                                        }
                                        
                                        keypoints_map[current_frame_num]->bbox2d_list[cam_id].push_back(bbox);
                                    }
                                }
                                
                                std::cout << "Auto YOLO: Added " << yolo_bboxes[cam_id].size() 
                                         << " detections for camera " << cam_id << ", frame " << current_frame_num << std::endl;
                            }
                        }
                    }
                }
            }
        }

        // show frames in the buffer if selected
        if (video_loaded && (!play_video)) {
            static int select_corr_head = 0;
            ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frames in the buffer")) {
                {
                    for (u32 i = 0; i < scene->size_of_buffer; i++) {
                        int seletable_frame_id =
                            (i + read_head) % scene->size_of_buffer;
                        char label[32];
                        if (input_is_imgs) {
                            snprintf(
                                label, sizeof(label), "%d: %s",
                                scene->display_buffer[0][seletable_frame_id]
                                    .frame_number,
                                imgs_names[i].c_str());
                        } else {
                            sprintf(label, "Frame %d",
                                    scene->display_buffer[0][seletable_frame_id]
                                        .frame_number);
                        }
                        if (ImGui::Selectable(label, pause_selected == i)) {
                            // start from the lowest frame
                            pause_selected = i;
                        }
                    }
                }

                if (ImGui::IsKeyPressed(ImGuiKey_Comma, true)) {
                    if (pause_selected > 0) {
                        pause_selected--;
                    }
                };

                if (ImGui::IsKeyPressed(ImGuiKey_Period, true)) {
                    if (pause_selected < (scene->size_of_buffer - 1)) {
                        pause_selected++;
                    }
                };
            }
            ImGui::End();

            select_corr_head =
                (pause_selected + read_head) % scene->size_of_buffer;
            current_frame_num =
                scene->display_buffer[0][select_corr_head].frame_number;

            // Automatic YOLO detection for current frame
            if (auto_yolo_labeling && !yolo_model_path.empty() && skeleton->has_bbox) {
                if (!frameHasYoloDetections(current_frame_num, keypoints_map, skeleton) && 
                    yolo_processed_frames.find(current_frame_num) == yolo_processed_frames.end()) {
                    
                    // Mark frame as processed to avoid duplicate processing
                    yolo_processed_frames.insert(current_frame_num);
                    
                    // Enable YOLO detection flag
                    yolo_detection = true;
                    
                    std::cout << "Auto YOLO: Processing frame " << current_frame_num << std::endl;
                    
                    // Run YOLO inference on all cameras for this frame
                    for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                        unsigned char* frame_data = scene->display_buffer[cam_id][select_corr_head].frame;
                        
                        if (frame_data) {
                            yolo_predictions[cam_id] = runYoloInference(
                                yolo_model_path, frame_data,
                                scene->image_width[cam_id],
                                scene->image_height[cam_id]);
                                
                            yolo_bboxes[cam_id].clear();
                            for (const auto& pred : yolo_predictions[cam_id]) {
                                yolo_bboxes[cam_id].emplace_back(pred);
                            }
                        }
                    }
                    
                    // Add YOLO detections to main bounding box system
                    if (!yolo_bboxes.empty() && std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(), [](const auto& cam_bboxes) { return !cam_bboxes.empty(); })) {
                        for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                            if (!yolo_bboxes[cam_id].empty()) {
                                // Ensure keypoints structure exists
                                bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                                if (!keypoints_find) {
                                    KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene, skeleton);
                                    keypoints_map[current_frame_num] = keypoints;
                                }
                                
                                // Add YOLO detections to main bounding box system
                                for (const auto& yolo_bbox : yolo_bboxes[cam_id]) {
                                    if (yolo_bbox.is_valid) {
                                        BoundingBox bbox;
                                        
                                        bbox.rect = new ImPlotRect(
                                            yolo_bbox.x_min, yolo_bbox.x_max,
                                            yolo_bbox.y_min, yolo_bbox.y_max
                                        );
                                        
                                        bbox.state = RectTwoPoints;
                                        bbox.class_id = yolo_bbox.class_id;
                                        bbox.confidence = yolo_bbox.confidence;
                                        bbox.has_bbox_keypoints = false;
                                        bbox.bbox_keypoints2d = nullptr;
                                        bbox.active_kp_id = nullptr;
                                        
                                        if (skeleton->has_bbox && skeleton->has_skeleton && skeleton->num_nodes > 0) {
                                            allocate_bbox_keypoints(&bbox, scene, skeleton);
                                        }
                                        
                                        keypoints_map[current_frame_num]->bbox2d_list[cam_id].push_back(bbox);
                                    }
                                }
                                
                                std::cout << "Auto YOLO: Added " << yolo_bboxes[cam_id].size() 
                                         << " detections for camera " << cam_id << ", frame " << current_frame_num << std::endl;
                            }
                        }
                    }
                }
            }

            for (int j = 0; j < scene->num_cams; j++) {
                if (scene->use_cpu_buffer) {
                    // upload_texture(&scene->image_texture[j],
                    // scene->display_buffer[j][select_corr_head].frame,
                    // scene->image_width[j], scene->image_height[j]);
                    ck(cudaMemcpy(
                        scene->pbo_cuda[j].cuda_buffer,
                        scene->display_buffer[j][select_corr_head].frame,
                        scene->image_width[j] * scene->image_height[j] * 4,
                        cudaMemcpyHostToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(
                        scene->image_width[j],
                        scene->image_height[j]); // Needs no arguments
                                                 // because texture and PBO
                                                 // are bound
                    unbind_pbo();
                    unbind_texture();
                } else {
                    ck(cudaMemcpy(
                        scene->pbo_cuda[j].cuda_buffer,
                        scene->display_buffer[j][select_corr_head].frame,
                        scene->image_width[j] * scene->image_height[j] * 4,
                        cudaMemcpyDeviceToDevice));
                    bind_pbo(&scene->pbo_cuda[j].pbo);
                    bind_texture(&scene->image_texture[j]);
                    upload_image_pbo_to_texture(
                        scene->image_width[j],
                        scene->image_height[j]); // Needs no arguments
                                                 // because texture and PBO
                                                 // are bound
                    unbind_pbo();
                    unbind_texture();
                }
            }
        }

        if (toggle_play_status && play_video) {
            play_video = false;
            toggle_play_status = false;
        }

        if (plot_keypoints_flag) {
            if (keypoints_map.find(current_frame_num) == keypoints_map.end()) {
                keypoints_find = false;
            } else {
                keypoints_find = true;
            }
        }

        // Render a video frame
        if (video_loaded) {
            for (int j = 0; j < scene->num_cams; j++) {

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
                    if (j % 4 == 0) {
                        window_pos.y = 200.0;
                        window_pos.x = (j / 4.0) * 500;
                    } else if (j % 4 == 1) {
                        window_pos.y = 600.0;
                        window_pos.x = (j - 1) / 4.0 * 500;
                    } else if (j % 4 == 2) {
                        window_pos.y = 1000.0;
                        window_pos.x = (j - 1) / 4.0 * 500;
                    } else {
                        window_pos.y = 1400.0;
                        window_pos.x = (j - 1) / 4.0 * 500;
                    }
                }

                ImGui::SetNextWindowPos(window_pos, ImGuiCond_FirstUseEver);
                ImGui::Begin(camera_names[j].c_str());
                ImGui::BeginGroup();
                std::string scene_name = "scene view" + std::to_string(j);
                ImGui::BeginChild(
                    scene_name.c_str(),
                    ImVec2(0,
                           -ImGui::GetFrameHeightWithSpacing())); // Leave room
                                                                  // for 1 line
                                                                  // below
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                // ImGui::Image((void*)(intptr_t)image_texture[j],
                // avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size,
                                      ImPlotFlags_Equal |
                                          ImPlotAxisFlags_AutoFit |
                                          ImPlotFlags_Crosshairs)) {
                    ImPlot::PlotImage(
                        "##no_image_name",
                        (ImTextureID)(intptr_t)scene->image_texture[j],
                        ImVec2(0, 0),
                        ImVec2(scene->image_width[j], scene->image_height[j]));

                    if (plot_keypoints_flag) {
                        // plot arena for testing camera parameters
                        // gui_plot_perimeter(&camera_params[j],
                        // scene->image_height[j]); if (scene->num_cams > 1)
                        // {
                        //     gui_plot_world_coordinates(&camera_params[j],
                        //     j, scene->image_height[j]);
                        // }

                        // labeling
                        if (ImPlot::IsPlotHovered()) {
                            is_view_focused[j] = true;

                            if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                                // create keypoints
                                if (!keypoints_find) {
                                    // not found
                                    KeyPoints *keypoints =
                                        (KeyPoints *)malloc(sizeof(KeyPoints));
                                    allocate_keypoints(keypoints, scene,
                                                       skeleton);
                                    keypoints_map[current_frame_num] =
                                        keypoints;
                                }
                            }

                            if (keypoints_find && skeleton->has_skeleton && !skeleton->has_bbox) {
                                u32 *kp = &(keypoints_map[current_frame_num]
                                                ->active_id[j]);
                                if (ImGui::IsKeyPressed(ImGuiKey_W, false)) {
                                    // labeling sequentially each view
                                    ImPlotPoint mouse =
                                        ImPlot::GetPlotMousePos();
                                    keypoints_map[current_frame_num]
                                        ->keypoints2d[j][*kp]
                                        .position = {mouse.x, mouse.y};
                                    keypoints_map[current_frame_num]
                                        ->keypoints2d[j][*kp]
                                        .is_labeled = true;
                                    keypoints_map[current_frame_num]
                                        ->keypoints2d[j][*kp]
                                        .is_triangulated = false;
                                    if (*kp < (skeleton->num_nodes - 1)) {
                                        (*kp)++;
                                    }
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_A, true)) {
                                    if (*kp <= 0) {
                                        *kp = 0;
                                    } else
                                        (*kp)--;
                                }

                                if (ImGui::IsKeyPressed(ImGuiKey_D, true)) {
                                    if (*kp >= skeleton->num_nodes - 1) {
                                        *kp = skeleton->num_nodes - 1;
                                    } else
                                        (*kp)++;
                                }

                                if (ImGui::IsKeyPressed(
                                        ImGuiKey_E,
                                        false)) // skip to the last keypoint
                                {
                                    *kp = skeleton->num_nodes - 1;
                                }

                                if (ImGui::IsKeyPressed(
                                        ImGuiKey_Q,
                                        false)) // go to the first keypoint
                                {
                                    *kp = 0;
                                }

                                // delete all keypoint on a frame
                                if (ImGui::IsKeyPressed(ImGuiKey_Backspace,
                                                        false)) {
                                    free_keypoints(
                                        keypoints_map[current_frame_num],
                                        scene);
                                    keypoints_map.erase(current_frame_num);
                                    keypoints_find = false;
                                }
                            }
                            
                            // Bounding box input handling
                            if (skeleton->has_bbox) {
                                if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle, false)) {
                                    // Ensure keypoints structure exists for bounding boxes, even if skeleton has 0 keypoints
                                    bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                                    if (!keypoints_find) {
                                        KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene, skeleton);
                                        keypoints_map[current_frame_num] = keypoints;
                                    }
                                    
                                    ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                    BoundingBox new_bbox;
                                    new_bbox.rect = new ImPlotRect(mouse.x, mouse.x, mouse.y, mouse.y);
                                    new_bbox.state = RectOnePoint;
                                    new_bbox.class_id = current_bbox_class;  // Use currently selected class
                                    new_bbox.confidence = 1.0f;
                                    new_bbox.has_bbox_keypoints = false;
                                    new_bbox.bbox_keypoints2d = nullptr;
                                    new_bbox.active_kp_id = nullptr;
                                    
                                    // Allocate keypoints for this bounding box only if skeleton has both bbox and skeleton
                                    if (skeleton->has_bbox && skeleton->has_skeleton && skeleton->num_nodes > 0) {
                                        allocate_bbox_keypoints(&new_bbox, scene, skeleton);
                                    }
                                    
                                    keypoints_map[current_frame_num]->bbox2d_list[j].push_back(new_bbox);
                                }
                                
                                // Handle dragging for multiple bounding boxes
                                for (auto& bbox : keypoints_map[current_frame_num]->bbox2d_list[j]) {
                                    if (bbox.state == RectOnePoint && ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                                        ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                        bbox.rect->X.Max = mouse.x;
                                        bbox.rect->Y.Min = mouse.y;
                                    }
                                    
                                    if (bbox.state == RectOnePoint && ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
                                        bbox.state = RectTwoPoints;
                                    }
                                }
                                
                            }
                        } else {
                            is_view_focused[j] = false;
                        }

                        // Plot bounding boxes (both with and without keypoints)
                        if (skeleton->has_bbox) {
                            bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                            if (!keypoints_find) {
                                KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                                allocate_keypoints(keypoints, scene, skeleton);
                                keypoints_map[current_frame_num] = keypoints;
                            }
                            
                            // Determine which bbox is active (under mouse cursor)
                            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                            int active_bbox_idx = -1;
                            
                            // Check which bbox the mouse is hovering over
                            for (int bbox_idx = 0; bbox_idx < keypoints_map[current_frame_num]->bbox2d_list[j].size(); bbox_idx++) {
                                BoundingBox& bbox = keypoints_map[current_frame_num]->bbox2d_list[j][bbox_idx];
                                if (bbox.rect && bbox.state == RectTwoPoints && 
                                    is_point_in_bbox(mouse.x, mouse.y, bbox.rect)) {
                                    active_bbox_idx = bbox_idx;
                                    break;
                                }
                            }
                            
                            // Plot multiple bounding boxes
                            for (int bbox_idx = 0; bbox_idx < keypoints_map[current_frame_num]->bbox2d_list[j].size(); bbox_idx++) {
                                BoundingBox& bbox = keypoints_map[current_frame_num]->bbox2d_list[j][bbox_idx];
                                if (bbox.rect) {
                                    // Get color based on class_id
                                    ImVec4 bbox_color = ImVec4(0.5f, 1.0f, 1.0f, 1.0f); // Default fallback
                                    if (bbox.class_id >= 0 && bbox.class_id < bbox_class_colors.size()) {
                                        bbox_color = bbox_class_colors[bbox.class_id];
                                    }
                                    
                                    // Reduce opacity for inactive bboxes
                                    bool is_active_bbox = (bbox_idx == active_bbox_idx);
                                    if (!is_active_bbox) {
                                        bbox_color.w = 0.6f; // Make inactive bboxes more transparent
                                    }
                                    
                                    // Draw completed bounding boxes
                                    if (bbox.state == RectTwoPoints) {
                                        bool bbox_clicked = false, bbox_hovered = false, bbox_held = false;
                                        
                                        // Store previous rect for this specific bbox
                                        static std::map<std::pair<int, int>, ImPlotRect> bbox_prev_rects; // frame -> {camera, bbox_idx} -> prev_rect
                                        auto bbox_key = std::make_pair(j, bbox_idx);
                                        
                                        // Initialize previous rect if not exists
                                        if (bbox_prev_rects.find(bbox_key) == bbox_prev_rects.end()) {
                                            bbox_prev_rects[bbox_key] = *bbox.rect;
                                        }
                                        
                                        ImPlotRect prev_bbox_rect = bbox_prev_rects[bbox_key];
                                        
                                        // Only allow interaction if this is the active bbox or no bbox is active
                                        ImPlotDragToolFlags drag_flags = ImPlotDragToolFlags_None;
                                        if (!is_active_bbox && active_bbox_idx != -1) {
                                            drag_flags = ImPlotDragToolFlags_NoInputs;
                                        }
                                        
                                        bool bbox_modified = ImPlot::DragRect(1000 + bbox_idx, &bbox.rect->X.Min, &bbox.rect->Y.Min, 
                                                       &bbox.rect->X.Max, &bbox.rect->Y.Max, bbox_color, drag_flags,
                                                       &bbox_clicked, &bbox_hovered, &bbox_held);
                                        
                                        if (bbox_clicked || bbox_held || bbox_modified) {
                                            active_bbox_idx = bbox_idx; 
                                        }

                                        bool is_active_bbox = (bbox_idx == active_bbox_idx);
                                        
                                        // Scale bbox keypoints if bbox was resized and this is the active bbox
                                        if (bbox_modified && bbox.has_bbox_keypoints && bbox.bbox_keypoints2d && is_active_bbox) {
                                            scale_bbox_keypoints(&bbox, scene, skeleton, &prev_bbox_rect, bbox.rect);
                                            bbox_prev_rects[bbox_key] = *bbox.rect; // Update stored previous rect
                                        }
                                        
                                        // Update previous rect when not being dragged
                                        if (!bbox_held) {
                                            bbox_prev_rects[bbox_key] = *bbox.rect;
                                        }
                                        
                                        // Handle keyboard shortcuts when hovering over bounding box
                                        if (bbox_hovered) {
                                            // Update hovered bbox info for display
                                            hovered_bbox_cam = j;
                                            hovered_bbox_idx = bbox_idx;
                                            hovered_bbox_confidence = bbox.confidence;
                                            hovered_bbox_class = bbox.class_id;
                                            
                                            // Delete bounding box from current camera when 'T' key is pressed while hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_T, false)) {
                                                // Clean up bbox keypoints before deletion
                                                if (bbox.has_bbox_keypoints && bbox.bbox_keypoints2d) {
                                                    free(bbox.bbox_keypoints2d);
                                                    bbox.bbox_keypoints2d = nullptr;
                                                    free(bbox.active_kp_id);
                                                    bbox.active_kp_id = nullptr;
                                                }
                                                // Mark for deletion by setting state to RectNull
                                                delete bbox.rect;
                                                bbox.rect = nullptr;
                                                bbox.state = RectNull;
                                                bbox.has_bbox_keypoints = false;
                                            }
                                            
                                            // Delete bounding box from all cameras when 'F' key is pressed while hovering
                                            if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                                                // Find this bbox's class_id and delete all bboxes with same class from all cameras
                                                int target_class_id = bbox.class_id;
                                                for (int cam_idx = 0; cam_idx < scene->num_cams; cam_idx++) {
                                                    auto& bbox_list = keypoints_map[current_frame_num]->bbox2d_list[cam_idx];
                                                    for (auto& other_bbox : bbox_list) {
                                                        if (other_bbox.class_id == target_class_id && 
                                                            other_bbox.state != RectNull && 
                                                            other_bbox.rect != nullptr) {
                                                            // Clean up bbox keypoints before deletion
                                                            if (other_bbox.has_bbox_keypoints && other_bbox.bbox_keypoints2d) {
                                                                free(other_bbox.bbox_keypoints2d);
                                                                other_bbox.bbox_keypoints2d = nullptr;
                                                                free(other_bbox.active_kp_id);
                                                                other_bbox.active_kp_id = nullptr;
                                                            }
                                                            delete other_bbox.rect;
                                                            other_bbox.rect = nullptr;
                                                            other_bbox.state = RectNull;
                                                            other_bbox.has_bbox_keypoints = false;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Draw bounding boxes being created (one point set)
                                    else if (bbox.state == RectOnePoint) {
                                        // Draw a preview rectangle while dragging
                                        double xs[5] = {bbox.rect->X.Min, bbox.rect->X.Max, bbox.rect->X.Max, bbox.rect->X.Min, bbox.rect->X.Min};
                                        double ys[5] = {bbox.rect->Y.Max, bbox.rect->Y.Max, bbox.rect->Y.Min, bbox.rect->Y.Min, bbox.rect->Y.Max};
                                        ImPlot::SetNextLineStyle(bbox_color, 2.0f);
                                        ImPlot::PlotLine("##bbox_preview", xs, ys, 5);
                                    }
                                    
                                    // Plot keypoints within this bounding box (only if keypoints are enabled and skeleton has keypoints)
                                    if (bbox.state == RectTwoPoints && keypoints_find && skeleton->has_skeleton && bbox.has_bbox_keypoints) {
                                        bool is_saved = true;
                                        // Only allow interaction with keypoints if this is the active bbox
                                        gui_plot_bbox_keypoints(&bbox, skeleton, j, scene->num_cams, is_active_bbox, is_saved, bbox_idx);
                                        
                                        // Handle keypoint labeling with W key for bounding box keypoints (only on active bbox)
                                        if (is_active_bbox && ImGui::IsKeyPressed(ImGuiKey_W, false)) {
                                            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
                                            if (is_point_in_bbox(mouse.x, mouse.y, bbox.rect)) {
                                                u32 active_kp = bbox.active_kp_id[j];
                                                if (active_kp < skeleton->num_nodes) {
                                                    bbox.bbox_keypoints2d[j][active_kp].position = {mouse.x, mouse.y};
                                                    bbox.bbox_keypoints2d[j][active_kp].is_labeled = true;
                                                    bbox.bbox_keypoints2d[j][active_kp].is_triangulated = false;
                                                    constrain_keypoint_to_bbox(&bbox.bbox_keypoints2d[j][active_kp], bbox.rect);
                                                    if (active_kp < (skeleton->num_nodes - 1)) {
                                                        bbox.active_kp_id[j]++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (keypoints_find) {
                            // Only plot keypoints if skeleton has keypoints and we're not in bbox+keypoints mode
                            if (skeleton->has_skeleton && !skeleton->has_bbox) {
                                gui_plot_keypoints(
                                    keypoints_map.at(current_frame_num), skeleton,
                                    j, scene->num_cams);
                            }
                            
                            // think more general solution of multiple sets
                            // of keypoints
                            if (skeleton->name == "Rat4Box" ||
                                skeleton->name == "Rat4Box3Ball") {
                                gui_plot_bbox_from_keypoints(
                                    keypoints_map.at(current_frame_num),
                                    skeleton, j, 4, 5);
                            }
                        }
                    }
                    ImPlot::EndPlot();
                }

                ImGui::EndChild();

                if (to_display_frame_number ==
                    (dc_context->total_num_frame - 1)) {
                    if (ImGui::Button(ICON_FK_REPEAT)) {
                        // seek to zero
                        for (int i = 0; i < scene->num_cams; i++) {
                            scene->seek_context[i].seek_frame = 0;
                            scene->seek_context[i].use_seek = true;
                            scene->seek_context[i].seek_accurate = false;
                        }

                        for (int i = 0; i < scene->num_cams; i++) {
                            // synchronize seeking
                            while (!(scene->seek_context[i].seek_done)) {
                                std::this_thread::sleep_for(
                                    std::chrono::milliseconds(1));
                                // std::cout << "Seeking Cam" << i << ", "
                                // << scene->seek_context[i].seek_done <<
                                // std::endl;
                            }
                        }

                        for (int i = 0; i < scene->num_cams; i++) {
                            scene->seek_context[i].seek_done = false;
                        }

                        to_display_frame_number =
                            scene->seek_context[0].seek_frame;
                        read_head = 0;
                        just_seeked = true;
                        slider_frame_number = to_display_frame_number;
                    }
                } else {
                    if (ImGui::Button(play_video ? ICON_FK_PAUSE
                                                 : ICON_FK_PLAY)) {
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
                if (ImGui::Button(ICON_FK_PLUS)) {
                    // advance_clicks++;
                    play_video = true;
                    toggle_play_status = true;
                }
                ImGui::PopButtonRepeat();
                ImGui::SameLine();

                slider_just_changed =
                    ImGui::SliderInt("##frame count", &slider_frame_number, 0,
                                     dc_context->estimated_num_frames);

                if (slider_just_changed) {
                    std::cout << "main, seeking: " << slider_frame_number
                              << std::endl;

                    for (int i = 0; i < scene->num_cams; i++) {
                        scene->seek_context[i].seek_frame =
                            (uint64_t)slider_frame_number;
                        scene->seek_context[i].use_seek = true;
                        scene->seek_context[i].seek_accurate = false;
                    }

                    for (int i = 0; i < scene->num_cams; i++) {
                        // synchronize seeking
                        while (!(scene->seek_context[i].seek_done)) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(1));
                            // std::cout << "Seeking Cam" << i << ", " <<
                            // scene->seek_context[i].seek_done <<
                            // std::endl;
                        }
                    }

                    for (int i = 0; i < scene->num_cams; i++) {
                        scene->seek_context[i].seek_done = false;
                    }

                    // std::cout << "Main thread seeking done to frame: " <<
                    // scene->seek_context[0].seek_frame << std::endl;
                    to_display_frame_number = scene->seek_context[0].seek_frame;
                    read_head = 0;
                    just_seeked = true;
                    slider_frame_number = to_display_frame_number;
                }

                ImGui::EndGroup();
                ImGui::End();
            }

            if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
                play_video = !play_video;
                if (!play_video) {
                    pause_selected = 0;
                }
            }
            
            // Bounding box class switching keybinds
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
                // Switch to previous class
                if (bbox_class_names.size() > 0) {
                    current_bbox_class = (current_bbox_class - 1 + bbox_class_names.size()) % bbox_class_names.size();
                }
            }
            
            if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
                if (bbox_class_names.size() > 0) {
                    // Switch to next class
                    current_bbox_class = (current_bbox_class + 1) % bbox_class_names.size();
                }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_N, false)) {
                create_new_bbox_class();
            }
        }

        if (plot_keypoints_flag) {
            if (ImGui::Begin("Keypoints")) {

                const float TEXT_BASE_HEIGHT =
                    ImGui::GetTextLineHeightWithSpacing();
                
                // Bounding Box Class Management Section
                ImGui::SeparatorText("Bounding Box Classes");
                
                // Current class selection combo
                if (ImGui::BeginCombo("Current Class", bbox_class_names[current_bbox_class].c_str())) {
                    for (int i = 0; i < bbox_class_names.size(); i++) {
                        bool is_selected = (current_bbox_class == i);
                        
                        // Show color indicator next to class name
                        ImGui::ColorButton("##color", bbox_class_colors[i], ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoBorder, ImVec2(15, 15));
                        ImGui::SameLine();
                        
                        if (ImGui::Selectable(bbox_class_names[i].c_str(), is_selected)) {
                            current_bbox_class = i;
                        }
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                
                // Add new class
                ImGui::SetNextItemWidth(200);
                if (ImGui::InputTextWithHint("##new_class", "Enter new class name...", new_class_name_buffer, sizeof(new_class_name_buffer))) {
                    // Input changed
                }
                ImGui::SameLine();
                if (ImGui::Button("Add Class") && strlen(new_class_name_buffer) > 0) {
                    bbox_class_names.push_back(std::string(new_class_name_buffer));
                    // Generate a unique color for the new class (HSV with different hues)
                    float hue = (bbox_class_colors.size() * 0.618034f); // Golden ratio for nice color distribution
                    while (hue > 1.0f) hue -= 1.0f;
                    ImVec4 new_color = (ImVec4)ImColor::HSV(hue, 0.8f, 1.0f);
                    bbox_class_colors.push_back(new_color);
                    current_bbox_class = bbox_class_names.size() - 1;
                    memset(new_class_name_buffer, 0, sizeof(new_class_name_buffer));
                }
                
                // Edit current class color
                if (current_bbox_class >= 0 && current_bbox_class < bbox_class_colors.size()) {
                    ImGui::SetNextItemWidth(200);
                    ImGui::ColorEdit3("Class Color", (float*)&bbox_class_colors[current_bbox_class], ImGuiColorEditFlags_NoInputs);
                }
                
                // Delete class (only if not the last one)
                if (bbox_class_names.size() > 1) {
                    ImGui::SameLine();
                    if (ImGui::Button("Delete Class")) {
                        bbox_class_names.erase(bbox_class_names.begin() + current_bbox_class);
                        bbox_class_colors.erase(bbox_class_colors.begin() + current_bbox_class);
                        if (current_bbox_class >= bbox_class_names.size()) {
                            current_bbox_class = bbox_class_names.size() - 1;
                        }
                    }
                }
                
                ImGui::Separator();
                
                // Check if skeleton is valid and has nodes before creating the table
                if (skeleton && skeleton->num_nodes > 0) {
                    const int rows_count = scene->num_cams;
                    const int columns_count = skeleton->num_nodes + 1;

                    static ImGuiTableFlags table_flags =
                        ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                        ImGuiTableFlags_SizingFixedFit |
                        ImGuiTableFlags_BordersOuter |
                        ImGuiTableFlags_BordersInnerH |
                        ImGuiTableFlags_Hideable | ImGuiTableFlags_Resizable |
                        ImGuiTableFlags_HighlightHoveredColumn;

                    if (ImGui::BeginTable(
                            "table_angled_headers", columns_count, table_flags,
                            ImVec2(0.0f, TEXT_BASE_HEIGHT * 12))) {
                        ImGui::TableSetupColumn(
                            "Name", ImGuiTableColumnFlags_NoHide |
                                        ImGuiTableColumnFlags_NoReorder);
                        for (int column = 1; column < columns_count; column++)
                            ImGui::TableSetupColumn(
                                skeleton->node_names[column - 1].c_str(),
                                ImGuiTableColumnFlags_AngledHeader |
                                    ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupScrollFreeze(1, 2);

                        ImGui::
                            TableAngledHeadersRow(); // Draw angled headers
                                                     // for all columns with
                                                     // the
                                                     // ImGuiTableColumnFlags_AngledHeader
                                                     // flag.
                        ImGui::TableHeadersRow(); // Draw remaining headers
                                                  // and allow access to
                                                  // context-menu and other
                                                  // functions.

                        for (int row = 0; row < rows_count; row++) {
                            ImGui::PushID(row);
                            ImGui::TableNextRow();

                            if (is_view_focused[row] && keypoints_find) {
                                ImU32 row_bg_color = ImGui::GetColorU32(
                                    ImVec4(0.7f, 0.3f, 0.3f, 0.65f));
                                ImGui::TableSetBgColor(
                                    ImGuiTableBgTarget_RowBg0, row_bg_color);
                            }

                            ImGui::TableSetColumnIndex(0);
                            ImGui::AlignTextToFramePadding();
                            ImGui::Text(camera_names[row].c_str());
                            for (int column = 1; column < columns_count;
                                 column++)
                                if (ImGui::TableSetColumnIndex(column)) {
                                    if (keypoints_find) {
                                        ImVec4 node_color;
                                        if (keypoints_map[current_frame_num]
                                                ->active_id[row] ==
                                            column - 1) {
                                            node_color = (ImVec4)ImColor::HSV(
                                                0.8, 1.0f, 1.0f);
                                        } else {
                                            if (keypoints_map[current_frame_num]
                                                    ->keypoints2d[row]
                                                                 [column - 1]
                                                    .is_labeled) {
                                                node_color =
                                                    skeleton
                                                        ->node_colors[column -
                                                                      1];
                                                node_color.w = 0.9;
                                            }
                                        }

                                        if (keypoints_map[current_frame_num]
                                                ->keypoints2d[row][column - 1]
                                                .is_triangulated) {
                                            ImGui::TextColored(
                                                ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                                "T");
                                        }

                                        ImU32 cell_bg_color =
                                            ImGui::GetColorU32(node_color);
                                        ImGui::TableSetBgColor(
                                            ImGuiTableBgTarget_CellBg,
                                            cell_bg_color);
                                    }
                                }
                            ImGui::PopID();
                        }
                        ImGui::EndTable();
                    }
                } else {
                    ImGui::Text("Bounding box mode: No keypoints to display");
                }
            }
            ImGui::End();
        }

        if (plot_keypoints_flag) {
            if (ImGui::Begin("Labeling Tool")) {

                if (scene->num_cams > 1) {
                    bool keypoint_triangulated_all = true;
                    if (keypoints_find) {
                        for (int i = 0; i < scene->num_cams; i++) {
                            for (int j = 0; j < skeleton->num_nodes; j++) {
                                if (!keypoints_map.at(current_frame_num)
                                         ->keypoints2d[i][j]
                                         .is_triangulated) {
                                    keypoint_triangulated_all = false;
                                }
                            }
                        }
                    } else {
                        keypoint_triangulated_all = false;
                    }

                    bool enabled = keypoints_find;
                    bool apply_color = !keypoint_triangulated_all && enabled;
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

                    ImGui::BeginDisabled(!enabled);
                    if (ImGui::Button("Triangulate")) {
                        reprojection(keypoints_map.at(current_frame_num),
                                     skeleton, camera_params, scene);
                    }
                    ImGui::EndDisabled();

                    if (skeleton->has_bbox && keypoints_find) {
                        if (ImGui::Button("Triangulate Bounding Boxes") || 
                            ImGui::IsKeyPressed(ImGuiKey_B, false)) {
                            triangulate_bounding_boxes(keypoints_map.at(current_frame_num),
                                                      skeleton, camera_params, scene, current_frame_num);
                        }
                    }

                    if (apply_color) {
                        ImGui::PopStyleColor(3);
                    }

                    if (keypoints_find) {
                        if (ImGui::IsKeyPressed(ImGuiKey_S,
                                                false)) // triangulate
                        {
                            reprojection(keypoints_map.at(current_frame_num),
                                         skeleton, camera_params, scene);
                        }
                    }
                } else {
                    // Display message when skeleton is not properly loaded
                    ImGui::Text("Please load a skeleton to view keypoints.");
                }

                if (ImGui::Button("Update keypoints working directory")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = root_dir;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseKeypointsFolder",
                        "Choose keypoints working directory", nullptr, config);
                }
                ImGui::SameLine();
                ImGui::Text("%s", keypoints_root_folder.c_str());

                if (ImGui::Button("Save Labeled Data") ||
                    (ImGui::GetIO().KeyCtrl &&
                     ImGui::IsKeyPressed(ImGuiKey_S, false))) {
                    save_keypoints(keypoints_map, skeleton,
                                   keypoints_root_folder, scene->num_cams,
                                   camera_names, &input_is_imgs, imgs_names);
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
                        if (load_keypoints_depreciated(
                                keypoints_map, skeleton, keypoints_root_folder,
                                scene, camera_names, error_message)) {
                            free_all_keypoints(keypoints_map, scene);
                            show_error = true;
                        }

                    } else {
                        std::string most_recent_folder;
                        if (find_most_recent_labels(keypoints_root_folder,
                                                    most_recent_folder,
                                                    error_message)) {
                            show_error = true;
                        } else {
                            if (load_keypoints(most_recent_folder,
                                               keypoints_map, skeleton, scene,
                                               camera_names, error_message)) {
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
                    config.path = keypoints_root_folder;
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "LoadFromSelected", "Load from selected", nullptr,
                        config);
                }

                // Find next frame with actual labeled keypoints
                auto next_labeled_frame_it = keypoints_map.end();
                for (auto it = keypoints_map.upper_bound(current_frame_num); it != keypoints_map.end(); ++it) {
                    const auto& [frame_num, keypoints] = *it;
                    if (!keypoints) continue;
                    
                    bool has_labels = false;
                    
                    if (skeleton->has_skeleton) {
                        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS && !has_labels; cam_id++) {
                            for (int kp_id = 0; kp_id < skeleton->num_nodes && !has_labels; kp_id++) {
                                if (keypoints->keypoints2d[cam_id][kp_id].is_labeled) {
                                    has_labels = true;
                                }
                            }
                        }
                    }
                    
                    if (!has_labels && skeleton->has_bbox) {
                        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS && !has_labels; cam_id++) {
                            for (const auto& bbox : keypoints->bbox2d_list[cam_id]) {
                                if (bbox.confidence >= 1.0f && bbox.state == RectTwoPoints) {
                                    has_labels = true;
                                    break;
                                }
                                if (bbox.confidence >= 1.0f && bbox.has_bbox_keypoints) {
                                    for (int kp_id = 0; kp_id < skeleton->num_nodes && !has_labels; kp_id++) {
                                        if (bbox.bbox_keypoints2d && bbox.bbox_keypoints2d[cam_id] && bbox.bbox_keypoints2d[cam_id][kp_id].is_labeled) {
                                            has_labels = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if (has_labels) {
                        next_labeled_frame_it = it;
                        break;
                    }
                }
                
                // If no labeled frame found after current, wrap around to beginning
                if (next_labeled_frame_it == keypoints_map.end()) {
                    for (auto it = keypoints_map.begin(); it != keypoints_map.upper_bound(current_frame_num); ++it) {
                        const auto& [frame_num, keypoints] = *it;
                        if (!keypoints) continue;
                        
                        bool has_labels = false;
                        
                        if (skeleton->has_skeleton) {
                            for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS && !has_labels; cam_id++) {
                                for (int kp_id = 0; kp_id < skeleton->num_nodes && !has_labels; kp_id++) {
                                    if (keypoints->keypoints2d[cam_id][kp_id].is_labeled) {
                                        has_labels = true;
                                    }
                                }
                            }
                        }
                        
                        if (!has_labels && skeleton->has_bbox) {
                            for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS && !has_labels; cam_id++) {
                                for (const auto& bbox : keypoints->bbox2d_list[cam_id]) {
                                    if (bbox.confidence >= 1.0f && bbox.state == RectTwoPoints) {
                                        has_labels = true;
                                        break;
                                    }
                                    if (bbox.confidence >= 1.0f && bbox.has_bbox_keypoints) {
                                        for (int kp_id = 0; kp_id < skeleton->num_nodes && !has_labels; kp_id++) {
                                            if (bbox.bbox_keypoints2d && bbox.bbox_keypoints2d[cam_id] && bbox.bbox_keypoints2d[cam_id][kp_id].is_labeled) {
                                                has_labels = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (has_labels) {
                            next_labeled_frame_it = it;
                            break;
                        }
                    }
                }

                ImGui::Separator();
                if (next_labeled_frame_it != keypoints_map.end()) {
                    ImGui::Text("Next labeled frame : %d", (*next_labeled_frame_it).first);
                } else {
                    ImGui::Text("Next labeled frame : none");
                }
                if (ImGui::Button("Jump to Next Labeled Frame") ||
                    ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) {
                    if (next_labeled_frame_it != keypoints_map.end()) {
                        for (int i = 0; i < scene->num_cams; i++) {
                            scene->seek_context[i].seek_frame =
                                (uint64_t)(*next_labeled_frame_it).first;
                            scene->seek_context[i].use_seek = true;
                            scene->seek_context[i].seek_accurate = true;
                        }

                        for (int i = 0; i < scene->num_cams; i++) {
                            while (!(scene->seek_context[i].seek_done)) {
                                std::this_thread::sleep_for(
                                    std::chrono::milliseconds(1));
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
                }
                
                size_t labeled_count = 0;
                for (const auto& [frame_num, keypoints] : keypoints_map) {
                    if (!keypoints) continue;
                    
                    bool has_labels = false;
                    
                    if (skeleton->has_skeleton) {
                        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS && !has_labels; cam_id++) {
                            for (int kp_id = 0; kp_id < skeleton->num_nodes && !has_labels; kp_id++) {
                                if (keypoints->keypoints2d[cam_id][kp_id].is_labeled) {
                                    has_labels = true;
                                }
                            }
                        }
                    }
                    
                    if (!has_labels && skeleton->has_bbox) {
                        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS && !has_labels; cam_id++) {
                            for (const auto& bbox : keypoints->bbox2d_list[cam_id]) {
                                if (bbox.confidence >= 1.0f && bbox.state == RectTwoPoints) {
                                    has_labels = true;
                                    break;
                                }
                                if (bbox.confidence >= 1.0f && bbox.has_bbox_keypoints) {
                                    for (int kp_id = 0; kp_id < skeleton->num_nodes && !has_labels; kp_id++) {
                                        if (bbox.bbox_keypoints2d && bbox.bbox_keypoints2d[cam_id] && bbox.bbox_keypoints2d[cam_id][kp_id].is_labeled) {
                                            has_labels = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if (has_labels) {
                        labeled_count++;
                    }
                }
                
                ImGui::Text("Total labeled frames : %zu", labeled_count);

                if (ImGui::Button("Select YOLO")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = "/home/user/data"; // fix later
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseYoloModel", "Choose YOLO Model",
                        ".pt", config);
                }

                if (ImGuiFileDialog::Instance()->Display("ChooseYoloModel")) {
                    if (ImGuiFileDialog::Instance()->IsOk()) {
                        std::string model_path =
                            ImGuiFileDialog::Instance()->GetFilePathName();
                        if (!model_path.empty()) {
                            yolo_model_path = model_path;
                            std::cout << "Selected YOLO model: "
                                      << yolo_model_path << std::endl;
                        }
                    }
                    // close
                    ImGuiFileDialog::Instance()->Close();
                }

                // Show model path and run prediction button (only if model is selected)
                if (!yolo_model_path.empty()) {
                    ImGui::Text("Selected model: %s", yolo_model_path.c_str());
                    
                    // Automatic YOLO labeling checkbox
                    ImGui::Checkbox("Automatic YOLO Labeling", &auto_yolo_labeling);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Automatically run YOLO detection on current and subsequent frames");
                    }
                    
                    // YOLO parameter sliders
                    ImGui::SliderFloat("Confidence Threshold", &confidence_threshold, 0.01f, 0.99f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Minimum confidence score for detections");
                    }

                    ImGui::SliderFloat("NMS IoU Threshold", &iou_threshold, 0.01f, 0.99f, "%.2f");
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Non-Maximum Suppression IoU threshold");
                    }
                    
                    if (ImGui::Button("Run YOLO Prediction")) {
                        std::cout << "Running YOLO prediction on frame " << to_display_frame_number << std::endl;
                        
                        yolo_detection = true;
                        
                        // Clear existing bounding boxes for current frame before running inference
                        if (skeleton->has_bbox) {
                            bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                            if (keypoints_find) {
                                for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                                    auto& bbox_list = keypoints_map[current_frame_num]->bbox2d_list[cam_id];
                                    // Clear all bounding boxes for this camera
                                    for (auto& bbox : bbox_list) {
                                        if (bbox.rect) {
                                            delete bbox.rect;
                                            bbox.rect = nullptr;
                                        }
                                        if (bbox.has_bbox_keypoints && bbox.bbox_keypoints2d) {
                                            free(bbox.bbox_keypoints2d);
                                            bbox.bbox_keypoints2d = nullptr;
                                            free(bbox.active_kp_id);
                                            bbox.active_kp_id = nullptr;
                                        }
                                    }
                                    bbox_list.clear();
                                }
                                std::cout << "Cleared existing bounding boxes for frame " << current_frame_num << std::endl;
                            }
                        }

                        for (int cam_id=0; cam_id < scene->num_cams; cam_id++) {
                            unsigned char* frame_data = nullptr;
                            if (play_video && dc_context->decoding_flag) {
                                frame_data = scene->display_buffer[cam_id][read_head].frame;
                            } else if (!play_video) {
                                int select_corr_head = (pause_selected + read_head) % scene->size_of_buffer;
                                frame_data = scene->display_buffer[cam_id][select_corr_head].frame;
                            }

                            if (frame_data) {
                                yolo_predictions[cam_id] = runYoloInference(
                                    yolo_model_path, frame_data,
                                    scene->image_width[cam_id],
                                    scene->image_height[cam_id]);
                                    yolo_bboxes[cam_id].clear();
                                    for (const auto& pred : yolo_predictions[cam_id]) {
                                        yolo_bboxes[cam_id].emplace_back(pred);
                                    }

                                    std::cout << "YOLO predictions for camera " << cam_id << ": " << yolo_bboxes[cam_id].size() << " detections." << std::endl;
                            }
                        }

                        if (!yolo_bboxes.empty() && std::any_of(yolo_bboxes.begin(), yolo_bboxes.end(), [](const auto& cam_bboxes) { return !cam_bboxes.empty(); })) {
                            // Convert YOLO detections to main bounding box system
                            for (int cam_id = 0; cam_id < scene->num_cams; cam_id++) {
                                if (!yolo_bboxes[cam_id].empty()) {
                                    // Ensure keypoints structure exists
                                    bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                                    if (!keypoints_find) {
                                        KeyPoints *keypoints = (KeyPoints *)malloc(sizeof(KeyPoints));
                                        allocate_keypoints(keypoints, scene, skeleton);
                                        keypoints_map[current_frame_num] = keypoints;
                                    }
                                    
                                    // Add YOLO detections to main bounding box system
                                    for (const auto& yolo_bbox : yolo_bboxes[cam_id]) {
                                        if (yolo_bbox.is_valid) {
                                            BoundingBox bbox;
                                            
                                            // Create ImPlotRect from YOLO coordinates (no Y-axis flipping)
                                            bbox.rect = new ImPlotRect(
                                                yolo_bbox.x_min,  // X.Min
                                                yolo_bbox.x_max,  // X.Max
                                                yolo_bbox.y_min,  // Y.Min
                                                yolo_bbox.y_max   // Y.Max
                                            );
                                            
                                            bbox.state = RectTwoPoints;
                                            bbox.class_id = yolo_bbox.class_id;
                                            bbox.confidence = yolo_bbox.confidence;
                                            bbox.has_bbox_keypoints = false;
                                            bbox.bbox_keypoints2d = nullptr;
                                            bbox.active_kp_id = nullptr;
                                            
                                            // Allocate keypoints if skeleton supports both bbox and skeleton
                                            if (skeleton->has_bbox && skeleton->has_skeleton && skeleton->num_nodes > 0) {
                                                allocate_bbox_keypoints(&bbox, scene, skeleton);
                                            }
                                            
                                            // Add to main bounding box system
                                            keypoints_map[current_frame_num]->bbox2d_list[cam_id].push_back(bbox);
                                        }
                                    }
                                    
                                    std::cout << "Added " << yolo_bboxes[cam_id].size() 
                                             << " YOLO detections to main bounding box system for camera " << cam_id << std::endl;
                                }
                            }
                        }
                    }
                    
                    // Display detection info for hovered bounding box
                    if (skeleton->has_bbox) {
                        bool keypoints_find = keypoints_map.find(current_frame_num) != keypoints_map.end();
                        if (keypoints_find) {
                            ImGui::Separator();
                            ImGui::Text("Bounding Box Info:");
                            
                            if (hovered_bbox_cam >= 0 && hovered_bbox_idx >= 0) {
                                ImGui::Text("Camera: %d, Box: %d", hovered_bbox_cam, hovered_bbox_idx);
                                ImGui::Text("Class: %d, Confidence: %.1f%%", 
                                          hovered_bbox_class, hovered_bbox_confidence * 100.0f);
                            } else {
                                ImGui::Text("Hover over a bounding box to see details");
                            }
                        }
                    }
                }

                
            }
            ImGui::End();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseKeypointsFolder")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                keypoints_root_folder =
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
                if (load_keypoints(selected_folder, keypoints_map, skeleton,
                                   scene, camera_names, error_message)) {
                    free_all_keypoints(keypoints_map, scene);
                    show_error = true;
                }
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGui::IsKeyPressed(ImGuiKey_H, false)) {
            show_help_window = !show_help_window;
        }

        if (show_help_window) {
            if (ImGui::Begin("Help Menu")) {
                ImGui::Text("<Space>: toggle play and pause");
                ImGui::Text("<,>: previous image");
                ImGui::Text("<.>: next image");

                ImGui::SeparatorText("While hovering image");
                ImGui::Text("<c>: create keypoints on frame");
                ImGui::Text("<w>: drop active keypoint");
                ImGui::Text("<a>: active keypoint++ ");
                ImGui::Text("<d>: active keypoint--");
                ImGui::Text("<q>: active keypoint set to first node");
                ImGui::Text("<e>: active keypoint set to last node");
                ImGui::Text("<s> -> triangulate");
                ImGui::Text("<Backspace>: delete all keypoints");
                ImGui::Text("<Right Arrow>: next labeled frame");
                ImGui::Text("CTRL-S: save labels");

                ImGui::SeparatorText("Bounding box");
                ImGui::Text("<mouse middle button>: draw bbox, drag then release to finish drawing the bbox");
                ImGui::Text("Bboxes are created with the currently selected class in Keypoints panel");
                ImGui::Text("While hovering bounding boxes");
                ImGui::Text("<t>: delete bounding box from current camera");
                ImGui::Text("<f>: delete bounding box from all cameras");
                ImGui::Text("<z>: switch to previous bbox class");
                ImGui::Text("<x>: switch to next bbox class (creates new class if at end)");

                ImGui::SeparatorText("While hovering keypoints");
                ImGui::Text("<r>: delete active keypoint");
                ImGui::Text("<f>: delete active keypoint on all cameras");
                ImGui::Text("Click keypoint to active it");
            }
            ImGui::End();
        }

        // Skeleton Creator Window
        if (show_skeleton_creator)
        {
            ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Skeleton Creator", &show_skeleton_creator))
            {
                ImGui::SeparatorText("Skeleton Configuration");
                
                char name_buffer[256];
                strncpy(name_buffer, skeleton_creator_name.c_str(), sizeof(name_buffer));
                name_buffer[sizeof(name_buffer) - 1] = '\0';
                if (ImGui::InputText("Skeleton Name", name_buffer, sizeof(name_buffer))) {
                    skeleton_creator_name = std::string(name_buffer);
                }
                
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
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Image: %s (%dx%d)", path.filename().string().c_str(), background_width, background_height);
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
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No background image");
                }
                
                ImGui::SeparatorText("Interactive Editor");
                
                if (ImPlot::BeginPlot("Skeleton Creator", ImVec2(-1, 400), ImPlotFlags_Equal))
                {
                    ImPlot::SetupAxes("", "");
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImGuiCond_Always);

                    ImPlot::SetupAxisTicks(ImAxis_X1, nullptr, 0);
                    ImPlot::SetupAxisTicks(ImAxis_Y1, nullptr, 0);
                    
                    if (background_image_selected && background_texture != 0) {
                        ImPlot::PlotImage("##background", (void*)(intptr_t)background_texture, 
                                         ImPlotPoint(0, 0), ImPlotPoint(1, 1));
                    }
                    
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::GetIO().KeyCtrl) {
                        if (selected_node_for_edge < 0) {
                            ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
                            creator_nodes.emplace_back(mouse_pos.x, mouse_pos.y, next_node_id++);
                        }
                    }
                    
                    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                        selected_node_for_edge = -1;
                    }
                    
                    for (const auto& edge : creator_edges) {
                        const SkeletonCreatorNode* node1 = nullptr;
                        const SkeletonCreatorNode* node2 = nullptr;
                        
                        for (const auto& node : creator_nodes) {
                            if (node.id == edge.node1_id) node1 = &node;
                            if (node.id == edge.node2_id) node2 = &node;
                        }
                        
                        if (node1 && node2) {
                            double xs[2] = {node1->position.x, node2->position.x};
                            double ys[2] = {node1->position.y, node2->position.y};
                            ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), 2.0f);
                            ImPlot::PlotLine("##edge", xs, ys, 2);
                        }
                    }
                    
                    for (size_t i = 0; i < creator_nodes.size(); i++) {
                        auto& node = creator_nodes[i];
                        
                        bool clicked = false, hovered = false, held = false;
                        ImVec4 node_color = node.color;
                        
                        if (selected_node_for_edge == node.id) {
                            node_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); 
                        }
                        
                        bool modified = ImPlot::DragPoint(node.id, &node.position.x, &node.position.y, 
                                                        node_color, 8.0f, ImPlotDragToolFlags_None, 
                                                        &clicked, &hovered, &held);
                        
                        if (hovered) {
                            ImPlot::PlotText(node.name.c_str(), node.position.x, node.position.y + 0.03);
                            
                            if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                                int node_id_to_delete = node.id;
                                
                                creator_nodes.erase(creator_nodes.begin() + i);
                                
                                creator_edges.erase(std::remove_if(creator_edges.begin(), creator_edges.end(),
                                    [node_id_to_delete](const SkeletonCreatorEdge& edge) {
                                        return edge.node1_id == node_id_to_delete || edge.node2_id == node_id_to_delete;
                                    }), creator_edges.end());
                                
                                if (selected_node_for_edge == node_id_to_delete) {
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
                                for (const auto& existing_edge : creator_edges) {
                                    if ((existing_edge.node1_id == selected_node_for_edge && existing_edge.node2_id == node.id) ||
                                        (existing_edge.node1_id == node.id && existing_edge.node2_id == selected_node_for_edge)) {
                                        edge_exists = true;
                                        break;
                                    }
                                }
                                
                                if (!edge_exists) {
                                    creator_edges.emplace_back(selected_node_for_edge, node.id);
                                } else {
                                    // Delete the existing edge
                                    creator_edges.erase(std::remove_if(creator_edges.begin(), creator_edges.end(),
                                        [selected_node_for_edge, node_id = node.id](const SkeletonCreatorEdge& edge) {
                                            return (edge.node1_id == selected_node_for_edge && edge.node2_id == node_id) || (edge.node1_id == node_id && edge.node2_id == selected_node_for_edge);
                                        }), creator_edges.end());
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
                    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Selected node for edge creation. Ctrl+Click another node to create edge or remove existing edge, or press ESC to cancel.");
                }
                
                ImGui::SeparatorText("Help");
                ImGui::BulletText("Left-click in the plot area to add a new node");
                ImGui::BulletText("Drag nodes to reposition them");
                ImGui::BulletText("Ctrl+Click a node to select it for edge creation");
                ImGui::BulletText("Ctrl+Click another node to create an edge or remove an existing edge between them");
                ImGui::BulletText("Press ESC to cancel edge creation");
                ImGui::BulletText("Press R while hovering a node to delete it and its edges");
                
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
                    config.path = std::filesystem::current_path().string() + "/skeleton";
                    config.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "LoadSkeletonForEdit", "Load Skeleton",
                        ".json", config);
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
                        for (const auto& node : creator_nodes) {
                            node_names.push_back(node.name);
                            node_positions.push_back({node.position.x, node.position.y});
                        }
                        skeleton_json["node_names"] = node_names;
                        skeleton_json["node_positions"] = node_positions;
                        
                        std::vector<std::vector<int>> edges_array;
                        for (const auto& edge : creator_edges) {
                            int idx1 = -1, idx2 = -1;
                            for (size_t i = 0; i < creator_nodes.size(); i++) {
                                if (creator_nodes[i].id == edge.node1_id) idx1 = (int)i;
                                if (creator_nodes[i].id == edge.node2_id) idx2 = (int)i;
                            }
                            if (idx1 >= 0 && idx2 >= 0) {
                                edges_array.push_back({idx1, idx2});
                            }
                        }
                        skeleton_json["edges"] = edges_array;

                        std::string skeleton_dir = std::filesystem::current_path().string() + "/skeleton";
                        
                        std::string filename;
                        filename = skeleton_dir + "/" + skeleton_creator_name + ".json";
                        
                        std::ofstream file(filename);
                        file << skeleton_json.dump(4);
                        file.close();
                        std::cout << "Skeleton saved to: " << filename << " (with node positions)" << std::endl;
                    }
                }
                
                if (!creator_nodes.empty()) {
                    ImGui::SeparatorText("Nodes");
                    if (ImGui::BeginTable("NodeTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Position", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                        ImGui::TableHeadersRow();
                        
                        for (size_t i = 0; i < creator_nodes.size(); i++)
                        {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%d", creator_nodes[i].id);
                            
                            ImGui::TableSetColumnIndex(1);
                            char node_name_buffer[128];
                            strncpy(node_name_buffer, creator_nodes[i].name.c_str(), sizeof(node_name_buffer));
                            node_name_buffer[sizeof(node_name_buffer) - 1] = '\0';
                            ImGui::PushID(i);
                            if (ImGui::InputText("##name", node_name_buffer, sizeof(node_name_buffer))) {
                                creator_nodes[i].name = std::string(node_name_buffer);
                            }
                            ImGui::PopID();
                            
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("(%.2f, %.2f)", creator_nodes[i].position.x, creator_nodes[i].position.y);
                        }
                        ImGui::EndTable();
                    }
                }
            }
            ImGui::End();
        }

        if (show_error) {
            ImGui::OpenPopup("Error");
            show_error = false; // Reset the flag so it only opens once
        }

        if (ImGui::BeginPopupModal("Error", NULL,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("%s", error_message.c_str());
            ImGui::Separator();

            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
                show_error = false;
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

        if (just_seeked) {
            just_seeked = false;
        } else {
            if (dc_context->decoding_flag && play_video &&
                (to_display_frame_number < (dc_context->total_num_frame - 1))) {
                to_display_frame_number++;

                for (int j = 0; j < scene->num_cams; j++) {
                    scene->display_buffer[j][read_head].available_to_write =
                        true;
                }

                read_head = (read_head + 1) % scene->size_of_buffer;
                slider_frame_number = to_display_frame_number;
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
