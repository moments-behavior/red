// Example of streaming images computed on a GPU with CUDA to a PBO in
// OpenGL, rendered with DearImgui. Structure follows basic OpenGL
// example using the ImGui GLFW backend.

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
#include "shader_m.h"

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"
#include "Logger.h"
#include "gl_helper.h"
#include "decoder.h"
#include "IconsForkAwesome.h"
#include "implot.h"
#include "LabelManager.h"
#include "label_gui.h"
#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <imfilebrowser.h>
#include "yolo_detection.h"
#include "Simd/SimdLib.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


#define MAX_VIEWS 4

cv::dnn::Net nets[MAX_VIEWS];
std::vector<std::mutex> g_mutexes(MAX_VIEWS);
std::vector<std::condition_variable> g_cvs(MAX_VIEWS);
std::vector<bool> g_ready(MAX_VIEWS);
std::vector<std::vector<cv::Rect>> yolo_boxes(MAX_VIEWS);
unsigned char* yolo_input_frame[MAX_VIEWS];
unsigned char* yolo_input_frame_rgba[MAX_VIEWS];

cv::dnn::Net batch_net;
vector<cv::Mat> batch_outs(MAX_VIEWS);
vector<cv::Mat> batch_frame(MAX_VIEWS);
bool g_detection_ready{false};
int preprocess_counter(0);
std::mutex g_preprocess_mutex;
std::condition_variable g_preprocess_cv;

static void draw_cv_contours(std::vector<cv::Rect> boxes)
{
    int n = boxes.size();
    if (n == 0)
    {
        return;
    }

    double x[n];
    double y[n];

    for (int i=0; i<boxes.size(); i++)
    {
        //cout << "Camera #" << cam_num << " Box info " << boxes[i].x << "," << boxes[i].y << "," << boxes[i].width << "," << boxes[i].height << endl;
        
        x[i] = (double)boxes[i].x + (boxes[i].width/2);
        y[i] = (double)2200 - ((double)boxes[i].y + (boxes[i].height/2));

    }

    ImVec4 fill_color = ImVec4(0.8, 0.0, 0.8, 0.3);
    ImVec4 outline_color = ImVec4(1.0, 1.0, 1.0, 1.0);

    ImPlot::SetNextMarkerStyle(ImPlotMarker_Square,20, fill_color, 3, outline_color);
    ImPlot::PlotScatter("now", &x[0], &y[0], n);
}


void track_ball_cpu(yolo_param post_setting, int cam_idx, int no_frame_proc)
{
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    int length_image = 3208 * 2200;
    SimdBgraToBgr(yolo_input_frame_rgba[cam_idx], 3208, 2200, 3208 * 4, yolo_input_frame[cam_idx], 3208 * 3);

    double x_factor = 3208.0 / 640.0;
    double y_factor = 2200.0 / 640.0;


    batch_frame[cam_idx] = cv::Mat(3208 * 2200 * 3, 1, CV_8U, yolo_input_frame[cam_idx]).reshape(3, 2200);
    
    {
        std::lock_guard lk(g_preprocess_mutex);
        preprocess_counter++;
        // std::cout << cam_idx << " ready, preprocess_couter now " << preprocess_counter << std::endl;
    }
    g_preprocess_cv.notify_all();
    {
        std::unique_lock lk(g_preprocess_mutex);
        g_preprocess_cv.wait(lk, [&](){return g_detection_ready;});
        preprocess_counter--;
        // std::cout << cam_idx << " back to thread, preprocess_couter now is" << preprocess_counter << std::endl;
        if(preprocess_counter==0){g_detection_ready=false;}
        lk.unlock();
    }
    
    float *data = (float *)batch_outs[0].data;


    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    const int rows = 25200;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence > post_setting.conf_threshold)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, post_setting.size_class_list, CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > post_setting.conf_threshold && class_id.x==32)
            {
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    vector<cv::Rect> final_boxes;
    cv::dnn::NMSBoxes(boxes, confidences, post_setting.conf_threshold, post_setting.nma_threshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        final_boxes.push_back(box);
        //cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), 
          //             cv::Scalar(255, 178, 50), 3);
    }
    yolo_boxes.at(cam_idx) = final_boxes;
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}


void yolo_detection_thread(){
    while(true){        
        std::unique_lock lk(g_preprocess_mutex);
        g_preprocess_cv.wait(lk, [&](){return preprocess_counter>=MAX_VIEWS;});
        
        // std::cout << "detection thread working...\n";
        cv::Mat batch_blob;
        cv::dnn::blobFromImages(batch_frame, batch_blob, 1./255.,  cv::Size(640, 640),  cv::Scalar(), true, false);
        batch_net.setInput(batch_blob);
        batch_net.forward(batch_outs, batch_net.getUnconnectedOutLayersNames());
        g_detection_ready=true;
        // std::cout << "detection resutls ready\n";

        lk.unlock();
        g_preprocess_cv.notify_all();
    }
}


void track_ball_fast(cv::dnn::Net net, yolo_param post_setting, int cam_idx, int no_frame_proc)
{

    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    int length_image = 3208 * 2200;
    // rgba_to_rgb_cpu(yolo_input_frame_rgba[cam_idx], yolo_input_frame[cam_idx], length_image);
    
    SimdBgraToBgr(yolo_input_frame_rgba[cam_idx], 3208, 2200, 3208 * 4, yolo_input_frame[cam_idx], 3208 * 3);

    // int last_pixel_rgba = (length_image-1) * 4;
    // int last_pixel_rgb = (length_image-1) * 3;

	// printf("rgba image first pixel: %d %d %d %d \n", yolo_input_frame_rgba[cam_idx][0], yolo_input_frame_rgba[cam_idx][1], yolo_input_frame_rgba[cam_idx][2], yolo_input_frame_rgba[cam_idx][3]);
	// printf("rgba image last pixel: %d %d %d %d \n", yolo_input_frame_rgba[cam_idx][last_pixel_rgba], yolo_input_frame_rgba[cam_idx][last_pixel_rgba+1], yolo_input_frame_rgba[cam_idx][last_pixel_rgba+2], yolo_input_frame_rgba[cam_idx][last_pixel_rgba+3]);

    // printf("rgb image first pixel: %d %d %d \n", yolo_input_frame[cam_idx][0], yolo_input_frame[cam_idx][1], yolo_input_frame[cam_idx][2]);
	// printf("rgb image last pixel: %d %d %d \n", yolo_input_frame[cam_idx][last_pixel_rgb], yolo_input_frame[cam_idx][last_pixel_rgb+1], yolo_input_frame[cam_idx][last_pixel_rgb+2]);


    cv::Mat image = cv::Mat(3208 * 2200 * 3, 1, CV_8U, yolo_input_frame[cam_idx]).reshape(3, 2200);

    double x_factor = image.cols / 640.0;
    double y_factor = image.rows / 640.0;

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1./255.,  cv::Size(640, 640),  cv::Scalar(), true, false);
    net.setInput(blob);
    // Runs the forward pass to get output of the output layers
    vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());
    
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    const int rows = 25200;
    float *data = (float *)outs[0].data;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence > post_setting.conf_threshold)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, post_setting.size_class_list, CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > post_setting.conf_threshold && class_id.x==32)
            {
                float cx = data[0];
                float cy = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    vector<cv::Rect> final_boxes;
    cv::dnn::NMSBoxes(boxes, confidences, post_setting.conf_threshold, post_setting.nma_threshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        final_boxes.push_back(box);
        //cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), 
          //             cv::Scalar(255, 178, 50), 3);
    }
    yolo_boxes.at(cam_idx) = final_boxes;
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}



void yolo_thread(int cam_idx, yolo_param post_setting)
{
    std::cout << "yolo thread started" << cam_idx << std::endl;
    int no_frame_proc = 0;
    while (true)
    {
        std::unique_lock<std::mutex> ul(g_mutexes.at(cam_idx));
        g_cvs.at(cam_idx).wait(ul, [&]() {return g_ready.at(cam_idx);});
        // cv::Mat image = cv::Mat(IMG_WIDTH * IMG_HEIGHT * 4, 1, CV_8U, curr_frame_on_host[cam_idx]).reshape(4, IMG_HEIGHT);
        // track_ball(net[cam_idx], image, cam_idx);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // track_ball_cpu(post_setting, cam_idx, no_frame_proc);
        track_ball_fast(nets[cam_idx], post_setting, cam_idx, no_frame_proc);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

        g_ready.at(cam_idx) = false;
        ul.unlock();
        no_frame_proc++;
    }
}




int main(int, char**)
{

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Streamer Example", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL functions with GLEW
    glew_error_callback(glewInit());



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
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load a nice font
    io.Fonts->AddFontFromFileTTF("fonts/Roboto-Regular.ttf", 15.0f);
    // merge in icons from Font Awesome
    static const ImWchar icons_ranges[] = { ICON_MIN_FK, ICON_MAX_16_FK, 0 };
    ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF("fonts/forkawesome-webfont.ttf", 15.0f, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid



    // Create a OpenGL texture identifier
    GLuint image_texture[MAX_VIEWS];
    for(int j=0; j<MAX_VIEWS; j++){
        glGenTextures(1, &image_texture[j]);
        glBindTexture(GL_TEXTURE_2D, image_texture[j]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
    }

    // Our state
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);


    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory);
    file_dialog.SetPwd("/home/jinyao/dev");
    file_dialog.SetTitle("Select work directory");

    
    double result = 0.0;
    unsigned long long num_heads = 0;
    unsigned long long num_tails = 0;
    
    
    int size_pic = 3208 * 2200 * 4 *  sizeof(unsigned char);

    // allocate display buffer
    const int size_of_buffer = 8;

    // right now, allocate more than needed, maybe  switch to vector?, need to think about it  
    PictureBuffer display_buffer[MAX_VIEWS][size_of_buffer];
    for(int j=0; j<MAX_VIEWS; j++){
        for (int i = 0; i < size_of_buffer; i++) {
            display_buffer[j][i].frame = (unsigned char*)malloc(size_pic);
            clear_buffer_with_constant_image(display_buffer[j][i].frame, 3208, 2200);

            display_buffer[j][i].frame_number = 0;
            display_buffer[j][i].available_to_write = true;
        }
    }
    
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;


    int num_cams;
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

    SeekInfo seek_context[MAX_VIEWS];
    for(int j=0; j<MAX_VIEWS; j++){
        seek_context[j].use_seek=false;
        seek_context[j].seek_frame=0;
        seek_context[j].seek_done=false;
    }

    int slider_frame_number = 0;
    bool just_seeked = false;

    bool slider_just_changed = false;
    bool video_loaded = false;
    bool plot_keypoints_flag = false;
    int current_frame_num = 0;

    std::vector<Camera*> cams;
    SkelEnum skelEnum = SkelEnum::Rat10Target2;
    LabelManager *labelMgr = nullptr;
    
    
    // for yolo detection
    bool yolo_detection = false;
    std::vector<std::thread> yolo_threads;
    std::unique_lock<std::mutex> display_thread_locks[4];
    std::vector<string> class_list;

    for(int i=0; i<MAX_VIEWS; i++){
        yolo_input_frame[i] = (unsigned char*)malloc(3208 * 2200 * 3 * sizeof(uint8_t) + 4);
    }


    // Main loop
    while (!glfwWindowShouldClose(window))
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
                            
                            for(int i=0; i<num_cams; i++)
                            {
                                Camera* cam = new Camera(i, i, root_dir, skelEnum);
                                cams.push_back(cam);
                            }
                            labelMgr = new LabelManager(cams);
                            plot_keypoints_flag = true;};
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("Detection")){
                    
                    if (ImGui::MenuItem("YOLOv5")) { 
                        // std::string yolov5 = root_dir + "/yolo_models/yolov5s_batch4.onnx";
                        std::cout << "Load trained yolo models..." << std::endl;
                        std::string yolov5 = root_dir + "/yolo_models/yolov5s.onnx";
                        
                        // for(int i=0; i<num_cams; i++){
                        //     cv::dnn::Net net = cv::dnn::readNet(yolov5);
                        //     net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                        //     net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                        //     nets.push_back(net); 
                        // }

                        for (int i=0; i<num_cams; i++) {
                            nets[i] = cv::dnn::readNet(yolov5);
                            nets[i].setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                            nets[i].setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                        }

                        // batch_net = cv::dnn::readNet(yolov5);
                        // batch_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                        // batch_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                        
                        
                        yolo_param yolo_setting = yolo_param();
                        std::ifstream ifs("/home/jinyao/tracking/darknet/data/coco.names");
                        std::string line;
                        while (getline(ifs, line))
                        {
                            class_list.push_back(line);
                        }
                        yolo_setting.size_class_list = class_list.size();

                        // // start thread
                        for(int i=0; i<num_cams; i++){
                            yolo_threads.push_back(std::thread(&yolo_thread, i, yolo_setting));
                        }

                        // yolo_threads.push_back(std::thread(&yolo_detection_thread));
                        yolo_detection=true;
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
                input_file_names.push_back(entry.path());
            }
                
            std::sort(input_file_names.begin(), input_file_names.end());
            num_cams = input_file_names.size();

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

                // sync yolo detection 
                if(yolo_detection){
                    display_thread_locks[j] = std::unique_lock<std::mutex> (g_mutexes[j]);
                    yolo_input_frame_rgba[j]  = display_buffer[j][read_head].frame;
                    g_ready[j] = true;
                    display_thread_locks[j].unlock();
                    g_cvs[j].notify_one();
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
                            for(int j=0; j<num_cams; j++){
                                bind_texture(&image_texture[j]);
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[j][select_corr_head].frame);
                                unbind_texture();
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

                        for(int j=0; j<num_cams; j++){
                            bind_texture(&image_texture[j]);
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[j][select_corr_head].frame);
                            unbind_texture();
                        
                        }
                    }
                };
            }
            ImGui::Text("Frame number selected: %d", display_buffer[0][select_corr_head].frame_number);
            
            select_corr_head = (selected + read_head) % size_of_buffer;
            current_frame_num = display_buffer[0][select_corr_head].frame_number;
            
            ImGui::End();
        }


        if (toggle_play_status && play_video) {
            play_video = false;
            toggle_play_status = false;
        }

        if (plot_keypoints_flag){
            set_active_skel3D(labelMgr, current_frame_num);
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

                    // labeling 
                    if (plot_keypoints_flag){
                        set_active_skel2D(cams[j], current_frame_num);
                        plot_keypoints(labelMgr, cams[j], current_frame_num, draw_id_ptr);
                        
                    }


                    if (yolo_detection){
                        draw_cv_contours(yolo_boxes.at(j));
                    }


                    if (plot_keypoints_flag)
                    {
                        labeling_one_view(cams[j], current_frame_num);
                        keypoint_button(cams[j], j, current_frame_num, cams[j]->frameDataMapIter, labelMgr, draw_id_ptr);
                    }
    
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


            if (plot_keypoints_flag)
            {
            

                ImGui::Begin("Labeling Tool");

                for (int i=0; i<labelMgr->nCams; i++)
                {
                    for (int j=0; j<labelMgr->nNodes; j++)
                    {
                        if (j > 0) ImGui::SameLine();

                        ImGui::PushID(j);

                        if (labelMgr->isLabeled[i][j])
                        {
                            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(j / (float)cams[i]->frameData->nNodes, 0.6f, 0.6f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(j / (float)cams[i]->frameData->nNodes, 0.7f, 0.7f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(j / (float)cams[i]->frameData->nNodes, 0.8f, 0.8f));
                        }
                        else
                        {
                            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.2f, 0.2f, 0.2f));
                        }
                        
                        ImGui::Button(cams[i]->frameData->nodeNames[j].c_str());
                        ImGui::PopStyleColor(3);
                        ImGui::PopID();
                    }
                }

                

                static bool triangulate = false;
                ImGui::Checkbox("triangulate", &triangulate);
                if (triangulate)
                {
                    reprojection(labelMgr, cams, current_frame_num);
                }

                if (ImGui::Button("Reproject") || ImGui::IsKeyPressed(ImGuiKey_S, false))
                {
                    reprojection(labelMgr, cams, current_frame_num);
                }

                if (ImGui::IsKeyPressed(ImGuiKey_X, false))
                {
                    reprojection_overweight_activeKP(labelMgr, cams, current_frame_num);
                }

                if (ImGui::Button("Save Labeled Data"))
                {
                    for (int i=0; i<labelMgr->nCams; i++)
                    {
                        cams[i]->SaveSkelMap();
                    }
                    
                    labelMgr->SaveWorldKeyPoints();
                }

                if (ImGui::Button("Load Camera Skels"))
                {
                    labelMgr->LoadCameraSkels();
                    cout << "loaded 2D skeletons from each camera results directory" << endl;
                    labelMgr->LoadWorldSkels();
                    cout << "loaded 3d skeletons from world results directory" << endl;

                }

                ImGui::NewLine();
                

                // ImGui::Text("A -> Previous Image");
                // ImGui::Text("D -> Next Image");
                // ImGui::Text("Q -> ActiveKeypoint++ (while hovering image)");
                // ImGui::Text("E -> ActiveKeypoint-- (while hovering image)");
                // ImGui::Text("T -> ActiveKeypoint set to last node");
                // ImGui::Text("W -> Drop ActiveKeypoint (while hovering at desired image point)");
                // ImGui::Text("S -> Reproject eligible keypoints");
                ImGui::End();
            }




        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
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

        glfwSwapBuffers(window);
 
        
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

    glfwDestroyWindow(window);
    glfwTerminate();

    *stop_flag = true;
    // wait for threads to join
    for (auto& t : decoder_threads)
        t.join();

    for (auto& t : yolo_threads)
        t.join();

    return 0;
}
