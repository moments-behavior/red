#include "yolo_detection.h"
#include "kernel.cuh"

void read_yolo_labels(std::string label_names_file, yolo_param* post_setting)
{
    std::ifstream ifs(label_names_file);
    std::vector<std::string> class_list;
    std::string line;    

    while (std::getline(ifs, line))
    {
        class_list.push_back(line);
    }
    post_setting->size_class_list = class_list.size();
    post_setting->class_names = class_list;
}



void yolo_detection(cv::dnn::Net yolo_net, yolo_param* post_setting, unsigned char* yolo_input_frame, int camera_id)
{
    int length_image = 3208 * 2200;
    rgba_to_bgr_cpu(yolo_input_frames_rgba[camera_id], yolo_input_frame, length_image);

    // SimdBgraToBgr(yolo_input_frame_rgba[cam_idx], 3208, 2200, 3208 * 4, yolo_input_frame[cam_idx], 3208 * 3);

    cv::Mat image = cv::Mat(3208 * 2200 * 3, 1, CV_8U, yolo_input_frame).reshape(3, 2200);
    double x_factor = image.cols / 640.0;
    double y_factor = image.rows / 640.0;
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1./255.,  cv::Size(640, 640),  cv::Scalar(), true, false);
    yolo_net.setInput(blob);
    std::vector<cv::Mat> outs;
    yolo_net.forward(outs, yolo_net.getUnconnectedOutLayersNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    const int rows = 25200;
    float *data = (float *)outs[0].data;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence > post_setting->conf_threshold)
        {
            float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, post_setting->size_class_list, CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > post_setting->conf_threshold)
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
                classIds.push_back(class_id.x);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 7;
    }

    std::vector<int> indices;
    std::vector<cv::Rect> final_boxes;
    std::vector<std::string> final_labels;
    std::vector<int> final_class_ids;
    cv::dnn::NMSBoxes(boxes, confidences, post_setting->conf_threshold, post_setting->nma_threshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        final_boxes.push_back(box);
        std::stringstream stream;
        stream << " " << std::fixed << std::setprecision(2) << confidences[idx];
        std::string s = post_setting->class_names[classIds[idx]] + stream.str();
        final_labels.push_back(s);
        final_class_ids.push_back((int)classIds[idx]);
    }

    yolo_boxes.at(camera_id) = final_boxes;
    yolo_labels.at(camera_id) = final_labels;
    yolo_classid.at(camera_id) = final_class_ids;
}


void yolo_process(std::string onnx_file, yolo_param* post_setting, int camera_id)
{
    // load models 
    cv::dnn::Net yolo_net;
    yolo_net = cv::dnn::readNet(onnx_file);
    yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "model loaded" << std::endl;

    int no_frame_proc = 0;
    unsigned char* yolo_input_frame = (unsigned char*)malloc(3208 * 2200 * 3 * sizeof(uint8_t) + 4);
    while (true) {
        std::unique_lock<std::mutex> ul(g_mutexes[camera_id]);
        g_cvs[camera_id].wait(ul, [&]() {return g_ready[camera_id];});
        yolo_detection(yolo_net, post_setting, yolo_input_frame, camera_id);
        g_ready[camera_id] = false;
    }
}

void yolo_process_v8pose(std::string engine_file, int camera_id)
{
    // load models
    unsigned char *d_convert;
    CHECK(cudaMalloc((void **)&d_convert, 3208 * 2200 * 3));
    float *d_points;
    unsigned int *d_skeleton; 
    unsigned int skeleton[8] = {0, 2, 1, 2, 2, 3};

    YOLOv8_pose* yolov8_pose = new YOLOv8_pose(engine_file);
    yolov8_pose->make_pipe(true);

    cudaMalloc((void **)&d_points, sizeof(float) * 8);
    cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
    CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice));

    std::vector<Object> objs;
    float    score_thres = 0.3f;
    float    iou_thres   = 0.5f;
    int      topk        = 1;


    while (true) {
        std::unique_lock<std::mutex> ul(g_mutexes[camera_id]);
        g_cvs[camera_id].wait(ul, [&]() {return g_ready[camera_id];});

        // model detection here, assume frame on gpu 
        rgba2rgb_convert(d_convert, yolo_input_frames_rgba[camera_id], 3208, 2200, 0);
        yolov8_pose->preprocess_gpu(d_convert);
        yolov8_pose->infer();
        yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
        yolov8_pose->copy_keypoints_gpu(d_points, objs);
        gpu_draw_rat_pose(yolo_input_frames_rgba[camera_id], 3208, 2200, d_points, d_skeleton, yolov8_pose->stream);
                
        g_ready[camera_id] = false;
    }
}


void yolo_process_trt(std::string engine_file, int camera_id)
{
    // load models
    unsigned char *d_convert;
    CHECK(cudaMalloc((void **)&d_convert, 3208 * 2200 * 3));

    YOLOv8* yolov8 = new YOLOv8(engine_file);
    yolov8->make_pipe(true);

    std::vector<Object> objs;
    float    score_thres = 0.3f;
    float    iou_thres   = 0.5f;
    int      topk        = 1;

    while (true) {
        std::unique_lock<std::mutex> ul(g_mutexes[camera_id]);
        g_cvs[camera_id].wait(ul, [&]() {return g_ready[camera_id];});
        std::cout << "camera_yolo_thread" <<  camera_id << ": acquire lock" << std::endl; 

        // model detection here, assume frame on gpu 
        rgba2rgb_convert(d_convert, yolo_input_frames_rgba[camera_id], 3208, 2200, yolov8->stream);
        yolov8->preprocess_gpu(d_convert);
        yolov8->infer();
        yolov8->postprocess(objs);
        g_ready[camera_id] = false;
    }
}