#ifndef RED_YOLO
#define RED_YOLO
#include <opencv2/dnn.hpp>
#include <iostream>
#include <condition_variable>
#include "simd_acc.h"
#include <iomanip>
#include <thread> 
#include "global.h"
#include <string>
#include <fstream>
#include "yolov8_pose.h"
#include "yolov8_det.h"

struct yolo_sync {
    bool new_frame;
    bool detect_ready; 
};

struct yolo_param {
    float conf_threshold;
    float nma_threshold;
    int size_class_list;
    std::vector<std::string> class_names;
    yolo_param(): conf_threshold(0.50), nma_threshold(0.45) {}
};


struct yolo_results {
    std::vector<cv::Rect> yolo_boxes;
    std::vector<std::string> yolo_labels;
    std::vector<int> yolo_classid;
}; 

void yolo_process(std::string onnx_file, yolo_param* post_setting, int camera_id);
void read_yolo_labels(std::string label_names_file, yolo_param* post_setting);
void yolo_process_v8pose(std::string engine_file, int camera_id);
void yolo_process_trt(std::string engine_file, int camera_id);
#endif
