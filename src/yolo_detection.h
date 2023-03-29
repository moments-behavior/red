#pragma once
#include <opencv2/dnn.hpp>
#include <iostream>


struct yolo_sync {
    bool new_frame;
    bool detect_ready; 
};

struct yolo_param {
    float conf_threshold;
    float nma_threshold;
    int size_class_list;
    std::vector<std::string> class_names;
    yolo_param(): conf_threshold(0.40), nma_threshold(0.45) {}
};


struct yolo_results {
    std::vector<cv::Rect> yolo_boxes;
    std::vector<std::string> yolo_labels;
    std::vector<int> yolo_classid;
}; 

void yolo_process(std::string onnx_file, unsigned char* display_frame, yolo_param* post_setting, std::vector<cv::Rect>& yolo_boxes, std::vector<std::string>& yolo_labels, std::vector<int>& yolo_classes, yolo_sync* sync);