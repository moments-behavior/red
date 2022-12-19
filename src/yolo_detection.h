#pragma once
#include <opencv2/dnn.hpp>
#include <condition_variable>
#include "color_conversion_cpu.h"
#include <iostream>


struct yolo_param{
    float conf_threshold;
    float nma_threshold;
    int size_class_list;
    yolo_param(): conf_threshold(0.5), nma_threshold(0.4) {}
};

void yolo_detect_thread(std::mutex& g_mutex, std::condition_variable& g_cv, bool* g_ready, cv::dnn::Net* net, unsigned char* frame_img, unsigned char* yolo_input_frame, yolo_param yolo_setting, std::vector<cv::Rect>* yolo_box, bool* stop_flag);
