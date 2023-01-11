#pragma once
#include <opencv2/dnn.hpp>
#include <condition_variable>
#include "color_conversion_cpu.h"
#include <iostream>


struct yolo_param{
    float conf_threshold;
    float nma_threshold;
    int size_class_list;
    std::vector<string> class_names;
    yolo_param(): conf_threshold(0.50), nma_threshold(0.45) {}
};

