#pragma once
#include <opencv2/dnn.hpp>
#include <condition_variable>
#include "color_conversion_cpu.h"
#include <iostream>


struct yolo_param{
    float conf_threshold;
    float nma_threshold;
    int size_class_list;
    yolo_param(): conf_threshold(0.25), nma_threshold(0.4) {}
};

