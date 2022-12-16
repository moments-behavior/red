#pragma once
#include <opencv2/dnn.hpp>

struct yolo_param{
    float conf_threshold;
    float nma_threshold;
    yolo_param(): conf_threshold(0.5), nma_threshold(0.4) {}
};

