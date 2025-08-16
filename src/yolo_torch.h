#pragma once
#include "skeleton.h"
#include "types.h"
#include <torch/script.h>
#include <torch/torch.h>

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

    YoloBBox()
        : x_min(0), y_min(0), x_max(0), y_max(0), confidence(0.0f),
          class_id(-1), is_valid(false) {}

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

float calculateIoU(const YoloPrediction &a, const YoloPrediction &b);
std::vector<YoloPrediction> applyNMS(std::vector<YoloPrediction> &predictions,
                                     float iou_threshold,
                                     float confidence_threshold);
std::vector<YoloPrediction> runYoloInference(const std::string &model_path,
                                             unsigned char *frame_data,
                                             int width, int height);
bool frameHasYoloDetections(int frame_num,
                            const std::map<u32, KeyPoints *> &keypoints_map,
                            const SkeletonContext *skeleton);
