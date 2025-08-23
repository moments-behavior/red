#include "yolo_torch.h"
#include "global.h"
float calculateIoU(const YoloPrediction &a, const YoloPrediction &b) {
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

std::vector<YoloPrediction> applyNMS(std::vector<YoloPrediction> &predictions,
                                     float iou_threshold,
                                     float confidence_threshold) {
    predictions.erase(
        std::remove_if(predictions.begin(), predictions.end(),
                       [confidence_threshold](const YoloPrediction &pred) {
                           return pred.confidence < confidence_threshold;
                       }),
        predictions.end());

    std::vector<YoloPrediction> result;
    
    // Group predictions by class
    std::map<int, std::vector<YoloPrediction>> class_predictions;
    for (const auto &pred : predictions) {
        class_predictions[pred.class_id].push_back(pred);
    }
    
    
    for (auto &[class_id, class_preds] : class_predictions) {
        
        // Sort predictions for this class by confidence
        std::sort(class_preds.begin(), class_preds.end(),
                  [](const YoloPrediction &a, const YoloPrediction &b) {
                      return a.confidence > b.confidence;
                  });

        std::vector<bool> suppressed(class_preds.size(), false);
        int kept_count = 0;
        
        for (size_t i = 0; i < class_preds.size(); ++i) {
            if (suppressed[i])
                continue;

            result.push_back(class_preds[i]);
            kept_count++;

            // Suppress overlapping detections in the same class
            for (size_t j = i + 1; j < class_preds.size(); ++j) {
                if (suppressed[j])
                    continue;

                float iou = calculateIoU(class_preds[i], class_preds[j]);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

std::vector<YoloPrediction> runYoloInference(const std::string &model_path,
                                             unsigned char *frame_data,
                                             int width, int height) {
    std::vector<YoloPrediction> predictions;

    try {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file does not exist: " << model_path
                      << std::endl;
            throw std::runtime_error("Model file not found");
        }

        if (std::filesystem::file_size(model_path) == 0) {
            std::cerr << "Model file is empty: " << model_path << std::endl;
            throw std::runtime_error("Model file is empty");
        }

        std::cout << "Loading YOLO model from: " << model_path << std::endl;
        std::cout << "Model file size: "
                  << std::filesystem::file_size(model_path) << " bytes"
                  << std::endl;

        torch::jit::script::Module module;
        try {
            module = torch::jit::load(model_path);
        } catch (const c10::Error &e) {
            std::cerr
                << "Failed to load TorchScript model. This might be caused by:"
                << std::endl;
            std::cerr
                << "1. The file is not a valid TorchScript model (.pt format)"
                << std::endl;
            std::cerr
                << "2. The model was saved with a different PyTorch version"
                << std::endl;
            std::cerr << "3. The model file is corrupted" << std::endl;
            std::cerr << "4. The model is in .pth format (state dict) instead "
                         "of .pt (TorchScript)"
                      << std::endl;
            std::cerr << "PyTorch error: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load TorchScript model");
        }

        module.eval();

        if (!frame_data) {
            std::cerr << "Frame data is null" << std::endl;
            throw std::runtime_error("Invalid frame data");
        }

        std::cout << "Converting frame data to tensor (size: " << width << "x"
                  << height << ")" << std::endl;

        torch::Tensor tensor =
            torch::from_blob(frame_data, {height, width, 4}, torch::kUInt8);
        tensor = tensor.slice(2, 0, 3);
        tensor = tensor.permute({2, 0, 1});
        tensor = tensor.to(torch::kFloat) / 255.0;

        tensor = torch::nn::functional::interpolate(
            tensor.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions()
                                     .size(std::vector<int64_t>{640, 640})
                                     .mode(torch::kBilinear)
                                     .align_corners(false));

        std::cout << "Running inference..." << std::endl;

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);

        torch::Tensor output;
        try {
            output = module.forward(inputs).toTensor();
        } catch (const std::exception &e) {
            std::cerr << "Inference failed. The model might expect different "
                         "input format."
                      << std::endl;
            std::cerr << "Inference error: " << e.what() << std::endl;
            throw std::runtime_error("Inference failed");
        }

        std::cout << "Inference completed. Output shape: [";
        for (int i = 0; i < output.dim(); ++i) {
            std::cout << output.size(i);
            if (i < output.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        if (output.dim() == 3) {
            output = output.permute({0, 2, 1});
        }
        output = output.squeeze(0);

        if (output.dim() != 2) {
            std::cerr << "Unexpected output tensor dimensions: " << output.dim()
                      << std::endl;
            return predictions;
        }

        float conf_threshold = 0.25f;
        float scale_x = static_cast<float>(width) / 640.0f;
        float scale_y = static_cast<float>(height) / 640.0f;

        auto output_accessor = output.accessor<float, 2>();
        int num_detections = output.size(0);
        int output_features = output.size(1);

        std::cout << "Processing " << num_detections << " detections with "
                  << output_features << " features per detection" << std::endl;

        for (int i = 0; i < num_detections; ++i) {
            float confidence;
            int class_id;

            if (output_features == 6) {
                float class0_score = output_accessor[i][4];
                float class1_score = output_accessor[i][5];
                
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
                
            } else if (output_features == 5) {
                confidence = output_accessor[i][4];
                class_id = 0;
                
            } else if (output_features > 6) {
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
                std::cerr << "Unexpected output format with " << output_features
                          << " features" << std::endl;
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

        std::cout << "Found " << predictions.size()
                  << " raw detections before NMS" << std::endl;

        predictions =
            applyNMS(predictions, iou_threshold, confidence_threshold);

        std::cout << "YOLO inference completed. Found " << predictions.size()
                  << " detections after NMS." << std::endl;

        for (size_t i = 0; i < predictions.size(); ++i) {
            const auto &pred = predictions[i];
            std::cout << "Final Detection " << i << ": center=(" << pred.x
                      << "," << pred.y << ") size=(" << pred.w << "," << pred.h
                      << ") conf=" << pred.confidence
                      << " class=" << pred.class_id << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "PyTorch inference error: " << e.what() << std::endl;
    }

    return predictions;
}

bool frameHasYoloDetections(int frame_num,
                            const std::map<u32, KeyPoints *> &keypoints_map,
                            const SkeletonContext *skeleton) {
    if (!skeleton || !skeleton->has_bbox)
        return false;

    auto it = keypoints_map.find(frame_num);
    if (it == keypoints_map.end() || !it->second)
        return false;

    // Check if any camera has bounding boxes with confidence < 1.0 (indicating
    // YOLO detections)
    for (int cam_id = 0;
         cam_id < MAX_VIEWS && cam_id < it->second->bbox2d_list.size();
         cam_id++) {
        const auto &bbox_list = it->second->bbox2d_list[cam_id];
        for (const auto &bbox : bbox_list) {
            if (bbox.confidence < 1.0f && bbox.state == RectTwoPoints) {
                return true;
            }
        }
    }
    return false;
}
