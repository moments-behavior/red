#include "yolo_detection.h"
#include "simd_acc.h"
#include <iomanip>

void yolo_process(std::string onnx_file, unsigned char* display_frame, yolo_param* post_setting, std::vector<cv::Rect>& yolo_boxes, std::vector<std::string>& yolo_labels, std::vector<int>& yolo_classes)
{
    // load models 
    cv::dnn::Net yolo_net;
    yolo_net = cv::dnn::readNet(onnx_file);
    yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "model loaded" << std::endl;

    int no_frame_proc = 0;
    unsigned char* yolo_input_frame = (unsigned char*)malloc(3208 * 2200 * 3 * sizeof(uint8_t) + 4);
    while (true)
    {
        int length_image = 3208 * 2200;
        rgba_to_bgr_cpu(display_frame, yolo_input_frame, length_image);
        // SimdBgraToBgr(yolo_input_frame_rgba[cam_idx], 3208, 2200, 3208 * 4, yolo_input_frame[cam_idx], 3208 * 3);

        cv::Mat image = cv::Mat(3208 * 2200 * 3, 1, CV_8U, yolo_input_frame).reshape(3, 2200);
        double x_factor = image.cols / 640.0;
        double y_factor = image.rows / 640.0;
        cv::Mat blob;

        cv::dnn::blobFromImage(image, blob, 1./255.,  cv::Size(640, 640),  cv::Scalar(), true, false);
        yolo_net.setInput(blob);
        // Runs the forward pass to get output of the output layers
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

        // Perform non maximum suppression to eliminate redundant overlapping boxes with
        // lower confidences
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
        yolo_boxes = final_boxes;
        yolo_labels = final_labels;
        yolo_classes = final_class_ids;
    }
}

