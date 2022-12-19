#include "yolo_detection.h"

void track_ball_fast(cv::dnn::Net* net, unsigned char* frame_img, unsigned char* yolo_input_frame, yolo_param yolo_setting, std::vector<cv::Rect>* yolo_box)
{
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    // optimize cpu color conversion 
    rgba_to_rgb_cpu(frame_img, yolo_input_frame, 3208, 2200);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    
    cv::Mat image = cv::Mat(3208 * 2200 * 3, 1, CV_8U, yolo_input_frame).reshape(3, 2200);
    cv::Mat blob;

    double x_factor = image.cols / 640.0;
    double y_factor = image.rows / 640.0;

    cv::dnn::blobFromImage(image, blob, 1./255.,  cv::Size(640, 640),  cv::Scalar(), true, false);

    net->setInput(blob);

    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    // net.forward(outs, getOutputsNames(net));
    net->forward(outs, net->getUnconnectedOutLayersNames());
    

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    const int rows = 25200;
    float *data = (float *)outs[0].data;
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
            if (confidence > yolo_setting.conf_threshold)
            {
             float *classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, yolo_setting.size_class_list, CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > yolo_setting.conf_threshold && class_id.x==32)
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
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    std::vector<cv::Rect> final_boxes;
    cv::dnn::NMSBoxes(boxes, confidences, yolo_setting.conf_threshold, yolo_setting.nma_threshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        final_boxes.push_back(box);
        //cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), 
          //             cv::Scalar(255, 178, 50), 3);
    }
    *yolo_box = final_boxes;
}



void yolo_detect_thread(std::mutex& g_mutex, std::condition_variable& g_cv, bool* g_ready, cv::dnn::Net* net, unsigned char* frame_img, unsigned char* yolo_input_frame, yolo_param yolo_setting, std::vector<cv::Rect>* yolo_box, bool* stop_flag)
{

    while (!(*stop_flag))
    {
        std::unique_lock<std::mutex> ul(g_mutex);
        g_cv.wait(ul, [&]() {return *g_ready;});

        // track_ball(net[cam_idx], image, cam_idx);
        track_ball_fast(net, frame_img, yolo_input_frame, yolo_setting, yolo_box);

        *g_ready = false;
        ul.unlock();
        g_cv.notify_one();
    }
}