#ifndef RED_GLOBAL
#define RED_GLOBAL

#define MAX_VIEWS 16
extern std::vector<std::mutex> g_mutexes;
extern std::vector<std::condition_variable> g_cvs;
extern std::vector<bool> g_ready;
extern std::vector<std::vector<cv::Rect>> yolo_boxes;
extern std::vector<std::vector<std::string>> yolo_labels;
extern std::vector<std::vector<int>> yolo_classid;
extern std::vector<unsigned char*> yolo_input_frames_rgba;

#endif
