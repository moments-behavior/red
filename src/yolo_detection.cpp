#include "yolo_detection.h"


void yolo_detect_thread()
{
    while (true)
    {
        std::unique_lock<std::mutex> ul(g_mutexes.at(cam_idx));
        g_cvs.at(cam_idx).wait(ul, [&]() {return g_ready.at(cam_idx);});

        // track_ball(net[cam_idx], image, cam_idx);
        // track_ball_fast(net[cam_idx], cam_idx);

        g_ready.at(cam_idx) = false;
        ul.unlock();
        g_cvs.at(cam_idx).notify_one();
    }
}