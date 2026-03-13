#pragma once
#include "render.h"
#include "skeleton.h"
#include <chrono>
#include <iostream>
#include <vector>
struct ProjectContext {
    std::string root_dir;
    std::vector<std::string> input_file_names;
    std::vector<std::string> camera_names;
};

struct PlaybackState {
    int pause_selected = 0;
    bool slider_just_changed = false;
    bool play_video = false;
    int to_display_frame_number = 0;
    int read_head = 0;
    bool just_seeked = false;
    bool pause_seeked = false;
    int slider_frame_number = 0;
    double accumulated_play_time = 0.0;
    std::chrono::steady_clock::time_point last_play_time_start =
        std::chrono::steady_clock::now();
    int last_frame_num_playspeed = 0;
    std::chrono::steady_clock::time_point last_wall_time_playspeed =
        std::chrono::steady_clock::now();
    bool video_loaded = false;
    bool realtime_playback = true;
    float set_playback_speed = 1.0f;
    double inst_speed = 1.0;
    bool slider_text_editing = false;  // true while user is typing in slider
};

bool string_ends_with(const std::string &str, const std::string &suffix);
std::vector<std::string> string_split(std::string s, std::string delimiter);
std::string format_time(float t_seconds);
void seek_all_cameras(RenderScene *scene, int frame_number, double video_fps,
                      PlaybackState &state, bool seek_accurate);
void prepare_application_folders(std::string &red_data_dir,
                                 std::string &media_dir);
std::string dir_difference(const std::filesystem::path &a,
                           const std::filesystem::path &b);
bool ensure_dir_exists(std::string path_string, std::string *err);
bool ends_with_ci(std::string_view s, std::string_view suffix);
std::string get_home_directory();
