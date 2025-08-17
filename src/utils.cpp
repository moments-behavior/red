#include "utils.h"
#include "Logger.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

void prepare_application_folders(const std::string &data_dir,
                                 const std::vector<std::string> &folders,
                                 std::string &media_dir) {
    // create required folders
    for (const auto &folder : folders) {
        std::filesystem::path path = std::filesystem::path(data_dir) / folder;
        if (!std::filesystem::exists(path)) {
            if (std::filesystem::create_directories(path)) {
                std::cout << "Created " << folder << " folder..." << std::endl;
            }
        }
    }

    // default to empty
    media_dir.clear();

    // check for config.json
    std::filesystem::path config_path =
        std::filesystem::path(data_dir) / "config.json";
    if (std::filesystem::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            nlohmann::json j;
            f >> j;

            if (j.contains("media_folder") && j["media_folder"].is_string()) {
                media_dir = j["media_folder"].get<std::string>();
            }
        } catch (const std::exception &e) {
            std::cerr << "Failed to read/parse config.json: " << e.what()
                      << std::endl;
        }
    }
}

void seek_all_cameras(render_scene *scene, int frame_number, double video_fps,
                      PlaybackState &state, bool seek_accurate) {
    // Trigger seek request
    for (int i = 0; i < scene->num_cams; i++) {
        scene->seek_context[i].seek_frame = (uint64_t)frame_number;
        scene->seek_context[i].use_seek = true;
        scene->seek_context[i].seek_accurate = seek_accurate;
    }

    // Wait for seek to complete
    for (int i = 0; i < scene->num_cams; i++) {
        while (!scene->seek_context[i].seek_done) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Reset seek_done flags
    for (int i = 0; i < scene->num_cams; i++) {
        scene->seek_context[i].seek_done = false;
    }

    // Update playback state
    state.to_display_frame_number = scene->seek_context[0].seek_frame;
    state.read_head = 0;
    state.just_seeked = true;
    state.slider_frame_number = state.to_display_frame_number;

    state.accumulated_play_time = frame_number / video_fps;
    state.last_play_time_start = std::chrono::steady_clock::now();
    state.last_frame_num_playspeed = frame_number;
    state.last_wall_time_playspeed = std::chrono::steady_clock::now();
}

std::vector<std::string> string_split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

bool string_ends_with(const std::string &str, const std::string &suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool numerical_compare_substr(const std::string &s1, const std::string &s2) {

    std::size_t s1_start = s1.find("Cam") + 3;
    std::size_t s2_start = s2.find("Cam") + 3;

    std::size_t s1_end = s1.find("mp4");
    std::size_t s2_end = s2.find("mp4");

    std::string s1_substr = s1.substr(s1_start, s1_end - s1_start - 1);
    std::string s2_substr = s2.substr(s2_start, s2_end - s2_start - 1);

    std::cout << s1_substr << " , " << s2_substr << std::endl;

    int s1_int = std::stoi(s1_substr);
    int s2_int = std::stoi(s2_substr);

    std::cout << s1_int << " , " << s2_int << std::endl;

    return s1_int < s2_int;
}

std::string format_time(float t_seconds) {
    int total_seconds = static_cast<int>(t_seconds);
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    float seconds = t_seconds - (hours * 3600 + minutes * 60);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%04.1f", hours, minutes,
                  seconds);
    return std::string(buf);
}
