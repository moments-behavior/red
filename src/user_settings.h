#pragma once
#include "json.hpp"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

struct UserSettings {
    std::string default_project_root_path;
    std::string default_media_root_path;
};

inline void to_json(nlohmann::json &j, const UserSettings &s) {
    j = nlohmann::json{
        {"default_project_root_path", s.default_project_root_path},
        {"default_media_root_path", s.default_media_root_path}};
}

inline void from_json(const nlohmann::json &j, UserSettings &s) {
    s.default_project_root_path =
        j.value("default_project_root_path", std::string{});
    s.default_media_root_path =
        j.value("default_media_root_path", std::string{});
}

inline std::filesystem::path user_settings_path() {
    return std::filesystem::path(get_home_directory()) /
           ".config/red/user_settings.json";
}

inline UserSettings load_user_settings() {
    UserSettings s;
    std::filesystem::path p = user_settings_path();
    if (!std::filesystem::exists(p))
        return s;
    try {
        std::ifstream f(p);
        nlohmann::json j;
        f >> j;
        s = j.get<UserSettings>();
    } catch (const std::exception &e) {
        std::cerr << "Failed to read user_settings.json: " << e.what()
                  << std::endl;
    }
    return s;
}

inline bool save_user_settings(const UserSettings &s) {
    std::filesystem::path p = user_settings_path();
    try {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            std::cerr << "Cannot create settings dir: " << ec.message()
                      << std::endl;
            return false;
        }
        std::ofstream f(p, std::ios::binary);
        if (!f)
            return false;
        nlohmann::json j = s;
        f << j.dump(2);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Failed to write user_settings.json: " << e.what()
                  << std::endl;
        return false;
    }
}
