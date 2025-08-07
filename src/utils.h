#ifndef RED_UTILS
#define RED_UTILS
#include <iostream>
#include <vector>

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

#endif
