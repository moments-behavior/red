#pragma once
// Which video containers red accepts, and how a camera's file is found.
//
// ".mp4" used to be written out at every site that needed a camera's video --
// the open dialogs, camera discovery, reload, and every exporter that pulls
// frames from the source. The decoder never cared: FFmpegDemuxer hands the
// path to avformat_open_input, which opens AVI and MOV and MKV as readily as
// MP4. The restriction was string literals, not capability.
//
// Resolving by search rather than by a stored extension means projects made
// before this keep working with no migration, and a folder can hold a mix.
// The tie-break when a camera has more than one file is the order below.

#include <filesystem>
#include <string>
#include <cctype>

inline const char *const kVideoExts[] = {".mp4", ".avi", ".mov", ".mkv"};

// For an ImGuiFileDialog filter string.
inline const char *video_ext_filter() { return ".mp4,.avi,.mov,.mkv"; }

inline bool is_video_ext(const std::string &ext) {
    std::string e;
    for (char c : ext) e += (char)std::tolower((unsigned char)c);
    for (const char *v : kVideoExts)
        if (e == v) return true;
    return false;
}

// The video file for `cam` under `media_folder`, or "" if there is none.
inline std::string find_camera_video(const std::string &media_folder,
                                     const std::string &cam) {
    namespace fs = std::filesystem;
    for (const char *ext : kVideoExts) {
        fs::path p = fs::path(media_folder) / (cam + ext);
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
    return std::string();
}

// The same, but always returning a path: callers that only want to open it and
// report their own failure should not have to special-case "not found".
inline std::string camera_video_path(const std::string &media_folder,
                                     const std::string &cam) {
    std::string found = find_camera_video(media_folder, cam);
    if (!found.empty()) return found;
    return (std::filesystem::path(media_folder) / (cam + ".mp4")).string();
}
