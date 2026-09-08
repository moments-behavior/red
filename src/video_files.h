#pragma once
// Which video containers red accepts, and how a camera's file is found.
//
// ".mp4" used to be written out at each of the eleven sites that needed a
// camera's video -- the open dialogs, camera discovery, the reload path, and
// every exporter that pulls frames from the source. It is written once here
// instead, so the answer to "which containers?" lives in one place.
//
// That answer is currently mp4 alone. The decoder does not require it --
// FFmpegDemuxer hands the path to avformat_open_input, which opens AVI, MOV
// and MKV just as well -- but the tailcycle-dataset format specifies
// <cam>.mp4, so a project in another container could not round-trip through
// it, and there is no call for the others yet. Adding one is appending to the
// list below; every site follows it.
//
// Resolution is by search rather than by an extension stored on the project,
// so nothing needs migrating if that list grows. When a camera has more than
// one matching file, the order below decides.

#include <filesystem>
#include <string>
#include <cctype>

inline const char *const kVideoExts[] = {".mp4"};

// For an ImGuiFileDialog filter string.
inline const char *video_ext_filter() { return ".mp4"; }

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
