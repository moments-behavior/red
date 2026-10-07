#pragma once
// Which video containers and image formats red accepts, and how a camera's
// file is found.
//
// ".mp4" used to be written out at each of the eleven sites that needed a
// camera's video -- the open dialogs, camera discovery, the reload path, and
// every exporter that pulls frames from the source. It is written once here
// instead, so the answer to "which containers?" lives in one place.
//
// That answer is mp4 and avi, in either case (Cam1.MP4, Cam1.AVI). The
// decoder does not care -- FFmpegDemuxer hands the path to
// avformat_open_input, which opens MOV and MKV just as well -- so adding a
// container is appending to the list below (and to the dialog filter);
// every site follows it. The tailcycle-dataset format still specifies
// <cam>.mp4 for the videos it holds; tailcycle export extracts frames, so an
// avi project exports the same as an mp4 one.
//
// Resolution is by search rather than by an extension stored on the project,
// so nothing needs migrating if that list grows. When a camera has more than
// one matching file, the order below decides.

#include <filesystem>
#include <string>
#include <vector>
#include <cctype>

// Lowercase; matching is case-insensitive.
inline const char *const kVideoExts[] = {".mp4", ".avi"};

// For an ImGuiFileDialog filter string. The dialog matches case-sensitively,
// so both spellings are listed. The first entry is the default: a collection
// ("Name{...}") showing every format in either case; then one entry per
// format, again covering both spellings. A comma in an entry's name splits
// it into two entries unless it sits inside parentheses, hence ".mp4 / .MP4".
inline const char *video_ext_filter() {
    return "Videos (.mp4, .avi, any case){.mp4,.MP4,.avi,.AVI},"
           ".mp4 / .MP4{.mp4,.MP4},.avi / .AVI{.avi,.AVI}";
}

inline std::string lower_ext(const std::string &ext) {
    std::string e;
    for (char c : ext) e += (char)std::tolower((unsigned char)c);
    return e;
}

inline bool is_video_ext(const std::string &ext) {
    const std::string e = lower_ext(ext);
    for (const char *v : kVideoExts)
        if (e == v) return true;
    return false;
}

// ── Images ──
// What load_image_rgba (decoder.cpp) can decode: turbojpeg for JPEG, else
// stb_image, which reads PNG and BMP but not TIFF -- so TIFF is not listed,
// though it used to be offered and then failed to load.
inline const char *const kImageExts[] = {".jpg", ".jpeg", ".png", ".bmp"};

// Same shape as video_ext_filter().
inline const char *image_ext_filter() {
    return "Images (.jpg, .jpeg, .png, .bmp, any case){.jpg,.JPG,.jpeg,.JPEG,"
           ".png,.PNG,.bmp,.BMP},"
           ".jpg / .JPG{.jpg,.JPG},.jpeg / .JPEG{.jpeg,.JPEG},"
           ".png / .PNG{.png,.PNG},.bmp / .BMP{.bmp,.BMP}";
}

inline bool is_image_ext(const std::string &ext) {
    const std::string e = lower_ext(ext);
    for (const char *v : kImageExts)
        if (e == v) return true;
    return false;
}

// The video file for `cam` under `media_folder`, or "" if there is none.
// The extension may be in any case: <cam>.mp4, <cam>.MP4 and <cam>.Mp4 all
// resolve, which matters on case-sensitive filesystems (Linux).
inline std::string find_camera_video(const std::string &media_folder,
                                     const std::string &cam) {
    namespace fs = std::filesystem;
    std::error_code ec;
    // The common spelling first: one stat per container, no directory scan.
    for (const char *ext : kVideoExts) {
        fs::path p = fs::path(media_folder) / (cam + ext);
        if (fs::exists(p, ec)) return p.string();
    }
    // Otherwise any case, still in kVideoExts order.
    std::vector<fs::path> hits;
    for (fs::directory_iterator it(media_folder, ec), end; !ec && it != end;
         it.increment(ec))
        if (it->path().stem().string() == cam) hits.push_back(it->path());
    for (const char *ext : kVideoExts)
        for (const auto &p : hits)
            if (lower_ext(p.extension().string()) == ext) return p.string();
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
