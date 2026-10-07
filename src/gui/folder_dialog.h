#pragma once
// Folder pickers that show what is in the folder: the files it is being
// chosen for (videos, calibration .yaml, timestamp .csv) are listed, so the
// user sees they are in the right place, and a line under the list says what
// red found there. ImGuiFileDialog's own folder mode lists folders only, so
// these open as file dialogs filtered to those files; OK takes the folder
// being browsed (GetCurrentPath), whatever file is clicked.
#include "media_loader.h"
#include <ImGuiFileDialog.h>
#include <filesystem>
#include <functional>
#include <map>
#include <string>

// What the line under the list says for a folder -- e.g. "4 cameras
// (videos)" -- or "" when nothing useful is there.
using FolderDescriber = std::function<std::string(const std::string &dir)>;

struct FolderDialogKind {
    const char *filter;          // the files to list, ImGuiFileDialog syntax
    FolderDescriber describe;
};

inline std::map<std::string, FolderDialogKind> &folder_dialog_kinds() {
    static std::map<std::string, FolderDialogKind> kinds;
    return kinds;
}

inline void open_folder_dialog(const char *key, const char *title,
                               FolderDialogKind kind, IGFD::FileDialogConfig cfg) {
    cfg.countSelectionMax = 1;
    cfg.fileName.clear();
    cfg.flags |= ImGuiFileDialogFlags_OptionalFileName |
                 ImGuiFileDialogFlags_ReadOnlyFileNameField |
                 ImGuiFileDialogFlags_HideColumnType;
    const char *filter = kind.filter;
    folder_dialog_kinds()[key] = std::move(kind);
    ImGuiFileDialog::Instance()->OpenDialog(key, title, filter, cfg);
}

// Display() for a dialog opened with open_folder_dialog.
inline bool display_folder_dialog(const char *key) {
    auto it = folder_dialog_kinds().find(key);
    if (it == folder_dialog_kinds().end())
        return ImGuiFileDialog::Instance()->Display(
            key, ImGuiWindowFlags_NoCollapse, ImVec2(680, 440));

    IgfdHooks &hooks = igfd_hooks();
    hooks.name_label = "Selected:";
    const FolderDescriber &describe = it->second.describe;
    hooks.under_name = [&describe, key] {
        // Folders can be large (thousands of frames): look once per folder.
        static std::string last_dir, last_text;
        const std::string dir = ImGuiFileDialog::Instance()->GetCurrentPath();
        if (std::string(key) + '|' + dir != last_dir) {
            last_dir = std::string(key) + '|' + dir;
            try {
                last_text = describe ? describe(dir) : std::string();
            } catch (const std::exception &) {   // unreadable folder
                last_text.clear();
            }
        }
        if (last_text.empty())
            ImGui::TextDisabled("OK uses the folder you are in.");
        else
            ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f),
                               "%s -- OK uses this folder.", last_text.c_str());
        return true;
    };
    const bool done = ImGuiFileDialog::Instance()->Display(
        key, ImGuiWindowFlags_NoCollapse, ImVec2(680, 440));
    hooks = IgfdHooks{};
    return done;
}

// Counts the files in dir whose extension (any case) is one of exts.
inline int count_files_with_ext(const std::string &dir,
                                std::initializer_list<const char *> exts) {
    namespace fs = std::filesystem;
    int n = 0;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        std::string ext = it->path().extension().string();
        for (char &c : ext) c = (char)std::tolower((unsigned char)c);
        for (const char *e : exts)
            if (ext == e) { ++n; break; }
    }
    return n;
}

inline std::string plural(int n, const char *one, const char *many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

// The three folders red asks for.
inline FolderDialogKind media_folder_kind() {
    return {"Videos and images{.mp4,.MP4,.avi,.AVI,.jpg,.JPG,.jpeg,.JPEG,"
            ".png,.PNG,.bmp,.BMP}",
            [](const std::string &dir) -> std::string {
                MediaKind kind;
                const auto cams = discover_media_cameras(dir, &kind);
                if (cams.empty()) return "";
                const std::string n = plural((int)cams.size(), "camera", "cameras");
                switch (kind) {
                case MediaKind::Video: return n + " (videos)";
                case MediaKind::ImagesPerCamera: return n + " (a folder of images each)";
                default: return n + " (images)";
                }
            }};
}

inline FolderDialogKind calibration_folder_kind() {
    return {"Calibration{.yaml,.YAML,.yml,.csv,.CSV}",
            [](const std::string &dir) -> std::string {
                const int yaml = count_files_with_ext(dir, {".yaml", ".yml"});
                const int csv = count_files_with_ext(dir, {".csv"});
                if (yaml) return plural(yaml, "camera .yaml", "camera .yaml files");
                if (csv) return plural(csv, ".csv file", ".csv files");
                return "";
            }};
}

inline FolderDialogKind timestamps_folder_kind() {
    return {"Timestamps{.csv,.CSV,.json}",
            [](const std::string &dir) -> std::string {
                const int csv = count_files_with_ext(dir, {".csv"});
                const bool plan =
                    std::filesystem::exists(std::filesystem::path(dir) / "sync_plan.json");
                if (plan) return csv ? "sync_plan.json and " + plural(csv, ".csv", ".csv files")
                                     : "sync_plan.json";
                return csv ? plural(csv, "timestamp .csv", "timestamp .csv files") : "";
            }};
}
