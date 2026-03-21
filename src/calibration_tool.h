#pragma once
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace CalibrationTool {

struct CharucoSetup {
    int w = 5, h = 5;
    float square_side_length = 60.0f;
    float marker_side_length = 37.5f;
    int dictionary = 0;
};

struct CalibConfig {
    std::string img_path;
    std::string vid_path;
    std::vector<std::string> cam_ordered;
    CharucoSetup charuco_setup;
    int first_view = 0;
    std::vector<int> second_view_order;
    std::string global_registration_video;
    std::vector<std::string> world_coordinate_imgs;
    std::map<std::string, std::vector<std::vector<float>>> gt_pts;
    Eigen::Matrix3d world_frame_rotation = Eigen::Matrix3d::Identity(); // post-Procrustes rotation
    nlohmann::json ba_config;
    // Separate global registration media (set from CalibProject, not config.json)
    std::string global_reg_media_folder;
    std::string global_reg_media_type; // "videos" or "images"
};

// Parse config.json into CalibConfig. Returns true on success.
inline bool parse_config(const std::string &config_path,
                         CalibConfig &config, std::string &error) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        error = "Cannot open config file: " + config_path;
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception &e) {
        error = std::string("JSON parse error: ") + e.what();
        return false;
    }

    config.img_path = j.value("img_path", std::string{});
    config.vid_path = j.value("vid_path", std::string{});
    config.global_registration_video =
        j.value("global_registration_video", std::string{});

    if (j.contains("cam_ordered") && j["cam_ordered"].is_array()) {
        config.cam_ordered = j["cam_ordered"].get<std::vector<std::string>>();
    }
    if (config.cam_ordered.empty()) {
        error = "config.json has no cam_ordered array";
        return false;
    }

    if (j.contains("charuco_setup") && j["charuco_setup"].is_object()) {
        auto &cs = j["charuco_setup"];
        config.charuco_setup.w = cs.value("w", 5);
        config.charuco_setup.h = cs.value("h", 5);
        config.charuco_setup.square_side_length =
            cs.value("square_side_length", 60.0f);
        config.charuco_setup.marker_side_length =
            cs.value("marker_side_length", 37.5f);
        config.charuco_setup.dictionary = cs.value("dictionary", 0);
    }

    config.first_view = j.value("first_view", 0);

    if (j.contains("second_view_order") && j["second_view_order"].is_array()) {
        config.second_view_order =
            j["second_view_order"].get<std::vector<int>>();
    }

    if (j.contains("world_coordinate_imgs") &&
        j["world_coordinate_imgs"].is_array()) {
        config.world_coordinate_imgs =
            j["world_coordinate_imgs"].get<std::vector<std::string>>();
    }

    if (j.contains("gt_pts") && j["gt_pts"].is_object()) {
        for (auto &[key, val] : j["gt_pts"].items()) {
            std::vector<std::vector<float>> pts;
            for (auto &pt : val) {
                pts.push_back(pt.get<std::vector<float>>());
            }
            config.gt_pts[key] = std::move(pts);
        }
    }

    // Optional post-Procrustes rotation to match a specific world frame convention.
    // Specified as a 3x3 row-major array: [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]]
    // Applied after Procrustes alignment: R_cam_new = R_cam * world_frame_rotation^T
    // Example (multiview_calib convention): [[0,1,0],[1,0,0],[0,0,-1]]
    if (j.contains("world_frame_rotation") && j["world_frame_rotation"].is_array()) {
        auto &wfr = j["world_frame_rotation"];
        if (wfr.size() == 3) {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    config.world_frame_rotation(r, c) = wfr[r][c].get<double>();
            // Validate: must be a proper rotation (orthogonal, det=+1)
            double det = config.world_frame_rotation.determinant();
            Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            double orth_err = (config.world_frame_rotation * config.world_frame_rotation.transpose() - I).norm();
            if (std::abs(det - 1.0) > 0.01 || orth_err > 0.01) {
                error = "world_frame_rotation must be a proper rotation matrix (det=" +
                    std::to_string(det) + ", orthogonality error=" + std::to_string(orth_err) + ")";
                return false;
            }
        }
    }

    if (j.contains("ba_config")) {
        config.ba_config = j["ba_config"];
    }

    return true;
}

// Scan img_path for images matching cam_ordered serials.
// Returns map: {"serial_N.jpg" -> "/full/path/serial_N.jpg"}
// This is the format load_images() expects.
inline std::map<std::string, std::string>
discover_images(const CalibConfig &config) {
    namespace fs = std::filesystem;
    std::map<std::string, std::string> result;

    if (config.img_path.empty() || !fs::is_directory(config.img_path))
        return result;

    // Build set of known serials for fast lookup
    std::set<std::string> serials(config.cam_ordered.begin(),
                                  config.cam_ordered.end());

    for (const auto &entry : fs::directory_iterator(config.img_path)) {
        if (!entry.is_regular_file())
            continue;
        std::string filename = entry.path().filename().string();

        // Check extension
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".tiff" && ext != ".tif")
            continue;

        // Extract serial: everything before first underscore
        auto underscore = filename.find('_');
        if (underscore == std::string::npos)
            continue;
        std::string serial = filename.substr(0, underscore);

        if (serials.count(serial)) {
            result[filename] = entry.path().string();
        }
    }

    return result;
}

// Scan aruco_video_folder for videos matching cam_ordered serials.
// Returns map: {"serial" -> "/full/path/CamSerial.mp4"}
// Matches Cam{serial}.mp4/.avi/.mov/.mkv pattern.
inline std::map<std::string, std::string>
discover_aruco_videos(const std::string &video_folder,
                      const std::vector<std::string> &cam_ordered) {
    namespace fs = std::filesystem;
    std::map<std::string, std::string> result;

    if (video_folder.empty() || !fs::is_directory(video_folder))
        return result;

    const std::vector<std::string> exts = {".mp4", ".avi", ".mov", ".mkv"};

    // Build stem → path map
    std::map<std::string, std::string> stem_to_path;
    for (const auto &entry : fs::directory_iterator(video_folder)) {
        if (!entry.is_regular_file())
            continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(exts.begin(), exts.end(), ext) == exts.end())
            continue;
        stem_to_path[entry.path().stem().string()] = entry.path().string();
    }

    for (const auto &serial : cam_ordered) {
        std::string stem = "Cam" + serial;
        auto it = stem_to_path.find(stem);
        if (it == stem_to_path.end())
            it = stem_to_path.find(serial); // fallback: exact match
        if (it != stem_to_path.end())
            result[serial] = it->second;
    }
    return result;
}

// Result of scanning a folder for aruco media (images or videos).
struct ArucoMediaInfo {
    std::string type;                     // "images", "videos", or "" (nothing found)
    std::vector<std::string> serials;     // camera serials found (sorted)
    int file_count = 0;                   // total files matched
    std::string description;              // human-readable summary
};

// Scan a folder and auto-detect whether it contains aruco videos or images.
// Checks for videos first (Cam*.{mp4,avi,mov,mkv}), then images ({serial}_*.{jpg,png,...}).
inline ArucoMediaInfo detect_aruco_media(const std::string &folder) {
    namespace fs = std::filesystem;
    ArucoMediaInfo info;

    if (folder.empty() || !fs::is_directory(folder))
        return info;

    const std::vector<std::string> vid_exts = {".mp4", ".avi", ".mov", ".mkv"};
    const std::vector<std::string> img_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"};

    // Pass 1: look for videos (Cam*.{mp4,avi,mov,mkv})
    std::set<std::string> vid_serials;
    int vid_count = 0;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(vid_exts.begin(), vid_exts.end(), ext) == vid_exts.end())
            continue;
        std::string stem = entry.path().stem().string();
        if (stem.size() > 3 && stem.substr(0, 3) == "Cam") {
            vid_serials.insert(stem.substr(3));
            vid_count++;
        }
    }

    if (!vid_serials.empty()) {
        info.type = "videos";
        info.serials.assign(vid_serials.begin(), vid_serials.end());
        info.file_count = vid_count;
        info.description = "Found " + std::to_string(vid_count) + " videos (" +
                           std::to_string(vid_serials.size()) + " cameras)";
        return info;
    }

    // Pass 2: look for images ({serial}_*.{jpg,png,...})
    std::set<std::string> img_serials;
    int img_count = 0;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(img_exts.begin(), img_exts.end(), ext) == img_exts.end())
            continue;
        std::string filename = entry.path().filename().string();
        auto underscore = filename.find('_');
        if (underscore != std::string::npos) {
            img_serials.insert(filename.substr(0, underscore));
            img_count++;
        }
    }

    if (!img_serials.empty()) {
        info.type = "images";
        info.serials.assign(img_serials.begin(), img_serials.end());
        info.file_count = img_count;
        info.description = "Found " + std::to_string(img_count) + " images (" +
                           std::to_string(img_serials.size()) + " cameras)";
        return info;
    }

    return info; // type remains empty — nothing found
}

// Count images per camera from a discovered file map.
inline int count_images_per_camera(
    const std::map<std::string, std::string> &files,
    const std::string &serial) {
    int count = 0;
    std::string prefix = serial + "_";
    for (const auto &[filename, _] : files) {
        if (filename.substr(0, prefix.size()) == prefix)
            count++;
    }
    return count;
}

// Get the default output folder for calibration (parent of config file + /calibration).
inline std::string
default_output_folder(const std::string &config_path) {
    auto parent = std::filesystem::path(config_path).parent_path();
    return (parent / "calibration").string();
}

// ── Calibration Project (.redproj) ──────────────────────────────────────────

enum class CameraModel { Projective = 0, Telecentric = 1 };

struct CalibProject {
    std::string project_name;      // e.g. "MyCalibration"
    std::string project_path;      // folder containing .redproj
    std::string project_root_path; // parent of project_path

    // Unified aruco calibration (images or videos — auto-detected)
    std::string config_file;           // path to config.json
    std::string aruco_media_folder;    // folder with aruco images or videos
    std::string aruco_media_type;      // "images" or "videos" (auto-detected)

    // Global registration media (optional — separate from calibration media)
    std::string global_reg_media_folder; // folder with global reg images or videos
    std::string global_reg_media_type;   // "images" or "videos" (auto-detected)

    // Legacy fields (kept for backward compat with old .redproj files)
    std::string img_path;              // old: image directory (from config)
    std::string aruco_video_folder;    // old: folder with aruco board .mp4 videos

    // Laser refinement (optional — empty until laser phase)
    std::string media_folder;      // folder with laser .mp4 videos
    std::string calibration_folder; // folder with CamXXXX.yaml (input)
    std::vector<std::string> camera_names; // validated camera serials

    // Camera model
    CameraModel camera_model = CameraModel::Projective;

    // Telecentric calibration (DLT with known 3D landmarks)
    std::string landmark_labels_folder; // folder with Cam<serial>.csv 2D labels
    std::string landmarks_3d_file;      // CSV with 3D landmark coordinates

    // Output
    std::string aruco_output_folder;     // unified aruco calibration output
    std::string output_folder;           // legacy (migrated)
    std::string image_output_folder;     // legacy: aruco image YAML output
    std::string video_output_folder;     // legacy: aruco video YAML output
    std::string image_experimental_folder; // legacy: experimental image YAML output
    std::string video_experimental_folder; // legacy: experimental video YAML output
    std::string laser_output_folder;     // laser YAML output
    std::string tele_output_folder;      // telecentric DLT output

    // Last aruco calibration run parameters (persisted for reproducibility)
    int last_aruco_start_frame = 0;
    int last_aruco_stop_frame = 0;
    int last_aruco_frame_step = 0;       // 0 = not set (images or not yet run)
    int last_aruco_total_video_frames = 0;
    int last_aruco_cameras_used = 0;
    double last_aruco_mean_reproj = 0;

    // Last DLT calibration result (persisted for reopening)
    int dlt_method = -1;                  // -1 = not run, 0/1/2 = Linear/k1/k1k2
    double dlt_mean_rmse = 0;
    std::vector<double> dlt_per_camera_rmse;

    // Mode helpers
    bool has_aruco() const { return !config_file.empty() || !aruco_media_folder.empty(); }
    bool has_laser_input() const { return !calibration_folder.empty() && !media_folder.empty(); }
    bool is_telecentric() const { return camera_model == CameraModel::Telecentric; }
    bool aruco_is_video() const { return aruco_media_type == "videos"; }
    bool aruco_is_image() const { return aruco_media_type == "images"; }
    std::string effective_labels_folder() const {
        if (!landmark_labels_folder.empty()) return landmark_labels_folder;
        return project_path.empty() ? std::string{} : (project_path + "/red_data");
    }

    // Effective aruco media folder (unified field, with legacy fallback)
    std::string effective_aruco_media() const {
        if (!aruco_media_folder.empty()) return aruco_media_folder;
        // Legacy fallback: prefer video folder, then image path
        if (!aruco_video_folder.empty()) return aruco_video_folder;
        return img_path;
    }

    bool has_telecentric_input() const {
        return is_telecentric() && !media_folder.empty() &&
               !effective_labels_folder().empty() && !landmarks_3d_file.empty();
    }
};

inline void to_json(nlohmann::json &j, const CalibProject &p) {
    j = nlohmann::json{{"type", "calibration"},
                       {"project_name", p.project_name},
                       {"project_path", p.project_path},
                       {"project_root_path", p.project_root_path},
                       {"config_file", p.config_file},
                       {"aruco_media_folder", p.aruco_media_folder},
                       {"aruco_media_type", p.aruco_media_type},
                       {"global_reg_media_folder", p.global_reg_media_folder},
                       {"global_reg_media_type", p.global_reg_media_type},
                       // Legacy fields (kept for backward compat)
                       {"img_path", p.img_path},
                       {"aruco_video_folder", p.aruco_video_folder},
                       {"media_folder", p.media_folder},
                       {"calibration_folder", p.calibration_folder},
                       {"camera_names", p.camera_names},
                       {"camera_model", static_cast<int>(p.camera_model)},
                       {"landmark_labels_folder", p.landmark_labels_folder},
                       {"landmarks_3d_file", p.landmarks_3d_file},
                       {"aruco_output_folder", p.aruco_output_folder},
                       // Legacy output fields (kept for backward compat)
                       {"image_output_folder", p.image_output_folder},
                       {"video_output_folder", p.video_output_folder},
                       {"image_experimental_folder", p.image_experimental_folder},
                       {"video_experimental_folder", p.video_experimental_folder},
                       {"laser_output_folder", p.laser_output_folder},
                       {"tele_output_folder", p.tele_output_folder},
                       {"last_aruco_start_frame", p.last_aruco_start_frame},
                       {"last_aruco_stop_frame", p.last_aruco_stop_frame},
                       {"last_aruco_frame_step", p.last_aruco_frame_step},
                       {"last_aruco_total_video_frames", p.last_aruco_total_video_frames},
                       {"last_aruco_cameras_used", p.last_aruco_cameras_used},
                       {"last_aruco_mean_reproj", p.last_aruco_mean_reproj},
                       {"dlt_method", p.dlt_method},
                       {"dlt_mean_rmse", p.dlt_mean_rmse},
                       {"dlt_per_camera_rmse", p.dlt_per_camera_rmse}};
}

inline void from_json(const nlohmann::json &j, CalibProject &p) {
    p.project_name = j.value("project_name", std::string{});
    p.project_path = j.value("project_path", std::string{});
    p.project_root_path = j.value("project_root_path", std::string{});
    p.config_file = j.value("config_file", std::string{});

    // New unified aruco media fields
    p.aruco_media_folder = j.value("aruco_media_folder", std::string{});
    p.aruco_media_type = j.value("aruco_media_type", std::string{});
    p.global_reg_media_folder = j.value("global_reg_media_folder", std::string{});
    p.global_reg_media_type = j.value("global_reg_media_type", std::string{});

    // Legacy fields (read for backward compat)
    p.img_path = j.value("img_path", std::string{});
    p.aruco_video_folder = j.value("aruco_video_folder", std::string{});

    // Backward compat: migrate old fields → unified aruco_media_folder
    if (p.aruco_media_folder.empty()) {
        if (!p.aruco_video_folder.empty()) {
            p.aruco_media_folder = p.aruco_video_folder;
            p.aruco_media_type = "videos";
        } else if (!p.img_path.empty()) {
            p.aruco_media_folder = p.img_path;
            p.aruco_media_type = "images";
        }
    }

    p.media_folder = j.value("media_folder", std::string{});
    p.calibration_folder = j.value("calibration_folder", std::string{});
    p.camera_names = j.value("camera_names", std::vector<std::string>{});
    p.camera_model = static_cast<CameraModel>(j.value("camera_model", 0));
    p.landmark_labels_folder = j.value("landmark_labels_folder", std::string{});
    p.landmarks_3d_file = j.value("landmarks_3d_file", std::string{});

    // New unified output folder
    p.aruco_output_folder = j.value("aruco_output_folder", std::string{});

    // Legacy output fields
    p.image_output_folder = j.value("image_output_folder", std::string{});
    p.video_output_folder = j.value("video_output_folder", std::string{});
    p.image_experimental_folder = j.value("image_experimental_folder", std::string{});
    p.video_experimental_folder = j.value("video_experimental_folder", std::string{});
    // Backward compat: migrate old output_folder → image_output_folder
    if (p.image_output_folder.empty() && j.contains("output_folder"))
        p.image_output_folder = j.value("output_folder", std::string{});
    // Backward compat: migrate legacy output folders → unified aruco_output_folder
    if (p.aruco_output_folder.empty()) {
        if (!p.image_experimental_folder.empty())
            p.aruco_output_folder = p.image_experimental_folder;
        else if (!p.video_experimental_folder.empty())
            p.aruco_output_folder = p.video_experimental_folder;
        else if (!p.image_output_folder.empty())
            p.aruco_output_folder = p.image_output_folder;
        else if (!p.video_output_folder.empty())
            p.aruco_output_folder = p.video_output_folder;
    }

    p.laser_output_folder = j.value("laser_output_folder", std::string{});
    p.tele_output_folder = j.value("tele_output_folder", std::string{});
    p.last_aruco_start_frame = j.value("last_aruco_start_frame", 0);
    p.last_aruco_stop_frame = j.value("last_aruco_stop_frame", 0);
    p.last_aruco_frame_step = j.value("last_aruco_frame_step", 0);
    p.last_aruco_total_video_frames = j.value("last_aruco_total_video_frames", 0);
    p.last_aruco_cameras_used = j.value("last_aruco_cameras_used", 0);
    p.last_aruco_mean_reproj = j.value("last_aruco_mean_reproj", 0.0);
    p.dlt_method = j.value("dlt_method", -1);
    p.dlt_mean_rmse = j.value("dlt_mean_rmse", 0.0);
    p.dlt_per_camera_rmse = j.value("dlt_per_camera_rmse", std::vector<double>{});
}

inline bool save_project(const CalibProject &p,
                          const std::string &file,
                          std::string *err = nullptr) {
    try {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(fs::path(file).parent_path(), ec);
        if (ec) {
            if (err) *err = ec.message();
            return false;
        }
        std::ofstream ofs(file, std::ios::binary);
        if (!ofs) {
            if (err) *err = "Cannot open: " + file;
            return false;
        }
        nlohmann::json j = p;
        ofs << j.dump(2);
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    }
}

inline bool load_project(CalibProject *out,
                          const std::string &file,
                          std::string *err = nullptr) {
    if (!out) {
        if (err) *err = "Output pointer is null";
        return false;
    }
    try {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) {
            if (err) *err = "Cannot open: " + file;
            return false;
        }
        nlohmann::json j;
        ifs >> j;
        // Accept both old aruco-only projects and new unified projects
        std::string type = j.value("type", std::string{});
        if (!type.empty() && type != "calibration" && type != "laser_calibration") {
            if (err) *err = "Unknown project type: " + type;
            return false;
        }
        *out = j.get<CalibProject>();
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    }
}

// Derive camera serial numbers from YAML filenames in a folder.
// Looks for CamXXXX.yaml files and returns the "XXXX" part sorted.
inline std::vector<std::string>
derive_camera_names_from_yaml(const std::string &yaml_folder) {
    namespace fs = std::filesystem;
    std::vector<std::string> names;
    if (yaml_folder.empty() || !fs::is_directory(yaml_folder))
        return names;
    for (const auto &entry : fs::directory_iterator(yaml_folder)) {
        if (!entry.is_regular_file()) continue;
        std::string stem = entry.path().stem().string();
        std::string ext = entry.path().extension().string();
        if (ext == ".yaml" && stem.size() > 3 && stem.substr(0, 3) == "Cam") {
            names.push_back(stem.substr(3));
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

// Derive camera serial numbers from video filenames in a folder.
// Looks for Cam*.{mp4,avi,mov,mkv} and returns the serial part sorted.
inline std::vector<std::string>
derive_camera_names_from_videos(const std::string &video_folder) {
    namespace fs = std::filesystem;
    std::vector<std::string> names;
    if (video_folder.empty() || !fs::is_directory(video_folder))
        return names;
    const std::vector<std::string> exts = {".mp4", ".avi", ".mov", ".mkv"};
    for (const auto &entry : fs::directory_iterator(video_folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;
        std::string stem = entry.path().stem().string();
        if (stem.size() > 3 && stem.substr(0, 3) == "Cam") {
            names.push_back(stem.substr(3));
        }
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

// Count non-empty data lines in a CSV file (skips first line as header).
inline int count_csv_data_lines(const std::string &filepath) {
    namespace fs = std::filesystem;
    if (filepath.empty() || !fs::is_regular_file(filepath))
        return 0;
    std::ifstream f(filepath);
    std::string line;
    int count = 0;
    bool first = true;
    while (std::getline(f, line)) {
        if (first) { first = false; continue; }
        if (!line.empty()) count++;
    }
    return count;
}

// Validate which cameras have matching Cam<serial>.csv files in a labels folder.
// Returns a map: serial -> number of data lines (landmarks) in that CSV.
inline std::map<std::string, int>
validate_telecentric_labels(const std::string &labels_folder,
                            const std::vector<std::string> &camera_names) {
    namespace fs = std::filesystem;
    std::map<std::string, int> result;
    if (labels_folder.empty() || !fs::is_directory(labels_folder))
        return result;
    for (const auto &serial : camera_names) {
        std::string csv_path =
            (fs::path(labels_folder) / ("Cam" + serial + ".csv")).string();
        int n = count_csv_data_lines(csv_path);
        if (n > 0) result[serial] = n;
    }
    return result;
}

// Count number of 3D landmark points in a CSV file (rows after header).
inline int count_landmarks_3d(const std::string &filepath) {
    return count_csv_data_lines(filepath);
}

} // namespace CalibrationTool
