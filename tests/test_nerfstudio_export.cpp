// test_nerfstudio_export.cpp — Test the nerfstudio/3DGS export pipeline.
//
// Loads the mouse_active1 project, runs export_nerfstudio() to generate
// transforms.json and extract frames for a single test frame.
//
// Build: cmake target "test_nerfstudio_export" (no ImGui/Metal needed).
// Run:   ./test_nerfstudio_export [output_dir] [frame_num]

#include "camera.h"
#include "export_formats.h"
#include "opencv_yaml_io.h"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

static const char *PROJECT_PATH =
    "/Users/johnsonr/red_demos/mouse_active1/mouse_active1.redproj";

int main(int argc, char **argv) {
    std::string output_dir = "/tmp/red_nerfstudio_test";
    int frame_num = 39348; // best labeled frame

    if (argc > 1) output_dir = argv[1];
    if (argc > 2) frame_num = std::atoi(argv[2]);

    std::cout << "=== Nerfstudio Export Test ===\n";
    std::cout << "Output: " << output_dir << "\n";
    std::cout << "Frame:  " << frame_num << "\n\n";

    // Load project file
    std::ifstream pf(PROJECT_PATH);
    if (!pf.is_open()) {
        std::cerr << "Cannot open project: " << PROJECT_PATH << "\n";
        return 1;
    }
    nlohmann::json proj = nlohmann::json::parse(pf);

    std::string calib_folder = proj["calibration_folder"];
    std::string media_folder = proj["media_folder"];
    std::vector<std::string> camera_names;
    for (const auto &c : proj["camera_names"])
        camera_names.push_back(c.get<std::string>());

    std::cout << "Calibration: " << calib_folder << "\n";
    std::cout << "Media:       " << media_folder << "\n";
    std::cout << "Cameras:     " << camera_names.size() << "\n\n";

    // Load camera params
    std::vector<CameraParams> camera_params;
    for (const auto &cam : camera_names) {
        CameraParams cp;
        std::string err;
        std::string yaml_path = calib_folder + "/" + cam + ".yaml";
        if (!camera_load_params_from_yaml(yaml_path, cp, err)) {
            std::cerr << "Failed to load " << cam << ": " << err << "\n";
            return 1;
        }
        camera_params.push_back(cp);
        std::cout << "  " << cam << ": fx=" << std::fixed << std::setprecision(1)
                  << cp.k(0, 0) << " fy=" << cp.k(1, 1)
                  << " cx=" << cp.k(0, 2) << " cy=" << cp.k(1, 2) << "\n";
    }
    std::cout << "\n";

    // Print camera positions (world coordinates: C = -R^T * t)
    std::cout << "Camera positions (world, mm):\n";
    for (size_t i = 0; i < camera_names.size(); ++i) {
        Eigen::Vector3d C = -camera_params[i].r.transpose() * camera_params[i].tvec;
        std::cout << "  " << camera_names[i] << ": ["
                  << std::setprecision(1) << C.x() << ", "
                  << C.y() << ", " << C.z() << "]\n";
    }
    std::cout << "\n";

    // Build export config
    ExportFormats::ExportConfig cfg;
    cfg.format = ExportFormats::NERFSTUDIO;
    cfg.calibration_folder = calib_folder;
    cfg.media_folder = media_folder;
    cfg.output_folder = output_dir;
    cfg.camera_names = camera_names;
    cfg.camera_params = camera_params;
    cfg.jpeg_quality = 95;
    cfg.nerfstudio_frames = {frame_num};

    // Run export
    std::string status;
    std::atomic<int> img_counter{0};
    AnnotationMap amap; // empty — we use nerfstudio_frames instead

    std::cout << "Running export_nerfstudio()...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    bool ok = ExportFormats::export_nerfstudio(cfg, amap, &status, &img_counter);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Status: " << status << "\n";
    std::cout << "Images: " << img_counter.load() << "\n";
    std::cout << "Time:   " << std::fixed << std::setprecision(1) << elapsed << "s\n\n";

    if (!ok) {
        std::cerr << "Export failed!\n";
        return 1;
    }

    // Validate transforms.json
    std::string tj_path = output_dir + "/transforms.json";
    std::ifstream tjf(tj_path);
    if (!tjf.is_open()) {
        std::cerr << "Cannot open " << tj_path << "\n";
        return 1;
    }
    auto tj = nlohmann::json::parse(tjf);

    std::cout << "=== Validation ===\n";
    std::cout << "camera_model: " << tj["camera_model"] << "\n";
    std::cout << "frames: " << tj["frames"].size() << "\n\n";

    // Check each frame entry
    int errors = 0;
    for (const auto &frame : tj["frames"]) {
        auto mat = frame["transform_matrix"];
        // Check 4x4 dimensions
        if (mat.size() != 4 || mat[0].size() != 4) {
            std::cerr << "  BAD: " << frame["file_path"] << " matrix not 4x4\n";
            errors++;
            continue;
        }

        // Extract rotation part (3x3) and check orthonormality
        Eigen::Matrix3d R;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R(r, c) = mat[r][c].get<double>();

        Eigen::Matrix3d RtR = R.transpose() * R;
        double ortho_err = (RtR - Eigen::Matrix3d::Identity()).norm();
        if (ortho_err > 1e-4) {
            std::cerr << "  BAD: " << frame["file_path"].get<std::string>()
                      << " R not orthonormal (err=" << ortho_err << ")\n";
            errors++;
        }

        // Check det(R) = +1 (proper rotation, not reflection)
        double det = R.determinant();
        if (std::abs(det - 1.0) > 1e-4) {
            std::cerr << "  BAD: " << frame["file_path"].get<std::string>()
                      << " det(R)=" << det << " (should be +1)\n";
            errors++;
        }

        // Check that file exists
        std::string img_path = output_dir + "/" + frame["file_path"].get<std::string>();
        if (!fs::exists(img_path)) {
            std::cerr << "  MISSING: " << img_path << "\n";
            errors++;
        }
    }

    // Check camera spread
    double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9, z_min = 1e9, z_max = -1e9;
    for (const auto &frame : tj["frames"]) {
        auto mat = frame["transform_matrix"];
        double x = mat[0][3].get<double>();
        double y = mat[1][3].get<double>();
        double z = mat[2][3].get<double>();
        x_min = std::min(x_min, x); x_max = std::max(x_max, x);
        y_min = std::min(y_min, y); y_max = std::max(y_max, y);
        z_min = std::min(z_min, z); z_max = std::max(z_max, z);
    }
    std::cout << "Camera spread (mm):\n";
    std::cout << "  X: [" << std::setprecision(0) << x_min << ", " << x_max
              << "] span=" << (x_max - x_min) << "\n";
    std::cout << "  Y: [" << y_min << ", " << y_max
              << "] span=" << (y_max - y_min) << "\n";
    std::cout << "  Z: [" << z_min << ", " << z_max
              << "] span=" << (z_max - z_min) << "\n\n";

    if (errors == 0) {
        std::cout << "ALL CHECKS PASSED (" << tj["frames"].size()
                  << " frames validated)\n";
    } else {
        std::cerr << errors << " ERRORS found\n";
    }

    // List extracted images
    std::string img_dir = output_dir + "/images";
    if (fs::exists(img_dir)) {
        int count = 0;
        uintmax_t total_size = 0;
        for (const auto &entry : fs::directory_iterator(img_dir)) {
            if (entry.is_regular_file()) {
                count++;
                total_size += entry.file_size();
            }
        }
        std::cout << "\nExtracted " << count << " images ("
                  << (total_size / (1024 * 1024)) << " MB)\n";
    }

    return errors;
}
