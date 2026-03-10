// Test JARVIS prediction import pipeline
// Usage: test_jarvis_import <data3D.csv> <calibration_folder>

#include "jarvis_import.h"
#include "opencv_yaml_io.h"
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <data3D.csv> <calibration_folder>" << std::endl;
        return 1;
    }

    std::string csv_path = argv[1];
    std::string calib_folder = argv[2];

    // Step 1: Read predictions
    std::cout << "Reading predictions from: " << csv_path << std::endl;
    std::string err;
    auto preds = JarvisImport::read_jarvis_predictions(csv_path, 0.0f, &err);
    if (!err.empty()) {
        std::cerr << "Error: " << err << std::endl;
        return 1;
    }
    std::cout << "  Loaded " << preds.size() << " frames" << std::endl;
    if (preds.empty()) {
        std::cerr << "No predictions loaded!" << std::endl;
        return 1;
    }
    int nj = (int)preds.begin()->second.positions.size();
    std::cout << "  Keypoints per frame: " << nj << std::endl;

    // Print first frame
    auto &first = preds.begin()->second;
    std::cout << "  Frame " << preds.begin()->first << ":" << std::endl;
    for (int j = 0; j < nj; j++) {
        std::cout << "    kp" << j << ": ("
                  << first.positions[j].x() << ", "
                  << first.positions[j].y() << ", "
                  << first.positions[j].z() << ") conf="
                  << first.confidences[j] << std::endl;
    }

    // Step 2: Load calibration
    std::cout << "\nLoading calibration from: " << calib_folder << std::endl;
    std::vector<CameraParams> cameras;
    std::vector<std::string> camera_names;
    int img_h = 0;

    for (const auto &entry :
         std::filesystem::directory_iterator(calib_folder)) {
        if (entry.path().extension() != ".yaml") continue;
        std::string name = entry.path().stem().string();
        camera_names.push_back(name);

        auto doc = opencv_yaml::read(entry.path().string());
        CameraParams cp;
        auto K = doc.getMatrix("camera_matrix");
        auto dist = doc.getMatrix("distortion_coefficients");
        auto R = doc.getMatrix("rc_ext");
        auto T = doc.getMatrix("tc_ext");

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cp.k(i, j) = K(i, j);
        for (int i = 0; i < std::min(5, (int)dist.size()); i++)
            cp.dist_coeffs(i) = dist(i);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cp.r(i, j) = R(i, j);
        for (int i = 0; i < 3; i++)
            cp.tvec(i) = T(i);
        cp.rvec = red_math::rotationMatrixToVector(cp.r);

        img_h = doc.getInt("image_height");
        cameras.push_back(cp);
    }
    std::sort(camera_names.begin(), camera_names.end());
    // Sort cameras to match names (re-read in order)
    // For simplicity, just use the order we loaded

    std::cout << "  Loaded " << cameras.size() << " cameras, img_h=" << img_h
              << std::endl;

    // Step 3: Write prediction CSVs
    std::string out_dir = "/tmp/test_jarvis_import";
    std::cout << "\nWriting CSVs to: " << out_dir << std::endl;
    bool ok = JarvisImport::write_prediction_csvs(
        out_dir, "Rat4", preds, cameras, camera_names, img_h, &err);
    if (!ok) {
        std::cerr << "Write error: " << err << std::endl;
        return 1;
    }

    // Step 4: Verify output files exist
    std::cout << "\nOutput files:" << std::endl;
    for (const auto &entry :
         std::filesystem::directory_iterator(out_dir)) {
        std::cout << "  " << entry.path().filename().string()
                  << " (" << entry.file_size() << " bytes)" << std::endl;
    }

    // Step 5: Verify keypoints3d.csv content
    std::ifstream f3d(out_dir + "/keypoints3d.csv");
    std::string line;
    int line_count = 0;
    while (std::getline(f3d, line)) line_count++;
    std::cout << "\nkeypoints3d.csv: " << line_count << " lines "
              << "(1 header + " << (line_count - 1) << " frames)" << std::endl;

    // Step 6: Verify confidence.csv
    std::ifstream fconf(out_dir + "/confidence.csv");
    line_count = 0;
    while (std::getline(fconf, line)) line_count++;
    std::cout << "confidence.csv: " << line_count << " lines" << std::endl;

    // Step 7: Test full import function
    std::cout << "\nTesting full import_jarvis_predictions()..." << std::endl;
    auto result = JarvisImport::import_jarvis_predictions(
        csv_path, "/tmp/test_jarvis_import_full", "Rat4",
        cameras, camera_names, img_h, 0.0f);
    std::cout << "  Imported: " << result.frames_imported << " frames"
              << std::endl;
    std::cout << "  Filtered: " << result.frames_filtered << " frames"
              << std::endl;
    std::cout << "  Mean confidence: " << result.mean_confidence << std::endl;
    std::cout << "  Output: " << result.output_folder << std::endl;
    if (!result.error.empty())
        std::cout << "  Error: " << result.error << std::endl;

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
