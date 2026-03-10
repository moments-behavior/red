/*  convert_labels_v1_to_v2.cpp — Standalone RED label format converter
 *
 *  Converts old RED v1 label CSVs to the new v2 format.
 *
 *  v1 format:
 *    keypoints3d.csv:  SkeletonName\n  frame,idx,x,y,z,idx,x,y,z,...  (1E7 = unlabeled)
 *    <camera>.csv:     SkeletonName\n  frame,idx,x,y,idx,x,y,...      (1E7 = unlabeled)
 *    confidence.csv:   frame_id,kp0,kp1,...  (optional, from JARVIS import)
 *    annotations.json: extended annotations  (optional, copied as-is)
 *
 *  v2 format:
 *    keypoints3d.csv:  #red_csv v2\n  #skeleton Rat4\n  frame,x0,y0,z0,c0,x1,y1,z1,c1,...
 *                      (empty cells = unlabeled, c = confidence, empty = manual)
 *    <camera>.csv:     #red_csv v2\n  #skeleton Rat4\n  frame,x0,y0,c0,s0,x1,y1,c1,s1,...
 *                      (s = source: empty=manual, P=predicted, I=imported)
 *    annotations.json: copied as-is
 *
 *  Compile:  g++ -std=c++17 -o convert_labels convert_labels_v1_to_v2.cpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Sentinel used by v1 format ──
static constexpr double SENTINEL = 1e7;
static constexpr double SENTINEL_THRESH = 9.99e6; // anything above this is sentinel

static bool is_sentinel(double v) { return std::abs(v) > SENTINEL_THRESH; }

// ── Parsed v1 data ──

struct Keypoint3D {
    double x, y, z;
    bool labeled;
};

struct Keypoint2D {
    double x, y;
    bool labeled;
};

struct Frame3D {
    int frame_id;
    std::vector<Keypoint3D> keypoints; // indexed by kp_idx
};

struct Frame2D {
    int frame_id;
    std::vector<Keypoint2D> keypoints; // indexed by kp_idx
};

// confidence[frame_id][kp_idx] = value
using ConfidenceMap = std::map<int, std::vector<float>>;

// ── Timestamp generation ──
static std::string make_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", std::localtime(&t));
    return std::string(buf);
}

// ── CSV tokenizer (handles trailing commas, empty tokens) ──
static std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ','))
        tokens.push_back(token);
    return tokens;
}

// ── Parse v1 3D CSV ──
// Returns skeleton name via out parameter; populates frames vector.
static bool parse_3d_csv(const std::string &path, std::string &skeleton_name,
                          std::vector<Frame3D> &frames, int &max_kp_idx) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Error: Cannot open " << path << "\n";
        return false;
    }

    std::string line;
    // Line 0: skeleton name
    if (!std::getline(fin, line)) {
        std::cerr << "Error: Empty file " << path << "\n";
        return false;
    }
    // Strip trailing whitespace/CR
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
        line.pop_back();
    skeleton_name = line;
    max_kp_idx = -1;

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '\r' || line[0] == '\n') continue;
        auto tokens = split_csv(line);
        if (tokens.size() < 2) continue;

        Frame3D f;
        f.frame_id = std::stoi(tokens[0]);

        // Remaining tokens: groups of (idx, x, y, z)
        size_t i = 1;
        while (i + 3 < tokens.size()) {
            int idx = std::stoi(tokens[i]);
            double x = std::stod(tokens[i + 1]);
            double y = std::stod(tokens[i + 2]);
            double z = std::stod(tokens[i + 3]);
            i += 4;

            // Grow vector if needed
            if (idx >= (int)f.keypoints.size())
                f.keypoints.resize(idx + 1, {0, 0, 0, false});

            bool labeled = !(is_sentinel(x) || is_sentinel(y) || is_sentinel(z));
            f.keypoints[idx] = {x, y, z, labeled};
            if (idx > max_kp_idx) max_kp_idx = idx;
        }

        frames.push_back(std::move(f));
    }
    return true;
}

// ── Parse v1 2D CSV ──
static bool parse_2d_csv(const std::string &path, std::string &skeleton_name,
                          std::vector<Frame2D> &frames, int &max_kp_idx) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Error: Cannot open " << path << "\n";
        return false;
    }

    std::string line;
    if (!std::getline(fin, line)) {
        std::cerr << "Error: Empty file " << path << "\n";
        return false;
    }
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
        line.pop_back();
    skeleton_name = line;
    max_kp_idx = -1;

    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '\r' || line[0] == '\n') continue;
        auto tokens = split_csv(line);
        if (tokens.size() < 2) continue;

        Frame2D f;
        f.frame_id = std::stoi(tokens[0]);

        // Remaining tokens: groups of (idx, x, y)
        size_t i = 1;
        while (i + 2 < tokens.size()) {
            int idx = std::stoi(tokens[i]);
            double x = std::stod(tokens[i + 1]);
            double y = std::stod(tokens[i + 2]);
            i += 3;

            if (idx >= (int)f.keypoints.size())
                f.keypoints.resize(idx + 1, {0, 0, false});

            bool labeled = !(is_sentinel(x) || is_sentinel(y));
            f.keypoints[idx] = {x, y, labeled};
            if (idx > max_kp_idx) max_kp_idx = idx;
        }

        frames.push_back(std::move(f));
    }
    return true;
}

// ── Parse confidence.csv ──
static bool parse_confidence(const std::string &path, ConfidenceMap &conf,
                              int &num_kp_cols) {
    std::ifstream fin(path);
    if (!fin) return false; // optional file, not an error

    std::string line;
    // Header: frame_id,kp0,kp1,...
    if (!std::getline(fin, line)) return false;

    // Count keypoint columns from header
    auto header_tokens = split_csv(line);
    num_kp_cols = (int)header_tokens.size() - 1; // minus frame_id column

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto tokens = split_csv(line);
        if (tokens.size() < 2) continue;

        int frame_id = std::stoi(tokens[0]);
        std::vector<float> vals;
        for (size_t i = 1; i < tokens.size(); ++i) {
            try {
                vals.push_back(std::stof(tokens[i]));
            } catch (...) {
                vals.push_back(0.0f);
            }
        }
        conf[frame_id] = std::move(vals);
    }
    return true;
}

// ── Write v2 3D CSV ──
static void write_3d_v2(const std::string &path, const std::string &skeleton,
                         const std::vector<Frame3D> &frames, int num_kps,
                         const ConfidenceMap &conf) {
    std::ofstream fout(path);
    fout << "#red_csv v2\n";
    fout << "#skeleton " << skeleton << "\n";

    // Header row
    fout << "frame";
    for (int k = 0; k < num_kps; ++k)
        fout << ",x" << k << ",y" << k << ",z" << k << ",c" << k;
    fout << "\n";

    for (const auto &f : frames) {
        fout << f.frame_id;
        for (int k = 0; k < num_kps; ++k) {
            if (k < (int)f.keypoints.size() && f.keypoints[k].labeled) {
                fout << "," << f.keypoints[k].x
                     << "," << f.keypoints[k].y
                     << "," << f.keypoints[k].z;
                // Confidence
                auto cit = conf.find(f.frame_id);
                if (cit != conf.end() && k < (int)cit->second.size()) {
                    fout << "," << cit->second[k];
                } else {
                    fout << ","; // empty = manual
                }
            } else {
                fout << ",,,,"; // empty = unlabeled (x,y,z,c all empty)
            }
        }
        fout << "\n";
    }
}

// ── Write v2 2D CSV ──
static void write_2d_v2(const std::string &path, const std::string &skeleton,
                         const std::vector<Frame2D> &frames, int num_kps,
                         const ConfidenceMap &conf, bool has_confidence) {
    std::ofstream fout(path);
    fout << "#red_csv v2\n";
    fout << "#skeleton " << skeleton << "\n";

    // Header row
    fout << "frame";
    for (int k = 0; k < num_kps; ++k)
        fout << ",x" << k << ",y" << k << ",c" << k << ",s" << k;
    fout << "\n";

    for (const auto &f : frames) {
        fout << f.frame_id;
        for (int k = 0; k < num_kps; ++k) {
            if (k < (int)f.keypoints.size() && f.keypoints[k].labeled) {
                fout << "," << f.keypoints[k].x
                     << "," << f.keypoints[k].y;
                // Confidence + source flag
                auto cit = conf.find(f.frame_id);
                if (has_confidence && cit != conf.end() && k < (int)cit->second.size()) {
                    fout << "," << cit->second[k] << ",P"; // predicted
                } else {
                    fout << ",,"; // empty confidence, empty source = manual
                }
            } else {
                fout << ",,,,"; // empty = unlabeled (x,y,c,s all empty)
            }
        }
        fout << "\n";
    }
}

// ── Usage ──
static void print_usage(const char *argv0) {
    std::cerr << "Usage: " << argv0 << " <input_folder> [--output <output_folder>]\n\n"
              << "  <input_folder>   Path to a v1 label folder (timestamped, e.g.\n"
              << "                   labeled_data/2026_03_10_00_43_32/)\n"
              << "  --output <dir>   Write v2 output into <dir>/<new_timestamp>/\n"
              << "                   Default: sibling of input with _v2 suffix\n\n"
              << "The converter reads v1 CSVs, optional confidence.csv, and\n"
              << "annotations.json, then writes v2 format to a new timestamped folder.\n"
              << "Original files are never modified.\n";
}

// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Check for --help/-h anywhere in arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::string input_folder = argv[1];
    std::string output_root;

    // Parse optional --output flag
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_root = argv[++i];
        }
    }

    // Validate input folder
    if (!fs::is_directory(input_folder)) {
        std::cerr << "Error: Not a directory: " << input_folder << "\n";
        return 1;
    }

    // Determine output root
    if (output_root.empty()) {
        // Place output as sibling: labeled_data/ → labeled_data_v2/
        fs::path inp(input_folder);
        fs::path parent = inp.parent_path();
        output_root = parent.string() + "_v2";
    }

    std::string timestamp = make_timestamp();
    std::string output_folder = output_root + "/" + timestamp;
    fs::create_directories(output_folder);

    std::cout << "=== RED Label Converter v1 -> v2 ===" << std::endl;
    std::cout << "Input:  " << fs::canonical(input_folder).string() << std::endl;
    std::cout << "Output: " << fs::absolute(output_folder).string() << std::endl;
    std::cout << std::endl;

    // ── Step 1: Find all CSV files in input ──
    std::string kp3d_path = input_folder + "/keypoints3d.csv";
    std::string conf_path = input_folder + "/confidence.csv";
    std::string annot_path = input_folder + "/annotations.json";

    // Discover camera CSV files (everything except keypoints3d.csv and confidence.csv)
    std::vector<std::string> camera_names;
    for (const auto &entry : fs::directory_iterator(input_folder)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        if (fname == "keypoints3d.csv" || fname == "confidence.csv" ||
            fname == "annotations.json")
            continue;
        if (entry.path().extension() == ".csv") {
            camera_names.push_back(entry.path().stem().string());
        }
    }
    std::sort(camera_names.begin(), camera_names.end());

    std::cout << "Cameras found: " << camera_names.size() << std::endl;
    for (const auto &cam : camera_names)
        std::cout << "  " << cam << std::endl;

    // ── Step 2: Parse confidence.csv (optional) ──
    ConfidenceMap confidence;
    int conf_kp_cols = 0;
    bool has_confidence = parse_confidence(conf_path, confidence, conf_kp_cols);
    if (has_confidence) {
        std::cout << "\nConfidence file: " << confidence.size()
                  << " frames, " << conf_kp_cols << " keypoint columns" << std::endl;
    } else {
        std::cout << "\nConfidence file: not present (all keypoints marked as manual)"
                  << std::endl;
    }

    // ── Step 3: Convert keypoints3d.csv ──
    int total_3d_frames = 0;
    int total_3d_labeled = 0;
    int total_3d_unlabeled = 0;
    int num_kps_3d = 0;

    if (fs::exists(kp3d_path)) {
        std::string skeleton_name;
        std::vector<Frame3D> frames;
        int max_idx = -1;

        if (!parse_3d_csv(kp3d_path, skeleton_name, frames, max_idx)) {
            std::cerr << "Error: Failed to parse keypoints3d.csv\n";
            return 1;
        }

        num_kps_3d = max_idx + 1;
        total_3d_frames = (int)frames.size();

        // Count labeled/unlabeled keypoints
        for (const auto &f : frames) {
            for (int k = 0; k < num_kps_3d; ++k) {
                if (k < (int)f.keypoints.size() && f.keypoints[k].labeled)
                    ++total_3d_labeled;
                else
                    ++total_3d_unlabeled;
            }
        }

        write_3d_v2(output_folder + "/keypoints3d.csv", skeleton_name,
                     frames, num_kps_3d, confidence);

        std::cout << "\nkeypoints3d.csv:" << std::endl;
        std::cout << "  Skeleton:   " << skeleton_name << std::endl;
        std::cout << "  Frames:     " << total_3d_frames << std::endl;
        std::cout << "  Keypoints:  " << num_kps_3d << " per frame" << std::endl;
        std::cout << "  Labeled:    " << total_3d_labeled << std::endl;
        std::cout << "  Unlabeled:  " << total_3d_unlabeled
                  << " (sentinels removed)" << std::endl;
    } else {
        std::cout << "\nkeypoints3d.csv: not found (single-camera project?)"
                  << std::endl;
    }

    // ── Step 4: Convert per-camera 2D CSVs ──
    int total_2d_frames = 0;
    int total_2d_labeled = 0;
    int total_2d_unlabeled = 0;
    int conf_merged = 0;

    for (const auto &cam : camera_names) {
        std::string cam_in = input_folder + "/" + cam + ".csv";
        std::string cam_out = output_folder + "/" + cam + ".csv";

        std::string skeleton_name;
        std::vector<Frame2D> frames;
        int max_idx = -1;

        if (!parse_2d_csv(cam_in, skeleton_name, frames, max_idx)) {
            std::cerr << "Warning: Failed to parse " << cam << ".csv, skipping\n";
            continue;
        }

        int num_kps_2d = max_idx + 1;

        // Count stats
        for (const auto &f : frames) {
            for (int k = 0; k < num_kps_2d; ++k) {
                if (k < (int)f.keypoints.size() && f.keypoints[k].labeled) {
                    ++total_2d_labeled;
                    // Count confidence merges
                    if (has_confidence && confidence.count(f.frame_id))
                        ++conf_merged;
                } else {
                    ++total_2d_unlabeled;
                }
            }
            ++total_2d_frames;
        }

        write_2d_v2(cam_out, skeleton_name, frames, num_kps_2d,
                     confidence, has_confidence);
    }

    std::cout << "\n2D camera CSVs:" << std::endl;
    std::cout << "  Camera files:      " << camera_names.size() << std::endl;
    std::cout << "  Total frame rows:  " << total_2d_frames << std::endl;
    std::cout << "  Labeled keypoints: " << total_2d_labeled << std::endl;
    std::cout << "  Unlabeled:         " << total_2d_unlabeled
              << " (sentinels removed)" << std::endl;
    if (has_confidence) {
        std::cout << "  Confidence merged: " << conf_merged
                  << " entries (source=P)" << std::endl;
    }

    // ── Step 5: Copy annotations.json (as-is) ──
    if (fs::exists(annot_path)) {
        fs::copy_file(annot_path, output_folder + "/annotations.json",
                      fs::copy_options::overwrite_existing);
        std::cout << "\nannotations.json: copied" << std::endl;
    } else {
        std::cout << "\nannotations.json: not present" << std::endl;
    }

    // ── Summary ──
    std::cout << "\n=== Conversion Complete ===" << std::endl;
    std::cout << "Output: " << fs::absolute(output_folder).string() << std::endl;
    std::cout << "Files written: "
              << (fs::exists(kp3d_path) ? 1 : 0) + (int)camera_names.size()
              << " CSV" << (fs::exists(annot_path) ? " + annotations.json" : "")
              << std::endl;

    return 0;
}
