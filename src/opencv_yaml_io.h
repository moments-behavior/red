#pragma once
// opencv_yaml_io.h — Lightweight reader/writer for OpenCV-format YAML files.
// Handles !!opencv-matrix blocks and scalar entries (image_width, etc.).
// No OpenCV dependency.

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace opencv_yaml {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace detail {

inline std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Parse a flat list of doubles from a line or multiline "data: [ ... ]" block.
inline std::vector<double> parse_data_values(const std::string &block) {
    std::vector<double> vals;
    std::string clean;
    for (char c : block) {
        if (c == '[' || c == ']')
            continue;
        if (c == ',')
            clean += ' ';
        else
            clean += c;
    }
    std::istringstream iss(clean);
    double v;
    while (iss >> v)
        vals.push_back(v);
    return vals;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Parsed representation of an OpenCV YAML file
// ---------------------------------------------------------------------------
struct YamlFile {
    // Scalar entries: key -> string value
    std::unordered_map<std::string, std::string> scalars;
    // Matrix entries: key -> {rows, cols, data}
    struct MatrixEntry {
        int rows = 0, cols = 0;
        std::vector<double> data;
    };
    std::unordered_map<std::string, MatrixEntry> matrices;

    // Convenience: read a scalar as int
    int getInt(const std::string &key) const {
        auto it = scalars.find(key);
        if (it == scalars.end())
            throw std::runtime_error("YAML key not found: " + key);
        return std::stoi(it->second);
    }

    // Convenience: read a scalar as double
    double getDouble(const std::string &key) const {
        auto it = scalars.find(key);
        if (it == scalars.end())
            throw std::runtime_error("YAML key not found: " + key);
        return std::stod(it->second);
    }

    // Convenience: read a matrix as Eigen::MatrixXd
    Eigen::MatrixXd getMatrix(const std::string &key) const {
        auto it = matrices.find(key);
        if (it == matrices.end())
            throw std::runtime_error("YAML matrix not found: " + key);
        const auto &m = it->second;
        Eigen::MatrixXd mat(m.rows, m.cols);
        for (int r = 0; r < m.rows; r++)
            for (int c = 0; c < m.cols; c++)
                mat(r, c) = m.data[r * m.cols + c];
        return mat;
    }

    bool hasKey(const std::string &key) const {
        return scalars.count(key) || matrices.count(key);
    }
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
// Parses files like:
//   %YAML:1.0
//   ---
//   image_width: 3208
//   camera_matrix: !!opencv-matrix
//      rows: 3
//      cols: 3
//      dt: d
//      data: [ 3200.5, 0., 1604., 0., 3200.5, 1200., 0., 0., 1. ]
//
inline YamlFile read(const std::string &path) {
    YamlFile result;
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open YAML file: " + path);

    std::string line;
    std::string current_matrix_key;
    int mat_rows = 0, mat_cols = 0;
    std::string data_accum;
    bool in_data = false;

    auto finish_matrix = [&]() {
        if (!current_matrix_key.empty() && !data_accum.empty()) {
            YamlFile::MatrixEntry entry;
            entry.rows = mat_rows;
            entry.cols = mat_cols;
            entry.data = detail::parse_data_values(data_accum);
            result.matrices[current_matrix_key] = entry;
        }
        current_matrix_key.clear();
        data_accum.clear();
        in_data = false;
        mat_rows = mat_cols = 0;
    };

    while (std::getline(file, line)) {
        std::string trimmed = detail::trim(line);
        if (trimmed.empty() || trimmed[0] == '#' || trimmed == "---" ||
            trimmed.substr(0, 5) == "%YAML")
            continue;

        // If we're accumulating data lines for a matrix
        if (in_data) {
            data_accum += " " + trimmed;
            // Check if the closing bracket is present
            if (trimmed.find(']') != std::string::npos) {
                finish_matrix();
            }
            continue;
        }

        // Check for "key: value" pattern
        size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;

        std::string key = detail::trim(line.substr(0, colon));
        std::string value = detail::trim(line.substr(colon + 1));

        // Inside a matrix block — look for rows, cols, dt, data
        if (!current_matrix_key.empty()) {
            if (key == "rows") {
                mat_rows = std::stoi(value);
            } else if (key == "cols") {
                mat_cols = std::stoi(value);
            } else if (key == "dt") {
                // type tag — we always read as double
            } else if (key == "data") {
                data_accum = value;
                in_data = true;
                if (value.find(']') != std::string::npos) {
                    finish_matrix();
                }
            } else {
                // New top-level key — finish previous matrix
                finish_matrix();
                // Re-process this line as a new entry
                if (value.find("!!opencv-matrix") != std::string::npos) {
                    current_matrix_key = key;
                } else {
                    result.scalars[key] = value;
                }
            }
        } else {
            // Top-level key
            if (value.find("!!opencv-matrix") != std::string::npos) {
                current_matrix_key = key;
            } else {
                result.scalars[key] = value;
            }
        }
    }

    // Handle trailing matrix
    if (!current_matrix_key.empty())
        finish_matrix();

    return result;
}

// ---------------------------------------------------------------------------
// Writer — emit OpenCV-compatible YAML
// ---------------------------------------------------------------------------

class YamlWriter {
  public:
    explicit YamlWriter(const std::string &path) : ofs_(path) {
        if (!ofs_.is_open())
            throw std::runtime_error("Cannot open YAML for writing: " + path);
        ofs_ << "%YAML:1.0\n---\n";
    }

    void writeScalar(const std::string &key, int value) {
        ofs_ << key << ": " << value << "\n";
    }

    void writeScalar(const std::string &key, double value) {
        ofs_ << key << ": " << value << "\n";
    }

    void writeMatrix(const std::string &key, const Eigen::MatrixXd &mat) {
        ofs_ << key << ": !!opencv-matrix\n";
        ofs_ << "   rows: " << mat.rows() << "\n";
        ofs_ << "   cols: " << mat.cols() << "\n";
        ofs_ << "   dt: d\n";
        ofs_ << "   data: [ ";
        for (int r = 0; r < mat.rows(); r++) {
            for (int c = 0; c < mat.cols(); c++) {
                if (r > 0 || c > 0)
                    ofs_ << ", ";
                ofs_ << mat(r, c);
            }
        }
        ofs_ << " ]\n";
    }

    bool isOpen() const { return ofs_.is_open(); }
    void close() { ofs_.close(); }

  private:
    std::ofstream ofs_;
};

} // namespace opencv_yaml
