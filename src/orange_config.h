#pragma once
// Export designed sensor ROI crops into orange camera configs.
//
// orange (the acquisition app) reads one flat JSON per camera, named
// {serial}.json, from its config folder. RED updates these files IN PLACE:
// only "width", "height", "offsetx", "offsety" are (re)written; every other
// key is preserved.
//
// IMPORTANT: today orange only understands "width"/"height" (ROI anchored at
// the sensor origin) — it hardcodes OffsetX/OffsetY to 0 at camera open
// (orange src/camera.cpp:402) and never reads offset keys. The "offsetx"/
// "offsety" keys written here are ignored until orange's
// load_camera_json_config_files() is patched to consume them. Unknown keys
// are safe: orange reads by key name.
//
// orange's GUI snaps all ROI values to multiples of 16; the actual EVT camera
// increments are device-dependent and orange's non-GUI path does not snap,
// so values exported here should already be multiples of 16.

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "crop_calibration.h"

namespace OrangeConfig {

struct ExportResult {
    bool success = false;
    std::string error;
    std::vector<std::string> warnings;
    int updated = 0;
};

// full_widths/full_heights: optional per-spec-camera full-frame dims (from the
// stage-1 YAMLs) used to validate offset+size <= sensor; pass empty to skip.
inline ExportResult update_orange_configs(
    const std::string &config_folder,
    const CropCalibration::CropSpec &spec,
    const std::vector<int> &full_widths = {},
    const std::vector<int> &full_heights = {}) {
    namespace fs = std::filesystem;
    ExportResult result;

    if (!fs::is_directory(config_folder)) {
        result.error = "Not a directory: " + config_folder;
        return result;
    }
    if (spec.cameras.empty()) {
        result.error = "Crop spec has no cameras";
        return result;
    }

    for (size_t i = 0; i < spec.cameras.size(); i++) {
        const auto &crop = spec.cameras[i];
        std::string path = config_folder + "/" + crop.serial + ".json";
        if (!fs::exists(path)) {
            result.warnings.push_back("No orange config for camera " +
                                      crop.serial + " (" + path + ") — skipped");
            continue;
        }

        nlohmann::json j;
        try {
            std::ifstream f(path);
            f >> j;
        } catch (const std::exception &e) {
            result.error = "Parse error in " + path + ": " + e.what();
            return result;
        }

        for (int v : {crop.offset_x, crop.offset_y, crop.width, crop.height})
            if (v % 16 != 0) {
                result.warnings.push_back(
                    "Camera " + crop.serial +
                    ": ROI value not a multiple of 16 — the camera may "
                    "reject or adjust it");
                break;
            }
        if (i < full_widths.size() && i < full_heights.size() &&
            full_widths[i] > 0 && full_heights[i] > 0 &&
            (crop.offset_x + crop.width > full_widths[i] ||
             crop.offset_y + crop.height > full_heights[i]))
            result.warnings.push_back(
                "Camera " + crop.serial + ": crop extends beyond full frame " +
                std::to_string(full_widths[i]) + "x" +
                std::to_string(full_heights[i]));

        j["width"] = crop.width;
        j["height"] = crop.height;
        j["offsetx"] = crop.offset_x;  // consumed once orange is patched
        j["offsety"] = crop.offset_y;

        try {
            std::ofstream f(path, std::ios::trunc);
            if (!f.is_open()) {
                result.error = "Cannot write " + path;
                return result;
            }
            f << j.dump(4) << "\n";
        } catch (const std::exception &e) {
            result.error = "Write error for " + path + ": " + e.what();
            return result;
        }
        result.updated++;
    }

    if (result.updated == 0) {
        result.error = "No orange configs matched any camera serial in " +
                       config_folder;
        return result;
    }
    result.success = true;
    return result;
}

}  // namespace OrangeConfig
