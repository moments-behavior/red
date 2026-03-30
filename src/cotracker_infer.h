#pragma once
#include <array>
#include <string>
#include <torch/script.h>
#include <vector>

class CoTrackerInfer {
public:
    struct CoTrackerResult {
        std::vector<std::array<float, 2>> tracks; // [N] points: {x_px, y_px}
        std::vector<float> vis;                    // [N] visibility in [0,1]
    };

    CoTrackerInfer() = default;
    ~CoTrackerInfer() = default;

    // Load TorchScript .pt model
    bool load(const std::string &model_path);
    bool isLoaded() const { return loaded_; }

    // Propagate keypoints across T frames.
    // frames_rgba: one RGBA uint8 buffer per frame (all same width/height).
    // query_pts[n] = {frame_idx, x_px, y_px} for the n-th query point.
    // Returns one CoTrackerResult per frame.
    std::vector<CoTrackerResult>
    track(const std::vector<unsigned char *> &frames_rgba, int width, int height,
          const std::vector<std::array<float, 3>> &query_pts);

private:
    bool loaded_ = false;
    torch::jit::Module model_;

    // Downsample factor applied before running the model (for speed)
    static constexpr float CT_SCALE = 0.25f;
};
