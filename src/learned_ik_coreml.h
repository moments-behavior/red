#pragma once
// learned_ik_coreml.h — Learned IK inference via CoreML
//
// Loads a trained .mlpackage that maps 3D keypoints to MuJoCo qpos.
// Input:  kp3d [1, 24, 3] (float32) — keypoints in MuJoCo frame (meters)
//         valid_mask [1, 24] (float32) — 1.0 if triangulated, 0.0 otherwise
// Output: qpos [1, 68] (float32) — MuJoCo joint angles
//
// The arena alignment (R, t, scale) must be applied to keypoints BEFORE
// calling predict(). The model expects keypoints in MuJoCo world frame.
//
// macOS only. Inference: ~0.2ms on Apple Silicon (CPU+ANE).

#ifdef __APPLE__

#include <string>
#include <vector>

struct LearnedIKState {
    bool loaded = false;
    std::string status;
    std::string model_path;

    // Opaque pointer to MLModel
    void *model = nullptr;

    // Model dimensions (set on load)
    int n_keypoints = 24;
    int n_qpos = 68;

    // Timing
    float last_inference_ms = 0.0f;
};

// Load .mlpackage model. Returns true on success.
bool learned_ik_init(LearnedIKState &s, const std::string &mlpackage_path);

// Run inference: keypoints (MuJoCo frame, meters) → qpos.
// kp3d: [n_keypoints * 3] flattened (x0,y0,z0, x1,y1,z1, ...)
// valid: [n_keypoints] validity mask (1.0 or 0.0)
// qpos_out: [n_qpos] output joint angles (caller must allocate)
// Returns true on success.
bool learned_ik_predict(LearnedIKState &s,
                        const float *kp3d, const float *valid,
                        float *qpos_out);

void learned_ik_cleanup(LearnedIKState &s);

#endif // __APPLE__
