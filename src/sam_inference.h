#pragma once
// sam_inference.h — MobileSAM segmentation via ONNX Runtime
//
// Optional dependency: compile with -DRED_HAS_ONNXRUNTIME to enable.
// Without ONNX Runtime, all functions are stubs that return false.
//
// Architecture: MobileSAM has two stages:
//   1. Image encoder (~8ms GPU, ~100ms CPU) — run once per frame, cached
//   2. Prompt decoder (~4ms) — run per click/bbox, near-instant
//
// Usage:
//   sam_init(state, "encoder.onnx", "decoder.onnx");
//   auto mask = sam_segment(state, rgb, w, h, fg_points, bg_points);
//   auto polygon = sam_mask_to_polygon(mask);

#include "types.h"
#include <cstdint>
#include <string>
#include <vector>

struct SamMask {
    std::vector<uint8_t> data; // H x W binary mask (0 or 255)
    int width = 0;
    int height = 0;
    float iou_score = 0.0f;
    bool valid = false;
};

struct SamState {
    bool loaded = false;
    bool available = false; // true if ONNX Runtime is compiled in

    // Cached image embedding (avoid re-encoding same frame)
    int cached_frame = -1;
    int cached_cam = -1;

    // Model paths
    std::string encoder_path;
    std::string decoder_path;

    // Status
    std::string status;
    float last_encode_ms = 0;
    float last_decode_ms = 0;

#ifdef RED_HAS_ONNXRUNTIME
    // ONNX Runtime session handles (opaque, managed internally)
    void *encoder_session = nullptr;
    void *decoder_session = nullptr;
    void *env = nullptr;

    // Cached encoder output
    std::vector<float> image_embedding;
    int embed_dim = 0;
    int embed_h = 0;
    int embed_w = 0;
#endif
};

// Initialize SAM with model paths. Returns true if models loaded successfully.
inline bool sam_init(SamState &s, const char *encoder_onnx,
                     const char *decoder_onnx) {
#ifdef RED_HAS_ONNXRUNTIME
    s.encoder_path = encoder_onnx;
    s.decoder_path = decoder_onnx;
    s.available = true;

    // TODO: Create ONNX Runtime InferenceSession for encoder and decoder
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sam");
    // Ort::SessionOptions opts;
    // opts.SetIntraOpNumThreads(4);
    // #ifdef __APPLE__
    // opts.AppendExecutionProvider_CoreML(0);
    // #endif
    // encoder_session = new Ort::Session(env, encoder_onnx, opts);
    // decoder_session = new Ort::Session(env, decoder_onnx, opts);

    s.status = "SAM models loaded";
    s.loaded = true;
    return true;
#else
    (void)encoder_onnx; (void)decoder_onnx;
    s.available = false;
    s.status = "ONNX Runtime not available (compile with -DRED_HAS_ONNXRUNTIME)";
    return false;
#endif
}

// Run SAM segmentation with point prompts and optional bbox prompt.
// fg_points/bg_points are in image coordinates (top-left origin).
// Returns a binary mask.
inline SamMask sam_segment(SamState &s, const uint8_t *rgb, int w, int h,
                            const std::vector<tuple_d> &fg_points,
                            const std::vector<tuple_d> &bg_points,
                            const double *bbox_prompt = nullptr,
                            int frame_num = -1, int cam_idx = -1) {
    SamMask result;
    result.width = w;
    result.height = h;

#ifdef RED_HAS_ONNXRUNTIME
    if (!s.loaded) {
        s.status = "SAM not initialized";
        return result;
    }

    // TODO: Implementation outline:
    // 1. If frame/cam changed, run encoder on the image
    //    - Resize to 1024x1024
    //    - Normalize (ImageNet mean/std)
    //    - Run encoder session → image_embedding
    //    - Cache embedding + frame/cam
    //
    // 2. Build prompt tensors:
    //    - point_coords: Nx2 float (fg + bg points, scaled to 1024x1024)
    //    - point_labels: Nx1 float (1 for fg, 0 for bg)
    //    - If bbox_prompt: add 2 corner points with label 2,3
    //    - mask_input: 1x1x256x256 zeros (no prior mask)
    //    - has_mask_input: [0.0]
    //
    // 3. Run decoder session → masks (1x3xHxW), iou_predictions (1x3)
    //    - Select mask with highest IoU
    //    - Threshold at 0.0 → binary mask
    //    - Resize back to original image size

    s.status = "SAM inference not yet implemented";
    return result;
#else
    (void)rgb; (void)w; (void)h;
    (void)fg_points; (void)bg_points;
    (void)bbox_prompt; (void)frame_num; (void)cam_idx;
    s.status = "ONNX Runtime not available";
    return result;
#endif
}

// Convert a binary mask to polygon contours (for storing in Camera2D::mask_polygons).
// Uses a simple marching-squares contour follower.
inline std::vector<std::vector<tuple_d>> sam_mask_to_polygon(const SamMask &mask,
                                                              double simplify_eps = 2.0) {
    std::vector<std::vector<tuple_d>> polygons;
    if (!mask.valid || mask.data.empty()) return polygons;

    // Simple contour extraction: scan rows for transitions, then trace boundary.
    // For a production implementation, use cv::findContours or a dedicated
    // marching squares library. This is a placeholder that returns the
    // bounding rectangle of the mask as a single polygon.
    int w = mask.width, h = mask.height;
    int x_min = w, x_max = 0, y_min = h, y_max = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (mask.data[y * w + x] > 127) {
                x_min = std::min(x_min, x);
                x_max = std::max(x_max, x);
                y_min = std::min(y_min, y);
                y_max = std::max(y_max, y);
            }
        }
    }

    if (x_max > x_min && y_max > y_min) {
        std::vector<tuple_d> poly = {
            {(double)x_min, (double)y_min},
            {(double)x_max, (double)y_min},
            {(double)x_max, (double)y_max},
            {(double)x_min, (double)y_max},
        };
        polygons.push_back(std::move(poly));
    }

    (void)simplify_eps; // TODO: Ramer-Douglas-Peucker simplification
    return polygons;
}

// Release SAM resources
inline void sam_cleanup(SamState &s) {
#ifdef RED_HAS_ONNXRUNTIME
    // TODO: delete encoder_session, decoder_session, env
    s.encoder_session = nullptr;
    s.decoder_session = nullptr;
    s.env = nullptr;
#endif
    s.loaded = false;
    s.cached_frame = -1;
}
