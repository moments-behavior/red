#pragma once
// sam_inference.h — SAM segmentation via ONNX Runtime
//
// Supports both MobileSAM (~9MB, fastest) and SAM 2.1 Tiny (~117MB, higher quality).
// Optional dependency: compile with -DRED_HAS_ONNXRUNTIME to enable.
// Without ONNX Runtime, all functions are stubs that return false.
//
// Usage:
//   SamState sam;
//   sam_init(sam, SamModel::MobileSAM, "encoder.onnx", "decoder.onnx");
//   auto mask = sam_segment(sam, rgb, w, h, fg_points, bg_points);
//   auto polygons = sam_mask_to_polygon(mask);

#include "types.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef RED_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#endif

// ── Types ──

struct SamMask {
    std::vector<uint8_t> data; // H x W binary mask (0 or 255)
    int width = 0;
    int height = 0;
    float iou_score = 0.0f;
    bool valid = false;
};

enum class SamModel { MobileSAM, SAM2 };

struct SamState {
    bool loaded = false;
    bool available = false;
    SamModel model = SamModel::MobileSAM;

    // Model paths
    std::string encoder_path;
    std::string decoder_path;

    // Status
    std::string status;
    float last_encode_ms = 0;
    float last_decode_ms = 0;

    // Cached embedding (avoid re-encoding same frame)
    int cached_frame = -1;
    int cached_cam = -1;

    // Original image dimensions for coordinate scaling
    int orig_w = 0, orig_h = 0;
    // Scale factor used during preprocessing (for coordinate mapping)
    float scale = 1.0f;
    int pad_x = 0, pad_y = 0; // padding offsets (MobileSAM only)

#ifdef RED_HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> encoder_session;
    std::unique_ptr<Ort::Session> decoder_session;

    // Cached encoder outputs
    std::vector<float> image_embedding;    // [1,256,64,64]
    std::vector<float> high_res_feats_0;   // SAM2 only: [1,32,256,256]
    std::vector<float> high_res_feats_1;   // SAM2 only: [1,64,128,128]
#endif
};

// ── Helpers (internal) ──

namespace sam_detail {

// Bilinear resize a single-channel float buffer
inline std::vector<float> bilinear_resize(const float *src, int sw, int sh,
                                           int dw, int dh) {
    std::vector<float> dst(dw * dh);
    float sx = (float)sw / dw;
    float sy = (float)sh / dh;
    for (int y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(sh - 1, y0 + 1);
        float wy = fy - y0;
        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(sw - 1, x0 + 1);
            float wx = fx - x0;
            float v = (1 - wy) * ((1 - wx) * src[y0 * sw + x0] + wx * src[y0 * sw + x1]) +
                      wy * ((1 - wx) * src[y1 * sw + x0] + wx * src[y1 * sw + x1]);
            dst[y * dw + x] = v;
        }
    }
    return dst;
}

// Preprocess image for MobileSAM: longest-side resize to 1024, zero-pad to square
// Returns CHW float32 [3, 1024, 1024] with ImageNet normalization (0-255 range)
inline std::vector<float> preprocess_mobilesam(const uint8_t *rgb, int w, int h,
                                                float &out_scale, int &out_pad_x,
                                                int &out_pad_y) {
    constexpr int target = 1024;
    constexpr float mean[3] = {123.675f, 116.28f, 103.53f};
    constexpr float inv_std[3] = {1.0f / 58.395f, 1.0f / 57.12f, 1.0f / 57.375f};

    // Longest-side resize
    float scale = (float)target / std::max(w, h);
    int nw = (int)(w * scale + 0.5f);
    int nh = (int)(h * scale + 0.5f);
    out_scale = scale;
    out_pad_x = 0;
    out_pad_y = 0;

    // Resize each channel via bilinear interpolation, then normalize + pad
    std::vector<float> result(3 * target * target, 0.0f);

    for (int c = 0; c < 3; ++c) {
        // Extract channel
        std::vector<float> chan(w * h);
        for (int i = 0; i < w * h; ++i)
            chan[i] = (float)rgb[i * 3 + c];

        // Resize
        auto resized = bilinear_resize(chan.data(), w, h, nw, nh);

        // Normalize and place into padded output
        float *dst = result.data() + c * target * target;
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x)
                dst[y * target + x] = (resized[y * nw + x] - mean[c]) * inv_std[c];
    }
    return result;
}

// Preprocess image for SAM 2.1: direct resize to 1024x1024
// Returns CHW float32 [3, 1024, 1024] with ImageNet normalization (0-1 range)
inline std::vector<float> preprocess_sam2(const uint8_t *rgb, int w, int h) {
    constexpr int target = 1024;
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float inv_std[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

    std::vector<float> result(3 * target * target);

    for (int c = 0; c < 3; ++c) {
        std::vector<float> chan(w * h);
        for (int i = 0; i < w * h; ++i)
            chan[i] = (float)rgb[i * 3 + c];

        auto resized = bilinear_resize(chan.data(), w, h, target, target);

        float *dst = result.data() + c * target * target;
        for (int i = 0; i < target * target; ++i)
            dst[i] = (resized[i] / 255.0f - mean[c]) * inv_std[c];
    }
    return result;
}

// Marching-squares contour extraction from binary mask.
// Returns outer boundary polygon(s) in image coordinates.
inline std::vector<std::vector<tuple_d>> extract_contours(const uint8_t *mask,
                                                           int w, int h) {
    std::vector<std::vector<tuple_d>> polygons;

    // Visited edge tracking to avoid duplicate contours
    // Each cell (x,y) in the (w-1)x(h-1) grid has a 4-bit marching square case.
    // We trace boundary segments between foreground and background.
    std::vector<bool> visited(w * h, false);

    // Simple boundary tracing: find connected boundary pixels and trace them
    // A boundary pixel is foreground with at least one background 4-neighbor
    auto is_fg = [&](int x, int y) -> bool {
        if (x < 0 || x >= w || y < 0 || y >= h) return false;
        return mask[y * w + x] > 127;
    };

    auto is_boundary = [&](int x, int y) -> bool {
        if (!is_fg(x, y)) return false;
        return !is_fg(x - 1, y) || !is_fg(x + 1, y) ||
               !is_fg(x, y - 1) || !is_fg(x, y + 1);
    };

    // 8-connected boundary tracing (Moore neighborhood)
    const int dx8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy8[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int sy = 0; sy < h; ++sy) {
        for (int sx = 0; sx < w; ++sx) {
            if (visited[sy * w + sx] || !is_boundary(sx, sy))
                continue;

            std::vector<tuple_d> contour;
            int cx = sx, cy = sy;
            int dir = 0; // start direction

            do {
                contour.push_back({(double)cx, (double)cy});
                visited[cy * w + cx] = true;

                // Find next boundary pixel (Moore neighbor tracing)
                bool found = false;
                int start_dir = (dir + 5) % 8; // backtrack direction + 1
                for (int i = 0; i < 8; ++i) {
                    int d = (start_dir + i) % 8;
                    int nx = cx + dx8[d];
                    int ny = cy + dy8[d];
                    if (is_boundary(nx, ny)) {
                        cx = nx;
                        cy = ny;
                        dir = d;
                        found = true;
                        break;
                    }
                }
                if (!found) break;
            } while (cx != sx || cy != sy);

            if (contour.size() >= 3)
                polygons.push_back(std::move(contour));
        }
    }
    return polygons;
}

// Douglas-Peucker polygon simplification
inline std::vector<tuple_d> simplify_polygon(const std::vector<tuple_d> &pts,
                                              double epsilon) {
    if (pts.size() <= 2) return pts;

    // Find point with max distance from line(first, last)
    double max_dist = 0;
    size_t max_idx = 0;
    double ax = pts.front().x, ay = pts.front().y;
    double bx = pts.back().x, by = pts.back().y;
    double dx = bx - ax, dy = by - ay;
    double len2 = dx * dx + dy * dy;

    for (size_t i = 1; i + 1 < pts.size(); ++i) {
        double dist;
        if (len2 < 1e-12) {
            double ex = pts[i].x - ax, ey = pts[i].y - ay;
            dist = std::sqrt(ex * ex + ey * ey);
        } else {
            double t = ((pts[i].x - ax) * dx + (pts[i].y - ay) * dy) / len2;
            t = std::clamp(t, 0.0, 1.0);
            double px = ax + t * dx - pts[i].x;
            double py = ay + t * dy - pts[i].y;
            dist = std::sqrt(px * px + py * py);
        }
        if (dist > max_dist) {
            max_dist = dist;
            max_idx = i;
        }
    }

    if (max_dist > epsilon) {
        auto left = std::vector<tuple_d>(pts.begin(), pts.begin() + max_idx + 1);
        auto right = std::vector<tuple_d>(pts.begin() + max_idx, pts.end());
        auto rl = simplify_polygon(left, epsilon);
        auto rr = simplify_polygon(right, epsilon);
        rl.insert(rl.end(), rr.begin() + 1, rr.end());
        return rl;
    }
    return {pts.front(), pts.back()};
}

} // namespace sam_detail

// ── Public API ──

// Initialize SAM with model paths. Returns true if models loaded successfully.
inline bool sam_init(SamState &s, SamModel model_type,
                     const char *encoder_onnx, const char *decoder_onnx) {
#ifdef RED_HAS_ONNXRUNTIME
    // Clean up any previous session
    s.decoder_session.reset();
    s.encoder_session.reset();
    s.loaded = false;
    s.available = true;
    s.model = model_type;
    s.encoder_path = encoder_onnx;
    s.decoder_path = decoder_onnx;
    s.cached_frame = -1;
    s.cached_cam = -1;

    try {
        if (!s.env)
            s.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "red_sam");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
        // Try CoreML EP for GPU acceleration, fall back to CPU
        try {
            uint32_t coreml_flags = 0;
            // COREML_FLAG_USE_CPU_ONLY = 0, default uses ANE/GPU
            (void)OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags);
            s.status = "Loading with CoreML...";
        } catch (...) {
            s.status = "CoreML unavailable, using CPU...";
        }
#endif

        s.encoder_session = std::make_unique<Ort::Session>(
            *s.env, encoder_onnx, opts);
        s.decoder_session = std::make_unique<Ort::Session>(
            *s.env, decoder_onnx, opts);

        const char *model_name = (model_type == SamModel::MobileSAM)
                                     ? "MobileSAM" : "SAM 2.1";
        s.status = std::string(model_name) + " loaded";
        s.loaded = true;
        return true;
    } catch (const Ort::Exception &e) {
        s.status = std::string("ONNX error: ") + e.what();
        s.loaded = false;
        return false;
    } catch (const std::exception &e) {
        s.status = std::string("Error: ") + e.what();
        s.loaded = false;
        return false;
    }
#else
    (void)model_type; (void)encoder_onnx; (void)decoder_onnx;
    s.available = false;
    s.status = "ONNX Runtime not available (compile with -DRED_HAS_ONNXRUNTIME)";
    return false;
#endif
}

// Run encoder on an image, cache the embedding
inline bool sam_encode(SamState &s, const uint8_t *rgb, int w, int h,
                       int frame_num, int cam_idx) {
#ifdef RED_HAS_ONNXRUNTIME
    if (!s.loaded) return false;

    // Skip if already cached (guard against default sentinel -1)
    if (frame_num >= 0 && cam_idx >= 0 &&
        s.cached_frame == frame_num && s.cached_cam == cam_idx)
        return true;

    auto t0 = std::chrono::steady_clock::now();

    s.orig_w = w;
    s.orig_h = h;

    // Preprocess
    std::vector<float> input_tensor;
    if (s.model == SamModel::MobileSAM) {
        input_tensor = sam_detail::preprocess_mobilesam(rgb, w, h,
                                                         s.scale, s.pad_x, s.pad_y);
    } else {
        input_tensor = sam_detail::preprocess_sam2(rgb, w, h);
        s.scale = 1024.0f / std::max(w, h); // for coordinate mapping
        s.pad_x = 0;
        s.pad_y = 0;
    }

    // Create input tensor
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape = {1, 3, 1024, 1024};
    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());

    // Run encoder
    const char *input_names[] = {"image"};

    if (s.model == SamModel::MobileSAM) {
        const char *output_names[] = {"image_embeddings"};
        auto outputs = s.encoder_session->Run(
            Ort::RunOptions{nullptr}, input_names, &input_val, 1,
            output_names, 1);

        // Copy embedding: [1, 256, 64, 64]
        auto &emb = outputs[0];
        auto emb_info = emb.GetTensorTypeAndShapeInfo();
        size_t emb_size = emb_info.GetElementCount();
        s.image_embedding.resize(emb_size);
        std::memcpy(s.image_embedding.data(), emb.GetTensorData<float>(),
                     emb_size * sizeof(float));
    } else {
        // SAM 2.1: 3 outputs
        const char *output_names[] = {"image_embed", "high_res_feats_0",
                                       "high_res_feats_1"};
        auto outputs = s.encoder_session->Run(
            Ort::RunOptions{nullptr}, input_names, &input_val, 1,
            output_names, 3);

        // image_embed [1, 256, 64, 64]
        {
            auto &t = outputs[0];
            size_t n = t.GetTensorTypeAndShapeInfo().GetElementCount();
            s.image_embedding.resize(n);
            std::memcpy(s.image_embedding.data(), t.GetTensorData<float>(),
                         n * sizeof(float));
        }
        // high_res_feats_0 [1, 32, 256, 256]
        {
            auto &t = outputs[1];
            size_t n = t.GetTensorTypeAndShapeInfo().GetElementCount();
            s.high_res_feats_0.resize(n);
            std::memcpy(s.high_res_feats_0.data(), t.GetTensorData<float>(),
                         n * sizeof(float));
        }
        // high_res_feats_1 [1, 64, 128, 128]
        {
            auto &t = outputs[2];
            size_t n = t.GetTensorTypeAndShapeInfo().GetElementCount();
            s.high_res_feats_1.resize(n);
            std::memcpy(s.high_res_feats_1.data(), t.GetTensorData<float>(),
                         n * sizeof(float));
        }
    }

    s.cached_frame = frame_num;
    s.cached_cam = cam_idx;

    auto t1 = std::chrono::steady_clock::now();
    s.last_encode_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return true;
#else
    (void)rgb; (void)w; (void)h; (void)frame_num; (void)cam_idx;
    return false;
#endif
}

// Run decoder with point/bbox prompts. Returns binary mask.
inline SamMask sam_decode(SamState &s,
                           const std::vector<tuple_d> &fg_points,
                           const std::vector<tuple_d> &bg_points,
                           const double *bbox_prompt = nullptr) {
    SamMask result;
    result.width = s.orig_w;
    result.height = s.orig_h;

#ifdef RED_HAS_ONNXRUNTIME
    if (!s.loaded || s.image_embedding.empty()) {
        s.status = "No cached embedding";
        return result;
    }

    auto t0 = std::chrono::steady_clock::now();

    // Build point prompts
    // MobileSAM: coords in 1024x1024 space (longest-side scaled + padded)
    // SAM 2.1:   coords in 1024x1024 space (directly scaled)
    int n_fg = (int)fg_points.size();
    int n_bg = (int)bg_points.size();
    int n_bbox = bbox_prompt ? 2 : 0;
    int n_points = n_fg + n_bg + n_bbox;

    if (n_points == 0) {
        s.status = "No prompts";
        return result;
    }

    std::vector<float> coords(n_points * 2);
    std::vector<float> labels(n_points);

    int idx = 0;
    auto map_coord = [&](double img_x, double img_y, float &ox, float &oy) {
        if (s.model == SamModel::MobileSAM) {
            ox = (float)(img_x * s.scale) + s.pad_x;
            oy = (float)(img_y * s.scale) + s.pad_y;
        } else {
            // SAM 2.1: direct resize to 1024x1024
            ox = (float)(img_x * 1024.0 / s.orig_w);
            oy = (float)(img_y * 1024.0 / s.orig_h);
        }
    };

    // Foreground points (label = 1)
    for (const auto &pt : fg_points) {
        map_coord(pt.x, pt.y, coords[idx * 2], coords[idx * 2 + 1]);
        labels[idx] = 1.0f;
        idx++;
    }
    // Background points (label = 0)
    for (const auto &pt : bg_points) {
        map_coord(pt.x, pt.y, coords[idx * 2], coords[idx * 2 + 1]);
        labels[idx] = 0.0f;
        idx++;
    }
    // Bbox corners (labels 2 and 3)
    if (bbox_prompt) {
        map_coord(bbox_prompt[0], bbox_prompt[1],
                  coords[idx * 2], coords[idx * 2 + 1]);
        labels[idx] = 2.0f;
        idx++;
        map_coord(bbox_prompt[2], bbox_prompt[3],
                  coords[idx * 2], coords[idx * 2 + 1]);
        labels[idx] = 3.0f;
        idx++;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // mask_input: zeros (no prior mask)
    std::vector<float> mask_input(1 * 1 * 256 * 256, 0.0f);
    float has_mask_input[1] = {0.0f};

    // Tensor shapes
    std::array<int64_t, 3> coords_shape = {1, (int64_t)n_points, 2};
    std::array<int64_t, 2> labels_shape = {1, (int64_t)n_points};
    std::array<int64_t, 4> mask_in_shape = {1, 1, 256, 256};
    std::array<int64_t, 1> has_mask_shape = {1};

    if (s.model == SamModel::MobileSAM) {
        // MobileSAM decoder inputs
        std::array<int64_t, 4> emb_shape = {1, 256, 64, 64};
        float orig_size[2] = {(float)s.orig_h, (float)s.orig_w};
        std::array<int64_t, 1> orig_shape = {2};

        Ort::Value inputs[] = {
            Ort::Value::CreateTensor<float>(mem_info, s.image_embedding.data(),
                                             s.image_embedding.size(),
                                             emb_shape.data(), emb_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, coords.data(), coords.size(),
                                             coords_shape.data(), coords_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, labels.data(), labels.size(),
                                             labels_shape.data(), labels_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, mask_input.data(),
                                             mask_input.size(),
                                             mask_in_shape.data(), mask_in_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, has_mask_input, 1,
                                             has_mask_shape.data(), has_mask_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, orig_size, 2,
                                             orig_shape.data(), orig_shape.size()),
        };

        const char *input_names[] = {"image_embeddings", "point_coords",
                                      "point_labels", "mask_input",
                                      "has_mask_input", "orig_im_size"};
        const char *output_names[] = {"masks", "iou_predictions", "low_res_masks"};

        auto outputs = s.decoder_session->Run(
            Ort::RunOptions{nullptr}, input_names, inputs, 6, output_names, 3);

        // masks: [1, 4, H, W] at original image size
        auto &masks_tensor = outputs[0];
        auto &iou_tensor = outputs[1];
        auto masks_shape = masks_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int n_masks = (int)masks_shape[1];
        int mh = (int)masks_shape[2];
        int mw = (int)masks_shape[3];
        const float *masks_data = masks_tensor.GetTensorData<float>();
        const float *iou_data = iou_tensor.GetTensorData<float>();

        // Select best mask by IoU
        int best = 0;
        for (int i = 1; i < n_masks; ++i)
            if (iou_data[i] > iou_data[best]) best = i;

        result.iou_score = iou_data[best];
        result.width = mw;
        result.height = mh;
        result.data.resize(mw * mh);

        const float *best_mask = masks_data + best * mh * mw;
        for (int i = 0; i < mh * mw; ++i)
            result.data[i] = (best_mask[i] > 0.0f) ? 255 : 0;

        result.valid = true;

    } else {
        // SAM 2.1 decoder inputs
        std::array<int64_t, 4> emb_shape = {1, 256, 64, 64};
        std::array<int64_t, 4> hr0_shape = {1, 32, 256, 256};
        std::array<int64_t, 4> hr1_shape = {1, 64, 128, 128};

        Ort::Value inputs[] = {
            Ort::Value::CreateTensor<float>(mem_info, s.image_embedding.data(),
                                             s.image_embedding.size(),
                                             emb_shape.data(), emb_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, s.high_res_feats_0.data(),
                                             s.high_res_feats_0.size(),
                                             hr0_shape.data(), hr0_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, s.high_res_feats_1.data(),
                                             s.high_res_feats_1.size(),
                                             hr1_shape.data(), hr1_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, coords.data(), coords.size(),
                                             coords_shape.data(), coords_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, labels.data(), labels.size(),
                                             labels_shape.data(), labels_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, mask_input.data(),
                                             mask_input.size(),
                                             mask_in_shape.data(), mask_in_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, has_mask_input, 1,
                                             has_mask_shape.data(), has_mask_shape.size()),
        };

        const char *input_names[] = {"image_embed", "high_res_feats_0",
                                      "high_res_feats_1", "point_coords",
                                      "point_labels", "mask_input",
                                      "has_mask_input"};
        const char *output_names[] = {"masks", "iou_predictions"};

        auto outputs = s.decoder_session->Run(
            Ort::RunOptions{nullptr}, input_names, inputs, 7, output_names, 2);

        // masks: [1, 3, 256, 256] — need to resize to original image size
        auto &masks_tensor = outputs[0];
        auto &iou_tensor = outputs[1];
        auto masks_shape = masks_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int n_masks = (int)masks_shape[1];
        int mh = (int)masks_shape[2]; // 256
        int mw = (int)masks_shape[3]; // 256
        const float *masks_data = masks_tensor.GetTensorData<float>();
        const float *iou_data = iou_tensor.GetTensorData<float>();

        // Select best mask by IoU
        int best = 0;
        for (int i = 1; i < n_masks; ++i)
            if (iou_data[i] > iou_data[best]) best = i;

        result.iou_score = iou_data[best];

        // Resize mask from 256x256 to original image size
        const float *best_mask = masks_data + best * mh * mw;
        auto resized = sam_detail::bilinear_resize(best_mask, mw, mh,
                                                    s.orig_w, s.orig_h);

        result.width = s.orig_w;
        result.height = s.orig_h;
        result.data.resize(s.orig_w * s.orig_h);
        for (int i = 0; i < s.orig_w * s.orig_h; ++i)
            result.data[i] = (resized[i] > 0.0f) ? 255 : 0;

        result.valid = true;
    }

    auto t1 = std::chrono::steady_clock::now();
    s.last_decode_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return result;
#else
    (void)fg_points; (void)bg_points; (void)bbox_prompt;
    s.status = "ONNX Runtime not available";
    return result;
#endif
}

// High-level: encode if needed, then decode. This is the main entry point.
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

    if (!sam_encode(s, rgb, w, h, frame_num, cam_idx)) {
        s.status = "Encoder failed";
        return result;
    }

    return sam_decode(s, fg_points, bg_points, bbox_prompt);
#else
    (void)rgb; (void)fg_points; (void)bg_points;
    (void)bbox_prompt; (void)frame_num; (void)cam_idx;
    s.status = "ONNX Runtime not available";
    return result;
#endif
}

// Convert a binary mask to polygon contours with simplification.
inline std::vector<std::vector<tuple_d>> sam_mask_to_polygon(const SamMask &mask,
                                                              double simplify_eps = 2.0) {
    std::vector<std::vector<tuple_d>> polygons;
    if (!mask.valid || mask.data.empty()) return polygons;

    auto raw = sam_detail::extract_contours(mask.data.data(), mask.width, mask.height);

    for (auto &poly : raw) {
        if (simplify_eps > 0 && poly.size() > 4) {
            auto simplified = sam_detail::simplify_polygon(poly, simplify_eps);
            if (simplified.size() >= 3)
                polygons.push_back(std::move(simplified));
        } else if (poly.size() >= 3) {
            polygons.push_back(std::move(poly));
        }
    }
    return polygons;
}

// Release SAM resources
inline void sam_cleanup(SamState &s) {
#ifdef RED_HAS_ONNXRUNTIME
    s.decoder_session.reset();
    s.encoder_session.reset();
    s.image_embedding.clear();
    s.high_res_feats_0.clear();
    s.high_res_feats_1.clear();
#endif
    s.loaded = false;
    s.cached_frame = -1;
    s.cached_cam = -1;
    s.status.clear();
}
