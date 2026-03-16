// superpoint_coreml.mm — CoreML SuperPoint inference implementation
//
// Pipeline: CVPixelBuffer BGRA → vImage resize → vDSP grayscale+scale →
//           MLMultiArray [1,1,H,W] → CoreML inference (ANE) →
//           softmax → score map → NMS → top-K → bilinear interp descriptors → L2-norm

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include "superpoint_coreml.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <vector>

// ── Context ──

struct SuperPointCoreMLContext {
    MLModel *model = nil;
    bool available = false;
    int input_h = 480;
    int input_w = 640;
    int cell = 8; // SuperPoint cell size
};

// ── Helpers ──

static MLModel *sp_load_mlpackage(const std::string &path, std::string &err) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:path.c_str()]];

        NSError *error = nil;
        NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
        if (!compiled) {
            err = "Compile failed: " +
                  std::string(error.localizedDescription.UTF8String);
            return nil;
        }

        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:config
                                                   error:&error];
        if (!model) {
            err = "Load failed: " +
                  std::string(error.localizedDescription.UTF8String);
            return nil;
        }
        return model;
    }
}

// ── Public API ──

SuperPointCoreMLHandle superpoint_coreml_create(const char *model_path,
                                                  int input_h, int input_w) {
    auto *ctx = new SuperPointCoreMLContext();
    ctx->input_h = input_h;
    ctx->input_w = input_w;

    // Read actual dimensions from model_info.json alongside the model
    {
        std::string dir = std::string(model_path);
        auto slash = dir.rfind('/');
        if (slash != std::string::npos) dir = dir.substr(0, slash);
        std::string info_path = dir + "/model_info.json";
        FILE *f = fopen(info_path.c_str(), "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = 0;
            fclose(f);
            std::string s(buf);
            // Parse "input_height": 2200 and "input_width": 3208
            auto parse_int = [&](const std::string &key) -> int {
                auto pos = s.find("\"" + key + "\"");
                if (pos == std::string::npos) return 0;
                pos = s.find(':', pos);
                if (pos == std::string::npos) return 0;
                // Skip whitespace after colon
                pos++;
                while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
                try { return std::stoi(s.substr(pos)); } catch (...) { return 0; }
            };
            int h = parse_int("input_height");
            int w = parse_int("input_width");
            if (h > 0 && w > 0) {
                ctx->input_h = h;
                ctx->input_w = w;
                fprintf(stderr, "[SuperPointCoreML] model_info.json: %dx%d\n", w, h);
            }
        }
    }

    // Check macOS 13+
    if (@available(macOS 13.0, *)) {
        // OK
    } else {
        fprintf(stderr, "[SuperPointCoreML] Requires macOS 13+\n");
        return ctx;
    }

    std::string err;
    ctx->model = sp_load_mlpackage(model_path, err);
    if (!ctx->model) {
        fprintf(stderr, "[SuperPointCoreML] Failed to load model: %s\n", err.c_str());
        return ctx;
    }

    ctx->available = true;
    fprintf(stderr, "[SuperPointCoreML] Model loaded: %s (%dx%d)\n",
            model_path, ctx->input_w, ctx->input_h);
    return ctx;
}

bool superpoint_coreml_available(SuperPointCoreMLHandle handle) {
    return handle && handle->available;
}

void superpoint_coreml_destroy(SuperPointCoreMLHandle handle) {
    if (handle) {
        handle->model = nil;
        delete handle;
    }
}

// ── Feature extraction ──

SuperPointFeatures superpoint_coreml_extract(SuperPointCoreMLHandle handle,
                                              CVPixelBufferRef pixel_buffer,
                                              int max_keypoints,
                                              float score_threshold) {
    SuperPointFeatures result;
    if (!handle || !handle->available || !pixel_buffer) return result;

    @autoreleasepool {
        int src_w = (int)CVPixelBufferGetWidth(pixel_buffer);
        int src_h = (int)CVPixelBufferGetHeight(pixel_buffer);
        result.image_width = src_w;
        result.image_height = src_h;

        int model_h = handle->input_h;
        int model_w = handle->input_w;
        int cell = handle->cell;
        int hc = model_h / cell; // grid height (e.g. 60)
        int wc = model_w / cell; // grid width  (e.g. 80)

        // Step 1: Resize BGRA CVPixelBuffer to model input size using vImage
        CVPixelBufferRef resized = NULL;
        if (src_w != model_w || src_h != model_h) {
            CVReturn cr = CVPixelBufferCreate(kCFAllocatorDefault, model_w, model_h,
                                               kCVPixelFormatType_32BGRA, NULL, &resized);
            if (cr != kCVReturnSuccess) return result;

            CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
            CVPixelBufferLockBaseAddress(resized, 0);

            vImage_Buffer src_buf = {
                CVPixelBufferGetBaseAddress(pixel_buffer),
                (vImagePixelCount)src_h, (vImagePixelCount)src_w,
                CVPixelBufferGetBytesPerRow(pixel_buffer)};
            vImage_Buffer dst_buf = {
                CVPixelBufferGetBaseAddress(resized),
                (vImagePixelCount)model_h, (vImagePixelCount)model_w,
                CVPixelBufferGetBytesPerRow(resized)};

            vImageScale_ARGB8888(&src_buf, &dst_buf, NULL, kvImageNoFlags);

            CVPixelBufferUnlockBaseAddress(resized, 0);
            CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
        } else {
            resized = (CVPixelBufferRef)CFRetain(pixel_buffer);
        }

        // Step 2: Convert BGRA → grayscale float [0,1] → MLMultiArray [1,1,H,W]
        int n = model_h * model_w;
        NSArray *shape = @[@1, @1, @(model_h), @(model_w)];
        NSError *err = nil;
        MLMultiArray *input_arr = [[MLMultiArray alloc] initWithShape:shape
                                                            dataType:MLMultiArrayDataTypeFloat32
                                                               error:&err];
        if (!input_arr) {
            if (resized != pixel_buffer) CFRelease(resized);
            return result;
        }

        CVPixelBufferLockBaseAddress(resized, kCVPixelBufferLock_ReadOnly);
        const uint8_t *bgra = (const uint8_t *)CVPixelBufferGetBaseAddress(resized);
        size_t stride = CVPixelBufferGetBytesPerRow(resized);
        float *dst = (float *)input_arr.dataPointer;

        // Extract grayscale: Y = 0.299*R + 0.587*G + 0.114*B (BGRA layout)
        // Deinterleave to planar, then use vDSP for speed
        std::vector<uint8_t> gray_u8(n);
        for (int y = 0; y < model_h; y++) {
            const uint8_t *row = bgra + y * stride;
            for (int x = 0; x < model_w; x++) {
                int b = row[x * 4 + 0];
                int g = row[x * 4 + 1];
                int r = row[x * 4 + 2];
                gray_u8[y * model_w + x] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
            }
        }
        CVPixelBufferUnlockBaseAddress(resized, kCVPixelBufferLock_ReadOnly);
        if (resized != pixel_buffer) CFRelease(resized);

        // uint8 → float32 [0,1] via vDSP
        vDSP_vfltu8(gray_u8.data(), 1, dst, 1, (vDSP_Length)n);
        float scale = 1.0f / 255.0f;
        vDSP_vsmul(dst, 1, &scale, dst, 1, (vDSP_Length)n);

        // Step 3: CoreML inference
        NSString *input_name = @"image";
        MLFeatureValue *fv = [MLFeatureValue featureValueWithMultiArray:input_arr];
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{input_name: fv}
                                                              error:&err];
        if (!provider) return result;

        id<MLFeatureProvider> output = [handle->model predictionFromFeatures:provider
                                                                       error:&err];
        if (!output) {
            fprintf(stderr, "[SuperPointCoreML] Inference failed: %s\n",
                    err.localizedDescription.UTF8String);
            return result;
        }

        // Get output arrays
        // semi: [1, 65, hc, wc], desc: [1, 256, hc, wc]
        NSArray *output_names = [output featureNames].allObjects;
        MLMultiArray *semi_arr = nil;
        MLMultiArray *desc_arr = nil;

        for (NSString *name in output_names) {
            MLMultiArray *arr = [output featureValueForName:name].multiArrayValue;
            if (!arr) continue;
            // Identify by shape: 65-channel = semi, 256-channel = desc
            if (arr.shape.count >= 4) {
                int ch = [arr.shape[1] intValue];
                if (ch == 65) semi_arr = arr;
                else if (ch == 256) desc_arr = arr;
            }
        }

        if (!semi_arr || !desc_arr) {
            fprintf(stderr, "[SuperPointCoreML] Missing semi or desc output\n");
            return result;
        }

        // Step 4: Post-process semi → score map → NMS → top-K keypoints
        // semi shape: [1, 65, hc, wc] — 64 spatial bins + 1 dustbin
        int semi_total = 65 * hc * wc;
        std::vector<float> semi_f32(semi_total);

        // Convert to float32 if needed
        if (semi_arr.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16 *fp16 = (const __fp16 *)semi_arr.dataPointer;
            vImage_Buffer src_buf = {(void *)fp16, 1, (vImagePixelCount)semi_total,
                                     semi_total * sizeof(__fp16)};
            vImage_Buffer dst_buf = {semi_f32.data(), 1, (vImagePixelCount)semi_total,
                                     semi_total * sizeof(float)};
            vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);
        } else {
            const float *src = (const float *)semi_arr.dataPointer;
            memcpy(semi_f32.data(), src, semi_total * sizeof(float));
        }

        // Softmax over channel dimension (65) for each spatial location
        // Then take max over first 64 channels (dustbin = channel 64)
        int grid_n = hc * wc;
        std::vector<float> score_map(grid_n);
        std::vector<int> best_bin(grid_n);

        for (int i = 0; i < grid_n; i++) {
            // Gather 65 values for this grid cell
            float max_val = -1e9f;
            for (int c = 0; c < 65; c++) {
                float v = semi_f32[c * grid_n + i]; // CHW layout
                if (v > max_val) max_val = v;
            }
            // Softmax
            float exp_sum = 0;
            float best_score = 0;
            int best_c = 0;
            for (int c = 0; c < 65; c++) {
                float ev = expf(semi_f32[c * grid_n + i] - max_val);
                if (c < 64) {
                    if (ev > best_score) { best_score = ev; best_c = c; }
                }
                exp_sum += ev;
            }
            score_map[i] = best_score / exp_sum;
            best_bin[i] = best_c;
        }

        // NMS: suppress non-maxima in 3x3 grid neighborhoods
        std::vector<bool> is_max(grid_n, true);
        for (int gy = 0; gy < hc; gy++) {
            for (int gx = 0; gx < wc; gx++) {
                int idx = gy * wc + gx;
                float s = score_map[idx];
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dy == 0 && dx == 0) continue;
                        int ny = gy + dy, nx = gx + dx;
                        if (ny < 0 || ny >= hc || nx < 0 || nx >= wc) continue;
                        if (score_map[ny * wc + nx] >= s && (ny * wc + nx) < idx) {
                            is_max[idx] = false;
                            break;
                        }
                        if (score_map[ny * wc + nx] > s) {
                            is_max[idx] = false;
                            break;
                        }
                    }
                    if (!is_max[idx]) break;
                }
            }
        }

        // Collect candidates above threshold
        struct Candidate { float score; int gx, gy, bin; };
        std::vector<Candidate> candidates;
        candidates.reserve(grid_n / 4);

        for (int gy = 0; gy < hc; gy++) {
            for (int gx = 0; gx < wc; gx++) {
                int idx = gy * wc + gx;
                if (!is_max[idx]) continue;
                if (score_map[idx] < score_threshold) continue;
                candidates.push_back({score_map[idx], gx, gy, best_bin[idx]});
            }
        }

        // Sort by score descending, take top-K
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &a, const Candidate &b) {
                      return a.score > b.score;
                  });
        int num_kpts = std::min((int)candidates.size(), max_keypoints);

        // Convert grid coordinates to pixel coordinates in model input space
        // bin layout: 8x8 grid within each cell, flattened row-major (0..63)
        // pixel = cell_origin + sub-pixel offset
        float scale_x = (float)src_w / model_w;
        float scale_y = (float)src_h / model_h;

        result.keypoints.resize(num_kpts);
        result.scores.resize(num_kpts);

        for (int i = 0; i < num_kpts; i++) {
            const auto &c = candidates[i];
            int sub_y = c.bin / cell;
            int sub_x = c.bin % cell;
            // Pixel in model-input space
            float px = c.gx * cell + sub_x + 0.5f;
            float py = c.gy * cell + sub_y + 0.5f;
            // Scale back to original image space
            result.keypoints[i] = Eigen::Vector2d(px * scale_x, py * scale_y);
            result.scores[i] = c.score;
        }

        // Step 5: Bilinear interpolate descriptors at keypoint locations
        // desc shape: [1, 256, hc, wc] — sample at grid positions
        int desc_total = 256 * grid_n;
        std::vector<float> desc_f32(desc_total);

        if (desc_arr.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16 *fp16 = (const __fp16 *)desc_arr.dataPointer;
            vImage_Buffer src_buf = {(void *)fp16, 1, (vImagePixelCount)desc_total,
                                     desc_total * sizeof(__fp16)};
            vImage_Buffer dst_buf = {desc_f32.data(), 1, (vImagePixelCount)desc_total,
                                     desc_total * sizeof(float)};
            vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);
        } else {
            const float *src = (const float *)desc_arr.dataPointer;
            memcpy(desc_f32.data(), src, desc_total * sizeof(float));
        }

        result.descriptors_flat.resize(num_kpts * 256);
        result.num_keypoints = num_kpts;

        for (int i = 0; i < num_kpts; i++) {
            // Map keypoint to descriptor grid coordinates
            // Keypoint is in original image space; convert to model space first
            float mx = (float)result.keypoints[i].x() / scale_x;
            float my = (float)result.keypoints[i].y() / scale_y;
            // Grid coordinate (center of each cell is at gx+0.5*cell, gy+0.5*cell)
            float gx = mx / cell - 0.5f;
            float gy = my / cell - 0.5f;

            // Bilinear interpolation
            int gx0 = std::max(0, (int)floorf(gx));
            int gy0 = std::max(0, (int)floorf(gy));
            int gx1 = std::min(wc - 1, gx0 + 1);
            int gy1 = std::min(hc - 1, gy0 + 1);
            float fx = gx - gx0;
            float fy = gy - gy0;

            float w00 = (1 - fx) * (1 - fy);
            float w01 = (1 - fx) * fy;
            float w10 = fx * (1 - fy);
            float w11 = fx * fy;

            float *out = &result.descriptors_flat[i * 256];
            for (int d = 0; d < 256; d++) {
                // desc_f32 layout: [256, hc, wc] (CHW)
                int base = d * grid_n;
                out[d] = w00 * desc_f32[base + gy0 * wc + gx0]
                       + w01 * desc_f32[base + gy1 * wc + gx0]
                       + w10 * desc_f32[base + gy0 * wc + gx1]
                       + w11 * desc_f32[base + gy1 * wc + gx1];
            }

            // L2-normalize descriptor
            float norm_sq = 0;
            for (int d = 0; d < 256; d++) norm_sq += out[d] * out[d];
            float inv_norm = (norm_sq > 1e-12f) ? 1.0f / sqrtf(norm_sq) : 0.0f;
            vDSP_vsmul(out, 1, &inv_norm, out, 1, 256);
        }

    } // @autoreleasepool
    return result;
}
