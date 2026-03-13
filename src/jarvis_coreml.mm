// jarvis_coreml.mm — CoreML implementation for JARVIS inference
//
// Zero-copy CVPixelBuffer path: VideoToolbox → IOSurface → CoreML → ANE/GPU

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include "jarvis_coreml.h"
#include <filesystem>

// ── Helpers ──

static MLModel *load_mlpackage(const std::string &path, std::string &err) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:path.c_str()]];

        // Compile .mlpackage to temporary .mlmodelc
        NSError *error = nil;
        NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
        if (!compiled) {
            err = "Compile failed: " +
                  std::string(error.localizedDescription.UTF8String);
            return nil;
        }

        // Use CPU + ANE only — avoids GPU contention with Metal rendering
        // pipeline which causes bimodal performance (300ms vs 1200ms).
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

// Create a resized CVPixelBuffer using vImage (Accelerate, NEON-optimized)
// Convert a BGRA CVPixelBuffer to an ImageNet-normalized CHW float MLMultiArray.
// JARVIS training normalizes: (pixel/255 - mean) / std with ImageNet values.
// CoreML's ImageType only does scale=1/255 (no normalization), so we do it manually.
// Uses vDSP for vectorized float conversion + normalization (~10x faster than scalar).
static MLMultiArray *pixelbuf_to_normalized_array(CVPixelBufferRef pb, int w, int h) {
    // ImageNet normalization: scale = 1/(255*std), offset = -mean/std
    // Combined: output = pixel * scale + offset
    static const float scale[3] = {
        1.0f / (255.0f * 0.229f),  // R
        1.0f / (255.0f * 0.224f),  // G
        1.0f / (255.0f * 0.225f),  // B
    };
    static const float offset[3] = {
        -0.485f / 0.229f,  // R
        -0.456f / 0.224f,  // G
        -0.406f / 0.225f,  // B
    };

    int n = w * h;
    NSArray *shape = @[@1, @3, @(h), @(w)];
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                  dataType:MLMultiArrayDataTypeFloat32
                                                     error:&err];
    if (!arr) return nil;

    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    uint8_t *src = (uint8_t *)CVPixelBufferGetBaseAddress(pb);
    size_t stride = CVPixelBufferGetBytesPerRow(pb);
    float *dst = (float *)arr.dataPointer;

    // Step 1: Deinterleave BGRA rows into planar uint8 B, G, R buffers.
    // Handle stride padding by processing row-by-row.
    std::vector<uint8_t> planar(n * 3); // B, G, R planes
    uint8_t *pB = planar.data();
    uint8_t *pG = pB + n;
    uint8_t *pR = pG + n;

    for (int y = 0; y < h; y++) {
        const uint8_t *row = src + y * stride;
        int off = y * w;
        for (int x = 0; x < w; x++) {
            pB[off + x] = row[x * 4 + 0];
            pG[off + x] = row[x * 4 + 1];
            pR[off + x] = row[x * 4 + 2];
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

    // Step 2: Convert uint8 → float and normalize using vDSP.
    // vDSP_vfltu8: uint8 → float,  vDSP_vsmsa: x * scale + offset
    float *dstR = dst;              // Channel 0 = R
    float *dstG = dst + n;          // Channel 1 = G
    float *dstB = dst + 2 * n;      // Channel 2 = B

    vDSP_vfltu8(pR, 1, dstR, 1, (vDSP_Length)n);
    vDSP_vsmsa(dstR, 1, &scale[0], &offset[0], dstR, 1, (vDSP_Length)n);

    vDSP_vfltu8(pG, 1, dstG, 1, (vDSP_Length)n);
    vDSP_vsmsa(dstG, 1, &scale[1], &offset[1], dstG, 1, (vDSP_Length)n);

    vDSP_vfltu8(pB, 1, dstB, 1, (vDSP_Length)n);
    vDSP_vsmsa(dstB, 1, &scale[2], &offset[2], dstB, 1, (vDSP_Length)n);

    return arr;
}

// Resize a CVPixelBuffer to dst_w x dst_h using vImage (squish, matching JARVIS).
static CVPixelBufferRef resize_pixelbuf(CVPixelBufferRef src, int dst_w, int dst_h) {
    int src_w = (int)CVPixelBufferGetWidth(src);
    int src_h = (int)CVPixelBufferGetHeight(src);
    size_t src_stride = CVPixelBufferGetBytesPerRow(src);

    CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    uint8_t *src_data = (uint8_t *)CVPixelBufferGetBaseAddress(src);

    vImage_Buffer src_buf = {src_data, (vImagePixelCount)src_h,
                             (vImagePixelCount)src_w, src_stride};

    CVPixelBufferRef dst = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, dst_w, dst_h,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    if (status != kCVReturnSuccess || !dst) {
        CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
        return NULL;
    }
    CVPixelBufferLockBaseAddress(dst, 0);

    uint8_t *dst_data = (uint8_t *)CVPixelBufferGetBaseAddress(dst);
    size_t dst_stride = CVPixelBufferGetBytesPerRow(dst);
    vImage_Buffer dst_buf = {dst_data, (vImagePixelCount)dst_h,
                             (vImagePixelCount)dst_w, dst_stride};

    vImageScale_ARGB8888(&src_buf, &dst_buf, NULL, kvImageNoFlags);

    CVPixelBufferUnlockBaseAddress(dst, 0);
    CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    return dst;
}

// Crop a region from a CVPixelBuffer using vImage
static CVPixelBufferRef crop_pixelbuf(CVPixelBufferRef src,
                                       int cx, int cy, int crop_size) {
    int src_w = (int)CVPixelBufferGetWidth(src);
    int src_h = (int)CVPixelBufferGetHeight(src);
    int half = crop_size / 2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(src_w, cx + half);
    int y1 = std::min(src_h, cy + half);
    int cw = x1 - x0, ch = y1 - y0;

    CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    uint8_t *src_data = (uint8_t *)CVPixelBufferGetBaseAddress(src);
    size_t src_stride = CVPixelBufferGetBytesPerRow(src);

    // Create crop destination
    CVPixelBufferRef dst = NULL;
    CVReturn cr_status = CVPixelBufferCreate(kCFAllocatorDefault, cw, ch,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    if (cr_status != kCVReturnSuccess || !dst) {
        CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
        return NULL;
    }
    CVPixelBufferLockBaseAddress(dst, 0);
    uint8_t *dst_data = (uint8_t *)CVPixelBufferGetBaseAddress(dst);
    size_t dst_stride = CVPixelBufferGetBytesPerRow(dst);

    // Copy rows with vImage
    vImage_Buffer src_buf = {src_data + y0 * src_stride + x0 * 4,
                             (vImagePixelCount)ch, (vImagePixelCount)cw, src_stride};
    vImage_Buffer dst_buf = {dst_data, (vImagePixelCount)ch,
                             (vImagePixelCount)cw, dst_stride};
    vImageCopyBuffer(&src_buf, &dst_buf, 4, kvImageNoFlags);

    CVPixelBufferUnlockBaseAddress(dst, 0);
    CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    return dst;
}

// Extract heatmap argmax from MLMultiArray
struct HMPeak { float x, y, confidence; };

static HMPeak heatmap_argmax(MLMultiArray *hm, int channel, int hm_h, int hm_w) {
    // hm shape: [1, C, H, W] — access channel-th plane
    const void *raw = hm.dataPointer;
    vDSP_Length n = (vDSP_Length)(hm_h * hm_w);
    float max_val = -1e9f;
    vDSP_Length max_idx = 0;

    if (hm.dataType == MLMultiArrayDataTypeFloat16) {
        // Convert float16 plane to float32, then use vDSP_maxvi
        const __fp16 *fp16 = (const __fp16 *)raw + channel * n;
        // Use Accelerate vImageConvert for NEON-accelerated fp16→fp32
        vImage_Buffer src_buf = {(void *)fp16, 1, n, n * sizeof(__fp16)};
        std::vector<float> fp32(n);
        vImage_Buffer dst_buf = {fp32.data(), 1, n, n * sizeof(float)};
        vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, kvImageNoFlags);
        vDSP_maxvi(fp32.data(), 1, &max_val, &max_idx, n);
    } else {
        const float *data = (const float *)raw + channel * n;
        vDSP_maxvi(data, 1, &max_val, &max_idx, n);
    }

    HMPeak peak;
    peak.x = (float)(max_idx % hm_w) * 2.0f; // stride-2 heatmap
    peak.y = (float)(max_idx / hm_w) * 2.0f;
    peak.confidence = std::min(max_val, 255.0f) / 255.0f;
    return peak;
}

// ── Public API ──

bool jarvis_coreml_available() {
    if (@available(macOS 13.0, *)) return true;
    return false;
}

bool jarvis_coreml_init(JarvisCoreMLState &s, const std::string &model_dir,
                         const JarvisModelConfig &cfg) {
    s.available = jarvis_coreml_available();
    if (!s.available) {
        s.status = "CoreML requires macOS 13+";
        return false;
    }

    // Release any previously loaded models (prevents pointer-overwrite leak)
    jarvis_coreml_cleanup(s);

    // Store full config for display, and flat copies for inference hot path
    s.config = cfg;
    s.center_input_size = cfg.center_input_size;
    s.keypoint_input_size = cfg.keypoint_input_size;
    s.num_joints = cfg.num_joints;

    // Load models
    std::string cd_path = model_dir + "/center_detect.mlpackage";
    std::string kd_path = model_dir + "/keypoint_detect.mlpackage";

    if (!std::filesystem::exists(cd_path) || !std::filesystem::exists(kd_path)) {
        s.status = "CoreML .mlpackage files not found";
        return false;
    }

    s.status = "Compiling CenterDetect (first time may take ~15s)...";
    std::string err;
    MLModel *cd = load_mlpackage(cd_path, err);
    if (!cd) { s.status = err; return false; }

    s.status = "Compiling KeypointDetect...";
    MLModel *kd = load_mlpackage(kd_path, err);
    if (!kd) { s.status = err; return false; }

    s.center_model = (__bridge_retained void *)cd;
    s.keypoint_model = (__bridge_retained void *)kd;
    s.loaded = true;
    s.status = "CoreML loaded (" + std::to_string(s.num_joints) + " joints, GPU/ANE)";
    return true;
}

bool jarvis_coreml_predict_frame(
    JarvisCoreMLState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pixel_buffers,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &/*skeleton*/,
    int num_cameras,
    float confidence_threshold) {

    if (!s.loaded) { s.status = "Not loaded"; return false; }

    {
        auto t0 = std::chrono::steady_clock::now();
        int num_cams = (int)pixel_buffers.size();
        int num_joints = s.num_joints;

        auto &fa = get_or_create_frame(amap, frame_num,
                        num_joints, num_cams);

        MLModel *cd_model = (__bridge MLModel *)s.center_model;
        MLModel *kd_model = (__bridge MLModel *)s.keypoint_model;

        // Helper: find the largest 4D heatmap output from a CoreML prediction
        auto find_heatmap = [](id<MLFeatureProvider> output) -> MLMultiArray * {
            MLMultiArray *best = nil;
            for (NSString *name in output.featureNames) {
                MLFeatureValue *fv = [output featureValueForName:name];
                if (fv.multiArrayValue && fv.multiArrayValue.shape.count == 4) {
                    MLMultiArray *arr = fv.multiArrayValue;
                    int h = [arr.shape[2] intValue];
                    if (!best || h > [best.shape[2] intValue])
                        best = arr;
                }
            }
            return best;
        };

        // Per-camera pipeline: CenterDetect → crop → KeypointDetect
        // Processing each camera fully before moving to the next improves
        // cache locality and enables early exit when center confidence is low.
        float center_ms_total = 0, kp_ms_total = 0;

        for (int c = 0; c < num_cams; ++c) {
            @autoreleasepool {
            if (!pixel_buffers[c]) continue;

            // --- CenterDetect ---
            auto tc0 = std::chrono::steady_clock::now();
            int sz = s.center_input_size;
            CVPixelBufferRef resized = resize_pixelbuf(pixel_buffers[c], sz, sz);
            if (!resized) continue;

            // Convert to ImageNet-normalized float tensor (matching JARVIS training)
            MLMultiArray *cd_tensor = pixelbuf_to_normalized_array(resized, sz, sz);
            CVPixelBufferRelease(resized);
            if (!cd_tensor) continue;

            NSError *error = nil;
            id<MLFeatureProvider> cd_input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                    @{@"image": [MLFeatureValue featureValueWithMultiArray:cd_tensor]}
                    error:&error];
            if (!cd_input || error) continue;
            error = nil;
            id<MLFeatureProvider> cd_output = [cd_model predictionFromFeatures:cd_input error:&error];
            if (!cd_output || error) continue;

            MLMultiArray *cd_hm = find_heatmap(cd_output);
            if (!cd_hm) continue;

            int cd_hm_h = [cd_hm.shape[2] intValue];
            int cd_hm_w = [cd_hm.shape[3] intValue];
            // Non-uniform downsampling scale (squish resize, matching JARVIS)
            float ds_x = (float)cam_widths[c] / sz;
            float ds_y = (float)cam_heights[c] / sz;

            HMPeak center = heatmap_argmax(cd_hm, 0, cd_hm_h, cd_hm_w);
            center.x *= ds_x;
            center.y *= ds_y;
            auto tc1 = std::chrono::steady_clock::now();
            center_ms_total += std::chrono::duration<float, std::milli>(tc1 - tc0).count();

            // Skip KeypointDetect if center confidence is too low
            if (center.confidence < confidence_threshold) continue;

            // --- KeypointDetect ---
            auto tk0 = std::chrono::steady_clock::now();
            int bbox_size = s.keypoint_input_size;
            int half = bbox_size / 2;
            int cx = std::clamp((int)center.x, half, cam_widths[c] - half);
            int cy = std::clamp((int)center.y, half, cam_heights[c] - half);

            CVPixelBufferRef crop = crop_pixelbuf(pixel_buffers[c], cx, cy, bbox_size);
            if (!crop) continue;

            // Normalize crop with ImageNet mean/std (matching JARVIS training)
            MLMultiArray *kd_tensor = pixelbuf_to_normalized_array(crop, bbox_size, bbox_size);
            CVPixelBufferRelease(crop);
            if (!kd_tensor) continue;

            error = nil;
            id<MLFeatureProvider> kd_input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                    @{@"image": [MLFeatureValue featureValueWithMultiArray:kd_tensor]}
                    error:&error];
            if (!kd_input || error) continue;
            error = nil;
            id<MLFeatureProvider> kd_output = [kd_model predictionFromFeatures:kd_input error:&error];
            if (!kd_output || error) continue;

            MLMultiArray *kd_hm = find_heatmap(kd_output);
            if (!kd_hm) continue;

            int kd_hm_h = [kd_hm.shape[2] intValue];
            int kd_hm_w = [kd_hm.shape[3] intValue];
            int n_joints = std::min([kd_hm.shape[1] intValue], (int)fa.cameras[c].keypoints.size());

            for (int k = 0; k < n_joints; ++k) {
                HMPeak peak = heatmap_argmax(kd_hm, k, kd_hm_h, kd_hm_w);
                float img_x = peak.x + (float)(cx - half);
                float img_y = peak.y + (float)(cy - half);

                auto &kp = fa.cameras[c].keypoints[k];
                kp.x = img_x;
                kp.y = cam_heights[c] - img_y; // image → ImPlot
                kp.labeled = (peak.confidence >= confidence_threshold);
                kp.confidence = peak.confidence;
                kp.source = LabelSource::Predicted;
            }
            auto tk1 = std::chrono::steady_clock::now();
            kp_ms_total += std::chrono::duration<float, std::milli>(tk1 - tk0).count();
            } // @autoreleasepool per camera
        }
        s.last_center_ms = center_ms_total;
        s.last_keypoint_ms = kp_ms_total;

        auto t4 = std::chrono::steady_clock::now();
        s.last_total_ms = std::chrono::duration<float, std::milli>(t4 - t0).count();
        s.status = "Predicted " + std::to_string(num_joints) + " joints on " +
                   std::to_string(num_cams) + " cameras in " +
                   std::to_string((int)s.last_total_ms) + " ms (CoreML)";
        return true;
    }
}

// Create a normalized CHW float MLMultiArray directly from RGB24 data at target size.
// Resizes using vImage bilinear interpolation, then normalizes with vDSP.
static MLMultiArray *rgb24_to_normalized_array(const uint8_t *rgb, int src_w, int src_h,
                                                int dst_w, int dst_h) {
    static const float scale[3] = {
        1.0f / (255.0f * 0.229f), 1.0f / (255.0f * 0.224f), 1.0f / (255.0f * 0.225f)
    };
    static const float offset[3] = {
        -0.485f / 0.229f, -0.456f / 0.224f, -0.406f / 0.225f
    };

    int n = dst_w * dst_h;
    NSArray *shape = @[@1, @3, @(dst_h), @(dst_w)];
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                  dataType:MLMultiArrayDataTypeFloat32
                                                     error:&err];
    if (!arr) return nil;

    // Resize RGB24 using vImage: first create a planar-8 ARGB buffer, resize, extract channels
    // Simpler approach: resize each RGB channel separately as Planar8
    // Even simpler: resize the interleaved RGB24 by treating as 3-channel, then deinterleave
    // Simplest: bilinear sample directly into the output array
    float *dst = (float *)arr.dataPointer;
    float *dstR = dst;
    float *dstG = dst + n;
    float *dstB = dst + 2 * n;

    float sx = (float)src_w / dst_w;
    float sy = (float)src_h / dst_h;
    int src_stride = src_w * 3;

    for (int y = 0; y < dst_h; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = std::max(0, (int)fy);
        int y1 = std::min(src_h - 1, y0 + 1);
        float wy = fy - y0;

        for (int x = 0; x < dst_w; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = std::max(0, (int)fx);
            int x1 = std::min(src_w - 1, x0 + 1);
            float wx = fx - x0;

            // Bilinear interpolation for R, G, B
            const uint8_t *p00 = rgb + y0 * src_stride + x0 * 3;
            const uint8_t *p10 = rgb + y0 * src_stride + x1 * 3;
            const uint8_t *p01 = rgb + y1 * src_stride + x0 * 3;
            const uint8_t *p11 = rgb + y1 * src_stride + x1 * 3;

            float w00 = (1-wx)*(1-wy), w10 = wx*(1-wy), w01 = (1-wx)*wy, w11 = wx*wy;
            int idx = y * dst_w + x;

            for (int ch = 0; ch < 3; ch++) {
                float v = w00*p00[ch] + w10*p10[ch] + w01*p01[ch] + w11*p11[ch];
                float *plane = (ch == 0) ? dstR : (ch == 1) ? dstG : dstB;
                plane[idx] = v * scale[ch] + offset[ch];
            }
        }
    }
    return arr;
}

// Crop a region from RGB24 data and create a normalized MLMultiArray
static MLMultiArray *rgb24_crop_to_normalized_array(const uint8_t *rgb, int src_w, int src_h,
                                                     int cx, int cy, int crop_size, int dst_size) {
    static const float scale[3] = {
        1.0f / (255.0f * 0.229f), 1.0f / (255.0f * 0.224f), 1.0f / (255.0f * 0.225f)
    };
    static const float offset[3] = {
        -0.485f / 0.229f, -0.456f / 0.224f, -0.406f / 0.225f
    };

    int half = crop_size / 2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(src_w, cx + half);
    int y1 = std::min(src_h, cy + half);
    int cw = x1 - x0, ch = y1 - y0;
    if (cw <= 0 || ch <= 0) return nil;

    int n = dst_size * dst_size;
    NSArray *shape = @[@1, @3, @(dst_size), @(dst_size)];
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                  dataType:MLMultiArrayDataTypeFloat32
                                                     error:&err];
    if (!arr) return nil;

    float *dst = (float *)arr.dataPointer;
    float *dstR = dst;
    float *dstG = dst + n;
    float *dstB = dst + 2 * n;

    float sx = (float)cw / dst_size;
    float sy = (float)ch / dst_size;
    int src_stride = src_w * 3;

    for (int y = 0; y < dst_size; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int iy0 = std::max(0, (int)fy);
        int iy1 = std::min(ch - 1, iy0 + 1);
        float wy = fy - iy0;

        for (int x = 0; x < dst_size; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int ix0 = std::max(0, (int)fx);
            int ix1 = std::min(cw - 1, ix0 + 1);
            float wx = fx - ix0;

            const uint8_t *p00 = rgb + (y0+iy0)*src_stride + (x0+ix0)*3;
            const uint8_t *p10 = rgb + (y0+iy0)*src_stride + (x0+ix1)*3;
            const uint8_t *p01 = rgb + (y0+iy1)*src_stride + (x0+ix0)*3;
            const uint8_t *p11 = rgb + (y0+iy1)*src_stride + (x0+ix1)*3;

            float w00 = (1-wx)*(1-wy), w10 = wx*(1-wy), w01 = (1-wx)*wy, w11 = wx*wy;
            int idx = y * dst_size + x;

            for (int c = 0; c < 3; c++) {
                float v = w00*p00[c] + w10*p10[c] + w01*p01[c] + w11*p11[c];
                float *plane = (c == 0) ? dstR : (c == 1) ? dstG : dstB;
                plane[idx] = v * scale[c] + offset[c];
            }
        }
    }
    return arr;
}

bool jarvis_coreml_predict_frame_rgb(
    JarvisCoreMLState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<const uint8_t *> &rgb_buffers,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &/*skeleton*/,
    int num_cameras,
    float confidence_threshold) {

    if (!s.loaded) { s.status = "Not loaded"; return false; }

    auto t0 = std::chrono::steady_clock::now();
    int num_cams = (int)rgb_buffers.size();
    int num_joints = s.num_joints;
    auto &fa = get_or_create_frame(amap, frame_num, num_joints, num_cams);

    MLModel *cd_model = (__bridge MLModel *)s.center_model;
    MLModel *kd_model = (__bridge MLModel *)s.keypoint_model;

    auto find_heatmap = [](id<MLFeatureProvider> output) -> MLMultiArray * {
        MLMultiArray *best = nil;
        for (NSString *name in output.featureNames) {
            MLFeatureValue *fv = [output featureValueForName:name];
            if (fv.multiArrayValue && fv.multiArrayValue.shape.count == 4) {
                MLMultiArray *arr = fv.multiArrayValue;
                int h = [arr.shape[2] intValue];
                if (!best || h > [best.shape[2] intValue])
                    best = arr;
            }
        }
        return best;
    };

    for (int c = 0; c < num_cams; ++c) {
        @autoreleasepool {
        if (!rgb_buffers[c]) continue;
        int w = cam_widths[c], h = cam_heights[c];

        // CenterDetect: resize RGB directly to 320×320 normalized tensor
        int sz = s.center_input_size;
        MLMultiArray *cd_tensor = rgb24_to_normalized_array(rgb_buffers[c], w, h, sz, sz);
        if (!cd_tensor) continue;

        NSError *error = nil;
        id<MLFeatureProvider> cd_input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{@"image": [MLFeatureValue featureValueWithMultiArray:cd_tensor]}
                error:&error];
        if (!cd_input || error) continue;
        id<MLFeatureProvider> cd_output = [cd_model predictionFromFeatures:cd_input error:&error];
        if (!cd_output || error) continue;

        MLMultiArray *cd_hm = find_heatmap(cd_output);
        if (!cd_hm) continue;

        int cd_hm_h = [cd_hm.shape[2] intValue];
        int cd_hm_w = [cd_hm.shape[3] intValue];
        float ds_x = (float)w / sz;
        float ds_y = (float)h / sz;

        HMPeak center = heatmap_argmax(cd_hm, 0, cd_hm_h, cd_hm_w);
        center.x *= ds_x;
        center.y *= ds_y;
        if (center.confidence < confidence_threshold) continue;

        // KeypointDetect: crop + resize from RGB directly
        int bbox_size = s.keypoint_input_size;
        int half = bbox_size / 2;
        int cx = std::clamp((int)center.x, half, w - half);
        int cy = std::clamp((int)center.y, half, h - half);

        MLMultiArray *kd_tensor = rgb24_crop_to_normalized_array(
            rgb_buffers[c], w, h, cx, cy, bbox_size, bbox_size);
        if (!kd_tensor) continue;

        error = nil;
        id<MLFeatureProvider> kd_input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{@"image": [MLFeatureValue featureValueWithMultiArray:kd_tensor]}
                error:&error];
        if (!kd_input || error) continue;
        id<MLFeatureProvider> kd_output = [kd_model predictionFromFeatures:kd_input error:&error];
        if (!kd_output || error) continue;

        MLMultiArray *kd_hm = find_heatmap(kd_output);
        if (!kd_hm) continue;

        int kd_hm_h = [kd_hm.shape[2] intValue];
        int kd_hm_w = [kd_hm.shape[3] intValue];
        int n_joints = std::min([kd_hm.shape[1] intValue], (int)fa.cameras[c].keypoints.size());

        for (int k = 0; k < n_joints; ++k) {
            HMPeak peak = heatmap_argmax(kd_hm, k, kd_hm_h, kd_hm_w);
            float img_x = peak.x + (float)(cx - half);
            float img_y = peak.y + (float)(cy - half);

            auto &kp = fa.cameras[c].keypoints[k];
            kp.x = img_x;
            kp.y = h - img_y;  // image → ImPlot
            kp.labeled = (peak.confidence >= confidence_threshold);
            kp.confidence = peak.confidence;
            kp.source = LabelSource::Predicted;
        }
        } // @autoreleasepool
    }

    auto t1 = std::chrono::steady_clock::now();
    s.last_total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    s.status = "Predicted " + std::to_string(num_joints) + " joints on " +
               std::to_string(num_cams) + " cameras in " +
               std::to_string((int)s.last_total_ms) + " ms (CoreML/RGB)";
    return true;
}

void jarvis_coreml_cleanup(JarvisCoreMLState &s) {
    if (s.center_model) {
        CFRelease(s.center_model);
        s.center_model = nullptr;
    }
    if (s.keypoint_model) {
        CFRelease(s.keypoint_model);
        s.keypoint_model = nullptr;
    }
    s.loaded = false;
}
