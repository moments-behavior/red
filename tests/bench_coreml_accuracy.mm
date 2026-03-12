// tests/bench_coreml_accuracy.mm — Compare ONNX (ground truth) vs CoreML predictions
//
// Loads a test JPEG, runs CenterDetect + KeypointDetect through both:
//   1. ONNX Runtime CPU (with correct ImageNet normalization in preprocessing)
//   2. Native CoreML .mlpackage (normalization baked into model or not)
//
// Reports: center detection agreement, per-joint keypoint error, confidence
// correlation, and timing. Run BEFORE and AFTER re-exporting .mlpackage
// with fixed normalization to see the impact.

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

#ifdef RED_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const char *MODEL_DIR =
    "/Users/johnsonr/red_projects/hurdles_rat24a/jarvis_models/mouseJan30";
static const char *TEST_JPEG =
    "/Users/johnsonr/red_projects/annotate1/export/"
    "2026_03_10_08_45_34/train/2026_03_10_08_41_51/Cam2002486/Frame_5998.jpg";

static constexpr int CENTER_SIZE = 320;
static constexpr int KP_SIZE = 704;

// ---------------------------------------------------------------------------
// ONNX preprocessing (ImageNet normalization — the CORRECT reference)
// ---------------------------------------------------------------------------
namespace onnx_preprocess {

static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float INV_STD[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

inline std::vector<float> preprocess(const uint8_t *rgb, int src_w, int src_h, int dst_size) {
    std::vector<float> result(3 * dst_size * dst_size);
    float sx = (float)src_w / dst_size;
    float sy = (float)src_h / dst_size;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dst_size; ++y) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = std::max(0, (int)std::floor(fy));
            int y1 = std::min(src_h - 1, y0 + 1);
            float wy = fy - y0;
            for (int x = 0; x < dst_size; ++x) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = std::max(0, (int)std::floor(fx));
                int x1 = std::min(src_w - 1, x0 + 1);
                float wx = fx - x0;
                float v = (1 - wy) * ((1 - wx) * rgb[(y0 * src_w + x0) * 3 + c] +
                                       wx * rgb[(y0 * src_w + x1) * 3 + c]) +
                          wy * ((1 - wx) * rgb[(y1 * src_w + x0) * 3 + c] +
                                 wx * rgb[(y1 * src_w + x1) * 3 + c]);
                result[c * dst_size * dst_size + y * dst_size + x] =
                    (v / 255.0f - MEAN[c]) * INV_STD[c];
            }
        }
    }
    return result;
}

inline std::vector<float> preprocess_crop(const uint8_t *rgb, int img_w, int img_h,
                                           int cx, int cy, int crop_size, int dst_size) {
    int half = crop_size / 2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(img_w, cx + half);
    int y1 = std::min(img_h, cy + half);
    int cw = x1 - x0, ch = y1 - y0;
    std::vector<uint8_t> crop(cw * ch * 3);
    for (int y = 0; y < ch; ++y) {
        const uint8_t *src_row = rgb + ((y0 + y) * img_w + x0) * 3;
        std::memcpy(crop.data() + y * cw * 3, src_row, cw * 3);
    }
    return preprocess(crop.data(), cw, ch, dst_size);
}

struct Peak { float x, y, confidence; };

inline Peak heatmap_argmax(const float *hm, int h, int w) {
    int max_idx = 0;
    float max_val = hm[0];
    for (int i = 1; i < h * w; ++i) {
        if (hm[i] > max_val) { max_val = hm[i]; max_idx = i; }
    }
    return {(float)(max_idx % w) * 2.0f, (float)(max_idx / w) * 2.0f,
            std::min(max_val, 255.0f) / 255.0f};
}

} // namespace onnx_preprocess

// ---------------------------------------------------------------------------
// CoreML helpers
// ---------------------------------------------------------------------------
static MLModel *load_mlpackage(const std::string &path, std::string &err) {
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    NSError *error = nil;
    NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
    if (!compiled) {
        err = std::string(error.localizedDescription.UTF8String);
        return nil;
    }
    MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:config error:&error];
    if (!model) {
        err = std::string(error.localizedDescription.UTF8String);
        return nil;
    }
    CFRetain((__bridge CFTypeRef)model);
    return model;
}

// Create CVPixelBuffer from RGB data (convert RGB → BGRA for CoreML)
static CVPixelBufferRef create_bgra_pixelbuf(const uint8_t *rgb, int w, int h) {
    CVPixelBufferRef pb = NULL;
    CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                        kCVPixelFormatType_32BGRA, NULL, &pb);
    CVPixelBufferLockBaseAddress(pb, 0);
    uint8_t *dst = (uint8_t *)CVPixelBufferGetBaseAddress(pb);
    size_t stride = CVPixelBufferGetBytesPerRow(pb);
    for (int y = 0; y < h; ++y) {
        const uint8_t *src_row = rgb + y * w * 3;
        uint8_t *dst_row = dst + y * stride;
        for (int x = 0; x < w; ++x) {
            dst_row[x * 4 + 0] = src_row[x * 3 + 2]; // B
            dst_row[x * 4 + 1] = src_row[x * 3 + 1]; // G
            dst_row[x * 4 + 2] = src_row[x * 3 + 0]; // R
            dst_row[x * 4 + 3] = 255;                  // A
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
}

// Bilinear resize using vImage
static CVPixelBufferRef resize_pixelbuf(CVPixelBufferRef src, int dst_w, int dst_h) {
    int src_w = (int)CVPixelBufferGetWidth(src);
    int src_h = (int)CVPixelBufferGetHeight(src);
    CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    vImage_Buffer src_buf = {CVPixelBufferGetBaseAddress(src),
                             (vImagePixelCount)src_h, (vImagePixelCount)src_w,
                             CVPixelBufferGetBytesPerRow(src)};
    CVPixelBufferRef dst = NULL;
    CVPixelBufferCreate(kCFAllocatorDefault, dst_w, dst_h,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    CVPixelBufferLockBaseAddress(dst, 0);
    vImage_Buffer dst_buf = {CVPixelBufferGetBaseAddress(dst),
                             (vImagePixelCount)dst_h, (vImagePixelCount)dst_w,
                             CVPixelBufferGetBytesPerRow(dst)};
    vImageScale_ARGB8888(&src_buf, &dst_buf, NULL, kvImageNoFlags);
    CVPixelBufferUnlockBaseAddress(dst, 0);
    CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    return dst;
}

// Crop a region from CVPixelBuffer
static CVPixelBufferRef crop_pixelbuf(CVPixelBufferRef src, int cx, int cy, int crop_size) {
    int src_w = (int)CVPixelBufferGetWidth(src);
    int src_h = (int)CVPixelBufferGetHeight(src);
    int half = crop_size / 2;
    int x0 = std::max(0, cx - half), y0 = std::max(0, cy - half);
    int x1 = std::min(src_w, cx + half), y1 = std::min(src_h, cy + half);
    int cw = x1 - x0, ch = y1 - y0;
    CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    uint8_t *src_data = (uint8_t *)CVPixelBufferGetBaseAddress(src);
    size_t src_stride = CVPixelBufferGetBytesPerRow(src);
    CVPixelBufferRef dst = NULL;
    CVPixelBufferCreate(kCFAllocatorDefault, cw, ch, kCVPixelFormatType_32BGRA, NULL, &dst);
    CVPixelBufferLockBaseAddress(dst, 0);
    uint8_t *dst_data = (uint8_t *)CVPixelBufferGetBaseAddress(dst);
    size_t dst_stride = CVPixelBufferGetBytesPerRow(dst);
    vImage_Buffer sb = {src_data + y0 * src_stride + x0 * 4,
                        (vImagePixelCount)ch, (vImagePixelCount)cw, src_stride};
    vImage_Buffer db = {dst_data, (vImagePixelCount)ch, (vImagePixelCount)cw, dst_stride};
    vImageCopyBuffer(&sb, &db, 4, kvImageNoFlags);
    CVPixelBufferUnlockBaseAddress(dst, 0);
    CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    return dst;
}

// Extract heatmap argmax from MLMultiArray
struct CMLPeak { float x, y, confidence; };

static CMLPeak coreml_heatmap_argmax(MLMultiArray *hm, int channel, int hm_h, int hm_w) {
    const void *raw = hm.dataPointer;
    vDSP_Length n = (vDSP_Length)(hm_h * hm_w);
    float max_val = -1e9f;
    vDSP_Length max_idx = 0;
    if (hm.dataType == MLMultiArrayDataTypeFloat16) {
        const __fp16 *fp16 = (const __fp16 *)raw + channel * n;
        vImage_Buffer sb = {(void *)fp16, 1, n, n * sizeof(__fp16)};
        std::vector<float> fp32(n);
        vImage_Buffer db = {fp32.data(), 1, n, n * sizeof(float)};
        vImageConvert_Planar16FtoPlanarF(&sb, &db, kvImageNoFlags);
        vDSP_maxvi(fp32.data(), 1, &max_val, &max_idx, n);
    } else {
        const float *data = (const float *)raw + channel * n;
        vDSP_maxvi(data, 1, &max_val, &max_idx, n);
    }
    return {(float)(max_idx % hm_w) * 2.0f, (float)(max_idx / hm_w) * 2.0f,
            std::min(max_val, 255.0f) / 255.0f};
}

static MLMultiArray *find_heatmap(id<MLFeatureProvider> output) {
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
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== bench_coreml_accuracy: ONNX vs CoreML prediction comparison ===\n\n");

#ifndef RED_HAS_ONNXRUNTIME
    fprintf(stderr, "ERROR: Requires ONNX Runtime.\n");
    return 1;
#else

    // --- Load test image ---
    if (!std::filesystem::exists(TEST_JPEG)) {
        fprintf(stderr, "ERROR: Test JPEG not found: %s\n", TEST_JPEG);
        return 1;
    }
    int img_w, img_h, ch;
    uint8_t *rgb = stbi_load(TEST_JPEG, &img_w, &img_h, &ch, 3);
    assert(rgb && "Failed to load test image");
    printf("Test image: %dx%d\n", img_w, img_h);
    printf("Model dir:  %s\n\n", MODEL_DIR);

    // --- Load ONNX models ---
    std::string cd_onnx = std::string(MODEL_DIR) + "/center_detect.onnx";
    std::string kd_onnx = std::string(MODEL_DIR) + "/keypoint_detect.onnx";
    assert(std::filesystem::exists(cd_onnx));
    assert(std::filesystem::exists(kd_onnx));

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench_accuracy");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session cd_sess(env, cd_onnx.c_str(), opts);
    Ort::Session kd_sess(env, kd_onnx.c_str(), opts);
    printf("ONNX models loaded.\n");

    // --- Load CoreML models ---
    std::string cd_ml = std::string(MODEL_DIR) + "/center_detect.mlpackage";
    std::string kd_ml = std::string(MODEL_DIR) + "/keypoint_detect.mlpackage";
    assert(std::filesystem::exists(cd_ml));
    assert(std::filesystem::exists(kd_ml));

    std::string err;
    MLModel *cd_coreml = load_mlpackage(cd_ml, err);
    if (!cd_coreml) { fprintf(stderr, "CoreML CenterDetect load failed: %s\n", err.c_str()); return 1; }
    MLModel *kd_coreml = load_mlpackage(kd_ml, err);
    if (!kd_coreml) { fprintf(stderr, "CoreML KeypointDetect load failed: %s\n", err.c_str()); return 1; }
    printf("CoreML models loaded.\n\n");

    // ================================================================
    // ONNX CenterDetect (ground truth)
    // ================================================================
    printf("--- CenterDetect ---\n");
    auto onnx_cd_input = onnx_preprocess::preprocess(rgb, img_w, img_h, CENTER_SIZE);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> cd_shape = {1, 3, CENTER_SIZE, CENTER_SIZE};
    Ort::AllocatorWithDefaultOptions alloc;

    auto in_name = cd_sess.GetInputNameAllocated(0, alloc);
    auto out0 = cd_sess.GetOutputNameAllocated(0, alloc);
    auto out1 = cd_sess.GetOutputNameAllocated(1, alloc);
    const char *ins[] = {in_name.get()};
    const char *outs[] = {out0.get(), out1.get()};

    Ort::Value cd_iv = Ort::Value::CreateTensor<float>(
        mem, onnx_cd_input.data(), onnx_cd_input.size(), cd_shape.data(), cd_shape.size());

    auto t0 = std::chrono::steady_clock::now();
    auto cd_outputs = cd_sess.Run(Ort::RunOptions{nullptr}, ins, &cd_iv, 1, outs, 2);
    auto t1 = std::chrono::steady_clock::now();
    float onnx_cd_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    auto &cd_hm = cd_outputs[1];
    auto cd_hm_shape = cd_hm.GetTensorTypeAndShapeInfo().GetShape();
    int cd_hm_h = (int)cd_hm_shape[2], cd_hm_w = (int)cd_hm_shape[3];
    auto onnx_center = onnx_preprocess::heatmap_argmax(cd_hm.GetTensorData<float>(), cd_hm_h, cd_hm_w);
    float ds_x = (float)img_w / CENTER_SIZE;
    float ds_y = (float)img_h / CENTER_SIZE;
    onnx_center.x *= ds_x;
    onnx_center.y *= ds_y;

    printf("  ONNX:   center=(%6.1f, %6.1f) conf=%.4f  [%.1f ms]\n",
           onnx_center.x, onnx_center.y, onnx_center.confidence, onnx_cd_ms);

    // ================================================================
    // CoreML CenterDetect
    // ================================================================
    @autoreleasepool {
        CVPixelBufferRef pb = create_bgra_pixelbuf(rgb, img_w, img_h);
        CVPixelBufferRef resized = resize_pixelbuf(pb, CENTER_SIZE, CENTER_SIZE);
        CVPixelBufferRelease(pb);

        NSError *error = nil;
        id<MLFeatureProvider> cd_input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{@"image": [MLFeatureValue featureValueWithPixelBuffer:resized]}
                error:&error];

        t0 = std::chrono::steady_clock::now();
        id<MLFeatureProvider> cd_output = [cd_coreml predictionFromFeatures:cd_input error:&error];
        t1 = std::chrono::steady_clock::now();
        float coreml_cd_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        CVPixelBufferRelease(resized);

        if (!cd_output) {
            printf("  CoreML CenterDetect failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        MLMultiArray *cd_hm_ml = find_heatmap(cd_output);
        int ml_hm_h = [cd_hm_ml.shape[2] intValue], ml_hm_w = [cd_hm_ml.shape[3] intValue];
        CMLPeak coreml_center = coreml_heatmap_argmax(cd_hm_ml, 0, ml_hm_h, ml_hm_w);
        coreml_center.x *= ds_x;
        coreml_center.y *= ds_y;

        printf("  CoreML: center=(%6.1f, %6.1f) conf=%.4f  [%.1f ms]\n",
               coreml_center.x, coreml_center.y, coreml_center.confidence, coreml_cd_ms);

        float cd_dx = coreml_center.x - onnx_center.x;
        float cd_dy = coreml_center.y - onnx_center.y;
        float cd_dist = std::sqrt(cd_dx * cd_dx + cd_dy * cd_dy);
        float cd_conf_diff = std::abs(coreml_center.confidence - onnx_center.confidence);
        printf("  DELTA:  dist=%.1f px, conf_diff=%.4f\n\n", cd_dist, cd_conf_diff);

        // ================================================================
        // KeypointDetect — use ONNX center for both (fair comparison)
        // ================================================================
        printf("--- KeypointDetect (using ONNX center for both) ---\n");
        int half = KP_SIZE / 2;
        int cx = std::clamp((int)onnx_center.x, half, img_w - half);
        int cy = std::clamp((int)onnx_center.y, half, img_h - half);
        printf("  Crop center: (%d, %d), size: %dx%d\n\n", cx, cy, KP_SIZE, KP_SIZE);

        // ONNX KeypointDetect
        auto onnx_kd_input = onnx_preprocess::preprocess_crop(rgb, img_w, img_h, cx, cy, KP_SIZE, KP_SIZE);
        std::array<int64_t, 4> kd_shape = {1, 3, KP_SIZE, KP_SIZE};
        auto kd_in = kd_sess.GetInputNameAllocated(0, alloc);
        auto kd_o0 = kd_sess.GetOutputNameAllocated(0, alloc);
        auto kd_o1 = kd_sess.GetOutputNameAllocated(1, alloc);
        const char *kd_ins[] = {kd_in.get()};
        const char *kd_outs[] = {kd_o0.get(), kd_o1.get()};
        Ort::Value kd_iv = Ort::Value::CreateTensor<float>(
            mem, onnx_kd_input.data(), onnx_kd_input.size(), kd_shape.data(), kd_shape.size());

        t0 = std::chrono::steady_clock::now();
        auto kd_outputs = kd_sess.Run(Ort::RunOptions{nullptr}, kd_ins, &kd_iv, 1, kd_outs, 2);
        t1 = std::chrono::steady_clock::now();
        float onnx_kd_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        auto &kd_hm = kd_outputs[1];
        auto kd_hm_shape = kd_hm.GetTensorTypeAndShapeInfo().GetShape();
        int n_joints = (int)kd_hm_shape[1];
        int kd_hm_h = (int)kd_hm_shape[2], kd_hm_w = (int)kd_hm_shape[3];

        std::vector<onnx_preprocess::Peak> onnx_kps(n_joints);
        for (int j = 0; j < n_joints; ++j) {
            onnx_kps[j] = onnx_preprocess::heatmap_argmax(
                kd_hm.GetTensorData<float>() + j * kd_hm_h * kd_hm_w, kd_hm_h, kd_hm_w);
        }

        // CoreML KeypointDetect
        CVPixelBufferRef full_pb = create_bgra_pixelbuf(rgb, img_w, img_h);
        CVPixelBufferRef crop = crop_pixelbuf(full_pb, cx, cy, KP_SIZE);
        CVPixelBufferRelease(full_pb);

        id<MLFeatureProvider> kd_input_ml =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                @{@"image": [MLFeatureValue featureValueWithPixelBuffer:crop]}
                error:&error];

        t0 = std::chrono::steady_clock::now();
        id<MLFeatureProvider> kd_output_ml = [kd_coreml predictionFromFeatures:kd_input_ml error:&error];
        t1 = std::chrono::steady_clock::now();
        float coreml_kd_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        CVPixelBufferRelease(crop);

        if (!kd_output_ml) {
            printf("  CoreML KeypointDetect failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        MLMultiArray *kd_hm_ml = find_heatmap(kd_output_ml);
        int ml_kd_h = [kd_hm_ml.shape[2] intValue], ml_kd_w = [kd_hm_ml.shape[3] intValue];
        int ml_n_joints = [kd_hm_ml.shape[1] intValue];
        int cmp_joints = std::min(n_joints, ml_n_joints);

        // ================================================================
        // Comparison
        // ================================================================
        printf("  %-6s  %8s %8s %7s   %8s %8s %7s   %7s %8s\n",
               "Joint", "ONNX_x", "ONNX_y", "O_conf", "CML_x", "CML_y", "C_conf", "dist", "conf_d");
        printf("  %-6s  %8s %8s %7s   %8s %8s %7s   %7s %8s\n",
               "-----", "------", "------", "------", "-----", "-----", "------", "----", "------");

        float sum_dist = 0, max_dist = 0;
        float sum_conf_diff = 0;
        int n_agree = 0; // both within 10px

        for (int j = 0; j < cmp_joints; ++j) {
            CMLPeak cml_kp = coreml_heatmap_argmax(kd_hm_ml, j, ml_kd_h, ml_kd_w);

            float dx = cml_kp.x - onnx_kps[j].x;
            float dy = cml_kp.y - onnx_kps[j].y;
            float dist = std::sqrt(dx * dx + dy * dy);
            float conf_d = std::abs(cml_kp.confidence - onnx_kps[j].confidence);

            sum_dist += dist;
            max_dist = std::max(max_dist, dist);
            sum_conf_diff += conf_d;
            if (dist < 10.0f) n_agree++;

            printf("  %4d    %7.1f  %7.1f  %6.4f   %7.1f  %7.1f  %6.4f   %6.1f   %6.4f%s\n",
                   j, onnx_kps[j].x, onnx_kps[j].y, onnx_kps[j].confidence,
                   cml_kp.x, cml_kp.y, cml_kp.confidence,
                   dist, conf_d,
                   dist > 10.0f ? " ***" : "");
        }

        printf("\n=== SUMMARY ===\n");
        printf("  Joints compared:     %d\n", cmp_joints);
        printf("  Agreement (<10px):   %d / %d (%.0f%%)\n",
               n_agree, cmp_joints, 100.0f * n_agree / cmp_joints);
        printf("  Mean keypoint error: %.1f px\n", sum_dist / cmp_joints);
        printf("  Max keypoint error:  %.1f px\n", max_dist);
        printf("  Mean conf diff:      %.4f\n", sum_conf_diff / cmp_joints);
        printf("  ONNX timing:         CD=%.1f ms + KD=%.1f ms\n", onnx_cd_ms, onnx_kd_ms);
        printf("  CoreML timing:       CD=%.1f ms + KD=%.1f ms\n", coreml_cd_ms, coreml_kd_ms);
        printf("\n");

        if (sum_dist / cmp_joints > 20.0f) {
            printf("  *** LARGE DIVERGENCE: CoreML and ONNX predictions differ significantly.\n");
            printf("      This likely indicates a normalization mismatch in the CoreML export.\n");
        } else if (sum_dist / cmp_joints < 5.0f) {
            printf("  GOOD: CoreML and ONNX predictions are closely aligned.\n");
            printf("      Normalization appears correct.\n");
        } else {
            printf("  MODERATE: Some divergence between CoreML and ONNX.\n");
            printf("      May be due to float16 precision or resize interpolation differences.\n");
        }
    }

    stbi_image_free(rgb);
    printf("\n=== bench_coreml_accuracy complete ===\n");
    return 0;
#endif
}
