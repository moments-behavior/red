// tests/bench_jarvis.cpp — JARVIS inference pipeline benchmark
//
// Measures per-step timing for the JARVIS CenterDetect + KeypointDetect
// pipeline on real 3208x2200 JPEG images from annotate1 export.
//
// Steps benchmarked:
//   (a) stbi_load: disk -> RGB
//   (b) preprocess CenterDetect: resize 3208x2200 -> 320x320 + normalize
//   (c) CenterDetect ONNX inference (CPU)
//   (d) preprocess KeypointDetect: crop 704x704 + normalize
//   (e) KeypointDetect ONNX inference (CPU)
//   (f) Heatmap argmax postprocessing
//
// Each step runs 5 iterations; reports mean +/- stddev.
// Also tests CoreML EP for comparison.

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#ifdef RED_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#endif

// ---------------------------------------------------------------------------
// Preprocessing (copied from jarvis_inference.h to avoid pulling in all deps)
// ---------------------------------------------------------------------------
namespace jarvis_detail {

static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float INV_STD[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

inline std::vector<float> preprocess(const uint8_t *rgb, int src_w, int src_h,
                                      int dst_size) {
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
    int cw = x1 - x0;
    int ch = y1 - y0;

    std::vector<uint8_t> crop(cw * ch * 3);
    for (int y = 0; y < ch; ++y) {
        const uint8_t *src_row = rgb + ((y0 + y) * img_w + x0) * 3;
        uint8_t *dst_row = crop.data() + y * cw * 3;
        std::memcpy(dst_row, src_row, cw * 3);
    }
    return preprocess(crop.data(), cw, ch, dst_size);
}

struct HeatmapPeak { float x, y, confidence; };

inline HeatmapPeak heatmap_argmax(const float *heatmap, int h, int w) {
    int max_idx = 0;
    float max_val = heatmap[0];
    for (int i = 1; i < h * w; ++i) {
        if (heatmap[i] > max_val) {
            max_val = heatmap[i];
            max_idx = i;
        }
    }
    HeatmapPeak peak;
    peak.x = (float)(max_idx % w) * 2.0f;
    peak.y = (float)(max_idx / w) * 2.0f;
    peak.confidence = std::min(max_val, 255.0f) / 255.0f;
    return peak;
}

} // namespace jarvis_detail

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

struct TimingStats {
    double mean_ms = 0;
    double stddev_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
    std::vector<double> samples;
};

static TimingStats compute_stats(const std::vector<double> &times) {
    TimingStats s;
    s.samples = times;
    if (times.empty()) return s;
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    s.mean_ms = sum / times.size();
    double sq_sum = 0;
    for (double t : times) sq_sum += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = std::sqrt(sq_sum / times.size());
    s.min_ms = *std::min_element(times.begin(), times.end());
    s.max_ms = *std::max_element(times.begin(), times.end());
    return s;
}

static void print_stats(const char *label, const TimingStats &s) {
    printf("  %-35s %7.2f +/- %5.2f ms  [min=%.2f, max=%.2f]\n",
           label, s.mean_ms, s.stddev_ms, s.min_ms, s.max_ms);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const char *MODEL_DIR = "/Users/johnsonr/src/red/models/jarvis_mouseJan30";
static const char *JPEG_DIR  = "/Users/johnsonr/red_projects/annotate1/export/"
                               "2026_03_10_08_45_34/train/2026_03_10_08_41_51";

static constexpr int IMG_W = 3208;
static constexpr int IMG_H = 2200;
static constexpr int CENTER_SIZE = 320;
static constexpr int KP_SIZE = 704;
static constexpr int NUM_JOINTS = 24;
static constexpr int NUM_CAMS = 16;
static constexpr int NUM_ITERS = 5;

static const char *CAMERA_NAMES[] = {
    "Cam2002486", "Cam2002487", "Cam2005325", "Cam2006050",
    "Cam2006051", "Cam2006052", "Cam2006054", "Cam2006055",
    "Cam2006515", "Cam2006516", "Cam2008665", "Cam2008666",
    "Cam2008667", "Cam2008668", "Cam2008669", "Cam2008670"
};

// ---------------------------------------------------------------------------
// Benchmark: CPU execution provider
// ---------------------------------------------------------------------------
#ifdef RED_HAS_ONNXRUNTIME

struct BenchResult {
    TimingStats stbi_load;
    TimingStats preprocess_center;
    TimingStats center_infer;
    TimingStats preprocess_kp;
    TimingStats kp_infer;
    TimingStats heatmap_argmax;
    TimingStats total_per_camera;
};

static BenchResult run_bench(const char *label, Ort::Session &center_sess,
                              Ort::Session &kp_sess) {
    printf("\n=== %s ===\n", label);
    BenchResult result;

    std::string jpeg_path = std::string(JPEG_DIR) + "/Cam2002486/Frame_5998.jpg";

    // Pre-load image for iterations that skip disk I/O
    int w0, h0, ch0;
    uint8_t *preloaded = stbi_load(jpeg_path.c_str(), &w0, &h0, &ch0, 3);
    assert(preloaded && "Failed to load test JPEG");
    printf("  Image: %dx%d (%zu bytes RGB)\n\n", w0, h0, (size_t)w0 * h0 * 3);

    // Use image center as a reasonable center detection result for crop benchmarking
    int center_x = w0 / 2;
    int center_y = h0 / 2;

    // -- Warmup: one full pass to trigger any lazy JIT/compilation --
    {
        auto input = jarvis_detail::preprocess(preloaded, w0, h0, CENTER_SIZE);
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> shape = {1, 3, CENTER_SIZE, CENTER_SIZE};

        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = center_sess.GetInputNameAllocated(0, alloc);
        auto out0 = center_sess.GetOutputNameAllocated(0, alloc);
        auto out1 = center_sess.GetOutputNameAllocated(1, alloc);
        const char *ins[] = {in_name.get()};
        const char *outs[] = {out0.get(), out1.get()};

        Ort::Value iv = Ort::Value::CreateTensor<float>(
            mem, input.data(), input.size(), shape.data(), shape.size());
        auto outputs = center_sess.Run(Ort::RunOptions{nullptr}, ins, &iv, 1, outs, 2);
        printf("  Warmup CenterDetect done.\n");

        // Get center from warmup to use for keypoint
        auto &hm = outputs[1];
        auto hm_shape = hm.GetTensorTypeAndShapeInfo().GetShape();
        int hm_h = (int)hm_shape[2], hm_w = (int)hm_shape[3];
        auto peak = jarvis_detail::heatmap_argmax(hm.GetTensorData<float>(), hm_h, hm_w);
        float ds_x = (float)w0 / CENTER_SIZE;
        float ds_y = (float)h0 / CENTER_SIZE;
        center_x = (int)(peak.x * ds_x);
        center_y = (int)(peak.y * ds_y);
        printf("  Detected center: (%d, %d), confidence=%.3f\n", center_x, center_y, peak.confidence);

        // Warmup keypoint
        auto kp_input = jarvis_detail::preprocess_crop(preloaded, w0, h0,
            center_x, center_y, KP_SIZE, KP_SIZE);
        std::array<int64_t, 4> kp_shape = {1, 3, KP_SIZE, KP_SIZE};

        auto kp_in = kp_sess.GetInputNameAllocated(0, alloc);
        auto kp_o0 = kp_sess.GetOutputNameAllocated(0, alloc);
        auto kp_o1 = kp_sess.GetOutputNameAllocated(1, alloc);
        const char *kp_ins[] = {kp_in.get()};
        const char *kp_outs[] = {kp_o0.get(), kp_o1.get()};

        Ort::Value kv = Ort::Value::CreateTensor<float>(
            mem, kp_input.data(), kp_input.size(), kp_shape.data(), kp_shape.size());
        kp_sess.Run(Ort::RunOptions{nullptr}, kp_ins, &kv, 1, kp_outs, 2);
        printf("  Warmup KeypointDetect done.\n\n");
    }

    // -- Benchmark each step --
    std::vector<double> t_stbi, t_preproc_center, t_center_infer;
    std::vector<double> t_preproc_kp, t_kp_infer, t_argmax, t_total;

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        auto t_all_start = Clock::now();

        // (a) stbi_load
        auto ta0 = Clock::now();
        int w, h, ch;
        uint8_t *rgb = stbi_load(jpeg_path.c_str(), &w, &h, &ch, 3);
        auto ta1 = Clock::now();
        t_stbi.push_back(std::chrono::duration<double, std::milli>(ta1 - ta0).count());
        assert(rgb);

        // (b) Preprocess CenterDetect
        auto tb0 = Clock::now();
        auto center_input = jarvis_detail::preprocess(rgb, w, h, CENTER_SIZE);
        auto tb1 = Clock::now();
        t_preproc_center.push_back(std::chrono::duration<double, std::milli>(tb1 - tb0).count());

        // (c) CenterDetect ONNX inference
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> shape = {1, 3, CENTER_SIZE, CENTER_SIZE};
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = center_sess.GetInputNameAllocated(0, alloc);
        auto out0 = center_sess.GetOutputNameAllocated(0, alloc);
        auto out1 = center_sess.GetOutputNameAllocated(1, alloc);
        const char *ins[] = {in_name.get()};
        const char *outs[] = {out0.get(), out1.get()};

        Ort::Value iv = Ort::Value::CreateTensor<float>(
            mem, center_input.data(), center_input.size(), shape.data(), shape.size());

        auto tc0 = Clock::now();
        auto center_outputs = center_sess.Run(
            Ort::RunOptions{nullptr}, ins, &iv, 1, outs, 2);
        auto tc1 = Clock::now();
        t_center_infer.push_back(std::chrono::duration<double, std::milli>(tc1 - tc0).count());

        // Extract center from heatmap (for crop location)
        auto &hm = center_outputs[1];
        auto hm_shape = hm.GetTensorTypeAndShapeInfo().GetShape();
        int hm_h = (int)hm_shape[2], hm_w = (int)hm_shape[3];
        float ds_x = (float)w / CENTER_SIZE;
        float ds_y = (float)h / CENTER_SIZE;
        auto cpeak = jarvis_detail::heatmap_argmax(hm.GetTensorData<float>(), hm_h, hm_w);
        int cx = (int)(cpeak.x * ds_x);
        int cy = (int)(cpeak.y * ds_y);

        // (d) Preprocess KeypointDetect (crop)
        int half = KP_SIZE / 2;
        cx = std::clamp(cx, half, w - half);
        cy = std::clamp(cy, half, h - half);

        auto td0 = Clock::now();
        auto kp_input = jarvis_detail::preprocess_crop(
            rgb, w, h, cx, cy, KP_SIZE, KP_SIZE);
        auto td1 = Clock::now();
        t_preproc_kp.push_back(std::chrono::duration<double, std::milli>(td1 - td0).count());

        // (e) KeypointDetect ONNX inference
        std::array<int64_t, 4> kp_shape_arr = {1, 3, KP_SIZE, KP_SIZE};
        auto kp_in = kp_sess.GetInputNameAllocated(0, alloc);
        auto kp_o0 = kp_sess.GetOutputNameAllocated(0, alloc);
        auto kp_o1 = kp_sess.GetOutputNameAllocated(1, alloc);
        const char *kp_ins[] = {kp_in.get()};
        const char *kp_outs[] = {kp_o0.get(), kp_o1.get()};

        Ort::Value kv = Ort::Value::CreateTensor<float>(
            mem, kp_input.data(), kp_input.size(), kp_shape_arr.data(), kp_shape_arr.size());

        auto te0 = Clock::now();
        auto kp_outputs = kp_sess.Run(
            Ort::RunOptions{nullptr}, kp_ins, &kv, 1, kp_outs, 2);
        auto te1 = Clock::now();
        t_kp_infer.push_back(std::chrono::duration<double, std::milli>(te1 - te0).count());

        // (f) Heatmap argmax (all joints)
        auto &kp_hm = kp_outputs[1];
        auto kp_hm_shape = kp_hm.GetTensorTypeAndShapeInfo().GetShape();
        int n_joints = (int)kp_hm_shape[1];
        int kp_hm_h = (int)kp_hm_shape[2];
        int kp_hm_w = (int)kp_hm_shape[3];
        const float *kp_hm_data = kp_hm.GetTensorData<float>();

        auto tf0 = Clock::now();
        std::vector<jarvis_detail::HeatmapPeak> keypoints(n_joints);
        for (int j = 0; j < n_joints; ++j) {
            keypoints[j] = jarvis_detail::heatmap_argmax(
                kp_hm_data + j * kp_hm_h * kp_hm_w, kp_hm_h, kp_hm_w);
        }
        auto tf1 = Clock::now();
        t_argmax.push_back(std::chrono::duration<double, std::milli>(tf1 - tf0).count());

        auto t_all_end = Clock::now();
        t_total.push_back(std::chrono::duration<double, std::milli>(t_all_end - t_all_start).count());

        stbi_image_free(rgb);

        // Print keypoints on first iter
        if (iter == 0) {
            printf("  Keypoints (iter 0, %d joints, heatmap %dx%d):\n", n_joints, kp_hm_w, kp_hm_h);
            int bbox_hw = KP_SIZE / 2;
            for (int j = 0; j < std::min(n_joints, 6); ++j) {
                float img_x = keypoints[j].x + (float)(cx - bbox_hw);
                float img_y = keypoints[j].y + (float)(cy - bbox_hw);
                printf("    joint %2d: hm=(%.1f,%.1f) -> img=(%.1f,%.1f) conf=%.3f\n",
                       j, keypoints[j].x, keypoints[j].y, img_x, img_y, keypoints[j].confidence);
            }
            if (n_joints > 6) printf("    ... (%d more)\n", n_joints - 6);
        }
    }

    result.stbi_load = compute_stats(t_stbi);
    result.preprocess_center = compute_stats(t_preproc_center);
    result.center_infer = compute_stats(t_center_infer);
    result.preprocess_kp = compute_stats(t_preproc_kp);
    result.kp_infer = compute_stats(t_kp_infer);
    result.heatmap_argmax = compute_stats(t_argmax);
    result.total_per_camera = compute_stats(t_total);

    printf("\n  Per-step timing (%d iterations):\n", NUM_ITERS);
    print_stats("(a) stbi_load (disk->RGB)", result.stbi_load);
    print_stats("(b) preprocess CenterDetect", result.preprocess_center);
    print_stats("(c) CenterDetect inference", result.center_infer);
    print_stats("(d) preprocess KeypointDetect", result.preprocess_kp);
    print_stats("(e) KeypointDetect inference", result.kp_infer);
    print_stats("(f) heatmap argmax (24 joints)", result.heatmap_argmax);
    print_stats("    TOTAL per camera", result.total_per_camera);

    printf("\n  Projected 16-camera time:\n");
    double total16 = result.total_per_camera.mean_ms * NUM_CAMS;
    printf("    Sequential: %.0f ms (%.1f fps)\n", total16, 1000.0 / total16);
    printf("    Inference-only x16: %.0f ms\n",
           (result.center_infer.mean_ms + result.kp_infer.mean_ms) * NUM_CAMS);

    stbi_image_free(preloaded);
    return result;
}

// ---------------------------------------------------------------------------
// Multi-camera benchmark (all 16 cameras, one pass each)
// ---------------------------------------------------------------------------
static void run_multi_camera_bench(const char *label, Ort::Session &center_sess,
                                    Ort::Session &kp_sess) {
    printf("\n=== %s: All 16 cameras ===\n", label);

    auto t_all_start = Clock::now();
    double sum_load = 0, sum_center_pre = 0, sum_center_inf = 0;
    double sum_kp_pre = 0, sum_kp_inf = 0, sum_argmax = 0;

    for (int c = 0; c < NUM_CAMS; ++c) {
        std::string jpeg_path = std::string(JPEG_DIR) + "/" + CAMERA_NAMES[c] + "/Frame_5998.jpg";
        if (!std::filesystem::exists(jpeg_path)) {
            printf("  %s: SKIP (no JPEG)\n", CAMERA_NAMES[c]);
            continue;
        }

        // (a) load
        auto ta0 = Clock::now();
        int w, h, ch;
        uint8_t *rgb = stbi_load(jpeg_path.c_str(), &w, &h, &ch, 3);
        auto ta1 = Clock::now();
        sum_load += std::chrono::duration<double, std::milli>(ta1 - ta0).count();
        if (!rgb) continue;

        // (b) preprocess center
        auto tb0 = Clock::now();
        auto center_input = jarvis_detail::preprocess(rgb, w, h, CENTER_SIZE);
        auto tb1 = Clock::now();
        sum_center_pre += std::chrono::duration<double, std::milli>(tb1 - tb0).count();

        // (c) center inference
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> shape = {1, 3, CENTER_SIZE, CENTER_SIZE};
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = center_sess.GetInputNameAllocated(0, alloc);
        auto out0 = center_sess.GetOutputNameAllocated(0, alloc);
        auto out1 = center_sess.GetOutputNameAllocated(1, alloc);
        const char *ins[] = {in_name.get()};
        const char *outs[] = {out0.get(), out1.get()};
        Ort::Value iv = Ort::Value::CreateTensor<float>(
            mem, center_input.data(), center_input.size(), shape.data(), shape.size());

        auto tc0 = Clock::now();
        auto center_outputs = center_sess.Run(
            Ort::RunOptions{nullptr}, ins, &iv, 1, outs, 2);
        auto tc1 = Clock::now();
        sum_center_inf += std::chrono::duration<double, std::milli>(tc1 - tc0).count();

        // extract center
        auto &hm = center_outputs[1];
        auto hm_shape_v = hm.GetTensorTypeAndShapeInfo().GetShape();
        int hm_h = (int)hm_shape_v[2], hm_w = (int)hm_shape_v[3];
        float ds_x = (float)w / CENTER_SIZE;
        float ds_y = (float)h / CENTER_SIZE;
        auto cpeak = jarvis_detail::heatmap_argmax(hm.GetTensorData<float>(), hm_h, hm_w);
        int cx = std::clamp((int)(cpeak.x * ds_x), KP_SIZE / 2, w - KP_SIZE / 2);
        int cy = std::clamp((int)(cpeak.y * ds_y), KP_SIZE / 2, h - KP_SIZE / 2);

        // (d) preprocess keypoint
        auto td0 = Clock::now();
        auto kp_input = jarvis_detail::preprocess_crop(rgb, w, h, cx, cy, KP_SIZE, KP_SIZE);
        auto td1 = Clock::now();
        sum_kp_pre += std::chrono::duration<double, std::milli>(td1 - td0).count();

        // (e) keypoint inference
        std::array<int64_t, 4> kp_shape = {1, 3, KP_SIZE, KP_SIZE};
        auto kp_in = kp_sess.GetInputNameAllocated(0, alloc);
        auto kp_o0 = kp_sess.GetOutputNameAllocated(0, alloc);
        auto kp_o1 = kp_sess.GetOutputNameAllocated(1, alloc);
        const char *kp_ins[] = {kp_in.get()};
        const char *kp_outs[] = {kp_o0.get(), kp_o1.get()};
        Ort::Value kv = Ort::Value::CreateTensor<float>(
            mem, kp_input.data(), kp_input.size(), kp_shape.data(), kp_shape.size());

        auto te0 = Clock::now();
        auto kp_outputs = kp_sess.Run(
            Ort::RunOptions{nullptr}, kp_ins, &kv, 1, kp_outs, 2);
        auto te1 = Clock::now();
        sum_kp_inf += std::chrono::duration<double, std::milli>(te1 - te0).count();

        // (f) argmax
        auto &kp_hm = kp_outputs[1];
        auto kp_hm_shape = kp_hm.GetTensorTypeAndShapeInfo().GetShape();
        int n_joints = (int)kp_hm_shape[1];
        int kp_hm_h = (int)kp_hm_shape[2];
        int kp_hm_w = (int)kp_hm_shape[3];
        const float *kp_hm_data = kp_hm.GetTensorData<float>();

        auto tf0 = Clock::now();
        for (int j = 0; j < n_joints; ++j) {
            jarvis_detail::heatmap_argmax(kp_hm_data + j * kp_hm_h * kp_hm_w, kp_hm_h, kp_hm_w);
        }
        auto tf1 = Clock::now();
        sum_argmax += std::chrono::duration<double, std::milli>(tf1 - tf0).count();

        printf("  %s: center=(%4d,%4d) conf=%.3f\n", CAMERA_NAMES[c], cx, cy, cpeak.confidence);
        stbi_image_free(rgb);
    }

    auto t_all_end = Clock::now();
    double total_wall = std::chrono::duration<double, std::milli>(t_all_end - t_all_start).count();

    printf("\n  16-camera totals:\n");
    printf("    stbi_load:           %7.1f ms\n", sum_load);
    printf("    preprocess center:   %7.1f ms\n", sum_center_pre);
    printf("    CenterDetect infer:  %7.1f ms\n", sum_center_inf);
    printf("    preprocess kp:       %7.1f ms\n", sum_kp_pre);
    printf("    KeypointDetect infer:%7.1f ms\n", sum_kp_inf);
    printf("    heatmap argmax:      %7.1f ms\n", sum_argmax);
    printf("    ---------------------------------\n");
    printf("    Wall clock total:    %7.1f ms\n", total_wall);
    double infer_only = sum_center_inf + sum_kp_inf;
    printf("    Inference only:      %7.1f ms (%.1f%%)\n", infer_only, 100.0 * infer_only / total_wall);
    printf("    Preprocessing:       %7.1f ms (%.1f%%)\n",
           sum_center_pre + sum_kp_pre,
           100.0 * (sum_center_pre + sum_kp_pre) / total_wall);
    printf("    I/O (load):          %7.1f ms (%.1f%%)\n", sum_load, 100.0 * sum_load / total_wall);
    printf("    Effective FPS:       %7.1f\n", 1000.0 / total_wall);
}

#endif // RED_HAS_ONNXRUNTIME

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== bench_jarvis: JARVIS inference pipeline benchmark ===\n\n");

#ifndef RED_HAS_ONNXRUNTIME
    fprintf(stderr, "ERROR: This benchmark requires ONNX Runtime (RED_HAS_ONNXRUNTIME).\n");
    return 1;
#else
    // Check models
    std::string center_path = std::string(MODEL_DIR) + "/center_detect.onnx";
    std::string kp_path = std::string(MODEL_DIR) + "/keypoint_detect.onnx";

    if (!std::filesystem::exists(center_path)) {
        fprintf(stderr, "ERROR: CenterDetect model not found: %s\n", center_path.c_str());
        return 1;
    }
    if (!std::filesystem::exists(kp_path)) {
        fprintf(stderr, "ERROR: KeypointDetect model not found: %s\n", kp_path.c_str());
        return 1;
    }
    printf("Models: %s\n", MODEL_DIR);

    // Check test image
    std::string test_jpeg = std::string(JPEG_DIR) + "/Cam2002486/Frame_5998.jpg";
    if (!std::filesystem::exists(test_jpeg)) {
        fprintf(stderr, "ERROR: Test JPEG not found: %s\n", test_jpeg.c_str());
        return 1;
    }
    printf("Test image: %s\n", test_jpeg.c_str());
    printf("Iterations: %d\n", NUM_ITERS);

    // ================================================================
    // Part 1: CPU Execution Provider
    // ================================================================
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench_jarvis_cpu");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        printf("\nLoading models (CPU EP, 4 threads)...\n");
        auto tl0 = Clock::now();
        Ort::Session center_sess(env, center_path.c_str(), opts);
        Ort::Session kp_sess(env, kp_path.c_str(), opts);
        auto tl1 = Clock::now();
        printf("  Model load time: %.0f ms\n",
               std::chrono::duration<double, std::milli>(tl1 - tl0).count());

        // Print model info
        {
            Ort::AllocatorWithDefaultOptions alloc;
            auto ci = center_sess.GetInputNameAllocated(0, alloc);
            auto co0 = center_sess.GetOutputNameAllocated(0, alloc);
            auto co1 = center_sess.GetOutputNameAllocated(1, alloc);
            printf("  CenterDetect:  input='%s', outputs=['%s','%s']\n",
                   ci.get(), co0.get(), co1.get());

            auto ki = kp_sess.GetInputNameAllocated(0, alloc);
            auto ko0 = kp_sess.GetOutputNameAllocated(0, alloc);
            auto ko1 = kp_sess.GetOutputNameAllocated(1, alloc);
            printf("  KeypointDetect: input='%s', outputs=['%s','%s']\n",
                   ki.get(), ko0.get(), ko1.get());

            // Print input/output shapes
            auto c_in_info = center_sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            printf("  CenterDetect input shape: [");
            for (auto d : c_in_info.GetShape()) printf("%lld,", d);
            printf("]\n");

            auto k_in_info = kp_sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            printf("  KeypointDetect input shape: [");
            for (auto d : k_in_info.GetShape()) printf("%lld,", d);
            printf("]\n");
        }

        run_bench("CPU EP (4 threads)", center_sess, kp_sess);
        run_multi_camera_bench("CPU EP (4 threads)", center_sess, kp_sess);
    }

    // ================================================================
    // Part 2: CoreML Execution Provider
    // ================================================================
#ifdef __APPLE__
    {
        printf("\n\n======================================================\n");
        printf("CoreML Execution Provider\n");
        printf("======================================================\n");

        Ort::Env env_coreml(ORT_LOGGING_LEVEL_WARNING, "bench_jarvis_coreml");
        Ort::SessionOptions opts_coreml;
        opts_coreml.SetIntraOpNumThreads(1);
        opts_coreml.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        bool coreml_ok = false;
        try {
            uint32_t coreml_flags = 0; // default: ANE/GPU
            (void)OrtSessionOptionsAppendExecutionProvider_CoreML(opts_coreml, coreml_flags);
            coreml_ok = true;
            printf("  CoreML EP registered.\n");
        } catch (const std::exception &e) {
            printf("  CoreML EP failed: %s\n", e.what());
        } catch (...) {
            printf("  CoreML EP failed (unknown error).\n");
        }

        if (coreml_ok) {
            printf("  Loading models (CoreML EP)...\n");
            try {
                auto tl0 = Clock::now();
                Ort::Session center_sess(env_coreml, center_path.c_str(), opts_coreml);
                Ort::Session kp_sess(env_coreml, kp_path.c_str(), opts_coreml);
                auto tl1 = Clock::now();
                printf("  Model load time: %.0f ms\n",
                       std::chrono::duration<double, std::milli>(tl1 - tl0).count());

                run_bench("CoreML EP", center_sess, kp_sess);
                run_multi_camera_bench("CoreML EP", center_sess, kp_sess);
            } catch (const Ort::Exception &e) {
                printf("  CoreML session creation failed: %s\n", e.what());
                printf("  (This is expected for models with decomposed InstanceNorm)\n");
            } catch (const std::exception &e) {
                printf("  CoreML session creation failed: %s\n", e.what());
            }
        }
    }
#endif

    // ================================================================
    // Part 3: Thread count comparison
    // ================================================================
    {
        printf("\n\n======================================================\n");
        printf("Thread count comparison (CPU EP)\n");
        printf("======================================================\n");

        Ort::Env env_threads(ORT_LOGGING_LEVEL_WARNING, "bench_jarvis_threads");

        for (int nthreads : {1, 2, 4, 8}) {
            Ort::SessionOptions opts_t;
            opts_t.SetIntraOpNumThreads(nthreads);
            opts_t.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            Ort::Session center_sess(env_threads, center_path.c_str(), opts_t);
            Ort::Session kp_sess(env_threads, kp_path.c_str(), opts_t);

            char label[64];
            snprintf(label, sizeof(label), "CPU EP (%d thread%s)",
                     nthreads, nthreads > 1 ? "s" : "");
            run_bench(label, center_sess, kp_sess);
        }
    }

    printf("\n=== bench_jarvis complete ===\n");
    return 0;
#endif
}
