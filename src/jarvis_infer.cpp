#include "jarvis_infer.h"
#include <NvInfer.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <torch/torch.h>

// ---------------------------------------------------------------------------
// ImageNet BGR normalization constants (channel order: B, G, R)
// ---------------------------------------------------------------------------
static constexpr float BGR_MEAN[3] = {0.406f, 0.456f, 0.485f}; // B, G, R
static constexpr float BGR_STD[3]  = {0.225f, 0.224f, 0.229f}; // B, G, R

// ---------------------------------------------------------------------------
// TRT Logger
// ---------------------------------------------------------------------------
void JarvisTRTLogger::log(Severity severity, const char *msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << "\n";
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
JarvisInfer::JarvisInfer() = default;

JarvisInfer::~JarvisInfer() {
    if (center_in_d_)  cudaFree(center_in_d_);
    if (center_out_d_) cudaFree(center_out_d_);
    if (kp_in_d_)      cudaFree(kp_in_d_);
    if (kp_out_d_)     cudaFree(kp_out_d_);
    if (stream_)       cudaStreamDestroy(stream_);
    if (center_ctx_)   { delete center_ctx_;   center_ctx_ = nullptr; }
    if (kp_ctx_)       { delete kp_ctx_;       kp_ctx_ = nullptr; }
    if (center_engine_){ delete center_engine_; center_engine_ = nullptr; }
    if (kp_engine_)    { delete kp_engine_;    kp_engine_ = nullptr; }
    if (trt_runtime_)  { delete trt_runtime_;  trt_runtime_ = nullptr; }
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------
bool JarvisInfer::load(const std::string &center_path,
                       const std::string &kp_path) {
    loaded_ = false;

    bool center_is_engine = center_path.size() > 7 &&
                            center_path.substr(center_path.size() - 7) == ".engine";
    bool kp_is_engine     = kp_path.size() > 7 &&
                            kp_path.substr(kp_path.size() - 7) == ".engine";

    if (center_is_engine && kp_is_engine) {
        // ---- TensorRT path ----
        auto loadEngineBytes = [](const std::string &path,
                                  std::vector<char> &buf) -> bool {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) return false;
            f.seekg(0, std::ios::end);
            buf.resize(f.tellg());
            f.seekg(0, std::ios::beg);
            f.read(buf.data(), (std::streamsize)buf.size());
            return f.good();
        };

        std::vector<char> center_buf, kp_buf;
        if (!loadEngineBytes(center_path, center_buf)) {
            std::cerr << "[JarvisInfer] Cannot open: " << center_path << "\n";
            return false;
        }
        if (!loadEngineBytes(kp_path, kp_buf)) {
            std::cerr << "[JarvisInfer] Cannot open: " << kp_path << "\n";
            return false;
        }

        trt_runtime_ = nvinfer1::createInferRuntime(trt_logger_);
        if (!trt_runtime_) {
            std::cerr << "[JarvisInfer] Failed to create TRT runtime\n";
            return false;
        }

        center_engine_ = trt_runtime_->deserializeCudaEngine(center_buf.data(),
                                                              center_buf.size());
        kp_engine_     = trt_runtime_->deserializeCudaEngine(kp_buf.data(),
                                                              kp_buf.size());
        if (!center_engine_ || !kp_engine_) {
            std::cerr << "[JarvisInfer] Failed to deserialize TRT engine\n";
            return false;
        }

        center_ctx_ = center_engine_->createExecutionContext();
        kp_ctx_     = kp_engine_->createExecutionContext();

        // Discover tensor names
        auto discoverNames = [](nvinfer1::ICudaEngine *eng,
                                std::string &in_name,
                                std::string &out_name) {
            for (int i = 0; i < eng->getNbIOTensors(); ++i) {
                const char *name = eng->getIOTensorName(i);
                if (eng->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
                    in_name = name;
                else
                    out_name = name;
            }
        };
        discoverNames(center_engine_, center_in_name_, center_out_name_);
        discoverNames(kp_engine_,     kp_in_name_,    kp_out_name_);

        // Discover center and KP output sizes from engine tensor shapes
        {
            auto co = center_engine_->getTensorShape(center_out_name_.c_str());
            if (co.nbDims >= 4) center_out_size_ = (int)co.d[2];
            auto ki = kp_engine_->getTensorShape(kp_in_name_.c_str());
            if (ki.nbDims >= 4) kp_input_size_ = (int)ki.d[2];
            auto ko = kp_engine_->getTensorShape(kp_out_name_.c_str());
            if (ko.nbDims >= 4) kp_out_size_ = (int)ko.d[2];
        }
        std::cout << "[JarvisInfer] Center out=" << center_out_size_
                  << "  KP input=" << kp_input_size_
                  << "  KP out=" << kp_out_size_ << "\n";

        // Allocate GPU buffers
        cudaStreamCreate(&stream_);
        cudaMalloc(&center_in_d_,  1 * 3 * CENTER_SIZE * CENTER_SIZE * sizeof(float));
        cudaMalloc(&center_out_d_, 1 * 1 * center_out_size_ * center_out_size_ * sizeof(float));
        cudaMalloc(&kp_in_d_,      1 * 3 * kp_input_size_ * kp_input_size_ * sizeof(float));
        cudaMalloc(&kp_out_d_,     1 * NUM_KP * kp_out_size_ * kp_out_size_ * sizeof(float));

        // Set persistent tensor addresses
        center_ctx_->setTensorAddress(center_in_name_.c_str(),  center_in_d_);
        center_ctx_->setTensorAddress(center_out_name_.c_str(), center_out_d_);
        kp_ctx_->setTensorAddress(kp_in_name_.c_str(),  kp_in_d_);
        kp_ctx_->setTensorAddress(kp_out_name_.c_str(), kp_out_d_);

        use_trt_ = true;
        loaded_  = true;
        std::cout << "[JarvisInfer] Loaded TRT engines OK\n";
        return true;
    }

    // ---- LibTorch fallback path (.pt) ----
    try {
        center_module_ = torch::jit::load(center_path, torch::kCPU);
        center_module_.eval();
        kp_module_ = torch::jit::load(kp_path, torch::kCPU);
        kp_module_.eval();

        // Probe output shapes with dummy forward passes
        {
            auto device = torch::kCPU;
            torch::NoGradGuard ng;
            try {
                auto probe_c = torch::zeros({1, 3, CENTER_SIZE, CENTER_SIZE}, device);
                auto out_c   = center_module_.forward({probe_c});
                // Model may return a tensor or a tuple — take first element
                at::Tensor hmap_c = out_c.isTuple()
                    ? out_c.toTuple()->elements()[0].toTensor()
                    : out_c.toTensor();
                center_out_size_ = (int)hmap_c.size(2);

                auto probe_kp = torch::zeros({1, 3, kp_input_size_, kp_input_size_}, device);
                auto out_kp   = kp_module_.forward({probe_kp});
                at::Tensor hmap_kp = out_kp.isTuple()
                    ? out_kp.toTuple()->elements()[0].toTensor()
                    : out_kp.toTensor();
                kp_out_size_ = (int)hmap_kp.size(2);

                std::cout << "[JarvisInfer] Center out=" << center_out_size_
                          << "  KP input=" << kp_input_size_
                          << "  KP out=" << kp_out_size_ << "\n";
            } catch (const std::exception &e) {
                std::cerr << "[JarvisInfer] Probe failed: " << e.what() << "\n";
            }
        }

        use_trt_ = false;
        loaded_  = true;
        std::cout << "[JarvisInfer] Loaded TorchScript models OK\n";
        return true;
    } catch (const c10::Error &e) {
        std::cerr << "[JarvisInfer] torch::jit::load failed: " << e.what() << "\n";
        return false;
    }
}

// ---------------------------------------------------------------------------
// Preprocessing helpers
// ---------------------------------------------------------------------------
at::Tensor JarvisInfer::preprocessFrame(unsigned char *frame_rgba, int width,
                                         int height, int out_size) {
    // RGBA uint8 → float [0,1] tensor (H, W, 4)
    auto rgba = torch::from_blob(frame_rgba, {height, width, 4}, torch::kUInt8)
                    .to(torch::kFloat32)
                    .div(255.0f);

    // Extract R, G, B channels (permute gives (4,H,W) so slice dim 0)
    auto chw = rgba.permute({2, 0, 1}); // (4, H, W)
    auto R   = chw[0];                  // (H, W)
    auto G   = chw[1];
    auto B   = chw[2];

    // Normalize with BGR mean/std (channel 0 = B, 1 = G, 2 = R)
    auto Bn = (B - BGR_MEAN[0]) / BGR_STD[0];
    auto Gn = (G - BGR_MEAN[1]) / BGR_STD[1];
    auto Rn = (R - BGR_MEAN[2]) / BGR_STD[2];

    auto img = torch::stack({Bn, Gn, Rn}, 0).unsqueeze(0); // (1, 3, H, W)

    // Resize to out_size × out_size
    img = torch::nn::functional::interpolate(
        img,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{out_size, out_size})
            .mode(torch::kBilinear)
            .align_corners(false));

    return img.contiguous();
}

at::Tensor JarvisInfer::cropAroundCenter(unsigned char *frame_rgba, int width,
                                          int height, int cx_px, int cy_px) {
    // Build full-image BGR-normalized tensor (1, 3, H, W)
    auto rgba = torch::from_blob(frame_rgba, {height, width, 4}, torch::kUInt8)
                    .to(torch::kFloat32)
                    .div(255.0f);
    auto chw = rgba.permute({2, 0, 1}); // (4, H, W)
    auto Bn  = (chw[2] - BGR_MEAN[0]) / BGR_STD[0];
    auto Gn  = (chw[1] - BGR_MEAN[1]) / BGR_STD[1];
    auto Rn  = (chw[0] - BGR_MEAN[2]) / BGR_STD[2];
    auto img = torch::stack({Bn, Gn, Rn}, 0).unsqueeze(0); // (1, 3, H, W)

    // Replicate-pad by kp_input_size_/2 on all sides so any crop is in-bounds
    int half = kp_input_size_ / 2;
    auto padded = torch::nn::functional::pad(
        img,
        torch::nn::functional::PadFuncOptions({half, half, half, half})
            .mode(torch::kReplicate)); // (1, 3, H+kp_input_size_, W+kp_input_size_)

    // In padded space, original pixel (x, y) maps to padded (x+half, y+half).
    // The function is called with cx_px = cx_center - half, cy_px = cy_center - half,
    // so the desired crop top-left in padded space is (cx_px+half, cy_px+half) = (cx_center, cy_center).
    int y0 = std::max(0, cy_px + half);
    int x0 = std::max(0, cx_px + half);
    auto crop = padded
                    .slice(2, y0, y0 + kp_input_size_)
                    .slice(3, x0, x0 + kp_input_size_)
                    .contiguous();

    return crop; // (1, 3, kp_input_size_, kp_input_size_)
}

std::pair<int, int> JarvisInfer::argmax2D(const at::Tensor &heatmap) {
    // heatmap: (H, W)
    auto flat    = heatmap.flatten();
    auto max_idx = flat.argmax().item<int64_t>();
    int  W       = (int)heatmap.size(1);
    int  row     = (int)(max_idx / W);
    int  col     = (int)(max_idx % W);
    return {row, col};
}

// ---------------------------------------------------------------------------
// predict() — dispatch to TRT or LibTorch
// ---------------------------------------------------------------------------
JarvisInfer::JarvisResult JarvisInfer::predict(unsigned char *frame_rgba,
                                                int width, int height) {
    if (!loaded_) return JarvisResult{};
    if (use_trt_) return predictTRT(frame_rgba, width, height);
    return predictTorch(frame_rgba, width, height);
}

// ---------------------------------------------------------------------------
// LibTorch inference
// ---------------------------------------------------------------------------
JarvisInfer::JarvisResult JarvisInfer::predictTorch(unsigned char *frame_rgba,
                                                     int width, int height) {
    JarvisResult result{};
    result.detected = false;

    torch::NoGradGuard no_grad;
    auto device = torch::kCPU;

    // --- CenterDetect ---
    auto center_in = preprocessFrame(frame_rgba, width, height, CENTER_SIZE);
    center_in       = center_in.to(device);

    at::Tensor center_out;
    try {
        auto raw = center_module_.forward({center_in});
        center_out = raw.isTuple()
            ? raw.toTuple()->elements()[0].toTensor()
            : raw.toTensor();
    } catch (const c10::Error &e) {
        std::cerr << "[JarvisInfer] CenterDetect forward failed: " << e.what() << "\n";
        return result;
    }
    // center_out: (1, 1, center_out_size_, center_out_size_)
    auto hmap_c = center_out.squeeze().cpu();

    auto [row_c, col_c] = argmax2D(hmap_c);
    float center_conf   = torch::sigmoid(hmap_c[row_c][col_c]).item<float>();

    // Scale argmax from output heatmap space to original image coords
    float cx_px = (float)col_c / center_out_size_ * width;
    float cy_px = (float)row_c / center_out_size_ * height;

    result.cx          = cx_px;
    result.cy          = cy_px;
    result.center_conf = center_conf;

    if (center_conf < conf_threshold_) return result;

    // --- KeypointDetect ---
    int cx_i  = (int)std::round(cx_px);
    int cy_i  = (int)std::round(cy_px);
    int half  = kp_input_size_ / 2;

    auto kp_in = cropAroundCenter(frame_rgba, width, height, cx_i - half, cy_i - half);
    kp_in       = kp_in.to(device);

    at::Tensor kp_out;
    try {
        auto raw = kp_module_.forward({kp_in});
        kp_out = raw.isTuple()
            ? raw.toTuple()->elements()[0].toTensor()
            : raw.toTensor();
    } catch (const c10::Error &e) {
        std::cerr << "[JarvisInfer] KeypointDetect forward failed: " << e.what() << "\n";
        return result;
    }
    // kp_out: (1, 24, kp_out_size_, kp_out_size_)
    auto kp_cpu = kp_out.squeeze(0).cpu(); // (24, kp_out_size_, kp_out_size_)

    // Scale factor from kp_out_size_ back to kp_input_size_ (heatmap → crop space)
    float scale = (float)kp_input_size_ / kp_out_size_; // = 2.0

    // Top-left of crop in original image
    float crop_x0 = cx_px - half;
    float crop_y0 = cy_px - half;

    for (int k = 0; k < NUM_KP; ++k) {
        auto hmap_k          = kp_cpu[k]; // (kp_out_size_, kp_out_size_)
        auto [row_k, col_k]  = argmax2D(hmap_k);
        float kp_conf        = torch::sigmoid(hmap_k[row_k][col_k]).item<float>();

        // Convert heatmap position → crop pixel → original image pixel
        float kp_x = crop_x0 + (float)col_k * scale;
        float kp_y = crop_y0 + (float)row_k * scale;

        result.kp[k][0] = kp_x;
        result.kp[k][1] = kp_y;
        result.kp[k][2] = kp_conf;
    }

    result.detected = true;
    return result;
}

// ---------------------------------------------------------------------------
// TensorRT inference
// ---------------------------------------------------------------------------
JarvisInfer::JarvisResult JarvisInfer::predictTRT(unsigned char *frame_rgba,
                                                   int width, int height) {
    JarvisResult result{};
    result.detected = false;

    // --- CenterDetect ---
    auto center_in = preprocessFrame(frame_rgba, width, height, CENTER_SIZE).cpu().contiguous();

    cudaMemcpyAsync(center_in_d_, center_in.data_ptr<float>(),
                    1 * 3 * CENTER_SIZE * CENTER_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    if (!center_ctx_->enqueueV3(stream_)) {
        std::cerr << "[JarvisInfer] CenterDetect TRT enqueue failed\n";
        return result;
    }

    std::vector<float> center_out_h(1 * 1 * center_out_size_ * center_out_size_);
    cudaMemcpyAsync(center_out_h.data(), center_out_d_,
                    center_out_h.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Find argmax in (center_out_size_ × center_out_size_) flat buffer
    int max_idx_c = (int)(std::max_element(center_out_h.begin(), center_out_h.end()) -
                          center_out_h.begin());
    int row_c     = max_idx_c / center_out_size_;
    int col_c     = max_idx_c % center_out_size_;
    float raw_c   = center_out_h[max_idx_c];
    float center_conf = 1.0f / (1.0f + std::exp(-raw_c)); // sigmoid

    float cx_px = (float)col_c / center_out_size_ * width;
    float cy_px = (float)row_c / center_out_size_ * height;

    result.cx          = cx_px;
    result.cy          = cy_px;
    result.center_conf = center_conf;

    if (center_conf < conf_threshold_) return result;

    // --- KeypointDetect ---
    int half = kp_input_size_ / 2;
    int cx_i = (int)std::round(cx_px);
    int cy_i = (int)std::round(cy_px);

    auto kp_in = cropAroundCenter(frame_rgba, width, height, cx_i - half, cy_i - half)
                     .cpu().contiguous();

    cudaMemcpyAsync(kp_in_d_, kp_in.data_ptr<float>(),
                    1 * 3 * kp_input_size_ * kp_input_size_ * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    if (!kp_ctx_->enqueueV3(stream_)) {
        std::cerr << "[JarvisInfer] KeypointDetect TRT enqueue failed\n";
        return result;
    }

    std::vector<float> kp_out_h(1 * NUM_KP * kp_out_size_ * kp_out_size_);
    cudaMemcpyAsync(kp_out_h.data(), kp_out_d_,
                    kp_out_h.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    float scale   = (float)kp_input_size_ / kp_out_size_;
    float crop_x0 = cx_px - half;
    float crop_y0 = cy_px - half;
    int   map_size = kp_out_size_ * kp_out_size_;

    for (int k = 0; k < NUM_KP; ++k) {
        const float *ch = kp_out_h.data() + k * map_size;
        int max_idx_k   = (int)(std::max_element(ch, ch + map_size) - ch);
        int row_k       = max_idx_k / kp_out_size_;
        int col_k       = max_idx_k % kp_out_size_;
        float raw_k     = ch[max_idx_k];
        float kp_conf   = 1.0f / (1.0f + std::exp(-raw_k));

        result.kp[k][0] = crop_x0 + (float)col_k * scale;
        result.kp[k][1] = crop_y0 + (float)row_k * scale;
        result.kp[k][2] = kp_conf;
    }

    result.detected = true;
    return result;
}
