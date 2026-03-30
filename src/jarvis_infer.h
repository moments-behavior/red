#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <torch/script.h>

// TensorRT logger (minimal)
class JarvisTRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override;
};

class JarvisInfer {
public:
    struct JarvisResult {
        float cx, cy;        // center in original image pixel coords
        float center_conf;   // peak value from center heatmap
        float kp[24][3];     // [x_px, y_px, conf] per keypoint
        bool detected;
    };

    JarvisInfer();
    ~JarvisInfer();

    // Load engines: .engine → TensorRT, .pt → LibTorch fallback
    bool load(const std::string &center_path, const std::string &kp_path);
    bool isLoaded() const { return loaded_; }
    void setConfThreshold(float t) { conf_threshold_ = t; }

    // Run JARVIS 2D inference on one RGBA uint8 camera frame
    JarvisResult predict(unsigned char *frame_rgba, int width, int height);

private:
    JarvisResult predictTRT(unsigned char *frame_rgba, int width, int height);
    JarvisResult predictTorch(unsigned char *frame_rgba, int width, int height);

    // Preprocess RGBA frame to (1, 3, out_size, out_size) BGR-normalized float tensor
    at::Tensor preprocessFrame(unsigned char *frame_rgba, int width, int height,
                                int out_size);

    // Crop (1, 3, KP_SIZE, KP_SIZE) around center from full-frame tensor
    at::Tensor cropAroundCenter(unsigned char *frame_rgba, int width, int height,
                                int cx_px, int cy_px);

    // Find argmax location in a 2D heatmap tensor (H, W) → returns {row, col}
    std::pair<int, int> argmax2D(const at::Tensor &heatmap);

    bool loaded_ = false;
    bool use_trt_ = false;
    float conf_threshold_ = 0.3f;

    // TRT members
    JarvisTRTLogger trt_logger_;
    nvinfer1::IRuntime *trt_runtime_ = nullptr;
    nvinfer1::ICudaEngine *center_engine_ = nullptr;
    nvinfer1::ICudaEngine *kp_engine_ = nullptr;
    nvinfer1::IExecutionContext *center_ctx_ = nullptr;
    nvinfer1::IExecutionContext *kp_ctx_ = nullptr;
    std::string center_in_name_, center_out_name_;
    std::string kp_in_name_, kp_out_name_;
    void *center_in_d_ = nullptr;   // GPU: (1,3,320,320) float
    void *center_out_d_ = nullptr;  // GPU: (1,1,320,320) float
    void *kp_in_d_ = nullptr;       // GPU: (1,3,832,832) float
    void *kp_out_d_ = nullptr;      // GPU: (1,24,416,416) float
    cudaStream_t stream_ = nullptr;

    // LibTorch members
    torch::jit::Module center_module_;
    torch::jit::Module kp_module_;

    static constexpr int CENTER_SIZE = 320;  // center detect input size
    static constexpr int NUM_KP = 24;
    // Discovered at load time from the model's actual output shape
    int center_out_size_ = 80;   // CenterDetect output H==W (stride 4)
    int kp_input_size_   = 704;  // KEYPOINTDETECT.BOUNDING_BOX_SIZE
    int kp_out_size_     = 176;  // KeypointDetect output H==W (stride 4)
};
