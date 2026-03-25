// Test ONNX Runtime end-to-end inference on CPU and CUDA EPs.
// Programmatically generates a minimal ONNX model (Add op) as raw protobuf
// bytes -- no protobuf library dependency required.
//
// Build (MSVC):
//   cl /std:c++17 /EHsc /I..\lib\onnxruntime\include
//      test_ort_inference.cpp ..\lib\onnxruntime\lib\onnxruntime.lib
// Run:
//   test_ort_inference.exe

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Simple test framework
// ---------------------------------------------------------------------------
static int g_passed = 0, g_failed = 0;

#define TEST_BEGIN(name)                              \
    do {                                              \
        const char* _test_name = (name);              \
        printf("  TEST: %-55s ", _test_name);         \
        try {

#define TEST_END()                                    \
            printf("PASS\n"); g_passed++;             \
        } catch (const Ort::Exception& e) {           \
            printf("FAIL (ORT): %s\n", e.what());    \
            g_failed++;                               \
        } catch (const std::exception& e) {           \
            printf("FAIL: %s\n", e.what());           \
            g_failed++;                               \
        }                                             \
    } while (0)

#define ASSERT_TRUE(cond, msg)                        \
    do {                                              \
        if (!(cond)) throw std::runtime_error(msg);   \
    } while (0)

#define ASSERT_NEAR(a, b, eps, msg)                   \
    do {                                              \
        if (std::fabs((a) - (b)) > (eps))             \
            throw std::runtime_error(msg);            \
    } while (0)

// ---------------------------------------------------------------------------
// Minimal protobuf encoder -- just enough to emit an ONNX ModelProto
// ---------------------------------------------------------------------------
// Protobuf wire types:  0 = varint, 2 = length-delimited
namespace pb {

static void append_varint(std::vector<uint8_t>& buf, uint64_t v) {
    do {
        uint8_t b = v & 0x7F;
        v >>= 7;
        if (v) b |= 0x80;
        buf.push_back(b);
    } while (v);
}

static void append_tag(std::vector<uint8_t>& buf, int field, int wire) {
    append_varint(buf, (uint64_t(field) << 3) | wire);
}

static void append_bytes(std::vector<uint8_t>& buf, int field,
                         const std::vector<uint8_t>& data) {
    append_tag(buf, field, 2);
    append_varint(buf, data.size());
    buf.insert(buf.end(), data.begin(), data.end());
}

static void append_bytes(std::vector<uint8_t>& buf, int field,
                         const void* data, size_t len) {
    append_tag(buf, field, 2);
    append_varint(buf, len);
    auto p = static_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + len);
}

static void append_string(std::vector<uint8_t>& buf, int field,
                           const char* s) {
    append_bytes(buf, field, s, strlen(s));
}

static void append_int64(std::vector<uint8_t>& buf, int field, int64_t v) {
    append_tag(buf, field, 0);
    append_varint(buf, static_cast<uint64_t>(v));
}

// Encode a repeated-packed int64 field.
static void append_packed_int64(std::vector<uint8_t>& buf, int field,
                                 const std::vector<int64_t>& vals) {
    std::vector<uint8_t> inner;
    for (auto v : vals) append_varint(inner, static_cast<uint64_t>(v));
    append_bytes(buf, field, inner);
}

// Encode a repeated-packed float field.
static void append_packed_float(std::vector<uint8_t>& buf, int field,
                                 const std::vector<float>& vals) {
    size_t nbytes = vals.size() * sizeof(float);
    append_tag(buf, field, 2);
    append_varint(buf, nbytes);
    auto p = reinterpret_cast<const uint8_t*>(vals.data());
    buf.insert(buf.end(), p, p + nbytes);
}

}  // namespace pb

// ---------------------------------------------------------------------------
// Build a minimal ONNX model: Z = Add(X, Y)
//   X : float input  [N, C]  (dynamic N, C fixed)
//   Y : float constant [C]   = {1, 1, 1}
//   Z : float output [N, C]
//
// ONNX protobuf schema reference (field numbers):
//   ModelProto:  1=ir_version, 7=graph, 8=opset_import
//   GraphProto:  1=node, 2=name, 5=initializer, 11=input, 12=output
//   NodeProto:   1=input(repeated string), 2=output(repeated string),
//                3=name, 4=op_type
//   TensorProto: 1=dims(packed int64), 2=data_type, 4=float_data(packed),
//                8=name
//   ValueInfoProto:  1=name, 2=type
//   TypeProto:       1=tensor_type
//   TypeProto.Tensor:  1=elem_type, 2=shape
//   TensorShapeProto:  1=dim (repeated)
//   TensorShapeProto.Dim:  1=dim_value(int64), 2=dim_param(string)
//   OperatorSetIdProto: 2=version
// ---------------------------------------------------------------------------

// Make a TensorShapeProto.Dimension (dim_value variant)
static std::vector<uint8_t> make_dim_value(int64_t v) {
    std::vector<uint8_t> buf;
    pb::append_int64(buf, 1, v);  // dim_value
    return buf;
}

// Make a TensorShapeProto.Dimension (dim_param variant -- symbolic)
static std::vector<uint8_t> make_dim_param(const char* s) {
    std::vector<uint8_t> buf;
    pb::append_string(buf, 2, s);  // dim_param
    return buf;
}

// Make a TensorShapeProto from a list of dimensions
static std::vector<uint8_t> make_shape(
    const std::vector<std::vector<uint8_t>>& dims) {
    std::vector<uint8_t> buf;
    for (auto& d : dims) pb::append_bytes(buf, 1, d);  // repeated dim
    return buf;
}

// TypeProto.Tensor: elem_type + shape
static std::vector<uint8_t> make_tensor_type(
    int32_t elem_type, const std::vector<uint8_t>& shape) {
    std::vector<uint8_t> buf;
    pb::append_int64(buf, 1, elem_type);  // elem_type (1 = FLOAT)
    pb::append_bytes(buf, 2, shape);       // shape
    return buf;
}

// TypeProto: tensor_type
static std::vector<uint8_t> make_type_proto(
    const std::vector<uint8_t>& tensor_type) {
    std::vector<uint8_t> buf;
    pb::append_bytes(buf, 1, tensor_type);  // tensor_type (field 1)
    return buf;
}

// ValueInfoProto: name + type
static std::vector<uint8_t> make_value_info(
    const char* name, const std::vector<uint8_t>& type_proto) {
    std::vector<uint8_t> buf;
    pb::append_string(buf, 1, name);
    pb::append_bytes(buf, 2, type_proto);
    return buf;
}

// TensorProto (float initializer)
static std::vector<uint8_t> make_float_tensor(
    const char* name, const std::vector<int64_t>& dims,
    const std::vector<float>& data) {
    std::vector<uint8_t> buf;
    pb::append_packed_int64(buf, 1, dims);   // dims
    pb::append_int64(buf, 2, 1);              // data_type = FLOAT
    pb::append_packed_float(buf, 4, data);    // float_data
    pb::append_string(buf, 8, name);          // name
    return buf;
}

// NodeProto: Add(X, Y) -> Z
static std::vector<uint8_t> make_add_node(
    const char* x, const char* y, const char* z, const char* name) {
    std::vector<uint8_t> buf;
    pb::append_string(buf, 1, x);     // input
    pb::append_string(buf, 1, y);     // input
    pb::append_string(buf, 2, z);     // output
    pb::append_string(buf, 3, name);  // name
    pb::append_string(buf, 4, "Add"); // op_type
    return buf;
}

// OperatorSetIdProto: version only (domain = "" = default ONNX domain)
static std::vector<uint8_t> make_opset(int64_t version) {
    std::vector<uint8_t> buf;
    pb::append_int64(buf, 2, version);  // version
    return buf;
}

// Build the complete ONNX model as bytes. C = number of channels.
static std::vector<uint8_t> build_add_model(int64_t C) {
    // --- shapes ---
    auto dim_N = make_dim_param("N");   // dynamic batch
    auto dim_C = make_dim_value(C);

    auto shape_NC = make_shape({dim_N, dim_C});
    auto shape_C  = make_shape({dim_C});

    auto tt_NC = make_tensor_type(1, shape_NC);  // FLOAT
    auto tt_C  = make_tensor_type(1, shape_C);

    auto tp_NC = make_type_proto(tt_NC);
    auto tp_C  = make_type_proto(tt_C);

    auto vi_X = make_value_info("X", tp_NC);
    auto vi_Y = make_value_info("Y", tp_C);
    auto vi_Z = make_value_info("Z", tp_NC);

    // --- initializer for Y (all ones) ---
    std::vector<float> ones(static_cast<size_t>(C), 1.0f);
    auto init_Y = make_float_tensor("Y", {C}, ones);

    // --- node: Z = Add(X, Y) ---
    auto node = make_add_node("X", "Y", "Z", "add0");

    // --- graph ---
    std::vector<uint8_t> graph;
    pb::append_bytes(graph, 1, node);      // node
    pb::append_string(graph, 2, "g");      // name
    pb::append_bytes(graph, 5, init_Y);    // initializer
    pb::append_bytes(graph, 11, vi_X);     // input (X)
    pb::append_bytes(graph, 11, vi_Y);     // input (Y -- also in initializer)
    pb::append_bytes(graph, 12, vi_Z);     // output

    // --- opset ---
    auto opset = make_opset(13);

    // --- model ---
    std::vector<uint8_t> model;
    pb::append_int64(model, 1, 7);           // ir_version = 7
    pb::append_bytes(model, 7, graph);       // graph
    pb::append_bytes(model, 8, opset);       // opset_import

    return model;
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------
struct InferenceResult {
    std::vector<float> output;
    double elapsed_ms;
};

static InferenceResult run_inference(Ort::Session& session,
                                     const std::vector<float>& input,
                                     const std::vector<int64_t>& shape) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(input.data()), input.size(),
        shape.data(), shape.size());

    const char* input_names[]  = {"X"};
    const char* output_names[] = {"Z"};

    auto t0 = std::chrono::high_resolution_clock::now();
    auto results = session.Run(Ort::RunOptions{nullptr},
                               input_names, &input_tensor, 1,
                               output_names, 1);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& out = results[0];
    const float* data = out.GetTensorData<float>();
    auto info = out.GetTensorTypeAndShapeInfo();
    size_t count = static_cast<size_t>(info.GetElementCount());

    return {std::vector<float>(data, data + count), ms};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_model_creation() {
    TEST_BEGIN("Generate minimal ONNX model bytes") {
        auto model = build_add_model(3);
        ASSERT_TRUE(model.size() > 20, "Model too small");
        // Quick sanity: starts with protobuf tag for field 1, varint
        ASSERT_TRUE(model[0] == 0x08, "Bad protobuf start tag");
    } TEST_END();
}

static void test_load_session_cpu(Ort::Env& env,
                                   const std::vector<uint8_t>& model_data) {
    TEST_BEGIN("Load session on CPU EP") {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, model_data.data(), model_data.size(), opts);
        ASSERT_TRUE(session.GetInputCount() == 1 || session.GetInputCount() == 2,
                     "Expected 1 or 2 inputs");
        ASSERT_TRUE(session.GetOutputCount() == 1, "Expected 1 output");
    } TEST_END();
}

static void test_io_names_and_shapes(Ort::Env& env,
                                      const std::vector<uint8_t>& model_data) {
    TEST_BEGIN("Verify input/output names and shapes") {
        Ort::SessionOptions opts;
        Ort::Session session(env, model_data.data(), model_data.size(), opts);

        auto in_names  = session.GetInputNames();
        auto out_names = session.GetOutputNames();

        // The runtime may or may not expose Y as an input (since it is
        // both an input and an initializer).  X must always be there.
        bool found_x = false;
        for (auto& n : in_names) if (n == "X") found_x = true;
        ASSERT_TRUE(found_x, "Input 'X' not found");

        ASSERT_TRUE(out_names.size() == 1 && out_names[0] == "Z",
                     "Output 'Z' not found");

        // Shape of X should be [N, 3] -- N may be -1 (dynamic)
        auto ti = session.GetInputTypeInfo(0);
        auto tensor_info = ti.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        ASSERT_TRUE(shape.size() == 2, "Expected rank-2 input");
        ASSERT_TRUE(shape[1] == 3, "Expected dim[1] == 3");

        printf("[shape=(%lld,%lld)] ",
               (long long)shape[0], (long long)shape[1]);
    } TEST_END();
}

static void test_cpu_inference(Ort::Env& env,
                                const std::vector<uint8_t>& model_data) {
    TEST_BEGIN("CPU inference: Z = X + 1") {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, model_data.data(), model_data.size(), opts);

        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        std::vector<int64_t> shape = {1, 3};
        auto result = run_inference(session, input, shape);

        ASSERT_TRUE(result.output.size() == 3, "Output size mismatch");
        ASSERT_NEAR(result.output[0], 2.0f, 1e-5f, "output[0] != 2.0");
        ASSERT_NEAR(result.output[1], 3.0f, 1e-5f, "output[1] != 3.0");
        ASSERT_NEAR(result.output[2], 4.0f, 1e-5f, "output[2] != 4.0");
        printf("[%.1fms] ", result.elapsed_ms);
    } TEST_END();
}

static bool g_cuda_available = false;

static void test_cuda_inference(Ort::Env& env,
                                 const std::vector<uint8_t>& model_data) {
    TEST_BEGIN("CUDA EP inference: Z = X + 1") {
        // Check if CUDA EP is available
        auto providers = Ort::GetAvailableProviders();
        bool has_cuda = false;
        for (auto& p : providers)
            if (p == "CUDAExecutionProvider") has_cuda = true;
        ASSERT_TRUE(has_cuda, "CUDAExecutionProvider not available");

        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        cuda_opts.arena_extend_strategy = 1;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        opts.AppendExecutionProvider_CUDA(cuda_opts);

        Ort::Session session(env, model_data.data(), model_data.size(), opts);

        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        std::vector<int64_t> shape = {1, 3};
        auto result = run_inference(session, input, shape);

        ASSERT_TRUE(result.output.size() == 3, "Output size mismatch");
        ASSERT_NEAR(result.output[0], 2.0f, 1e-5f, "output[0] != 2.0");
        ASSERT_NEAR(result.output[1], 3.0f, 1e-5f, "output[1] != 3.0");
        ASSERT_NEAR(result.output[2], 4.0f, 1e-5f, "output[2] != 4.0");
        printf("[%.1fms] ", result.elapsed_ms);
        g_cuda_available = true;
    } TEST_END();
}

static void test_cuda_cpu_match(Ort::Env& env,
                                 const std::vector<uint8_t>& model_data) {
    TEST_BEGIN("CUDA vs CPU output match (larger input)") {
        ASSERT_TRUE(g_cuda_available, "Skipped: CUDA not available");

        const int64_t N = 64, C = 3;
        std::vector<float> input(N * C);
        for (size_t i = 0; i < input.size(); i++)
            input[i] = static_cast<float>(i) * 0.1f;
        std::vector<int64_t> shape = {N, C};

        // CPU
        {
            Ort::SessionOptions opts;
            Ort::Session session(env, model_data.data(),
                                 model_data.size(), opts);
            auto cpu_result = run_inference(session, input, shape);

            // CUDA
            Ort::SessionOptions cuda_so;
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            cuda_so.AppendExecutionProvider_CUDA(cuda_opts);
            Ort::Session cuda_session(env, model_data.data(),
                                      model_data.size(), cuda_so);
            auto gpu_result = run_inference(cuda_session, input, shape);

            ASSERT_TRUE(cpu_result.output.size() == gpu_result.output.size(),
                         "Size mismatch");
            float max_diff = 0.0f;
            for (size_t i = 0; i < cpu_result.output.size(); i++) {
                float d = std::fabs(cpu_result.output[i] -
                                    gpu_result.output[i]);
                if (d > max_diff) max_diff = d;
            }
            ASSERT_TRUE(max_diff < 1e-5f, "CPU/CUDA output divergence");
            printf("[max_diff=%.2e] ", max_diff);
        }
    } TEST_END();
}

static void test_batch_sizes(Ort::Env& env,
                              const std::vector<uint8_t>& model_data) {
    TEST_BEGIN("Dynamic batch sizes (1, 4, 16, 128)") {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Use CUDA if available, else CPU
        if (g_cuda_available) {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        }

        Ort::Session session(env, model_data.data(), model_data.size(), opts);

        const int64_t C = 3;
        int64_t batches[] = {1, 4, 16, 128};
        for (int64_t N : batches) {
            std::vector<float> input(static_cast<size_t>(N * C), 5.0f);
            std::vector<int64_t> shape = {N, C};
            auto result = run_inference(session, input, shape);
            ASSERT_TRUE(
                result.output.size() == static_cast<size_t>(N * C),
                "Wrong output size for batch");
            // All outputs should be 5.0 + 1.0 = 6.0
            for (size_t i = 0; i < result.output.size(); i++) {
                ASSERT_NEAR(result.output[i], 6.0f, 1e-5f,
                             "Wrong value in batch test");
            }
        }
        printf("[batches: 1,4,16,128 OK] ");
    } TEST_END();
}

static void test_throughput(Ort::Env& env,
                             const std::vector<uint8_t>& model_data) {
    const int ITERS = 1000;
    const int64_t N = 16, C = 3;
    std::vector<float> input(static_cast<size_t>(N * C), 2.5f);
    std::vector<int64_t> shape = {N, C};

    // --- CPU throughput ---
    {
        char label[128];
        snprintf(label, sizeof(label),
                 "CPU throughput (%d iters, batch %lld)", ITERS, (long long)N);
        TEST_BEGIN(label) {
            Ort::SessionOptions opts;
            opts.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            Ort::Session session(env, model_data.data(),
                                 model_data.size(), opts);

            // Warm up
            for (int i = 0; i < 10; i++)
                run_inference(session, input, shape);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++)
                run_inference(session, input, shape);
            auto t1 = std::chrono::high_resolution_clock::now();

            double total_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            double per_iter = total_ms / ITERS;
            double throughput = ITERS / (total_ms / 1000.0);
            printf("[%.3f ms/iter, %.0f infer/s] ", per_iter, throughput);
        } TEST_END();
    }

    // --- CUDA throughput ---
    if (g_cuda_available) {
        char label[128];
        snprintf(label, sizeof(label),
                 "CUDA throughput (%d iters, batch %lld)", ITERS,
                 (long long)N);
        TEST_BEGIN(label) {
            Ort::SessionOptions opts;
            opts.SetGraphOptimizationLevel(
                GraphOptimizationLevel::ORT_ENABLE_ALL);
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            Ort::Session session(env, model_data.data(),
                                 model_data.size(), opts);

            // Warm up
            for (int i = 0; i < 10; i++)
                run_inference(session, input, shape);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITERS; i++)
                run_inference(session, input, shape);
            auto t1 = std::chrono::high_resolution_clock::now();

            double total_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            double per_iter = total_ms / ITERS;
            double throughput = ITERS / (total_ms / 1000.0);
            printf("[%.3f ms/iter, %.0f infer/s] ", per_iter, throughput);
        } TEST_END();
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== ONNX Runtime End-to-End Inference Test ===\n\n");

    // List providers
    {
        auto providers = Ort::GetAvailableProviders();
        printf("  Available EPs:");
        for (auto& p : providers) printf(" %s", p.c_str());
        printf("\n\n");
    }

    // Build the model
    auto model_data = build_add_model(3);
    printf("  Model size: %zu bytes\n\n", model_data.size());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_ort_inference");

    // Run tests
    test_model_creation();
    test_load_session_cpu(env, model_data);
    test_io_names_and_shapes(env, model_data);
    test_cpu_inference(env, model_data);
    test_cuda_inference(env, model_data);
    test_cuda_cpu_match(env, model_data);
    test_batch_sizes(env, model_data);
    test_throughput(env, model_data);

    // Summary
    printf("\n========================================\n");
    printf("  PASSED: %d   FAILED: %d   TOTAL: %d\n",
           g_passed, g_failed, g_passed + g_failed);
    printf("========================================\n");

    return g_failed > 0 ? 1 : 0;
}
