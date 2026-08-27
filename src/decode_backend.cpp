#include "decode_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#if defined(RED_HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace red {
namespace {

DecodeBackend g_backend = DecodeBackend::Hardware;
std::string g_reason;
std::once_flag g_resolve_once;
std::atomic<int> g_num_cams{1};

// Accepts the names this project's docs use as well as the ones a user is
// likely to reach for first.
bool matches(const char *v, std::initializer_list<const char *> names) {
    for (const char *n : names)
        if (std::strcmp(v, n) == 0) return true;
    return false;
}

#if defined(RED_HAVE_CUDA)
bool probe_cuda(std::string &why) {
    // cudart, not the driver API: libcudart loads the driver itself with
    // dlopen and reports a missing or too-old one as an ordinary error code.
    // cuInit() through ck() would LOG(FATAL) -> exit(1) instead, which is the
    // silent disappearance this whole path exists to avoid.
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        why = std::string("cudaGetDeviceCount: ") + cudaGetErrorName(err);
        return false;
    }
    if (count <= 0) {
        why = "no CUDA devices present";
        return false;
    }
    // Raw cuInit, deliberately not wrapped in ck(): a failure here must fall
    // back, not terminate. NVDEC lives behind the driver API, so a driver that
    // cannot initialize rules the hardware path out even with a device listed.
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        const char *name = nullptr;
        cuGetErrorName(res, &name);
        why = std::string("cuInit: ") + (name ? name : "unknown error");
        return false;
    }
    return true;
}
#endif

void resolve() {
    const char *env = std::getenv("RED_DECODE_BACKEND");
    if (env && *env) {
        if (matches(env, {"sw", "cpu", "software", "ffmpeg"})) {
            g_backend = DecodeBackend::Software;
            g_reason = "forced by RED_DECODE_BACKEND";
            return;
        }
        if (matches(env, {"hw", "gpu", "hardware"})) {
            g_backend = DecodeBackend::Hardware;
            g_reason = "forced by RED_DECODE_BACKEND";
            return;
        }
        if (!matches(env, {"auto"})) {
            std::cerr << "[decode] ignoring RED_DECODE_BACKEND=" << env
                      << " (expected auto, hw or sw)\n";
        }
    }

#if defined(__APPLE__)
    // VideoToolbox is part of the OS on every Mac this builds for, so there is
    // nothing to probe; sw is reachable only by the env override above, which
    // is what exercises the software path on a developer machine.
    g_backend = DecodeBackend::Hardware;
    g_reason = "VideoToolbox";
#elif defined(RED_HAVE_CUDA)
    std::string why;
    if (probe_cuda(why)) {
        g_backend = DecodeBackend::Hardware;
        g_reason = "NVDEC available";
    } else {
        g_backend = DecodeBackend::Software;
        g_reason = why;
    }
#else
    g_backend = DecodeBackend::Software;
    g_reason = "built without CUDA (-DRED_ENABLE_CUDA=OFF)";
#endif
}

void resolve_once() {
    std::call_once(g_resolve_once, [] {
        resolve();
        std::cerr << "[decode] backend: "
                  << (g_backend == DecodeBackend::Software ? "software"
                                                           : "hardware")
                  << " (" << g_reason << ")\n";
    });
}

} // namespace

DecodeBackend decode_backend() {
    resolve_once();
    return g_backend;
}

const char *decode_backend_name() {
    resolve_once();
    return g_backend == DecodeBackend::Software ? "software" : "hardware";
}

const char *decode_backend_reason() {
    resolve_once();
    return g_reason.c_str();
}

void sw_decode_set_camera_count(int num_cams) {
    g_num_cams.store(std::max(1, num_cams));
}

int sw_decode_threads_per_camera() {
    if (const char *env = std::getenv("RED_SW_DECODE_THREADS")) {
        int n = std::atoi(env);
        if (n > 0) return n;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    // Cap at 4: libavcodec's frame threading delays output by thread_count
    // frames, which shows up as seek latency, and the returns flatten out well
    // before that on a single 1-2 MP stream.
    int per_cam = (int)hw / g_num_cams.load();
    return std::clamp(per_cam, 1, 4);
}

} // namespace red
