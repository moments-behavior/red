#pragma once
// decode_backend.h -- runtime choice between hardware and software video decode.
//
// Hardware means NVDEC + CUDA-GL interop on Linux/Windows, VideoToolbox +
// Metal on macOS. Software means libavcodec decoding straight to RGBA in host
// memory (sw_decoder.h), uploaded with a plain texture upload.
//
// The choice is resolved once, on first call, and cached for the process
// lifetime -- decoder threads, the render loop and the teardown path all have
// to agree on it, and a probe that could answer differently mid-run would
// leave frames in the wrong kind of memory.
#include "red_build_config.h"

namespace red {

enum class DecodeBackend {
    Hardware,
    Software,
};

// Resolved on first call; every later call returns the cached answer.
DecodeBackend decode_backend();

inline bool decode_backend_is_software() {
    return decode_backend() == DecodeBackend::Software;
}

// "hardware" / "software", and a short human-readable reason for the choice
// (which probe failed, or which env var forced it). Both valid for the
// process lifetime.
const char *decode_backend_name();
const char *decode_backend_reason();

// Software decode thread budget. Call sw_decode_set_camera_count() once,
// before spawning decoder threads: N cameras each letting libavcodec size its
// own thread pool from hardware_concurrency() oversubscribes the machine
// badly on a 16-camera rig.
void sw_decode_set_camera_count(int num_cams);
int sw_decode_threads_per_camera();

} // namespace red
