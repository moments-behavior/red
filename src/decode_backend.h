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
#include <string>

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

// Can the hardware decoder actually handle this stream?
//
// NVDEC advertises what it supports per codec, chroma and resolution, and a
// GPU that cannot take a stream will simply produce no frames. NvDecoder does
// ask -- and throws "Codec not supported on this GPU" -- but it asks from
// inside a CUVID parser callback, and a C++ exception does not travel back
// through NVIDIA's C frame: the parser stops and nothing is printed. So the
// question has to be asked here, before any decoding starts.
//
// Returns true when hardware is not in use or when it can decode the stream.
// `why` is filled in when it cannot.
// `device_index` is the CUDA device red will decode on, so a multi-GPU
// machine is asked about the right one; ignored on macOS.
bool hw_can_decode_stream(int av_codec_id, int chroma_format,
                          int bit_depth_minus8, int width, int height,
                          int device_index, std::string *why);

// Switch to software for the rest of the process, with a reason. Only safe
// before any decode buffers are allocated or decoder threads spawned --
// load_videos calls it between opening the demuxers and allocating, which is
// the one point where the codec is known and nothing has been built on the
// earlier answer yet.
void decode_backend_force_software(const std::string &why);

// Software decode thread budget. Call sw_decode_set_camera_count() once,
// before spawning decoder threads: N cameras each letting libavcodec size its
// own thread pool from hardware_concurrency() oversubscribes the machine
// badly on a 16-camera rig.
void sw_decode_set_camera_count(int num_cams);
int sw_decode_threads_per_camera();

} // namespace red
