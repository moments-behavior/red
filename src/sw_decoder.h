#pragma once
// sw_decoder.h -- libavcodec software decode path.
//
// Same contract as the NVDEC decoder in decoder.cpp: one thread per camera,
// pulling packets from the shared FFmpegDemuxer, publishing RGBA frames into
// the camera's PictureBuffer ring under the same seek and canonical-timeline
// (sync_plan) semantics. The only difference is where the pixels live --
// always host memory here, never a CUdeviceptr.
//
// Platform-neutral on purpose: it builds on macOS too, where
// RED_DECODE_BACKEND=sw selects it over VideoToolbox. That is the only way to
// exercise this path on a machine with no NVIDIA GPU.
#include "decoder.h"

#include <string>

class FFmpegDemuxer;
namespace sync_plan {
struct SyncCam;
}

// Signature matches decoder_process() so the dispatcher can forward verbatim.
//
// use_cpu_buffer is accepted for that symmetry but not consulted: this decoder
// always memcpy's into host memory, so the caller must allocate the ring slots
// with calloc rather than cudaMalloc whenever the software backend is active
// (see render_allocate_scene_memory).
void sw_decoder_process(DecoderContext *dc_context, FFmpegDemuxer *demuxer,
                        std::string cam_name, PictureBuffer *display_buffer,
                        int size_of_buffer, SeekInfo *seek_info,
                        bool use_cpu_buffer,
                        const sync_plan::SyncCam *sync_cam = nullptr);
