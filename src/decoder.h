#ifndef RED_DECODER
#define RED_DECODER
#include "red_build_config.h"
#include "ColorSpace.h"
#include "FFmpegDemuxer.h"
#include "NvCodecUtils.h"
#if defined(RED_HAVE_CUDA)
#include "NvDecoder.h"
#include <cuda.h>
#endif
#ifdef __APPLE__
extern "C" {
#include <libavcodec/avcodec.h>
}
#include <CoreVideo/CoreVideo.h>
#endif
#include <atomic>
#include <cstdint>

namespace sync_plan {
struct SyncCam;
}

struct SeekInfo {
    bool use_seek;
    bool seek_done;
    uint64_t seek_frame;
    bool seek_accurate;
};

// Byte order of PictureBuffer::frame. There is no single convention here --
// each platform's uploader dictates one, and a producer that guesses wrong
// shows up as swapped red and blue rather than as a crash:
//   macOS   metal_upload_texture() replaceRegion's straight into an
//           MTLPixelFormatBGRA8Unorm texture, so the bytes must be BGRA.
//   GL      render.cpp uploads GL_RGBA, matching NVDEC's Nv12ToColor32<RGBA32>.
#if defined(__APPLE__)
#define RED_FRAME_BGRA 1
#endif

struct PictureBuffer {
    // RGBA, or BGRA where RED_FRAME_BGRA is defined -- see above.
    unsigned char *frame;
    std::atomic<int> frame_number;
    std::atomic<bool> available_to_write;
    // Sync-fix mode: this slot is a duplicate standing in for a frame the
    // camera dropped. Written before the available_to_write=false publish, so
    // any consumer that sees the slot filled may read it.
    std::atomic<bool> dropped;
#ifdef __APPLE__
    // Phase 2/3: decoded CVPixelBuffer (retained by decoder, released by main thread)
    CVPixelBufferRef pixel_buffer;
#endif
};

struct DecoderContext {
    std::atomic<bool> decoding_flag;
    std::atomic<bool> stop_flag;
    int total_num_frame;
    int estimated_num_frames;
    int gpu_index;
    int seek_interval;
    double video_fps;
    // Canonical-timeline desync fix (sync_plan.h). When active, decoders emit
    // canonical trigger slots instead of mp4 frame indices: frame_number,
    // seek_frame, latest_decoded_frame and total/estimated counts are all in
    // canonical-slot space, and slots a camera dropped hold a duplicate of the
    // nearest decoded frame with PictureBuffer::dropped set. Decoders sample
    // the flag at thread start and at each seek servicing (every toggle issues
    // a seek), so a decoder never mixes modes within one seek epoch.
    std::atomic<bool> sync_fix_active;
    int64_t sync_canonical_len;
};

#if defined(RED_HAVE_CUDA)
void decoder_get_image_from_gpu(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth,
                                int nHeight);
#endif
void decoder_clear_buffer_with_constant_image(unsigned char *image_pt,
                                              int width, int height);
void decoder_print_one_display_buffer(unsigned char *image_pt, int width,
                                      int height, int channels);
void decoder_process(DecoderContext *dc_context, FFmpegDemuxer *demuxer,
                     std::string cam_name, PictureBuffer *display_buffer,
                     int size_of_buffer, SeekInfo *seek_info,
                     bool use_cpu_buffer,
                     const sync_plan::SyncCam *sync_cam = nullptr);
void image_loader(DecoderContext *dc_context,
                  const std::vector<std::string> &img_list_vector,
                  PictureBuffer *display_buffer, int size_of_buffer,
                  SeekInfo *seek_info, bool use_cpu_buffer,
                  std::string cam_name, std::string root_dir,
                  std::string file_ext);
#endif
