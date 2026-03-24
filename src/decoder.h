#ifndef RED_DECODER
#define RED_DECODER
#include "ColorSpace.h"
#include "FFmpegDemuxer.h"
#include "NvCodecUtils.h"
#if !defined(__APPLE__)
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

struct SeekInfo {
    bool use_seek;
    bool seek_done;
    uint64_t seek_frame;
    bool seek_accurate;
};

struct PictureBuffer {
    unsigned char *frame;
    std::atomic<int> frame_number;
    std::atomic<bool> available_to_write;
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
};

#ifndef __APPLE__
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
                     bool use_cpu_buffer);
void image_loader(DecoderContext *dc_context,
                  const std::vector<std::string> &img_list_vector,
                  PictureBuffer *display_buffer, int size_of_buffer,
                  SeekInfo *seek_info, bool use_cpu_buffer,
                  std::string cam_name, std::string root_dir,
                  std::string file_ext);
#endif
