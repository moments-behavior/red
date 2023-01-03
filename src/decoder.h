#ifndef RED_DECODER
#define RED_DECODER

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"

struct SeekInfo
{
    bool use_seek;
    bool seek_done;
    uint64_t seek_frame;
};

struct PictureBuffer
{
    unsigned char *frame;
    int frame_number;
    bool available_to_write;
};

struct DecoderContext
{
    bool decoding_flag;
    bool stop_flag;
    int total_num_frame;
    int estimated_num_frames;
    int gpu_index;
};

void decoder_get_image_from_gpu(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight);
void decoder_clear_buffer_with_constant_image(unsigned char *image_pt, int width, int height);
void decoder_print_one_display_buffer(unsigned char *image_pt, int width, int height, int channels);
void decoder_process(const char *input_file_name, DecoderContext *dc_context, PictureBuffer *display_buffer, int size_of_buffer, SeekInfo *seek_context);

#endif
