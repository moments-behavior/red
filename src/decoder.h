#ifndef RED_DECODER
#define RED_DECODER

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"

struct SeekInfo{
    bool use_seek;
    bool seek_done;
    uint64_t seek_frame;
};

struct PictureBuffer{
	unsigned char* frame;
	int frame_number;
	bool available_to_write;
};

void get_image_from_gpu(CUdeviceptr dpSrc, uint8_t* pDst, int nWidth, int nHeight);
void clear_buffer_with_constant_image(unsigned char* image_pt, int width, int height);
void print_one_display_buffer(unsigned char* image_pt, int width, int height, int channels);
void decoder_process(const char* input_file_name, int gpu_id, PictureBuffer* display_buffer, bool* decoding_flag, int size_of_buffer, bool* stop_flag, SeekInfo* seek_context, int* total_num_frame, int* estimated_num_frames);

#endif
