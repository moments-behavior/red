#pragma once

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"
#include "buffer_utils.h"

struct PictureBuffer{
	unsigned char* frame;
	int frame_number;
	bool available_to_write;
};

void decoder_process(const char* input_file_name, int gpu_id, PictureBuffer* display_buffer, bool* decoding_flag, int size_of_buffer, bool* stop_flag, SeekContext* seek_context, int* total_num_frame, int* estimated_num_frames);