#pragma once

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"

void decoder_process(const char *szInFilePath, int gpu_id, unsigned char* display_buffer, bool* decoding_flag);
