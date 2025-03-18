#include "decoder.h"

void decoder_get_image_from_gpu(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight)
{
    CUDA_MEMCPY2D m = {0};
    m.WidthInBytes = nWidth;
    m.Height = nHeight;
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)dpSrc;
    m.srcPitch = m.WidthInBytes;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
    m.dstPitch = m.WidthInBytes;
    cuMemcpy2D(&m);
}

void decoder_clear_buffer_with_constant_image(unsigned char *image_pt, int width, int height)
{
    int counter = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            *(image_pt + counter) = 45;
            *(image_pt + counter + 1) = 85;
            *(image_pt + counter + 2) = 255;
            *(image_pt + counter + 3) = 255;
            counter += 4;
        }
    }
}

void decoder_print_one_display_buffer(unsigned char *image_pt, int width, int height, int channels)
{
    int counter = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < channels; k++)
            {
                printf("%x ", *(image_pt + counter));
                counter++;
            }
            printf("  ");
        }
        printf("\n");
    }
}

inline void decoder_check_input_files(const char *sz_in_file_path)
{
    std::ifstream fpIn(sz_in_file_path, std::ios::in | std::ios::binary);
    if (fpIn.fail())
    {
        std::ostringstream err;
        err << "Unable to open input file: " << sz_in_file_path << std::endl;
        throw std::invalid_argument(err.str());
    }
}

void decoder_process(DecoderContext *dc_context, FFmpegDemuxer* demuxer, PictureBuffer *display_buffer, int size_of_buffer, SeekInfo *seek_info, bool use_cpu_buffer)
{
    CUdeviceptr pTmpImage = 0;
    ck(cuInit(0));
    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, dc_context->gpu_index, 0);
    size_t nVideoBytes = 0;
    PacketData pktinfo;

    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));
    int nWidth = 0, nHeight = 0;

    int nFrameReturned = 0, nFrame = 0, iMatrix = 0;
    uint8_t *pVideo = nullptr;
    uint8_t *pFrame;

    int buffer_head = 0;

    bool seek_success_flag;
    bool demux_success;

    double video_length = demuxer->GetDuration();
    double frame_rate = demuxer->GetFramerate();
    std::cout << "Video framerate: " << frame_rate << std::endl;
    std::cout << "Video length: " << video_length << std::endl;

    if (demuxer->GetNumFrames() == 0) {
        dc_context->estimated_num_frames = int(video_length * frame_rate);
    } else {
        dc_context->estimated_num_frames = demuxer->GetNumFrames()-1;
    }

    std::cout << "estimated_num_frames:" << dc_context->estimated_num_frames << std::endl;
    int size_in_bytes; 
    bool skip_first_decode_after_seek = false;
    do
    {

        // todo: need to make seek_context thread safe
        if (seek_info->use_seek)
        {
            // demuxer.Flush();
            std::cout << "target_frame_number:" << seek_info->seek_frame << std::endl;
            
            // assume every 10s is a keyframe, double check if your video is like that
            uint64_t key_frame_num = demuxer->FindClosestKeyFrameFNI(seek_info->seek_frame, dc_context->seek_interval);
            std::cout << "seeking to: " << key_frame_num << std::endl;
            SeekContext s = SeekContext(key_frame_num);

            seek_success_flag = demuxer->Seek(s, pVideo, nVideoBytes, pktinfo);
            // std::cout << "seek_success_flag: " << seek_success_flag << std::endl;

            // reset the display buffer after seeking
            for (int i = 0; i < size_of_buffer; i++)
            {
                // if (use_cpu_buffer) {
                //     decoder_clear_buffer_with_constant_image(display_buffer[i].frame, 3208, 2200);
                // }
                display_buffer[i].available_to_write = true;
            }
            // nFrameReturned = dec.Decode(pVideo, nVideoBytes, CUVID_PKT_DISCONTINUITY, pktinfo.pts);
            nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
            // std::cout << "nFrameReturned right after seeking: " << nFrameReturned << std::endl;

            for (int i = 0; i < nFrameReturned; i++)
            {
                // decode frame and conversion
                pFrame = dec.GetFrame();
            }

            auto temp_nFrameReturned = dec.Decode(pVideo, nVideoBytes);
            // std::cout << "not sure about this: " << temp_nFrameReturned << std::endl; 

            if (seek_info->seek_accurate) {
                // seek acurate implementation
                uint64_t curr_frame = key_frame_num - 1;
                // keep decoding till the target frame
                while (curr_frame != seek_info->seek_frame) {
                    demux_success = demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!demux_success)
                    {
                        // end of stream
                        std::cout << "Demux error..." << std::endl;
                        nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
                        dc_context->total_num_frame = nFrame + nFrameReturned;
                    }
                    else
                    {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                    while (nFrameReturned != 0)
                    {
                        curr_frame++;
                        if (curr_frame == seek_info->seek_frame) {
                            // reach the decoded frame 
                            skip_first_decode_after_seek = true;
                            goto jump;
                        } else {
                            dec.GetFrame();
                        }
                        nFrameReturned--;
                    }
                }
                jump:; // break out of loop 
            } else {
                seek_info->seek_frame = key_frame_num;
            }
            
            // dec.setReconfigParams(NULL, NULL);
            buffer_head = 0;
            nFrame = seek_info->seek_frame;
            display_buffer[0].frame_number = -1;
            seek_info->use_seek = false;
            seek_info->seek_done = true;
            // std::cout << "seek thread done " << temp_nFrameReturned << std::endl; 
        }
        else
        {
            if (!skip_first_decode_after_seek) {            
                demux_success = demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                if (!demux_success)
                {
                    // end of stream
                    std::cout << "Demux error..." << std::endl;
                    nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
                    dc_context->total_num_frame = nFrame + nFrameReturned;
                }
                else
                {
                    nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                }
            } else {
                skip_first_decode_after_seek = false;
            }
            
            if (!nFrame && nFrameReturned)
            {
                LOG(INFO) << dec.GetVideoInfo();
                // Get output frame size from decoder
                nWidth = dec.GetWidth();
                nHeight = dec.GetHeight();
                size_in_bytes = nWidth * nHeight * 4;
                cuMemAlloc(&pTmpImage, size_in_bytes);
            }

            for (int i = 0; i < nFrameReturned; i++)
            {
                // decode frame and conversion
                pFrame = dec.GetFrame();
                iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
                Nv12ToColor32<RGBA32>(pFrame, dec.GetWidth(), (uint8_t *)pTmpImage, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight(), iMatrix);

                if (nFrame == 0)
                {
                    if (use_cpu_buffer) {
                        decoder_get_image_from_gpu(pTmpImage, display_buffer[buffer_head].frame, 4 * dec.GetWidth(), dec.GetHeight());
                    } else {
                        cudaMemcpy(display_buffer[buffer_head].frame, (uint8_t *)pTmpImage, size_in_bytes, cudaMemcpyDeviceToDevice);
                    }
                    display_buffer[buffer_head].available_to_write = false;
                    dc_context->decoding_flag = true;
                    display_buffer[buffer_head].frame_number = nFrame;
                }
                else
                {
                    while (!display_buffer[buffer_head].available_to_write && !(dc_context->stop_flag) && !(seek_info->use_seek))
                    {
                        // if the next frame hasn't been displayed, the queue is full, sleep
                        // std::cout << "thread wait, " << display_buffer[buffer_head].available_to_write << ", " << buffer_head << ", " << display_buffer[buffer_head].frame_number << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    if (use_cpu_buffer) {
                        decoder_get_image_from_gpu(pTmpImage, display_buffer[buffer_head].frame, 4 * dec.GetWidth(), dec.GetHeight());
                    } else {
                        cudaMemcpy(display_buffer[buffer_head].frame, (uint8_t *)pTmpImage, size_in_bytes, cudaMemcpyDeviceToDevice);
                    }
                    display_buffer[buffer_head].available_to_write = false;
                    display_buffer[buffer_head].frame_number = nFrame;
                }
                nFrame = nFrame + 1;
                buffer_head = (buffer_head + 1) % size_of_buffer;
                // for debugging purpose
                if (!demux_success)
                {
                    std::cout << "total_num_frame: " << dc_context->total_num_frame << std::endl;
                }
            }
        }
    } while (!(dc_context->stop_flag));
}

void image_loader(DecoderContext *dc_context, std::vector<std::string> img_list, PictureBuffer *display_buffer, int size_of_buffer, SeekInfo *seek_info, bool use_cpu_buffer)
{
    int buffer_head = 0;
    int frame_number = 0;
    while(!(dc_context->stop_flag))
    {
        if (frame_number == 0)
        {
            cv::Mat image = cv::imread(img_list[frame_number], cv::IMREAD_COLOR);  
            cv::Mat image_rgba;
            cv::cvtColor(image, image_rgba, cv::COLOR_BGR2RGBA);
            size_t buffer_size = image.total() * image.elemSize(); // Rows * Cols * Channels
            memcpy(display_buffer[buffer_head].frame, image_rgba.data, buffer_size);

            display_buffer[buffer_head].available_to_write = false;
            dc_context->decoding_flag = true;
            display_buffer[buffer_head].frame_number = frame_number;
        }
        while (!display_buffer[buffer_head].available_to_write && !(dc_context->stop_flag) && !(seek_info->use_seek))
        {
            cv::Mat image = cv::imread(img_list[frame_number], cv::IMREAD_COLOR);  
            cv::Mat image_rgba;
            cv::cvtColor(image, image_rgba, cv::COLOR_BGR2RGBA);
            size_t buffer_size = image.total() * image.elemSize(); // Rows * Cols * Channels
            memcpy(display_buffer[buffer_head].frame, image_rgba.data, buffer_size);

            display_buffer[buffer_head].available_to_write = false;
            display_buffer[buffer_head].frame_number = frame_number;
        }
        frame_number = frame_number + 1;
        buffer_head = (buffer_head + 1) % size_of_buffer;
    }
}