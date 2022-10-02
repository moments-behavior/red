#include "decoder.h"

void decoder_process(const char *input_file_name, int gpu_id, PictureBuffer* display_buffer, bool* decoding_flag, int size_of_buffer, bool* stop_flag, SeekContext* seek_context, int* total_num_frame, int* estimated_num_frames)
{
    CheckInputFile(input_file_name);
    std::cout << input_file_name << std::endl;
    
    CUdeviceptr pTmpImage = 0;
    ck(cuInit(0));
    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, gpu_id, 0);

    std::map<std::string, std::string> m;
    size_t nVideoBytes = 0;
    PacketData pktinfo;


    FFmpegDemuxer demuxer(input_file_name, m);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    int nWidth = 0, nHeight = 0;

    int nFrameReturned = 0, nFrame = 0, iMatrix = 0;
    uint8_t *pVideo = nullptr;
    uint8_t *pFrame;
    
    int buffer_head=0;

    bool seek_success_flag;
    bool demux_success;


    double video_length = demuxer.GetDuration();
    double frame_rate = demuxer.GetFramerate();
    *estimated_num_frames = int(video_length * frame_rate);
    std::cout << "estimated_num_frames:" << *estimated_num_frames << std::endl;


    do{

        if (seek_context->use_seek) {

            //demuxer.Flush();
            std::cout << "target_frame_number:" << seek_context->seek_frame << std::endl;
            // assume every 10s is a keyframe, double check if your video is like that 
            seek_context->seek_frame = demuxer.FindClosestKeyFrame(seek_context->seek_frame, 10);

            seek_success_flag = demuxer.Seek(*seek_context, pVideo, nVideoBytes, pktinfo);
            std::cout << "seek_success_flag: "  << seek_success_flag << std::endl;

            // reset the display buffer after seeking  
            for (int i = 0; i < size_of_buffer; i++) {
                clear_buffer_with_constant_image(display_buffer[i].frame, 3208, 2200); 
                display_buffer[i].available_to_write = true;
            }
            //nFrameReturned = dec.Decode(pVideo, nVideoBytes, CUVID_PKT_DISCONTINUITY, pktinfo.pts);
            nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
            std::cout << "nFrameReturned right after seeking: " << nFrameReturned << std::endl;
           
            for (int i = 0; i < nFrameReturned; i++) {
                // decode frame and conversion
                pFrame = dec.GetFrame();
            }

            dec.Decode(pVideo, nVideoBytes);

            //dec.setReconfigParams(NULL, NULL);
            buffer_head = 0;
            nFrame = seek_context->seek_frame;

            seek_context->use_seek = false;
               
        }
        else {
            demux_success = demuxer.Demux(pVideo, nVideoBytes, pktinfo);

            if (!demux_success) 
            {
                // end of stream
                nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
                *total_num_frame = nFrame + nFrameReturned; 
            }
            else {
                nFrameReturned = dec.Decode(pVideo, nVideoBytes);
            }
            

            if (!nFrame && nFrameReturned)
            {
                LOG(INFO) << dec.GetVideoInfo();
                // Get output frame size from decoder
                nWidth = dec.GetWidth(); nHeight = dec.GetHeight();
                int size_in_bytes = nWidth * nHeight * 4;
                cuMemAlloc(&pTmpImage, size_in_bytes);
            }


            for (int i = 0; i < nFrameReturned; i++) {
                // decode frame and conversion
                pFrame = dec.GetFrame();
                iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
                Nv12ToColor32<RGBA32>(pFrame, dec.GetWidth(), (uint8_t*)pTmpImage, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight(), iMatrix);

                if (nFrame == 0) {
                    GetImage(pTmpImage, display_buffer[buffer_head].frame, 4 * dec.GetWidth(), dec.GetHeight());
                    display_buffer[buffer_head].available_to_write = false;
                    *decoding_flag = true;
                    display_buffer[buffer_head].frame_number = nFrame;
                }
                else {
                    while (!display_buffer[buffer_head].available_to_write && !(*stop_flag) && !(seek_context->use_seek)) {
                        // if the next frame hasn't been displayed, the queue is full, sleep  
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }

                    GetImage(pTmpImage, display_buffer[buffer_head].frame, 4 * dec.GetWidth(), dec.GetHeight());
                    display_buffer[buffer_head].available_to_write = false;
                    display_buffer[buffer_head].frame_number = nFrame;
                }
                nFrame = nFrame + 1;
                buffer_head = (buffer_head + 1) % size_of_buffer;

                // for debugging purpose 
                if (!demux_success){
                    std::cout << "total_num_frame: " << *total_num_frame << std::endl;
                }
            }
                      
        }
    } while (!(*stop_flag));
}