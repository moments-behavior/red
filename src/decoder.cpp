#include "decoder.h"

void decoder_process(const char *szInFilePath, int gpu_id, unsigned char* display_buffer, bool* decoding_flag)
{

    ck(cudaSetDevice(gpu_id));

    ck(cuInit(0));

    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, gpu_id));


    CUcontext cuContext = NULL;
    // createCudaContext(&cuContext, gpu_id, CU_CTX_SCHED_BLOCKING_SYNC);
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    CheckInputFile(szInFilePath);
    std::cout << szInFilePath << std::endl;
    
    FFmpegDemuxer demuxer(szInFilePath);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    
    int nWidth = (demuxer.GetWidth() + 1) & ~1; // make sure it is even
    int nPitch = nWidth * 4;
    int nVideoBytes = 0, nFrameReturned = 0, iMatrix = 0;
    uint8_t *pVideo = NULL;
    uint8_t *pFrame;
    int nFrame=0;
    
	int size_pic_nv12 = 10586400 *  sizeof(unsigned char);
	int size_pic_rgba = 3208 * 2200 * 4 *  sizeof(unsigned char);
    
    unsigned char *frame_nv12;
    cudaMalloc((void **)&frame_nv12, size_pic_nv12);


    bool demuxer_output=true;

    do{
        demuxer_output = demuxer.Demux(&pVideo, &nVideoBytes);
        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
        
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();


        for (int i = 0; i < nFrameReturned; i++) {
            pFrame = dec.GetFrame();
            iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
         
            // copy to temp buffer 
            cudaMemcpy(frame_nv12, pFrame, size_pic_nv12, cudaMemcpyDeviceToDevice);
            // covernt to rgba format for display
            Nv12ToColor32<RGBA32>(frame_nv12, dec.GetWidth(), display_buffer, nPitch, dec.GetWidth(), dec.GetHeight(), iMatrix);
        }	            
        nFrame += nFrameReturned;
        std::cout << nFrame << std::endl;
    } while (*decoding_flag);
}