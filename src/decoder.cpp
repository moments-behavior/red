#include "decoder.h"
#include "global.h"
#ifndef __APPLE__
#include "AppDecUtils.h"
#endif

void decoder_clear_buffer_with_constant_image(unsigned char *image_pt,
                                              int width, int height) {
    int counter = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            *(image_pt + counter) = 45;
            *(image_pt + counter + 1) = 85;
            *(image_pt + counter + 2) = 255;
            *(image_pt + counter + 3) = 255;
            counter += 4;
        }
    }
}

void decoder_print_one_display_buffer(unsigned char *image_pt, int width,
                                      int height, int channels) {
    int counter = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                printf("%x ", *(image_pt + counter));
                counter++;
            }
            printf("  ");
        }
        printf("\n");
    }
}

inline void decoder_check_input_files(const char *sz_in_file_path) {
    std::ifstream fpIn(sz_in_file_path, std::ios::in | std::ios::binary);
    if (fpIn.fail()) {
        std::ostringstream err;
        err << "Unable to open input file: " << sz_in_file_path << std::endl;
        throw std::invalid_argument(err.str());
    }
}

#ifndef __APPLE__

void decoder_get_image_from_gpu(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth,
                                int nHeight) {
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

void decoder_process(DecoderContext *dc_context, FFmpegDemuxer *demuxer,
                     std::string cam_name, PictureBuffer *display_buffer,
                     int size_of_buffer, SeekInfo *seek_info,
                     bool use_cpu_buffer) {
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
        dc_context->estimated_num_frames = demuxer->GetNumFrames() - 1;
    }

    std::cout << "estimated_num_frames:" << dc_context->estimated_num_frames
              << std::endl;
    int size_in_bytes;
    bool skip_first_decode_after_seek = false;
    do {
        if (seek_info->use_seek) {
            uint64_t key_frame_num = demuxer->FindClosestKeyFrameFNI(
                seek_info->seek_frame, dc_context->seek_interval);
            SeekContext s = SeekContext(key_frame_num);

            seek_success_flag = demuxer->Seek(s, pVideo, nVideoBytes, pktinfo);

            // reset the display buffer after seeking
            for (int i = 0; i < size_of_buffer; i++) {
                display_buffer[i].available_to_write = true;
            }
            nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);

            for (int i = 0; i < nFrameReturned; i++) {
                pFrame = dec.GetFrame();
            }

            auto temp_nFrameReturned = dec.Decode(pVideo, nVideoBytes);

            if (seek_info->seek_accurate) {
                uint64_t curr_frame = key_frame_num - 1;
                while (curr_frame != seek_info->seek_frame) {
                    demux_success =
                        demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!demux_success) {
                        std::cout << "Demux error..." << std::endl;
                        nFrameReturned =
                            dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
                        dc_context->total_num_frame = nFrame + nFrameReturned;
                    } else {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                    while (nFrameReturned != 0) {
                        curr_frame++;
                        if (curr_frame == seek_info->seek_frame) {
                            skip_first_decode_after_seek = true;
                            goto jump;
                        } else {
                            dec.GetFrame();
                        }
                        nFrameReturned--;
                    }
                }
            jump:;
            } else {
                seek_info->seek_frame = key_frame_num;
            }

            buffer_head = 0;
            nFrame = seek_info->seek_frame;
            latest_decoded_frame[cam_name].store(seek_info->seek_frame);
            display_buffer[0].frame_number = -1;
            seek_info->use_seek = false;
            seek_info->seek_done = true;
        } else {
            if (window_need_decoding[cam_name].load()) {
                if (!skip_first_decode_after_seek) {
                    demux_success =
                        demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!demux_success) {
                        nFrameReturned =
                            dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
                        dc_context->total_num_frame = nFrame + nFrameReturned;
                    } else {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                } else {
                    skip_first_decode_after_seek = false;
                }

                if (!nFrame && nFrameReturned) {
                    LOG(INFO) << dec.GetVideoInfo();
                    nWidth = dec.GetWidth();
                    nHeight = dec.GetHeight();
                    size_in_bytes = nWidth * nHeight * 4;
                    cuMemAlloc(&pTmpImage, size_in_bytes);
                }

                for (int i = 0; i < nFrameReturned; i++) {
                    pFrame = dec.GetFrame();
                    iMatrix = dec.GetVideoFormatInfo()
                                  .video_signal_description.matrix_coefficients;
                    if (nFrame == 0) {
                        if (use_cpu_buffer) {
                            Nv12ToColor32<RGBA32>(
                                pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                                4 * dec.GetWidth(), dec.GetWidth(),
                                dec.GetHeight(), iMatrix);
                            decoder_get_image_from_gpu(
                                pTmpImage, display_buffer[buffer_head].frame,
                                4 * dec.GetWidth(), dec.GetHeight());
                        } else {
                            Nv12ToColor32<RGBA32>(
                                pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                                4 * dec.GetWidth(), dec.GetWidth(),
                                dec.GetHeight(), iMatrix);
                            cudaMemcpy(display_buffer[buffer_head].frame,
                                       (uint8_t *)pTmpImage, size_in_bytes,
                                       cudaMemcpyDeviceToDevice);
                        }
                        display_buffer[buffer_head].available_to_write = false;
                        dc_context->decoding_flag = true;
                        display_buffer[buffer_head].frame_number = nFrame;
                    } else {
                        while (
                            !display_buffer[buffer_head].available_to_write &&
                            !(dc_context->stop_flag) &&
                            !(seek_info->use_seek)) {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(1));
                        }
                        if (use_cpu_buffer) {
                            Nv12ToColor32<RGBA32>(
                                pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                                4 * dec.GetWidth(), dec.GetWidth(),
                                dec.GetHeight(), iMatrix);
                            decoder_get_image_from_gpu(
                                pTmpImage, display_buffer[buffer_head].frame,
                                4 * dec.GetWidth(), dec.GetHeight());
                        } else {
                            Nv12ToColor32<RGBA32>(
                                pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                                4 * dec.GetWidth(), dec.GetWidth(),
                                dec.GetHeight(), iMatrix);
                            cudaMemcpy(display_buffer[buffer_head].frame,
                                       (uint8_t *)pTmpImage, size_in_bytes,
                                       cudaMemcpyDeviceToDevice);
                        }

                        display_buffer[buffer_head].available_to_write = false;
                        display_buffer[buffer_head].frame_number = nFrame;
                        latest_decoded_frame[cam_name].store(nFrame);
                    }
                    nFrame = nFrame + 1;
                    buffer_head = (buffer_head + 1) % size_of_buffer;
                    if (!demux_success) {
                        std::cout << "total_num_frame: "
                                  << dc_context->total_num_frame << std::endl;
                    }
                }
            }
        }
    } while (!(dc_context->stop_flag));
}

#else // __APPLE__

// macOS decoder using FFmpeg + VideoToolbox hardware acceleration
// Falls back to software decode if VideoToolbox is unavailable.

#include <Accelerate/Accelerate.h>
#include <CoreVideo/CoreVideo.h>
#include <pthread.h>

// Convert a VideoToolbox CVPixelBuffer (NV12) directly to RGBA using vImage,
// writing into dst_rgba (w*h*4 bytes). info/info_init are cached per thread.
// Returns true on success; false if the pixel format is not NV12 (caller should
// fall back to sws_scale).
static bool mac_vimage_nv12_to_rgba(AVFrame *hw_frame, uint8_t *dst_rgba,
                                     int w, int h,
                                     vImage_YpCbCrToARGB &info,
                                     bool &info_init) {
    if (hw_frame->format != AV_PIX_FMT_VIDEOTOOLBOX)
        return false;

    CVPixelBufferRef pixbuf = (CVPixelBufferRef)hw_frame->data[3];
    OSType pix_fmt = CVPixelBufferGetPixelFormatType(pixbuf);

    bool full_range = (pix_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
    if (!full_range && pix_fmt != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
        return false;  // unsupported pixel format — caller falls back to sws

    if (!info_init) {
        vImage_YpCbCrPixelRange range = full_range
            ? (vImage_YpCbCrPixelRange){ 0, 128, 255, 255, 255, 0, 255, 0 }
            : (vImage_YpCbCrPixelRange){ 16, 128, 235, 240, 255, 1, 255, 0 };
        vImageConvert_YpCbCrToARGB_GenerateConversion(
            kvImage_YpCbCrToARGBMatrix_ITU_R_601_4,
            &range, &info,
            kvImage420Yp8_CbCr8, kvImageARGB8888,
            kvImageNoFlags);
        info_init = true;
    }

    CVPixelBufferLockBaseAddress(pixbuf, kCVPixelBufferLock_ReadOnly);

    uint8_t *y_ptr    = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pixbuf, 0);
    size_t   y_stride = CVPixelBufferGetBytesPerRowOfPlane(pixbuf, 0);
    uint8_t *uv_ptr   = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pixbuf, 1);
    size_t   uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pixbuf, 1);

    vImage_Buffer y_buf  = { y_ptr,  (vImagePixelCount)h,   (vImagePixelCount)w,   y_stride  };
    vImage_Buffer uv_buf = { uv_ptr, (vImagePixelCount)h/2, (vImagePixelCount)w/2, uv_stride };
    vImage_Buffer dst    = { dst_rgba, (vImagePixelCount)h, (vImagePixelCount)w, (size_t)w * 4 };

    // permuteMap: reorder vImage's ARGB output to RGBA
    // ARGB[0]=A, [1]=R, [2]=G, [3]=B  →  RGBA: dest[0]=R, [1]=G, [2]=B, [3]=A
    const uint8_t permuteMap[4] = { 1, 2, 3, 0 };
    vImageConvert_420Yp8_CbCr8ToARGB8888(
        &y_buf, &uv_buf, &dst, &info, permuteMap, 255, kvImageNoFlags);

    CVPixelBufferUnlockBaseAddress(pixbuf, kCVPixelBufferLock_ReadOnly);
    return true;
}

static enum AVPixelFormat mac_get_hw_format(AVCodecContext *,
                                            const enum AVPixelFormat *pix_fmts) {
    for (const enum AVPixelFormat *p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_VIDEOTOOLBOX)
            return *p;
    }
    // Return first available software format as fallback
    return pix_fmts[0];
}

void decoder_process(DecoderContext *dc_context, FFmpegDemuxer *demuxer,
                     std::string cam_name, PictureBuffer *display_buffer,
                     int size_of_buffer, SeekInfo *seek_info,
                     bool /*use_cpu_buffer*/) {
    // Run on performance cores to maximise decode/convert throughput
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    // --- Setup ---
    AVCodecID codec_id = demuxer->GetVideoCodec();
    const AVCodec *codec = avcodec_find_decoder(codec_id);
    if (!codec) {
        std::cerr << "[decoder_process mac] Cannot find decoder for codec "
                  << codec_id << std::endl;
        return;
    }

    AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "[decoder_process mac] Cannot allocate codec context\n";
        return;
    }

    // Try VideoToolbox hardware acceleration
    AVBufferRef *hw_device_ctx = nullptr;
    bool use_hw = (av_hwdevice_ctx_create(&hw_device_ctx,
                                          AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                                          nullptr, nullptr, 0) == 0);
    if (use_hw) {
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        codec_ctx->get_format = mac_get_hw_format;
        av_buffer_unref(&hw_device_ctx);
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "[decoder_process mac] Cannot open codec\n";
        avcodec_free_context(&codec_ctx);
        return;
    }

    AVFrame *hw_frame = av_frame_alloc();
    AVFrame *sw_frame = av_frame_alloc();
    SwsContext *sws_ctx = nullptr;
    AVPixelFormat last_src_fmt = AV_PIX_FMT_NONE;
    vImage_YpCbCrToARGB vimg_info = {};
    bool vimg_initialized = false;

    int w = (int)demuxer->GetWidth();
    int h = (int)demuxer->GetHeight();
    double video_length = demuxer->GetDuration();
    double frame_rate = demuxer->GetFramerate();
    std::cout << "[mac] Video framerate: " << frame_rate
              << "  length: " << video_length << std::endl;

    if (demuxer->GetNumFrames() == 0) {
        dc_context->estimated_num_frames = (int)(video_length * frame_rate);
    } else {
        dc_context->estimated_num_frames = (int)demuxer->GetNumFrames() - 1;
    }
    std::cout << "[mac] estimated_num_frames: "
              << dc_context->estimated_num_frames << std::endl;

    size_t nVideoBytes = 0;
    uint8_t *pVideo = nullptr;
    PacketData pktinfo;
    int nFrame = 0;
    int buffer_head = 0;

    // --- Main loop ---
    while (!(dc_context->stop_flag)) {
        if (seek_info->use_seek) {
            // Seek to nearest keyframe before the target
            uint64_t key_frame_num = demuxer->FindClosestKeyFrameFNI(
                seek_info->seek_frame, dc_context->seek_interval);
            SeekContext s(key_frame_num);
            demuxer->Seek(s, pVideo, nVideoBytes, pktinfo);
            avcodec_flush_buffers(codec_ctx);

            // Reset display buffer
            for (int i = 0; i < size_of_buffer; i++) {
                display_buffer[i].available_to_write = true;
            }

            // Send the keyframe packet first
            if (pVideo && nVideoBytes > 0) {
                AVPacket *pkt = av_packet_alloc();
                av_new_packet(pkt, (int)nVideoBytes);
                memcpy(pkt->data, pVideo, nVideoBytes);
                pkt->pts = pktinfo.pts;
                pkt->dts = pktinfo.dts;
                avcodec_send_packet(codec_ctx, pkt);
                av_packet_unref(pkt);
                av_packet_free(&pkt);
            }

            // Accurate forward seek: decode from keyframe to target frame,
            // discarding intermediate frames, so display_buffer[0] contains
            // exactly seek_frame and nFrame is set correctly.
            uint64_t curr_frame = key_frame_num;
            bool seek_complete = false;

            // Decode forward, discarding frames until we reach seek_frame
            while (!seek_complete && !(dc_context->stop_flag)) {
                int ret = avcodec_receive_frame(codec_ctx, hw_frame);
                if (ret == AVERROR(EAGAIN)) {
                    // Decoder needs more input — demux the next packet
                    bool ok = demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!ok) break;
                    AVPacket *pkt = av_packet_alloc();
                    av_new_packet(pkt, (int)nVideoBytes);
                    memcpy(pkt->data, pVideo, nVideoBytes);
                    pkt->pts = pktinfo.pts;
                    pkt->dts = pktinfo.dts;
                    avcodec_send_packet(codec_ctx, pkt);
                    av_packet_unref(pkt);
                    av_packet_free(&pkt);
                    continue;
                }
                if (ret < 0) break;

                if (curr_frame < seek_info->seek_frame) {
                    // Not at target yet — discard and advance
                    av_frame_unref(hw_frame);
                    curr_frame++;
                    continue;
                }

                // curr_frame == seek_frame: decode this frame into display_buffer[0]
                bool converted = mac_vimage_nv12_to_rgba(
                    hw_frame, display_buffer[0].frame, w, h,
                    vimg_info, vimg_initialized);
                if (!converted) {
                    // Fallback: transfer to CPU then sws_scale
                    AVFrame *frame_to_scale = hw_frame;
                    if (hw_frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
                        if (av_hwframe_transfer_data(sw_frame, hw_frame, 0) < 0) {
                            av_frame_unref(hw_frame);
                            break;
                        }
                        frame_to_scale = sw_frame;
                    }
                    AVPixelFormat src_fmt = (AVPixelFormat)frame_to_scale->format;
                    if (!sws_ctx || src_fmt != last_src_fmt) {
                        if (sws_ctx) sws_freeContext(sws_ctx);
                        last_src_fmt = src_fmt;
                        sws_ctx = sws_getContext(w, h, src_fmt, w, h, AV_PIX_FMT_RGBA,
                                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
                    }
                    if (!sws_ctx) { av_frame_unref(hw_frame); break; }
                    uint8_t *dst[4] = {display_buffer[0].frame, nullptr, nullptr, nullptr};
                    int dst_stride[4] = {w * 4, 0, 0, 0};
                    sws_scale(sws_ctx, (const uint8_t *const *)frame_to_scale->data,
                              frame_to_scale->linesize, 0, h, dst, dst_stride);
                    converted = true;
                }
                if (converted) {
                    display_buffer[0].frame_number = (int)seek_info->seek_frame;
                    display_buffer[0].available_to_write = false;
                    dc_context->decoding_flag = true;
                }
                av_frame_unref(hw_frame);
                seek_complete = true;
                curr_frame++;
            }

            buffer_head = seek_complete ? 1 : 0;
            nFrame = (int)curr_frame;
            latest_decoded_frame[cam_name].store((int)seek_info->seek_frame);
            seek_info->use_seek = false;
            seek_info->seek_done = true;
            continue;
        }

        if (!window_need_decoding[cam_name].load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Demux one packet
        bool demux_ok = demuxer->Demux(pVideo, nVideoBytes, pktinfo);

        AVPacket *pkt = av_packet_alloc();
        if (demux_ok && pVideo && nVideoBytes > 0) {
            av_new_packet(pkt, (int)nVideoBytes);
            memcpy(pkt->data, pVideo, nVideoBytes);
            pkt->pts = pktinfo.pts;
            pkt->dts = pktinfo.dts;
            avcodec_send_packet(codec_ctx, pkt);
        } else {
            // End of stream — flush decoder
            avcodec_send_packet(codec_ctx, nullptr);
        }
        av_packet_unref(pkt);
        av_packet_free(&pkt);

        // Receive all decoded frames from this packet
        while (true) {
            int ret = avcodec_receive_frame(codec_ctx, hw_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                break;

            // Wait for a free buffer slot (except for the very first frame)
            if (nFrame > 0) {
                while (!display_buffer[buffer_head].available_to_write &&
                       !(dc_context->stop_flag) && !(seek_info->use_seek)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            if (dc_context->stop_flag || seek_info->use_seek) {
                av_frame_unref(hw_frame);
                break;
            }

            // Convert to RGBA: try fast vImage path first, fall back to sws_scale
            bool converted = mac_vimage_nv12_to_rgba(
                hw_frame, display_buffer[buffer_head].frame, w, h,
                vimg_info, vimg_initialized);
            if (!converted) {
                // Fallback: transfer to CPU then sws_scale
                AVFrame *frame_to_scale = hw_frame;
                if (hw_frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
                    if (av_hwframe_transfer_data(sw_frame, hw_frame, 0) < 0) {
                        av_frame_unref(hw_frame);
                        continue;
                    }
                    frame_to_scale = sw_frame;
                }
                AVPixelFormat src_fmt = (AVPixelFormat)frame_to_scale->format;
                if (!sws_ctx || src_fmt != last_src_fmt) {
                    if (sws_ctx) sws_freeContext(sws_ctx);
                    last_src_fmt = src_fmt;
                    sws_ctx = sws_getContext(w, h, src_fmt, w, h, AV_PIX_FMT_RGBA,
                                             SWS_BILINEAR, nullptr, nullptr, nullptr);
                    if (!sws_ctx) { av_frame_unref(hw_frame); continue; }
                }
                uint8_t *dst[4] = {display_buffer[buffer_head].frame, nullptr,
                                   nullptr, nullptr};
                int dst_stride[4] = {w * 4, 0, 0, 0};
                sws_scale(sws_ctx, (const uint8_t *const *)frame_to_scale->data,
                          frame_to_scale->linesize, 0, h, dst, dst_stride);
            }

            display_buffer[buffer_head].frame_number = nFrame;
            display_buffer[buffer_head].available_to_write = false;

            if (nFrame == 0) {
                dc_context->decoding_flag = true;
            } else {
                latest_decoded_frame[cam_name].store(nFrame);
            }

            nFrame++;
            buffer_head = (buffer_head + 1) % size_of_buffer;

            av_frame_unref(hw_frame);
        }

        if (!demux_ok) {
            dc_context->total_num_frame = nFrame;
        }
    }

    // Cleanup
    if (sws_ctx)
        sws_freeContext(sws_ctx);
    av_frame_free(&hw_frame);
    av_frame_free(&sw_frame);
    avcodec_free_context(&codec_ctx);
}

#endif // __APPLE__

void image_loader(DecoderContext *dc_context,
                  const std::vector<std::string> &img_list_vector,
                  PictureBuffer *display_buffer, int size_of_buffer,
                  SeekInfo *seek_info, bool use_cpu_buffer,
                  std::string cam_name, std::string root_dir,
                  std::string file_ext) {
    int buffer_head = 0;
    int frame_number = 0;
    dc_context->total_num_frame = img_list_vector.size();
    dc_context->estimated_num_frames = img_list_vector.size();
    while (!(dc_context->stop_flag)) {
        if (seek_info->use_seek) {
            // reset the display buffer after seeking
            for (int i = 0; i < size_of_buffer; i++) {
                display_buffer[i].available_to_write = true;
            }
            buffer_head = 0;
            frame_number = seek_info->seek_frame;
            display_buffer[0].frame_number = -1;
            seek_info->use_seek = false;
            seek_info->seek_done = true;
            latest_decoded_frame[cam_name].store(seek_info->seek_frame);
        } else {
            if (frame_number < (int)img_list_vector.size()) {
                if (frame_number == 0) {
                    std::string file_name = root_dir + "/" + cam_name + "_" +
                                            img_list_vector[frame_number] +
                                            "." + file_ext;
                    cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
                    cv::Mat image_rgba;
                    cv::cvtColor(image, image_rgba, cv::COLOR_BGR2RGBA);
                    size_t buffer_size =
                        image_rgba.total() *
                        image_rgba.elemSize(); // Rows * Cols * Channels
                    memcpy(display_buffer[buffer_head].frame, image_rgba.data,
                           buffer_size);
                    display_buffer[buffer_head].available_to_write = false;
                    dc_context->decoding_flag = true;
                    display_buffer[buffer_head].frame_number = frame_number;
                } else {
                    while (!display_buffer[buffer_head].available_to_write &&
                           !(dc_context->stop_flag) && !(seek_info->use_seek)) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1));
                    }
                    std::string file_name = root_dir + "/" + cam_name + "_" +
                                            img_list_vector[frame_number] +
                                            "." + file_ext;
                    cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
                    cv::Mat image_rgba;
                    cv::cvtColor(image, image_rgba, cv::COLOR_BGR2RGBA);
                    size_t buffer_size =
                        image_rgba.total() *
                        image_rgba.elemSize(); // Rows * Cols * Channels
                    memcpy(display_buffer[buffer_head].frame, image_rgba.data,
                           buffer_size);
                    display_buffer[buffer_head].available_to_write = false;
                    display_buffer[buffer_head].frame_number = frame_number;
                    latest_decoded_frame[cam_name].store(frame_number);
                }
                frame_number = frame_number + 1;
                buffer_head = (buffer_head + 1) % size_of_buffer;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}
