#include "decoder.h"
#include "global.h"
#ifndef __APPLE__
#include "AppDecUtils.h"
#else
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
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
    if (demuxer->GetNumFrames() == 0) {
        dc_context->estimated_num_frames = int(video_length * frame_rate);
    } else {
        dc_context->estimated_num_frames = demuxer->GetNumFrames() - 1;
    }
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

// macOS decoder: Phase 3 — async VideoToolbox via VTAsyncDecoder.
// Phases 2+3: decoded frames are stored as CVPixelBufferRef (IOSurface-backed)
// in display_buffer[*].pixel_buffer; the main thread imports them into Metal
// textures via CVMetalTextureCache + compute shader (no CPU color conversion).

#include <pthread.h>
#include "vt_async_decoder.h"

// Helper: release any CVPixelBuffer retained in a buffer slot and reset it.
static inline void mac_release_pixbuf_slot(PictureBuffer &slot) {
    if (slot.pixel_buffer) {
        CFRelease(slot.pixel_buffer);
        slot.pixel_buffer = nullptr;
    }
}

void decoder_process(DecoderContext *dc_context, FFmpegDemuxer *demuxer,
                     std::string cam_name, PictureBuffer *display_buffer,
                     int size_of_buffer, SeekInfo *seek_info,
                     bool /*use_cpu_buffer*/) {
    // Run on performance cores
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    int w = (int)demuxer->GetWidth();
    int h = (int)demuxer->GetHeight();
    double video_length = demuxer->GetDuration();
    double frame_rate   = demuxer->GetFramerate();
    double timebase     = demuxer->GetTimebase();
    if (demuxer->GetNumFrames() == 0)
        dc_context->estimated_num_frames = (int)(video_length * frame_rate);
    else
        dc_context->estimated_num_frames = (int)demuxer->GetNumFrames() - 1;

    (void)w; (void)h;  // used by caller via scene->image_width/height

    AVCodecID codec_id = demuxer->GetVideoCodec();
    uint8_t *extradata      = demuxer->GetExtradata();
    int      extradata_size = demuxer->GetExtradataSize();

    VTAsyncDecoder vt_dec;
    bool use_vt = vt_dec.init(extradata, extradata_size, codec_id);
    if (!use_vt)
        std::cerr << "[mac] VTAsyncDecoder init failed; no video decode available\n";

    size_t   nVideoBytes = 0;
    uint8_t *pVideo      = nullptr;
    PacketData pktinfo;
    int nFrame      = 0;
    int buffer_head = 0;

    // -----------------------------------------------------------------------
    // Lambda: store a decoded CVPixelBuffer into the next free display slot
    // -----------------------------------------------------------------------
    auto store_frame = [&](CVPixelBufferRef pb, int frame_num) {
        // Slot 0 is stored without waiting (dc_context->decoding_flag not yet set)
        if (frame_num > 0) {
            while (!display_buffer[buffer_head].available_to_write &&
                   !dc_context->stop_flag && !seek_info->use_seek) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (dc_context->stop_flag || seek_info->use_seek) {
            CFRelease(pb);
            return false;
        }
        // Release any old pixel buffer still in this slot
        mac_release_pixbuf_slot(display_buffer[buffer_head]);

        display_buffer[buffer_head].pixel_buffer        = pb;  // transfer ownership
        display_buffer[buffer_head].frame_number        = frame_num;
        display_buffer[buffer_head].available_to_write  = false;

        if (frame_num == 0)
            dc_context->decoding_flag = true;
        else
            latest_decoded_frame[cam_name].store(frame_num);

        nFrame++;
        buffer_head = (buffer_head + 1) % size_of_buffer;
        return true;
    };

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------
    while (!dc_context->stop_flag) {
        // ---- SEEK ----
        if (seek_info->use_seek) {
            // 1. Drain any in-flight async decodes and clear buffered frames
            vt_dec.flush();

            // 2. Seek demuxer to nearest keyframe at or before the target
            uint64_t key_frame_num = demuxer->FindClosestKeyFrameFNI(
                seek_info->seek_frame, dc_context->seek_interval);
            SeekContext sc(key_frame_num);
            demuxer->Seek(sc, pVideo, nVideoBytes, pktinfo);

            // 3. Release all pixel buffers in display slots
            for (int i = 0; i < size_of_buffer; i++) {
                mac_release_pixbuf_slot(display_buffer[i]);
                display_buffer[i].available_to_write = true;
            }

            // 4. Recreate VT session (seek invalidates existing decode state)
            vt_dec.destroy();
            use_vt = vt_dec.init(extradata, extradata_size, codec_id);

            // 5. Forward-decode from keyframe to the target frame using
            //    blocking submissions (synchronous for seek accuracy).
            uint64_t curr_frame = key_frame_num;
            bool     seek_done  = false;

            // Prime the session with the keyframe packet
            if (pVideo && nVideoBytes > 0) {
                vt_dec.submit_blocking(pVideo, nVideoBytes, pktinfo.pts, pktinfo.dts,
                                       timebase, (pktinfo.flags & AV_PKT_FLAG_KEY) != 0);
                // After submit_blocking, callback has fired; drain_one() retrieves it
                CVPixelBufferRef pb = vt_dec.drain_one();
                if (pb) {
                    if (curr_frame < seek_info->seek_frame) {
                        CFRelease(pb);
                        curr_frame++;
                    } else {
                        display_buffer[0].pixel_buffer       = pb;
                        display_buffer[0].frame_number       = (int)seek_info->seek_frame;
                        display_buffer[0].available_to_write = false;
                        dc_context->decoding_flag            = true;
                        seek_done = true;
                    }
                }
            }

            while (!seek_done && !dc_context->stop_flag) {
                bool ok = demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                if (!ok) break;
                if (!pVideo || nVideoBytes == 0) continue;

                vt_dec.submit_blocking(pVideo, nVideoBytes, pktinfo.pts, pktinfo.dts,
                                       timebase, (pktinfo.flags & AV_PKT_FLAG_KEY) != 0);
                CVPixelBufferRef pb = vt_dec.drain_one();
                if (!pb) continue;

                if (curr_frame < seek_info->seek_frame) {
                    CFRelease(pb);
                    curr_frame++;
                } else {
                    display_buffer[0].pixel_buffer       = pb;
                    display_buffer[0].frame_number       = (int)seek_info->seek_frame;
                    display_buffer[0].available_to_write = false;
                    dc_context->decoding_flag            = true;
                    curr_frame++;
                    seek_done = true;
                }
            }

            buffer_head = 1;
            nFrame      = (int)curr_frame;
            latest_decoded_frame[cam_name].store((int)seek_info->seek_frame);
            seek_info->use_seek  = false;
            seek_info->seek_done = true;
            continue;
        }

        // ---- NORMAL PLAYBACK ----
        if (!window_need_decoding[cam_name].load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        bool demux_ok = demuxer->Demux(pVideo, nVideoBytes, pktinfo);
        if (demux_ok && pVideo && nVideoBytes > 0) {
            vt_dec.submit(pVideo, nVideoBytes, pktinfo.pts, pktinfo.dts,
                          timebase, (pktinfo.flags & AV_PKT_FLAG_KEY) != 0);
        } else {
            // End of stream
            dc_context->total_num_frame = nFrame;
        }

        // Drain decoded frames from the reorder queue
        while (true) {
            CVPixelBufferRef pb = vt_dec.pop_next();
            if (!pb) break;
            if (!store_frame(pb, nFrame))
                break;
        }
    }

    // Cleanup: release any retained pixel buffers
    vt_dec.flush();
    for (int i = 0; i < size_of_buffer; i++)
        mac_release_pixbuf_slot(display_buffer[i]);
}

#endif // __APPLE__

// Helper: load image as RGBA into display buffer slot
static inline bool load_image_rgba(const std::string &file_name,
                                   unsigned char *dst_buf,
                                   size_t *out_size) {
#ifdef __APPLE__
    int w, h, ch;
    unsigned char *data = stbi_load(file_name.c_str(), &w, &h, &ch, 4);
    if (!data) return false;
    size_t buf_size = (size_t)w * h * 4;
    memcpy(dst_buf, data, buf_size);
    stbi_image_free(data);
    if (out_size) *out_size = buf_size;
    return true;
#else
    cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
    if (image.empty()) return false;
    cv::Mat image_rgba;
    cv::cvtColor(image, image_rgba, cv::COLOR_BGR2RGBA);
    size_t buf_size = image_rgba.total() * image_rgba.elemSize();
    memcpy(dst_buf, image_rgba.data, buf_size);
    if (out_size) *out_size = buf_size;
    return true;
#endif
}

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
                    load_image_rgba(file_name,
                                    display_buffer[buffer_head].frame, nullptr);
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
                    load_image_rgba(file_name,
                                    display_buffer[buffer_head].frame, nullptr);
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
