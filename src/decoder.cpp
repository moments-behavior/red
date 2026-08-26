#include "decoder.h"
#include "global.h"
#include "sync_plan.h"
#if !defined(__APPLE__)
#include "AppDecUtils.h"
#endif
#include "../lib/ImGuiFileDialog/stb/stb_image.h"  // all platforms: PNG/non-JPEG fallback
#if defined(__APPLE__) || defined(_WIN32)
#include <turbojpeg.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstring>

void decoder_clear_buffer_with_constant_image(unsigned char *image_pt,
                                              int width, int height) {
    // One 32-bit store per pixel rather than four byte stores. This runs for
    // every slot of every camera's ring buffer on project load
    // (num_cams x size_of_buffer x w x h pixels), so on a 17-camera rig it is
    // several GB of writes and the byte-wise version dominated load time.
    // memcpy builds the pattern so the byte order stays correct on any endian.
    const unsigned char px[4] = {45, 85, 255, 255};
    std::uint32_t v;
    std::memcpy(&v, px, sizeof(v));
    std::fill_n(reinterpret_cast<std::uint32_t *>(image_pt),
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                v);
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
                     bool use_cpu_buffer,
                     const sync_plan::SyncCam *sync_cam) {
  try {
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

    // Canonical-timeline mode: sampled at thread start and at every seek
    // servicing (each toggle issues a seek, so modes never mix mid-stream).
    // nFrame stays in mp4-index space; next_slot is the next canonical slot
    // to publish. pTmpImage always holds the last converted frame, which
    // makes it the natural duplicate source for gap/trailing fills.
    bool sync_on = sync_cam && dc_context->sync_fix_active.load();
    int64_t canonical_len = dc_context->sync_canonical_len;
    int64_t next_slot = 0;
    bool have_content = false;      // pTmpImage holds a frame of this epoch
    bool first_store_done = false;

    double video_length = demuxer->GetDuration();
    double frame_rate = demuxer->GetFramerate();
    // In sync mode the loader owns total/estimated (both = canonical_len).
    if (!sync_on) {
        if (demuxer->GetNumFrames() == 0) {
            dc_context->estimated_num_frames = int(video_length * frame_rate);
        } else {
            dc_context->estimated_num_frames = demuxer->GetNumFrames() - 1;
        }
    }
    int size_in_bytes;
    bool skip_first_decode_after_seek = false;

    // Copy the converted RGBA frame in pTmpImage into a ring slot and publish
    // it under `label` (mp4 index in passthrough, canonical slot in sync
    // mode). available_to_write is published LAST — see the note below.
    auto store_slot = [&](int64_t label, bool is_dropped) {
        if (first_store_done) {
            while (!display_buffer[buffer_head].available_to_write &&
                   !(dc_context->stop_flag) && !(seek_info->use_seek)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (dc_context->stop_flag || seek_info->use_seek) return false;
        }
        if (use_cpu_buffer) {
            decoder_get_image_from_gpu(pTmpImage,
                                       display_buffer[buffer_head].frame,
                                       4 * nWidth, nHeight);
        } else {
            cudaMemcpy(display_buffer[buffer_head].frame,
                       (uint8_t *)pTmpImage, size_in_bytes,
                       cudaMemcpyDeviceToDevice);
        }
        display_buffer[buffer_head].frame_number = (int)label;
        display_buffer[buffer_head].dropped = is_dropped;
        if (!first_store_done) {
            dc_context->decoding_flag = true;
            first_store_done = true;
        } else {
            latest_decoded_frame[cam_name].store((int)label);
        }
        // Publish available_to_write LAST so a consumer that sees it false is
        // guaranteed the matching frame_number/dropped/frame data.
        display_buffer[buffer_head].available_to_write = false;
        buffer_head = (buffer_head + 1) % size_of_buffer;
        return true;
    };
    do {
        if (seek_info->use_seek) {
            // Re-sample the sync mode: every toggle issues a seek, so this is
            // the epoch boundary where the decoder may switch modes.
            sync_on = sync_cam && dc_context->sync_fix_active.load();
            canonical_len = dc_context->sync_canonical_len;

            // In sync mode seek_frame is a canonical slot; the demuxer works
            // in mp4 frame indices. pos() of a slot inside a gap is the first
            // frame after the gap — the natural freeze source for that slot.
            // Sync seeks are forced accurate: the non-accurate keyframe snap
            // is per-camera, which would put different canonical instants in
            // the same ring slot.
            int64_t target = (int64_t)seek_info->seek_frame;
            uint64_t mp4_target = seek_info->seek_frame;
            bool seek_accurate = seek_info->seek_accurate;
            if (sync_on) {
                mp4_target = (uint64_t)sync_cam->seek_pos(target);
                seek_accurate = true;
            }

            uint64_t key_frame_num = demuxer->FindClosestKeyFrameFNI(
                mp4_target, dc_context->seek_interval);
            SeekContext s = SeekContext(key_frame_num);

            seek_success_flag = demuxer->Seek(s, pVideo, nVideoBytes, pktinfo);

            // reset the display buffer after seeking
            for (int i = 0; i < size_of_buffer; i++) {
                display_buffer[i].available_to_write = true;
            }

            if (!seek_success_flag) {
                std::cerr << "[decoder " << cam_name
                          << "] Seek failed for frame "
                          << seek_info->seek_frame << "; skipping\n";
                seek_info->use_seek = false;
                seek_info->seek_done = true;
                continue;
            }

            nFrameReturned = dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);

            for (int i = 0; i < nFrameReturned; i++) {
                pFrame = dec.GetFrame();
            }

            auto temp_nFrameReturned = dec.Decode(pVideo, nVideoBytes);

            if (seek_accurate) {
                uint64_t curr_frame = key_frame_num - 1;
                while (curr_frame != mp4_target) {
                    demux_success =
                        demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!demux_success) {
                        std::cout << "Demux error..." << std::endl;
                        nFrameReturned =
                            dec.Decode(NULL, 0, CUVID_PKT_DISCONTINUITY);
                        if (!sync_on)
                            dc_context->total_num_frame = nFrame + nFrameReturned;
                    } else {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                    // EOF guard: the demuxer is exhausted AND the decoder has
                    // been fully flushed (no buffered frames left), so the
                    // requested frame lies past the true end of the stream —
                    // e.g. the user typed a frame number beyond the video, or
                    // beyond an over-estimated frame count. Land on the last
                    // real frame instead of looping forever: without this the
                    // thread spins (demux fails -> flush returns 0 -> repeat)
                    // and hangs the whole UI, because seek_all_cameras
                    // busy-waits on seek_done which would never be set.
                    if (!demux_success && nFrameReturned == 0) {
                        seek_info->seek_frame = curr_frame;
                        break;
                    }
                    while (nFrameReturned != 0) {
                        curr_frame++;
                        if (curr_frame == mp4_target) {
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
            if (sync_on) {
                // The normal loop stores the leftover frames: the first one
                // (mp4_target) converts first, then back-fills slots from
                // next_slot with it — which labels ring slot 0 as `target`
                // whether or not this camera has a real frame there.
                nFrame = (int)mp4_target;
                next_slot = target;
                have_content = false;
                latest_decoded_frame[cam_name].store((int)target);
            } else {
                nFrame = seek_info->seek_frame;
                latest_decoded_frame[cam_name].store(seek_info->seek_frame);
            }
            first_store_done = true;
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
                        if (!sync_on)
                            dc_context->total_num_frame = nFrame + nFrameReturned;
                    } else {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                } else {
                    skip_first_decode_after_seek = false;
                }

                if (!pTmpImage && nFrameReturned) {
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
                    if (!sync_on) {
                        Nv12ToColor32<RGBA32>(
                            pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                            4 * dec.GetWidth(), dec.GetWidth(),
                            dec.GetHeight(), iMatrix);
                        store_slot(nFrame, false);
                        nFrame = nFrame + 1;
                    } else {
                        int64_t c = sync_cam->slot_of_pos(nFrame);
                        if (c < next_slot) {
                            // Frame precedes the current epoch's window (a
                            // seek past this camera's end). Keep it as the
                            // duplicate source so trailing fills can freeze
                            // it; don't publish a slot for it.
                            if (!have_content) {
                                Nv12ToColor32<RGBA32>(
                                    pFrame, dec.GetWidth(),
                                    (uint8_t *)pTmpImage, 4 * dec.GetWidth(),
                                    dec.GetWidth(), dec.GetHeight(), iMatrix);
                                have_content = true;
                            }
                            nFrame = nFrame + 1;
                            continue;
                        }
                        if (!have_content) {
                            // Gap at the head: no previous frame exists, so
                            // the back-fill duplicates this first frame.
                            Nv12ToColor32<RGBA32>(
                                pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                                4 * dec.GetWidth(), dec.GetWidth(),
                                dec.GetHeight(), iMatrix);
                            have_content = true;
                            while (next_slot < c) {
                                if (!store_slot(next_slot, true)) break;
                                next_slot++;
                            }
                        } else {
                            // Interior gap: fills freeze the PREVIOUS frame
                            // (still in pTmpImage), then the new frame is
                            // converted and stored at its canonical slot.
                            while (next_slot < c) {
                                if (!store_slot(next_slot, true)) break;
                                next_slot++;
                            }
                            Nv12ToColor32<RGBA32>(
                                pFrame, dec.GetWidth(), (uint8_t *)pTmpImage,
                                4 * dec.GetWidth(), dec.GetWidth(),
                                dec.GetHeight(), iMatrix);
                        }
                        if (next_slot == c && store_slot(c, false))
                            next_slot = c + 1;
                        nFrame = nFrame + 1;
                    }
                    if (!demux_success) {
                        std::cout << "total_num_frame: "
                                  << dc_context->total_num_frame << std::endl;
                    }
                }

                // End of stream, sync mode: total_num_frame is owned by the
                // loader (= canonical_len). Trailing fill: a camera whose
                // span ends before canonical_len must keep publishing
                // duplicate slots, or the shared min-decoded playback cap
                // would freeze at its last real frame and stall every camera.
                if (sync_on && !demux_success && nFrameReturned == 0 &&
                    have_content) {
                    while (next_slot < canonical_len) {
                        if (!store_slot(next_slot, true)) break;
                        next_slot++;
                    }
                }
            } else {
                // Not decoding (e.g. batch predict idles decoders to free CPU).
                // Sleep instead of busy-spinning at 100% CPU. The seek check at the
                // top of the loop still runs every iteration, so a seek request is
                // serviced promptly even while idle. Mirrors the macOS path.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } while (!(dc_context->stop_flag));

  } catch (const std::exception &e) {
    std::cerr << "[decoder " << cam_name << "] Fatal error: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "[decoder " << cam_name << "] Unknown fatal error"
              << std::endl;
  }
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
                     bool /*use_cpu_buffer*/,
                     const sync_plan::SyncCam *sync_cam) {
  try {
    // Run on performance cores
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    // Canonical-timeline mode: sampled at thread start and at every seek
    // servicing (each toggle issues a seek, so modes never mix mid-stream).
    bool sync_on = sync_cam && dc_context->sync_fix_active.load();
    int64_t canonical_len = dc_context->sync_canonical_len;

    int w = (int)demuxer->GetWidth();
    int h = (int)demuxer->GetHeight();
    double video_length = demuxer->GetDuration();
    double frame_rate   = demuxer->GetFramerate();
    double timebase     = demuxer->GetTimebase();
    // In sync mode the loader owns total/estimated (both = canonical_len).
    if (!sync_on) {
        if (demuxer->GetNumFrames() == 0)
            dc_context->estimated_num_frames = (int)(video_length * frame_rate);
        else
            dc_context->estimated_num_frames = (int)demuxer->GetNumFrames() - 1;
    }

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
    int nFrame      = 0;   // next mp4 frame index expected from the decoder
    int buffer_head = 0;

    // Sync mode emission state: the ring must receive the canonical slot
    // sequence next_slot, next_slot+1, ... where slots this camera dropped
    // hold a duplicate of the nearest decoded frame (held_pb).
    int64_t next_slot = 0;
    CVPixelBufferRef held_pb = nullptr;   // decoder-owned retain of last content
    bool first_store_done = false;
    int64_t packets_in_flight = 0;  // async submits not yet popped (sync mode)
    int eof_stall = 0;              // iterations waited on in-flight frames at EOF

    // -----------------------------------------------------------------------
    // Lambda: publish one pixel buffer (ownership transfers) into the next
    // ring slot under the label `label` (mp4 index or canonical slot).
    // -----------------------------------------------------------------------
    auto store_slot = [&](CVPixelBufferRef pb, int64_t label, bool is_dropped) {
        // The first slot ever is stored without waiting (dc_context->
        // decoding_flag not yet set, so the consumer frees nothing yet).
        if (first_store_done) {
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
        display_buffer[buffer_head].frame_number        = (int)label;
        display_buffer[buffer_head].dropped             = is_dropped;
        display_buffer[buffer_head].available_to_write  = false;

        if (!first_store_done) {
            dc_context->decoding_flag = true;
            first_store_done = true;
        } else {
            latest_decoded_frame[cam_name].store((int)label);
        }

        buffer_head = (buffer_head + 1) % size_of_buffer;
        return true;
    };

    // -----------------------------------------------------------------------
    // Lambda: handle one decoded frame (mp4 index frame_num). Passthrough
    // stores it as-is; sync mode remaps it to its canonical slot, first
    // back-filling any slots this camera has no frame for with a duplicate of
    // the previous frame (or of this frame, for a gap at the head).
    // -----------------------------------------------------------------------
    auto store_frame = [&](CVPixelBufferRef pb, int frame_num) {
        if (!sync_on) {
            if (!store_slot(pb, frame_num, false)) return false;
            nFrame++;
            return true;
        }
        int64_t c = sync_cam->slot_of_pos(frame_num);
        if (c < next_slot) {  // stale frame from before the current epoch
            CFRelease(pb);
            nFrame++;
            return true;
        }
        CVPixelBufferRef dup_src = held_pb ? held_pb : pb;
        while (next_slot < c) {
            CFRetain(dup_src);
            if (!store_slot(dup_src, next_slot, true)) {
                CFRelease(pb);  // pb was never handed to a slot
                return false;
            }
            next_slot++;
        }
        if (held_pb) CFRelease(held_pb);
        held_pb = pb;
        CFRetain(held_pb);
        if (!store_slot(pb, c, false)) return false;
        next_slot = c + 1;
        nFrame++;
        return true;
    };

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------
    while (!dc_context->stop_flag) {
        // ---- SEEK ----
        if (seek_info->use_seek) {
            // Re-sample the sync mode: every toggle issues a seek, so this is
            // the epoch boundary where the decoder may switch modes.
            sync_on = sync_cam && dc_context->sync_fix_active.load();
            canonical_len = dc_context->sync_canonical_len;
            if (held_pb) {
                CFRelease(held_pb);
                held_pb = nullptr;
            }

            // In sync mode seek_frame is a canonical slot; the demuxer works
            // in mp4 frame indices. seek_pos() of a slot inside a gap is the
            // first frame after the gap — the natural freeze source there.
            int64_t target = (int64_t)seek_info->seek_frame;
            uint64_t mp4_target = seek_info->seek_frame;
            bool target_dropped = false;
            if (sync_on) {
                mp4_target = (uint64_t)sync_cam->seek_pos(target);
                target_dropped = !sync_cam->has(target);
            }

            // 1. Drain any in-flight async decodes and clear buffered frames
            vt_dec.flush();

            // 2. Seek demuxer to nearest keyframe at or before the target
            uint64_t key_frame_num = demuxer->FindClosestKeyFrameFNI(
                mp4_target, dc_context->seek_interval);
            SeekContext sc(key_frame_num);
            bool seek_ok = demuxer->Seek(sc, pVideo, nVideoBytes, pktinfo);

            // 3. Release all pixel buffers in display slots
            for (int i = 0; i < size_of_buffer; i++) {
                mac_release_pixbuf_slot(display_buffer[i]);
                display_buffer[i].available_to_write = true;
            }

            if (!seek_ok) {
                std::cerr << "[decoder " << cam_name
                          << "] Seek failed for frame "
                          << seek_info->seek_frame << "; skipping\n";
                seek_info->use_seek  = false;
                seek_info->seek_done = true;
                continue;
            }

            // 4. Recreate VT session (seek invalidates existing decode state)
            vt_dec.destroy();
            use_vt = vt_dec.init(extradata, extradata_size, codec_id);
            packets_in_flight = 0;
            eof_stall = 0;

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
                    if (curr_frame < mp4_target) {
                        CFRelease(pb);
                        curr_frame++;
                    } else {
                        display_buffer[0].pixel_buffer       = pb;
                        display_buffer[0].frame_number       = (int)target;
                        display_buffer[0].dropped            = target_dropped;
                        display_buffer[0].available_to_write = false;
                        dc_context->decoding_flag            = true;
                        if (sync_on) {
                            held_pb = pb;
                            CFRetain(held_pb);
                        }
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

                if (curr_frame < mp4_target) {
                    CFRelease(pb);
                    curr_frame++;
                } else {
                    display_buffer[0].pixel_buffer       = pb;
                    display_buffer[0].frame_number       = (int)target;
                    display_buffer[0].dropped            = target_dropped;
                    display_buffer[0].available_to_write = false;
                    dc_context->decoding_flag            = true;
                    if (sync_on) {
                        held_pb = pb;
                        CFRetain(held_pb);
                    }
                    curr_frame++;
                    seek_done = true;
                }
            }

            buffer_head = 1;
            first_store_done = true;
            if (sync_on) {
                // Ring continues at the slot after the target; the decoder
                // resumes at the mp4 frame after the one just stored.
                next_slot = target + 1;
                nFrame    = (int)(mp4_target + 1);
            } else {
                nFrame = (int)curr_frame;
            }
            latest_decoded_frame[cam_name].store((int)target);
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
            packets_in_flight++;
            eof_stall = 0;
        } else if (!sync_on) {
            // End of stream
            dc_context->total_num_frame = nFrame;
        } else if (packets_in_flight > 0 && eof_stall < 100) {
            // End of stream but async decodes are still in flight — filling
            // now would make the late real frames look stale and get dropped.
            eof_stall++;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
            // End of stream, sync mode: total_num_frame is owned by the
            // loader (= canonical_len). Trailing fill: a camera whose span
            // ends before canonical_len must keep publishing duplicate slots,
            // or the shared min-decoded playback cap would freeze at its last
            // real frame and stall every camera.
            while (next_slot < canonical_len && held_pb) {
                CFRetain(held_pb);
                if (!store_slot(held_pb, next_slot, true)) break;
                next_slot++;
            }
        }

        // Drain decoded frames from the reorder queue
        while (true) {
            CVPixelBufferRef pb = vt_dec.pop_next();
            if (!pb) break;
            if (packets_in_flight > 0) packets_in_flight--;
            if (!store_frame(pb, nFrame))
                break;
        }
    }

    // Cleanup: release any retained pixel buffers
    if (held_pb) {
        CFRelease(held_pb);
        held_pb = nullptr;
    }
    vt_dec.flush();
    for (int i = 0; i < size_of_buffer; i++)
        mac_release_pixbuf_slot(display_buffer[i]);

  } catch (const std::exception &e) {
    std::cerr << "[decoder " << cam_name << "] Fatal error: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "[decoder " << cam_name << "] Unknown fatal error"
              << std::endl;
  }
}

#endif // __APPLE__

// Helper: load image as RGBA into display buffer slot
static inline bool load_image_rgba(const std::string &file_name,
                                   unsigned char *dst_buf,
                                   size_t *out_size) {
#if defined(__APPLE__) || defined(_WIN32)
    // Try turbojpeg first for JPEG files (SIMD-accelerated, ~2-3x faster than stbi)
    {
        FILE *fp = fopen(file_name.c_str(), "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long fsize = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            // Check JPEG magic bytes (FFD8)
            unsigned char magic[2] = {};
            fread(magic, 1, 2, fp);
            if (fsize > 2 && magic[0] == 0xFF && magic[1] == 0xD8) {
                fseek(fp, 0, SEEK_SET);
                std::vector<unsigned char> jpeg_buf(fsize);
                fread(jpeg_buf.data(), 1, fsize, fp);
                fclose(fp);

                tjhandle tj = tjInitDecompress();
                if (tj) {
                    int w, h, tj_subsamp, tj_cs;
                    if (tjDecompressHeader3(tj, jpeg_buf.data(), fsize,
                                            &w, &h, &tj_subsamp, &tj_cs) == 0) {
                        if (tjDecompress2(tj, jpeg_buf.data(), fsize,
                                          dst_buf, w, 0, h, TJPF_RGBA,
                                          TJFLAG_FASTDCT) == 0) {
                            if (out_size) *out_size = (size_t)w * h * 4;
                            tjDestroy(tj);
                            return true;
                        }
                    }
                    tjDestroy(tj);
                }
            } else {
                fclose(fp);
            }
        }
    }
    // Fallback to stbi for non-JPEG formats (PNG, TIFF, etc.)
    int w, h, ch;
    unsigned char *data = stbi_load(file_name.c_str(), &w, &h, &ch, 4);
    if (!data) return false;
    size_t buf_size = (size_t)w * h * 4;
    memcpy(dst_buf, data, buf_size);
    stbi_image_free(data);
    if (out_size) *out_size = buf_size;
    return true;
#else
    // Linux: uses stbi (OpenCV removed)
    int w, h, ch;
    unsigned char *data = stbi_load(file_name.c_str(), &w, &h, &ch, 4);
    if (!data) return false;
    size_t buf_size = (size_t)w * h * 4;
    memcpy(dst_buf, data, buf_size);
    stbi_image_free(data);
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
