#include "sw_decoder.h"

#include "FFmpegDemuxer.h"
#include "decode_backend.h"
#include "global.h"
#include "sync_plan.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <chrono>
#include <cstring>
#include <deque>
#include <iostream>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// SwVideoDecoder -- the libavcodec half of the NvDecoder interface used by
// decoder_process: submit a packet, learn how many frames became available,
// pop them one at a time in display order.
//
// Frames are recycled through a free list rather than reallocated: on a
// 16-camera rig this runs tens of thousands of times a second, and
// av_frame_alloc/free per frame showed up as pure allocator churn.
// ---------------------------------------------------------------------------
class SwVideoDecoder {
  public:
    ~SwVideoDecoder() {
        recycle_ready();
        if (last_returned_) {
            av_frame_free(&last_returned_);
        }
        for (AVFrame *f : pool_) {
            AVFrame *tmp = f;
            av_frame_free(&tmp);
        }
        pool_.clear();
        if (pkt_) av_packet_free(&pkt_);
        if (ctx_) avcodec_free_context(&ctx_);
    }

    bool init(AVCodecID codec_id, int threads, int hint_width,
              int hint_height) {
        const AVCodec *codec = avcodec_find_decoder(codec_id);
        if (!codec) {
            std::cerr << "[sw_decoder] no decoder for codec id " << codec_id
                      << "\n";
            return false;
        }
        ctx_ = avcodec_alloc_context3(codec);
        if (!ctx_) return false;

        // The demuxer hands us Annex-B for H.264/HEVC (it runs the
        // h264_mp4toannexb / hevc_mp4toannexb filter), which carries its own
        // in-band SPS/PPS -- so no extradata is set here on purpose. Other
        // codecs come through as raw container packets, which libavcodec
        // parses the same way. Dimensions are only a hint; the real ones come
        // off the first decoded frame.
        ctx_->width = hint_width;
        ctx_->height = hint_height;
        ctx_->thread_count = threads;
        ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

        if (avcodec_open2(ctx_, codec, nullptr) < 0) {
            std::cerr << "[sw_decoder] avcodec_open2 failed for "
                      << avcodec_get_name(codec_id) << "\n";
            avcodec_free_context(&ctx_);
            return false;
        }
        pkt_ = av_packet_alloc();
        if (!pkt_) return false;
        codec_name_ = avcodec_get_name(codec_id);
        return true;
    }

    // Submit one compressed packet. Returns the number of frames that became
    // available (0 is normal while the decoder fills its reorder window).
    int Decode(const uint8_t *data, size_t size) {
        if (!ctx_ || !data || size == 0) return 0;

        // A packet after Drain() means the stream resumed (a seek landed
        // without going through Reset, say). The decoder is in draining state
        // and would reject it, so flush back into a normal one first.
        if (eof_sent_) {
            avcodec_flush_buffers(ctx_);
            recycle_ready();
            eof_sent_ = false;
        }

        // pkt_ borrows the demuxer's buffer; avcodec_send_packet copies what
        // it needs. Clear it on every path out so no stale pointer outlives
        // the demuxer's next Demux() call.
        pkt_->data = const_cast<uint8_t *>(data);
        pkt_->size = (int)size;
        int ret = avcodec_send_packet(ctx_, pkt_);
        int n = 0;
        if (ret == AVERROR(EAGAIN)) {
            // Output queue full. Drain and retry once -- the send-then-drain
            // loop below should prevent this, but a codec that buffers
            // differently would otherwise silently drop a packet.
            n = receive_all();
            ret = avcodec_send_packet(ctx_, pkt_);
        }
        pkt_->data = nullptr;
        pkt_->size = 0;
        if (ret < 0) {
            log_send_error(ret);
            return n;
        }
        return n + receive_all();
    }

    // End of stream: flush the reorder window. Idempotent -- returns 0 once
    // the decoder is exhausted, which is what the caller's EOF loop expects.
    int Drain() {
        if (!ctx_) return 0;
        if (!eof_sent_) {
            avcodec_send_packet(ctx_, nullptr);
            eof_sent_ = true;
        }
        return receive_all();
    }

    // Post-seek discontinuity: drop everything buffered, in the decoder and in
    // our queue. Mandatory with frame threading -- without it the first frames
    // after a seek are stale ones from before it.
    void Reset() {
        if (!ctx_) return;
        avcodec_flush_buffers(ctx_);
        recycle_ready();
        eof_sent_ = false;
    }

    // Oldest ready frame, in display order. Valid until the next GetFrame()
    // or Reset(); the decoder keeps ownership.
    AVFrame *GetFrame() {
        if (last_returned_) {
            av_frame_unref(last_returned_);
            pool_.push_back(last_returned_);
            last_returned_ = nullptr;
        }
        if (ready_.empty()) return nullptr;
        last_returned_ = ready_.front();
        ready_.pop_front();
        return last_returned_;
    }

    const std::string &codec_name() const { return codec_name_; }
    int thread_count() const { return ctx_ ? ctx_->thread_count : 0; }

  private:
    void log_send_error(int ret) {
        char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, buf, sizeof(buf));
        std::cerr << "[sw_decoder] avcodec_send_packet: " << buf << "\n";
    }

    AVFrame *take_frame() {
        if (pool_.empty()) return av_frame_alloc();
        AVFrame *f = pool_.back();
        pool_.pop_back();
        return f;
    }

    void recycle_ready() {
        while (!ready_.empty()) {
            AVFrame *f = ready_.front();
            ready_.pop_front();
            av_frame_unref(f);
            pool_.push_back(f);
        }
    }

    int receive_all() {
        int n = 0;
        for (;;) {
            AVFrame *f = take_frame();
            if (!f) break;
            int ret = avcodec_receive_frame(ctx_, f);
            if (ret == 0) {
                ready_.push_back(f);
                n++;
                continue;
            }
            av_frame_unref(f);
            pool_.push_back(f);
            break; // EAGAIN or EOF
        }
        return n;
    }

    AVCodecContext *ctx_ = nullptr;
    AVPacket *pkt_ = nullptr;
    std::deque<AVFrame *> ready_;
    std::vector<AVFrame *> pool_;
    AVFrame *last_returned_ = nullptr;
    bool eof_sent_ = false;
    std::string codec_name_;
};

// Output byte order demanded by this platform's uploader (see decoder.h).
#if defined(RED_FRAME_BGRA)
constexpr AVPixelFormat kFrameFormat = AV_PIX_FMT_BGRA;
constexpr const char *kFrameFormatName = "BGRA";
#else
constexpr AVPixelFormat kFrameFormat = AV_PIX_FMT_RGBA;
constexpr const char *kFrameFormatName = "RGBA";
#endif

// ---------------------------------------------------------------------------
// RgbaConverter -- YUV frame to the packed 32-bit colour the ring buffers hold.
//
// The destination size is fixed to the demuxer's reported dimensions, which
// are what render_allocate_scene_memory sized the slots with. Scaling a frame
// that disagrees is cheaper than the buffer overrun the alternative would be.
// ---------------------------------------------------------------------------
class RgbaConverter {
  public:
    RgbaConverter(int dst_width, int dst_height)
        : dst_w_(dst_width), dst_h_(dst_height) {}

    ~RgbaConverter() {
        if (sws_) sws_freeContext(sws_);
    }

    bool convert(const AVFrame *src, uint8_t *dst) {
        if (!src || src->width <= 0 || src->height <= 0) return false;

        if (!sws_ || src->width != src_w_ || src->height != src_h_ ||
            src->format != src_fmt_) {
            if (sws_) sws_freeContext(sws_);
            sws_ = sws_getContext(src->width, src->height,
                                  (AVPixelFormat)src->format, dst_w_, dst_h_,
                                  kFrameFormat, SWS_BILINEAR, nullptr, nullptr,
                                  nullptr);
            if (!sws_) {
                std::cerr << "[sw_decoder] sws_getContext failed\n";
                return false;
            }
            src_w_ = src->width;
            src_h_ = src->height;
            src_fmt_ = src->format;
            apply_colorspace(src);
            if (src_w_ != dst_w_ || src_h_ != dst_h_) {
                std::cerr << "[sw_decoder] scaling " << src_w_ << "x" << src_h_
                          << " to " << dst_w_ << "x" << dst_h_
                          << " (container reported the latter)\n";
            }
        }

        uint8_t *dst_data[4] = {dst, nullptr, nullptr, nullptr};
        int dst_linesize[4] = {dst_w_ * 4, 0, 0, 0};
        sws_scale(sws_, src->data, src->linesize, 0, src->height, dst_data,
                  dst_linesize);
        return true;
    }

  private:
    // Match the NVDEC path, which picks its conversion matrix from the
    // stream's matrix_coefficients. Left at the swscale default a BT.709
    // stream comes out visibly off in the greens.
    void apply_colorspace(const AVFrame *src) {
        int table = SWS_CS_DEFAULT;
        switch (src->colorspace) {
        case AVCOL_SPC_BT709:
            table = SWS_CS_ITU709;
            break;
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M:
            table = SWS_CS_ITU601;
            break;
        case AVCOL_SPC_SMPTE240M:
            table = SWS_CS_SMPTE240M;
            break;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            table = SWS_CS_BT2020;
            break;
        default:
            break;
        }
        const int src_range = (src->color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
        const int *coeffs = sws_getCoefficients(table);
        sws_setColorspaceDetails(sws_, coeffs, src_range,
                                 sws_getCoefficients(SWS_CS_DEFAULT), 1, 0,
                                 1 << 16, 1 << 16);
    }

    SwsContext *sws_ = nullptr;
    int dst_w_ = 0, dst_h_ = 0;
    int src_w_ = -1, src_h_ = -1, src_fmt_ = -1;
};

} // namespace

void sw_decoder_process(DecoderContext *dc_context, FFmpegDemuxer *demuxer,
                        std::string cam_name, PictureBuffer *display_buffer,
                        int size_of_buffer, SeekInfo *seek_info,
                        bool /*use_cpu_buffer*/,
                        const sync_plan::SyncCam *sync_cam) {
  try {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    size_t nVideoBytes = 0;
    PacketData pktinfo;
    uint8_t *pVideo = nullptr;

    const int nWidth = (int)demuxer->GetWidth();
    const int nHeight = (int)demuxer->GetHeight();
    const int size_in_bytes = nWidth * nHeight * 4;

    SwVideoDecoder dec;
    if (!dec.init(demuxer->GetVideoCodec(), red::sw_decode_threads_per_camera(),
                  nWidth, nHeight)) {
        std::cerr << "[sw_decoder " << cam_name
                  << "] no software decoder available; camera will not play\n";
        return;
    }
    RgbaConverter converter(nWidth, nHeight);

    // Staging frame, mirroring pTmpImage in the NVDEC path: always holds the
    // most recently converted frame, which makes it the duplicate source for
    // sync-mode gap and trailing fills.
    std::vector<uint8_t> staging((size_t)size_in_bytes);

    int nFrameReturned = 0, nFrame = 0;
    int buffer_head = 0;
    bool seek_success_flag;
    bool demux_success = true;
    bool stream_exhausted = false;  // demuxer dry AND decoder fully drained

    // Canonical-timeline mode: sampled at thread start and at every seek
    // servicing (each toggle issues a seek, so modes never mix mid-stream).
    bool sync_on = sync_cam && dc_context->sync_fix_active.load();
    int64_t canonical_len = dc_context->sync_canonical_len;
    int64_t next_slot = 0;
    bool have_content = false;   // staging holds a frame of this epoch
    bool first_store_done = false;
    bool have_reported_info = false;

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
    bool skip_first_decode_after_seek = false;

    // Convert one decoded frame into the staging buffer.
    auto stage = [&](const AVFrame *frame) {
        if (!have_reported_info) {
            std::cout << "[sw_decoder " << cam_name << "] " << dec.codec_name()
                      << " " << frame->width << "x" << frame->height << " "
                      << av_get_pix_fmt_name((AVPixelFormat)frame->format)
                      << " -> " << kFrameFormatName << ", "
                      << dec.thread_count() << " threads" << std::endl;
            have_reported_info = true;
        }
        converter.convert(frame, staging.data());
    };

    // Copy the staged RGBA frame into a ring slot and publish it under
    // `label` (mp4 index in passthrough, canonical slot in sync mode).
    // available_to_write is published LAST -- see the note below.
    auto store_slot = [&](int64_t label, bool is_dropped) {
        if (first_store_done) {
            while (!display_buffer[buffer_head].available_to_write &&
                   !(dc_context->stop_flag) && !(seek_info->use_seek)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (dc_context->stop_flag || seek_info->use_seek) return false;
        }
        std::memcpy(display_buffer[buffer_head].frame, staging.data(),
                    size_in_bytes);
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
            // in mp4 frame indices. Sync seeks are forced accurate: the
            // non-accurate keyframe snap is per-camera, which would put
            // different canonical instants in the same ring slot.
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
                std::cerr << "[sw_decoder " << cam_name
                          << "] Seek failed for frame " << seek_info->seek_frame
                          << "; skipping\n";
                seek_info->use_seek = false;
                seek_info->seek_done = true;
                continue;
            }

            dec.Reset();
            // Seek() already returned the keyframe packet, so submit it here.
            // Unlike the NVDEC path this keeps its frame count: every frame
            // the decoder produces has to be counted exactly once or the
            // stepping below lands short of mp4_target, and a frame index off
            // by even one puts annotations on the wrong image. NVDEC gets away
            // with discarding this count because its parser returns nothing
            // for a lone first packet; libavcodec with thread_count == 1 can
            // return the keyframe immediately.
            nFrameReturned = dec.Decode(pVideo, nVideoBytes);

            if (seek_accurate) {
                // curr_frame is the index of the last frame accounted for;
                // starting one below the keyframe means "none yet, the next
                // frame produced is key_frame_num".
                uint64_t curr_frame = key_frame_num - 1;
                for (;;) {
                    while (nFrameReturned != 0) {
                        curr_frame++;
                        if (curr_frame == mp4_target) {
                            // Leave the target queued for the main loop.
                            skip_first_decode_after_seek = true;
                            goto jump;
                        }
                        dec.GetFrame();
                        nFrameReturned--;
                    }
                    demux_success =
                        demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!demux_success) {
                        nFrameReturned = dec.Drain();
                        if (!sync_on)
                            dc_context->total_num_frame = nFrame + nFrameReturned;
                    } else {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                    // EOF guard: the demuxer is exhausted AND the decoder has
                    // been fully drained, so the requested frame lies past the
                    // true end of the stream. Land on the last real frame
                    // instead of looping forever -- seek_all_cameras busy-waits
                    // on seek_done, so a spin here hangs the whole UI.
                    if (!demux_success && nFrameReturned == 0) {
                        seek_info->seek_frame = curr_frame;
                        break;
                    }
                }
            jump:;
            } else {
                seek_info->seek_frame = key_frame_num;
            }

            buffer_head = 0;
            if (sync_on) {
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
                stream_exhausted = false;
                if (!skip_first_decode_after_seek) {
                    demux_success =
                        demuxer->Demux(pVideo, nVideoBytes, pktinfo);
                    if (!demux_success) {
                        nFrameReturned = dec.Drain();
                        if (!sync_on)
                            dc_context->total_num_frame = nFrame + nFrameReturned;
                        stream_exhausted = (nFrameReturned == 0);
                    } else {
                        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
                    }
                } else {
                    skip_first_decode_after_seek = false;
                }

                for (int i = 0; i < nFrameReturned; i++) {
                    AVFrame *frame = dec.GetFrame();
                    if (!frame) break;
                    if (!sync_on) {
                        stage(frame);
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
                                stage(frame);
                                have_content = true;
                            }
                            nFrame = nFrame + 1;
                            continue;
                        }
                        if (!have_content) {
                            // Gap at the head: no previous frame exists, so
                            // the back-fill duplicates this first frame.
                            stage(frame);
                            have_content = true;
                            while (next_slot < c) {
                                if (!store_slot(next_slot, true)) break;
                                next_slot++;
                            }
                        } else {
                            // Interior gap: fills freeze the PREVIOUS frame
                            // (still in staging), then the new frame is
                            // converted and stored at its canonical slot.
                            while (next_slot < c) {
                                if (!store_slot(next_slot, true)) break;
                                next_slot++;
                            }
                            stage(frame);
                        }
                        if (next_slot == c && store_slot(c, false))
                            next_slot = c + 1;
                        nFrame = nFrame + 1;
                    }
                }

                // End of stream, sync mode: total_num_frame is owned by the
                // loader (= canonical_len). Trailing fill: a camera whose span
                // ends before canonical_len must keep publishing duplicate
                // slots, or the shared min-decoded playback cap would freeze
                // at its last real frame and stall every camera.
                if (sync_on && !demux_success && nFrameReturned == 0 &&
                    have_content) {
                    while (next_slot < canonical_len) {
                        if (!store_slot(next_slot, true)) break;
                        next_slot++;
                    }
                }

                // Fully drained and out of packets: nothing more will ever
                // come from this camera until a seek. Sleep rather than
                // spinning on a demuxer that keeps failing -- the seek check
                // at the top of the loop still runs every iteration.
                if (stream_exhausted) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else {
                // Not decoding (e.g. batch predict idles decoders to free CPU).
                // Sleep instead of busy-spinning at 100% CPU. The seek check at
                // the top of the loop still runs every iteration, so a seek
                // request is serviced promptly even while idle.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } while (!(dc_context->stop_flag));

  } catch (const std::exception &e) {
    std::cerr << "[sw_decoder " << cam_name << "] Fatal error: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "[sw_decoder " << cam_name << "] Unknown fatal error"
              << std::endl;
  }
}
