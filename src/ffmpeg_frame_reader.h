#pragma once
// ffmpeg_frame_reader.h — FFmpeg-based frame extraction for JARVIS export.
// Replaces cv::VideoCapture + cv::imwrite with FFmpeg C API + stb_image_write.
// Uses VideoToolbox hardware decoding on macOS for near-native performance.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ffmpeg_reader {

// FFmpeg's format/codec initialization (avformat_open_input,
// avformat_find_stream_info, avcodec_open2) touches shared internal tables
// that aren't thread-safe across simultaneous opens. On Linux this shows up
// as "double free or corruption (!prev)" when 16 parallel threads open
// video files at once during calibration. Serializing just the open() step
// is cheap and avoids the races — decoding itself stays fully parallel.
inline std::mutex &ffmpeg_open_mutex() {
    static std::mutex m;
    return m;
}

// ---------------------------------------------------------------------------
// FrameReader — open a video, seek to frames, decode to RGB24
// ---------------------------------------------------------------------------
class FrameReader {
  public:
    FrameReader() = default;
    ~FrameReader() { close(); }

    // Non-copyable
    FrameReader(const FrameReader &) = delete;
    FrameReader &operator=(const FrameReader &) = delete;

    bool open(const std::string &path, bool use_hw_accel = true) {
        close();

        // Serialize FFmpeg format/codec setup. See ffmpeg_open_mutex() above.
        std::lock_guard<std::mutex> lk(ffmpeg_open_mutex());

        fmt_ctx_ = avformat_alloc_context();
        if (!fmt_ctx_) return false;
        fmt_ctx_->max_analyze_duration = 5 * AV_TIME_BASE;

        if (avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, nullptr) < 0) {
            fmt_ctx_ = nullptr;
            return false;
        }
        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
            return false;
        }

        video_stream_ = -1;
        for (unsigned i = 0; i < fmt_ctx_->nb_streams; i++) {
            if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_ = (int)i;
                break;
            }
        }
        if (video_stream_ < 0) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
            return false;
        }

        AVCodecParameters *par = fmt_ctx_->streams[video_stream_]->codecpar;
        const AVCodec *codec = avcodec_find_decoder(par->codec_id);
        if (!codec) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
            return false;
        }

        codec_ctx_ = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codec_ctx_, par);

        if (use_hw_accel && init_hw_decoder(codec))
            hw_enabled_ = true;

        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            avcodec_free_context(&codec_ctx_);
            avformat_close_input(&fmt_ctx_);
            codec_ctx_ = nullptr;
            fmt_ctx_ = nullptr;
            return false;
        }

        width_ = codec_ctx_->width;
        height_ = codec_ctx_->height;

        // Compute FPS
        AVRational tb = fmt_ctx_->streams[video_stream_]->avg_frame_rate;
        if (tb.num > 0 && tb.den > 0)
            fps_ = (double)tb.num / (double)tb.den;
        else
            fps_ = 30.0;

        frame_ = av_frame_alloc();
        sw_frame_ = av_frame_alloc();
        rgb_frame_ = av_frame_alloc();
        pkt_ = av_packet_alloc();

        // Allocate the RGB buffer with:
        //   - align=1 so rows are tightly packed (linesize == 3*width). The
        //     caller reads rgb[i*3+c] assuming tight packing, so using a
        //     larger alignment (which rounds linesize up with padding) would
        //     misalign every row past the first — produced garbage calib
        //     with reprojection errors in the billions of pixels.
        //   - allocated via av_malloc so the pointer itself is 32-byte
        //     aligned (satisfies sws_scale's SIMD load alignment).
        //   - with 1 KiB of extra slack past the end for sws_scale's SIMD
        //     tail stores, which otherwise corrupted the neighbouring
        //     glibc malloc chunk and tripped "double free or corruption"
        //     on the next free().
        int rgb_size =
            av_image_get_buffer_size(AV_PIX_FMT_RGB24, width_, height_, 1);
        rgb_buffer_ = (uint8_t *)av_malloc(rgb_size + 1024);
        if (!rgb_buffer_) {
            close();
            return false;
        }
        av_image_fill_arrays(rgb_frame_->data, rgb_frame_->linesize,
                             rgb_buffer_, AV_PIX_FMT_RGB24, width_,
                             height_, 1);

        opened_ = true;
        return true;
    }

    void close() {
        if (sws_ctx_) {
            sws_freeContext(sws_ctx_);
            sws_ctx_ = nullptr;
        }
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (rgb_frame_) {
            av_frame_free(&rgb_frame_);
            rgb_frame_ = nullptr;
        }
        if (sw_frame_) {
            av_frame_free(&sw_frame_);
            sw_frame_ = nullptr;
        }
        if (frame_) {
            av_frame_free(&frame_);
            frame_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        if (fmt_ctx_) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
            hw_device_ctx_ = nullptr;
        }
        if (rgb_buffer_) {
            av_free(rgb_buffer_);
            rgb_buffer_ = nullptr;
        }
        opened_ = false;
        hw_enabled_ = false;
        current_frame_ = -1;
        sws_src_fmt_ = AV_PIX_FMT_NONE;
    }

    // Seek to a specific frame number and decode it.
    // Returns pointer to RGB24 data (width * height * 3), or nullptr on failure.
    // The pointer is valid until the next readFrame() or close().
    //
    // Optimized for sequential access: if the target frame is ahead of the
    // current decode position (and not too far away), we simply continue
    // decoding forward without seeking. This avoids the costly seek + flush
    // + re-decode-from-keyframe overhead that dominated JARVIS export time.
    const uint8_t *readFrame(int frame_num) {
        if (!opened_) return nullptr;

        AVStream *st = fmt_ctx_->streams[video_stream_];

        bool need_seek = (current_frame_ < 0) ||
                         (frame_num < current_frame_) ||
                         (frame_num - current_frame_ > 300);

        if (need_seek) {
            int64_t target_ts =
                (int64_t)(frame_num * av_q2d(av_inv_q(st->time_base)) / fps_);
            if (av_seek_frame(fmt_ctx_, video_stream_, target_ts,
                              AVSEEK_FLAG_BACKWARD) < 0)
                return nullptr;
            avcodec_flush_buffers(codec_ctx_);
            current_frame_ = -1;
        }

        while (true) {
            int ret = av_read_frame(fmt_ctx_, pkt_);
            if (ret < 0) return nullptr;
            if (pkt_->stream_index != video_stream_) {
                av_packet_unref(pkt_);
                continue;
            }

            ret = avcodec_send_packet(codec_ctx_, pkt_);
            av_packet_unref(pkt_);
            if (ret < 0) return nullptr;

            while (true) {
                ret = avcodec_receive_frame(codec_ctx_, frame_);
                if (ret == AVERROR(EAGAIN)) break;
                if (ret < 0) return nullptr;

                int64_t pts = frame_->pts;
                if (pts == AV_NOPTS_VALUE)
                    pts = frame_->best_effort_timestamp;
                int decoded_frame =
                    (int)(pts * av_q2d(st->time_base) * fps_ + 0.5);

                if (decoded_frame >= frame_num) {
                    if (!convertToRGB()) return nullptr;
                    current_frame_ = decoded_frame;
                    return rgb_buffer_;
                }
            }
        }
    }

    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    bool isOpened() const { return opened_; }

    int frameCount() const {
        if (!opened_ || video_stream_ < 0) return 0;
        int64_t nb = fmt_ctx_->streams[video_stream_]->nb_frames;
        if (nb > 0) return (int)nb;
        // Estimate from duration and fps
        double dur = fmt_ctx_->duration / (double)AV_TIME_BASE;
        return (dur > 0 && fps_ > 0) ? (int)(dur * fps_ + 0.5) : 0;
    }

  private:
    // Try to initialize VideoToolbox hardware decoding
    bool init_hw_decoder(const AVCodec *codec) {
#ifdef __APPLE__
        // Check if codec supports videotoolbox
        for (int i = 0;; i++) {
            const AVCodecHWConfig *cfg = avcodec_get_hw_config(codec, i);
            if (!cfg)
                break;
            if (cfg->device_type == AV_HWDEVICE_TYPE_VIDEOTOOLBOX &&
                cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
                if (av_hwdevice_ctx_create(&hw_device_ctx_,
                                           AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                                           nullptr, nullptr, 0) >= 0) {
                    codec_ctx_->hw_device_ctx =
                        av_buffer_ref(hw_device_ctx_);
                    return true;
                }
            }
        }
#endif
        (void)codec;
        return false;
    }

    // Convert decoded frame (HW or SW) to RGB24 in rgb_buffer_
    bool convertToRGB() {
        AVFrame *src = frame_;

        // If HW frame, transfer to CPU first
        if (hw_enabled_ && frame_->format ==
                               AV_PIX_FMT_VIDEOTOOLBOX) {
            av_frame_unref(sw_frame_);
            if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0)
                return false;
            src = sw_frame_;
        }

        // Create or recreate sws context if pixel format changed
        auto src_fmt = (AVPixelFormat)src->format;
        if (src_fmt != sws_src_fmt_) {
            if (sws_ctx_)
                sws_freeContext(sws_ctx_);
            sws_ctx_ =
                sws_getContext(width_, height_, src_fmt, width_, height_,
                               AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR,
                               nullptr, nullptr, nullptr);
            if (!sws_ctx_)
                return false;
            sws_src_fmt_ = src_fmt;
        }

        sws_scale(sws_ctx_, src->data, src->linesize, 0, height_,
                   rgb_frame_->data, rgb_frame_->linesize);
        return true;
    }

    AVFormatContext *fmt_ctx_ = nullptr;
    AVCodecContext *codec_ctx_ = nullptr;
    AVBufferRef *hw_device_ctx_ = nullptr;
    SwsContext *sws_ctx_ = nullptr;
    AVFrame *frame_ = nullptr;
    AVFrame *sw_frame_ = nullptr; // for HW→CPU transfer
    AVFrame *rgb_frame_ = nullptr;
    AVPacket *pkt_ = nullptr;
    uint8_t *rgb_buffer_ = nullptr;   // av_malloc'd, aligned+padded for sws_scale
    int video_stream_ = -1;
    int width_ = 0, height_ = 0;
    double fps_ = 0.0;
    bool opened_ = false;
    bool hw_enabled_ = false;
    int current_frame_ = -1;
    AVPixelFormat sws_src_fmt_ = AV_PIX_FMT_NONE;
};

} // namespace ffmpeg_reader
