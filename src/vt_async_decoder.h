#pragma once
#ifdef __APPLE__

#include <stdint.h>
#include <mutex>
#include <queue>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
}
#include <CoreVideo/CoreVideo.h>
#include <CoreMedia/CoreMedia.h>
#include <VideoToolbox/VideoToolbox.h>

// ---------------------------------------------------------------------------
// VTAsyncDecoder — wraps one VTDecompressionSessionRef per camera.
//
// Usage:
//   init(extradata, size, codec_id)   — create session from stream extradata
//   submit(avcc_data, size, pts, dts) — submit one compressed packet (async)
//   pop_next()                        — dequeue next decoded frame in PTS order
//                                       (returns nullptr if not yet ready)
//   flush()                           — drain any delayed frames, empty queue
//   destroy()                         — invalidate session, release resources
// ---------------------------------------------------------------------------

class VTAsyncDecoder {
public:
    VTAsyncDecoder() = default;
    ~VTAsyncDecoder() { destroy(); }

    // Initialize from codec parameters.
    // extradata: AVCodecContext->extradata (AVCC/HVCC format)
    // codec_id : AV_CODEC_ID_H264 or AV_CODEC_ID_HEVC
    // Returns true on success.
    bool init(const uint8_t *extradata, int extradata_size, AVCodecID codec_id);

    // Submit one Annex-B or AVCC packet for asynchronous decode.
    // pts / dts are in stream timebase ticks; timebase_sec converts to seconds.
    void submit(const uint8_t *data, size_t size,
                int64_t pts, int64_t dts, double timebase_sec,
                bool is_keyframe);

    // Return the next decoded frame in PTS order, or nullptr if not yet ready.
    // The caller takes ownership and must CFRelease() the returned buffer.
    CVPixelBufferRef pop_next();

    // Like submit() but blocks until the frame has been decoded and the
    // output callback has fired.  Used during seek for frame-accurate access.
    void submit_blocking(const uint8_t *data, size_t size,
                         int64_t pts, int64_t dts, double timebase_sec,
                         bool is_keyframe);

    // Pop one decoded frame regardless of reorder-queue depth.
    // Used during seek when we know the callback has already fired.
    CVPixelBufferRef drain_one();

    // Drain all pending frames from VT and empty the reorder queue.
    // Releases all retained CVPixelBuffers.
    void flush();

    // Invalidate the VT session and release all resources.
    void destroy();

    // True if init() has been called successfully.
    bool is_initialized() const { return session_ != nullptr; }

private:
    // Callback invoked by VideoToolbox on decode completion (any thread).
    static void output_callback(void *ctx, void *source_frame_ref_con,
                                OSStatus status, VTDecodeInfoFlags flags,
                                CVImageBufferRef image_buffer,
                                CMTime pts, CMTime duration);

    // Convert Annex-B start-code packets to AVCC length-prefix format.
    static std::vector<uint8_t> annexb_to_avcc(const uint8_t *data, size_t size);

    // Build CMVideoFormatDescriptionRef from H.264 AVCC extradata.
    static bool make_fmt_desc_h264(const uint8_t *extra, int extra_size,
                                   CMVideoFormatDescriptionRef *out);
    // Build CMVideoFormatDescriptionRef from HEVC HVCC extradata.
    static bool make_fmt_desc_hevc(const uint8_t *extra, int extra_size,
                                   CMVideoFormatDescriptionRef *out);

    struct FrameEntry {
        CVPixelBufferRef buf;
        CMTime           pts;
        bool operator>(const FrameEntry &o) const {
            // min-heap: lowest PTS at top
            return CMTimeCompare(pts, o.pts) > 0;
        }
    };

    using MinHeap = std::priority_queue<FrameEntry,
                                        std::vector<FrameEntry>,
                                        std::greater<FrameEntry>>;

    VTDecompressionSessionRef       session_  = nullptr;
    CMVideoFormatDescriptionRef     fmt_desc_ = nullptr;
    MinHeap                         queue_;
    std::mutex                      mutex_;
    bool                            flushing_ = false;

    // How many frames to buffer before emitting (handles B-frame reordering)
    static constexpr int REORDER_DEPTH = 4;
};

// Convenience: build a CMSampleBufferRef from AVCC data + timing.
CMSampleBufferRef vt_make_sample_buffer(const uint8_t *avcc_data, size_t size,
                                        CMTime pts, CMTime dts,
                                        CMVideoFormatDescriptionRef fmt_desc);

#endif // __APPLE__
