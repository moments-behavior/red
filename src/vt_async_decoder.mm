#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#include "vt_async_decoder.h"

#include <arpa/inet.h>  // htonl
#include <cstring>
#include <cassert>

// ---------------------------------------------------------------------------
// Helper: build a CMSampleBufferRef from raw AVCC data + timing
// ---------------------------------------------------------------------------

CMSampleBufferRef vt_make_sample_buffer(const uint8_t *avcc_data, size_t size,
                                        CMTime pts, CMTime dts,
                                        CMVideoFormatDescriptionRef fmt_desc) {
    // Copy data into a malloced block so VT owns the memory.
    void *data_copy = malloc(size);
    if (!data_copy) return NULL;
    memcpy(data_copy, avcc_data, size);

    CMBlockBufferRef block = NULL;
    OSStatus err = CMBlockBufferCreateWithMemoryBlock(
        kCFAllocatorDefault,
        data_copy, size,
        kCFAllocatorMalloc,  // releases via free() when block is deallocated
        NULL,
        0, size, 0,
        &block);
    if (err != noErr || !block) { free(data_copy); return NULL; }

    CMSampleTimingInfo timing = {
        .presentationTimeStamp = pts,
        .decodeTimeStamp       = dts,
        .duration              = kCMTimeInvalid,
    };

    size_t sample_size = size;
    CMSampleBufferRef sample = NULL;
    err = CMSampleBufferCreate(
        kCFAllocatorDefault,
        block,
        TRUE,          // dataReady
        NULL, NULL,    // makeDataReadyCallback
        fmt_desc,
        1,             // numSamples
        1, &timing,    // numSampleTimingEntries
        1, &sample_size,
        &sample);
    CFRelease(block);
    if (err != noErr) return NULL;
    return sample;
}

// ---------------------------------------------------------------------------
// Annex-B → AVCC conversion
// ---------------------------------------------------------------------------

std::vector<uint8_t> VTAsyncDecoder::annexb_to_avcc(const uint8_t *data, size_t size) {
    // Collect the start position of each NAL unit's DATA (after start code)
    struct NalSpan { size_t data_begin; size_t data_end; };
    std::vector<NalSpan> spans;

    size_t i = 0;
    while (i + 2 < size) {
        bool sc4 = (i + 3 < size &&
                    data[i]==0 && data[i+1]==0 && data[i+2]==0 && data[i+3]==1);
        bool sc3 = (!sc4 && data[i]==0 && data[i+1]==0 && data[i+2]==1);

        if (sc4 || sc3) {
            size_t data_begin = i + (sc4 ? 4 : 3);
            // Find end of this NAL (start of next start code or end of buf)
            size_t j = data_begin;
            bool found_next = false;
            while (j + 2 < size) {
                bool n4 = (j + 3 < size &&
                           data[j]==0 && data[j+1]==0 && data[j+2]==0 && data[j+3]==1);
                bool n3 = (!n4 && data[j]==0 && data[j+1]==0 && data[j+2]==1);
                if (n4 || n3) { found_next = true; break; }
                j++;
            }
            // If the loop ended by running out of buffer (not by finding a start code),
            // extend j to the true end so we don't truncate the last NAL.
            if (!found_next)
                j = size;
            if (j > data_begin)
                spans.push_back({data_begin, j});
            i = j;
        } else {
            i++;
        }
    }

    // Build AVCC output
    std::vector<uint8_t> out;
    out.reserve(size);
    for (auto &s : spans) {
        size_t nal_sz = s.data_end - s.data_begin;
        uint32_t be = htonl((uint32_t)nal_sz);
        const uint8_t *p = (const uint8_t *)&be;
        out.insert(out.end(), p, p + 4);
        out.insert(out.end(), data + s.data_begin, data + s.data_end);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Format description creation
// ---------------------------------------------------------------------------

bool VTAsyncDecoder::make_fmt_desc_h264(const uint8_t *extra, int extra_size,
                                        CMVideoFormatDescriptionRef *out) {
    // AVCC extradata layout:
    //   [0]     = configurationVersion (1)
    //   [1]     = profile
    //   [2]     = profileCompatibility
    //   [3]     = level
    //   [4]     = 0xff (lengthSizeMinusOne = 3 → 4-byte lengths)
    //   [5]     = 0xe1 | numSPS
    //   [6..7]  = spsLength (big-endian)
    //   [8..8+spsLength-1] = SPS
    //   [8+spsLength] = numPPS
    //   [9+spsLength..10+spsLength] = ppsLength
    //   ...     = PPS
    if (!extra || extra_size < 7) return false;

    size_t pos = 5;
    int num_sps = extra[pos++] & 0x1f;
    if (num_sps < 1 || pos + 2 > (size_t)extra_size) return false;

    size_t sps_len = ((size_t)extra[pos] << 8) | extra[pos + 1];
    pos += 2;
    if (pos + sps_len > (size_t)extra_size) return false;
    const uint8_t *sps = extra + pos;
    pos += sps_len;

    if (pos >= (size_t)extra_size) return false;
    int num_pps = extra[pos++];
    if (num_pps < 1 || pos + 2 > (size_t)extra_size) return false;

    size_t pps_len = ((size_t)extra[pos] << 8) | extra[pos + 1];
    pos += 2;
    if (pos + pps_len > (size_t)extra_size) return false;
    const uint8_t *pps = extra + pos;

    const uint8_t *param_sets[2] = { sps, pps };
    size_t param_sizes[2]        = { sps_len, pps_len };

    OSStatus err = CMVideoFormatDescriptionCreateFromH264ParameterSets(
        kCFAllocatorDefault, 2, param_sets, param_sizes, 4, out);
    return (err == noErr);
}

bool VTAsyncDecoder::make_fmt_desc_hevc(const uint8_t *extra, int extra_size,
                                        CMVideoFormatDescriptionRef *out) {
    // HEVC uses HEVCDecoderConfigurationRecord.
    // We pass the whole extradata as a CFData and let CM parse it.
    // This API is available macOS 10.13+.
    CFDataRef data = CFDataCreate(kCFAllocatorDefault, extra, extra_size);
    if (!data) return false;

    // Parse VPS/SPS/PPS from HVCC extradata (offset 22 contains array_completeness)
    // The standard approach: use CMVideoFormatDescriptionCreateFromHEVCParameterSets
    // which takes individual parameter set NALs.
    // Parsing HVCC is complex; for now use a simplified approach:
    // attempt direct creation from the raw extradata.
    CFRelease(data);

    // HVCC layout (simplified, assumes standard encoder output):
    //  [0]      = configurationVersion
    //  ...      = profile/tier/level info (21 bytes)
    //  [22]     = numArrays
    //  for each array:
    //    [0]    = NAL_type (0x21=VPS, 0x22=SPS, 0x23=PPS)
    //    [1..2] = numNalus
    //    for each nalu: [0..1]=length, [2..] = data

    if (!extra || extra_size < 23) return false;
    int num_arrays = extra[22];
    size_t pos = 23;

    std::vector<const uint8_t *> param_sets;
    std::vector<size_t>          param_sizes;

    for (int a = 0; a < num_arrays && pos + 3 <= (size_t)extra_size; a++) {
        pos++;  // NAL type (we don't need to filter; VPS/SPS/PPS all go in)
        int num_nalus = ((int)extra[pos] << 8) | extra[pos + 1];
        pos += 2;
        for (int n = 0; n < num_nalus && pos + 2 <= (size_t)extra_size; n++) {
            size_t nalu_len = ((size_t)extra[pos] << 8) | extra[pos + 1];
            pos += 2;
            if (pos + nalu_len > (size_t)extra_size) break;
            param_sets.push_back(extra + pos);
            param_sizes.push_back(nalu_len);
            pos += nalu_len;
        }
    }

    if (param_sets.empty()) return false;

    OSStatus err = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        kCFAllocatorDefault,
        param_sets.size(),
        param_sets.data(),
        param_sizes.data(),
        4,    // nalUnitHeaderLength
        NULL, // extensions
        out);
    return (err == noErr);
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::init
// ---------------------------------------------------------------------------

bool VTAsyncDecoder::init(const uint8_t *extradata, int extradata_size,
                          AVCodecID codec_id) {
    destroy();

    bool ok = false;
    if (codec_id == AV_CODEC_ID_H264) {
        ok = make_fmt_desc_h264(extradata, extradata_size, &fmt_desc_);
    } else if (codec_id == AV_CODEC_ID_HEVC) {
        ok = make_fmt_desc_hevc(extradata, extradata_size, &fmt_desc_);
    }
    if (!ok || !fmt_desc_) {
        fprintf(stderr, "[VTAsync] Failed to create format description\n");
        return false;
    }

    // Request IOSurface-backed BGRA output. VideoToolbox reads the color space
    // metadata from the stream (BT.601/BT.709, full/video range) and applies
    // the correct matrix internally, producing display-ready pixels for any
    // video source. IOSurface backing allows zero-copy import as a Metal
    // texture via CVMetalTextureCacheCreateTextureFromImage.
    NSDictionary *out_props = @{
        (id)kCVPixelBufferPixelFormatTypeKey:
            @(kCVPixelFormatType_32BGRA),
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
    };

    VTDecompressionOutputCallbackRecord cb = {
        .decompressionOutputCallback        = output_callback,
        .decompressionOutputRefCon          = this,
    };

    OSStatus err = VTDecompressionSessionCreate(
        kCFAllocatorDefault,
        fmt_desc_,
        NULL,
        (__bridge CFDictionaryRef)out_props,
        &cb,
        &session_);

    if (err != noErr) {
        fprintf(stderr, "[VTAsync] VTDecompressionSessionCreate failed: %d\n", (int)err);
        if (fmt_desc_) { CFRelease(fmt_desc_); fmt_desc_ = nullptr; }
        return false;
    }

    flushing_ = false;
    return true;
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::output_callback — called by VT on an internal queue
// ---------------------------------------------------------------------------

void VTAsyncDecoder::output_callback(void *ctx, void */*source*/,
                                     OSStatus status, VTDecodeInfoFlags /*flags*/,
                                     CVImageBufferRef image_buffer,
                                     CMTime pts, CMTime /*duration*/) {
    if (status != noErr || !image_buffer) return;

    VTAsyncDecoder *self = reinterpret_cast<VTAsyncDecoder *>(ctx);
    CVPixelBufferRef pb = (CVPixelBufferRef)image_buffer;
    CFRetain(pb);

    std::lock_guard<std::mutex> lk(self->mutex_);
    self->queue_.push({pb, pts});
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::submit
// ---------------------------------------------------------------------------

void VTAsyncDecoder::submit(const uint8_t *data, size_t size,
                            int64_t pts, int64_t dts, double timebase_sec,
                            bool is_keyframe) {
    if (!session_) return;

    // Convert Annex-B → AVCC for VT
    std::vector<uint8_t> avcc = annexb_to_avcc(data, size);
    if (avcc.empty()) return;

    // Build CMTime from stream timestamps
    auto make_time = [&](int64_t ts) -> CMTime {
        if (ts == AV_NOPTS_VALUE)
            return kCMTimeInvalid;
        return CMTimeMakeWithSeconds(ts * timebase_sec, 1000000);
    };
    CMTime pts_t = make_time(pts);
    CMTime dts_t = make_time(dts);

    CMSampleBufferRef sample = vt_make_sample_buffer(
        avcc.data(), avcc.size(), pts_t, dts_t, fmt_desc_);
    if (!sample) return;

    VTDecodeInfoFlags info_out = 0;
    VTDecompressionSessionDecodeFrame(
        session_, sample,
        kVTDecodeFrame_EnableAsynchronousDecompression,
        NULL, &info_out);

    CFRelease(sample);
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::submit_blocking — synchronous version for seek
// ---------------------------------------------------------------------------

void VTAsyncDecoder::submit_blocking(const uint8_t *data, size_t size,
                                     int64_t pts, int64_t dts,
                                     double timebase_sec, bool /*is_keyframe*/) {
    if (!session_) return;

    std::vector<uint8_t> avcc = annexb_to_avcc(data, size);
    if (avcc.empty()) return;

    auto make_time = [&](int64_t ts) -> CMTime {
        if (ts == AV_NOPTS_VALUE)
            return kCMTimeInvalid;
        return CMTimeMakeWithSeconds(ts * timebase_sec, 1000000);
    };

    CMSampleBufferRef sample = vt_make_sample_buffer(
        avcc.data(), avcc.size(), make_time(pts), make_time(dts), fmt_desc_);
    if (!sample) return;

    VTDecodeInfoFlags info_out = 0;
    // Submit asynchronously, then wait for all pending frames to complete.
    // VTDecompressionSessionFinishDelayedFrames blocks until the output
    // callback has fired for every pending frame, giving us sync-like behavior.
    VTDecompressionSessionDecodeFrame(
        session_, sample,
        kVTDecodeFrame_EnableAsynchronousDecompression,
        NULL, &info_out);
    CFRelease(sample);

    VTDecompressionSessionFinishDelayedFrames(session_);
    // After this returns, the output callback has fired and the frame
    // is in the reorder queue — drain_one() can retrieve it immediately.
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::drain_one — pop regardless of reorder depth (seek phase)
// ---------------------------------------------------------------------------

CVPixelBufferRef VTAsyncDecoder::drain_one() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (queue_.empty()) return nullptr;
    FrameEntry e = queue_.top();
    queue_.pop();
    return e.buf;  // caller owns
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::pop_next
// ---------------------------------------------------------------------------

CVPixelBufferRef VTAsyncDecoder::pop_next() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (queue_.empty()) return nullptr;
    // Emit once the queue depth exceeds REORDER_DEPTH, or if flushing (EOS/seek)
    if ((int)queue_.size() >= REORDER_DEPTH || flushing_) {
        FrameEntry e = queue_.top();
        queue_.pop();
        return e.buf;  // caller owns; must CFRelease
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::flush
// ---------------------------------------------------------------------------

void VTAsyncDecoder::flush() {
    if (session_) {
        // Block until VT has delivered all pending decoded frames to the callback
        VTDecompressionSessionFinishDelayedFrames(session_);
    }
    // Mark flushing so pop_next() drains the queue completely
    {
        std::lock_guard<std::mutex> lk(mutex_);
        flushing_ = true;
    }
    // Drain and release any buffered CVPixelBuffers
    while (true) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (queue_.empty()) break;
        FrameEntry e = queue_.top();
        queue_.pop();
        CFRelease(e.buf);
    }
    flushing_ = false;
}

// ---------------------------------------------------------------------------
// VTAsyncDecoder::destroy
// ---------------------------------------------------------------------------

void VTAsyncDecoder::destroy() {
    if (session_) {
        VTDecompressionSessionInvalidate(session_);
        CFRelease(session_);
        session_ = nullptr;
    }
    if (fmt_desc_) {
        CFRelease(fmt_desc_);
        fmt_desc_ = nullptr;
    }
    // Release any remaining buffered frames
    std::lock_guard<std::mutex> lk(mutex_);
    while (!queue_.empty()) {
        CFRelease(queue_.top().buf);
        queue_.pop();
    }
    flushing_ = false;
}

#endif // __APPLE__
