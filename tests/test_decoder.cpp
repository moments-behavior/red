// test_decoder.cpp -- backend conformance test for the per-camera decoder.
//
// Drives the real decoder_process() thread against a generated video and
// checks the two things a labeling tool cannot afford to get wrong:
//
//   1. sequential playback delivers consecutive frame numbers, and
//   2. an accurate seek to frame N delivers frame N -- not N-1, not the
//      keyframe before it.
//
// The check is on PIXELS, not just on the label the decoder attached: the
// fixture video encodes its frame index in every pixel (R = (N%8)*32,
// G = (N/8)*32), so a frame carrying the wrong image fails even if the
// bookkeeping around it looks right. Generate one with:
//
//   ffmpeg -f lavfi -i "nullsrc=s=320x240:r=30:d=2" \
//     -vf "format=gbrp,geq=r='mod(N\,8)*32':g='floor(N/8)*32':b='0',format=yuv420p" \
//     -c:v libx264 -qp 0 -g 10 out.mp4
//
// Whichever backend red::decode_backend() resolves is the one under test, so
//   RED_DECODE_BACKEND=sw ./release/test_decoder out.mp4
//   RED_DECODE_BACKEND=hw ./release/test_decoder out.mp4
// exercise software and hardware through exactly the code path the app uses.

#include "test_framework.h"

// decoder.cpp's image-sequence loader pulls in stb_image; red.cpp normally
// provides the implementation and is not linked into test targets.
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../src/stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "../src/FFmpegDemuxer.h"
#include "../src/decode_backend.h"
#include "../src/decoder.h"
#include "../src/global.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#endif

namespace {

constexpr int kRing = 8;
const std::string kCam = "cam";

// Block until the producer publishes into `head`. Returns false on timeout
// rather than spinning forever -- a decoder that stalls should fail the test,
// not hang CI.
bool wait_slot(PictureBuffer *ring, int head, int timeout_ms = 8000) {
    auto start = std::chrono::steady_clock::now();
    while (ring[head].available_to_write.load()) {
        auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        if (waited > timeout_ms) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Centre pixel of a published slot, as RGB. The software backend fills
// PictureBuffer::frame with RGBA; VideoToolbox instead hands over a 32BGRA
// CVPixelBuffer and leaves .frame untouched, so both shapes are read here.
bool read_center_rgb(const PictureBuffer &pb, int w, int h, int rgb[3]) {
#ifdef __APPLE__
    if (pb.pixel_buffer) {
        CVPixelBufferRef b = pb.pixel_buffer;
        if (CVPixelBufferLockBaseAddress(b, kCVPixelBufferLock_ReadOnly) != 0)
            return false;
        const uint8_t *base = (const uint8_t *)CVPixelBufferGetBaseAddress(b);
        size_t stride = CVPixelBufferGetBytesPerRow(b);
        size_t bw = CVPixelBufferGetWidth(b), bh = CVPixelBufferGetHeight(b);
        bool ok = false;
        if (base && bw > 0 && bh > 0) {
            const uint8_t *px = base + (bh / 2) * stride + (bw / 2) * 4;
            rgb[0] = px[2]; // BGRA
            rgb[1] = px[1];
            rgb[2] = px[0];
            ok = true;
        }
        CVPixelBufferUnlockBaseAddress(b, kCVPixelBufferLock_ReadOnly);
        return ok;
    }
#endif
    if (!pb.frame) return false;
    const unsigned char *px = pb.frame + ((size_t)(h / 2) * w + (w / 2)) * 4;
    // Read with the DISPLAY's byte order (decoder.h), not the decoder's. A
    // producer writing RGBA where the uploader wants BGRA is invisible to a
    // test that reads back with the producer's own convention -- both agree on
    // the wrong answer and only the screen shows red where blue should be.
#if defined(RED_FRAME_BGRA)
    rgb[0] = px[2];
    rgb[1] = px[1];
    rgb[2] = px[0];
#else
    rgb[0] = px[0];
    rgb[1] = px[1];
    rgb[2] = px[2];
#endif
    return true;
}

int quantize32(int v, int *err) {
    int step = (v + 16) / 32;
    if (step > 7) step = 7;
    if (step < 0) step = 0;
    *err = std::abs(v - step * 32);
    return step;
}

// Frame index recovered from the fixture's colour encoding, or -1 if the
// pixel is too far off the 32-step grid to be trusted.
int index_from_pixel(const PictureBuffer &pb, int w, int h) {
    int rgb[3];
    if (!read_center_rgb(pb, w, h, rgb)) return -1;
    int er = 0, eg = 0;
    int lo = quantize32(rgb[0], &er);
    int hi = quantize32(rgb[1], &eg);
    if (er > 12 || eg > 12) {
        fprintf(stderr, "  (pixel %d,%d,%d is off the fixture grid)\n", rgb[0],
                rgb[1], rgb[2]);
        return -1;
    }
    return hi * 8 + lo;
}

void release(PictureBuffer *ring, int head) {
#ifdef __APPLE__
    if (ring[head].pixel_buffer) {
        CFRelease(ring[head].pixel_buffer);
        ring[head].pixel_buffer = nullptr;
    }
#endif
    ring[head].available_to_write = true;
}

// One accurate seek, mirroring seek_all_cameras(): request, wait for
// seek_done, clear it. The producer restarts at ring slot 0.
void do_seek(SeekInfo &si, int frame, bool accurate) {
    si.seek_frame = (uint64_t)frame;
    si.seek_accurate = accurate;
    si.use_seek = true;
    auto start = std::chrono::steady_clock::now();
    while (!si.seek_done) {
        auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        if (waited > 8000) {
            fprintf(stderr, "FAIL: seek to %d never completed\n", frame);
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    si.seek_done = false;
}


// ---------------------------------------------------------------------------
// Image-sequence path. image_loader() feeds the same ring from stills rather
// than a video, through load_image_rgba() -- a separate loader with its own
// channel-order decision, which is exactly where a red/blue swap hid.
// JPEG and PNG take different code paths (turbojpeg vs stb_image), so both are
// covered.
// ---------------------------------------------------------------------------
constexpr int kImgCount = 12;
constexpr int kImgBlue = 160;

bool write_fixture_image(const std::string &path, int idx, int w, int h,
                         bool jpeg) {
    std::vector<unsigned char> rgb((size_t)w * h * 3);
    const unsigned char r = (unsigned char)((idx % 8) * 32);
    const unsigned char g = (unsigned char)((idx / 8) * 32);
    for (size_t p = 0; p < (size_t)w * h; ++p) {
        rgb[p * 3 + 0] = r;
        rgb[p * 3 + 1] = g;
        rgb[p * 3 + 2] = (unsigned char)kImgBlue;
    }
    if (jpeg)
        return stbi_write_jpg(path.c_str(), w, h, 3, rgb.data(), 95) != 0;
    return stbi_write_png(path.c_str(), w, h, 3, rgb.data(), w * 3) != 0;
}

void run_image_loader_test(const std::string &dir, const char *ext, bool jpeg,
                           int w, int h) {
    printf("\n[image sequence: .%s]\n", ext);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const std::string cam = "imgcam";
    std::vector<std::string> names;
    for (int i = 0; i < kImgCount; i++) {
        char stem[32];
        snprintf(stem, sizeof(stem), "%06d", i);
        names.emplace_back(stem);
        const std::string path = dir + "/" + cam + "_" + stem + "." + ext;
        if (!write_fixture_image(path, i, w, h, jpeg)) {
            fprintf(stderr, "FAIL: could not write %s\n", path.c_str());
            ++g_fail;
            return;
        }
    }

    window_need_decoding[cam].store(true);
    latest_decoded_frame[cam].store(0);

    PictureBuffer *ring = new PictureBuffer[kRing]();
    for (int i = 0; i < kRing; i++) {
        ring[i].frame = (unsigned char *)calloc((size_t)w * h * 4, 1);
        ring[i].frame_number = -1;
        ring[i].available_to_write = true;
        ring[i].dropped = false;
#ifdef __APPLE__
        ring[i].pixel_buffer = nullptr;
#endif
    }

    DecoderContext dc{};
    SeekInfo si{};
    si.use_seek = false;
    si.seek_done = false;
    si.seek_frame = 0;
    si.seek_accurate = false;

    std::thread loader(image_loader, &dc, std::cref(names), ring, kRing, &si,
                       true, cam, dir, std::string(ext));

    int head = 0;
    for (int expected = 0; expected < kImgCount; expected++) {
        if (!wait_slot(ring, head)) {
            fprintf(stderr, "FAIL: image %d never arrived\n", expected);
            ++g_fail;
            break;
        }
        const int label = ring[head].frame_number.load();
        const int from_pixels = index_from_pixel(ring[head], w, h);
        int rgb[3] = {-1, -1, -1};
        const bool readable = read_center_rgb(ring[head], w, h, rgb);
        if (label != expected || from_pixels != expected) {
            fprintf(stderr, "FAIL: image slot %d -> label %d, pixels %d "
                            "(want %d)\n", head, label, from_pixels, expected);
            ++g_fail;
        } else if (!readable || std::abs(rgb[2] - kImgBlue) > 12) {
            fprintf(stderr,
                    "FAIL: image %d blue=%d want %d (rgb %d,%d,%d) -- "
                    "channel order wrong?\n",
                    expected, rgb[2], kImgBlue, rgb[0], rgb[1], rgb[2]);
            ++g_fail;
        } else {
            ++g_pass;
        }
        release(ring, head);
        head = (head + 1) % kRing;
    }
    printf("  %d images checked\n", kImgCount);

    dc.stop_flag = true;
    loader.join();
    for (int i = 0; i < kRing; i++) free(ring[i].frame);
    delete[] ring;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_decoder <video.mp4> [--expect-blue N]\n");
        return 2;
    }
    const char *video_path = argv[1];
    // The fixture paints a constant blue channel; asserting it catches a
    // red/blue swap, which the frame-index checks below cannot see (the index
    // lives in R and G, and R/B swaps leave G alone).
    int expect_blue = -1;
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--expect-blue") && i + 1 < argc)
            expect_blue = atoi(argv[++i]);

    printf("backend: %s (%s)\n", red::decode_backend_name(),
           red::decode_backend_reason());

    std::map<std::string, std::string> opts;
    FFmpegDemuxer demuxer(video_path, opts);
    const int w = (int)demuxer.GetWidth();
    const int h = (int)demuxer.GetHeight();
    const int n_frames = (int)demuxer.GetNumFrames();
    printf("video: %dx%d, %d frames, gop interval %lld\n", w, h, n_frames,
           (long long)demuxer.FindKeyFrameInterval());
    EXPECT_TRUE(n_frames >= 40);

    red::sw_decode_set_camera_count(1);
    window_need_decoding[kCam].store(true);
    latest_decoded_frame[kCam].store(0);

    PictureBuffer *ring = new PictureBuffer[kRing]();
    for (int i = 0; i < kRing; i++) {
        ring[i].frame = (unsigned char *)calloc((size_t)w * h * 4, 1);
        ring[i].frame_number = -1;
        ring[i].available_to_write = true;
        ring[i].dropped = false;
#ifdef __APPLE__
        ring[i].pixel_buffer = nullptr;
#endif
    }

    DecoderContext dc{};
    dc.gpu_index = 0;
    dc.seek_interval = (int)demuxer.FindKeyFrameInterval();
    dc.video_fps = demuxer.GetFramerate();
    dc.total_num_frame = n_frames;
    dc.estimated_num_frames = n_frames - 1;
    dc.sync_fix_active = false;
    dc.sync_canonical_len = 0;

    SeekInfo si{};
    si.use_seek = false;
    si.seek_done = false;
    si.seek_frame = 0;
    si.seek_accurate = false;

    std::thread decoder(decoder_process, &dc, &demuxer, kCam, ring, kRing, &si,
                        true, nullptr);

    // ---- Sequential playback ------------------------------------------
    printf("\n[sequential]\n");
    const int kSeq = 24;
    int head = 0;
    for (int expected = 0; expected < kSeq; expected++) {
        if (!wait_slot(ring, head)) {
            fprintf(stderr, "FAIL: timed out waiting for frame %d\n", expected);
            ++g_fail;
            break;
        }
        int label = ring[head].frame_number.load();
        int from_pixels = index_from_pixel(ring[head], w, h);
        if (label != expected || from_pixels != expected) {
            fprintf(stderr, "FAIL: slot %d -> label %d, pixels %d (want %d)\n",
                    head, label, from_pixels, expected);
            ++g_fail;
        } else {
            ++g_pass;
        }
        if (expect_blue >= 0) {
            int rgb[3] = {-1, -1, -1};
            if (!read_center_rgb(ring[head], w, h, rgb)) {
                fprintf(stderr, "FAIL: frame %d unreadable\n", expected);
                ++g_fail;
            } else if (std::abs(rgb[2] - expect_blue) > 12) {
                fprintf(stderr,
                        "FAIL: frame %d blue=%d want %d (rgb %d,%d,%d) -- "
                        "channel order wrong?\n",
                        expected, rgb[2], expect_blue, rgb[0], rgb[1], rgb[2]);
                ++g_fail;
            } else {
                ++g_pass;
            }
        }
        release(ring, head);
        head = (head + 1) % kRing;
    }

    // ---- Accurate seeks ------------------------------------------------
    // Targets straddle keyframe boundaries (gop 10) on purpose: one on a
    // keyframe, the rest at varying offsets into a GOP, and one backwards.
    printf("\n[accurate seek]\n");
    const int targets[] = {5, 17, 30, 33, 8, 20};
    for (int target : targets) {
        do_seek(si, target, true);
        head = 0;
        bool ok = true;
        // The seeked frame and the two after it: a seek that lands correctly
        // but leaves the decoder mis-primed shows up on the follow-on frames.
        for (int k = 0; k < 3; k++) {
            int expected = target + k;
            if (!wait_slot(ring, head)) {
                fprintf(stderr, "FAIL: seek %d, timed out on frame %d\n",
                        target, expected);
                ok = false;
                break;
            }
            int label = ring[head].frame_number.load();
            int from_pixels = index_from_pixel(ring[head], w, h);
            if (label != expected || from_pixels != expected) {
                fprintf(stderr,
                        "FAIL: seek %d, slot %d -> label %d, pixels %d "
                        "(want %d)\n",
                        target, head, label, from_pixels, expected);
                ok = false;
            }
            release(ring, head);
            head = (head + 1) % kRing;
        }
        if (ok) {
            ++g_pass;
            printf("  seek %2d ok\n", target);
        } else {
            ++g_fail;
        }
    }

    // ---- Non-accurate seek ---------------------------------------------
    // Playback seeks snap back to the enclosing keyframe and report where
    // they landed in seek_frame; the delivered pixels must match that.
    printf("\n[keyframe seek]\n");
    for (int target : {23, 44}) {
        do_seek(si, target, false);
        int landed = (int)si.seek_frame;
        head = 0;
        if (!wait_slot(ring, head)) {
            fprintf(stderr, "FAIL: keyframe seek %d timed out\n", target);
            ++g_fail;
            continue;
        }
        int label = ring[head].frame_number.load();
        int from_pixels = index_from_pixel(ring[head], w, h);
        EXPECT_TRUE(landed <= target);
        if (label == landed && from_pixels == landed) {
            ++g_pass;
            printf("  seek %2d landed on keyframe %d ok\n", target, landed);
        } else {
            fprintf(stderr,
                    "FAIL: keyframe seek %d -> landed %d, label %d, pixels %d\n",
                    target, landed, label, from_pixels);
            ++g_fail;
        }
        release(ring, head);
    }

    dc.stop_flag = true;
    decoder.join();

    run_image_loader_test(
        (std::filesystem::path(video_path).parent_path() / "imgseq").string(),
        "png", false, 64, 48);
    run_image_loader_test(
        (std::filesystem::path(video_path).parent_path() / "imgseq").string(),
        "jpg", true, 64, 48);

    for (int i = 0; i < kRing; i++) {
#ifdef __APPLE__
        if (ring[i].pixel_buffer) CFRelease(ring[i].pixel_buffer);
#endif
        free(ring[i].frame);
    }
    delete[] ring;

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
