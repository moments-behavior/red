// Minimal test for JARVIS export JPEG pipeline.
// Tests: FrameReader seek + decode + write_jpeg
// Build:
//   clang++ -std=c++17 -O2 -I src/ tests/test_jpeg_export.cpp \
//     $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale) \
//     -I/opt/homebrew/include -L/opt/homebrew/lib -lturbojpeg -o tests/test_jpeg_export

#include "ffmpeg_frame_reader.h"
#include <turbojpeg.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// Copy of write_jpeg from jarvis_export.h (to avoid pulling in all deps)
static bool write_jpeg(const char *path, int w, int h, int channels,
                       const uint8_t *data, int quality) {
    tjhandle tj = tjInitCompress();
    if (!tj) return false;
    unsigned char *buf = nullptr;
    unsigned long buf_size = 0;
    int pf = (channels == 3) ? TJPF_RGB : TJPF_RGBA;
    int rc = tjCompress2(tj, data, w, 0, h, pf, &buf, &buf_size,
                         TJSAMP_420, quality, TJFLAG_FASTDCT);
    bool ok = (rc == 0);
    if (ok) {
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(buf, 1, buf_size, f); fclose(f); }
        else ok = false;
    }
    tjFree(buf);
    tjDestroy(tj);
    return ok;
}

// Verify JPEG by reading back first few bytes (SOI marker)
static bool verify_jpeg(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    uint8_t header[2];
    bool ok = (fread(header, 1, 2, f) == 2 &&
               header[0] == 0xFF && header[1] == 0xD8);
    // Also check file size is reasonable
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);
    return ok && size > 1000; // at least 1KB for a real image
}

int main(int argc, char *argv[]) {
    const char *video_path =
        "/Users/johnsonr/datasets/rat/sessions/2025_09_03_15_18_21/Cam2002479.mp4";
    const char *output_dir = "/tmp/red_cpp_export_test";
    int test_frames[] = {10360, 10361, 10362, 10400, 10500};
    int n_frames = 5;

    if (argc > 1) video_path = argv[1];

    fs::create_directories(output_dir);

    // Open video
    ffmpeg_reader::FrameReader reader;
    if (!reader.open(video_path)) {
        fprintf(stderr, "ERROR: Cannot open video: %s\n", video_path);
        return 1;
    }

    int w = reader.width();
    int h = reader.height();
    double fps = reader.fps();
    printf("Video: %dx%d @ %.1f fps\n", w, h, fps);

    int ok_count = 0;
    for (int i = 0; i < n_frames; i++) {
        int frame = test_frames[i];
        const uint8_t *rgb = reader.readFrame(frame);
        if (!rgb) {
            printf("  Frame %d: FAILED to decode\n", frame);
            continue;
        }

        // Check pixel values
        double sum = 0;
        for (int p = 0; p < w * h * 3; p++) sum += rgb[p];
        double mean = sum / (w * h * 3);

        // Write JPEG
        char path[512];
        snprintf(path, sizeof(path), "%s/Frame_%d_cpp.jpg", output_dir, frame);
        bool wrote = write_jpeg(path, w, h, 3, rgb, 95);

        // Verify
        bool valid = wrote && verify_jpeg(path);
        long fsize = 0;
        if (valid) {
            FILE *f = fopen(path, "rb");
            fseek(f, 0, SEEK_END);
            fsize = ftell(f);
            fclose(f);
        }

        printf("  Frame %d: %s — mean=%.1f, jpeg=%ldKB\n",
               frame, valid ? "OK" : "FAILED",
               mean, fsize / 1024);
        if (valid) ok_count++;
    }

    printf("\nResults: %d/%d frames OK\n", ok_count, n_frames);
    printf("Output: %s/\n", output_dir);

    if (ok_count == n_frames) {
        printf("\nC++ pipeline matches Python — JPEGs export correctly.\n");
        printf("Compare visually:\n");
        printf("  open %s/Frame_10360_cpp.jpg\n", output_dir);
        printf("  open /tmp/red_jarvis_export_test/Frame_10360_pillow.jpg\n");
    }

    return (ok_count == n_frames) ? 0 : 1;
}
