// tests/test_jarvis_hybridnet_smoke.cpp
//
// Smoke-test the JARVIS HybridNet 3D inference pipeline end-to-end against
// real vertcyl1 data. No UI. Decodes one frame from each of the 16 H.264
// videos, loads camera calibration, runs predict_frame, prints 3D keypoints,
// and compares to the reference data3D.csv from the older April-19 model run.
//
// Build: cmake --build release --target test_jarvis_hybridnet_smoke -j
// Run:   ./release/test_jarvis_hybridnet_smoke

// stb_image generated in this TU (decoder.cpp references it).
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#include "annotation.h"
#include "camera.h"
#include "jarvis_hybridnet.h"
#include "red_math.h"
#include "skeleton.h"
// `logger` singleton lives in src/utils.cpp — pulled in via TEST_GUI_SRC_FILES.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

struct DecodedFrame {
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
};

static DecodedFrame decode_first_frame(const std::string &path) {
    DecodedFrame out;
    AVFormatContext *fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) return out;
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        avformat_close_input(&fmt); return out;
    }
    int vidx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; ++i) {
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vidx = i; break;
        }
    }
    if (vidx < 0) { avformat_close_input(&fmt); return out; }
    const AVCodec *codec = avcodec_find_decoder(fmt->streams[vidx]->codecpar->codec_id);
    AVCodecContext *cc = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(cc, fmt->streams[vidx]->codecpar);
    if (avcodec_open2(cc, codec, nullptr) < 0) {
        avcodec_free_context(&cc); avformat_close_input(&fmt); return out;
    }

    AVFrame *frame = av_frame_alloc();
    AVPacket *pkt = av_packet_alloc();
    bool got = false;
    while (!got && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == vidx) {
            if (avcodec_send_packet(cc, pkt) == 0) {
                if (avcodec_receive_frame(cc, frame) == 0) got = true;
            }
        }
        av_packet_unref(pkt);
    }
    if (got) {
        out.w = frame->width;
        out.h = frame->height;
        out.rgb.resize(static_cast<size_t>(out.w) * out.h * 3);
        SwsContext *sws = sws_getContext(
            out.w, out.h, static_cast<AVPixelFormat>(frame->format),
            out.w, out.h, AV_PIX_FMT_RGB24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        uint8_t *dst[1] = {out.rgb.data()};
        int dst_stride[1] = {out.w * 3};
        sws_scale(sws, frame->data, frame->linesize, 0, out.h, dst, dst_stride);
        sws_freeContext(sws);
    }
    av_frame_free(&frame); av_packet_free(&pkt);
    avcodec_free_context(&cc); avformat_close_input(&fmt);
    return out;
}

int main(int /*argc*/, char ** /*argv*/) {
    const std::string MODEL_DIR =
        "/data0/quanshare/mouse_merge_24kp_aug/onnx";
    const std::string VIDEO_DIR =
        "/data0/quanshare/mouse_w_cyl_w_calib/vertical_cyl/2026_04_17_16_49_04";
    const std::string CALIB_DIR =
        "/data0/quanshare/mouse_w_cyl_w_calib/calib/calibration";
    const std::string REF_CSV =
        "/data0/quanshare/mouse_w_cyl_w_calib/new_model_prediction/vertical_cyl/data3D.csv";

    const std::vector<std::string> cam_names = {
        "Cam2002486", "Cam2002487", "Cam2005325", "Cam2006050",
        "Cam2006051", "Cam2006052", "Cam2006054", "Cam2006055",
        "Cam2006515", "Cam2006516", "Cam2008665", "Cam2008666",
        "Cam2008667", "Cam2008668", "Cam2008669", "Cam2008670",
    };
    const int N = static_cast<int>(cam_names.size());

#ifndef RED_HAS_ONNXRUNTIME
    std::cerr << "Built without RED_HAS_ONNXRUNTIME — cannot run.\n";
    return 1;
#else
    std::cout << "Loading HybridNet ONNX models from " << MODEL_DIR << "...\n";
    JarvisHybridNetState state;
    if (!jarvis_hybridnet_load(state, MODEL_DIR)) {
        std::cerr << "FAILED: jarvis_hybridnet_load\n";
        return 1;
    }
    std::cout << "  cfg.num_cameras=" << state.cfg.num_cameras
              << " cfg.num_joints=" << state.cfg.num_joints
              << " cfg.bbox=" << state.cfg.keypoint_bbox_size
              << "\n";

    std::cout << "Loading 16 calibration YAMLs from " << CALIB_DIR << "...\n";
    std::vector<CameraParams> cam_params(N);
    for (int c = 0; c < N; ++c) {
        std::string yaml = CALIB_DIR + "/" + cam_names[c] + ".yaml";
        std::string err;
        if (!camera_load_params_from_yaml(yaml, cam_params[c], err)) {
            std::cerr << "FAILED calib " << cam_names[c] << ": " << err << "\n";
            return 1;
        }
    }

    std::cout << "Decoding frame 0 from each of " << N << " videos...\n";
    std::vector<DecodedFrame> frames(N);
    std::vector<const uint8_t *> rgbs(N);
    std::vector<int> widths(N), heights(N);
    for (int c = 0; c < N; ++c) {
        frames[c] = decode_first_frame(VIDEO_DIR + "/" + cam_names[c] + ".mp4");
        if (frames[c].rgb.empty()) {
            std::cerr << "FAILED decode " << cam_names[c] << ".mp4\n";
            return 1;
        }
        rgbs[c] = frames[c].rgb.data();
        widths[c] = frames[c].w;
        heights[c] = frames[c].h;
    }
    std::cout << "  first cam: " << widths[0] << "x" << heights[0] << "\n";

    SkeletonContext skel{};
    skel.num_nodes = state.cfg.num_joints;
    skel.num_edges = 0;
    skel.has_skeleton = true;

    AnnotationMap annotations;
    std::cout << "Running jarvis_hybridnet_predict_frame on frame 0...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = jarvis_hybridnet_predict_frame(
        state, rgbs, widths, heights, cam_params,
        annotations, skel, 0u);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        std::cerr << "predict_frame returned false (likely <2 cams detected center).\n";
        return 2;
    }
    std::cout << "  total: " << dt_ms << " ms"
              << "  (CenterDetect=" << state.last_center_ms
              << "  effTrack=" << state.last_efftrack_ms
              << "  Hybrid3D=" << state.last_hybrid3d_ms << ")\n";

    auto it = annotations.find(0u);
    if (it == annotations.end()) {
        std::cerr << "annotations map missing frame 0\n";
        return 3;
    }
    const FrameAnnotation &fa = it->second;

    std::cout << "\n=== Predicted 3D keypoints (frame 0) ===\n";
    std::cout << "Keypoint            x         y         z    conf\n";
    for (int j = 0; j < state.cfg.num_joints; ++j) {
        const std::string &name =
            (j < static_cast<int>(state.cfg.keypoint_names.size()))
            ? state.cfg.keypoint_names[j] : std::to_string(j);
        std::printf("%-18s %9.3f %9.3f %9.3f  %.4f\n",
                    name.c_str(), fa.kp3d[j].x, fa.kp3d[j].y, fa.kp3d[j].z,
                    fa.kp3d[j].confidence);
    }

    std::ifstream ref(REF_CSV);
    if (!ref) {
        std::cout << "\n(no reference CSV at " << REF_CSV << " — skipping compare)\n";
        return 0;
    }
    std::string line;
    std::getline(ref, line);   // names
    std::getline(ref, line);   // x/y/z/conf
    std::getline(ref, line);   // frame 0
    std::vector<double> vals;
    size_t pos = 0;
    while (pos < line.size()) {
        size_t comma = line.find(',', pos);
        if (comma == std::string::npos) comma = line.size();
        vals.push_back(std::stod(line.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    if (vals.size() < static_cast<size_t>(state.cfg.num_joints) * 4) {
        std::cerr << "Reference row too short: " << vals.size() << " values\n";
        return 0;
    }

    std::cout << "\n=== Reference (data3D.csv frame 0) vs prediction ===\n";
    std::cout << "Keypoint           x_ref     y_ref     z_ref  c_ref     delta\n";
    double sum_dist = 0; double max_dist = 0;
    for (int j = 0; j < state.cfg.num_joints; ++j) {
        double rx = vals[j * 4 + 0], ry = vals[j * 4 + 1];
        double rz = vals[j * 4 + 2], rc = vals[j * 4 + 3];
        double dx = fa.kp3d[j].x - rx, dy = fa.kp3d[j].y - ry, dz = fa.kp3d[j].z - rz;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        const std::string &name =
            (j < static_cast<int>(state.cfg.keypoint_names.size()))
            ? state.cfg.keypoint_names[j] : std::to_string(j);
        std::printf("%-18s %9.3f %9.3f %9.3f  %.4f   %7.2f mm\n",
                    name.c_str(), rx, ry, rz, rc, dist);
        sum_dist += dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::cout << "\nMean 3D delta: " << (sum_dist / state.cfg.num_joints) << " mm\n";
    std::cout << "Max  3D delta: " << max_dist << " mm\n";
    std::cout << "(ref is from older April-19 model; some delta is expected)\n";
    return 0;
#endif
}
