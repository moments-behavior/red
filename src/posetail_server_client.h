#pragma once
// posetail_server_client.h — Client for the posetail HTTP inference server.
//
// One round-trip = one 16-frame chunk × N cameras × N query points. Mirrors
// the I/O of posetail_predict_chunk() in posetail_infer.h so the same callers
// can swap between local ONNX and remote server. The server lives at
// posetail/server/server.py — see SERVER.md for the wire format.
//
// Wire format:
//   POST /predict
//     form field "metadata" : JSON {cameras, coords, query_times?}
//     repeated file "images" : <cam_name>__<frame_idx>.png   (PNG of the
//                                                              256×256 crop)
//   Response: an .npz (ZIP of .npy) with keys coords_pred, vis_pred, conf_pred,
//   ... we only read those three.
//
// Build deps (header-only, no system libs):
//   lib/httplib/httplib.h
//   lib/miniz/miniz.h  (+ miniz.c, compiled in once globally)
//   OpenCV (already linked by red) for PNG encode and resize.

#include "camera.h"
#include "posetail_infer.h"  // for posetail_detail::compute_crop_box / T_CHUNK / CROP_SIZE

#include <Eigen/Core>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// cpp-httplib brings in <thread>/<mutex>/<atomic>; keep it isolated to a TU
// that defines CPPHTTPLIB_OPENSSL_SUPPORT only if we ever need HTTPS. The
// posetail server is plain HTTP (port 8000), so we leave SSL off.
#include "httplib.h"

// miniz: ZIP reader for the .npz response. miniz.c is compiled once at
// the CMake level; here we only include the header.
#include "miniz.h"


struct PosetailServerState {
    // User-editable URL like "http://10.102.10.88:8000".
    std::string url;
    bool reachable = false;
    int n_frames = 0;       // from /info — must match posetail_detail::T_CHUNK
    int image_size = 0;     // from /info — must match posetail_detail::CROP_SIZE
    std::string device;     // from /info, informational only
    std::string mode_3d;    // from /info, informational only
    std::string status;     // last-action message for the UI

    // Most-recent-call timings.
    float last_total_ms = 0.0f;       // crop+encode+upload+infer+download+decode
    float last_encode_ms = 0.0f;      // crop + cv::imencode of all images
    float last_request_ms = 0.0f;     // POST round-trip
    float last_decode_ms = 0.0f;      // npz/npy parse + tensor unpack

    // Last call's vis/conf for UI display. Indexed [t][n] like the local path.
    std::vector<std::vector<float>> last_vis;
    std::vector<std::vector<float>> last_conf;
};


namespace posetail_server_detail {

// Split "http://host:port" into ("http://host:port", "/") for httplib::Client.
// cpp-httplib's Client takes scheme+host[+port] only; the path goes on Get/Post.
// We also accept "host:port" (no scheme) and default to http.
inline std::string normalize_url(const std::string &u) {
    if (u.rfind("http://", 0) == 0 || u.rfind("https://", 0) == 0)
        return u;
    return std::string("http://") + u;
}

// Read [start..end) little-endian into a uint64_t. ZIP/NPY headers are LE.
inline uint64_t read_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
inline uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

// Minimal .npy parser. Reads dtype string (e.g. "<f4") and shape tuple from
// the ASCII Python-literal header. Returns the raw data pointer + element
// count. We don't validate everything — just enough to consume coords_pred /
// vis_pred / conf_pred which are always little-endian float32, C-order.
struct NpyView {
    bool ok = false;
    std::string descr;          // e.g. "<f4"
    std::vector<int64_t> shape; // row-major
    const uint8_t *data = nullptr;
    size_t data_bytes = 0;
};

inline NpyView parse_npy(const uint8_t *buf, size_t n) {
    NpyView v;
    if (n < 10) return v;
    if (std::memcmp(buf, "\x93NUMPY", 6) != 0) return v;
    uint8_t major = buf[6];
    // uint8_t minor = buf[7];  (unused)
    size_t header_len = 0;
    size_t data_off = 0;
    if (major == 1) {
        header_len = read_u16_le(buf + 8);
        data_off = 10 + header_len;
    } else {
        // v2/v3: 4-byte header_len
        if (n < 12) return v;
        header_len = read_u32_le(buf + 8);
        data_off = 12 + header_len;
    }
    if (data_off > n) return v;

    std::string header((const char *)buf + (data_off - header_len), header_len);

    // Extract 'descr':
    {
        auto k = header.find("'descr'");
        if (k == std::string::npos) k = header.find("\"descr\"");
        if (k == std::string::npos) return v;
        auto q1 = header.find('\'', k + 7);
        auto q2 = (q1 == std::string::npos) ? std::string::npos
                                            : header.find('\'', q1 + 1);
        if (q2 == std::string::npos) return v;
        v.descr = header.substr(q1 + 1, q2 - q1 - 1);
    }
    // Extract 'shape': (N, M, ...) — could be empty for scalars.
    {
        auto k = header.find("'shape'");
        if (k == std::string::npos) return v;
        auto lp = header.find('(', k);
        auto rp = (lp == std::string::npos) ? std::string::npos
                                            : header.find(')', lp);
        if (rp == std::string::npos) return v;
        std::string inside = header.substr(lp + 1, rp - lp - 1);
        // Strip whitespace, then split on ','.
        std::string cur;
        for (char c : inside) {
            if (c == ',') {
                // trim
                while (!cur.empty() && (cur.back() == ' ' || cur.back() == '\t'))
                    cur.pop_back();
                size_t s = 0;
                while (s < cur.size() && (cur[s] == ' ' || cur[s] == '\t')) ++s;
                if (s < cur.size()) {
                    try { v.shape.push_back((int64_t)std::stoll(cur.substr(s))); }
                    catch (...) { return v; }
                }
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        // trailing element (no comma — e.g. shape (5))
        std::string trimmed;
        for (char c : cur) if (c != ' ' && c != '\t') trimmed.push_back(c);
        if (!trimmed.empty()) {
            try { v.shape.push_back((int64_t)std::stoll(trimmed)); }
            catch (...) { return v; }
        }
    }

    // Compute element size from descr ("<f4", "<f8", "<i4", "|b1"...).
    size_t esz = 0;
    if (v.descr.size() >= 3) {
        try { esz = (size_t)std::stoul(v.descr.substr(2)); } catch (...) {}
    }
    if (esz == 0) return v;
    int64_t nelem = 1;
    for (auto d : v.shape) nelem *= d;
    v.data = buf + data_off;
    v.data_bytes = (size_t)nelem * esz;
    if (data_off + v.data_bytes > n) return v;
    v.ok = true;
    return v;
}

// Extract one named entry from an .npz (= ZIP of .npy) using miniz.
// Returns the raw .npy bytes (caller passes to parse_npy).
inline std::vector<uint8_t> npz_extract(const std::vector<uint8_t> &zip_bytes,
                                         const std::string &name) {
    std::vector<uint8_t> out;
    mz_zip_archive z;
    std::memset(&z, 0, sizeof(z));
    if (!mz_zip_reader_init_mem(&z, zip_bytes.data(), zip_bytes.size(), 0))
        return out;

    int idx = mz_zip_reader_locate_file(&z, (name + ".npy").c_str(), nullptr, 0);
    if (idx < 0) {
        // np.savez sometimes writes without the .npy suffix in older formats,
        // but the standard is to include it. Try without just in case.
        idx = mz_zip_reader_locate_file(&z, name.c_str(), nullptr, 0);
    }
    if (idx < 0) {
        mz_zip_reader_end(&z);
        return out;
    }
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&z, idx, &st)) {
        mz_zip_reader_end(&z);
        return out;
    }
    out.resize((size_t)st.m_uncomp_size);
    if (!mz_zip_reader_extract_to_mem(&z, idx, out.data(), out.size(), 0))
        out.clear();
    mz_zip_reader_end(&z);
    return out;
}

// One per-camera prepared image: PNG-encoded 256×256 BGR, with the K_scaled
// and offset that go into the metadata for this camera.
struct PreparedImage {
    std::vector<uint8_t> png_bytes;
    Eigen::Matrix3d k_scaled;
    Eigen::Vector2f offset;
};

// Crop+resize one camera's full-resolution RGBA frame to a 256×256 BGR PNG,
// using the same crop box logic as the local ONNX path. The crop box must
// match across the whole 16-frame chunk for that camera, so callers pass the
// pre-computed CropBox (one per camera, shared across all 16 frames).
inline std::vector<uint8_t> encode_crop_png(
    const uint8_t *rgba, int src_w, int src_h,
    const posetail_detail::CropBox &box, int png_compress_level = 1) {
    if (!rgba) return {};
    // Clamp box to image bounds defensively (compute_crop_box already does
    // this, but the display buffer may have stale dims for transitions).
    int x0 = std::max(0, box.x0);
    int y0 = std::max(0, box.y0);
    int w = std::min(box.w, src_w - x0);
    int h = std::min(box.h, src_h - y0);
    if (w <= 0 || h <= 0) return {};

    // Wrap the full RGBA frame as a Mat header (no copy), take a sub-rect
    // view (no copy), and ONLY convert+resize the crop. The naive version
    // — cvtColor on the full 3216×2208 RGBA first — burned ~28 MB × cams ×
    // frames in CPU work that gets immediately thrown away.
    cv::Mat src(src_h, src_w, CV_8UC4, (void *)rgba);
    cv::Mat crop_rgba = src(cv::Rect(x0, y0, w, h));
    cv::Mat crop_bgr;
    cv::cvtColor(crop_rgba, crop_bgr, cv::COLOR_RGBA2BGR);
    cv::Mat resized;
    cv::resize(crop_bgr, resized, cv::Size(posetail_detail::CROP_SIZE,
                                            posetail_detail::CROP_SIZE),
               0, 0, cv::INTER_LINEAR);
    std::vector<uint8_t> out;
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, png_compress_level};
    cv::imencode(".png", resized, out, params);
    return out;
}

}  // namespace posetail_server_detail


// Hit /info and populate state. Returns true if the server responded and
// the model matches what red expects (n_frames=16, image_size=256).
inline bool posetail_server_probe(PosetailServerState &s) {
    s.reachable = false;
    if (s.url.empty()) {
        s.status = "Server URL is empty";
        return false;
    }
    std::string base = posetail_server_detail::normalize_url(s.url);
    httplib::Client cli(base);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(5, 0);
    auto res = cli.Get("/info");
    if (!res) {
        s.status = "Cannot reach server (" + base + "): " +
                   httplib::to_string(res.error());
        return false;
    }
    if (res->status != 200) {
        s.status = "GET /info failed: HTTP " + std::to_string(res->status);
        return false;
    }
    // Very small JSON; do scrappy parsing instead of pulling in a dep.
    const std::string &body = res->body;
    auto find_int = [&](const char *key) -> int {
        std::string k = std::string("\"") + key + "\":";
        auto p = body.find(k);
        if (p == std::string::npos) return -1;
        p += k.size();
        while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) ++p;
        int sign = 1;
        if (p < body.size() && body[p] == '-') { sign = -1; ++p; }
        int v = 0;
        while (p < body.size() && body[p] >= '0' && body[p] <= '9') {
            v = v * 10 + (body[p] - '0');
            ++p;
        }
        return sign * v;
    };
    auto find_str = [&](const char *key) -> std::string {
        std::string k = std::string("\"") + key + "\":";
        auto p = body.find(k);
        if (p == std::string::npos) return {};
        p += k.size();
        while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) ++p;
        if (p >= body.size() || body[p] != '"') return {};
        ++p;
        auto end = body.find('"', p);
        if (end == std::string::npos) return {};
        return body.substr(p, end - p);
    };
    s.n_frames = find_int("n_frames");
    s.image_size = find_int("image_size");
    s.device = find_str("device");
    s.mode_3d = find_str("mode_3d");
    s.reachable = true;

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "Server OK: n_frames=%d image_size=%d device=%s mode=%s",
                  s.n_frames, s.image_size, s.device.c_str(),
                  s.mode_3d.c_str());
    s.status = buf;
    if (s.n_frames != posetail_detail::T_CHUNK ||
        s.image_size != posetail_detail::CROP_SIZE) {
        s.status += "  (WARNING: doesn't match red's hardcoded 16×256)";
    }
    return true;
}


// Run one chunk through the remote server. Drop-in replacement for
// posetail_predict_chunk() — same inputs, same PosetailChunkResult layout.
//
// cam_names_opt : optional per-camera names. If empty, uses "0", "1", ... in
//                 order. The server matches uploaded images to cameras by
//                 name, so anything consistent is fine.
inline PosetailChunkResult posetail_server_predict_chunk(
    PosetailServerState &s,
    const std::vector<const uint8_t *> &frames_rgba_per_cam_per_t,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const std::vector<CameraParams> &cams,
    const std::vector<Eigen::Vector3d> &seed_3d,
    int seed_t = 0,
    const std::vector<std::string> &cam_names_opt = {}) {
    using namespace posetail_detail;
    using namespace posetail_server_detail;

    PosetailChunkResult r;
    int num_cams = (int)cams.size();
    int N = (int)seed_3d.size();
    if (num_cams == 0 || N == 0) {
        r.error = "No cameras or queries";
        return r;
    }
    if ((int)frames_rgba_per_cam_per_t.size() != num_cams * T_CHUNK) {
        r.error = "frames buffer size mismatch (need cams*16)";
        return r;
    }
    if (s.url.empty()) {
        r.error = "Server URL not set";
        return r;
    }

    auto t0 = std::chrono::steady_clock::now();

    // Per-camera crop boxes from seed 3D — identical to the local path so the
    // model sees the same input geometry.
    std::vector<CropBox> boxes(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        boxes[c] = compute_crop_box(seed_3d, cams[c], cam_widths[c],
                                    cam_heights[c]);
    }

    // ── Encode all (cam, frame) images to PNG ──
    // PNG @ compression level 1 ≈ ~2-3× JPEG-90 in bytes for 256×256 crops but
    // lossless; the model was trained on raw frames so we keep it lossless.
    // Switch to JPEG here if upload time is the bottleneck.
    httplib::MultipartFormDataItems items;
    items.reserve((size_t)num_cams * T_CHUNK + 1);

    auto t_enc0 = std::chrono::steady_clock::now();
    int encoded = 0, skipped = 0;
    for (int c = 0; c < num_cams; ++c) {
        std::string cam_name = (c < (int)cam_names_opt.size() &&
                                 !cam_names_opt[c].empty())
            ? cam_names_opt[c]
            : std::to_string(c);
        for (int t = 0; t < T_CHUNK; ++t) {
            const uint8_t *rgba = frames_rgba_per_cam_per_t[c * T_CHUNK + t];
            std::vector<uint8_t> png =
                encode_crop_png(rgba, cam_widths[c], cam_heights[c], boxes[c]);
            if (png.empty()) {
                skipped++;
                // Send a tiny solid-grey PNG instead so the server still
                // sees T_CHUNK images and we don't break the request.
                cv::Mat grey(CROP_SIZE, CROP_SIZE, CV_8UC3,
                             cv::Scalar(128, 128, 128));
                cv::imencode(".png", grey, png);
            } else {
                encoded++;
            }
            httplib::MultipartFormData item;
            item.name = "images";
            char fname[64];
            std::snprintf(fname, sizeof(fname), "%s__%06d.png",
                          cam_name.c_str(), t);
            item.filename = fname;
            item.content_type = "image/png";
            item.content.assign((const char *)png.data(), png.size());
            items.push_back(std::move(item));
        }
    }
    auto t_enc1 = std::chrono::steady_clock::now();
    s.last_encode_ms =
        std::chrono::duration<float, std::milli>(t_enc1 - t_enc0).count();

    // ── Build the metadata JSON ──
    // mat = K_scaled (256×256-space), dist = first 5 distortion coeffs,
    // ext = [R | t; 0 0 0 1] world→camera, offset = scaled (x0*sx, y0*sy).
    // Because the uploaded PNG is already 256×256, the server's
    // resize_camera_group computes scale=1.0 and leaves everything alone.
    std::ostringstream meta;
    meta << std::scientific;
    meta.precision(9);
    meta << "{\"cameras\":[";
    for (int c = 0; c < num_cams; ++c) {
        if (c) meta << ",";
        std::string cam_name = (c < (int)cam_names_opt.size() &&
                                 !cam_names_opt[c].empty())
            ? cam_names_opt[c]
            : std::to_string(c);
        const auto &cam = cams[c];
        const auto &box = boxes[c];

        meta << "{\"name\":\"" << cam_name << "\",\"type\":\"pinhole\","
             << "\"mat\":[["
             << box.K_scaled(0, 0) << "," << box.K_scaled(0, 1) << ","
             << box.K_scaled(0, 2) << "],["
             << box.K_scaled(1, 0) << "," << box.K_scaled(1, 1) << ","
             << box.K_scaled(1, 2) << "],["
             << box.K_scaled(2, 0) << "," << box.K_scaled(2, 1) << ","
             << box.K_scaled(2, 2) << "]],"
             << "\"dist\":["
             << cam.dist_coeffs(0) << "," << cam.dist_coeffs(1) << ","
             << cam.dist_coeffs(2) << "," << cam.dist_coeffs(3) << ","
             << cam.dist_coeffs(4) << "],"
             << "\"ext\":[";
        for (int i = 0; i < 3; ++i) {
            meta << "[" << cam.r(i, 0) << "," << cam.r(i, 1) << ","
                 << cam.r(i, 2) << "," << cam.tvec(i) << "],";
        }
        meta << "[0,0,0,1]],"
             << "\"offset\":[" << (double)box.offset(0) << ","
             << (double)box.offset(1) << "]}";
    }
    meta << "],\"coords\":[";
    for (int n = 0; n < N; ++n) {
        if (n) meta << ",";
        meta << "[" << seed_3d[n](0) << "," << seed_3d[n](1) << ","
             << seed_3d[n](2) << "]";
    }
    meta << "],\"query_times\":[";
    for (int n = 0; n < N; ++n) {
        if (n) meta << ",";
        meta << seed_t;
    }
    meta << "]}";

    httplib::MultipartFormData meta_item;
    meta_item.name = "metadata";
    meta_item.content = meta.str();
    meta_item.content_type = "application/json";
    items.push_back(std::move(meta_item));

    // ── POST ──
    std::string base = normalize_url(s.url);
    httplib::Client cli(base);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(120, 0);   // big POSTs + GPU forward can take a while
    cli.set_write_timeout(60, 0);

    auto t_req0 = std::chrono::steady_clock::now();
    auto res = cli.Post("/predict", items);
    auto t_req1 = std::chrono::steady_clock::now();
    s.last_request_ms =
        std::chrono::duration<float, std::milli>(t_req1 - t_req0).count();

    if (!res) {
        r.error = std::string("HTTP error: ") + httplib::to_string(res.error());
        return r;
    }
    if (res->status != 200) {
        // Server returns 400 text/plain on validation errors; surface it.
        r.error = "HTTP " + std::to_string(res->status) + ": " +
                  res->body.substr(0, 400);
        return r;
    }

    // ── Decode the NPZ ──
    auto t_dec0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> body(res->body.begin(), res->body.end());

    auto unpack = [&](const char *key,
                       std::vector<std::vector<float>> &dst_2d_TN1,
                       std::vector<std::vector<Eigen::Vector3d>> *dst_kp = nullptr)
        -> bool {
        std::vector<uint8_t> npy = npz_extract(body, key);
        if (npy.empty()) {
            r.error = std::string("NPZ missing key: ") + key;
            return false;
        }
        NpyView v = parse_npy(npy.data(), npy.size());
        if (!v.ok) {
            r.error = std::string("Malformed .npy for key: ") + key;
            return false;
        }
        if (v.descr != "<f4") {
            r.error = std::string("Unexpected dtype ") + v.descr +
                      " for key " + key + " (need <f4)";
            return false;
        }
        const float *fp = reinterpret_cast<const float *>(v.data);

        if (dst_kp) {
            // coords_pred: (B=1, T, N, 3)
            if (v.shape.size() != 4 ||
                v.shape[0] != 1 ||
                v.shape[1] != T_CHUNK ||
                v.shape[2] != N ||
                v.shape[3] != 3) {
                r.error = std::string("Bad shape for ") + key;
                return false;
            }
            dst_kp->assign(T_CHUNK, std::vector<Eigen::Vector3d>(N));
            for (int t = 0; t < T_CHUNK; ++t)
                for (int n = 0; n < N; ++n) {
                    const float *p = fp + ((t * N) + n) * 3;
                    (*dst_kp)[t][n] = Eigen::Vector3d(p[0], p[1], p[2]);
                }
        } else {
            // vis/conf: server returns either (B=1, T, N, 1) or (B=1, T, N).
            // Docs say the former, but the live posetail-odyssey checkpoint
            // omits the trailing-1 for conf_pred. Accept both: same N*T
            // float layout in memory either way.
            bool ok_shape = false;
            if (v.shape.size() == 4 &&
                v.shape[0] == 1 && v.shape[1] == T_CHUNK &&
                v.shape[2] == N && v.shape[3] == 1)
                ok_shape = true;
            else if (v.shape.size() == 3 &&
                     v.shape[0] == 1 && v.shape[1] == T_CHUNK &&
                     v.shape[2] == N)
                ok_shape = true;
            if (!ok_shape) {
                r.error = std::string("Bad shape for ") + key;
                return false;
            }
            dst_2d_TN1.assign(T_CHUNK, std::vector<float>(N, 0.0f));
            for (int t = 0; t < T_CHUNK; ++t)
                for (int n = 0; n < N; ++n)
                    dst_2d_TN1[t][n] = fp[(t * N + n)];
        }
        return true;
    };

    std::vector<std::vector<float>> _dummy;
    if (!unpack("coords_pred", _dummy, &r.kp3d)) return r;
    if (!unpack("vis_pred", r.vis)) return r;
    if (!unpack("conf_pred", r.conf)) return r;

    auto t_dec1 = std::chrono::steady_clock::now();
    s.last_decode_ms =
        std::chrono::duration<float, std::milli>(t_dec1 - t_dec0).count();
    auto t1 = std::chrono::steady_clock::now();
    s.last_total_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    s.last_vis = r.vis;
    s.last_conf = r.conf;

    fprintf(stderr,
            "[PoseTail/server] cams=%d N=%d T=%d  encode=%.1f ms (%d ok, %d "
            "filled) request=%.1f ms decode=%.1f ms  total=%.1f ms\n",
            num_cams, N, T_CHUNK, s.last_encode_ms, encoded, skipped,
            s.last_request_ms, s.last_decode_ms, s.last_total_ms);
    r.ok = true;
    return r;
}
