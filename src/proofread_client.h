#pragma once
// proofread_client.h — Client for the JARVIS pose-proofreading server.
//
// One round-trip = every session that has finished IK, with its bad-frame
// indices (residual >= threshold) already debounced into per-cluster
// worst-frame picks. Red uses this to drive a proofreading workflow:
// the user picks an (animal, session) row, walks the bad frames, and
// re-labels keypoints in red as usual.
//
// Server lives at mouse_dashboard/app.py — see /api/bad_frames_all.
//
// Wire format:
//   GET /api/bad_frames_all?residual_threshold_mm=25&min_gap=50
//   200 OK, application/json:
//     {
//       "threshold_mm": 25.0,
//       "min_gap":      50,
//       "variant":      "original",
//       "generated_at": "2026-06-17T12:34:56",
//       "n_sessions":   45,
//       "n_bad_total":  24332,
//       "sessions": [
//         {
//           "animal":          "rat",
//           "session":         "2026_05_21_12_57_09",
//           "date":            "2026_05_21",
//           "recording_path":  "/mnt/free/rat/2026_05_21_12_57_09",
//           "prediction_path": "/home/user/mouse_foundation_model_data/...",
//           "n_frames_total":  108871,
//           "n_frames_bad":    234,
//           "frames":          [12, 87, 145, ...],
//           "residuals_mm":    [42.1, 38.5, 31.2, ...]
//         },
//         ...
//       ]
//     }
//
// Build deps (header-only):
//   lib/httplib/httplib.h  (already pulled in by posetail_server_client.h)
//   src/json.hpp           (nlohmann::json, already in red)

#include "httplib.h"
#include "json.hpp"
#include "miniz.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>


struct ProofreadSession {
    std::string animal;
    std::string session;
    std::string date;
    std::string recording_path;
    std::string prediction_path;
    int n_frames_total = 0;
    int n_frames_bad   = 0;
    std::vector<int>   frames;        // frame indices to relabel
    std::vector<float> residuals_mm;  // parallel to `frames`
};


struct ProofreadState {
    // User-editable URL like "http://10.102.10.138:8000" — the lab Linux
    // box that runs mouse_dashboard's uvicorn server. NOTE: this is a
    // *different* host from the posetail server (which lives at .88) —
    // don't copy that default; it'll return 404 from a posetail process.
    std::string url = "http://10.102.10.138:8000";
    // Residual cutoff (mm). Default matches the dashboard default.
    float residual_threshold_mm = 25.0f;
    // Cluster debounce window in frames. Defaults to 50, which collapses
    // runs of adjacent bad frames into one worst-frame pick — usually what
    // you want for relabeling.
    int   min_gap = 50;

    bool reachable = false;
    std::string status;             // last-action message for the UI
    std::vector<ProofreadSession> sessions;
    int n_bad_total = 0;
    std::string generated_at;       // ISO-ish timestamp from the server
    float last_ms = 0.0f;           // round-trip time
};


namespace proofread_client_detail {

// Split "http://host:port" into the base URL httplib::Client wants.
inline std::string normalize_url(const std::string &raw) {
    std::string s = raw;
    while (!s.empty() && (s.back() == '/' || s.back() == ' ')) s.pop_back();
    if (s.rfind("http://", 0) == std::string::npos &&
        s.rfind("https://", 0) == std::string::npos) {
        s = "http://" + s;
    }
    return s;
}

}  // namespace proofread_client_detail


// Refresh `s.sessions` from the server. Returns true on a successful
// response with valid JSON; on failure, sets `s.status` and leaves
// `s.sessions` untouched so the UI keeps showing the previous results.
inline bool proofread_fetch(ProofreadState &s) {
    s.reachable = false;
    if (s.url.empty()) {
        s.status = "Server URL is empty";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();

    const std::string base = proofread_client_detail::normalize_url(s.url);
    httplib::Client cli(base);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(30, 0);   // /api/bad_frames_all scans 50 sessions

    char path[256];
    std::snprintf(path, sizeof(path),
                   "/api/bad_frames_all?residual_threshold_mm=%.3f&min_gap=%d",
                   s.residual_threshold_mm, s.min_gap);
    auto res = cli.Get(path);
    if (!res) {
        s.status = "Cannot reach server (" + base + "): " +
                   httplib::to_string(res.error());
        return false;
    }
    if (res->status != 200) {
        s.status = "GET /api/bad_frames_all failed: HTTP " +
                   std::to_string(res->status);
        return false;
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(res->body);
    } catch (const std::exception &e) {
        s.status = std::string("Bad JSON from server: ") + e.what();
        return false;
    }

    std::vector<ProofreadSession> out;
    out.reserve(j.value("n_sessions", 0));
    for (const auto &item : j.value("sessions", nlohmann::json::array())) {
        ProofreadSession ps;
        ps.animal          = item.value("animal", std::string{});
        ps.session         = item.value("session", std::string{});
        ps.date            = item.value("date", std::string{});
        ps.recording_path  = item.value("recording_path", std::string{});
        ps.prediction_path = item.value("prediction_path", std::string{});
        ps.n_frames_total  = item.value("n_frames_total", 0);
        ps.n_frames_bad    = item.value("n_frames_bad", 0);
        if (item.contains("frames")) {
            ps.frames = item["frames"].get<std::vector<int>>();
        }
        if (item.contains("residuals_mm")) {
            ps.residuals_mm = item["residuals_mm"].get<std::vector<float>>();
        }
        out.push_back(std::move(ps));
    }
    s.sessions     = std::move(out);
    s.n_bad_total  = j.value("n_bad_total", 0);
    s.generated_at = j.value("generated_at", std::string{});
    s.reachable    = true;
    s.status       = "OK: " + std::to_string(s.sessions.size()) +
                     " sessions, " + std::to_string(s.n_bad_total) +
                     " bad frames";

    const auto t1 = std::chrono::steady_clock::now();
    s.last_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    return true;
}


// Download the OpenCV-format Cam*.yaml zip for one session and extract it
// into `dest_dir` (created if missing). Returns true on success.
// `error_out` is filled with a human-readable message on failure.
//
// Useful before red's load_videos: the server stores the calibration that
// JARVIS used for this session at /home/user/red_data/calib/<date>/calibration/,
// and we want pm.calibration_folder pointing at a local mirror so the
// rest of red's project plumbing has its yamls.
inline bool proofread_fetch_calib(const ProofreadState &s,
                                   const std::string &animal,
                                   const std::string &session,
                                   const std::filesystem::path &dest_dir,
                                   std::string *error_out = nullptr) {
    auto set_err = [&](const std::string &m) {
        if (error_out) *error_out = m;
        return false;
    };
    if (s.url.empty())     return set_err("Server URL is empty");
    if (animal.empty())    return set_err("Animal is empty");
    if (session.empty())   return set_err("Session is empty");

    const std::string base = proofread_client_detail::normalize_url(s.url);
    httplib::Client cli(base);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(15, 0);

    char path[512];
    std::snprintf(path, sizeof(path),
                   "/api/session_calib_zip?animal=%s&session=%s",
                   animal.c_str(), session.c_str());
    auto res = cli.Get(path);
    if (!res) {
        return set_err(std::string("Cannot reach server: ") +
                        httplib::to_string(res.error()));
    }
    if (res->status != 200) {
        return set_err("HTTP " + std::to_string(res->status) +
                        " from " + path);
    }

    std::error_code ec;
    std::filesystem::create_directories(dest_dir, ec);
    if (ec) return set_err("Cannot create dest dir " + dest_dir.string() +
                            ": " + ec.message());

    mz_zip_archive zip;
    mz_zip_zero_struct(&zip);
    if (!mz_zip_reader_init_mem(&zip, res->body.data(),
                                 res->body.size(), 0)) {
        return set_err("Bad zip from server");
    }
    const mz_uint n = mz_zip_reader_get_num_files(&zip);
    for (mz_uint i = 0; i < n; ++i) {
        char fname[256];
        if (!mz_zip_reader_get_filename(&zip, i, fname, sizeof(fname))) {
            mz_zip_reader_end(&zip);
            return set_err("Bad filename entry in zip");
        }
        // Reject path-traversal entries: only flat Cam*.yaml allowed.
        if (std::string(fname).find("..") != std::string::npos ||
            std::string(fname).find('/')  != std::string::npos) {
            continue;
        }
        const auto out_path = dest_dir / fname;
        if (!mz_zip_reader_extract_to_file(&zip, i,
                                            out_path.string().c_str(), 0)) {
            mz_zip_reader_end(&zip);
            return set_err("Failed to extract " + std::string(fname));
        }
    }
    mz_zip_reader_end(&zip);
    return true;
}
