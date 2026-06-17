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

#include <chrono>
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
    // User-editable URL like "http://10.102.10.88:8000" — same machine that
    // serves the JARVIS dashboard.
    std::string url = "http://10.102.10.88:8000";
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
