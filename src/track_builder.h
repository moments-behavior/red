#pragma once
// track_builder.h — Build multi-view tracks from pairwise feature matches.
// Reads pairwise_matches.json (from Python SuperPoint+LightGlue pipeline)
// and produces the landmarks map consumed by FeatureRefinement.
//
// Header-only. Uses union-find to merge pairwise matches into multi-view tracks.

#include "json.hpp"
#include <Eigen/Core>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace TrackBuilder {
namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct PairwiseMatch {
    std::string cam_a, cam_b;
    std::vector<Eigen::Vector2d> pts_a, pts_b;
    std::vector<float> scores;
};

struct TrackResult {
    // landmarks map: camera_name -> {track_id -> pixel_coords}
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
    int num_tracks = 0;
    int num_observations = 0;
    std::string error;
};

// ─────────────────────────────────────────────────────────────────────────────
// Union-Find
// ─────────────────────────────────────────────────────────────────────────────

class UnionFind {
public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        for (int i = 0; i < n; i++) parent_[i] = i;
    }

    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]]; // path halving
            x = parent_[x];
        }
        return x;
    }

    void unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;
        if (rank_[rx] < rank_[ry]) std::swap(rx, ry);
        parent_[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Keypoint identity hash
// ─────────────────────────────────────────────────────────────────────────────

// Same SuperPoint keypoint in different pairs has identical float coords
// (same extraction run). Quantize to 0.01 px for robust hashing.
inline uint64_t keypoint_hash(double x, double y) {
    int32_t ix = static_cast<int32_t>(std::round(x * 100.0));
    int32_t iy = static_cast<int32_t>(std::round(y * 100.0));
    return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(iy));
}

// ─────────────────────────────────────────────────────────────────────────────
// Load pairwise matches from JSON
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<PairwiseMatch> load_pairwise_matches(const std::string &path, std::string *error = nullptr) {
    std::vector<PairwiseMatch> matches;

    std::ifstream f(path);
    if (!f.is_open()) {
        if (error) *error = "Cannot open " + path;
        return matches;
    }

    try {
        nlohmann::json j;
        f >> j;
        const auto &pairs = j["pairs"];
        for (auto &[key, val] : pairs.items()) {
            // Key format: "serial_a-serial_b"
            auto dash = key.find('-');
            if (dash == std::string::npos) continue;

            PairwiseMatch m;
            m.cam_a = key.substr(0, dash);
            m.cam_b = key.substr(dash + 1);

            const auto &pts_a = val["pts_a"];
            const auto &pts_b = val["pts_b"];
            int n = (int)pts_a.size();
            m.pts_a.resize(n);
            m.pts_b.resize(n);
            for (int i = 0; i < n; i++) {
                m.pts_a[i] = Eigen::Vector2d(pts_a[i][0].get<double>(), pts_a[i][1].get<double>());
                m.pts_b[i] = Eigen::Vector2d(pts_b[i][0].get<double>(), pts_b[i][1].get<double>());
            }

            if (val.contains("scores")) {
                const auto &sc = val["scores"];
                m.scores.resize(n);
                for (int i = 0; i < n; i++)
                    m.scores[i] = sc[i].get<float>();
            }

            matches.push_back(std::move(m));
        }
    } catch (const std::exception &e) {
        if (error) *error = std::string("Error parsing pairwise_matches: ") + e.what();
        return {};
    }

    return matches;
}

// Load all pairwise_matches*.json files from a directory and merge
inline std::vector<PairwiseMatch> load_all_pairwise_matches(const std::string &dir, std::string *error = nullptr) {
    std::vector<PairwiseMatch> all;

    // Look for the single merged file first
    std::string single = dir + "/pairwise_matches.json";
    if (fs::exists(single))
        return load_pairwise_matches(single, error);

    // Otherwise look for per-set files
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.path().filename().string().find("pairwise_matches") == 0 &&
            entry.path().extension() == ".json") {
            auto matches = load_pairwise_matches(entry.path().string(), error);
            if (error && !error->empty()) return {};
            all.insert(all.end(), std::make_move_iterator(matches.begin()),
                        std::make_move_iterator(matches.end()));
        }
    }
    return all;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build tracks from pairwise matches
// ─────────────────────────────────────────────────────────────────────────────

inline TrackResult build_tracks(const std::vector<PairwiseMatch> &matches) {
    TrackResult result;

    // Phase 1: assign global IDs to each unique (camera, keypoint) pair
    // Key: camera serial -> keypoint_hash -> global_id
    std::unordered_map<std::string, std::unordered_map<uint64_t, int>> cam_kpt_ids;
    // Also store the actual coordinates for each global ID
    std::vector<std::pair<std::string, Eigen::Vector2d>> id_to_obs; // global_id -> (cam, pixel)

    auto get_or_assign = [&](const std::string &cam, const Eigen::Vector2d &pt) -> int {
        uint64_t h = keypoint_hash(pt.x(), pt.y());
        auto &cam_map = cam_kpt_ids[cam];
        auto it = cam_map.find(h);
        if (it != cam_map.end()) return it->second;
        int id = (int)id_to_obs.size();
        cam_map[h] = id;
        id_to_obs.push_back({cam, pt});
        return id;
    };

    // Assign all keypoints
    for (const auto &m : matches) {
        for (int i = 0; i < (int)m.pts_a.size(); i++) {
            get_or_assign(m.cam_a, m.pts_a[i]);
            get_or_assign(m.cam_b, m.pts_b[i]);
        }
    }

    int total_kpts = (int)id_to_obs.size();
    if (total_kpts == 0) {
        result.error = "No keypoints found in pairwise matches";
        return result;
    }

    // Phase 2: union matched keypoints
    UnionFind uf(total_kpts);
    for (const auto &m : matches) {
        for (int i = 0; i < (int)m.pts_a.size(); i++) {
            int ga = cam_kpt_ids[m.cam_a][keypoint_hash(m.pts_a[i].x(), m.pts_a[i].y())];
            int gb = cam_kpt_ids[m.cam_b][keypoint_hash(m.pts_b[i].x(), m.pts_b[i].y())];
            uf.unite(ga, gb);
        }
    }

    // Phase 3: collect connected components
    std::unordered_map<int, std::vector<int>> components; // root -> [global_ids]
    for (int i = 0; i < total_kpts; i++)
        components[uf.find(i)].push_back(i);

    // Phase 4: build landmarks map from components with 2+ cameras
    int track_id = 0;
    for (const auto &[root, members] : components) {
        // Check distinct camera count
        std::unordered_map<std::string, int> cam_to_member; // cam -> first member index
        for (int gid : members) {
            const auto &[cam, pt] = id_to_obs[gid];
            if (cam_to_member.find(cam) == cam_to_member.end())
                cam_to_member[cam] = gid;
        }

        if ((int)cam_to_member.size() < 2) continue;

        // One observation per camera (pick first seen)
        for (const auto &[cam, gid] : cam_to_member) {
            result.landmarks[cam][track_id] = id_to_obs[gid].second;
            result.num_observations++;
        }
        track_id++;
    }

    result.num_tracks = track_id;

    fprintf(stderr, "[TrackBuilder] %d keypoints -> %d tracks (%d observations) from %d pairs\n",
            total_kpts, result.num_tracks, result.num_observations, (int)matches.size());

    return result;
}

} // namespace TrackBuilder
