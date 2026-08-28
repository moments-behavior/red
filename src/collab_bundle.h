#pragma once
// collab_bundle.h -- project sharing: manifests, transfer planning, and
// resumable content-addressed blob transfer.
//
// Sharing a project means moving the .redproj, the skeleton, the calibration
// folder, the most recent labels, and (optionally) the media. Media is the
// hard part: a 65MP rig produces gigabytes per session, and pushing that
// through a single relay is slow and bandwidth-expensive.
//
// Three things make that bearable:
//   * CONTENT ADDRESSING. A file's id is its SHA-256, so a file already
//     present on the receiving machine transfers zero bytes. Seed the videos
//     once by USB or rsync and a later clone verifies and sends nothing.
//   * CHUNKING + RESUME. Transfers move in 4 MiB chunks into a .part file, so
//     an interrupted 40 GB download resumes instead of restarting.
//   * OPT-IN CATEGORIES. "Metadata and labels only" is a first-class choice;
//     media is selectable per file.
//
// Path rewriting is mandatory, not optional. ProjectManager stores absolute
// machine-local paths (project_path, media_folder, calibration_folder,
// skeleton_file). Shipping those verbatim would produce a project that points
// at directories which do not exist on the receiving machine -- and would also
// defeat content addressing, since the same project would hash differently on
// every machine. The manifest carries the .redproj as JSON with those fields
// relativized; the clone re-absolutizes them against the new root.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "collab_hash.h"
#include "collab_io.h"
#include "collab_ops.h"
#include "json.hpp"

namespace collab {

namespace fs = std::filesystem;

constexpr int kManifestVersion = 1;

// =========================================================================
// Categories
// =========================================================================

// What a file is, so the UI can offer "everything except media" without
// pattern-matching on paths.
namespace category {
inline const char *kProject     = "project";
inline const char *kSkeleton    = "skeleton";
inline const char *kCalibration = "calibration";
inline const char *kLabels      = "labels";
inline const char *kMedia       = "media";
}  // namespace category

inline bool is_media(const std::string &c) { return c == category::kMedia; }

// =========================================================================
// Manifest
// =========================================================================

struct BundleEntry {
    std::string rel_path;   // forward slashes, relative to the project root
    uint64_t    size = 0;
    std::string sha256;     // lowercase hex -- this IS the blob id
    std::string category;
};

inline void to_json(nlohmann::json &j, const BundleEntry &e) {
    j = nlohmann::json{{"rel_path", e.rel_path},
                       {"size", e.size},
                       {"sha256", e.sha256},
                       {"category", e.category}};
}

inline void from_json(const nlohmann::json &j, BundleEntry &e) {
    e.rel_path = j.value("rel_path", std::string{});
    e.size = j.value("size", (uint64_t)0);
    e.sha256 = j.value("sha256", std::string{});
    e.category = j.value("category", std::string{});
}

struct Manifest {
    int         version = kManifestVersion;
    std::string project_name;
    RoomBinding binding;
    std::string created_by;       // peer id
    std::string created_by_name;  // display name
    int64_t     created_ms = 0;

    // The .redproj contents with machine-local absolute paths stripped. Never
    // shipped as a blob -- see the header comment.
    nlohmann::json project_json;

    std::vector<BundleEntry> entries;

    uint64_t total_bytes() const {
        uint64_t n = 0;
        for (const auto &e : entries) n += e.size;
        return n;
    }

    uint64_t media_bytes() const {
        uint64_t n = 0;
        for (const auto &e : entries)
            if (is_media(e.category)) n += e.size;
        return n;
    }
};

inline void to_json(nlohmann::json &j, const Manifest &m) {
    nlohmann::json entries = nlohmann::json::array();
    for (const auto &e : m.entries) {
        nlohmann::json je;
        to_json(je, e);
        entries.push_back(je);
    }
    nlohmann::json jb;
    to_json(jb, m.binding);

    j = nlohmann::json{{"version", m.version},
                       {"project_name", m.project_name},
                       {"binding", jb},
                       {"created_by", m.created_by},
                       {"created_by_name", m.created_by_name},
                       {"created_ms", m.created_ms},
                       {"project_json", m.project_json},
                       {"entries", entries}};
}

inline bool manifest_from_json(const nlohmann::json &j, Manifest &m,
                               std::string *err = nullptr) {
    if (!j.is_object()) {
        if (err) *err = "manifest is not an object";
        return false;
    }
    m.version = j.value("version", 0);
    if (m.version != kManifestVersion) {
        if (err)
            *err = "manifest version " + std::to_string(m.version) +
                   " is not supported (this build speaks version " +
                   std::to_string(kManifestVersion) + ")";
        return false;
    }
    m.project_name = j.value("project_name", std::string{});
    if (j.contains("binding")) from_json(j.at("binding"), m.binding);
    m.created_by = j.value("created_by", std::string{});
    m.created_by_name = j.value("created_by_name", std::string{});
    m.created_ms = j.value("created_ms", (int64_t)0);
    m.project_json = j.value("project_json", nlohmann::json::object());

    m.entries.clear();
    if (j.contains("entries") && j["entries"].is_array()) {
        for (const auto &je : j["entries"]) {
            BundleEntry e;
            from_json(je, e);
            // A blob id that is not a full SHA-256 hex string would let a
            // hostile manifest steer the blob store's path.
            if (e.sha256.size() != 64 || e.rel_path.empty()) {
                if (err) *err = "manifest contains a malformed entry";
                return false;
            }
            m.entries.push_back(std::move(e));
        }
    }
    return true;
}

// =========================================================================
// Building a manifest
// =========================================================================

// One file to include, resolved by the caller. Kept as plain strings so this
// header does not need project.h (which pulls in the skeleton and camera
// model) and stays unit-testable on its own.
struct FileRef {
    std::string abs_path;
    std::string rel_path;
    std::string category;
};

// Normalizes to forward slashes so a manifest written on Windows resolves on
// Linux and vice versa.
inline std::string to_posix(const fs::path &p) {
    std::string s = p.generic_string();
    return s;
}

// Enumerates a directory tree into FileRefs, relative to `project_root`.
// Skips the .collab directory: sync bookkeeping is per-machine and must never
// be shipped to a peer.
inline void scan_dir(const fs::path &dir, const fs::path &project_root,
                     const std::string &cat, std::vector<FileRef> &out) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return;

    for (auto it = fs::recursive_directory_iterator(
             dir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        const fs::path &p = it->path();
        if (it->is_directory(ec)) {
            if (p.filename() == ".collab") it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;

        const fs::path rel = fs::relative(p, project_root, ec);
        if (ec || rel.empty()) continue;
        if (to_posix(rel).rfind(".collab/", 0) == 0) continue;

        FileRef r;
        r.abs_path = p.string();
        r.rel_path = to_posix(rel);
        r.category = cat;
        out.push_back(std::move(r));
    }
}

// Same, but places the tree under a synthetic prefix instead of computing a
// path relative to the project root.
//
// This is what makes a bundle self-contained. media_folder in particular is
// usually OUTSIDE the project directory, so a true relative path would be
// "../../../scratch/rig7/media" -- meaningless on the receiving machine and
// an escape from the destination root. Everything is re-homed under
// media/, calibration/, labeled_data/ instead, which is exactly the layout
// rewrite_project_paths() re-anchors the .redproj to.
inline void scan_dir_prefixed(const fs::path &dir, const std::string &prefix,
                              const std::string &cat,
                              std::vector<FileRef> &out) {
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return;

    for (auto it = fs::recursive_directory_iterator(
             dir, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        const fs::path &p = it->path();
        if (it->is_directory(ec)) {
            if (p.filename() == ".collab") it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;

        const fs::path rel = fs::relative(p, dir, ec);
        if (ec || rel.empty()) continue;

        FileRef r;
        r.abs_path = p.string();
        r.rel_path = prefix + "/" + to_posix(rel);
        r.category = cat;
        out.push_back(std::move(r));
    }
}

// Hashes every file into manifest entries. This is the slow step for media --
// hashing 40 GB takes minutes -- so it reports progress and honors a cancel
// flag. Runs on the sync thread, never on the GUI thread.
inline bool build_manifest(const std::vector<FileRef> &files, Manifest &out,
                           std::string *err = nullptr,
                           int *progress_done = nullptr,
                           const bool *cancel = nullptr) {
    out.entries.clear();
    out.entries.reserve(files.size());

    for (const FileRef &f : files) {
        if (cancel && *cancel) {
            if (err) *err = "cancelled";
            return false;
        }
        BundleEntry e;
        e.rel_path = f.rel_path;
        e.category = f.category;
        e.size = size_of_file(f.abs_path);
        if (!sha256_file_hex(f.abs_path, e.sha256, err)) return false;
        out.entries.push_back(std::move(e));
        if (progress_done) ++(*progress_done);
    }
    return true;
}

// =========================================================================
// Transfer planning
// =========================================================================

struct TransferPlan {
    std::vector<BundleEntry> already_present;  // byte-identical locally
    std::vector<BundleEntry> needed;
    uint64_t bytes_needed = 0;
    uint64_t bytes_skipped = 0;

    size_t file_count() const {
        return already_present.size() + needed.size();
    }
};

// Decides what actually has to move.
//
// A file counts as present only if the size matches AND the content hashes to
// the expected digest. Size alone is not enough -- a truncated or edited file
// of the same length would silently pass, and the whole point of content
// addressing is that "I already have this" is a verified claim.
//
// `include` lets the caller drop whole categories (the "metadata and labels
// only" choice) without rebuilding the manifest.
inline TransferPlan plan_transfer(
    const Manifest &m, const fs::path &dest_root,
    bool include_media = true) {
    TransferPlan plan;

    for (const BundleEntry &e : m.entries) {
        if (!include_media && is_media(e.category)) {
            plan.bytes_skipped += e.size;
            continue;
        }

        const fs::path local = dest_root / e.rel_path;
        bool present = false;
        if (file_exists(local) && size_of_file(local) == e.size) {
            std::string hex;
            if (sha256_file_hex(local.string(), hex, nullptr) &&
                hex == e.sha256)
                present = true;
        }

        if (present) {
            plan.already_present.push_back(e);
        } else {
            plan.needed.push_back(e);
            plan.bytes_needed += e.size;
        }
    }
    return plan;
}

// Human-readable byte count for the confirmation dialog.
inline std::string format_bytes(uint64_t n) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(n);
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char buf[64];
    if (u == 0) std::snprintf(buf, sizeof(buf), "%llu %s",
                              static_cast<unsigned long long>(n), units[u]);
    else std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return std::string(buf);
}

// =========================================================================
// Resumable blob transfer
// =========================================================================

// Partial downloads land here, named by content hash so a resumed transfer
// finds its own work regardless of which file in the project wanted it.
inline fs::path incoming_dir(const fs::path &project_root) {
    return project_root / ".collab" / "incoming";
}

inline fs::path part_path(const fs::path &project_root,
                          const std::string &sha256_hex) {
    return incoming_dir(project_root) / (sha256_hex + ".part");
}

// A blob id must be exactly 64 lowercase hex characters. Everything that turns
// a peer-supplied hash into a filesystem path goes through this first, so a
// malicious "../../etc/passwd" can never become a path.
inline bool valid_blob_id(const std::string &hex) {
    if (hex.size() != 64) return false;
    for (char c : hex) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) return false;
    }
    return true;
}

// How many bytes of this blob are already on disk. Zero means start fresh.
//
// The *_at variants take the staging directory directly so the relay can reuse
// this machinery for its own blob store; the project-root forms below are thin
// wrappers for the client side.
inline uint64_t resume_offset_at(const fs::path &staging,
                                 const std::string &sha256_hex) {
    if (!valid_blob_id(sha256_hex)) return 0;
    return size_of_file(staging / (sha256_hex + ".part"));
}

inline uint64_t resume_offset(const fs::path &project_root,
                              const std::string &sha256_hex) {
    return resume_offset_at(incoming_dir(project_root), sha256_hex);
}

// Appends one received chunk. Chunks always arrive in order for a given blob,
// so `offset` is a consistency check rather than a seek: a mismatch means the
// sender and receiver disagree about progress and the transfer must restart
// rather than silently produce a corrupt file.
inline bool append_chunk_at(const fs::path &staging,
                            const std::string &sha256_hex, uint64_t offset,
                            const void *data, size_t len,
                            std::string *err = nullptr) {
    if (!valid_blob_id(sha256_hex)) {
        if (err) *err = "malformed blob id";
        return false;
    }
    const fs::path p = staging / (sha256_hex + ".part");
    if (!ensure_dir(p.parent_path(), err)) return false;

    const uint64_t have = size_of_file(p);
    if (have != offset) {
        if (err)
            *err = "chunk offset " + std::to_string(offset) + " does not match "
                   "the " + std::to_string(have) + " bytes already received";
        return false;
    }

    std::FILE *f = std::fopen(p.string().c_str(), "ab");
    if (!f) {
        if (err) *err = "cannot open " + p.string();
        return false;
    }
    const bool ok = (len == 0) || (std::fwrite(data, 1, len, f) == len);
    if (ok) std::fflush(f);
    std::fclose(f);
    if (!ok && err) *err = "short write to " + p.string();
    return ok;
}

// Verifies the completed .part against its claimed digest and moves it into
// place. The hash check is the whole safety net for the transfer: a truncated
// or tampered blob dies here rather than becoming a silently corrupt video.
inline bool finalize_blob_at(const fs::path &staging,
                             const std::string &sha256_hex,
                             const fs::path &dest, std::string *err = nullptr) {
    if (!valid_blob_id(sha256_hex)) {
        if (err) *err = "malformed blob id";
        return false;
    }
    const fs::path p = staging / (sha256_hex + ".part");
    if (!file_exists(p)) {
        if (err) *err = "no received data for " + sha256_hex;
        return false;
    }

    std::string actual;
    if (!sha256_file_hex(p.string(), actual, err)) return false;
    if (actual != sha256_hex) {
        if (err)
            *err = "content hash mismatch: expected " + sha256_hex +
                   ", got " + actual + " -- discarding";
        std::error_code ec;
        fs::remove(p, ec);
        return false;
    }

    if (!dest.parent_path().empty() && !ensure_dir(dest.parent_path(), err))
        return false;

    std::error_code ec;
    fs::rename(p, dest, ec);
    if (ec) {
        // A cross-filesystem rename fails; fall back to copy-then-remove.
        fs::copy_file(p, dest, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            if (err) *err = "cannot place " + dest.string() + ": " + ec.message();
            return false;
        }
        fs::remove(p, ec);
    }
    return true;
}

inline bool append_chunk(const fs::path &project_root,
                         const std::string &sha256_hex, uint64_t offset,
                         const void *data, size_t len,
                         std::string *err = nullptr) {
    return append_chunk_at(incoming_dir(project_root), sha256_hex, offset, data,
                           len, err);
}

inline bool finalize_blob(const fs::path &project_root,
                          const std::string &sha256_hex, const fs::path &dest,
                          std::string *err = nullptr) {
    return finalize_blob_at(incoming_dir(project_root), sha256_hex, dest, err);
}

// =========================================================================
// Path rewriting on clone
// =========================================================================

// Rewrites the machine-local absolute paths in a shipped .redproj so it points
// at the receiving machine's copy. Anything not listed here is left alone, so
// project settings, JARVIS model lists, and pump config survive the trip.
//
// setup_project() rebuilds camera_params from calibration_folder on load, so
// shipping the calibration files is enough -- the derived parameters need no
// special handling.
inline nlohmann::json rewrite_project_paths(const nlohmann::json &project_json,
                                            const fs::path &new_root,
                                            const std::string &project_name) {
    nlohmann::json j = project_json;
    const std::string root = to_posix(new_root);

    j["project_path"] = root;
    j["project_root_path"] = to_posix(new_root.parent_path());
    if (!project_name.empty()) j["project_name"] = project_name;

    // These are stored as absolute paths but always live under the project
    // root in a shared bundle, so re-anchor whatever relative tail they had.
    auto reanchor = [&](const char *key, const char *fallback) {
        const std::string cur = j.value(key, std::string{});
        std::string tail = cur.empty() ? std::string(fallback)
                                       : to_posix(fs::path(cur).filename());
        if (cur.empty()) tail = fallback;
        j[key] = root + "/" + tail;
    };
    reanchor("calibration_folder", "calibration");
    reanchor("media_folder", "media");
    reanchor("keypoints_root_folder", "labeled_data");
    if (!j.value("skeleton_file", std::string{}).empty())
        reanchor("skeleton_file", "skeleton.json");

    return j;
}

}  // namespace collab
