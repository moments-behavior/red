#pragma once
// collab_io.h -- crash-safe file primitives for the collaboration layer.
//
// Nothing in RED writes files atomically: save_project_manager_json
// (project.h), save_user_settings, and every CSV writer truncate in place. A
// background sync thread writing while the user saves would corrupt both. The
// discipline here is the one PredictionWriter already follows -- never let a
// partial write be observable as a complete file.
//
// write_file_atomic() writes to a sibling temp file, flushes it to stable
// storage, then renames over the target. Rename is atomic within a filesystem,
// so a reader sees either the whole old file or the whole new one, never a
// truncated middle. The temp file is a *sibling* on purpose: renaming across
// filesystems is not atomic and would silently degrade to a copy.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "collab_hash.h"

namespace collab {

namespace fs = std::filesystem;

// =========================================================================
// Randomness and time
// =========================================================================

// Cryptographically-seeded random bytes. Used for peer ids, auth nonces, and
// temp-file suffixes. std::random_device is the only entropy source available
// without adding a dependency; on the platforms RED targets it is backed by
// the OS CSPRNG. Deliberately NOT the fixed-seed mt19937 used elsewhere in the
// tree for reproducible sampling -- a predictable auth nonce defeats the
// challenge-response entirely.
inline std::vector<uint8_t> random_bytes(size_t n) {
    static thread_local std::random_device rd;
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<uint8_t>(rd() & 0xff);
    return out;
}

inline std::string random_hex(size_t n_bytes) {
    const std::vector<uint8_t> b = random_bytes(n_bytes);
    return to_hex(b.data(), b.size());
}

// Wall-clock milliseconds since the Unix epoch. Display and presence only --
// never an ordering authority. Peer clocks disagree, and the pump-events work
// already documents that this rig's clocks drift by seconds. Ordering is the
// Lamport clock in collab_ops.h.
inline int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
        .count();
}

// =========================================================================
// Directories
// =========================================================================

inline bool ensure_dir(const fs::path &dir, std::string *err = nullptr) {
    std::error_code ec;
    if (fs::exists(dir, ec)) {
        if (fs::is_directory(dir, ec)) return true;
        if (err) *err = dir.string() + " exists and is not a directory";
        return false;
    }
    fs::create_directories(dir, ec);
    if (ec) {
        if (err) *err = "cannot create " + dir.string() + ": " + ec.message();
        return false;
    }
    return true;
}

// =========================================================================
// Reading
// =========================================================================

inline bool read_file(const fs::path &path, std::string &out,
                      std::string *err = nullptr) {
    std::FILE *f = std::fopen(path.string().c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open " + path.string();
        return false;
    }
    out.clear();
    char buf[65536];
    for (;;) {
        const size_t n = std::fread(buf, 1, sizeof(buf), f);
        if (n > 0) out.append(buf, n);
        if (n < sizeof(buf)) {
            if (std::ferror(f)) {
                if (err) *err = "read error on " + path.string();
                std::fclose(f);
                return false;
            }
            break;
        }
    }
    std::fclose(f);
    return true;
}

// =========================================================================
// Atomic write
// =========================================================================

// Forces the file's contents to stable storage. Without this the rename can
// land before the data does, so a crash leaves a correctly-named empty file --
// which is worse than a partial one, because it looks valid.
inline bool flush_to_disk(std::FILE *f) {
    if (std::fflush(f) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return ::fsync(fileno(f)) == 0;
#endif
}

inline bool write_file_atomic(const fs::path &path, const void *data,
                              size_t len, std::string *err = nullptr) {
    const fs::path dir = path.parent_path();
    if (!dir.empty() && !ensure_dir(dir, err)) return false;

    // Sibling temp file: same directory means the rename stays within one
    // filesystem and therefore stays atomic.
    const fs::path tmp =
        path.string() + ".tmp-" + random_hex(8);

    std::FILE *f = std::fopen(tmp.string().c_str(), "wb");
    if (!f) {
        if (err) *err = "cannot create temp file " + tmp.string();
        return false;
    }

    bool ok = true;
    if (len > 0 && std::fwrite(data, 1, len, f) != len) {
        if (err) *err = "short write to " + tmp.string();
        ok = false;
    }
    if (ok && !flush_to_disk(f)) {
        if (err) *err = "cannot flush " + tmp.string();
        ok = false;
    }
    std::fclose(f);

    if (!ok) {
        std::error_code ec;
        fs::remove(tmp, ec);
        return false;
    }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        if (err)
            *err = "cannot rename " + tmp.string() + " -> " + path.string() +
                   ": " + ec.message();
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

inline bool write_file_atomic(const fs::path &path, const std::string &data,
                              std::string *err = nullptr) {
    return write_file_atomic(path, data.data(), data.size(), err);
}

// =========================================================================
// Append
// =========================================================================

// Appends one newline-terminated record to a log, flushing before returning.
//
// The op log is append-only and never rewritten, so it does not need the
// temp-and-rename dance -- but it DOES need the flush, otherwise ops sit in
// stdio's buffer and a crash loses edits the UI already showed as recorded.
// A torn final line is tolerable and expected: the reader in collab_log.h
// drops a trailing partial line rather than failing the whole log.
inline bool append_line(const fs::path &path, const std::string &line,
                        std::string *err = nullptr) {
    const fs::path dir = path.parent_path();
    if (!dir.empty() && !ensure_dir(dir, err)) return false;

    std::FILE *f = std::fopen(path.string().c_str(), "ab");
    if (!f) {
        if (err) *err = "cannot open for append: " + path.string();
        return false;
    }
    bool ok = true;
    if (!line.empty() && std::fwrite(line.data(), 1, line.size(), f) != line.size())
        ok = false;
    if (ok && std::fputc('\n', f) == EOF) ok = false;
    if (ok && !flush_to_disk(f)) ok = false;
    std::fclose(f);
    if (!ok && err) *err = "append failed on " + path.string();
    return ok;
}

// =========================================================================
// Size / existence helpers
// =========================================================================

inline bool file_exists(const fs::path &path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec);
}

inline uint64_t size_of_file(const fs::path &path) {
    std::error_code ec;
    const auto sz = fs::file_size(path, ec);
    if (ec) return 0;
    return static_cast<uint64_t>(sz);
}

}  // namespace collab
