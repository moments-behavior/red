#pragma once
// prediction_store.h — on-disk, memory-mapped store for JARVIS batch predictions,
// kept SEPARATE from the manual-label AnnotationMap so predicting a whole video
// never floods the Labeling Tool.
//
// Only 3D + confidence is stored (the video overlay reprojects 3D->2D on the
// fly), so a frame is 4*num_keypoints float32 = 800 B for 50 keypoints. The
// store is SPARSE: frames where the subject is out of view / prediction failed
// are simply absent (frame() returns nullptr), which is the common case for the
// bout workflow. mmap gives out-of-core access — a 1 hr @ 800 fps video (~2.15 GB
// dense) pages in only the frames currently on screen.
//
// File format (little-endian; native on x86/arm64):
//   Header (page 0, first 40 bytes used):
//     0  u32 magic            = 0x52505231 ('RPR1')
//     4  u32 version          = 1
//     8  u32 total_video_frames
//     12 u32 n_stored         — frames actually present
//     16 u32 fps
//     20 u16 num_keypoints
//     22 u16 elements_per_frame  (= 4 * num_keypoints)
//     24 u64 data_offset      (= 4096, page-aligned)
//     32 u64 index_offset     — byte offset of the frame index (footer)
//   Data section (offset 4096):
//     n_stored contiguous float32[elements_per_frame] blocks, in frame order
//   Index section (offset index_offset):
//     n_stored * (u32 frame_number, u32 data_index), sorted ascending by frame
//
// The writer streams data blocks straight to disk as each frame is predicted,
// holding only the (tiny) index in RAM; the index is written as a footer on
// finalize(). The reader mmaps read-only and does an O(log n) binary search.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace predstore {

static constexpr uint32_t kMagic = 0x52505231u;  // 'RPR1'
static constexpr uint32_t kVersion = 1u;
static constexpr uint64_t kDataOffset = 4096u;   // page-aligned data section

// ── Streaming writer ─────────────────────────────────────────────────────────
// Usage:
//   PredictionWriter w;
//   w.open(path, num_keypoints);
//   for each predicted frame: w.add_frame(frame_num, floats /* 4*num_kp */);
//   w.finalize(total_video_frames, fps);   // writes footer index + header
struct PredictionWriter {
    PredictionWriter() = default;
    ~PredictionWriter() { if (fp_) std::fclose(fp_); }

    PredictionWriter(const PredictionWriter &) = delete;
    PredictionWriter &operator=(const PredictionWriter &) = delete;

    bool open(const std::string &path, int num_keypoints) {
        close();
        fp_ = std::fopen(path.c_str(), "wb+");
        if (!fp_) return false;
        path_ = path;
        num_kp_ = (uint16_t)num_keypoints;
        epf_ = (uint16_t)(num_keypoints * 4);
        // Reserve the header page; data starts at kDataOffset.
        std::vector<uint8_t> zero(kDataOffset, 0);
        if (std::fwrite(zero.data(), 1, zero.size(), fp_) != zero.size()) {
            close(); return false;
        }
        index_.clear();
        finalized_ = false;
        return true;
    }

    bool is_open() const { return fp_ != nullptr && !finalized_; }
    uint32_t stored_frames() const { return (uint32_t)index_.size(); }

    // Append one frame's elements_per_frame float32 values. `values` must point
    // to exactly 4*num_keypoints floats (x,y,z,conf per keypoint). Frames must
    // be added in strictly increasing frame_number order (the batch predictor
    // already does this), which keeps the index sorted for the reader.
    bool add_frame(uint32_t frame_number, const float *values) {
        if (!is_open()) return false;
        uint32_t data_index = (uint32_t)index_.size();
        if (std::fwrite(values, sizeof(float), epf_, fp_) != (size_t)epf_)
            return false;
        index_.push_back({frame_number, data_index});
        return true;
    }

    // Write the footer index and back-patch the header. Safe to call once.
    bool finalize(uint32_t total_video_frames, uint32_t fps) {
        if (!fp_ || finalized_) return false;
        // Current position is the end of the data section = index offset.
        long ix = std::ftell(fp_);
        if (ix < 0) return false;
        uint64_t index_offset = (uint64_t)ix;
        // Footer: n_stored * (u32 frame, u32 data_index), already sorted.
        if (!index_.empty() &&
            std::fwrite(index_.data(), sizeof(IndexEntry), index_.size(), fp_)
                != index_.size())
            return false;

        // Back-patch the header.
        uint8_t hdr[40];
        std::memset(hdr, 0, sizeof(hdr));
        uint32_t n_stored = (uint32_t)index_.size();
        std::memcpy(hdr + 0,  &kMagic, 4);
        std::memcpy(hdr + 4,  &kVersion, 4);
        std::memcpy(hdr + 8,  &total_video_frames, 4);
        std::memcpy(hdr + 12, &n_stored, 4);
        std::memcpy(hdr + 16, &fps, 4);
        std::memcpy(hdr + 20, &num_kp_, 2);
        std::memcpy(hdr + 22, &epf_, 2);
        uint64_t data_off = kDataOffset;
        std::memcpy(hdr + 24, &data_off, 8);
        std::memcpy(hdr + 32, &index_offset, 8);
        if (std::fseek(fp_, 0, SEEK_SET) != 0) return false;
        if (std::fwrite(hdr, 1, sizeof(hdr), fp_) != sizeof(hdr)) return false;
        std::fflush(fp_);
        std::fclose(fp_);
        fp_ = nullptr;
        finalized_ = true;
        return true;
    }

    void close() {
        if (fp_) { std::fclose(fp_); fp_ = nullptr; }
        index_.clear();
        finalized_ = false;
    }

  private:
    struct IndexEntry { uint32_t frame; uint32_t data_index; };
    std::FILE *fp_ = nullptr;
    std::string path_;
    uint16_t num_kp_ = 0;
    uint16_t epf_ = 0;
    bool finalized_ = false;
    std::vector<IndexEntry> index_;
};

// ── Memory-mapped reader ─────────────────────────────────────────────────────
struct PredictionReader {
    PredictionReader() = default;
    explicit PredictionReader(const std::string &path) { open(path); }
    ~PredictionReader() { close(); }

    PredictionReader(const PredictionReader &) = delete;
    PredictionReader &operator=(const PredictionReader &) = delete;
    PredictionReader(PredictionReader &&o) noexcept { move_from(o); }
    PredictionReader &operator=(PredictionReader &&o) noexcept {
        if (this != &o) { close(); move_from(o); }
        return *this;
    }

    bool open(const std::string &path) {
        close();
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) < 0) { ::close(fd); return false; }
        file_size_ = (size_t)st.st_size;
        if (file_size_ < kDataOffset) { ::close(fd); file_size_ = 0; return false; }
        data_ = (const uint8_t *)mmap(nullptr, file_size_, PROT_READ,
                                      MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (data_ == MAP_FAILED) { data_ = nullptr; file_size_ = 0; return false; }
        madvise((void *)data_, file_size_, MADV_RANDOM);
        return parse_header();
    }

    void close() {
        if (data_) { munmap((void *)data_, file_size_); data_ = nullptr; }
        file_size_ = 0; index_ = nullptr; data_section_ = nullptr;
        total_video_frames_ = n_stored_ = fps_ = 0; num_kp_ = epf_ = 0;
    }

    bool is_open() const { return data_ != nullptr && index_ != nullptr; }
    uint32_t total_frames() const { return total_video_frames_; }
    uint32_t stored_frames() const { return n_stored_; }
    uint32_t fps() const { return fps_; }
    uint16_t num_keypoints() const { return num_kp_; }
    uint16_t elements_per_frame() const { return epf_; }

    // Returns a pointer to elements_per_frame floats (x,y,z,conf per keypoint)
    // for `frame_num`, or nullptr if that frame has no stored prediction.
    const float *frame(uint32_t frame_num) const {
        if (!is_open()) return nullptr;
        uint32_t lo = 0, hi = n_stored_;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t fn = index_[mid * 2];
            if (fn < frame_num) lo = mid + 1;
            else if (fn > frame_num) hi = mid;
            else {
                uint32_t di = index_[mid * 2 + 1];
                size_t off = data_offset_ + (size_t)di * epf_ * 4;
                if (off + (size_t)epf_ * 4 > file_size_) return nullptr;
                return (const float *)(data_ + off);
            }
        }
        return nullptr;
    }

    // Sequential access to the i-th stored frame (0..stored_frames-1), in
    // ascending frame order. Writes the absolute frame number to *frame_out.
    // For one-pass scans (e.g. building the Pose Stats time-series) this is
    // O(1) per frame vs. frame()'s O(log n) binary search.
    const float *stored_at(uint32_t i, uint32_t *frame_out) const {
        if (!is_open() || i >= n_stored_) return nullptr;
        if (frame_out) *frame_out = index_[i * 2];
        uint32_t di = index_[i * 2 + 1];
        size_t off = data_offset_ + (size_t)di * epf_ * 4;
        if (off + (size_t)epf_ * 4 > file_size_) return nullptr;
        return (const float *)(data_ + off);
    }

  private:
    void move_from(PredictionReader &o) {
        data_ = o.data_; file_size_ = o.file_size_;
        total_video_frames_ = o.total_video_frames_; n_stored_ = o.n_stored_;
        fps_ = o.fps_; num_kp_ = o.num_kp_; epf_ = o.epf_;
        data_offset_ = o.data_offset_; index_ = o.index_;
        data_section_ = o.data_section_;
        o.data_ = nullptr; o.file_size_ = 0; o.index_ = nullptr;
        o.data_section_ = nullptr;
    }

    bool parse_header() {
        uint32_t magic; std::memcpy(&magic, data_ + 0, 4);
        if (magic != kMagic) { close(); return false; }
        std::memcpy(&total_video_frames_, data_ + 8, 4);
        std::memcpy(&n_stored_, data_ + 12, 4);
        std::memcpy(&fps_, data_ + 16, 4);
        std::memcpy(&num_kp_, data_ + 20, 2);
        std::memcpy(&epf_, data_ + 22, 2);
        std::memcpy(&data_offset_, data_ + 24, 8);
        uint64_t index_offset; std::memcpy(&index_offset, data_ + 32, 8);
        // Bounds-check the index footer.
        if (index_offset + (size_t)n_stored_ * 8 > file_size_) { close(); return false; }
        index_ = (const uint32_t *)(data_ + index_offset);
        data_section_ = data_ + data_offset_;
        return true;
    }

    const uint8_t *data_ = nullptr;
    size_t file_size_ = 0;
    uint32_t total_video_frames_ = 0, n_stored_ = 0, fps_ = 0;
    uint16_t num_kp_ = 0, epf_ = 0;
    uint64_t data_offset_ = kDataOffset;
    const uint32_t *index_ = nullptr;      // (frame, data_index) pairs
    const uint8_t *data_section_ = nullptr;
};

// ── Lightweight header probe (for the "Saved Predictions" picker) ────────────
// Reads just the 40-byte header without mmapping the whole file.
struct StoreHeaderInfo {
    bool ok = false;
    uint32_t total_frames = 0, n_stored = 0, fps = 0;
    uint16_t num_keypoints = 0, elements_per_frame = 0;
};

inline StoreHeaderInfo read_store_header(const std::string &path) {
    StoreHeaderInfo h;
    std::FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) return h;
    uint8_t buf[40];
    size_t n = std::fread(buf, 1, sizeof(buf), fp);
    std::fclose(fp);
    if (n < sizeof(buf)) return h;
    uint32_t magic; std::memcpy(&magic, buf + 0, 4);
    if (magic != kMagic) return h;
    std::memcpy(&h.total_frames, buf + 8, 4);
    std::memcpy(&h.n_stored, buf + 12, 4);
    std::memcpy(&h.fps, buf + 16, 4);
    std::memcpy(&h.num_keypoints, buf + 20, 2);
    std::memcpy(&h.elements_per_frame, buf + 22, 2);
    h.ok = true;
    return h;
}

}  // namespace predstore
