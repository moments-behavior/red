#pragma once

// traj_reader.h — Memory-mapped reader for sparse prediction binary format
//
// Adapted from Green's TrajReader for RED's bout inspector. Key difference:
// predictions are SPARSE — most video frames have no predictions. The binary
// stores only frames with data, plus a sorted frame-number index for O(log n)
// lookup via binary search.
//
// File format (v3, sparse):
//   Header (32 bytes):
//     magic (4B) = 0x024E5247
//     version (4B) = 3
//     total_video_frames (4B)    — length of the full video
//     n_stored_frames (4B)       — number of frames with data
//     header_size (4B)           — byte offset where data starts (page-aligned)
//     fps (4B)
//     num_keypoints (2B)
//     elements_per_frame (2B)    — e.g., 200 for 50 kp × 4 (x,y,z,conf)
//   Frame index (n_stored × 8 bytes):
//     sorted array of (frame_number: u32, data_index: u32) pairs
//   Data section (page-aligned):
//     contiguous float32 arrays, one per stored frame
//     Each frame: elements_per_frame × 4 bytes
//
// Usage:
//   PredictionReader preds("fly_predictions.bin");
//   int n = preds.total_frames();        // 1,714,786
//   int stored = preds.stored_frames();  // 568,073
//   const float* pose = preds.frame(42000);  // binary search, returns nullptr if not stored
//   if (pose) { float x0 = pose[0], y0 = pose[1], z0 = pose[2], conf0 = pose[3]; }
//
// Thread safety: read-only after construction.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct PredictionReader {
    PredictionReader() = default;

    explicit PredictionReader(const std::string &path) { open(path); }

    ~PredictionReader() { close(); }

    // Non-copyable, movable
    PredictionReader(const PredictionReader &) = delete;
    PredictionReader &operator=(const PredictionReader &) = delete;
    PredictionReader(PredictionReader &&o) noexcept
        : data_(o.data_), file_size_(o.file_size_),
          total_video_frames_(o.total_video_frames_),
          n_stored_(o.n_stored_), fps_(o.fps_),
          num_keypoints_(o.num_keypoints_), epf_(o.epf_),
          index_(o.index_), data_section_(o.data_section_) {
        o.data_ = nullptr;
        o.file_size_ = 0;
    }
    PredictionReader &operator=(PredictionReader &&o) noexcept {
        if (this != &o) {
            close();
            data_ = o.data_;  file_size_ = o.file_size_;
            total_video_frames_ = o.total_video_frames_;
            n_stored_ = o.n_stored_;  fps_ = o.fps_;
            num_keypoints_ = o.num_keypoints_;  epf_ = o.epf_;
            index_ = o.index_;  data_section_ = o.data_section_;
            o.data_ = nullptr;  o.file_size_ = 0;
        }
        return *this;
    }

    void open(const std::string &path) {
        close();
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return;
        struct stat st;
        if (fstat(fd, &st) < 0) { ::close(fd); return; }
        file_size_ = static_cast<size_t>(st.st_size);
        data_ = static_cast<const uint8_t *>(
            mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
        if (data_ == MAP_FAILED) { data_ = nullptr; return; }
        madvise(const_cast<uint8_t *>(data_), file_size_, MADV_RANDOM);
        parse_header();
    }

    void close() {
        if (data_) {
            munmap(const_cast<uint8_t *>(data_), file_size_);
            data_ = nullptr;
            file_size_ = 0;
        }
    }

    bool is_open() const { return data_ != nullptr; }

    uint32_t total_frames() const { return total_video_frames_; }
    uint32_t stored_frames() const { return n_stored_; }
    uint32_t fps() const { return fps_; }
    uint16_t num_keypoints() const { return num_keypoints_; }
    uint16_t elements_per_frame() const { return epf_; }

    // Look up a frame by absolute video frame number.
    // Returns pointer to elements_per_frame floats, or nullptr if frame has no data.
    // O(log n) via binary search on the sorted frame index.
    const float *frame(uint32_t frame_num) const {
        if (!data_ || !index_ || frame_num >= total_video_frames_)
            return nullptr;
        // Binary search the index
        uint32_t lo = 0, hi = n_stored_;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t fn = index_[mid * 2];
            if (fn < frame_num)
                lo = mid + 1;
            else if (fn > frame_num)
                hi = mid;
            else {
                uint32_t data_idx = index_[mid * 2 + 1];
                size_t byte_off = static_cast<size_t>(data_idx) * epf_ * 4;
                size_t byte_end = byte_off + static_cast<size_t>(epf_) * 4;
                if (data_section_ + byte_end > data_ + file_size_)
                    return nullptr;  // corrupt index — out of bounds
                return reinterpret_cast<const float *>(data_section_ + byte_off);
            }
        }
        return nullptr;
    }

    // Check if a frame has stored data (without returning the pointer).
    bool has_frame(uint32_t frame_num) const {
        return frame(frame_num) != nullptr;
    }

    // Prefetch pages for a frame range (call when selecting a bout).
    void prefetch_range(uint32_t start_frame, uint32_t end_frame) const {
        if (!data_ || !index_) return;
        // Find first and last stored frames in range
        // Binary search for start
        uint32_t lo = 0, hi = n_stored_;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (index_[mid * 2] < start_frame) lo = mid + 1;
            else hi = mid;
        }
        uint32_t first = lo;
        // Binary search for end
        hi = n_stored_;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (index_[mid * 2] <= end_frame) lo = mid + 1;
            else hi = mid;
        }
        uint32_t last = lo;  // one past the last
        if (first >= last) return;

        uint32_t first_idx = index_[first * 2 + 1];
        uint32_t last_idx = index_[(last - 1) * 2 + 1];
        const uint8_t *start = data_section_ + static_cast<size_t>(first_idx) * epf_ * 4;
        const uint8_t *end_ptr = data_section_ + static_cast<size_t>(last_idx + 1) * epf_ * 4;
        // Clamp to file bounds
        const uint8_t *file_end = data_ + file_size_;
        if (end_ptr > file_end) end_ptr = file_end;
        if (start >= end_ptr) return;
        size_t len = static_cast<size_t>(end_ptr - start);
        len = (len + 4095) & ~size_t(4095);
        madvise(const_cast<uint8_t *>(start), len, MADV_WILLNEED);
    }

private:
    static constexpr uint32_t MAGIC = 0x024E5247;
    static constexpr uint32_t VERSION = 3;

    void parse_header() {
        if (file_size_ < 32) { close(); return; }
        uint32_t magic, version;
        std::memcpy(&magic, data_, 4);
        std::memcpy(&version, data_ + 4, 4);
        if (magic != MAGIC || version != VERSION) { close(); return; }

        std::memcpy(&total_video_frames_, data_ + 8, 4);
        std::memcpy(&n_stored_, data_ + 12, 4);
        uint32_t header_size;
        std::memcpy(&header_size, data_ + 16, 4);
        std::memcpy(&fps_, data_ + 20, 4);
        std::memcpy(&num_keypoints_, data_ + 24, 2);
        std::memcpy(&epf_, data_ + 26, 2);

        // Frame index starts at byte 32
        size_t index_bytes = static_cast<size_t>(n_stored_) * 8;
        if (32 + index_bytes > file_size_) { close(); return; }
        index_ = reinterpret_cast<const uint32_t *>(data_ + 32);

        // Data section
        if (header_size > file_size_) { close(); return; }
        data_section_ = data_ + header_size;
    }

    const uint8_t *data_ = nullptr;
    size_t file_size_ = 0;
    uint32_t total_video_frames_ = 0;
    uint32_t n_stored_ = 0;
    uint32_t fps_ = 0;
    uint16_t num_keypoints_ = 0;
    uint16_t epf_ = 0;  // elements per frame
    const uint32_t *index_ = nullptr;      // [n_stored * 2]: (frame_num, data_idx) pairs
    const uint8_t *data_section_ = nullptr; // start of float data
};
