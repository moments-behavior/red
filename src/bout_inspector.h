#pragma once

// bout_inspector.h — Bout Inspector data structures and DuckDB wrapper
//
// The Bout Inspector is a Green-inspired mode for inspecting dense 3D
// predictions organized into walking bouts. It uses:
//   1. DuckDB for bout metadata (start/end frame, speed, IK residual, status)
//   2. PredictionReader (traj_reader.h) for mmap'd sparse 3D predictions
//   3. RED's existing video/calibration/skeleton infrastructure
//
#include "duckdb.hpp"
#include <cmath>
#include <string>
#include <vector>

// ── Bout row (matches DuckDB schema) ──────────────────────────────────

struct BoutRow {
    int id = -1;
    int start_frame = 0;
    int end_frame = 0;
    int n_frames = 0;
    float duration_s = 0;
    float mean_speed = 0;
    float max_speed = 0;
    float mean_confidence = 0;
    float scut_z_mean = 0;
    float ik_mean_mm = NAN;
    int status = 0;       // 0=pending, 1=accepted, 2=rejected
    std::string notes;
};

enum BoutStatus { BoutPending = 0, BoutAccepted = 1, BoutRejected = 2 };

// ── Filter state ──────────────────────────────────────────────────────

struct BoutFilterState {
    int status_filter = 0;      // 0=all, 1=pending, 2=accepted, 3=rejected
    float min_duration_s = 0;
    float min_speed = 0;
    bool dirty = true;
};

// ── Inspector state ───────────────────────────────────────────────────

struct BoutInspectorState {
    bool active = false;
    int selected_bout_id = -1;
    int selected_row = -1;      // index into filtered_bouts
    std::vector<BoutRow> filtered_bouts;
    BoutFilterState filters;

    // Playback clamp
    bool clamp_playback = true;
    int clamp_start = 0;
    int clamp_end = 0;

    // Predictions file path
    std::string predictions_path;
    std::string db_path;
    float fps = 800.0f;  // video frame rate (for duration computation)
};

// ── DuckDB wrapper ────────────────────────────────────────────────────

class BoutDatabase {
public:
    BoutDatabase() = default;

    bool open(const std::string &path) {
        try {
            db_ = std::make_unique<duckdb::DuckDB>(path);
            con_ = std::make_unique<duckdb::Connection>(*db_);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool is_open() const { return con_ != nullptr; }

    void close() {
        con_.reset();
        db_.reset();
    }

    std::vector<BoutRow> query_bouts(const BoutFilterState &f) {
        std::vector<BoutRow> rows;
        if (!con_) return rows;

        std::string sql =
            "SELECT id, start_frame, end_frame, n_frames, duration_s, "
            "mean_speed, max_speed, mean_confidence, scut_z_mean, "
            "ik_mean_mm, status, notes FROM bouts";

        std::vector<std::string> where;
        if (f.status_filter == 1) where.push_back("status = 0");
        else if (f.status_filter == 2) where.push_back("status = 1");
        else if (f.status_filter == 3) where.push_back("status = 2");
        if (f.min_duration_s > 0)
            where.push_back("duration_s >= " + std::to_string(f.min_duration_s));
        if (f.min_speed > 0)
            where.push_back("mean_speed >= " + std::to_string(f.min_speed));

        if (!where.empty()) {
            sql += " WHERE ";
            for (size_t i = 0; i < where.size(); ++i) {
                if (i > 0) sql += " AND ";
                sql += where[i];
            }
        }
        sql += " ORDER BY start_frame";

        try {
            auto result = con_->Query(sql);
            if (result->HasError()) return rows;
            while (auto chunk = result->Fetch()) {
                for (uint64_t r = 0; r < chunk->size(); ++r) {
                    BoutRow row;
                    row.id = chunk->GetValue(0, r).GetValue<int32_t>();
                    row.start_frame = chunk->GetValue(1, r).GetValue<int32_t>();
                    row.end_frame = chunk->GetValue(2, r).GetValue<int32_t>();
                    row.n_frames = chunk->GetValue(3, r).GetValue<int32_t>();
                    row.duration_s = chunk->GetValue(4, r).GetValue<float>();
                    row.mean_speed = chunk->GetValue(5, r).IsNull() ? 0 : chunk->GetValue(5, r).GetValue<float>();
                    row.max_speed = chunk->GetValue(6, r).IsNull() ? 0 : chunk->GetValue(6, r).GetValue<float>();
                    row.mean_confidence = chunk->GetValue(7, r).IsNull() ? 0 : chunk->GetValue(7, r).GetValue<float>();
                    row.scut_z_mean = chunk->GetValue(8, r).IsNull() ? 0 : chunk->GetValue(8, r).GetValue<float>();
                    row.ik_mean_mm = chunk->GetValue(9, r).IsNull() ? NAN : chunk->GetValue(9, r).GetValue<float>();
                    row.status = chunk->GetValue(10, r).IsNull() ? 0 : chunk->GetValue(10, r).GetValue<int32_t>();
                    row.notes = chunk->GetValue(11, r).IsNull() ? "" : chunk->GetValue(11, r).GetValue<std::string>();
                    rows.push_back(row);
                }
            }
        } catch (...) {}
        return rows;
    }

    void update_status(int bout_id, int status) {
        if (!con_) return;
        con_->Query("UPDATE bouts SET status = " + std::to_string(status) +
                     " WHERE id = " + std::to_string(bout_id));
    }

    void update_range(int bout_id, int start, int end, float fps) {
        if (!con_) return;
        int n = end - start + 1;
        float dur = n / fps;
        con_->Query("UPDATE bouts SET start_frame = " + std::to_string(start) +
                     ", end_frame = " + std::to_string(end) +
                     ", n_frames = " + std::to_string(n) +
                     ", duration_s = " + std::to_string(dur) +
                     " WHERE id = " + std::to_string(bout_id));
    }

    void delete_bout(int bout_id) {
        if (!con_) return;
        con_->Query("DELETE FROM bouts WHERE id = " + std::to_string(bout_id));
    }

    int insert_bout(int start, int end, float fps) {
        if (!con_) return -1;
        int n = end - start + 1;
        float dur = n / fps;
        auto result = con_->Query(
            "INSERT INTO bouts (id, start_frame, end_frame, n_frames, duration_s, status) "
            "VALUES ((SELECT COALESCE(MAX(id),0)+1 FROM bouts), " +
            std::to_string(start) + ", " + std::to_string(end) + ", " +
            std::to_string(n) + ", " + std::to_string(dur) + ", 0) RETURNING id");
        if (result->HasError()) return -1;
        auto chunk = result->Fetch();
        if (!chunk || chunk->size() == 0) return -1;
        return chunk->GetValue(0, 0).GetValue<int32_t>();
    }

    int bout_count() {
        if (!con_) return 0;
        auto result = con_->Query("SELECT COUNT(*) FROM bouts");
        auto chunk = result->Fetch();
        return chunk ? chunk->GetValue(0, 0).GetValue<int32_t>() : 0;
    }

private:
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> con_;
};

