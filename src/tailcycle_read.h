#ifndef RED_TAILCYCLE_READ
#define RED_TAILCYCLE_READ

// Reading tailcycle-dataset tables, shared by the importer and by the
// exporter's in-place save, which has to read the old tables to merge its rows
// into them. C++20 / Arrow only: include from tailcycle_import.cpp and
// tailcycle_export.cpp, never from a C++17 header.

#include "red_build_config.h"

#if defined(RED_HAVE_PARQUET)

#include "tailcycle_schema.h"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Tailcycle {

inline std::shared_ptr<arrow::Table> read_pq(const std::filesystem::path &p) {
    if (!std::filesystem::exists(p)) return nullptr;
    auto file = arrow::io::ReadableFile::Open(p.string());
    if (!file.ok()) return nullptr;
    auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
    if (!reader.ok()) return nullptr;
    auto t = (*reader)->ReadTable();
    return t.ok() ? *t : nullptr;
}

// A dictionary<int32,str> column, decoded row by row. Small tables arrive in
// one chunk; larger ones do not, so chunk offsets are tracked.
struct DictCol {
    std::vector<std::string> vals;
    bool ok = false;
    explicit DictCol(const std::shared_ptr<arrow::Table> &t, const char *name) {
        auto col = t->GetColumnByName(name);
        if (!col) return;
        for (int c = 0; c < col->num_chunks(); c++) {
            auto d = std::dynamic_pointer_cast<arrow::DictionaryArray>(col->chunk(c));
            if (!d) return;
            auto dv = std::dynamic_pointer_cast<arrow::StringArray>(d->dictionary());
            auto ix = std::dynamic_pointer_cast<arrow::Int32Array>(d->indices());
            if (!dv || !ix) return;
            for (int64_t i = 0; i < ix->length(); i++)
                vals.push_back(ix->IsNull(i) ? std::string() : dv->GetString(ix->Value(i)));
        }
        ok = true;
    }
};

struct NumCol {
    std::vector<double> vals;
    std::vector<bool> null;
    bool ok = false;
    explicit NumCol(const std::shared_ptr<arrow::Table> &t, const char *name) {
        auto col = t->GetColumnByName(name);
        if (!col) return;
        for (int c = 0; c < col->num_chunks(); c++) {
            auto a = col->chunk(c);
            for (int64_t i = 0; i < a->length(); i++) {
                null.push_back(a->IsNull(i));
                if (a->IsNull(i)) { vals.push_back(0.0); continue; }
                if (auto f = std::dynamic_pointer_cast<arrow::FloatArray>(a))
                    vals.push_back(f->Value(i));
                else if (auto d = std::dynamic_pointer_cast<arrow::DoubleArray>(a))
                    vals.push_back(d->Value(i));
                else if (auto n = std::dynamic_pointer_cast<arrow::Int32Array>(a))
                    vals.push_back(n->Value(i));
                else return;
            }
        }
        ok = true;
    }
};

// A plain utf8 column (instances.pq `notes`), with its nulls.
struct StrCol {
    std::vector<std::string> vals;
    std::vector<bool> null;
    bool ok = false;
    explicit StrCol(const std::shared_ptr<arrow::Table> &t, const char *name) {
        auto col = t->GetColumnByName(name);
        if (!col) return;
        for (int c = 0; c < col->num_chunks(); c++) {
            auto a = std::dynamic_pointer_cast<arrow::StringArray>(col->chunk(c));
            if (!a) return;
            for (int64_t i = 0; i < a->length(); i++) {
                null.push_back(a->IsNull(i));
                vals.push_back(a->IsNull(i) ? std::string() : a->GetString(i));
            }
        }
        ok = true;
    }
};

// §9: a box with x1 <= x0 or y1 <= y0 is empty and equivalent to no box.
inline bool box_nonempty(double x0, double y0, double x1, double y1) {
    return std::isfinite(x0) && std::isfinite(y0) && std::isfinite(x1) &&
           std::isfinite(y1) && x1 > x0 && y1 > y0;
}

// Which rows the importer turns into something red holds. A row it does not
// load is invisible in red, so the user cannot have edited it, and an in-place
// save must keep it rather than drop it. These must agree with
// tailcycle_import.cpp's read_session.
inline bool keypoint_row_loaded(const std::string &status, bool has_xy) {
    if (status == status::kMissing) return true;
    return (status == status::kVisible || status == status::kProjected) && has_xy;
}
inline bool point3d_row_loaded(const std::string &status, bool has_xyz) {
    return status == status::kVisible && has_xyz;
}
// `labeled` and `present` boxes become red boxes. `absent` never does: a box
// on an explicit negative is not a positive.
inline bool instance_box_loaded(const std::string &status, bool nonempty_box) {
    return nonempty_box && (status == status::kLabeled || status == status::kPresent);
}
// An `absent` row becomes red's absent mark on that view.
inline bool instance_absent_loaded(const std::string &status) {
    return status == status::kAbsent;
}

} // namespace Tailcycle

#endif // RED_HAVE_PARQUET
#endif // RED_TAILCYCLE_READ
