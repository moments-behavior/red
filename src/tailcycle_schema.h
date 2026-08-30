#ifndef RED_TAILCYCLE_SCHEMA
#define RED_TAILCYCLE_SCHEMA

// Arrow schemas and the status vocabularies for the `tailcycle-dataset`
// format (annotation_format.md). Shared by the writer and, later, the reader:
// the column types, the enum spellings and the null conventions are exactly
// the things that go wrong when a writer and a reader each declare their own.
//
// Two places where this follows a real file rather than the spec's prose:
//
//   - `status` is spelled dictionary<int8,str> in the spec tables, but
//     johnson-mouse-tracked writes dictionary<int32,str>, as do all its other
//     key columns. Arrow's Dictionary32Builder produces int32, so int32 is
//     both what real files carry and what the natural builder emits.
//   - `groups.pq` uses a plain string `group_id` while the label tables use a
//     dictionary. One row per group makes a dictionary pointless there.

#include "red_build_config.h"

#if defined(RED_HAVE_PARQUET)

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <memory>
#include <string>

namespace Tailcycle {

// ── status vocabularies (§7, §8, §9) ──
// Closed sets. A typo here is a validation error downstream rather than a
// silent third category, which is the whole point of them being closed.
namespace status {
// keypoints.pq
constexpr const char *kVisible   = "visible";
constexpr const char *kProjected = "projected";
constexpr const char *kMissing   = "missing";
constexpr const char *kUnlabeled = "unlabeled";
// instances.pq
constexpr const char *kLabeled = "labeled";
constexpr const char *kPresent = "present";
constexpr const char *kAbsent  = "absent";
// regions.pq
constexpr const char *kLabelledComplete = "labelled_complete";
} // namespace status

namespace labels {
constexpr const char *kAnnotated = "annotated";
constexpr const char *kTracked   = "tracked";
} // namespace labels

// Key columns are dictionary-encoded: the string is stored once and each row
// carries a small integer. On a dense session -- 8000 frames x 16 cameras x 24
// bodyparts -- this is the difference between a compact file and gigabytes of
// repeated "left_hindpaw".
inline std::shared_ptr<arrow::DataType> dict_str() {
    return arrow::dictionary(arrow::int32(), arrow::utf8());
}

// ── groups.pq (§6) ──
inline std::shared_ptr<arrow::Schema> groups_schema() {
    return arrow::schema({
        arrow::field("group_id", arrow::utf8(), /*nullable=*/false),
        arrow::field("n_frames", arrow::int32(), false),
        arrow::field("fps", arrow::float32(), true),
        arrow::field("source_video", arrow::utf8(), true),
        arrow::field("source_frame_start", arrow::int32(), true),
        arrow::field("source_frame_step", arrow::int32(), true),
        arrow::field("notes", arrow::utf8(), true),
    });
}

// Optional columns are OMITTED from the schema when nothing fills them, not
// written as an all-null column: johnson-mouse-tracked has no `score` column
// at all, and a reader must resolve columns by name against the actual schema
// rather than assume the full set is present. Writing all-null columns would
// work, but it would misrepresent "no scores exist" as "every score is
// unknown", and cost a column's worth of file for the privilege.

// ── keypoints.pq (§7) ──
// x/y are nullable because a `missing` row has no coordinates, and because a
// `visible` row may defer its position to points3d.pq. They are written as
// nulls, never NaN -- a null costs a validity bit and any reader sees the
// absence, where a sentinel has to be known about to be honoured.
inline std::shared_ptr<arrow::Schema> keypoints_schema(bool with_score) {
    arrow::FieldVector f = {
        arrow::field("group_id", dict_str(), false),
        arrow::field("frame", arrow::int32(), false),
        arrow::field("animal_id", dict_str(), false),
        arrow::field("camera", dict_str(), false),
        arrow::field("bodypart", dict_str(), false),
        arrow::field("status", dict_str(), false),
        arrow::field("x", arrow::float32(), true),
        arrow::field("y", arrow::float32(), true),
    };
    if (with_score) f.push_back(arrow::field("score", arrow::float32(), true));
    return arrow::schema(f);
}

// ── points3d.pq (§8) ──
// No `camera`: a 3D point is not a per-camera observation. `projected` is not
// in this table's vocabulary -- there is no per-camera visibility question to
// leave open.
inline std::shared_ptr<arrow::Schema> points3d_schema(bool with_score) {
    arrow::FieldVector f = {
        arrow::field("group_id", dict_str(), false),
        arrow::field("frame", arrow::int32(), false),
        arrow::field("animal_id", dict_str(), false),
        arrow::field("bodypart", dict_str(), false),
        arrow::field("status", dict_str(), false),
        arrow::field("x", arrow::float32(), true),
        arrow::field("y", arrow::float32(), true),
        arrow::field("z", arrow::float32(), true),
    };
    if (with_score) f.push_back(arrow::field("score", arrow::float32(), true));
    return arrow::schema(f);
}

// ── instances.pq (§9) ──
// Boxes are [x0,x1) x [y0,y1), top-left inclusive: width is x1-x0, so an
// integer box is directly usable as an array slice. red stores origin+extent,
// so the writer converts.
inline std::shared_ptr<arrow::Schema> instances_schema() {
    return arrow::schema({
        arrow::field("group_id", dict_str(), false),
        arrow::field("frame", arrow::int32(), false),
        arrow::field("animal_id", dict_str(), false),
        arrow::field("camera", dict_str(), false),
        arrow::field("x0", arrow::float32(), true),
        arrow::field("y0", arrow::float32(), true),
        arrow::field("x1", arrow::float32(), true),
        arrow::field("y1", arrow::float32(), true),
        arrow::field("status", dict_str(), false),
        arrow::field("notes", arrow::utf8(), true),
    });
}

// Write one table to `path`.
//
// store_schema() records the Arrow schema in the file's metadata, without
// which the dictionary columns come back as plain strings on read -- the file
// is still valid, but it no longer round-trips as the same schema, and a
// reader checking column types would reject it.
inline arrow::Status write_table(const std::shared_ptr<arrow::Table> &table,
                                 const std::string &path) {
    ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
    auto props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), out,
        parquet::DEFAULT_MAX_ROW_GROUP_LENGTH,
        parquet::default_writer_properties(), props));
    return out->Close();
}

} // namespace Tailcycle

#endif // RED_HAVE_PARQUET
#endif // RED_TAILCYCLE_SCHEMA
