#ifndef RED_TAILCYCLE_EXPORT
#define RED_TAILCYCLE_EXPORT

// Export a red project as a `tailcycle-dataset` session (annotation_format.md).
//
// Writes the format -- session.toml, calibration.toml, and the Parquet tables --
// and creates groups/<group_id>/ but leaves it EMPTY. Populating it is the
// caller's job, because a group must contain exactly its own frames: a consumer
// reads group frame f as the f-th frame of the media in that folder, and
// source_frame_start is provenance, not an offset to apply. Linking a whole
// recording into a group that starts elsewhere puts every index out, and a link
// resolves only on the machine that wrote it. red's export window extracts
// JPEGs; see export_tailcycle() in export_formats.h.
//
// Deliberately NOT header-only, unlike jarvis_export.h. Arrow 25 requires
// C++20 and red is C++17 everywhere, so every Arrow include lives in
// tailcycle_export.cpp, which CMake builds as its own C++20 target. This
// header stays plain C++17 so the GUI can include it.

#include "annotation.h"
#include "camera.h"
#include <string>
#include <utility>
#include <vector>

namespace TailcycleExport {

struct ExportConfig {
    std::string output_folder;      // dataset root; <root>/<split>/<session>/
    std::string split = "train";    // train | val | test -- a directory level (§2.1)
    std::string session_id;         // becomes the folder name, which IS the id

    std::vector<std::string> camera_names;
    std::vector<CameraParams> calibration;   // parallel to camera_names

    std::vector<std::string> node_names;         // the keypoint axis (§4)
    std::vector<std::pair<int, int>> edges;      // resolved to name pairs on write

    std::string group_id;           // folder under groups/; defaults to session_id
    int n_frames = 0;               // from the media, NOT the annotation range
    float fps = 0.0f;
    int source_frame_start = 0;     // red's frame_number is absolute; §6 rebases
    std::string source_video;

    std::string units = "mm";

    // Which label tables to write. §3 requires at least one of keypoints.pq
    // and points3d.pq, and a session may legitimately carry either alone.
    //
    // TwoD is the default because red's labels are per-camera 2D. Where those
    // 2D are themselves reprojections of a 3D solve -- which is what dense,
    // every-view labelling usually means -- writing both stores the same
    // information twice, and §8 says a derivation is not stored. ThreeD alone
    // is then the honest export.
    //
    // Whenever 3D is written it includes red's triangulated solve: asking for
    // the 3D layer and getting an empty one would be worse than not offering
    // the choice.
    enum class Layers { TwoD, TwoDAndThreeD, ThreeD };
    Layers layers = Layers::TwoD;

    // `labels` is closed at annotated|tracked and a session that is both must
    // be two sessions (§2.6). Rows partition by LabelSource/Kp3DSource; a
    // project with both produces <session>_annotated and <session>_tracked.
    bool export_annotated = true;
    bool export_tracked = true;

    // When set to "annotated" or "tracked", every row goes into ONE session
    // carrying that value, whatever each point's source says. Saving
    // corrections back over an existing session has to reproduce it -- editing
    // an imported session mixes sources, and the usual split would replace one
    // session with <name>_annotated and <name>_tracked sitting beside frames
    // that belong to neither.
    std::string force_labels;

    // The format's animal_id per instance, indexed by FrameAnnotation's
    // instance_id. Supplied when saving back over a session so its own names
    // survive; otherwise ids are generated as a00, a01, ... which is the
    // convention every dataset seen so far uses.
    std::vector<std::string> animal_ids;

    // Saving corrections back over a session that is already on disk, rather
    // than writing a new dataset. Only the label tables are rewritten:
    // session.toml, calibration.toml and groups.pq describe things a label
    // edit does not change, and rewriting them from an ExportConfig would
    // replace what is there with the subset red happens to model. A 3dzef
    // session.toml carries 44 keys -- the checkpoint that produced it, the
    // detector settings, the source paths, assoc_res_max_px -- and the
    // exporter writes 9 of them, so correcting one keypoint would have thrown
    // the other 35 away.
    bool in_place = false;

    std::string provenance_source;
    std::string annotator;          // empty when one annotator authored the root (§2.11)
};

struct ExportStats {
    int sessions_written = 0;
    int keypoint_rows = 0;
    int points3d_rows = 0;
    int instance_rows = 0;
    int frames_with_labels = 0;
    std::vector<std::string> warnings;
    std::vector<std::string> sessions;   // paths actually written
    double elapsed_seconds = 0.0;
};

// Returns false and fills `status` on refusal. Refusals are for cases where
// writing anything would produce a file that loads cleanly and is wrong:
// telecentric calibration (no aniposelib representation), an improper
// rotation (not expressible as Rodrigues), or a camera whose declared size
// disagrees with its media (validation rule 8).
bool export_session(const ExportConfig &config, const AnnotationMap &amap,
                    ExportStats *stats, std::string *status);

// True when red was built with Arrow/Parquet. The GUI hides the window
// otherwise rather than offering a button that cannot work.
bool available();

} // namespace TailcycleExport

#endif // RED_TAILCYCLE_EXPORT
