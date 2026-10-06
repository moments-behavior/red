// tests/test_gui.cpp
// Unit tests for pure math functions in gui.h.
// Compiled as the test_gui target (same deps as red, replacing red.cpp).

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "camera.h"
#include "deferred_queue.h"
#include "global.h"
#include "gui.h"
#include "gui/gui_keypoints.h"
#include "annotation_csv.h"
#include "gui/popup_stack.h"
#include "gui/toast.h"
#include "gui/transport_bar.h"
#include "project_handler.h"
#include "project.h"
#include "app_context.h"
#include "gui/bbox_tool.h"
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <thread>

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT_TRUE(expr)                                                      \
    do {                                                                       \
        if (expr) {                                                            \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: expected true: %s\n", __FILE__,    \
                    __LINE__, #expr);                                          \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_NEAR(a, b, eps)                                                 \
    do {                                                                       \
        float _a = (float)(a), _b = (float)(b), _e = (float)(eps);           \
        float _diff = fabsf(_a - _b);                                         \
        if (_diff <= _e) {                                                     \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: |%s - %s| = %g > %g\n",           \
                    __FILE__, __LINE__, #a, #b, (double)_diff, (double)_e);   \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)


// ---------------------------------------------------------------------------
// current_timestamp (AnnotationCSV)
// Format: YYYY_MM_DD_HH_MM_SS  (19 chars, underscores at 4,7,10,13,16)
// ---------------------------------------------------------------------------

static void test_current_date_time() {
    std::string dt = AnnotationCSV::current_timestamp();

    EXPECT_TRUE(dt.length() == 19);

    // Underscores at expected positions
    EXPECT_TRUE(dt[4]  == '_');
    EXPECT_TRUE(dt[7]  == '_');
    EXPECT_TRUE(dt[10] == '_');
    EXPECT_TRUE(dt[13] == '_');
    EXPECT_TRUE(dt[16] == '_');

    // All other characters are digits
    for (int i = 0; i < 19; i++) {
        if (i == 4 || i == 7 || i == 10 || i == 13 || i == 16)
            continue;
        EXPECT_TRUE(isdigit((unsigned char)dt[i]));
    }
}
// ---------------------------------------------------------------------------
// DeferredQueue
// ---------------------------------------------------------------------------

static void test_deferred_queue_basic() {
    DeferredQueue q;
    EXPECT_TRUE(q.size() == 0);

    int counter = 0;
    q.enqueue([&]() { counter += 1; });
    q.enqueue([&]() { counter += 10; });
    EXPECT_TRUE(q.size() == 2);

    q.flush();
    EXPECT_TRUE(counter == 11);
    EXPECT_TRUE(q.size() == 0);

    // Flush on empty queue is a no-op
    q.flush();
    EXPECT_TRUE(counter == 11);
}

static void test_deferred_queue_thread_safety() {
    DeferredQueue q;
    std::atomic<int> counter{0};

    // Enqueue from multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 100; j++)
                q.enqueue([&]() { counter++; });
        });
    }
    for (auto &t : threads)
        t.join();

    EXPECT_TRUE(q.size() == 1000);
    q.flush();
    EXPECT_TRUE(counter.load() == 1000);
    EXPECT_TRUE(q.size() == 0);
}

// ---------------------------------------------------------------------------
// PopupStack
// ---------------------------------------------------------------------------

static void test_popup_stack_basic() {
    PopupStack ps;
    EXPECT_TRUE(ps.pending.empty());
    EXPECT_FALSE(ps.has_active);

    ps.pushError("Something went wrong");
    EXPECT_TRUE(ps.pending.size() == 1);
    EXPECT_TRUE(ps.pending[0].type == PopupEntry::Error);
    EXPECT_TRUE(ps.pending[0].message == "Something went wrong");
    EXPECT_TRUE(ps.pending[0].title == "Error");
}

static void test_popup_stack_confirm() {
    PopupStack ps;
    bool confirmed = false;
    ps.pushConfirm("Delete?", "Are you sure?", [&]() { confirmed = true; });
    EXPECT_TRUE(ps.pending.size() == 1);
    EXPECT_TRUE(ps.pending[0].type == PopupEntry::Confirm);
    EXPECT_TRUE(ps.pending[0].on_confirm != nullptr);

    // Simulate calling on_confirm
    ps.pending[0].on_confirm();
    EXPECT_TRUE(confirmed);
}

static void test_popup_stack_fifo() {
    PopupStack ps;
    ps.pushError("First");
    ps.pushInfo("Info", "Second");
    ps.pushError("Third");
    EXPECT_TRUE(ps.pending.size() == 3);
    EXPECT_TRUE(ps.pending[0].message == "First");
    EXPECT_TRUE(ps.pending[1].message == "Second");
    EXPECT_TRUE(ps.pending[2].message == "Third");
}

// ---------------------------------------------------------------------------
// ToastQueue
// ---------------------------------------------------------------------------

static void test_toast_queue_basic() {
    ToastQueue tq;
    EXPECT_TRUE(tq.size() == 0);

    tq.push("Hello");
    EXPECT_TRUE(tq.size() == 1);
    EXPECT_TRUE(tq.toasts[0].level == Toast::Info);
    EXPECT_NEAR(tq.toasts[0].duration_sec, 4.0f, 0.01f);

    tq.pushSuccess("Done!");
    EXPECT_TRUE(tq.size() == 2);
    EXPECT_TRUE(tq.toasts[1].level == Toast::Success);
    EXPECT_NEAR(tq.toasts[1].duration_sec, 5.0f, 0.01f);

    tq.pushError("Bad!");
    EXPECT_TRUE(tq.size() == 3);
    EXPECT_TRUE(tq.toasts[2].level == Toast::Error);
    EXPECT_NEAR(tq.toasts[2].duration_sec, 8.0f, 0.01f);
}

// ---------------------------------------------------------------------------
// ProjectHandlerRegistry
// ---------------------------------------------------------------------------

static void test_project_handler_registry() {
    ProjectHandlerRegistry reg;
    EXPECT_TRUE(reg.size() == 0);

    int save_calls = 0;
    int load_calls = 0;
    std::string loaded_value;

    reg.add({"test_section",
             [&]() -> nlohmann::json {
                 save_calls++;
                 return {{"key", "value"}};
             },
             [&](const nlohmann::json &j) {
                 load_calls++;
                 loaded_value = j.value("key", std::string{});
             }});

    EXPECT_TRUE(reg.size() == 1);

    // Test save
    nlohmann::json j;
    j["existing"] = 42;
    project_handlers_save(reg, j);
    EXPECT_TRUE(save_calls == 1);
    EXPECT_TRUE(j.contains("test_section"));
    EXPECT_TRUE(j["test_section"]["key"] == "value");
    EXPECT_TRUE(j["existing"] == 42); // preserved

    // Test load
    project_handlers_load(reg, j);
    EXPECT_TRUE(load_calls == 1);
    EXPECT_TRUE(loaded_value == "value");

    // Test load with missing section (should silently skip)
    nlohmann::json j2;
    j2["other"] = "data";
    project_handlers_load(reg, j2);
    EXPECT_TRUE(load_calls == 1); // not called again
}

// ---------------------------------------------------------------------------
// TransportBarState: default initialization
// ---------------------------------------------------------------------------

static void test_transport_bar_state_defaults() {
    TransportBarState s;
    EXPECT_FALSE(s.slider_text_editing);
    EXPECT_TRUE(s.edit_buf == 0);
}

// ---------------------------------------------------------------------------
// TransportBarState: slider text-input pause/resume logic
// ---------------------------------------------------------------------------

static void test_transport_slider_text_edit_pause() {
    // Cmd+click on slider pauses permanently (like Space), then seeks on Enter.
    TransportBarState state;
    PlaybackState ps;
    ps.play_video = true;
    ps.video_loaded = true;
    ps.slider_frame_number = 500;

    // --- Frame 1: edit_buf syncs from slider_frame_number, then text input begins ---
    state.edit_buf = ps.slider_frame_number;  // sync (not text editing yet)
    EXPECT_TRUE(state.edit_buf == 500);

    bool text_input = true;
    if (text_input && !state.slider_text_editing) {
        if (ps.play_video) {
            ps.play_video = false;
            ps.pause_selected = 0;
        }
        state.slider_text_editing = true;
        ps.slider_text_editing = true;
    }

    EXPECT_TRUE(state.slider_text_editing);
    EXPECT_TRUE(ps.slider_text_editing);
    EXPECT_FALSE(ps.play_video);  // paused permanently

    // --- Frame 2: user types "1234" into edit_buf; external sync is blocked ---
    // (edit_buf sync skipped because slider_text_editing is true)
    ps.slider_frame_number = 600;  // simulate external write from frame advance
    state.edit_buf = 1234;         // user typed this
    // Our guard: if (!state.slider_text_editing) state.edit_buf = ps.slider_frame_number;
    // Since slider_text_editing is true, edit_buf stays at 1234
    EXPECT_TRUE(state.edit_buf == 1234);  // NOT overwritten by external 600

    // --- Frame 3: user presses Enter (IsItemDeactivatedAfterEdit) ---
    bool deactivated_after_edit = true;
    bool seek_called = false;
    int seek_target = -1;
    if (state.slider_text_editing && deactivated_after_edit) {
        seek_called = true;
        seek_target = state.edit_buf;
        state.slider_text_editing = false;
        ps.slider_text_editing = false;
    }

    EXPECT_TRUE(seek_called);
    EXPECT_TRUE(seek_target == 1234);  // seeks to what user typed
    EXPECT_FALSE(ps.play_video);       // stays paused
    EXPECT_FALSE(state.slider_text_editing);
    EXPECT_FALSE(ps.slider_text_editing);
}

static void test_transport_slider_text_edit_escape() {
    // Cmd+click then Escape cancels without seeking.
    TransportBarState state;
    PlaybackState ps;
    ps.play_video = true;
    ps.slider_frame_number = 500;

    // Enter text editing
    state.edit_buf = ps.slider_frame_number;
    state.slider_text_editing = true;
    ps.slider_text_editing = true;
    ps.play_video = false;
    ps.pause_selected = 0;

    // User presses Escape: IsItemDeactivatedAfterEdit = false,
    // IsItemActive = false, TempInputIsActive = false
    bool deactivated_after_edit = false;
    bool item_active = false;
    bool text_input = false;
    bool seek_called = false;

    if (state.slider_text_editing) {
        if (deactivated_after_edit) {
            seek_called = true;
        } else if (!item_active && !text_input) {
            // Cancel — no seek
            state.slider_text_editing = false;
            ps.slider_text_editing = false;
        }
    }

    EXPECT_FALSE(seek_called);
    EXPECT_FALSE(state.slider_text_editing);
    EXPECT_FALSE(ps.slider_text_editing);
    EXPECT_FALSE(ps.play_video);  // stays paused
}

static void test_slider_text_editing_blocks_frame_sync() {
    // When ps.slider_text_editing is true, the frame-advance code must NOT
    // overwrite ps.slider_frame_number (which would destroy the text input).
    PlaybackState ps;
    ps.slider_frame_number = 100;
    ps.slider_text_editing = true;
    ps.to_display_frame_number = 200;

    // Simulate the guarded frame-advance sync from red.cpp:
    // if (!ps.slider_text_editing)
    //     ps.slider_frame_number = ps.to_display_frame_number;
    if (!ps.slider_text_editing)
        ps.slider_frame_number = ps.to_display_frame_number;

    EXPECT_TRUE(ps.slider_frame_number == 100);  // NOT overwritten

    // When editing ends, sync resumes
    ps.slider_text_editing = false;
    if (!ps.slider_text_editing)
        ps.slider_frame_number = ps.to_display_frame_number;

    EXPECT_TRUE(ps.slider_frame_number == 200);  // now synced
}

// ---------------------------------------------------------------------------
// INI migration: v4→v5 DockId remap (sidebar dockspace removal)
// ---------------------------------------------------------------------------

static void test_ini_migration_dock_remap() {
    // Simulate a project ini with old sidebar dock references
    std::string content =
        "[Window][Labeling Tool]\n"
        "Pos=8,401\n"
        "Size=269,266\n"
        "Collapsed=0\n"
        "DockId=0x00000100,0\n"
        "\n"
        "[Window][Keypoints]\n"
        "Pos=8,669\n"
        "Size=448,301\n"
        "Collapsed=0\n"
        "DockId=0x00000100,1\n"
        "\n"
        "[Docking][Data]\n"
        "DockSpace ID=0x00000001 Window=0x1BBC0F80 Pos=280,21 Size=1448,956 CentralNode=1\n"
        "DockSpace ID=0x00000100 Window=0xFA8EA1CE Pos=8,29 Size=264,940 CentralNode=1\n";

    // Apply the v4→v5 migration logic (same as in app_context.h)
    bool changed = false;
    {
        const std::string old_dock = "DockId=0x00000100";
        const std::string new_dock = "DockId=0x00000009";
        size_t pos = 0;
        while ((pos = content.find(old_dock, pos)) != std::string::npos) {
            content.replace(pos, old_dock.size(), new_dock);
            pos += new_dock.size();
            changed = true;
        }
        const std::string stale_node = "DockSpace ID=0x00000100";
        pos = content.find(stale_node);
        if (pos != std::string::npos) {
            size_t line_end = content.find('\n', pos);
            if (line_end != std::string::npos)
                line_end += 1;
            else
                line_end = content.size();
            content.erase(pos, line_end - pos);
            changed = true;
        }
    }

    EXPECT_TRUE(changed);

    // Old dock ID should be gone
    EXPECT_TRUE(content.find("0x00000100") == std::string::npos);

    // New dock ID should be present (twice: one per window)
    EXPECT_TRUE(content.find("DockId=0x00000009,0") != std::string::npos);
    EXPECT_TRUE(content.find("DockId=0x00000009,1") != std::string::npos);

    // Stale DockSpace node line should be removed
    EXPECT_TRUE(content.find("DockSpace ID=0x00000100") == std::string::npos);

    // Main DockSpace should survive
    EXPECT_TRUE(content.find("DockSpace ID=0x00000001") != std::string::npos);
}

// ---------------------------------------------------------------------------
// INI migration: full chain (v1→v5)
// ---------------------------------------------------------------------------

static void test_ini_migration_full_chain() {
    // Simulate a very old ini with "File Browser" window + old sidebar dock
    std::string content =
        "[Window][File Browser]\n"
        "Pos=0,0\n"
        "Size=450,600\n"
        "Collapsed=0\n"
        "DockId=0x00000100,0\n"
        "\n"
        "[Window][Labeling Tool]\n"
        "Pos=8,401\n"
        "Size=269,266\n"
        "Collapsed=0\n"
        "DockId=0x00000100,1\n"
        "\n"
        "[Docking][Data]\n"
        "DockSpace ID=0x00000100 Window=0xFA8EA1CE Pos=8,29 Size=264,940 CentralNode=1\n";

    // Apply all migration steps from migrate_ini_window_names
    bool changed = false;

    // v1→v2: File Browser → Navigator
    changed |= migrate_ini_section(content,
        "[Window][File Browser]", "[Window][Navigator]");
    // v2→v3: Navigator → Controls
    changed |= migrate_ini_section(content,
        "[Window][Navigator]", "[Window][Controls]");
    // v3→v4: Remove Controls
    {
        const std::string header = "[Window][Controls]";
        size_t pos = content.find(header);
        if (pos != std::string::npos) {
            size_t section_end = content.find("\n[", pos + 1);
            if (section_end == std::string::npos)
                section_end = content.size();
            else
                section_end += 1;
            content.erase(pos, section_end - pos);
            changed = true;
        }
    }
    // v4→v5: Remap DockId 0x00000100 → 0x00000009
    {
        const std::string old_dock = "DockId=0x00000100";
        const std::string new_dock = "DockId=0x00000009";
        size_t pos = 0;
        while ((pos = content.find(old_dock, pos)) != std::string::npos) {
            content.replace(pos, old_dock.size(), new_dock);
            pos += new_dock.size();
            changed = true;
        }
        const std::string stale_node = "DockSpace ID=0x00000100";
        pos = content.find(stale_node);
        if (pos != std::string::npos) {
            size_t line_end = content.find('\n', pos);
            if (line_end != std::string::npos) line_end += 1;
            else line_end = content.size();
            content.erase(pos, line_end - pos);
            changed = true;
        }
    }

    EXPECT_TRUE(changed);

    // File Browser / Navigator / Controls should all be gone
    EXPECT_TRUE(content.find("File Browser") == std::string::npos);
    EXPECT_TRUE(content.find("Navigator") == std::string::npos);
    EXPECT_TRUE(content.find("Controls") == std::string::npos);

    // Old dock ID should be gone
    EXPECT_TRUE(content.find("0x00000100") == std::string::npos);

    // Labeling Tool should survive with new dock ID
    EXPECT_TRUE(content.find("[Window][Labeling Tool]") != std::string::npos);
    EXPECT_TRUE(content.find("DockId=0x00000009") != std::string::npos);
}

// ---------------------------------------------------------------------------
// INI migration: no-op on already-migrated content
// ---------------------------------------------------------------------------

static void test_ini_migration_idempotent() {
    // Content that has already been migrated to v5
    std::string content =
        "[Window][Labeling Tool]\n"
        "Pos=0,51\n"
        "Size=280,463\n"
        "Collapsed=0\n"
        "DockId=0x00000009,0\n"
        "\n"
        "[Docking][Data]\n"
        "DockNode ID=0x00000001 Pos=0,51 Size=1728,926 Split=X\n";

    std::string original = content;
    bool changed = false;

    // Run all migrations — nothing should change
    changed |= migrate_ini_section(content,
        "[Window][File Browser]", "[Window][Navigator]");
    changed |= migrate_ini_section(content,
        "[Window][Navigator]", "[Window][Controls]");
    {
        const std::string header = "[Window][Controls]";
        size_t pos = content.find(header);
        if (pos != std::string::npos) { changed = true; }
    }
    {
        const std::string old_dock = "DockId=0x00000100";
        size_t pos = content.find(old_dock);
        if (pos != std::string::npos) { changed = true; }
    }

    EXPECT_FALSE(changed);
    EXPECT_TRUE(content == original);
}

// ---------------------------------------------------------------------------
// PlaybackState: speed computation logic
// ---------------------------------------------------------------------------

static void test_playback_speed_computation() {
    PlaybackState ps;
    ps.video_loaded = true;
    ps.play_video = true;

    // Simulate: 30 frames elapsed over 1.0 second at 60fps
    // Expected: inst_speed = 30 / (60 * 1.0) = 0.5x
    ps.last_frame_num_playspeed = 0;
    int current_frame = 30;
    double wall_seconds = 1.0;
    double video_fps = 60.0;

    if (wall_seconds > 0.5 && ps.play_video) {
        int frame_delta = current_frame - ps.last_frame_num_playspeed;
        ps.inst_speed = frame_delta / (video_fps * wall_seconds);
        ps.last_frame_num_playspeed = current_frame;
    }

    EXPECT_NEAR(ps.inst_speed, 0.5, 0.001);
    EXPECT_TRUE(ps.last_frame_num_playspeed == 30);

    // Simulate: 60 more frames over 1.0 second = 1.0x realtime
    current_frame = 90;
    wall_seconds = 1.0;
    {
        int frame_delta = current_frame - ps.last_frame_num_playspeed;
        ps.inst_speed = frame_delta / (video_fps * wall_seconds);
        ps.last_frame_num_playspeed = current_frame;
    }

    EXPECT_NEAR(ps.inst_speed, 1.0, 0.001);
}

// ---------------------------------------------------------------------------
// PlaybackState: default initialization
// ---------------------------------------------------------------------------

static void test_playback_state_defaults() {
    PlaybackState ps;
    EXPECT_FALSE(ps.play_video);
    EXPECT_FALSE(ps.video_loaded);
    EXPECT_TRUE(ps.realtime_playback);
    EXPECT_NEAR(ps.set_playback_speed, 1.0f, 0.001f);
    EXPECT_NEAR(ps.inst_speed, 1.0, 0.001);
    EXPECT_TRUE(ps.slider_frame_number == 0);
    EXPECT_TRUE(ps.pause_selected == 0);
    EXPECT_FALSE(ps.slider_just_changed);
    EXPECT_FALSE(ps.just_seeked);
    EXPECT_FALSE(ps.pause_seeked);
    EXPECT_NEAR(ps.accumulated_play_time, 0.0, 0.001);
    EXPECT_FALSE(ps.slider_text_editing);
}


// ---------------------------------------------------------------------------
// reprojection(): what a triangulate does to an occluded view
// ---------------------------------------------------------------------------

// Three pinhole cameras on a baseline, no distortion, 1280x720. Camera 1 sits
// at the origin looking down +Z; 0 and 2 are shifted a metre either side, so a
// point in front of the rig is comfortably inside all three images.
static void build_rig(std::vector<CameraParams> &cams, RenderScene &scene,
                      std::vector<u32> &w, std::vector<u32> &h,
                      double shift_cam2 = 1.0) {
    const double f = 1000.0, cx = 640.0, cy = 360.0;
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = f; K(1, 1) = f; K(0, 2) = cx; K(1, 2) = cy;

    const double tx[3] = {1.0, 0.0, -shift_cam2};
    cams.resize(3);
    for (int i = 0; i < 3; i++) {
        cams[i].telecentric = false;
        cams[i].k = K;
        cams[i].dist_coeffs.setZero();
        cams[i].r = Eigen::Matrix3d::Identity();
        cams[i].tvec = Eigen::Vector3d(tx[i], 0.0, 0.0);
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.setZero();
        Rt.block<3, 3>(0, 0) = cams[i].r;
        Rt.col(3) = cams[i].tvec;
        cams[i].projection_mat = K * Rt;
    }
    w.assign(3, 1280);
    h.assign(3, 720);
    scene.num_cams = 3;
    scene.image_width = w.data();
    scene.image_height = h.data();
}

// Place `p` in view v as a manual label, using the same bottom-origin
// convention the rest of red stores keypoints in.
static void place_manual(FrameAnnotation &fa, const std::vector<CameraParams> &cams,
                         const RenderScene &scene, int v, u32 node,
                         const Eigen::Vector3d &p) {
    Eigen::Matrix<double, 5, 1> zero; zero.setZero();
    Eigen::Vector2d px = red_math::projectPointR(p, cams[v].r, cams[v].tvec,
                                                 cams[v].k, zero);
    fa.cameras[v].keypoints[node].x = px(0);
    fa.cameras[v].keypoints[node].y = (double)scene.image_height[v] - px(1);
    fa.cameras[v].keypoints[node].set_manual();
}

static void make_frame(FrameAnnotation &fa, u32 nodes) {
    fa.cameras.resize(3);
    for (auto &c : fa.cameras) c.keypoints.assign(nodes, Keypoint2D{});
    fa.kp3d.assign(nodes, Keypoint3D{});
}

// The question this answers: mark a node occluded in one view, label it in the
// other two, triangulate -- does the occluded view get a position to draw its
// cross at, without the occlusion assessment being lost?
static void test_reprojection_refreshes_occluded_view() {
    SkeletonContext skel;
    skel.num_nodes = 1;
    skel.num_edges = 0;

    std::vector<CameraParams> cams;
    RenderScene scene{};
    std::vector<u32> w, h;
    build_rig(cams, scene, w, h);

    FrameAnnotation fa;
    make_frame(fa, skel.num_nodes);

    const Eigen::Vector3d P(0.10, 0.05, 5.0);
    place_manual(fa, cams, scene, 1, 0, P);
    place_manual(fa, cams, scene, 2, 0, P);

    // View 0: judged hidden, and never placed -- no coordinates at all.
    fa.cameras[0].keypoints[0].set_manual();
    fa.cameras[0].keypoints[0].set_occluded();
    fa.cameras[0].keypoints[0].x = UNLABELED;
    fa.cameras[0].keypoints[0].y = UNLABELED;

    reprojection(fa, &skel, cams, &scene);

    // Two manual views were enough to solve.
    EXPECT_TRUE(fa.kp3d[0].exist);
    EXPECT_TRUE(fa.kp3d[0].is_triangulated());
    EXPECT_NEAR(fa.kp3d[0].x, P(0), 1e-6);
    EXPECT_NEAR(fa.kp3d[0].z, P(2), 1e-6);

    const Keypoint2D &occ = fa.cameras[0].keypoints[0];
    // The assessment stands, and it is still not a point you can use.
    EXPECT_TRUE(occ.is_occluded());
    EXPECT_FALSE(occ.usable());       // a position, but not one to act on
    EXPECT_TRUE(occ.has_pos);         // the solve gave it one to draw at
    EXPECT_TRUE(occ.is_manual());          // the author of the assessment survives
    // ...but it now has somewhere to draw the cross, and says where that
    // position came from.
    EXPECT_TRUE(occ.x != UNLABELED && occ.y != UNLABELED);
    EXPECT_TRUE(occ.reprojected);
    Eigen::Matrix<double, 5, 1> zero; zero.setZero();
    Eigen::Vector2d expect = red_math::projectPointR(P, cams[0].r, cams[0].tvec,
                                                     cams[0].k, zero);
    EXPECT_NEAR(occ.x, expect(0), 1e-4);
    EXPECT_NEAR(occ.y, (double)scene.image_height[0] - expect(1), 1e-4);

    // The views that were labelled keep their authorship through the refresh.
    EXPECT_TRUE(fa.cameras[1].keypoints[0].is_manual());
    EXPECT_TRUE(fa.cameras[1].keypoints[0].has_pos);
}

// The other half of the same path: if the solve lands outside this camera's
// image there is nowhere to put the cross, so the position must go rather than
// sit at a stale spot. Regression for the bug in 6d97e65.
static void test_reprojection_drops_offscreen_occluded_position() {
    SkeletonContext skel;
    skel.num_nodes = 1;
    skel.num_edges = 0;

    std::vector<CameraParams> cams;
    RenderScene scene{};
    std::vector<u32> w, h;
    build_rig(cams, scene, w, h);

    FrameAnnotation fa;
    make_frame(fa, skel.num_nodes);

    // First solve: in front of the rig, visible everywhere.
    const Eigen::Vector3d P1(0.0, 0.0, 5.0);
    place_manual(fa, cams, scene, 1, 0, P1);
    place_manual(fa, cams, scene, 2, 0, P1);
    fa.cameras[0].keypoints[0].set_manual();
    fa.cameras[0].keypoints[0].set_occluded();
    reprojection(fa, &skel, cams, &scene);
    EXPECT_TRUE(fa.cameras[0].keypoints[0].x != UNLABELED);

    // Second solve: far off to the side, still in front of the rig but well
    // outside camera 0's 1280px image.
    const Eigen::Vector3d P2(-40.0, 0.0, 5.0);
    place_manual(fa, cams, scene, 1, 0, P2);
    place_manual(fa, cams, scene, 2, 0, P2);
    reprojection(fa, &skel, cams, &scene);

    const Keypoint2D &occ = fa.cameras[0].keypoints[0];
    EXPECT_TRUE(occ.is_occluded());       // still judged hidden
    EXPECT_TRUE(occ.is_manual());         // still that judgement's author
    EXPECT_FALSE(occ.usable());
    EXPECT_FALSE(occ.has_pos);       // the solve left the frame, so none
    EXPECT_TRUE(occ.x == UNLABELED); // nothing to draw
    EXPECT_TRUE(occ.y == UNLABELED);
    EXPECT_FALSE(occ.reprojected);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

// A new project is Untitled: setting it up must not touch the disk (it has no
// folder yet -- creating "" failed with "No such file or directory").
static void test_setup_untitled_project() {
    auto skeleton_map = skeleton_get_all();
    ProjectManager pm;
    pm.untitled = true;
    pm.skeleton_name = skeleton_map.begin()->first;
    pm.camera_names = {"Cam1"};
    SkeletonContext skeleton;
    std::string err;
    EXPECT_TRUE(setup_project(pm, skeleton, skeleton_map, &err));
    if (!err.empty()) fprintf(stderr, "  setup_project: %s\n", err.c_str());
    EXPECT_TRUE(pm.project_path.empty());
    EXPECT_TRUE(pm.keypoints_root_folder.empty());
}

// Save Project's check: a new folder or an empty one is fine; a name that is
// already there (folder with files, or a file) is not.
static void test_untitled_save_problem() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "red_test_save_problem";
    fs::remove_all(root);
    fs::create_directories(root / "empty");
    fs::create_directories(root / "full");
    { std::ofstream(root / "full" / "x.txt") << "x"; }
    { std::ofstream(root / "afile") << "x"; }
    const std::string r = root.string();
    EXPECT_TRUE(untitled_save_problem("new", r).empty());
    EXPECT_TRUE(untitled_save_problem("empty", r).empty());
    EXPECT_FALSE(untitled_save_problem("full", r).empty());
    EXPECT_FALSE(untitled_save_problem("afile", r).empty());
    EXPECT_FALSE(untitled_save_problem("", r).empty());
    EXPECT_FALSE(untitled_save_problem("a/b", r).empty());
    fs::remove_all(root);
}

// ── Bbox tool, driven through real ImGui/ImPlot frames ──
// Headless: no window or renderer, a mouse fed through io events, and the
// same calls a camera view makes. Covers what only shows when the input
// code runs: drawing over an existing box, dragging an edge, and moving a
// box by its label.
namespace bbox_ui {
struct Headless {
    Headless() {
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.DisplaySize = ImVec2(800, 600);
        io.DeltaTime = 1.0f / 60.0f;
        io.IniFilename = nullptr;
        // Apply each frame's key and mouse events together; trickling
        // spreads them over frames, which a test scripting one state per
        // frame does not want.
        io.ConfigInputTrickleEventQueue = false;
        unsigned char *px; int w, h;
        io.Fonts->GetTexDataAsRGBA32(&px, &w, &h);
    }
    ~Headless() {
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
};

constexpr int kW = 640, kH = 480;

// One frame with the pointer at `mouse` (pixels), the left button and Shift
// as given; `body` runs inside the plot, which spans the plot coords
// [0,kW] x [0,kH]. The plot sees the pointer from the frame after it moves
// there, so a test hovers for two frames before acting.
template <typename Body>
void frame(BBoxToolState &st, ImVec2 mouse, bool down, bool shift, Body body,
           bool right_down = false) {
    ImGuiIO &io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Shift, shift);
    io.AddKeyEvent(ImGuiKey_LeftShift, shift);
    io.AddMousePosEvent(mouse.x, mouse.y);
    io.AddMouseButtonEvent(0, down);
    io.AddMouseButtonEvent(1, right_down);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    ImGui::Begin("view", nullptr, ImGuiWindowFlags_NoDecoration);
    if (ImPlot::BeginPlot("##cam", ImVec2(-1, -1), ImPlotFlags_NoMenus)) {
        const ImPlotAxisFlags lock = bbox_blocks_pan(st) ? ImPlotAxisFlags_Lock
                                                         : ImPlotAxisFlags_None;
        ImPlot::SetupAxes(nullptr, nullptr, lock, lock);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, kW, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, kH, ImPlotCond_Always);
        body();
        ImPlot::EndPlot();
    }
    ImGui::End();
    ImGui::Render();
}

// Pixel position of plot point (x, y), read inside a frame.
ImVec2 px(BBoxToolState &st, double x, double y) {
    ImVec2 p;
    frame(st, ImVec2(-1, -1), false, false, [&] { p = ImPlot::PlotToPixels(x, y); });
    return p;
}

// A frame with instance 0 holding a box on camera 0, plot x 100..300, y 100..300.
AnnotationMap one_box() {
    AnnotationMap amap;
    auto &fa = get_or_create_frame(amap, 0, 1, 1, 0);
    auto &e = fa.cameras[0].get_extras();
    e.bbox_x = 100; e.bbox_y = kH - 300; e.bbox_w = 200; e.bbox_h = 200;
    e.has_bbox = true;
    return amap;
}
} // namespace bbox_ui

static void test_bbox_ui_draw_over_box() {
    printf("  test_bbox_ui_draw_over_box...\n");
    using namespace bbox_ui;
    Headless ui;
    BBoxToolState st;
    LabelInfo info;
    AnnotationMap amap = one_box();
    int active = 0;
    auto input = [&] { bbox_handle_input(st, info, amap, 0, 0, active, 1, 1, kW, kH); };
    const ImVec2 a = px(st, 150, 250), b = px(st, 250, 150);   // inside the box
    frame(st, a, false, true, input);    // hover with Shift
    frame(st, a, false, true, input);
    frame(st, a, true, true, input);     // press: starts a box
    EXPECT_TRUE(st.drawing);
    frame(st, b, true, true, input);     // drag
    frame(st, b, false, true, input);    // let go: commits
    EXPECT_FALSE(st.drawing);
    // Instance 0 already had a box here: the new one is instance 1.
    EXPECT_TRUE((int)amap[0].size() == 2);
    if (amap[0].size() == 2) {
        const auto &e = *amap[0][1].cameras[0].extras;
        EXPECT_TRUE(amap[0][1].cameras[0].has_bbox());
        EXPECT_NEAR(e.bbox_x, 150, 2.0);
        EXPECT_NEAR(e.bbox_w, 100, 2.0);
        EXPECT_TRUE(active == 1);
    }
}

static void test_bbox_ui_drag_edge_and_label() {
    printf("  test_bbox_ui_drag_edge_and_label...\n");
    using namespace bbox_ui;
    Headless ui;
    BBoxToolState st;
    LabelInfo info;
    AnnotationMap amap = one_box();
    int active = 0;
    auto input = [&] { bbox_handle_input(st, info, amap, 0, 0, active, 1, 1, kW, kH); };
    auto &e = amap[0][0].cameras[0].get_extras();

    // Left edge: hover shows it, press and drag moves it to x = 50.
    const ImVec2 edge = px(st, 100, 200), to = px(st, 50, 200);
    frame(st, edge, false, false, input);
    frame(st, edge, false, false, input);
    EXPECT_TRUE(st.edge_mask == kEdgeL);
    frame(st, edge, true, false, input);
    EXPECT_TRUE(st.resizing);
    frame(st, to, true, false, input);
    frame(st, to, false, false, input);
    EXPECT_FALSE(st.resizing);
    EXPECT_NEAR(e.bbox_x, 50, 2.0);
    EXPECT_NEAR(e.bbox_w, 250, 2.0);    // the right edge stayed at 300

    // The label (top-left): drag moves the whole box, size kept.
    const ImPlotPoint lab = box_label_anchor(e.bbox_x, kH - e.bbox_y);
    const ImVec2 l0 = px(st, lab.x, lab.y), l1 = px(st, lab.x + 100, lab.y - 50);
    const double w0 = e.bbox_w, h0 = e.bbox_h, x0 = e.bbox_x, y0 = e.bbox_y;
    frame(st, l0, false, false, input);
    frame(st, l0, false, false, input);
    EXPECT_TRUE(st.edge_mask == kEdgeMove);
    frame(st, l0, true, false, input);
    frame(st, l1, true, false, input);
    frame(st, l1, false, false, input);
    EXPECT_NEAR(e.bbox_x, x0 + 100, 2.0);
    EXPECT_NEAR(e.bbox_y, y0 + 50, 2.0);   // image y grows downward
    EXPECT_NEAR(e.bbox_w, w0, 1e-9);
    EXPECT_NEAR(e.bbox_h, h0, 1e-9);
}

static void test_bbox_ui_right_click_menu() {
    printf("  test_bbox_ui_right_click_menu...\n");
    using namespace bbox_ui;
    Headless ui;
    BBoxToolState st;
    LabelInfo info;
    AnnotationMap amap = one_box();
    int active = 0;
    bool open = false;
    auto input = [&] {
        bbox_handle_input(st, info, amap, 0, 0, active, 1, 1, kW, kH);
        bbox_draw_menu(st, info, amap, 0, 0, active);
        open = ImGui::IsPopupOpen("##box_menu");
    };
    const ImVec2 in = px(st, 200, 200);
    frame(st, in, false, false, input);
    frame(st, in, false, false, input);
    EXPECT_FALSE(open);
    frame(st, in, false, false, input, /*right*/ true);
    frame(st, in, false, false, input);
    EXPECT_TRUE(open);
    EXPECT_TRUE(st.menu_instance == 0 && st.menu_cam == 0);
}

// Keypoints coloured by reprojection error: a pinhole camera looking down +z,
// a 3D point projecting to the principal point, and keypoints placed 0, 3 and
// 10 px from it -> green, yellow, red; a node without 3D is grey.
static void test_reprojection_error_colors() {
    printf("  test_reprojection_error_colors...\n");
    CameraParams cam;
    cam.k << 500, 0, 320, 0, 500, 240, 0, 0, 1;
    const double img_h = 480;
    FrameAnnotation fa = make_frame(4, 1);
    // ImPlot coords: y up, so image row 240 is plot y 480 - 240.
    const double off[] = {0, 3, 10};
    for (int n = 0; n < 3; ++n) {
        fa.kp3d[n].x = 0; fa.kp3d[n].y = 0; fa.kp3d[n].z = 10;
        fa.kp3d[n].set_triangulated();
        auto &kp = fa.cameras[0].keypoints[n];
        kp.x = 320 + off[n]; kp.y = img_h - 240; kp.set_manual();
    }
    auto &kp3 = fa.cameras[0].keypoints[3];   // placed, but no 3D
    kp3.x = 100; kp3.y = 100; kp3.set_manual();
    const auto c = reprojection_error_colors(fa, 0, 4, cam, img_h);
    EXPECT_TRUE(c.size() == 4);
    auto is = [](const ImVec4 &a, float r, float g) {
        return std::fabs(a.x - r) < 0.01f && std::fabs(a.y - g) < 0.01f;
    };
    EXPECT_TRUE(is(c[0], 0.25f, 0.9f));   // green
    EXPECT_TRUE(is(c[1], 1.0f, 0.85f));   // yellow
    EXPECT_TRUE(is(c[2], 1.0f, 0.25f));   // red
    EXPECT_TRUE(is(c[3], 0.6f, 0.6f));    // grey
}

int main() {
    test_current_date_time();

    // Infrastructure tests
    test_deferred_queue_basic();
    test_deferred_queue_thread_safety();
    test_popup_stack_basic();
    test_popup_stack_confirm();
    test_popup_stack_fifo();
    test_toast_queue_basic();
    test_project_handler_registry();
    test_setup_untitled_project();
    test_untitled_save_problem();
    test_bbox_ui_draw_over_box();
    test_bbox_ui_drag_edge_and_label();
    test_bbox_ui_right_click_menu();
    test_reprojection_error_colors();

    // Transport bar + UI overhaul tests
    test_transport_bar_state_defaults();
    test_transport_slider_text_edit_pause();
    test_transport_slider_text_edit_escape();
    test_slider_text_editing_blocks_frame_sync();
    test_ini_migration_dock_remap();
    test_ini_migration_full_chain();
    test_ini_migration_idempotent();
    test_playback_speed_computation();
    test_playback_state_defaults();

    // Reprojection write-back
    test_reprojection_refreshes_occluded_view();
    test_reprojection_drops_offscreen_occluded_position();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
