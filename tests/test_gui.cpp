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
#include "gui/calibration_tool_window.h"
#include "gui/popup_stack.h"
#include "gui/toast.h"
#include "gui/transport_bar.h"
#include "project_handler.h"
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
// current_date_time
// Format: YYYY_MM_DD_HH_MM_SS  (19 chars, underscores at 4,7,10,13,16)
// ---------------------------------------------------------------------------

static void test_current_date_time() {
    std::string dt = current_date_time();

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
// CalibrationToolState: default initialization
// ---------------------------------------------------------------------------

static void test_calib_state_defaults() {
    CalibrationToolState s;
    EXPECT_FALSE(s.show);
    EXPECT_FALSE(s.project_loaded);
    EXPECT_FALSE(s.dock_pending);
    EXPECT_TRUE(s.show_create_dialog);
    EXPECT_FALSE(s.config_loaded);
    EXPECT_FALSE(s.images_loaded);
    EXPECT_TRUE(s.status.empty());
    EXPECT_FALSE(s.img_running);
    EXPECT_FALSE(s.img_done);
    EXPECT_FALSE(s.vid_running);
    EXPECT_FALSE(s.vid_done);

    // Laser defaults
    EXPECT_FALSE(s.laser_ready);
    EXPECT_TRUE(s.laser_total_frames == 0);
    EXPECT_FALSE(s.laser_running);
    EXPECT_FALSE(s.laser_done);
    EXPECT_TRUE(s.laser_status.empty());
    EXPECT_FALSE(s.laser_show_detection);
    EXPECT_FALSE(s.laser_focus_window);
    EXPECT_TRUE(s.laser_progress != nullptr);

    // LaserVizState defaults
    EXPECT_TRUE(s.laser_viz.ready.empty());
    EXPECT_TRUE(s.laser_viz.pending.empty());
    EXPECT_FALSE(s.laser_viz.computing.load());
    EXPECT_TRUE(s.laser_viz.last_green_th == -1);
}

// ---------------------------------------------------------------------------
// CalibrationToolState: reset-on-close logic
// ---------------------------------------------------------------------------

static void test_calib_state_reset_on_close() {
    // Simulate a loaded project, then close
    CalibrationToolState s;
    s.show = true;
    s.project_loaded = true;
    s.show_create_dialog = false;
    s.config_loaded = true;
    s.images_loaded = true;
    s.img_done = true;
    s.status = "Calibration complete!";
    s.laser_ready = true;
    s.laser_done = true;
    s.laser_status = "Laser done";
    s.laser_show_detection = true;

    // Simulate closing (this is what DrawCalibrationToolWindow does when !state.show)
    s.show = false;
    // Inline the reset logic from the draw function
    s.project_loaded = false;
    s.show_create_dialog = true;
    s.config_loaded = false;
    s.images_loaded = false;
    s.img_done = false;
    s.vid_done = false;
    s.status.clear();
    s.laser_ready = false;
    s.laser_done = false;
    s.laser_status.clear();
    s.laser_show_detection = false;
    if (s.laser_viz.worker.joinable())
        s.laser_viz.worker.join();
    s.laser_viz.ready.clear();
    s.laser_viz.pending.clear();

    // Verify all state is reset
    EXPECT_FALSE(s.show);
    EXPECT_FALSE(s.project_loaded);
    EXPECT_TRUE(s.show_create_dialog);
    EXPECT_FALSE(s.config_loaded);
    EXPECT_FALSE(s.images_loaded);
    EXPECT_FALSE(s.img_done);
    EXPECT_FALSE(s.vid_done);
    EXPECT_TRUE(s.status.empty());
    EXPECT_FALSE(s.laser_ready);
    EXPECT_FALSE(s.laser_done);
    EXPECT_TRUE(s.laser_status.empty());
    EXPECT_FALSE(s.laser_show_detection);
    EXPECT_TRUE(s.laser_viz.ready.empty());
    EXPECT_TRUE(s.laser_viz.pending.empty());
}

// ---------------------------------------------------------------------------
// CalibrationToolCallbacks: all fields callable
// ---------------------------------------------------------------------------

static void test_calib_callbacks_callable() {
    CalibrationToolCallbacks cb;

    // All callbacks should be empty/null by default
    EXPECT_TRUE(!cb.load_images);
    EXPECT_TRUE(!cb.load_videos);
    EXPECT_TRUE(!cb.unload_media);
    EXPECT_TRUE(!cb.copy_default_layout);
    EXPECT_TRUE(!cb.switch_ini);
    EXPECT_TRUE(!cb.print_metadata);

    // Wire up with no-op lambdas and verify they're callable
    int call_count = 0;
    cb.load_images = [&](std::map<std::string, std::string> &) { call_count++; };
    cb.load_videos = [&]() { call_count++; };
    cb.unload_media = [&]() { call_count++; };
    cb.copy_default_layout = [&](const std::string &) { call_count++; };
    cb.switch_ini = [&](const std::string &) { call_count++; };
    cb.print_metadata = [&]() { call_count++; };

    std::map<std::string, std::string> dummy_files;
    cb.load_images(dummy_files);
    cb.load_videos();
    cb.unload_media();
    cb.copy_default_layout("/tmp/test");
    cb.switch_ini("/tmp/test");
    cb.print_metadata();

    EXPECT_TRUE(call_count == 6);
}

// ---------------------------------------------------------------------------
// CalibrationToolState: project initialization pattern
// ---------------------------------------------------------------------------

static void test_calib_state_project_init() {
    CalibrationToolState s;

    // Simulate the initialization pattern from red.cpp main()
    std::string default_root = "/tmp/test_root";
    std::string default_media = "/tmp/test_media";
    s.project.project_root_path = default_root;
    s.project.config_file = default_media;

    EXPECT_TRUE(s.project.project_root_path == default_root);
    EXPECT_TRUE(s.project.config_file == default_media);
    EXPECT_TRUE(s.project.project_name.empty());
    EXPECT_TRUE(s.project.project_path.empty());
}

// ---------------------------------------------------------------------------
// LaserVizState: double-buffer lifecycle
// ---------------------------------------------------------------------------

static void test_laser_viz_double_buffer() {
    LaserVizState lv;

    // Simulate producing results
    std::vector<LaserVizState::CamResult> results(4);
    for (int i = 0; i < 4; i++) {
        results[i].frame_num = 42;
        results[i].num_blobs = (i == 0) ? 1 : 0;
        results[i].total_mask_pixels = i * 10;
    }

    // Simulate background thread completing
    lv.pending = std::move(results);
    EXPECT_TRUE(lv.ready.empty());
    EXPECT_TRUE(lv.pending.size() == 4);

    // Simulate main thread consuming
    lv.ready = std::move(lv.pending);
    lv.pending.clear();
    EXPECT_TRUE(lv.ready.size() == 4);
    EXPECT_TRUE(lv.pending.empty());
    EXPECT_TRUE(lv.ready[0].num_blobs == 1);
    EXPECT_TRUE(lv.ready[0].frame_num == 42);
    EXPECT_TRUE(lv.ready[1].num_blobs == 0);
    EXPECT_TRUE(lv.ready[2].total_mask_pixels == 20);

    // Mark as uploaded
    for (auto &cr : lv.ready) cr.uploaded = false;
    lv.ready[0].uploaded = true;
    EXPECT_TRUE(lv.ready[0].uploaded);
    EXPECT_FALSE(lv.ready[1].uploaded);
}

// ---------------------------------------------------------------------------
// LaserVizState: params-changed detection
// ---------------------------------------------------------------------------

static void test_laser_viz_params_changed() {
    LaserVizState lv;
    lv.last_green_th = 100;
    lv.last_green_dom = 30;
    lv.last_min_blob = 5;
    lv.last_max_blob = 500;

    // Same params: no change
    auto params_changed = [&](int gt, int gd, int minb, int maxb) {
        return gt != lv.last_green_th || gd != lv.last_green_dom ||
               minb != lv.last_min_blob || maxb != lv.last_max_blob;
    };

    EXPECT_FALSE(params_changed(100, 30, 5, 500));
    EXPECT_TRUE(params_changed(101, 30, 5, 500));  // green_threshold changed
    EXPECT_TRUE(params_changed(100, 31, 5, 500));  // green_dominance changed
    EXPECT_TRUE(params_changed(100, 30, 6, 500));  // min_blob changed
    EXPECT_TRUE(params_changed(100, 30, 5, 501));  // max_blob changed
}

// ---------------------------------------------------------------------------
// CalibrationToolState: menu open patterns
// ---------------------------------------------------------------------------

static void test_calib_menu_open_patterns() {
    CalibrationToolState s;

    // "Create Calibration Project" menu item
    s.show = true;
    s.show_create_dialog = true;
    EXPECT_TRUE(s.show);
    EXPECT_TRUE(s.show_create_dialog);
    EXPECT_FALSE(s.project_loaded);

    // After project creation succeeds
    s.project_loaded = true;
    s.dock_pending = true;
    s.show_create_dialog = false;
    EXPECT_TRUE(s.project_loaded);
    EXPECT_TRUE(s.dock_pending);
    EXPECT_FALSE(s.show_create_dialog);

    // After docking completes
    s.dock_pending = false;
    EXPECT_FALSE(s.dock_pending);
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
// main
// ---------------------------------------------------------------------------

int main() {
    test_current_date_time();

    // Calibration tool extraction tests
    test_calib_state_defaults();
    test_calib_state_reset_on_close();
    test_calib_callbacks_callable();
    test_calib_state_project_init();
    test_laser_viz_double_buffer();
    test_laser_viz_params_changed();
    test_calib_menu_open_patterns();

    // Infrastructure tests
    test_deferred_queue_basic();
    test_deferred_queue_thread_safety();
    test_popup_stack_basic();
    test_popup_stack_confirm();
    test_popup_stack_fifo();
    test_toast_queue_basic();
    test_project_handler_registry();

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

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
