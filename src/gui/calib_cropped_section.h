#pragma once
// calib_cropped_section.h — Cropped-Sensor Refinement wizard.
//
// Two-stage workflow for rigs that record a sensor ROI crop:
//   1. Posts at FULL frame: click fixed posts, triangulate them with the
//      stage-1 (full-frame ChArUco) calibration → posts_3d.csv.
//   2. Crop transform: shift the calibration into the crop frame
//      (cx -= OffsetX, cy -= OffsetY, per-camera dims).
//   3. Posts at CROPPED frame: click the same posts, refine intrinsics only
//      (extrinsics + posts locked) → cropped_refined calibration.
// Pattern follows calib_kp_manual_section.h.

#include "calib_tool_state.h"
#include "app_context.h"
#include "calib_kp_manual_section.h"  // collect_manual_landmarks
#include "crop_calibration.h"
#include "cropped_refinement.h"
#include "crop_designer.h"  // crop_snap16
#include "orange_config.h"
#include "annotation_csv.h"
#include "gui/gui_helpers.h"
#include "imgui.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <filesystem>
#include <future>
#include <map>

// Load stage-1 or cropped calibration poses for the project's cameras.
inline bool cropped_load_poses(
    const std::string &folder, const std::vector<std::string> &cam_names,
    std::vector<CalibrationPipeline::CameraPose> &poses, std::string &err) {
    namespace fs = std::filesystem;
    std::string src = CropCalibration::resolve_calibration_folder(folder);
    poses.assign(cam_names.size(), {});
    for (size_t c = 0; c < cam_names.size(); c++) {
        std::string yaml_path = src + "/Cam" + cam_names[c] + ".yaml";
        if (!fs::exists(yaml_path)) {
            err = "Missing calibration file: " + yaml_path;
            return false;
        }
        try {
            auto yf = opencv_yaml::read(yaml_path);
            poses[c].K = yf.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dm = yf.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) poses[c].dist(j) = dm(j, 0);
            poses[c].R = yf.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd tm = yf.getMatrix("tc_ext");
            poses[c].t = Eigen::Vector3d(tm(0, 0), tm(1, 0), tm(2, 0));
        } catch (const std::exception &e) {
            err = "Error reading " + yaml_path + ": " + e.what();
            return false;
        }
    }
    return true;
}

// Discover still images named {serial}_{n}.{ext} for the given serials.
// Honors load_images()'s contract: the suffix after the first '_' must be a
// pure integer (std::stoi is called on it), and all images must share one
// extension — non-conforming files are skipped with a warning.
inline std::map<std::string, std::string> cropped_discover_images(
    const std::string &folder, const std::vector<std::string> &serials,
    std::vector<std::string> &warnings) {
    namespace fs = std::filesystem;
    std::map<std::string, std::string> result;
    if (folder.empty() || !fs::is_directory(folder)) return result;

    std::set<std::string> serial_set(serials.begin(), serials.end());
    const std::vector<std::string> img_exts = {".jpg", ".jpeg", ".png",
                                               ".tiff", ".tif"};
    std::string shared_ext;
    for (const auto &entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::string ext_lower = ext;
        std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (std::find(img_exts.begin(), img_exts.end(), ext_lower) ==
            img_exts.end())
            continue;
        std::string filename = entry.path().filename().string();
        auto underscore = filename.find('_');
        if (underscore == std::string::npos) continue;
        std::string serial = filename.substr(0, underscore);
        if (!serial_set.count(serial)) continue;

        std::string stem = entry.path().stem().string();
        std::string suffix = stem.substr(underscore + 1);
        if (suffix.empty() ||
            suffix.find_first_not_of("0123456789") != std::string::npos) {
            warnings.push_back("Skipping " + filename +
                               ": name after '_' must be a number");
            continue;
        }
        if (shared_ext.empty()) shared_ext = ext;
        if (ext != shared_ext) {
            warnings.push_back("Skipping " + filename + ": mixed extensions (" +
                               ext + " vs " + shared_ext + ")");
            continue;
        }
        result[filename] = entry.path().string();
    }
    return result;
}

inline void DrawCalibCroppedSection(CalibrationToolState &state,
                                    AppContext &ctx,
                                    const CalibrationToolCallbacks &cb) {
    auto &pm = ctx.pm;
    auto &ps = ctx.ps;
    auto *scene = ctx.scene;
    auto &imgs_names = ctx.imgs_names;
#ifdef __APPLE__
    auto &mac_last_uploaded_frame = ctx.mac_last_uploaded_frame;
#endif
    auto &cs = state.cropped;
    auto &proj = state.project;

    auto save_project_now = [&]() {
        std::string proj_file = proj.project_path + "/" + proj.project_name +
                                ".redproj";
        std::string save_err;
        CalibrationTool::save_project(proj, proj_file, &save_err);
    };

    // Poll async refinement
    if (cs.refine_running && cs.refine_future.valid()) {
        auto fstat = cs.refine_future.wait_for(std::chrono::milliseconds(0));
        if (fstat == std::future_status::ready) {
            cs.refine_result = cs.refine_future.get();
            cs.refine_running = false;
            cs.refine_done = true;
            if (cs.refine_result.success) {
                cs.status = "Refinement done: " +
                    std::to_string(cs.refine_result.mean_before).substr(0, 5) +
                    " -> " +
                    std::to_string(cs.refine_result.mean_after).substr(0, 5) +
                    " px";
                proj.cropped_refined_folder = cs.refine_result.output_folder;
                save_project_now();
            } else {
                cs.status = "Refinement failed: " + cs.refine_result.error;
            }
        }
    }

    // Poll async crop verification
    if (cs.verify_running && cs.verify_future.valid()) {
        auto fstat = cs.verify_future.wait_for(std::chrono::milliseconds(0));
        if (fstat == std::future_status::ready) {
            cs.verify_result = cs.verify_future.get();
            cs.verify_running = false;
            cs.verify_done = true;
            cs.status = cs.verify_result.success
                ? ("Verification: " +
                   std::to_string(cs.verify_result.mean_after).substr(0, 6) +
                   " px mean reproj")
                : ("Verification failed: " + cs.verify_result.error);
        }
    }

    // Shared deferred media loader (posts recorded as videos OR still
    // images — auto-detected from the folder contents).
    auto load_posts_media = [&](const std::string &folder, bool cropped_stage) {
        cb.deferred->enqueue([&state, &pm, &ps, &cb, &imgs_names, &ctx, scene,
                              folder, cropped_stage
#ifdef __APPLE__
                              , &mac_last_uploaded_frame
#endif
        ]() {
            auto &cs2 = state.cropped;
            auto info = CalibrationTool::detect_aruco_media(folder);
            if (info.type.empty()) {
                cs2.status = "No Cam*.mp4 videos or {serial}_{n} images found "
                             "in " + folder;
                return;
            }

            if (ps.video_loaded) cb.unload_media();
            imgs_names.clear();
#ifdef __APPLE__
            for (size_t ci = 0; ci < mac_last_uploaded_frame.size(); ci++)
                mac_last_uploaded_frame[ci] = -1;
#endif
            pm.media_folder = folder;
            pm.camera_names.clear();

            if (info.type == "videos") {
                for (const auto &cn : state.project.camera_names)
                    pm.camera_names.push_back("Cam" + cn);
                cb.load_videos();
            } else {
                // Still images: load_images derives camera names from the
                // {serial}_{n}.{ext} filenames (alphabetical == sorted
                // serials, matching project camera order).
                std::vector<std::string> warns;
                auto files = cropped_discover_images(
                    folder, state.project.camera_names, warns);
                if (files.empty()) {
                    cs2.status = "No usable {serial}_{n} images matched the "
                                 "project cameras in " + folder;
                    return;
                }
                cb.load_images(files);
                for (const auto &w : warns)
                    fprintf(stderr, "[cropped wizard] %s\n", w.c_str());
                if ((int)pm.camera_names.size() !=
                    (int)state.project.camera_names.size())
                    cs2.status = "WARNING: images found for " +
                        std::to_string(pm.camera_names.size()) + " of " +
                        std::to_string(state.project.camera_names.size()) +
                        " project cameras";
            }
            cb.print_metadata();

            if (cropped_stage) {
                cs2.cropped_videos_loaded = true;
                cs2.cropped_skeleton_ready = false;
            } else {
                cs2.fullframe_videos_loaded = true;
                cs2.fullframe_skeleton_ready = false;
            }
            if (cs2.status.empty() || cs2.status.rfind("WARNING", 0) != 0)
                cs2.status = std::string(info.type == "videos" ? "Videos"
                                                               : "Images") +
                    " loaded (" + std::to_string(pm.camera_names.size()) +
                    " cameras)";
        });
    };

    // Set up the posts skeleton; labels_subdir separates full-frame from
    // cropped clicks so they never mix.
    auto setup_posts_skeleton = [&](const char *labels_subdir,
                                    bool clear_labels) {
        setup_landmark_skeleton(ctx.skeleton, proj.posts_num, pm,
                                proj.project_path);
        pm.keypoints_root_folder =
            (std::filesystem::path(proj.project_path) / labels_subdir)
                .string();
        std::error_code ec;
        std::filesystem::create_directories(pm.keypoints_root_folder, ec);
        if (clear_labels) ctx.annotations.erase((u32)0);
        auto &fa = ctx.annotations[0];
        if ((int)fa.cameras.size() < (int)scene->num_cams)
            fa.cameras.resize(scene->num_cams);
        for (int c = 0; c < (int)scene->num_cams; c++)
            if ((int)fa.cameras[c].keypoints.size() < proj.posts_num)
                fa.cameras[c].keypoints.resize(proj.posts_num);
        if ((int)fa.kp3d.size() < proj.posts_num)
            fa.kp3d.resize(proj.posts_num);
    };

    // Load previously saved post labels: newest timestamped save under the
    // step's labels dir ({project}/{labels_subdir}/{timestamp}/). Returns a
    // status string. Sets up the skeleton first so clicks land in place.
    auto load_posts_labels = [&](const char *labels_subdir) -> std::string {
        namespace fs = std::filesystem;
        setup_posts_skeleton(labels_subdir, false);
        std::string latest;
        std::error_code ec;
        for (const auto &e : fs::directory_iterator(pm.keypoints_root_folder, ec)) {
            if (!e.is_directory()) continue;
            std::string name = e.path().filename().string();
            if (fs::exists(e.path() / "keypoints3d.csv") && name > latest)
                latest = name;
        }
        if (latest.empty())
            return "No saved labels found in " + pm.keypoints_root_folder;
        std::string folder = pm.keypoints_root_folder + "/" + latest;
        std::string err;
        if (AnnotationCSV::load_all(folder, ctx.annotations, ctx.skeleton.name,
                                    ctx.skeleton.num_nodes, scene->num_cams,
                                    pm.camera_names, err) != 0)
            return "Load failed: " + err;
        return "Labels loaded from " + folder;
    };

    // Harvest clicked posts (ImPlot y-up -> OpenCV y-down, per-camera heights)
    auto collect_posts = [&]() {
        std::vector<int> img_hs(scene->num_cams);
        for (u32 c = 0; c < scene->num_cams; c++)
            img_hs[c] = (int)scene->image_height[c];
        return collect_manual_landmarks(ctx.annotations, 0, proj.posts_num,
                                        scene->num_cams, proj.camera_names,
                                        img_hs);
    };

    // Label-count summary line
    auto label_summary = [&]() {
        int total_labeled = 0, cams_with_labels = 0;
        auto it = ctx.annotations.find(0);
        if (it != ctx.annotations.end()) {
            const auto &fa = it->second;
            for (int c = 0;
                 c < (int)scene->num_cams && c < (int)fa.cameras.size(); c++) {
                int n = 0;
                for (int k = 0; k < proj.posts_num &&
                                k < (int)fa.cameras[c].keypoints.size(); k++)
                    if (fa.cameras[c].keypoints[k].labeled) n++;
                if (n > 0) cams_with_labels++;
                total_labeled += n;
            }
        }
        ImGui::Text("Labels: %d posts across %d cameras", total_labeled,
                    cams_with_labels);
    };

    if (!ImGui::CollapsingHeader("Cropped-Sensor Refinement",
                                 ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::Indent();

    // ---- Stage-1 calibration source ----
    // Everything in phase 2 (post triangulation, crop transform, verify)
    // consumes this. Selectable here so a specific stage-1 run can be chosen
    // (the default resolution picks the LATEST timestamped subfolder).
    {
        ImGui::Text("Stage-1 calibration:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 160.0f);
        if (ImGui::InputText("##crop_stage1_dir", &proj.calibration_folder))
            save_project_now();
        ImGui::SameLine();
        if (ImGui::Button("Browse##crop_stage1")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            if (!proj.calibration_folder.empty())
                cfg.path = proj.calibration_folder;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCropStage1Dir",
                "Select Stage-1 Calibration (run folder or aruco root)",
                nullptr, cfg);
        }
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseCropStage1Dir", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                proj.calibration_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                save_project_now();
            }
            ImGuiFileDialog::Instance()->Close();
        }
        ImGui::SameLine();
        HelpMarker("The full-frame calibration phase 2 builds on: post\n"
                   "triangulation, the crop transform, and verification all\n"
                   "read Cam{serial}.yaml from here.\nAccepts a specific\n"
                   "timestamped run folder or the aruco output root (the\n"
                   "LATEST run is used then). Changing it after applying a\n"
                   "crop requires re-applying the crop.");
        if (!proj.calibration_folder.empty()) {
            std::string resolved = CropCalibration::resolve_calibration_folder(
                proj.calibration_folder);
            int found = 0;
            for (const auto &cam : proj.camera_names)
                if (std::filesystem::exists(resolved + "/Cam" + cam + ".yaml"))
                    found++;
            bool ok = found == (int)proj.camera_names.size() &&
                      !proj.camera_names.empty();
            ImVec4 col = ok ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f)
                            : ImVec4(1.0f, 0.6f, 0.2f, 1.0f);
            ImGui::TextColored(col, "Using: %s (%d/%d camera YAMLs)",
                               resolved.c_str(), found,
                               (int)proj.camera_names.size());
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                               "No stage-1 calibration selected");
        }
    }
    ImGui::Separator();

    // ════ Step 1: Posts at full frame ════
    if (ImGui::CollapsingHeader("1. Posts — Full Frame##crop",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();
        ImGui::TextWrapped(
            "Record the fixed posts at FULL sensor resolution (glass "
            "installed) — videos or still images ({serial}_{n}.png) — click "
            "each post in each view, and triangulate with the stage-1 "
            "calibration.");

        ImGui::Text("Media Folder:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 160.0f);
        ImGui::InputText("##crop_ff_vid", &proj.posts_fullframe_media_folder);
        ImGui::SameLine();
        if (ImGui::Button("Browse##crop_ff_vid")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            if (!proj.posts_fullframe_media_folder.empty())
                cfg.path = proj.posts_fullframe_media_folder;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCropFFVideoFolder", "Select Full-Frame Posts Videos",
                nullptr, cfg);
        }
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseCropFFVideoFolder", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk())
                proj.posts_fullframe_media_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::SameLine();
        ImGui::BeginDisabled(proj.posts_fullframe_media_folder.empty());
        if (ImGui::Button("Load Media##crop_ff")) {
            load_posts_media(proj.posts_fullframe_media_folder, false);
            save_project_now();
        }
        ImGui::EndDisabled();
        if (cs.fullframe_videos_loaded) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Loaded");
        }

        ImGui::SliderInt("Num Posts##crop", &proj.posts_num, 3, 20);
        ImGui::SameLine();
        HelpMarker("Each post gets a numbered, colored skeleton node.\n"
                   "Click posts in the SAME ORDER in every view — and again\n"
                   "in the same order in the cropped views (step 3).");

        ImGui::BeginDisabled(!cs.fullframe_videos_loaded);
        if (ImGui::Button("Setup Posts##crop_ff")) {
            setup_posts_skeleton("labeled_data_fullframe", false);
            cs.fullframe_skeleton_ready = true;
            cs.status = "Skeleton created. Click " +
                std::to_string(proj.posts_num) + " posts in each view.";
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Labels##crop_ff_load")) {
            cs.status = load_posts_labels("labeled_data_fullframe");
            if (cs.status.rfind("Labels loaded", 0) == 0)
                cs.fullframe_skeleton_ready = true;
        }
        ImGui::SameLine();
        HelpMarker("Loads the most recent Save Labels snapshot from\n"
                   "{project}/labeled_data_fullframe/ — no re-clicking\n"
                   "needed when re-triangulating against a new stage-1\n"
                   "calibration.");
        ImGui::EndDisabled();
        if (cs.fullframe_skeleton_ready) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                               "Ready to label");
        }

        if (cs.fullframe_skeleton_ready && scene->num_cams > 0) {
            label_summary();

            if (ImGui::Button("Save Labels##crop_ff")) {
                std::string save_err;
                std::string saved = AnnotationCSV::save_all(
                    pm.keypoints_root_folder, ctx.skeleton.name,
                    ctx.annotations, scene->num_cams, ctx.skeleton.num_nodes,
                    pm.camera_names, &save_err);
                cs.status = saved.empty() ? ("Save failed: " + save_err)
                                          : ("Labels saved to " + saved);
            }
            ImGui::SameLine();

            ImGui::BeginDisabled(proj.calibration_folder.empty());
            if (ImGui::Button("Triangulate Posts##crop_ff")) {
                std::vector<CalibrationPipeline::CameraPose> poses;
                std::string err;
                if (!cropped_load_poses(proj.calibration_folder,
                                        proj.camera_names, poses, err)) {
                    cs.status = "Error: " + err;
                } else {
                    auto landmarks = collect_posts();
                    std::map<int, Eigen::Vector3d> posts_3d;
                    // Flag clicks beyond the stage-1 calibration's observed
                    // radius (distortion is extrapolated there).
                    auto coverage = CropCalibration::load_coverage_radii(
                        proj.calibration_folder, proj.camera_names, poses);
                    cs.post_report =
                        CropCalibration::triangulate_posts_with_report(
                            proj.camera_names, landmarks, poses,
                            cs.post_accept_th, posts_3d, coverage);
                    std::string csv = proj.project_path + "/posts_3d.csv";
                    std::string werr;
                    if (!CropCalibration::write_posts_3d_csv(
                            csv, proj.posts_num, posts_3d, &werr)) {
                        cs.status = "Error: " + werr;
                    } else {
                        CropCalibration::write_posts_report_json(
                            proj.project_path + "/posts_3d_report.json",
                            cs.post_report);
                        proj.posts_3d_file = csv;
                        cs.posts_triangulated = true;
                        int accepted = 0;
                        for (const auto &r : cs.post_report)
                            if (r.accepted) accepted++;
                        cs.status = std::to_string(accepted) + "/" +
                            std::to_string(proj.posts_num) +
                            " posts triangulated OK -> " + csv;
                        save_project_now();
                    }
                }
            }
            ImGui::EndDisabled();
        }

        // Per-post quality table
        if (!cs.post_report.empty()) {
            if (ImGui::BeginTable("crop_post_tab", 5,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                    ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Post", 0, 50.0f);
                ImGui::TableSetupColumn("Views", 0, 50.0f);
                ImGui::TableSetupColumn("Mean (px)", 0, 80.0f);
                ImGui::TableSetupColumn("Max (px)", 0, 80.0f);
                ImGui::TableSetupColumn("Extrapolated", 0, 150.0f);
                ImGui::TableHeadersRow();
                for (const auto &r : cs.post_report) {
                    ImGui::TableNextRow();
                    ImVec4 col = r.accepted
                        ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                        : ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextColored(col, "%d", r.id);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%d", r.n_views);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::TextColored(col, "%.2f", r.mean_reproj);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.2f", r.max_reproj);
                    ImGui::TableSetColumnIndex(4);
                    if (!r.extrapolated_cams.empty()) {
                        std::string s;
                        for (const auto &c : r.extrapolated_cams)
                            s += (s.empty() ? "" : " ") + c;
                        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                                           "%s", s.c_str());
                    } else {
                        ImGui::TextDisabled("-");
                    }
                }
                ImGui::EndTable();
            }
            ImGui::TextDisabled(
                "Red rows: bad clicks or <2 views — fix and re-triangulate.\n"
                "Extrapolated: click lies beyond the stage-1 calibration's "
                "board coverage — distortion is unconstrained there; expect "
                "elevated error for that camera, not a bad click.");
        }
        ImGui::Unindent();
    }

    // ════ Step 2: Crop transform ════
    bool have_posts = cs.posts_triangulated || !proj.posts_3d_file.empty();
    if (ImGui::CollapsingHeader("2. Crop Transform##crop")) {
        ImGui::Indent();
        ImGui::TextWrapped(
            "Design the sensor ROI: one crop size for all cameras, positioned "
            "per camera — drag the rectangle on each camera view, or type "
            "offsets below. The stage-1 calibration is then shifted into the "
            "crop frame: cx -= OffsetX, cy -= OffsetY.");

        // Seed / sync spec rows with project cameras
        if (cs.crop_spec.cameras.size() != proj.camera_names.size()) {
            CropCalibration::CropSpec seeded;
            for (const auto &cn : proj.camera_names) {
                CropCalibration::CameraCrop c;
                c.serial = cn;
                for (const auto &old : cs.crop_spec.cameras)
                    if (old.serial == cn) c = old;
                seeded.cameras.push_back(c);
            }
            cs.crop_spec.cameras = seeded.cameras;
        }

        // Write the shared crop dims into every camera row (snapped/clamped).
        auto apply_shared_dims = [&]() {
            cs.crop_w = std::max(16, crop_snap16(cs.crop_w));
            cs.crop_h = std::max(16, crop_snap16(cs.crop_h));
            for (auto &row : cs.crop_spec.cameras) {
                row.width = cs.crop_w;
                row.height = cs.crop_h;
                row.offset_x = std::max(0, crop_snap16(row.offset_x));
                row.offset_y = std::max(0, crop_snap16(row.offset_y));
            }
        };

        // ---- Interactive designer ----
        if (ImGui::Checkbox("Design crop on camera views##crop",
                            &cs.designer_enabled)) {
            if (cs.designer_enabled) apply_shared_dims();
        }
        ImGui::SameLine();
        HelpMarker("Overlays a draggable, fixed-size rectangle on every "
                   "camera view — this is exactly the region orange will "
                   "record with the exported ROI.\nValues snap to multiples "
                   "of 16 (orange's convention). Posts outside a rectangle "
                   "are counted in the overlay label.");

        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::InputInt("Crop W##crop", &cs.crop_w, 16))
            apply_shared_dims();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::InputInt("Crop H##crop", &cs.crop_h, 16))
            apply_shared_dims();
        ImGui::SameLine();
        ImGui::BeginDisabled(scene->num_cams == 0);
        if (ImGui::Button("Auto-center on posts##crop")) {
            apply_shared_dims();
            auto it = ctx.annotations.find(0);
            for (u32 c = 0; c < scene->num_cams &&
                            c < cs.crop_spec.cameras.size(); c++) {
                if (it == ctx.annotations.end() ||
                    c >= it->second.cameras.size())
                    continue;
                const auto &kps = it->second.cameras[c].keypoints;
                double sx = 0, sy = 0;
                int n = 0;
                for (int k = 0; k < proj.posts_num && k < (int)kps.size();
                     k++) {
                    if (!kps[k].labeled || kps[k].x >= UNLABELED * 0.9)
                        continue;
                    sx += kps[k].x;
                    sy += (double)scene->image_height[c] - kps[k].y;  // y-down
                    n++;
                }
                if (n == 0) continue;
                auto &row = cs.crop_spec.cameras[c];
                int img_w = (int)scene->image_width[c];
                int img_h = (int)scene->image_height[c];
                row.offset_x = std::clamp(
                    crop_snap16((int)std::lround(sx / n - cs.crop_w / 2.0)),
                    0, std::max(0, img_w - cs.crop_w));
                row.offset_y = std::clamp(
                    crop_snap16((int)std::lround(sy / n - cs.crop_h / 2.0)),
                    0, std::max(0, img_h - cs.crop_h));
            }
            cs.status = "Crops centered on clicked posts";
        }
        ImGui::EndDisabled();

        if (ImGui::Button("Import crop_info.json##crop")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            if (!proj.crop_info_file.empty())
                cfg.path = std::filesystem::path(proj.crop_info_file)
                               .parent_path().string();
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCropInfoImport", "Choose crop_info.json", ".json", cfg);
        }
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseCropInfoImport", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string path =
                    ImGuiFileDialog::Instance()->GetFilePathName();
                CropCalibration::CropSpec spec;
                std::string err;
                if (CropCalibration::load_crop_spec(path, spec, &err)) {
                    // Merge by serial so row order follows the project
                    for (auto &row : cs.crop_spec.cameras)
                        for (const auto &imp : spec.cameras)
                            if (imp.serial == row.serial) row = imp;
                    // Shared dims follow the first imported camera
                    if (!spec.cameras.empty()) {
                        cs.crop_w = spec.cameras[0].width;
                        cs.crop_h = spec.cameras[0].height;
                    }
                    proj.crop_info_file = path;
                    cs.status = "Imported crop spec from " + path;
                    save_project_now();
                } else {
                    cs.status = "Error: " + err;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!cs.cropped_videos_loaded);
        if (ImGui::Button("Fill W/H from cropped media##crop")) {
            if (scene->num_cams > 0) {
                cs.crop_w = (int)scene->image_width[0];
                cs.crop_h = (int)scene->image_height[0];
                apply_shared_dims();
            }
        }
        ImGui::EndDisabled();

        if (ImGui::BeginTable("crop_spec_tab", 5,
                ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                ImGuiTableFlags_SizingFixedFit)) {
            ImGui::TableSetupColumn("Camera", 0, 100.0f);
            ImGui::TableSetupColumn("OffsetX", 0, 90.0f);
            ImGui::TableSetupColumn("OffsetY", 0, 90.0f);
            ImGui::TableSetupColumn("Width", 0, 90.0f);
            ImGui::TableSetupColumn("Height", 0, 90.0f);
            ImGui::TableHeadersRow();
            for (size_t c = 0; c < cs.crop_spec.cameras.size(); c++) {
                auto &row = cs.crop_spec.cameras[c];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Cam%s", row.serial.c_str());
                ImGui::PushID((int)c);
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::InputInt("##ox", &row.offset_x, 0))
                    row.offset_x = std::max(0, crop_snap16(row.offset_x));
                ImGui::TableSetColumnIndex(2);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::InputInt("##oy", &row.offset_y, 0))
                    row.offset_y = std::max(0, crop_snap16(row.offset_y));
                // Shared dims: display only
                ImGui::TableSetColumnIndex(3);
                ImGui::TextDisabled("%d", row.width);
                ImGui::TableSetColumnIndex(4);
                ImGui::TextDisabled("%d", row.height);
                ImGui::PopID();
            }
            ImGui::EndTable();
        }

        // ---- Orange config export ----
        ImGui::Separator();
        ImGui::Text("Orange configs:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 260.0f);
        ImGui::InputText("##crop_orange_dir", &proj.orange_config_folder);
        ImGui::SameLine();
        if (ImGui::Button("Browse##crop_orange")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            if (!proj.orange_config_folder.empty())
                cfg.path = proj.orange_config_folder;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseOrangeConfigDir", "Select Orange Config Folder",
                nullptr, cfg);
        }
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseOrangeConfigDir", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                proj.orange_config_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
                save_project_now();
            }
            ImGuiFileDialog::Instance()->Close();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(proj.orange_config_folder.empty() ||
                             cs.crop_spec.cameras.empty());
        if (ImGui::Button("Export ROI##crop_orange")) {
            apply_shared_dims();
            std::vector<int> full_ws(scene->num_cams), full_hs(scene->num_cams);
            for (u32 c = 0; c < scene->num_cams; c++) {
                full_ws[c] = (int)scene->image_width[c];
                full_hs[c] = (int)scene->image_height[c];
            }
            auto er = OrangeConfig::update_orange_configs(
                proj.orange_config_folder, cs.crop_spec, full_ws, full_hs);
            if (er.success) {
                cs.orange_status = "Updated " + std::to_string(er.updated) +
                    " orange config(s) in " + proj.orange_config_folder;
                save_project_now();
            } else {
                cs.orange_status = "Export failed: " + er.error;
            }
            for (const auto &w : er.warnings)
                cs.orange_status += "\n" + w;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(proj.orange_config_folder.empty() ||
                             cs.crop_spec.cameras.empty());
        if (ImGui::Button("Import ROI##crop_orange")) {
            std::vector<std::string> serials;
            for (const auto &row : cs.crop_spec.cameras)
                serials.push_back(row.serial);
            auto ir = OrangeConfig::import_orange_configs(
                proj.orange_config_folder, serials);
            if (ir.success) {
                // Merge by serial so row order follows the project
                for (auto &row : cs.crop_spec.cameras)
                    for (const auto &imp : ir.cameras)
                        if (imp.serial == row.serial) row = imp;
                // Shared dims follow the first imported camera; differing
                // per-camera dims in the configs are surfaced as a warning
                // (the designer enforces one shared crop size).
                cs.crop_w = ir.cameras[0].width;
                cs.crop_h = ir.cameras[0].height;
                for (const auto &imp : ir.cameras)
                    if (imp.width != cs.crop_w || imp.height != cs.crop_h)
                        ir.warnings.push_back(
                            "Camera " + imp.serial + " config is " +
                            std::to_string(imp.width) + "x" +
                            std::to_string(imp.height) + " — shared dims use " +
                            std::to_string(cs.crop_w) + "x" +
                            std::to_string(cs.crop_h));
                apply_shared_dims();
                cs.orange_status = "Imported " +
                    std::to_string(ir.cameras.size()) + " ROI(s) from " +
                    proj.orange_config_folder;
                save_project_now();
            } else {
                cs.orange_status = "Import failed: " + ir.error;
            }
            for (const auto &w : ir.warnings)
                cs.orange_status += "\n" + w;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        HelpMarker("Export rewrites width/height/offsetx/offsety in each "
                   "{serial}.json, preserving all other keys.\nImport loads "
                   "them back into the crop table (offsets default to 0 when "
                   "a config has no offset keys).\nNOTE: orange currently "
                   "only applies width/height (offsets are hardcoded to 0) — "
                   "it must be patched to read offsetx/offsety.");
        if (!cs.orange_status.empty())
            ImGui::TextWrapped("%s", cs.orange_status.c_str());

        // ---- Apply / verify ----
        ImGui::Separator();
        bool spec_ok = !cs.crop_spec.cameras.empty();
        for (const auto &c : cs.crop_spec.cameras)
            if (c.width <= 0 || c.height <= 0) spec_ok = false;

        auto do_apply_crop = [&]() -> bool {
            apply_shared_dims();
            std::string out = proj.project_path + "/cropped_calibration";
            auto cr = CropCalibration::apply_crop_to_calibration(
                proj.calibration_folder, cs.crop_spec, out);
            cs.crop_warnings = cr.warnings;
            if (cr.success) {
                cs.crop_applied = true;
                proj.cropped_calibration_folder = out;
                proj.crop_info_file = out + "/crop_info.json";
                cs.status = "Cropped calibration written to " + out;
                save_project_now();
                return true;
            }
            cs.status = "Error: " + cr.error;
            return false;
        };

        ImGui::BeginDisabled(!spec_ok || proj.calibration_folder.empty());
        if (ImGui::Button("Apply Crop##crop")) do_apply_crop();
        ImGui::EndDisabled();

        // Apply + Verify: shift the full-frame post clicks into the crop and
        // run the stage-2 refinement — residuals should be ~0 (same optics,
        // same session), so this validates the transform + the crop design.
        ImGui::SameLine();
        bool can_verify = spec_ok && !proj.calibration_folder.empty() &&
                          cs.fullframe_skeleton_ready &&
                          !proj.posts_3d_file.empty() && !cs.verify_running;
        ImGui::BeginDisabled(!can_verify);
        if (ImGui::Button("Apply Crop + Verify##crop")) {
            if (do_apply_crop()) {
                auto full_lm = collect_posts();
                cs.verify_dropped.clear();
                auto crop_lm = CropCalibration::shift_landmarks_to_crop(
                    full_lm, cs.crop_spec, cs.verify_dropped);

                CroppedRefinement::RefineConfig cfg;
                cfg.calibration_folder = proj.cropped_calibration_folder;
                cfg.posts_3d_csv = proj.posts_3d_file;
                cfg.output_folder = proj.project_path + "/cropped_verified";
                cfg.camera_names = proj.camera_names;
                cfg.free_focal = false;

                cs.verify_running = true;
                cs.verify_done = false;
                cs.status = "Verifying cropped calibration...";
                cs.verify_future = std::async(
                    std::launch::async, [cfg, lm = crop_lm]() {
                        return CroppedRefinement::run_cropped_refinement(cfg,
                                                                         lm);
                    });
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        HelpMarker("Requires step 1 (clicked posts + posts_3d.csv). Maps the "
                   "full-frame clicks into the crop and re-projects — "
                   "residuals near 0 px confirm the cropped calibration is "
                   "consistent. Posts outside a crop are reported.");
        if (cs.verify_running) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Verifying...");
        }

        if (cs.crop_applied || !proj.cropped_calibration_folder.empty()) {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                               "Cropped calibration ready");
        }
        for (const auto &w : cs.crop_warnings)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s",
                               w.c_str());

        // Verification results
        if (cs.verify_done && cs.verify_result.success) {
            const auto &vr = cs.verify_result;
            bool ok = vr.mean_after < 0.5;
            ImGui::TextColored(ok ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                                  : ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                "Verification: %.4f px mean reproj (%d obs)%s",
                vr.mean_after, vr.obs_used,
                ok ? " — consistent" : " — check crop/labels!");
            for (const auto &s : vr.per_camera)
                if (std::abs(s.d_cx) > 0.5 || std::abs(s.d_cy) > 0.5)
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                        "  Cam%s: principal moved %.2f,%.2f px in "
                        "verification — unexpected",
                        s.serial.c_str(), s.d_cx, s.d_cy);
        } else if (cs.verify_done && !cs.verify_result.success) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                "Verification failed: %s", cs.verify_result.error.c_str());
        }
        for (const auto &d : cs.verify_dropped)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s",
                               d.c_str());
        ImGui::Unindent();
    }

    // ════ Step 3: Posts at cropped frame + refinement ════
    bool have_cropped_calib = !proj.cropped_calibration_folder.empty();
    if (ImGui::CollapsingHeader("3. Posts — Cropped + Refine##crop")) {
        ImGui::Indent();
        if (!have_posts || !have_cropped_calib)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "Complete steps 1 (posts_3d.csv) and 2 (cropped calibration) "
                "first.");

        ImGui::Text("Video Folder:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 160.0f);
        ImGui::InputText("##crop_cr_vid", &proj.cropped_media_folder);
        ImGui::SameLine();
        if (ImGui::Button("Browse##crop_cr_vid")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            if (!proj.cropped_media_folder.empty())
                cfg.path = proj.cropped_media_folder;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseCropCrVideoFolder", "Select Cropped Posts Videos",
                nullptr, cfg);
        }
        if (ImGuiFileDialog::Instance()->Display(
                "ChooseCropCrVideoFolder", ImGuiWindowFlags_NoCollapse,
                ImVec2(680, 440))) {
            if (ImGuiFileDialog::Instance()->IsOk())
                proj.cropped_media_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::SameLine();
        ImGui::BeginDisabled(proj.cropped_media_folder.empty());
        if (ImGui::Button("Load Media##crop_cr")) {
            load_posts_media(proj.cropped_media_folder, true);
            save_project_now();
        }
        ImGui::EndDisabled();
        if (cs.cropped_videos_loaded) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Loaded");
        }

        ImGui::BeginDisabled(!cs.cropped_videos_loaded);
        if (ImGui::Button("Setup Posts##crop_cr")) {
            // Clear step-1 clicks: full-frame coordinates are wrong here.
            setup_posts_skeleton("labeled_data_cropped", true);
            cs.cropped_skeleton_ready = true;
            cs.status = "Skeleton created. Click the SAME " +
                std::to_string(proj.posts_num) +
                " posts in the SAME ORDER as step 1.";
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Labels##crop_cr_load")) {
            cs.status = load_posts_labels("labeled_data_cropped");
            if (cs.status.rfind("Labels loaded", 0) == 0)
                cs.cropped_skeleton_ready = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        HelpMarker("Post identity is the node index: post 3 in step 1 must\n"
                   "be clicked as node 3 here too. Node colors/names match.\n"
                   "Load Labels restores the most recent Save Labels snapshot\n"
                   "from {project}/labeled_data_cropped/ (cropped-frame\n"
                   "coordinates — do NOT load step-1 labels here).");
        if (cs.cropped_skeleton_ready) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                               "Ready to label");
        }

        if (cs.cropped_skeleton_ready && scene->num_cams > 0) {
            label_summary();
            if (ImGui::Button("Save Labels##crop_cr")) {
                std::string save_err;
                std::string saved = AnnotationCSV::save_all(
                    pm.keypoints_root_folder, ctx.skeleton.name,
                    ctx.annotations, scene->num_cams, ctx.skeleton.num_nodes,
                    pm.camera_names, &save_err);
                cs.status = saved.empty() ? ("Save failed: " + save_err)
                                          : ("Labels saved to " + saved);
            }
        }

        ImGui::Separator();
        ImGui::Checkbox("Free focal length##crop", &cs.refine_free_focal);
        ImGui::SameLine();
        HelpMarker("Default refines only the principal point (cx, cy) — "
                   "absorbs pointing drift.\nFreeing focal also absorbs the "
                   "plexiglass apparent-depth shift; needs more posts.");
        ImGui::SliderFloat("Principal prior##crop", &cs.refine_prior_principal,
                           0.001f, 1.0f, "%.3f",
                           ImGuiSliderFlags_Logarithmic);
        if (cs.refine_free_focal)
            ImGui::SliderFloat("Focal prior##crop", &cs.refine_prior_focal,
                               0.0001f, 2.0f, "%.4f",
                               ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Outlier threshold (px)##crop",
                           &cs.refine_outlier_th, 5.0f, 100.0f, "%.0f");
        ImGui::SliderFloat("Holdout fraction##crop", &cs.refine_holdout, 0.0f,
                           0.5f, "%.2f");

        bool can_refine = cs.cropped_skeleton_ready && have_posts &&
                          have_cropped_calib && !cs.refine_running;
        ImGui::BeginDisabled(!can_refine);
        if (ImGui::Button("Run Refinement##crop")) {
            cs.landmarks = collect_posts();

            CroppedRefinement::RefineConfig cfg;
            cfg.calibration_folder = proj.cropped_calibration_folder;
            cfg.posts_3d_csv = proj.posts_3d_file;
            cfg.output_folder = proj.project_path + "/cropped_refined";
            cfg.camera_names = proj.camera_names;
            cfg.free_focal = cs.refine_free_focal;
            cfg.prior_principal_weight = cs.refine_prior_principal;
            cfg.prior_focal_weight = cs.refine_prior_focal;
            cfg.outlier_th = cs.refine_outlier_th;
            cfg.holdout_fraction = cs.refine_holdout;

            cs.refine_running = true;
            cs.refine_done = false;
            cs.status = "Running cropped refinement...";
            cs.refine_future = std::async(
                std::launch::async,
                [cfg, lm = cs.landmarks, status_ptr = &cs.status]() {
                    return CroppedRefinement::run_cropped_refinement(
                        cfg, lm, status_ptr);
                });
        }
        ImGui::EndDisabled();
        if (cs.refine_running) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Running...");
        }
        ImGui::Unindent();
    }

    // ════ Step 4: Results ════
    if (cs.refine_done && cs.refine_result.success) {
        if (ImGui::CollapsingHeader("4. Results##crop",
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();
            const auto &rr = cs.refine_result;

            ImGui::Text("Before: %.3f px  ->  After: %.3f px "
                        "(%d obs, %d dropped)",
                        rr.mean_before, rr.mean_after, rr.obs_used,
                        rr.obs_dropped);
            if (rr.holdout_obs > 0) {
                double ratio = rr.mean_after > 1e-9
                    ? rr.holdout_after / rr.mean_after : 0.0;
                ImVec4 color = ratio > 1.5
                    ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
                    : ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
                ImGui::TextColored(color,
                    "Holdout: %.3f px (%.1fx train, %d obs) %s",
                    rr.holdout_after, ratio, rr.holdout_obs,
                    ratio > 1.5 ? "OVERFITTING" : "OK");
            }

            if (ImGui::BeginTable("crop_res_tab", 6,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders |
                    ImGuiTableFlags_SizingFixedFit)) {
                ImGui::TableSetupColumn("Camera", 0, 100.0f);
                ImGui::TableSetupColumn("Obs", 0, 45.0f);
                ImGui::TableSetupColumn("Before", 0, 70.0f);
                ImGui::TableSetupColumn("After", 0, 70.0f);
                ImGui::TableSetupColumn("dcx / dcy", 0, 110.0f);
                ImGui::TableSetupColumn("df", 0, 70.0f);
                ImGui::TableHeadersRow();
                for (const auto &s : rr.per_camera) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Cam%s", s.serial.c_str());
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%d", s.n_obs);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.3f", s.reproj_before);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.3f", s.reproj_after);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%+.2f / %+.2f", s.d_cx, s.d_cy);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%+.2f", s.d_f);
                }
                ImGui::EndTable();
            }

            for (const auto &w : rr.warnings)
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s",
                                   w.c_str());

            if (ImGui::Button("Open 3D Viewer##crop")) {
                state.loaded_result =
                    SuperPointRefinement::load_calib_result_from_folder(
                        rr.output_folder, proj.camera_names);
                state.loaded_result.mean_reproj_error = rr.mean_after;
                state.calib_viewer.result = &state.loaded_result;
                state.calib_viewer.show = true;
                state.calib_viewer.cached_selection = -2;
            }

            ImGui::Unindent();
        }
    }

    // ---- Status ----
    if (!cs.status.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", cs.status.c_str());
    }

    ImGui::Unindent();
    ImGui::Spacing();
}
