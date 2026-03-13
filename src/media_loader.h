#pragma once
#include "global.h"
#include "project.h"
#include "render.h"
#include "utils.h"
#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include "../lib/ImGuiFileDialog/stb/stb_image.h"
#endif

// Tear down existing media (decoder threads, demuxers, scene memory)
// so that load_images or load_videos can be called cleanly.
inline void
unload_media(PlaybackState &ps, ProjectManager &pm,
             std::vector<FFmpegDemuxer *> &demuxers,
             DecoderContext *dc_context,
             RenderScene *scene,
             std::vector<std::thread> &decoder_threads,
             std::vector<bool> &is_view_focused,
             std::unordered_map<std::string, bool> &window_was_decoding) {
    if (!ps.video_loaded)
        return;

    // Signal all decoder/image_loader threads to stop
    dc_context->stop_flag = true;
    for (auto &t : decoder_threads) {
        if (t.joinable())
            t.join();
    }
    decoder_threads.clear();
    dc_context->stop_flag = false;

    // Free demuxers
    for (auto *d : demuxers)
        delete d;
    demuxers.clear();

    // Free scene display buffers
    if (scene->display_buffer) {
        for (u32 j = 0; j < scene->num_cams; j++) {
            if (scene->display_buffer[j]) {
                for (u32 i = 0; i < scene->size_of_buffer; i++) {
#ifdef __APPLE__
                    free(scene->display_buffer[j][i].frame);
                    if (scene->display_buffer[j][i].pixel_buffer) {
                        CVPixelBufferRelease(scene->display_buffer[j][i].pixel_buffer);
                        scene->display_buffer[j][i].pixel_buffer = nullptr;
                    }
#else
                    if (scene->use_cpu_buffer)
                        free(scene->display_buffer[j][i].frame);
                    else
                        cudaFree(scene->display_buffer[j][i].frame);
#endif
                }
                free(scene->display_buffer[j]);
            }
        }
        free(scene->display_buffer);
        scene->display_buffer = nullptr;
    }

    // Free other scene arrays
    free(scene->image_width);
    scene->image_width = nullptr;
    free(scene->image_height);
    scene->image_height = nullptr;
    free(scene->seek_context);
    scene->seek_context = nullptr;

#ifndef __APPLE__
    // Free Linux GPU resources (PBOs, CUDA mappings, GL textures)
    if (scene->pbo_cuda) {
        for (u32 j = 0; j < scene->num_cams; j++) {
            unmap_cuda_resource(&scene->pbo_cuda[j].cuda_resource);
            cudaGraphicsUnregisterResource(scene->pbo_cuda[j].cuda_resource);
            glDeleteBuffers(1, &scene->pbo_cuda[j].pbo);
        }
    }
    if (scene->image_texture) {
        for (u32 j = 0; j < scene->num_cams; j++)
            glDeleteTextures(1, &scene->image_texture[j]);
        free(scene->image_texture);
        scene->image_texture = nullptr;
    }
#endif

    free(scene->pbo_cuda);
    scene->pbo_cuda = nullptr;
#ifdef __APPLE__
    free(scene->image_descriptor);
    scene->image_descriptor = nullptr;
#endif

    scene->num_cams = 0;
    scene->size_of_buffer = 0;

    // Clear playback and project media state
    is_view_focused.clear();
    pm.camera_names.clear();
    window_need_decoding.clear();
    window_was_decoding.clear();
    ps.video_loaded = false;
    ps.play_video = false;
    ps.to_display_frame_number = 0;
    ps.read_head = 0;
    ps.slider_frame_number = 0;
    ps.just_seeked = false;
    ps.pause_seeked = false;
    dc_context->decoding_flag = false;
    dc_context->total_num_frame = INT_MAX;
    dc_context->estimated_num_frames = 0;

    // Clear stale per-camera decoded frame counters
    latest_decoded_frame.clear();

    // Reset realtime playback (load_images sets false; load_videos expects true)
    ps.realtime_playback = true;
    ps.accumulated_play_time = 0.0;
    ps.last_play_time_start = std::chrono::steady_clock::now();
    ps.pause_selected = 0;
}

inline void
load_images(std::map<std::string, std::string> &selected_files,
            PlaybackState &ps, ProjectManager &pm,
            std::vector<std::string> &imgs_names, RenderScene *scene,
            DecoderContext *dc_context, int label_buffer_size,
            std::vector<std::thread> &decoder_threads,
            std::vector<bool> &is_view_focused,
            std::unordered_map<std::string, bool> &window_was_decoding) {

    std::string file_ext;
    for (const auto &elem : selected_files) {
        std::size_t cam_string_position = elem.first.find("_");
        std::string cam_name = elem.first.substr(0, cam_string_position);
        std::string file_full = elem.first.substr(cam_string_position + 1);

        window_need_decoding[cam_name].store(true);
        window_was_decoding[cam_name] = true;

        // split "123.jpg" -> name = "123", ext = "jpg"
        std::size_t dot_pos = file_full.rfind('.');
        std::string file_name = file_full.substr(0, dot_pos);
        file_ext = file_full.substr(dot_pos + 1);

        if (std::find(pm.camera_names.begin(), pm.camera_names.end(),
                      cam_name) == pm.camera_names.end()) {
            pm.camera_names.push_back(cam_name);
        }

        if (std::find(imgs_names.begin(), imgs_names.end(), file_name) ==
            imgs_names.end()) {
            imgs_names.push_back(file_name);
        }
    }

    auto to_number = [](const std::string &s) { return std::stoi(s); };

    std::sort(imgs_names.begin(), imgs_names.end(),
              [&](const std::string &a, const std::string &b) {
                  return to_number(a) < to_number(b);
              });

    dc_context->seek_interval = 1;
    dc_context->video_fps = 1;
    ps.realtime_playback = false;
    scene->num_cams = pm.camera_names.size();
    scene->image_width = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    scene->image_height = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    for (u32 j = 0; j < scene->num_cams; j++) {
        std::string file_name = pm.media_folder + "/" + pm.camera_names[j] +
                                "_" + imgs_names[0] + "." + file_ext;
#ifdef __APPLE__
        int w = 0, h = 0, ch = 0;
        stbi_info(file_name.c_str(), &w, &h, &ch);
        scene->image_width[j] = w;
        scene->image_height[j] = h;
#else
        cv::Mat image = cv::imread(file_name, cv::IMREAD_COLOR);
        scene->image_width[j] = image.cols;
        scene->image_height[j] = image.rows;
#endif
    }
    if (imgs_names.size() < (size_t)label_buffer_size) {
        label_buffer_size = imgs_names.size();
    }
    render_allocate_scene_memory(scene, label_buffer_size);
    for (int i = 0; i < scene->num_cams; i++) {
        decoder_threads.push_back(
            std::thread(&image_loader, dc_context, imgs_names,
                        scene->display_buffer[i], scene->size_of_buffer,
                        &scene->seek_context[i], scene->use_cpu_buffer,
                        pm.camera_names[i], pm.media_folder, file_ext));
        is_view_focused.push_back(false);
    }
    ps.video_loaded = true;
}

inline void
load_videos(std::map<std::string, std::string> &selected_files,
            PlaybackState &ps, ProjectManager &pm,
            std::unordered_map<std::string, bool> &window_was_decoding,
            std::vector<FFmpegDemuxer *> &demuxers, DecoderContext *dc_context,
            RenderScene *scene, int label_buffer_size,
            std::vector<std::thread> &decoder_threads,
            std::vector<bool> &is_view_focused) {
    // Track which camera names successfully loaded (in order) so that
    // camera_names stays in sync with demuxers after skipping failures.
    std::vector<std::string> loaded_cam_names;

    if (selected_files.empty()) {
        for (const auto &cam_string : pm.camera_names) {
            std::map<std::string, std::string> m;
            std::string media_filename =
                (std::filesystem::path(pm.media_folder) / (cam_string + ".mp4"))
                    .string();
            try {
                FFmpegDemuxer *demuxer =
                    new FFmpegDemuxer(media_filename.c_str(), m);
                if (demuxer->GetPixelFormat() == AV_PIX_FMT_NONE) {
                    std::cerr << "[load_videos] Skipping camera '" << cam_string
                              << "': broken header (no pixel format)" << std::endl;
                    delete demuxer;
                } else {
                    demuxers.push_back(demuxer);
                    loaded_cam_names.push_back(cam_string);
                    window_need_decoding[cam_string].store(true);
                    window_was_decoding[cam_string] = true;
                }
            } catch (const std::exception &e) {
                std::cerr << "[load_videos] Skipping camera '" << cam_string
                          << "': " << e.what() << std::endl;
            }
        }
        // Use the first successfully loaded demuxer for seek_interval/fps
        if (!demuxers.empty()) {
            dc_context->seek_interval =
                (int)demuxers[0]->FindKeyFrameInterval();
            // FindKeyFrameInterval consumes packets from demuxer 0; seek back
            demuxers[0]->SeekToStart();
            dc_context->video_fps = demuxers[0]->GetFramerate();
        }
        pm.camera_names = loaded_cam_names;
    } else {
        for (const auto &elem : selected_files) {
            std::size_t cam_string_mp4_position = elem.first.find("mp4");
            std::string cam_string =
                elem.first.substr(0, cam_string_mp4_position - 1);
            std::map<std::string, std::string> m;
            try {
                FFmpegDemuxer *demuxer =
                    new FFmpegDemuxer(elem.second.c_str(), m);
                if (demuxer->GetPixelFormat() == AV_PIX_FMT_NONE) {
                    std::cerr << "[load_videos] Skipping camera '" << cam_string
                              << "': broken header (no pixel format)" << std::endl;
                    delete demuxer;
                } else {
                    demuxers.push_back(demuxer);
                    loaded_cam_names.push_back(cam_string);
                    window_need_decoding[cam_string].store(true);
                    window_was_decoding[cam_string] = true;
                }
            } catch (const std::exception &e) {
                std::cerr << "[load_videos] Skipping camera '" << cam_string
                          << "' (" << elem.second << "): " << e.what()
                          << std::endl;
            }
        }
        if (!demuxers.empty()) {
            dc_context->seek_interval =
                (int)demuxers[0]->FindKeyFrameInterval();
            dc_context->video_fps = demuxers[0]->GetFramerate();
        }
        pm.camera_names = loaded_cam_names;
    }

    if (demuxers.empty()) {
        std::cerr << "[load_videos] No cameras could be loaded" << std::endl;
        return;
    }

    scene->num_cams = demuxers.size();
    scene->image_width = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    scene->image_height = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    for (u32 j = 0; j < scene->num_cams; j++) {
        scene->image_width[j] = demuxers[j]->GetWidth();
        scene->image_height[j] = demuxers[j]->GetHeight();
    }
    render_allocate_scene_memory(scene, label_buffer_size);

    for (int i = 0; i < scene->num_cams; i++) {
        decoder_threads.push_back(std::thread(
            &decoder_process, dc_context, demuxers[i], pm.camera_names[i],
            scene->display_buffer[i], scene->size_of_buffer,
            &scene->seek_context[i], scene->use_cpu_buffer));
        is_view_focused.push_back(false);
    }
    ps.video_loaded = true;
}
