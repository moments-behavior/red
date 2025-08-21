#include "trial_slicer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"
#include "FFmpegDemuxer.h"
#include "NvCodecUtils.h"
#include "NvDecoder.h"
#include "decoder.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>

TrialSlicer::TrialSlicer() 
    : export_in_progress(false), export_status("Ready"), current_progress(0), total_progress(0) {
}

TrialSlicer::~TrialSlicer() {
}

void TrialSlicer::addTrialMark(int frame_number, const std::string& label) {
    // Check if this frame already has a mark and remove it
    auto existing_mark = std::find_if(trial_marks.begin(), trial_marks.end(),
        [frame_number](const TrialMark& mark) {
            return mark.frame_number == frame_number;
        });
    
    if (existing_mark != trial_marks.end()) {
        trial_marks.erase(existing_mark);
    }
    
    // Add the new mark
    trial_marks.emplace_back(frame_number, label);
    generateTrialsFromMarks();
}

void TrialSlicer::removeLastMark() {
    if (!trial_marks.empty()) {
        trial_marks.pop_back();
        generateTrialsFromMarks();
    }
}

void TrialSlicer::clearAllMarks() {
    trial_marks.clear();
    trials.clear();
}

void TrialSlicer::generateTrialsFromMarks() {
    trials.clear();
    
    std::sort(trial_marks.begin(), trial_marks.end(), 
              [](const TrialMark& a, const TrialMark& b) {
                  return a.frame_number < b.frame_number;
              });
    
    if (!validateTrialSequence()) {
        return; 
    }
    
    int trial_counter = 1;
    
    for (size_t i = 0; i < trial_marks.size(); i += 2) {
        if (i + 1 < trial_marks.size() && 
            trial_marks[i].label == "start" && 
            trial_marks[i + 1].label == "end") {
            
            std::string trial_name = generateTrialName(trial_counter, 
                                                     trial_marks[i].frame_number, 
                                                     trial_marks[i + 1].frame_number);
            trials.emplace_back(trial_marks[i].frame_number, 
                              trial_marks[i + 1].frame_number, 
                              trial_name);
            trial_counter++;
        }
    }
}

std::string TrialSlicer::generateTrialName(int trial_index, int start_frame, int end_frame) {
    std::stringstream ss;
    ss << "trial" << std::setfill('0') << std::setw(3) << trial_index;
    return ss.str();
}

std::vector<Trial> TrialSlicer::getTrials() {
    generateTrialsFromMarks(); 
    return trials;
}

void TrialSlicer::removeTrialByIndex(int index) {
    if (index >= 0 && index < trials.size()) {
        trials.erase(trials.begin() + index);
    }
}

void TrialSlicer::clearAllTrials() {
    trials.clear();
    trial_marks.clear();
}

TrialMark* TrialSlicer::getLastMark() {
    if (trial_marks.empty()) {
        return nullptr;
    }
    return &trial_marks.back();
}

bool TrialSlicer::hasUnpairedMark() const {
    return !validateTrialSequence();
}

bool TrialSlicer::validateTrialSequence() const {
    if (trial_marks.empty()) {
        return true;
    }
    
    if (trial_marks.size() % 2 != 0) {
        return false;
    }
    
    std::vector<TrialMark> sorted_marks = trial_marks;
    std::sort(sorted_marks.begin(), sorted_marks.end(), 
              [](const TrialMark& a, const TrialMark& b) {
                  return a.frame_number < b.frame_number;
              });
    
    for (size_t i = 0; i < sorted_marks.size(); i++) {
        if (i % 2 == 0) { // Even indices should be "start"
            if (sorted_marks[i].label != "start") {
                return false;
            }
        } else { 
            if (sorted_marks[i].label != "end") {
                return false;
            }
        }
    }
    
    return true;
}

std::string TrialSlicer::getValidationError() const {
    if (trial_marks.empty()) {
        return "";
    }
    
    if (trial_marks.size() % 2 != 0) {
        return "Uneven number of marks - need matching start/end pairs";
    }
    
    std::vector<TrialMark> sorted_marks = trial_marks;
    std::sort(sorted_marks.begin(), sorted_marks.end(), 
              [](const TrialMark& a, const TrialMark& b) {
                  return a.frame_number < b.frame_number;
              });
    
    for (size_t i = 0; i < sorted_marks.size(); i++) {
        if (i % 2 == 0) { 
            if (sorted_marks[i].label != "start") {
                return "Invalid sequence: expected 'start' at frame " + 
                       std::to_string(sorted_marks[i].frame_number);
            }
        } else { 
            if (sorted_marks[i].label != "end") {
                return "Invalid sequence: expected 'end' at frame " + 
                       std::to_string(sorted_marks[i].frame_number);
            }
        }
    }
    
    return "";
}

bool TrialSlicer::exportTrials() {
    if (export_in_progress) {
        return false;
    }
    
    if (trials.empty()) {
        export_status = "No trials to export";
        return false;
    }
    
    if (config.camera_names.empty()) {
        export_status = "No cameras configured";
        return false;
    }
    
    std::filesystem::create_directories(config.output_directory);
    
    export_in_progress = true;
    export_status = "Starting export...";
    current_progress = 0;
    total_progress = trials.size() * config.camera_names.size();
    
    std::thread export_thread([this]() {
        bool success = true;
        
        for (size_t trial_idx = 0; trial_idx < trials.size() && success; trial_idx++) {
            const Trial& trial = trials[trial_idx];
            
            for (size_t cam_idx = 0; cam_idx < config.camera_names.size() && success; cam_idx++) {
                const std::string& cam_name = config.camera_names[cam_idx];
                
                export_status = "Exporting " + trial.name + " - " + cam_name;
                
                std::string input_path;
                auto path_it = config.camera_video_paths.find(cam_name);
                if (path_it != config.camera_video_paths.end()) {
                    input_path = path_it->second;
                    std::cout << "Trial Slicer: Using stored path for " << cam_name << ": " << input_path << std::endl;
                } else {
                    input_path = config.media_directory + "/" + cam_name + ".mp4";
                    std::cout << "Trial Slicer: No stored path found for " << cam_name << ", using fallback: " << input_path << std::endl;
                }
                
                std::string trial_folder = config.output_directory + "/" + trial.name;
                std::filesystem::create_directories(trial_folder);
                
                std::string output_filename = cam_name + ".mp4";
                std::string output_path = trial_folder + "/" + output_filename;
                
                if (!std::filesystem::exists(input_path)) {
                    export_status = "Error: Input file not found: " + input_path;
                    std::cout << "Trial Slicer: File does not exist: " << input_path << std::endl;
                    success = false;
                    break;
                }
                
                success = extractVideoSegment(input_path, output_path, 
                                            trial.start_frame, trial.end_frame, config.fps);
                
                if (!success) {
                    export_status = "Error exporting " + output_filename;
                    break;
                }
                
                current_progress++;
            }
        }
        
        if (success) {
            export_status = "Export completed successfully";
        }
        
        export_in_progress = false;
    });
    
    export_thread.detach();
    return true;
}

bool TrialSlicer::extractVideoSegment(const std::string& input_video_path, 
                                     const std::string& output_video_path,
                                     int start_frame, int end_frame, double fps) {
    std::cout << "Trial Slicer: Starting video extraction..." << std::endl;
    std::cout << "Trial Slicer: Input: " << input_video_path << std::endl;
    std::cout << "Trial Slicer: Output: " << output_video_path << std::endl;
    std::cout << "Trial Slicer: Frames: " << start_frame << " to " << end_frame << std::endl;
    std::cout << "Trial Slicer: OpenCV version: " << cv::getVersionString() << std::endl;
    
    // Initialize CUDA context
    CUcontext cuContext = nullptr;
    FFmpegDemuxer *demuxer = nullptr;
    NvDecoder *decoder = nullptr;
    CUdeviceptr pTmpImage = 0;
    std::ofstream output_file;
    
    try {
        ck(cuInit(0));
        createCudaContext(&cuContext, 0, 0);

        std::map<std::string, std::string> ffmpeg_options;
        demuxer = new FFmpegDemuxer(input_video_path.c_str(), ffmpeg_options);

        int width = demuxer->GetWidth();
        int height = demuxer->GetHeight();
        
        decoder = new NvDecoder(cuContext, true, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));

        ck(cuMemAlloc(&pTmpImage, width * height * 4));

        std::filesystem::path output_path(output_video_path);
        std::filesystem::create_directories(output_path.parent_path());
        
        std::string frames_dir = output_path.parent_path() / ("frames_" + output_path.stem().string());
        std::filesystem::create_directories(frames_dir);
        
        std::cout << "Trial Slicer: Saving frames to: " << frames_dir << std::endl;

        auto convert_frame_to_mat = [&](uint8_t *pFrame) -> cv::Mat {
            try {
                int iMatrix = decoder->GetVideoFormatInfo()
                                  .video_signal_description.matrix_coefficients;
                Nv12ToColor32<RGBA32>(pFrame, width, (uint8_t *)pTmpImage,
                                      4 * width, width, height, iMatrix);

                uint8_t *cpu_buffer = new uint8_t[width * height * 4];
                decoder_get_image_from_gpu(pTmpImage, cpu_buffer, 4 * width, height);

                cv::Mat rgba_frame(height, width, CV_8UC4, cpu_buffer);
                cv::Mat bgr_frame;
                cv::cvtColor(rgba_frame, bgr_frame, cv::COLOR_RGBA2BGR);

                cv::Mat result = bgr_frame.clone();
                delete[] cpu_buffer;

                return result;
            } catch (const std::exception &e) {
                std::cerr << "Error converting frame to Mat: " << e.what() << std::endl;
                return cv::Mat();
            }
        };

        SeekContext seek_ctx;
        seek_ctx.use_seek = true;
        seek_ctx.seek_frame = start_frame;
        seek_ctx.mode = EXACT_FRAME;
        seek_ctx.crit = BY_NUMBER;

        uint8_t *pVideo = nullptr;
        size_t nVideoBytes = 0;
        PacketData pktData;

        if (!demuxer->Seek(seek_ctx, pVideo, nVideoBytes, pktData)) {
            std::cerr << "Trial Slicer: Failed to seek to frame " << start_frame << std::endl;
            throw std::runtime_error("Failed to seek to start frame");
        }

        // Flush decoder
        int nFrameReturned = decoder->Decode(nullptr, 0, CUVID_PKT_DISCONTINUITY);
        for (int i = 0; i < nFrameReturned; i++) {
            decoder->GetFrame();
        }

        int frames_written = 0;
        int current_frame = start_frame;
        bool first_decode = true;
        std::vector<std::string> frame_files;

        while (current_frame <= end_frame) {
            if (first_decode) {
                nFrameReturned = decoder->Decode(pVideo, nVideoBytes);
                first_decode = false;
            } else {
                if (!demuxer->Demux(pVideo, nVideoBytes, pktData)) {
                    nFrameReturned = decoder->Decode(nullptr, 0, CUVID_PKT_DISCONTINUITY);
                    if (nFrameReturned == 0) break;
                } else {
                    nFrameReturned = decoder->Decode(pVideo, nVideoBytes);
                }
            }

            for (int i = 0; i < nFrameReturned && current_frame <= end_frame; i++) {
                uint8_t *pFrame = decoder->GetFrame();
                if (pFrame && current_frame >= start_frame) {
                    cv::Mat frame = convert_frame_to_mat(pFrame);
                    if (!frame.empty()) {
                        std::ostringstream frame_filename;
                        frame_filename << frames_dir << "/frame_" << std::setw(6) << std::setfill('0') << frames_written << ".png";
                        
                        if (cv::imwrite(frame_filename.str(), frame)) {
                            frame_files.push_back(frame_filename.str());
                            frames_written++;
                            
                            if (frames_written % 30 == 0) {
                                std::cout << "Trial Slicer: Saved " << frames_written << " frames" << std::endl;
                            }
                        } else {
                            std::cerr << "Trial Slicer: Failed to save frame: " << frame_filename.str() << std::endl;
                        }
                    }
                }
                current_frame++;
            }
        }

        std::cout << "Trial Slicer: Successfully extracted " << frames_written << " frames" << std::endl;
        
        std::cout << "Trial Slicer: Creating video with NVENC..." << std::endl;
        
        std::ostringstream nvenc_cmd;
        nvenc_cmd << "ffmpeg -y -r " << fps << " -i \"" << frames_dir << "/frame_%06d.png\" "
                  << "-c:v h264_nvenc -preset p1 -tune lossless -pix_fmt yuv420p \"" 
                  << output_video_path << "\" 2>/dev/null";
        
        std::cout << "Trial Slicer: Running NVENC command: " << nvenc_cmd.str() << std::endl;
        
        int result = system(nvenc_cmd.str().c_str());
        if (result != 0) {
            std::cerr << "Trial Slicer: NVENC encoding failed with code " << result << std::endl;
        } else {
            std::cout << "Trial Slicer: NVENC encoding completed successfully" << std::endl;
            
            std::cout << "Trial Slicer: Cleaning up temporary frame files..." << std::endl;
            std::filesystem::remove_all(frames_dir);
        }
        
        delete demuxer;
        delete decoder;
        if (pTmpImage) {
            cuMemFree(pTmpImage);
        }
        if (cuContext) {
            cuCtxDestroy(cuContext);
        }
        
        if (!std::filesystem::exists(output_video_path)) {
            std::cerr << "Trial Slicer: Output file was not created: " << output_video_path << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const std::exception &e) {
        std::cerr << "Trial Slicer: Error during video extraction: " << e.what() << std::endl;
        
        // Cleanup on error
        if (demuxer) delete demuxer;
        if (decoder) delete decoder;
        if (pTmpImage) cuMemFree(pTmpImage);
        if (cuContext) cuCtxDestroy(cuContext);
        
        return false;
    }
}

bool TrialSlicer::extractVideoSegmentGPU(const std::string& input_video_path,
                                        const std::string& output_video_path,
                                        int start_frame, int end_frame, double fps) {
    return extractVideoSegment(input_video_path, output_video_path, start_frame, end_frame, fps);
}

bool TrialSlicer::saveTrialMarksToFile(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "frame_number,label,timestamp\n";
    for (const auto& mark : trial_marks) {
        auto time_since_epoch = mark.timestamp.time_since_epoch();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_epoch).count();
        file << mark.frame_number << "," << mark.label << "," << ms << "\n";
    }
    
    file.close();
    return true;
}

bool TrialSlicer::loadTrialMarksFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    trial_marks.clear();
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string frame_str, label, timestamp_str;
        
        if (std::getline(ss, frame_str, ',') &&
            std::getline(ss, label, ',') &&
            std::getline(ss, timestamp_str)) {
            
            int frame_number = std::stoi(frame_str);
            trial_marks.emplace_back(frame_number, label);
        }
    }
    
    file.close();
    generateTrialsFromMarks();
    return true;
}

bool TrialSlicer::saveTrialsToFile(const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "trial_name,start_frame,end_frame\n";
    for (const auto& trial : trials) {
        file << trial.name << "," << trial.start_frame << "," << trial.end_frame << "\n";
    }
    
    file.close();
    return true;
}
