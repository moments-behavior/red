#pragma once

#include "FFmpegDemuxer.h"
#include "NvDecoder.h"
#include "decoder.h"
#include <chrono>
#include <map>
#include <string>
#include <vector>

struct TrialMark {
    int frame_number;
    std::string label; // "start" or "end"
    std::chrono::steady_clock::time_point timestamp;
    
    TrialMark(int frame, const std::string& mark_label) 
        : frame_number(frame), label(mark_label), timestamp(std::chrono::steady_clock::now()) {}
};

struct Trial {
    int start_frame;
    int end_frame;
    std::string name;
    bool valid;
    
    Trial() : start_frame(-1), end_frame(-1), name(""), valid(false) {}
    Trial(int start, int end, const std::string& trial_name) 
        : start_frame(start), end_frame(end), name(trial_name), valid(true) {}
};

struct TrialSlicerConfig {
    std::string output_directory;
    std::string video_prefix; // Prefix for output video files
    std::vector<std::string> camera_names;
    std::map<std::string, std::string> camera_video_paths; // Map camera name to full video path
    std::string media_directory;
    bool use_gpu_acceleration;
    double fps;
    
    TrialSlicerConfig() 
        : output_directory(""), video_prefix("trial"), 
          use_gpu_acceleration(true), fps(30.0) {}
};

class TrialSlicer {
private:
    std::vector<TrialMark> trial_marks;
    std::vector<Trial> trials;
    TrialSlicerConfig config;
    bool export_in_progress;
    std::string export_status;
    int current_progress;
    int total_progress;
    
    // Helper methods
    void generateTrialsFromMarks();
    bool extractVideoSegment(const std::string& input_video_path, 
                            const std::string& output_video_path,
                            int start_frame, int end_frame, double fps);
    bool extractVideoSegmentGPU(const std::string& input_video_path,
                               const std::string& output_video_path,
                               int start_frame, int end_frame, double fps);
    std::string generateTrialName(int trial_index, int start_frame, int end_frame);
    
public:
    TrialSlicer();
    ~TrialSlicer();
    
    // Mark management
    void addTrialMark(int frame_number, const std::string& label);
    void removeLastMark();
    void clearAllMarks();
    std::vector<TrialMark> getTrialMarks() const { return trial_marks; }
    
    // Trial management
    std::vector<Trial> getTrials();
    void removeTrialByIndex(int index);
    void clearAllTrials();
    int getTrialCount() const { return trials.size(); }
    
    // Configuration
    void setConfig(const TrialSlicerConfig& new_config) { config = new_config; }
    TrialSlicerConfig getConfig() const { return config; }
    
    // Export functionality
    bool exportTrials();
    bool isExportInProgress() const { return export_in_progress; }
    std::string getExportStatus() const { return export_status; }
    std::pair<int, int> getExportProgress() const { return {current_progress, total_progress}; }
    
    // File IO
    bool saveTrialMarksToFile(const std::string& filepath);
    bool loadTrialMarksFromFile(const std::string& filepath);
    bool saveTrialsToFile(const std::string& filepath);
    
    // Misc
    int getMarkCount() const { return trial_marks.size(); }
    TrialMark* getLastMark();
    bool hasUnpairedMark() const;
    bool validateTrialSequence() const;
    std::string getValidationError() const;
};
