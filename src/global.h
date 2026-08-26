#ifndef RED_GLOBAL
#define RED_GLOBAL

#define MAX_VIEWS 20

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "sync_plan.h"

extern std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
extern std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;

// Canonical-timeline desync fix. The plan is built (from Cam*_meta.csv
// sidecars or an imported sync_plan.json) in load_videos before decoder
// threads spawn, is immutable while they run, and is reset in unload_media
// after they join — decoder threads hold pointers into plan.cams.
struct SyncFixState {
    sync_plan::SyncPlan plan;  // plan.usable() gates the whole feature
};
extern SyncFixState g_sync_fix;

// Selected keypoint colormap (see keypoint_colors.h). Single source of truth
// so every skeleton-load path recolors consistently. < 0 == legacy HSV
// rainbow; >= 0 == an ImPlotColormap_ index. Seeded from UserSettings at
// startup and updated live when the user changes it in Settings.
extern int g_keypoint_colormap;
#endif
