#include "global.h"

std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;
SyncFixState g_sync_fix;
int g_keypoint_colormap = -1; // -1 == legacy HSV rainbow
