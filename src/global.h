#ifndef RED_GLOBAL
#define RED_GLOBAL

#define MAX_VIEWS 20

#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

extern std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
extern std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;
#endif
