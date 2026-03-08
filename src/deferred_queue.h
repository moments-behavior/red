#pragma once
#include <functional>
#include <mutex>
#include <vector>

// Thread-safe queue of callbacks flushed once per frame on the main thread.
// Replaces ad-hoc flag patterns (e.g. laser_load_pending) for deferred work.
class DeferredQueue {
  public:
    // Enqueue a callback from any thread. Thread-safe.
    void enqueue(std::function<void()> fn) {
        std::lock_guard<std::mutex> lock(mu_);
        pending_.push_back(std::move(fn));
    }

    // Call once per frame on the main thread (after ImGui::NewFrame()).
    // Executes and clears all pending callbacks.
    void flush() {
        std::vector<std::function<void()>> batch;
        {
            std::lock_guard<std::mutex> lock(mu_);
            batch.swap(pending_);
        }
        for (auto &fn : batch)
            fn();
    }

    // Returns number of pending callbacks (for testing).
    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return pending_.size();
    }

  private:
    mutable std::mutex mu_;
    std::vector<std::function<void()>> pending_;
};
