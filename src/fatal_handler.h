#pragma once
// Say what an uncaught exception was before the process dies.
//
// Uncaught, an exception ends in std::terminate -> abort(), which on Windows
// is a fail-fast in ucrtbase.dll (exception 0xc0000409) with nothing on the
// console: a red that crashed loading a video on a test PC left only that in
// Event Viewer. This prints the exception's what() to stderr first.
//
// MSVC keeps the terminate handler PER THREAD, so call this at the top of
// every thread that runs red code (main and the decoder entry points), not
// once. Elsewhere it is process-wide and calling it again is harmless.

#include <cstdio>
#include <cstdlib>
#include <exception>

inline void install_fatal_handler() {
    std::set_terminate([] {
        if (std::exception_ptr e = std::current_exception()) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception &x) {
                std::fprintf(stderr, "[red] fatal: uncaught exception: %s\n",
                             x.what());
            } catch (...) {
                std::fprintf(stderr, "[red] fatal: uncaught exception "
                                     "(not a std::exception)\n");
            }
        } else {
            std::fprintf(stderr, "[red] fatal: std::terminate called with no "
                                 "exception in flight\n");
        }
        std::fflush(stderr);
        std::abort();
    });
}
