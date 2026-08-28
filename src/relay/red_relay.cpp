// red_relay.cpp -- command-line driver for the RED collaboration relay.
//
// One collaborator runs this on a host the others can reach (a VPS, or any
// port-forwarded machine). Every RED client dials OUT to it, so no client
// needs an open port and machines behind unrelated NATs can collaborate.
//
// All the logic lives in relay/relay_core.h so the loopback test can drive a
// real relay in-process.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "relay/relay_core.h"

namespace fs = std::filesystem;
using collab::relay::Relay;
using collab::relay::kDefaultQuotaGb;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop.store(true); }

// =========================================================================
// main
// =========================================================================

static void usage() {
    std::printf(
        "red_relay -- collaboration relay for RED\n"
        "\n"
        "  red_relay --secrets rooms.json [options]\n"
        "\n"
        "  --port N          port to listen on (default 7373)\n"
        "  --data DIR        where ops, manifests, and blobs live "
        "(default ./relay-data)\n"
        "  --secrets FILE    room shared secrets (required)\n"
        "  --quota-gb N      blob store size cap (default %llu)\n"
        "  --help\n"
        "\n"
        "rooms.json:\n"
        "  { \"rooms\": { \"rig-a\": { \"psk\": "
        "\"a-long-random-shared-secret\" } } }\n"
        "\n"
        "Traffic is authenticated but NOT encrypted. Put the relay behind an\n"
        "SSH tunnel or a WireGuard/Tailscale network if the annotations are\n"
        "sensitive:\n"
        "  ssh -N -L 7373:localhost:7373 user@relay-host\n",
        (unsigned long long)kDefaultQuotaGb);
}

int main(int argc, char **argv) {
    uint16_t port = 7373;
    fs::path data_dir = "relay-data";
    fs::path secrets;
    uint64_t quota_gb = kDefaultQuotaGb;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--port") port = static_cast<uint16_t>(std::stoi(next("--port")));
        else if (a == "--data") data_dir = next("--data");
        else if (a == "--secrets") secrets = next("--secrets");
        else if (a == "--quota-gb") quota_gb = std::stoull(next("--quota-gb"));
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else {
            std::fprintf(stderr, "unknown argument: %s\n\n", a.c_str());
            usage();
            return 2;
        }
    }

    if (secrets.empty()) {
        std::fprintf(stderr, "--secrets is required\n\n");
        usage();
        return 2;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);
#endif

    Relay relay;
    std::string err;
    if (!relay.init(port, data_dir, secrets, quota_gb, &err)) {
        std::fprintf(stderr, "[relay] startup failed: %s\n", err.c_str());
        return 1;
    }
    relay.run(g_stop);
    return 0;
}
