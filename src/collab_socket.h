#pragma once
// collab_socket.h -- the ONLY file in the collaboration layer that touches the
// network. Everything above it (framing, ops, bundles) is pure logic over
// buffers so it can be tested without a port.
//
// A thin blocking TCP wrapper plus a poll() helper for the relay. No event
// loop library, no async framework -- the client does one blocking round trip
// on a background thread, and the relay multiplexes every connection through a
// single poll() rather than a thread per client.
//
// Portability: BSD sockets on POSIX, Winsock on Windows. The differences are
// confined to the shims at the top of this file.
//
// SIGPIPE: writing to a socket the peer has closed raises SIGPIPE by default,
// which kills the process. RED must never die because a relay went away, so
// sends use MSG_NOSIGNAL where it exists and SO_NOSIGPIPE where it does not.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace collab {
namespace net {

#ifdef _WIN32
using socket_t = SOCKET;
static constexpr socket_t kInvalidSocket = INVALID_SOCKET;
inline int close_socket(socket_t s) { return ::closesocket(s); }
inline int last_error() { return ::WSAGetLastError(); }
inline bool would_block(int e) {
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
}
inline bool timed_out(int e) { return e == WSAETIMEDOUT; }
#else
using socket_t = int;
static constexpr socket_t kInvalidSocket = -1;
inline int close_socket(socket_t s) { return ::close(s); }
inline int last_error() { return errno; }
inline bool would_block(int e) {
    return e == EWOULDBLOCK || e == EAGAIN || e == EINPROGRESS;
}
inline bool timed_out(int e) { return e == EAGAIN || e == EWOULDBLOCK; }
#endif

// Winsock needs a process-wide init. Idempotent and safe to call from any
// entry point; a no-op everywhere else.
inline bool startup() {
#ifdef _WIN32
    static bool done = false;
    static bool ok = false;
    if (!done) {
        WSADATA wsa;
        ok = (::WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
        done = true;
    }
    return ok;
#else
    return true;
#endif
}

inline std::string error_string(int e) {
#ifdef _WIN32
    return "winsock error " + std::to_string(e);
#else
    return std::string(std::strerror(e));
#endif
}

// ── Socket ──

// Move-only RAII wrapper. Copying would double-close.
class Socket {
  public:
    Socket() = default;
    explicit Socket(socket_t s) : fd_(s) {}
    ~Socket() { close(); }

    Socket(Socket &&o) noexcept : fd_(o.fd_) { o.fd_ = kInvalidSocket; }
    Socket &operator=(Socket &&o) noexcept {
        if (this != &o) {
            close();
            fd_ = o.fd_;
            o.fd_ = kInvalidSocket;
        }
        return *this;
    }
    Socket(const Socket &) = delete;
    Socket &operator=(const Socket &) = delete;

    bool valid() const { return fd_ != kInvalidSocket; }
    socket_t fd() const { return fd_; }

    socket_t release() {
        const socket_t s = fd_;
        fd_ = kInvalidSocket;
        return s;
    }

    void close() {
        if (fd_ != kInvalidSocket) {
            close_socket(fd_);
            fd_ = kInvalidSocket;
        }
    }

    // Half-closes the write side so the peer sees a clean EOF rather than a
    // reset. Used on a graceful Bye.
    void shutdown_write() {
        if (fd_ == kInvalidSocket) return;
#ifdef _WIN32
        ::shutdown(fd_, SD_SEND);
#else
        ::shutdown(fd_, SHUT_WR);
#endif
    }

    // ── Options ──

    bool set_nodelay(bool on = true) {
        const int v = on ? 1 : 0;
        return ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY,
                            reinterpret_cast<const char *>(&v),
                            sizeof(v)) == 0;
    }

    bool set_nonblocking(bool on) {
#ifdef _WIN32
        u_long mode = on ? 1 : 0;
        return ::ioctlsocket(fd_, FIONBIO, &mode) == 0;
#else
        int flags = ::fcntl(fd_, F_GETFL, 0);
        if (flags < 0) return false;
        flags = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
        return ::fcntl(fd_, F_SETFL, flags) == 0;
#endif
    }

    // Bounds how long a blocking recv/send will sit before returning. This is
    // what lets the sync thread notice its stop flag promptly instead of
    // hanging until TCP gives up, which can be minutes.
    bool set_timeouts(int ms) {
#ifdef _WIN32
        DWORD tv = static_cast<DWORD>(ms);
#else
        struct timeval tv;
        tv.tv_sec = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
#endif
        const bool a = ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO,
                                    reinterpret_cast<const char *>(&tv),
                                    sizeof(tv)) == 0;
        const bool b = ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO,
                                    reinterpret_cast<const char *>(&tv),
                                    sizeof(tv)) == 0;
        return a && b;
    }

    void suppress_sigpipe() {
#if defined(SO_NOSIGPIPE)
        const int v = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE,
                     reinterpret_cast<const char *>(&v), sizeof(v));
#endif
    }

    // ── Transfer ──

    // Sends every byte or fails. Short writes are normal on a socket, so the
    // loop is mandatory, not defensive.
    bool send_all(const void *data, size_t len, std::string *err = nullptr) {
        const char *p = static_cast<const char *>(data);
        size_t sent = 0;
        while (sent < len) {
#if defined(MSG_NOSIGNAL)
            const int flags = MSG_NOSIGNAL;
#else
            const int flags = 0;
#endif
            const auto n = ::send(fd_, p + sent, static_cast<int>(len - sent),
                                  flags);
            if (n > 0) {
                sent += static_cast<size_t>(n);
                continue;
            }
            const int e = last_error();
            if (n < 0 && would_block(e)) continue;  // timeout slice; retry
            if (err)
                *err = "send failed after " + std::to_string(sent) + "/" +
                       std::to_string(len) + " bytes: " + error_string(e);
            return false;
        }
        return true;
    }

    bool send_all(const std::vector<uint8_t> &b, std::string *err = nullptr) {
        return send_all(b.data(), b.size(), err);
    }

    enum class RecvResult { Data, Closed, Timeout, Error };

    // One recv. `got` is set only for Data. Timeout is a normal outcome given
    // set_timeouts() and means "nothing yet", not a failure.
    RecvResult recv_some(void *buf, size_t len, size_t &got,
                         std::string *err = nullptr) {
        got = 0;
        const auto n = ::recv(fd_, static_cast<char *>(buf),
                              static_cast<int>(len), 0);
        if (n > 0) {
            got = static_cast<size_t>(n);
            return RecvResult::Data;
        }
        if (n == 0) return RecvResult::Closed;
        const int e = last_error();
        if (would_block(e) || timed_out(e)) return RecvResult::Timeout;
        if (err) *err = "recv failed: " + error_string(e);
        return RecvResult::Error;
    }

  private:
    socket_t fd_ = kInvalidSocket;
};

// ── Client connect ──

// Resolves `host` (v4 or v6) and connects, giving up after `timeout_ms`.
//
// The connect itself is done non-blocking + poll so the timeout is honored --
// a blocking connect() to an unreachable host can hang for over a minute,
// which would freeze the sync thread's shutdown.
inline bool connect_to(const std::string &host, uint16_t port, int timeout_ms,
                       Socket &out, std::string *err = nullptr) {
    if (!startup()) {
        if (err) *err = "winsock init failed";
        return false;
    }

    struct addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;      // v4 or v6, whichever resolves
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *res = nullptr;
    const std::string port_s = std::to_string(port);
    const int gai = ::getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
    if (gai != 0 || !res) {
        if (err) *err = "cannot resolve " + host + ": " + ::gai_strerror(gai);
        return false;
    }

    std::string last_err = "no addresses for " + host;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        Socket s(::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
        if (!s.valid()) {
            last_err = "socket(): " + error_string(last_error());
            continue;
        }
        s.suppress_sigpipe();
        s.set_nonblocking(true);

        int rc = ::connect(s.fd(), ai->ai_addr,
                           static_cast<int>(ai->ai_addrlen));
        if (rc != 0) {
            const int e = last_error();
            if (!would_block(e)) {
                last_err = "connect(): " + error_string(e);
                continue;
            }
#ifdef _WIN32
            WSAPOLLFD pfd;
            pfd.fd = s.fd();
            pfd.events = POLLWRNORM;
            pfd.revents = 0;
            rc = ::WSAPoll(&pfd, 1, timeout_ms);
#else
            struct pollfd pfd;
            pfd.fd = s.fd();
            pfd.events = POLLOUT;
            pfd.revents = 0;
            rc = ::poll(&pfd, 1, timeout_ms);
#endif
            if (rc <= 0) {
                last_err = "connect to " + host + ":" + port_s +
                           (rc == 0 ? " timed out" : " failed while waiting");
                continue;
            }
            // poll() reporting writable does not by itself mean success --
            // a refused connection also wakes it. SO_ERROR is the real answer.
            int soerr = 0;
#ifdef _WIN32
            int slen = sizeof(soerr);
#else
            socklen_t slen = sizeof(soerr);
#endif
            if (::getsockopt(s.fd(), SOL_SOCKET, SO_ERROR,
                             reinterpret_cast<char *>(&soerr), &slen) != 0 ||
                soerr != 0) {
                last_err = "connect to " + host + ":" + port_s + " refused: " +
                           error_string(soerr);
                continue;
            }
        }

        s.set_nonblocking(false);
        s.set_nodelay(true);
        if (timeout_ms > 0) s.set_timeouts(timeout_ms);
        out = std::move(s);
        ::freeaddrinfo(res);
        return true;
    }

    ::freeaddrinfo(res);
    if (err) *err = last_err;
    return false;
}

// ── Listener (relay side) ──

class Listener {
  public:
    bool listen_on(uint16_t port, std::string *err = nullptr,
                   int backlog = 64) {
        if (!startup()) {
            if (err) *err = "winsock init failed";
            return false;
        }

        // AF_INET6 with V6ONLY off accepts both v4 and v6 clients on one
        // socket. If the host has no v6 stack at all, fall back to v4.
        Socket s(::socket(AF_INET6, SOCK_STREAM, 0));
        bool v6 = s.valid();
        if (!v6) s = Socket(::socket(AF_INET, SOCK_STREAM, 0));
        if (!s.valid()) {
            if (err) *err = "socket(): " + error_string(last_error());
            return false;
        }

        const int one = 1;
        ::setsockopt(s.fd(), SOL_SOCKET, SO_REUSEADDR,
                     reinterpret_cast<const char *>(&one), sizeof(one));

        int rc;
        if (v6) {
            const int off = 0;
            ::setsockopt(s.fd(), IPPROTO_IPV6, IPV6_V6ONLY,
                         reinterpret_cast<const char *>(&off), sizeof(off));
            struct sockaddr_in6 addr;
            std::memset(&addr, 0, sizeof(addr));
            addr.sin6_family = AF_INET6;
            addr.sin6_addr = in6addr_any;
            addr.sin6_port = htons(port);
            rc = ::bind(s.fd(), reinterpret_cast<struct sockaddr *>(&addr),
                        sizeof(addr));
        } else {
            struct sockaddr_in addr;
            std::memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY;
            addr.sin_port = htons(port);
            rc = ::bind(s.fd(), reinterpret_cast<struct sockaddr *>(&addr),
                        sizeof(addr));
        }
        if (rc != 0) {
            if (err)
                *err = "cannot bind port " + std::to_string(port) + ": " +
                       error_string(last_error());
            return false;
        }
        if (::listen(s.fd(), backlog) != 0) {
            if (err) *err = "listen(): " + error_string(last_error());
            return false;
        }
        s.set_nonblocking(true);
        sock_ = std::move(s);
        return true;
    }

    // Non-blocking. Returns false with `again = true` when there is simply no
    // pending connection.
    bool accept_one(Socket &out, bool &again, std::string *err = nullptr) {
        again = false;
        const socket_t c = ::accept(sock_.fd(), nullptr, nullptr);
        if (c == kInvalidSocket) {
            const int e = last_error();
            if (would_block(e)) {
                again = true;
                return false;
            }
            if (err) *err = "accept(): " + error_string(e);
            return false;
        }
        Socket s(c);
        s.suppress_sigpipe();
        s.set_nodelay(true);
        s.set_nonblocking(true);
        out = std::move(s);
        return true;
    }

    bool valid() const { return sock_.valid(); }
    socket_t fd() const { return sock_.fd(); }
    void close() { sock_.close(); }

    // The port actually bound. Needed when listening on port 0 to let the OS
    // pick a free one, which is how tests avoid colliding with a real relay.
    uint16_t port() const {
        struct sockaddr_storage ss;
#ifdef _WIN32
        int len = sizeof(ss);
#else
        socklen_t len = sizeof(ss);
#endif
        std::memset(&ss, 0, sizeof(ss));
        if (::getsockname(sock_.fd(), reinterpret_cast<struct sockaddr *>(&ss),
                          &len) != 0)
            return 0;
        if (ss.ss_family == AF_INET6)
            return ntohs(reinterpret_cast<struct sockaddr_in6 *>(&ss)->sin6_port);
        if (ss.ss_family == AF_INET)
            return ntohs(reinterpret_cast<struct sockaddr_in *>(&ss)->sin_port);
        return 0;
    }

  private:
    Socket sock_;
};

// ── poll ──

struct PollItem {
    socket_t fd = kInvalidSocket;
    bool want_read = false;
    bool want_write = false;
    bool can_read = false;
    bool can_write = false;
    bool hung_up = false;
};

// Returns the number of ready descriptors, 0 on timeout, -1 on error.
inline int poll_wait(std::vector<PollItem> &items, int timeout_ms) {
    if (items.empty()) return 0;

#ifdef _WIN32
    std::vector<WSAPOLLFD> pfds(items.size());
#else
    std::vector<struct pollfd> pfds(items.size());
#endif
    for (size_t i = 0; i < items.size(); ++i) {
        pfds[i].fd = items[i].fd;
        pfds[i].events = 0;
#ifdef _WIN32
        if (items[i].want_read) pfds[i].events |= POLLRDNORM;
        if (items[i].want_write) pfds[i].events |= POLLWRNORM;
#else
        if (items[i].want_read) pfds[i].events |= POLLIN;
        if (items[i].want_write) pfds[i].events |= POLLOUT;
#endif
        pfds[i].revents = 0;
    }

#ifdef _WIN32
    const int rc = ::WSAPoll(pfds.data(), static_cast<ULONG>(pfds.size()),
                             timeout_ms);
#else
    const int rc = ::poll(pfds.data(), static_cast<nfds_t>(pfds.size()),
                          timeout_ms);
#endif

    for (size_t i = 0; i < items.size(); ++i) {
#ifdef _WIN32
        items[i].can_read = (pfds[i].revents & POLLRDNORM) != 0;
        items[i].can_write = (pfds[i].revents & POLLWRNORM) != 0;
#else
        items[i].can_read = (pfds[i].revents & POLLIN) != 0;
        items[i].can_write = (pfds[i].revents & POLLOUT) != 0;
#endif
        items[i].hung_up = (pfds[i].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0;
    }
    return rc;
}

}  // namespace net
}  // namespace collab
