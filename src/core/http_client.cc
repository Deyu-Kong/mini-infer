#include "core/http_client.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstring>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace mini_infer {

namespace {
std::string to_lower(std::string s) {
    for (auto& c : s) if (c >= 'A' && c <= 'Z') c |= 0x20;
    return s;
}
bool send_all_raw(int fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) { if (errno == EINTR) continue; return false; }
        sent += static_cast<size_t>(n);
    }
    return true;
}
}  // namespace

bool HttpClient::parse_endpoint(const std::string& endpoint,
                                std::string& host, int& port) {
    std::string e = endpoint;
    // strip scheme
    if (e.rfind("http://", 0) == 0) e = e.substr(7);
    else if (e.rfind("https://", 0) == 0) e = e.substr(8);
    // strip trailing slash / path
    size_t slash = e.find('/');
    if (slash != std::string::npos) e = e.substr(0, slash);
    size_t colon = e.rfind(':');
    if (colon != std::string::npos) {
        host = e.substr(0, colon);
        try { port = std::stoi(e.substr(colon + 1)); }
        catch (...) { return false; }
    } else {
        host = e;
        port = 80;
    }
    return !host.empty();
}

HttpClient::HttpClient(const std::string& endpoint, int connect_timeout_sec)
    : connect_timeout_sec_(connect_timeout_sec) {
    if (!parse_endpoint(endpoint, host_, port_)) {
        throw std::runtime_error("invalid endpoint: " + endpoint);
    }
}

int HttpClient::connect_() const {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    // Non-blocking connect for timeout control.
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
        // Try hostname resolution.
        hostent* he = gethostbyname(host_.c_str());
        if (!he) { ::close(fd); return -1; }
        std::memcpy(&addr.sin_addr, he->h_addr, sizeof(in_addr));
    }

    int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) { ::close(fd); return -1; }

    fd_set wset;
    FD_ZERO(&wset);
    FD_SET(fd, &wset);
    timeval tv;
    tv.tv_sec = connect_timeout_sec_;
    tv.tv_usec = 0;
    int sel = ::select(fd + 1, nullptr, &wset, nullptr, &tv);
    if (sel <= 0) { ::close(fd); return -1; }

    int err = 0;
    socklen_t len = sizeof(err);
    getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
    if (err != 0) { ::close(fd); return -1; }

    // Back to blocking.
    fcntl(fd, F_SETFL, flags);
    timeval rwto;
    rwto.tv_sec = 120;
    rwto.tv_usec = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rwto, sizeof(rwto));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &rwto, sizeof(rwto));
    return fd;
}

bool HttpClient::send_all_(int fd, const std::string& s) {
    return send_all_raw(fd, s.data(), s.size());
}

bool HttpClient::read_until_headers_(int fd, std::string& buf, size_t& header_end) {
    char chunk[4096];
    while (true) {
        size_t pos = buf.find("\r\n\r\n");
        if (pos != std::string::npos) { header_end = pos; return true; }
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        buf.append(chunk, static_cast<size_t>(n));
        if (buf.size() > (1u << 24)) return false;  // header too large
    }
}

HttpResp HttpClient::post(const std::string& path, const std::string& json_body,
                          const std::string& content_type) {
    HttpResp r;
    int fd = connect_();
    if (fd < 0) { r.error = "connect failed"; return r; }
    std::ostringstream os;
    os << "POST " << path << " HTTP/1.1\r\n";
    os << "Host: " << host_ << ":" << port_ << "\r\n";
    os << "Content-Type: " << content_type << "\r\n";
    os << "Content-Length: " << json_body.size() << "\r\n";
    os << "Connection: close\r\n\r\n";
    os << json_body;
    if (!send_all_(fd, os.str())) { r.error = "send failed"; ::close(fd); return r; }

    std::string buf;
    size_t hend = 0;
    if (!read_until_headers_(fd, buf, hend)) {
        r.error = "no response"; ::close(fd); return r;
    }
    std::string head = buf.substr(0, hend);
    // Status line.
    size_t eol = head.find("\r\n");
    std::string status_line = (eol == std::string::npos) ? head : head.substr(0, eol);
    {
        std::istringstream iss(status_line);
        std::string ver;
        iss >> ver >> r.status;
    }
    // Body = everything after "\r\n\r\n".
    r.body = buf.substr(hend + 4);
    char chunk[8192];
    while (true) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        r.body.append(chunk, static_cast<size_t>(n));
    }
    // If the server chunk-encoded or kept body after close, we read until EOF.
    ::close(fd);
    if (r.status == 0) r.error = "bad status line";
    return r;
}

HttpResp HttpClient::get(const std::string& path) {
    HttpResp r;
    int fd = connect_();
    if (fd < 0) { r.error = "connect failed"; return r; }
    std::ostringstream os;
    os << "GET " << path << " HTTP/1.1\r\n";
    os << "Host: " << host_ << ":" << port_ << "\r\n";
    os << "Connection: close\r\n\r\n";
    if (!send_all_(fd, os.str())) { r.error = "send failed"; ::close(fd); return r; }

    std::string buf;
    size_t hend = 0;
    if (!read_until_headers_(fd, buf, hend)) { r.error = "no response"; ::close(fd); return r; }
    std::string head = buf.substr(0, hend);
    size_t eol = head.find("\r\n");
    std::string status_line = (eol == std::string::npos) ? head : head.substr(0, eol);
    {
        std::istringstream iss(status_line);
        std::string ver; iss >> ver >> r.status;
    }
    r.body = buf.substr(hend + 4);
    char chunk[8192];
    while (true) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        r.body.append(chunk, static_cast<size_t>(n));
    }
    ::close(fd);
    if (r.status == 0) r.error = "bad status line";
    return r;
}

bool HttpClient::post_stream(const std::string& path, const std::string& json_body,
                             const std::function<void(const std::string&)>& on_chunk,
                             const std::string& content_type) {
    int fd = connect_();
    if (fd < 0) return false;
    std::ostringstream os;
    os << "POST " << path << " HTTP/1.1\r\n";
    os << "Host: " << host_ << ":" << port_ << "\r\n";
    os << "Content-Type: " << content_type << "\r\n";
    os << "Content-Length: " << json_body.size() << "\r\n";
    os << "Connection: close\r\n\r\n";
    os << json_body;
    if (!send_all_(fd, os.str())) { ::close(fd); return false; }

    std::string buf;
    size_t hend = 0;
    if (!read_until_headers_(fd, buf, hend)) { ::close(fd); return false; }
    // Stream the body (including anything already buffered past headers).
    if (buf.size() > hend + 4) on_chunk(buf.substr(hend + 4));
    char chunk[8192];
    while (true) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        on_chunk(std::string(chunk, static_cast<size_t>(n)));
    }
    ::close(fd);
    return true;
}

}  // namespace mini_infer
