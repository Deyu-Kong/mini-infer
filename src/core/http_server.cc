#include "core/http_server.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_infer {

namespace {

std::string to_lower(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c |= 0x20;
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\r' || s[a] == '\n' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\r' || s[b - 1] == '\n' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

std::string url_decode(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        char c = in[i];
        if (c == '+') {
            out.push_back(' ');
        } else if (c == '%' && i + 2 < in.size()) {
            auto hex = [](char h) -> int {
                if (h >= '0' && h <= '9') return h - '0';
                if (h >= 'a' && h <= 'f') return h - 'a' + 10;
                if (h >= 'A' && h <= 'F') return h - 'A' + 10;
                return -1;
            };
            int hi = hex(in[i + 1]);
            int lo = hex(in[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
            } else {
                out.push_back(c);
            }
        } else {
            out.push_back(c);
        }
    }
    return out;
}

bool read_until(int fd, std::string& buf, size_t max_bytes,
                const std::function<bool(const std::string&)>& pred) {
    if (pred(buf)) return true;
    char chunk[4096];
    while (buf.size() < max_bytes) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return pred(buf);
        buf.append(chunk, static_cast<size_t>(n));
        if (pred(buf)) return true;
    }
    return pred(buf);
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

const char* reason(int s) {
    switch (s) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 500: return "Internal Server Error";
        default: return "OK";
    }
}

}  // namespace

std::unordered_map<std::string, std::string> parse_query(const std::string& q) {
    std::unordered_map<std::string, std::string> out;
    size_t start = 0;
    while (start <= q.size()) {
        size_t amp = q.find('&', start);
        std::string pair = q.substr(start, (amp == std::string::npos ? q.size() : amp) - start);
        if (!pair.empty()) {
            size_t eq = pair.find('=');
            std::string k = url_decode(eq == std::string::npos ? pair : pair.substr(0, eq));
            std::string v = eq == std::string::npos ? "" : url_decode(pair.substr(eq + 1));
            out[k] = v;
        }
        if (amp == std::string::npos) break;
        start = amp + 1;
    }
    return out;
}

std::string json_error(int status, const std::string& message) {
    std::string esc;
    esc.reserve(message.size());
    for (char c : message) {
        switch (c) {
            case '"': esc += "\\\""; break;
            case '\\': esc += "\\\\"; break;
            case '\n': esc += "\\n"; break;
            case '\r': esc += "\\r"; break;
            case '\t': esc += "\\t"; break;
            default: esc.push_back(c); break;
        }
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%d", status);
    return "{\"error\":{\"code\":" + std::string(buf) + ",\"message\":\"" + esc + "\"}}";
}

HttpServer::HttpServer(const std::string& host, int port, int num_threads)
    : host_(host), port_(port), num_threads_(num_threads) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket() failed");
    int yes = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#ifdef SO_REUSEPORT
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (host == "0.0.0.0" || host.empty()) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        throw std::runtime_error("invalid host: " + host);
    }
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error("bind() failed on port " + std::to_string(port_) +
                                 ": " + std::strerror(errno));
    }
    if (::listen(fd, 64) < 0) {
        ::close(fd);
        throw std::runtime_error("listen() failed");
    }
    listen_fd_ = fd;
    // Workers are spawned in start() (after running_ is set true) so their
    // cv_.wait predicate doesn't immediately evaluate to "not running".
}

HttpServer::~HttpServer() {
    stop();
    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
}

void HttpServer::stop() {
    if (!running_.exchange(false)) return;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    cv_.notify_all();
    if (accept_thread_.joinable()) accept_thread_.join();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        while (!conn_queue_.empty()) { ::close(conn_queue_.front()); conn_queue_.pop(); }
    }
    for (auto& w : workers_) {
        if (w.joinable()) {
            { std::lock_guard<std::mutex> lk(mtx_); conn_queue_.push(-1); }
            cv_.notify_one();
            w.join();
        }
    }
}

void HttpServer::route(const std::string& method, const std::string& path, Handler h) {
    Route r;
    r.method = to_lower(method);
    r.path = path;
    r.handler = std::move(h);
    std::lock_guard<std::mutex> lk(mtx_);
    routes_.push_back(std::move(r));
}

void HttpServer::start() {
    running_ = true;
    for (int i = 0; i < num_threads_; ++i)
        workers_.emplace_back([this] { worker_loop(); });
    accept_thread_ = std::thread([this] { accept_loop(); });
    while (running_) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop();
}

void HttpServer::accept_loop() {
    while (running_) {
        sockaddr_in client{};
        socklen_t len = sizeof(client);
        int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client), &len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (!running_) break;
            continue;
        }
        timeval tv{}; tv.tv_sec = 120; tv.tv_usec = 0;
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        {
            std::lock_guard<std::mutex> lk(mtx_);
            conn_queue_.push(fd);
        }
        cv_.notify_one();
    }
}

void HttpServer::worker_loop() {
    while (true) {
        int fd = -1;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return !conn_queue_.empty() || !running_; });
            if (!running_ && conn_queue_.empty()) return;
            fd = conn_queue_.front();
            conn_queue_.pop();
        }
        if (fd < 0) return;
        handle_connection(fd);
        ::close(fd);
    }
}

bool HttpServer::parse_request(int fd, HttpRequest& req) {
    std::string buf;
    auto found = [](const std::string& b) { return b.find("\r\n\r\n") != std::string::npos; };
    if (!read_until(fd, buf, 1 << 20, found)) return false;

    size_t hend = buf.find("\r\n\r\n");
    std::string head = buf.substr(0, hend);
    std::string body_extra = buf.substr(hend + 4);

    size_t eol = head.find("\r\n");
    if (eol == std::string::npos) return false;
    std::string line = head.substr(0, eol);
    {
        std::istringstream iss(line);
        iss >> req.method >> req.path;
        if (req.method.empty() || req.path.empty()) return false;
    }
    req.method = to_lower(req.method);
    size_t q = req.path.find('?');
    if (q != std::string::npos) {
        req.query = req.path.substr(q + 1);
        req.path = req.path.substr(0, q);
        req.params = parse_query(req.query);
    }

    size_t pos = eol + 2;
    while (pos < head.size()) {
        size_t nl = head.find("\r\n", pos);
        std::string hline = (nl == std::string::npos) ? head.substr(pos)
                                                       : head.substr(pos, nl - pos);
        if (hline.empty()) break;
        size_t colon = hline.find(':');
        if (colon != std::string::npos) {
            std::string k = to_lower(trim(hline.substr(0, colon)));
            std::string v = trim(hline.substr(colon + 1));
            req.headers[k] = v;
        }
        if (nl == std::string::npos) break;
        pos = nl + 2;
    }

    size_t content_len = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        try { content_len = static_cast<size_t>(std::stoul(it->second)); }
        catch (...) { content_len = 0; }
    }
    req.body = body_extra;
    if (req.body.size() < content_len) {
        size_t need = content_len - req.body.size();
        char chunk[8192];
        while (need > 0) {
            ssize_t n = recv(fd, chunk, std::min(need, sizeof(chunk)), 0);
            if (n <= 0) break;
            req.body.append(chunk, static_cast<size_t>(n));
            need -= static_cast<size_t>(n);
        }
    } else if (req.body.size() > content_len) {
        req.body.resize(content_len);
    }
    return true;
}

bool HttpServer::send_all_static(int fd, const char* data, size_t n) {
    return send_all_raw(fd, data, n);
}

void HttpServer::write_response(int fd, const HttpResponse& resp) {
    std::ostringstream os;
    os << "HTTP/1.1 " << resp.status << ' ' << reason(resp.status) << "\r\n";
    os << "Content-Type: " << resp.content_type << "\r\n";
    os << "Content-Length: " << resp.body.size() << "\r\n";
    os << "Connection: close\r\n";
    os << "Access-Control-Allow-Origin: *\r\n";
    os << "Access-Control-Allow-Headers: Content-Type\r\n";
    os << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n";
    std::string header = os.str();
    send_all_raw(fd, header.data(), header.size());
    send_all_raw(fd, resp.body.data(), resp.body.size());
}

void HttpServer::write_stream_headers(int fd) {
    std::ostringstream os;
    os << "HTTP/1.1 200 OK\r\n";
    os << "Content-Type: text/event-stream\r\n";
    os << "Cache-Control: no-cache\r\n";
    os << "Connection: close\r\n";
    os << "Access-Control-Allow-Origin: *\r\n";
    os << "Access-Control-Allow-Headers: Content-Type\r\n";
    os << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n";
    std::string header = os.str();
    send_all_raw(fd, header.data(), header.size());
}

void HttpServer::handle_connection(int fd) {
    HttpRequest req;
    if (!parse_request(fd, req)) {
        std::fprintf(stderr, "[http] malformed request\n");
        HttpResponse r; r.status = 400; r.body = json_error(400, "malformed request");
        write_response(fd, r);
        return;
    }

    if (req.method == "options") {
        HttpResponse r; r.status = 200; r.body = "{}";
        write_response(fd, r);
        return;
    }

    Route* match = nullptr;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& r : routes_)
            if (r.method == req.method && r.path == req.path) { match = &r; break; }
    }
    if (!match) {
        HttpResponse r; r.status = 404;
        r.body = json_error(404, "not found: " + req.method + " " + req.path);
        write_response(fd, r);
        return;
    }

    // Track whether the handler has started writing (so we don't double-write
    // on an exception, and so the catch-block can still send an error).
    std::atomic<bool> started{false};
    ResponseWriter w;
    w.send = [fd, &started](const HttpResponse& resp) {
        if (started.exchange(true)) return;
        write_response(fd, resp);
    };
    w.begin_stream = [fd, &started]() {
        if (started.exchange(true)) return;
        write_stream_headers(fd);
    };
    w.send_chunk = [fd](const std::string& chunk) {
        send_all_raw(fd, chunk.data(), chunk.size());
    };

    try {
        match->handler(req, w);
    } catch (const std::exception& e) {
        if (!started.exchange(true))
            write_response(fd, HttpResponse{500, "application/json", json_error(500, e.what())});
    }
}

}  // namespace mini_infer
