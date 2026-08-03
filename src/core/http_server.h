#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mini_infer {

/**
 * Minimal HTTP/1.1 request as parsed by HttpServer.
 *
 * `headers` keys are lower-cased. `params` are parsed from the query string.
 */
struct HttpRequest {
    std::string method;   // "GET" | "POST"
    std::string path;     // path without query string (e.g. "/generate")
    std::string query;    // raw query string (after '?'), may be empty
    std::string body;     // raw request body
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> params;
};

/** A normal (non-streaming) HTTP response. */
struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
};

/**
 * ResponseWriter — lets a single handler decide at runtime whether to send a
 * full JSON response OR an SSE stream. This avoids routing ambiguity for
 * endpoints like `/v1/chat/completions` that behave differently per request.
 *
 * Lifecycle: a handler MUST call exactly one of:
 *   • `send(HttpResponse)`            — one-shot JSON response, or
 *   • `begin_stream()` then any number of `send_chunk(...)` — SSE stream.
 * The connection closes when the handler returns.
 */
struct ResponseWriter {
    std::function<void(const HttpResponse&)> send;
    std::function<void()> begin_stream;
    std::function<void(const std::string&)> send_chunk;
};

/** Unified handler signature. */
using Handler = std::function<void(const HttpRequest&, ResponseWriter&)>;

/**
 * HttpServer — a tiny dependency-free HTTP/1.1 server built on POSIX sockets.
 *
 * Design goals for mini-infer:
 *   * No external HTTP library (keeps the build self-contained).
 *   * Fixed worker thread-pool; request handling calls into the single-
 *     threaded CUDA Engine safely (generation is serialized by a mutex
 *     supplied by the caller).
 *   * Supports both plain JSON responses and SSE streaming via a single
 *     `ResponseWriter` so one route can serve both modes.
 *
 * Not a general-purpose web server — just enough to serve the inference API.
 */
class HttpServer {
public:
    HttpServer(const std::string& host, int port, int num_threads = 4);
    ~HttpServer();

    /** Register a handler. Exact method + path match. */
    void route(const std::string& method, const std::string& path, Handler h);

    /** Blocking — spawns the accept loop. Returns when stop() is called. */
    void start();

    /** Signal the accept loop to stop and close the listening socket. */
    void stop();

private:
    struct Route {
        std::string method;
        std::string path;
        Handler handler;
    };

    void accept_loop();
    void worker_loop();
    void handle_connection(int fd);

    static bool parse_request(int fd, HttpRequest& req);
    static void write_response(int fd, const HttpResponse& resp);
    static void write_stream_headers(int fd);
    static bool send_all_static(int fd, const char* data, size_t n);

    std::string host_;
    int port_;
    int num_threads_ = 4;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<std::thread> workers_;

    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<int> conn_queue_;
    std::vector<Route> routes_;
};

/** Utility: parse a query string into key/value pairs (percent-decoded). */
std::unordered_map<std::string, std::string> parse_query(const std::string& q);

/** Utility: minimal JSON error body (no external dependency). */
std::string json_error(int status, const std::string& message);

}  // namespace mini_infer
