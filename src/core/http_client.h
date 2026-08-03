#pragma once

#include <functional>
#include <string>

namespace mini_infer {

/**
 * HttpClient — minimal HTTP/1.1 client over raw POSIX sockets.
 *
 * Implemented without libcurl so the engine stays dependency-free (only
 * nlohmann/json, which is already vendored, is used by callers for bodies).
 * Sufficient for the mini-infer online CLI to POST JSON to a running
 * `mini_infer serve` instance and read either a full JSON response or an
 * SSE stream.
 *
 * `endpoint` is `http://host:port` (path is supplied per-call).
 */
struct HttpResp {
    int status = 0;        // HTTP status code (0 on transport error)
    std::string body;      // full response body (for non-streaming)
    std::string error;     // non-empty if a transport error occurred
};

class HttpClient {
public:
    explicit HttpClient(const std::string& endpoint, int connect_timeout_sec = 10);

    /** Parse `host:port` and path out of an endpoint + path. */
    static bool parse_endpoint(const std::string& endpoint,
                               std::string& host, int& port);

    /** Synchronous JSON POST. Returns the full response body. */
    HttpResp post(const std::string& path, const std::string& json_body,
                 const std::string& content_type = "application/json");

    /** GET that returns the full body. */
    HttpResp get(const std::string& path);

    /**
     * Streaming POST. Calls `on_chunk` for each piece of the response body
     * as it arrives (the server uses `Connection: close`, so EOF terminates).
     * Useful for consuming SSE `data:` events from /v1/chat/completions.
     */
    bool post_stream(const std::string& path, const std::string& json_body,
                     const std::function<void(const std::string&)>& on_chunk,
                     const std::string& content_type = "application/json");

private:
    std::string host_;
    int port_ = 80;
    int connect_timeout_sec_ = 10;

    int connect_() const;
    static bool send_all_(int fd, const std::string& s);
    static bool read_until_headers_(int fd, std::string& buf, size_t& header_end);
};

}  // namespace mini_infer
