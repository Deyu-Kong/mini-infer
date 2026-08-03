#include "core/server.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/engine.h"
#include "core/http_client.h"
#include "core/http_server.h"
#include "core/tokenizer.h"
#include "model/model_config.h"
#include "model/safetensors_loader.h"
#include "model/transformer_model.h"
#include "speculative/draft_engine.h"
#include "speculative/spec_decoder.h"

namespace mini_infer {

namespace {

using json = nlohmann::json;

std::string basename(const std::string& path) {
    size_t slash = path.find_last_of('/');
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

// Build a ChatML prompt from a list of (role, content) messages. Falls back to
// a plain "Role: content" template when the tokenizer has no <|im_start|>.
std::string build_chatml(const Tokenizer& tok,
                         const std::vector<std::pair<std::string, std::string>>& msgs,
                         bool& has_im) {
    has_im = (tok.im_start_id() >= 0 && tok.im_end_id() >= 0);
    std::string out;
    if (has_im) {
        for (const auto& m : msgs)
            out += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
    } else {
        for (const auto& m : msgs) {
            if (m.first == "system")      out += "System: " + m.second + "\n";
            else if (m.first == "assistant") out += "Assistant: " + m.second + "\n";
            else                            out += "User: " + m.second + "\n";
        }
        out += "Assistant: ";
    }
    return out;
}

std::vector<int64_t> stop_ids_for(const Tokenizer& tok, bool has_im) {
    std::vector<int64_t> s;
    if (has_im && tok.im_end_id() >= 0) s.push_back(tok.im_end_id());
    if (tok.eos_token_id() >= 0) s.push_back(tok.eos_token_id());
    return s;
}

struct ServerContext {
    std::shared_ptr<TransformerModel> model;
    std::shared_ptr<TransformerModel> draft_model;
    std::shared_ptr<DraftEngine> draft_engine;
    std::shared_ptr<SpecDecoder> spec_decoder;
    std::shared_ptr<Engine> engine;
    std::unique_ptr<Tokenizer> tok;
    std::mutex gen_mutex;        // serialize generation (Engine is stateful)
    std::string model_name;
    int device = 0;
    int64_t max_seq_len = 2048;
    int64_t default_max_new_tokens = 512;
    bool has_draft = false;
};

struct GenParams {
    std::vector<int64_t> prompt_ids;
    std::vector<int64_t> stop_ids;
    int64_t max_new_tokens = 256;
    bool greedy = true;
    float temperature = 1.0f;
    float top_p = 0.9f;
    uint64_t seed = 42;
};

struct GenOutput {
    std::vector<int64_t> out_ids;
    std::string text;
    int64_t prompt_tokens = 0;
    int64_t generated_tokens = 0;
    double tokens_per_sec = 0.0;
    double elapsed = 0.0;
    std::string finish_reason = "length";
};

GenOutput do_generate(ServerContext& c, const GenParams& p) {
    std::lock_guard<std::mutex> lk(c.gen_mutex);
    GenOutput out;
    out.prompt_tokens = static_cast<int64_t>(p.prompt_ids.size());

    c.engine->clear_paged_sequences();
    if (p.greedy) {
        c.engine->set_sampling(SamplingMode::Greedy);
    } else {
        c.engine->set_sampling(SamplingMode::TopP, p.top_p, p.temperature, p.seed);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    if (c.has_draft) {
        if (p.greedy) c.spec_decoder->set_sampling_greedy();
        else          c.spec_decoder->set_sampling_top_p(p.top_p, p.temperature, p.seed);
        out.out_ids = c.spec_decoder->generate(p.prompt_ids, p.max_new_tokens, p.stop_ids);
    } else {
        out.out_ids = c.engine->generate_paged(p.prompt_ids, p.max_new_tokens, p.stop_ids);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    out.elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::vector<int64_t> gen_ids(out.out_ids.begin() + p.prompt_ids.size(),
                                 out.out_ids.end());
    out.generated_tokens = static_cast<int64_t>(gen_ids.size());
    out.text = c.tok->decode(gen_ids);
    out.tokens_per_sec = (out.elapsed > 0) ? out.generated_tokens / out.elapsed
                                          : c.engine->decode_tokens_per_sec();

    if (!gen_ids.empty()) {
        int64_t last = gen_ids.back();
        for (int64_t s : p.stop_ids)
            if (s == last) { out.finish_reason = "stop"; break; }
    }
    return out;
}

// Split text into small chunks for SSE emission. Breaks at word boundaries
// when feasible so streamed text stays readable.
std::vector<std::string> split_chunks(const std::string& text, size_t target = 8) {
    std::vector<std::string> chunks;
    if (text.empty()) return chunks;
    size_t i = 0;
    while (i < text.size()) {
        size_t end = std::min(i + target, text.size());
        if (end < text.size()) {
            size_t brk = text.find_first_of(" \n\t.,;:!?", end);
            if (brk != std::string::npos && brk - i <= target * 3) end = brk + 1;
        }
        chunks.push_back(text.substr(i, end - i));
        i = end;
    }
    return chunks;
}

void send_sse_chunk(ResponseWriter& w, const std::string& model_name,
                    const std::string& content) {
    json ev;
    ev["object"] = "chat.completion.chunk";
    ev["model"] = model_name;
    json choice; choice["index"] = 0;
    choice["delta"] = {{"content", content}};
    choice["finish_reason"] = nullptr;
    ev["choices"] = json::array({choice});
    w.send_chunk("data: " + ev.dump() + "\n\n");
}

struct ServeArgs {
    std::string model_dir;
    std::string draft_dir;
    std::string host = "0.0.0.0";
    int port = 8000;
    int device = 0;
    int64_t max_seq_len = 2048;
    int64_t max_new_tokens = 512;
    int gamma = 4;
    std::string python = "python3";
};

bool parse_serve_args(int argc, char** argv, ServeArgs& a, std::string& err) {
    auto need = [&](const std::string& f, int& i) -> std::string {
        if (i + 1 >= argc) { err = "missing value for " + f; return ""; }
        return argv[++i];
    };
    for (int i = 0; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--model")         a.model_dir = need(s, i);
        else if (s == "--draft" || s == "--spec-draft") a.draft_dir = need(s, i);
        else if (s == "--host")          a.host = need(s, i);
        else if (s == "--port")          a.port = std::stoi(need(s, i));
        else if (s == "--device")        a.device = std::stoi(need(s, i));
        else if (s == "--max-seq-len")   a.max_seq_len = std::stoll(need(s, i));
        else if (s == "--max-new-tokens") a.max_new_tokens = std::stoll(need(s, i));
        else if (s == "--gamma")         a.gamma = std::stoi(need(s, i));
        else if (s == "--python")        a.python = need(s, i);
        else if (s == "-h" || s == "--help") {
            std::printf(
                "Usage: mini_infer serve --model DIR [--draft DRAFT_DIR] "
                "[--host H] [--port P] [--device D] [--max-seq-len M] "
                "[--max-new-tokens N] [--gamma G] [--python PY]\n");
            std::exit(0);
        } else { err = "unknown flag: " + s; return false; }
    }
    if (a.model_dir.empty()) { err = "--model is required"; return false; }
    return true;
}

}  // namespace

int run_serve(int argc, char** argv) {
    ServeArgs args;
    std::string err;
    if (!parse_serve_args(argc, argv, args, err)) {
        std::fprintf(stderr, "[serve] %s\n", err.c_str());
        return 2;
    }

    ServerContext ctx;
    ctx.device = args.device;
    ctx.max_seq_len = args.max_seq_len;
    ctx.default_max_new_tokens = args.max_new_tokens;
    ctx.model_name = basename(args.model_dir);

    std::printf("[serve] loading model %s ...\n", args.model_dir.c_str());
    auto cfg = ModelConfig::load(args.model_dir + "/config.json");
    auto weight_idx = WeightIndex::load(args.model_dir);
    ctx.model = std::make_shared<TransformerModel>(cfg, args.device);
    ctx.model->load_weights(weight_idx);
    ctx.tok = std::make_unique<Tokenizer>(args.model_dir + "/tokenizer.json", args.python);
    ctx.engine = std::make_shared<Engine>(ctx.model, args.max_seq_len, args.device);

    if (!args.draft_dir.empty()) {
        std::printf("[serve] loading draft model %s ...\n", args.draft_dir.c_str());
        auto dcfg = ModelConfig::load(args.draft_dir + "/config.json");
        auto dwidx = WeightIndex::load(args.draft_dir);
        ctx.draft_model = std::make_shared<TransformerModel>(dcfg, args.device);
        ctx.draft_model->load_weights(dwidx);
        ctx.draft_engine = std::make_shared<DraftEngine>(ctx.draft_model, args.max_seq_len, args.device);
        ctx.spec_decoder = std::make_shared<SpecDecoder>(ctx.model, ctx.draft_engine,
                                                         args.max_seq_len, args.gamma, args.device);
        ctx.has_draft = true;
    }

    std::printf("[serve] model ready: %s (device=%d, max_seq_len=%ld, draft=%s)\n",
                ctx.model_name.c_str(), ctx.device, ctx.max_seq_len,
                ctx.has_draft ? "yes" : "no");

    HttpServer server(args.host, args.port);

    // GET /health
    server.route("GET", "/health", [&ctx](const HttpRequest&, ResponseWriter& w) {
        json j;
        j["status"] = "ok";
        j["model"] = ctx.model_name;
        j["device"] = ctx.device;
        j["draft"] = ctx.has_draft;
        j["max_seq_len"] = ctx.max_seq_len;
        HttpResponse r; r.body = j.dump(); w.send(r);
    });

    // GET /v1/models
    server.route("GET", "/v1/models", [&ctx](const HttpRequest&, ResponseWriter& w) {
        json data = json::array();
        data.push_back({{"id", ctx.model_name}, {"object", "model"},
                        {"created", 0}, {"owned_by", "mini-infer"}});
        json j; j["object"] = "list"; j["data"] = data;
        HttpResponse r; r.body = j.dump(); w.send(r);
    });

    // POST /tokenize
    server.route("POST", "/tokenize", [&ctx](const HttpRequest& req, ResponseWriter& w) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_null() || !j.contains("text")) {
            HttpResponse r; r.status = 400; r.body = json_error(400, "missing 'text'"); w.send(r); return;
        }
        auto ids = ctx.tok->encode(j["text"].get<std::string>());
        HttpResponse r; r.body = (json{{"ids", ids}}).dump(); w.send(r);
    });

    // POST /detokenize
    server.route("POST", "/detokenize", [&ctx](const HttpRequest& req, ResponseWriter& w) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_null() || !j.contains("ids")) {
            HttpResponse r; r.status = 400; r.body = json_error(400, "missing 'ids'"); w.send(r); return;
        }
        auto ids = j["ids"].get<std::vector<int64_t>>();
        HttpResponse r; r.body = (json{{"text", ctx.tok->decode(ids)}}).dump(); w.send(r);
    });

    // POST /generate
    server.route("POST", "/generate", [&ctx](const HttpRequest& req, ResponseWriter& w) {
        json j = json::parse(req.body, nullptr, false);
        if (j.is_null() || !j.contains("prompt")) {
            HttpResponse r; r.status = 400; r.body = json_error(400, "missing 'prompt'"); w.send(r); return;
        }
        GenParams p;
        std::string prompt = j["prompt"].get<std::string>();
        bool raw = j.value("raw", false);
        p.greedy = j.value("greedy", true);
        p.temperature = j.value("temperature", 1.0f);
        p.top_p = j.value("top_p", 0.9f);
        p.seed = j.value("seed", (uint64_t)42);
        p.max_new_tokens = j.value("max_new_tokens", (int64_t)ctx.default_max_new_tokens);

        bool has_im;
        if (raw) {
            p.prompt_ids = ctx.tok->encode(prompt);
            p.stop_ids = stop_ids_for(*ctx.tok, false);
        } else {
            std::string chat = build_chatml(*ctx.tok, {{"user", prompt}}, has_im);
            p.prompt_ids = ctx.tok->encode(chat);
            p.stop_ids = stop_ids_for(*ctx.tok, has_im);
        }

        GenOutput g = do_generate(ctx, p);

        json out;
        out["text"] = g.text;
        out["prompt_tokens"] = g.prompt_tokens;
        out["generated_tokens"] = g.generated_tokens;
        out["tokens_per_sec"] = g.tokens_per_sec;
        out["elapsed"] = g.elapsed;
        out["finish_reason"] = g.finish_reason;
        HttpResponse r; r.body = out.dump(); w.send(r);
    });

    // POST /v1/chat/completions (OpenAI-compatible; stream:true => SSE)
    server.route("POST", "/v1/chat/completions",
        [&ctx](const HttpRequest& req, ResponseWriter& w) {
            json j = json::parse(req.body, nullptr, false);
            if (j.is_null() || !j.contains("messages")) {
                HttpResponse r; r.status = 400; r.body = json_error(400, "missing 'messages'");
                w.send(r); return;
            }
            std::vector<std::pair<std::string, std::string>> msgs;
            for (const auto& m : j["messages"])
                msgs.push_back({m.value("role", "user"), m.value("content", "")});
            bool stream = j.value("stream", false);
            GenParams p;
            p.greedy = j.value("greedy", true);
            p.temperature = j.value("temperature", 1.0f);
            p.top_p = j.value("top_p", 0.9f);
            p.seed = j.value("seed", (uint64_t)42);
            p.max_new_tokens = j.value("max_tokens",
                           j.value("max_new_tokens", (int64_t)ctx.default_max_new_tokens));

            bool has_im;
            std::string chat = build_chatml(*ctx.tok, msgs, has_im);
            p.prompt_ids = ctx.tok->encode(chat);
            p.stop_ids = stop_ids_for(*ctx.tok, has_im);

            GenOutput g = do_generate(ctx, p);

            if (stream) {
                w.begin_stream();
                // initial role delta
                {
                    json ev; ev["object"] = "chat.completion.chunk"; ev["model"] = ctx.model_name;
                    json ch; ch["index"] = 0; ch["delta"] = {{"role", "assistant"}};
                    ch["finish_reason"] = nullptr; ev["choices"] = json::array({ch});
                    w.send_chunk("data: " + ev.dump() + "\n\n");
                }
                for (const auto& chunk : split_chunks(g.text))
                    send_sse_chunk(w, ctx.model_name, chunk);
                // final chunk
                {
                    json ev; ev["object"] = "chat.completion.chunk"; ev["model"] = ctx.model_name;
                    json ch; ch["index"] = 0; ch["delta"] = json::object();
                    ch["finish_reason"] = g.finish_reason; ev["choices"] = json::array({ch});
                    w.send_chunk("data: " + ev.dump() + "\n\n");
                }
                w.send_chunk("data: [DONE]\n\n");
            } else {
                json out;
                out["id"] = "chatcmpl-mini-infer";
                out["object"] = "chat.completion";
                out["created"] = static_cast<int64_t>(std::time(nullptr));
                out["model"] = ctx.model_name;
                json choice; choice["index"] = 0;
                choice["message"] = {{"role", "assistant"}, {"content", g.text}};
                choice["finish_reason"] = g.finish_reason;
                out["choices"] = json::array({choice});
                out["usage"] = {{"prompt_tokens", g.prompt_tokens},
                                {"completion_tokens", g.generated_tokens},
                                {"total_tokens", g.prompt_tokens + g.generated_tokens},
                                {"tokens_per_sec", g.tokens_per_sec}};
                HttpResponse r; r.body = out.dump(); w.send(r);
            }
        });

    std::printf("[serve] listening on http://%s:%d\n", args.host.c_str(), args.port);
    std::printf("[serve] endpoints:\n");
    std::printf("  GET  /health\n");
    std::printf("  GET  /v1/models\n");
    std::printf("  POST /generate\n");
    std::printf("  POST /v1/chat/completions  (stream=true supported)\n");
    std::printf("  POST /tokenize | /detokenize\n");
    std::fflush(stdout);

    server.start();
    return 0;
}

// ---------------------------------------------------------------------------
// `client` (online)
// ---------------------------------------------------------------------------
int run_client(int argc, char** argv) {
    std::string endpoint;
    std::string prompt = "Hello!";
    std::string system_prompt;
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 0.9f;
    bool greedy = false;
    bool stream = false;

    auto need = [&](const std::string& f, int& i) -> std::string {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", f.c_str()); std::exit(2); }
        return argv[++i];
    };
    for (int i = 0; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--remote" || s == "--endpoint") endpoint = need(s, i);
        else if (s == "--prompt")          prompt = need(s, i);
        else if (s == "--system")          system_prompt = need(s, i);
        else if (s == "--max-tokens" || s == "--max-new-tokens") max_tokens = std::stoi(need(s, i));
        else if (s == "--temperature")     temperature = std::stof(need(s, i));
        else if (s == "--top-p")           top_p = std::stof(need(s, i));
        else if (s == "--greedy")          greedy = true;
        else if (s == "--stream")          stream = true;
        else if (s == "-h" || s == "--help") {
            std::printf("Usage: mini_infer client --remote http://host:port --prompt TEXT "
                        "[--system S] [--max-tokens N] [--temperature T] [--top-p P] "
                        "[--greedy] [--stream]\n");
            std::exit(0);
        } else { std::fprintf(stderr, "unknown flag: %s\n", s.c_str()); return 2; }
    }
    if (endpoint.empty()) { std::fprintf(stderr, "--remote is required\n"); return 2; }

    HttpClient client(endpoint);

    json messages = json::array();
    if (!system_prompt.empty())
        messages.push_back({{"role", "system"}, {"content", system_prompt}});
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json body;
    body["messages"] = messages;
    body["max_tokens"] = max_tokens;
    body["temperature"] = temperature;
    body["top_p"] = top_p;
    body["greedy"] = greedy;
    body["stream"] = stream;

    const std::string path = "/v1/chat/completions";

    if (stream) {
        std::string sse_buf;
        bool ok = client.post_stream(path, body.dump(), [&](const std::string& chunk) {
            sse_buf += chunk;
            size_t pos = 0;
            while (true) {
                size_t nl = sse_buf.find('\n', pos);
                if (nl == std::string::npos) break;
                std::string line = sse_buf.substr(pos, nl - pos);
                pos = nl + 1;
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (line.rfind("data:", 0) == 0) {
                    std::string payload = line.substr(5);
                    while (!payload.empty() && payload[0] == ' ') payload.erase(0, 1);
                    if (payload == "[DONE]") return;
                    try {
                        json ev = json::parse(payload);
                        if (ev.contains("choices") && !ev["choices"].empty()) {
                            auto delta = ev["choices"][0].value("delta", json::object());
                            if (delta.contains("content"))
                                std::fputs(delta["content"].get<std::string>().c_str(), stdout),
                                std::fflush(stdout);
                        }
                    } catch (...) { /* ignore malformed */ }
                }
            }
            sse_buf.erase(0, pos);
        });
        std::fputc('\n', stdout);
        if (!ok) { std::fprintf(stderr, "\n[client] connection failed\n"); return 1; }
    } else {
        HttpResp r = client.post(path, body.dump());
        if (!r.error.empty() || r.status == 0) {
            std::fprintf(stderr, "[client] request failed: %s\n", r.error.c_str());
            return 1;
        }
        if (r.status >= 400) {
            std::fprintf(stderr, "[client] HTTP %d: %s\n", r.status, r.body.c_str());
            return 1;
        }
        try {
            json j = json::parse(r.body);
            if (j.contains("choices") && !j["choices"].empty())
                std::printf("%s\n", j["choices"][0]["message"]["content"].get<std::string>().c_str());
            if (j.contains("usage")) {
                auto u = j["usage"];
                std::fprintf(stderr, "[client] tokens=%lld/%lld, tps=%.2f tok/s\n",
                             static_cast<long long>(u.value("prompt_tokens", (int64_t)0)),
                             static_cast<long long>(u.value("completion_tokens", (int64_t)0)),
                             u.value("tokens_per_sec", 0.0));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[client] parse error: %s\nbody: %s\n", e.what(), r.body.c_str());
            return 1;
        }
    }
    return 0;
}

}  // namespace mini_infer
