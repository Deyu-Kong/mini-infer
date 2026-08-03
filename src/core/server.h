#pragma once

#include <string>

namespace mini_infer {

/**
 * `mini_infer serve` — start the HTTP inference server (online service).
 *
 * Loads the model once into GPU memory and exposes a JSON + OpenAI-compatible
 * REST API. Routes:
 *   GET  /health
 *   GET  /v1/models
 *   POST /tokenize        {text}            -> {ids:[...]}
 *   POST /detokenize      {ids:[...]}       -> {text}
 *   POST /generate        {prompt,...}      -> {text, generated_tokens, ...}
 *   POST /v1/chat/completions  (OpenAI-compatible, supports stream:true SSE)
 *
 * argv here starts AFTER the "serve" subcommand.
 */
int run_serve(int argc, char** argv);

/**
 * `mini_infer client` — online CLI client.
 *
 * Connects to a running `mini_infer serve` instance and issues a chat
 * completion, optionally streaming the response to stdout.
 *
 * argv here starts AFTER the "client" subcommand.
 */
int run_client(int argc, char** argv);

}  // namespace mini_infer
