#include "core/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace mini_infer {

namespace {
std::string slurp_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Tokenizer: cannot read " + path);
    std::ostringstream o; o << f.rdbuf();
    return o.str();
}
void write_file(const std::string& path, const std::string& s) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Tokenizer: cannot write " + path);
    f << s;
}
std::string tmp_path(const std::string& tag) {
    char tmpl[] = "/tmp/mini_infer_tok_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) throw std::runtime_error("Tokenizer: mkstemp failed");
    ::close(fd);
    return std::string(tmpl) + tag;
}
}  // namespace

Tokenizer::Tokenizer(const std::string& tokenizer_json_path,
                     const std::string& python_exe)
    : tokenizer_path_(tokenizer_json_path), python_exe_(python_exe) {
    // Locate the helper. It ships next to the binaries under scripts/.
    // For dev runs, fall back to the source-tree path.
    const char* env_path = std::getenv("MINI_INFER_HELPER");
    if (env_path) {
        helper_path_ = env_path;
    } else {
        // Try to find the helper relative to the executable.
        // For dev: use the source tree path.
        helper_path_ = "scripts/tokenize_helper.py";
    }

    // Probe special tokens via the helper.
    const std::string out = tmp_path(".txt");
    const std::string cmd = python_exe_ + " " + helper_path_ +
        " specials --tokenizer " + tokenizer_path_ +
        " > " + out + " 2>/dev/null";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("Tokenizer: failed to probe specials (rc=" +
                                 std::to_string(rc) + ")");
    }
    const std::string content = slurp_file(out);
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string key = line.substr(0, pos);
        const int64_t val = std::stoll(line.substr(pos + 1));
        if      (key == "eos")       eos_id_      = val;
        else if (key == "bos")       bos_id_      = val;
        else if (key == "im_start")  im_start_id_ = val;
        else if (key == "im_end")    im_end_id_   = val;
    }
    std::remove(out.c_str());
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    const std::string in  = tmp_path(".in.txt");
    const std::string out = tmp_path(".out.bin");
    write_file(in, text);

    const std::string cmd = python_exe_ + " " + helper_path_ +
        " encode --tokenizer " + tokenizer_path_ +
        " --input " + in + " --output " + out;
    int rc = std::system(cmd.c_str());
    std::remove(in.c_str());
    if (rc != 0) {
        std::remove(out.c_str());
        throw std::runtime_error("Tokenizer::encode failed (rc=" +
                                 std::to_string(rc) + ")");
    }

    // Read raw bytes (int64 little-endian) into vector.
    std::ifstream f(out, std::ios::binary | std::ios::ate);
    if (!f) {
        std::remove(out.c_str());
        throw std::runtime_error("Tokenizer::encode: cannot read output");
    }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    if (bytes % 8 != 0) {
        std::remove(out.c_str());
        throw std::runtime_error("Tokenizer::encode: bad output size");
    }
    std::vector<int64_t> ids(bytes / 8);
    f.read(reinterpret_cast<char*>(ids.data()), bytes);
    std::remove(out.c_str());
    return ids;
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
    const std::string in  = tmp_path(".in.bin");
    const std::string out = tmp_path(".out.txt");
    std::ofstream f(in, std::ios::binary);
    f.write(reinterpret_cast<const char*>(ids.data()),
            static_cast<std::streamsize>(ids.size() * sizeof(int64_t)));
    f.close();

    const std::string cmd = python_exe_ + " " + helper_path_ +
        " decode --tokenizer " + tokenizer_path_ +
        " --input " + in + " --output " + out;
    int rc = std::system(cmd.c_str());
    std::remove(in.c_str());
    if (rc != 0) {
        std::remove(out.c_str());
        throw std::runtime_error("Tokenizer::decode failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const std::string text = slurp_file(out);
    std::remove(out.c_str());
    return text;
}

}  // namespace mini_infer