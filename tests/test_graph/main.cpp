/**
 * test_graph — verify Graph insertion, linear-chain topo sort, and the
 * Qwen2.5 single-block pattern (RMSNorm -> Attn -> +res -> RMSNorm ->
 * MLP -> +res).
 */
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "core/graph.h"

using mini_infer::Graph;
using mini_infer::GraphNode;

static int g_failures = 0;
#define EXPECT(cond, msg)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__,     \
                         msg);                                                \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

int main() {
    std::printf("mini-infer :: graph test\n");
    std::printf("------------------------\n");

    // 1. Empty graph
    {
        Graph g;
        EXPECT(g.size() == 0, "empty graph");
        EXPECT(g.inputs().empty(), "no inputs");
        EXPECT(g.outputs().empty(), "no outputs");
        auto order = g.topological_order();
        EXPECT(order.empty(), "empty topo");
    }

    // 2. Single input + single output (linear chain of one)
    {
        Graph g;
        g.add_input("x");
        g.add_node("id", "identity", {"x"}, "y");
        g.add_output("y");
        auto order = g.topological_order();
        EXPECT(order.size() == 2, "topo size");
        EXPECT(order[0]->op == "input", "first is input");
        EXPECT(order[1]->op == "identity", "second is identity");
    }

    // 3. Qwen2.5 single-block pattern
    {
        Graph g;
        g.add_input("hidden");
        g.add_input("positions");
        g.add_node("attn_q", "rope",   {"hidden", "positions"}, "q");
        g.add_node("attn_k", "rope",   {"hidden", "positions"}, "k");
        g.add_node("sdpa",   "sdpa",   {"q", "k", "v"},        "attn_out");
        g.add_node("o_proj", "matmul", {"attn_out"},          "attn_proj");
        g.add_node("res1",   "add",    {"hidden", "attn_proj"}, "hidden2");
        g.add_node("rms2",   "rmsnorm", {"hidden2"},           "h_norm");
        g.add_node("mlp",    "swiglu_mlp", {"h_norm"},        "mlp_out");
        g.add_node("res2",   "add",    {"hidden2", "mlp_out"}, "next");
        g.add_output("next");
        auto order = g.topological_order();
        EXPECT(order.size() == 10, "10 nodes (2 inputs + 8 ops)");
        std::printf("  Qwen2.5 single block topo order:\n");
        for (auto* n : order) std::printf("    %s : %s\n",
                                          n->name.c_str(), n->op.c_str());
    }

    // 4. Print the block to a stringstream and check it contains expected parts.
    {
        Graph g;
        g.add_input("hidden");
        g.add_node("rms1", "rmsnorm", {"hidden"}, "h1");
        g.add_node("attn", "sdpa",    {"h1"},    "a");
        g.add_node("add",  "add",     {"hidden", "a"}, "out");
        g.add_output("out");
        std::ostringstream o;
        g.print(o);
        const std::string s = o.str();
        EXPECT(s.find("rms1") != std::string::npos, "print mentions rms1");
        EXPECT(s.find("attn") != std::string::npos, "print mentions attn");
        EXPECT(s.find("Graph (") != std::string::npos, "print header");
    }

    if (g_failures == 0) { std::printf("\nALL OK\n"); return 0; }
    std::fprintf(stderr, "\n%d failures\n", g_failures);
    return 1;
}