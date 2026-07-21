#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_infer {

/**
 * Computation graph node — a single op producing one output tensor.
 *
 *  - `name`   : human-readable, used for `print()`.
 *  - `op`     : op kind ("rmsnorm", "matmul", "add", "sdpa", "embedding",
 *               "rope", "swiglu_mlp", ...). The actual execution is
 *               dispatched elsewhere (W4+); the graph is only metadata.
 *  - `inputs` : ordered list of input tensor names (topo-sorted predecessors).
 *  - `output` : the output tensor name.
 *
 * Week-3 scope: linear-chain only (no fan-out / fan-in > 1). Topological
 * order is just `nodes_.size()` after insertion. Branching ops will land
 * in Week 5 alongside PagedAttention / speculative-decoding.
 */
struct GraphNode {
    std::string               name;
    std::string               op;
    std::vector<std::string>  inputs;
    std::string               output;
};

class Graph {
public:
    Graph() = default;

    // -------- inputs / outputs ---------------------------------------------
    GraphNode& add_input(const std::string& name);
    GraphNode& add_output(const std::string& name);
    const std::vector<std::string>& inputs()  const { return inputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

    // -------- ops ----------------------------------------------------------
    GraphNode& add_node(const std::string& name, const std::string& op,
                        const std::vector<std::string>& inputs,
                        const std::string& output);

    const std::vector<GraphNode>& nodes() const { return nodes_; }
    std::size_t size() const { return nodes_.size(); }
    void clear();

    // -------- analysis -----------------------------------------------------
    // Linear-chain topo order (the week-3 contract). Throws if any node has
    // multiple inputs whose order is ambiguous or if there is a cycle.
    std::vector<const GraphNode*> topological_order() const;

    // Pretty-print the graph to `os`. `indent` is in spaces.
    void print(std::ostream& os, int indent = 2) const;

private:
    std::vector<GraphNode>        nodes_;
    std::vector<std::string>      inputs_;
    std::vector<std::string>      outputs_;
    std::unordered_map<std::string, std::size_t> index_of_;  // tensor name -> node
};

}  // namespace mini_infer