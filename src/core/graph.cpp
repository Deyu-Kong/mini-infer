#include "core/graph.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mini_infer {

GraphNode& Graph::add_input(const std::string& name) {
    GraphNode n;
    n.name = "input:" + name;
    n.op   = "input";
    n.output = name;
    nodes_.push_back(std::move(n));
    index_of_[name] = nodes_.size() - 1;
    inputs_.push_back(name);
    return nodes_.back();
}

GraphNode& Graph::add_output(const std::string& name) {
    outputs_.push_back(name);
    return nodes_.back();  // last node (placeholder, never dereferenced)
}

GraphNode& Graph::add_node(const std::string& name, const std::string& op,
                           const std::vector<std::string>& inputs,
                           const std::string& output) {
    GraphNode n;
    n.name = name;
    n.op   = op;
    n.inputs = inputs;
    n.output = output;
    nodes_.push_back(std::move(n));
    index_of_[output] = nodes_.size() - 1;
    return nodes_.back();
}

void Graph::clear() {
    nodes_.clear();
    inputs_.clear();
    outputs_.clear();
    index_of_.clear();
}

std::vector<const GraphNode*> Graph::topological_order() const {
    std::vector<const GraphNode*> order;
    order.reserve(nodes_.size());

    // Linear-chain check: each non-input node's first input must be the
    // previous node's output. We allow a node to have additional inputs
    // (e.g. "positions", "weights"), but at most one of them is a "main"
    // dependency on the previous node.
    std::unordered_set<std::string> produced;
    for (const auto& in : inputs_) produced.insert(in);

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto& n = nodes_[i];
        if (n.op == "input") { order.push_back(&n); continue; }
        bool prev_ok = false;
        for (const auto& dep : n.inputs) {
            if (produced.count(dep)) { prev_ok = true; break; }
        }
        if (!prev_ok && !n.inputs.empty()) {
            throw std::runtime_error("Graph: node " + n.name +
                                     " has no dependency in chain");
        }
        order.push_back(&n);
        produced.insert(n.output);
    }

    // Detect accidental duplicates (same output name produced twice).
    if (produced.size() != nodes_.size()) {
        throw std::runtime_error("Graph: duplicate output names detected");
    }
    return order;
}

void Graph::print(std::ostream& os, int indent) const {
    const std::string pad(indent, ' ');
    os << pad << "Graph (" << nodes_.size() << " nodes)\n";
    os << pad << "  inputs : ";
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (i) os << ", ";
        os << inputs_[i];
    }
    os << "\n";
    for (const auto& n : nodes_) {
        if (n.op == "input") continue;
        os << pad << "  " << n.name << " : " << n.op << "(";
        for (size_t i = 0; i < n.inputs.size(); ++i) {
            if (i) os << ", ";
            os << n.inputs[i];
        }
        os << ") -> " << n.output << "\n";
    }
    os << pad << "  outputs: ";
    for (size_t i = 0; i < outputs_.size(); ++i) {
        if (i) os << ", ";
        os << outputs_[i];
    }
    os << "\n";
}

}  // namespace mini_infer