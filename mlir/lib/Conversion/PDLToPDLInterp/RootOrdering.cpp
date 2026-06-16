//===- RootOrdering.cpp - Optimal root ordering ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An implementation of Edmonds' optimal branching algorithm. This is a
// directed analogue of the minimum spanning tree problem for a given root.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/RootOrdering.h"

#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <queue>
#include <utility>

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

/// Returns the cycle implied by the specified parent relation, starting at the
/// given node.
static SmallVector<Value> getCycle(const DenseMap<Value, Value> &parents,
                                   Value rep) {
  SmallVector<Value> cycle;
  Value node = rep;
  do {
    cycle.push_back(node);
    node = parents.lookup(node);
    assert(node && "got an empty value in the cycle");
  } while (node != rep);
  return cycle;
}

/// Contracts the specified cycle in the given graph in-place.
/// The parentsCost map specifies, for each node in the cycle, the lowest cost
/// among the edges entering that node. Then, the nodes in the cycle C are
/// replaced with a single node v_C (the first node in the cycle). All edges
/// (u, v) entering the cycle, v \in C, are replaced with a single edge
/// (u, v_C) with an appropriately chosen cost, and the selected node v is
/// marked in the output map actualTarget[u]. All edges (u, v) leaving the
/// cycle, u \in C, are replaced with a single edge (v_C, v), and the selected
/// node u is marked in the ouptut map actualSource[v].
static void contract(RootOrderingGraph &graph, ArrayRef<Value> cycle,
                     const DenseMap<Value, unsigned> &parentDepths,
                     DenseMap<Value, Value> &actualSource,
                     DenseMap<Value, Value> &actualTarget) {
  Value rep = cycle.front();
  DenseSet<Value> cycleSet(cycle.begin(), cycle.end());

  // Now, contract the cycle, marking the actual sources and targets.
  DenseMap<Value, RootOrderingEntry> repEntries;
  for (auto &[target, edges] : graph) {
    if (cycleSet.contains(target)) {
      // Target in the cycle => edges incoming to the cycle or within the cycle.
      unsigned parentDepth = parentDepths.lookup(target);
      for (const auto &inner : edges) {
        Value source = inner.first;
        // Ignore edges within the cycle.
        if (cycleSet.contains(source))
          continue;

        // Edge incoming to the cycle.
        std::pair<unsigned, unsigned> cost = inner.second.cost;
        assert(parentDepth <= cost.first && "invalid parent depth");

        // Subtract the cost of the parent within the cycle from the cost of
        // the edge incoming to the cycle. This update ensures that the cost
        // of the minimum-weight spanning arborescence of the entire graph is
        // the cost of arborescence for the contracted graph plus the cost of
        // the cycle, no matter which edge in the cycle we choose to drop.
        cost.first -= parentDepth;
        auto it = repEntries.find(source);
        if (it == repEntries.end() || it->second.cost > cost) {
          actualTarget[source] = target;
          // Do not bother populating the connector (the connector is only
          // relevant for the final traversal, not for the optimal branching).
          repEntries[source].cost = cost;
        }
      }
      // Defer erasing graph[target] until after the loop; backward-shift
      // erase would otherwise invalidate the surrounding iterator.
    } else {
      // Target not in cycle => edges going away from or unrelated to the cycle.
      Value bestSource;
      std::pair<unsigned, unsigned> bestCost;
      edges.remove_if([&](const auto &inner) {
        Value source = inner.first;
        if (!cycleSet.contains(source))
          return false;
        // Going-away edge => get its cost and erase it.
        if (!bestSource || bestCost > inner.second.cost) {
          bestSource = source;
          bestCost = inner.second.cost;
        }
        return true;
      });

      // There were going-away edges, contract them.
      if (bestSource) {
        edges[rep].cost = bestCost;
        actualSource[target] = bestSource;
      }
    }
  }

  // Erase all in-cycle nodes from the graph. Done after the iteration above
  // because backward-shift erase relocates surviving entries.
  for (Value node : cycle)
    graph.erase(node);

  // Store the edges to the representative.
  graph[rep] = std::move(repEntries);
}

OptimalBranching::OptimalBranching(RootOrderingGraph graph, Value root)
    : graph(std::move(graph)), root(root) {}

unsigned OptimalBranching::solve() {
  // Initialize the parents and total cost.
  parents.clear();
  parents[root] = Value();
  unsigned totalCost = 0;

  // A map that stores the cost of the optimal local choice for each node
  // in a directed cycle. This map is cleared every time we seed the search.
  DenseMap<Value, unsigned> parentDepths;
  parentDepths.reserve(graph.size());

  // Determine if the optimal local choice results in an acyclic graph. This is
  // done by computing the optimal local choice and traversing up the computed
  // parents. On success, `parents` will contain the parent of each node.
  for (const auto &outer : graph) {
    Value node = outer.first;
    if (parents.count(node)) // already visited
      continue;

    // Follow the trail of best sources until we reach an already visited node.
    // The code will assert if we cannot reach an already visited node, i.e.,
    // the graph is not strongly connected.
    parentDepths.clear();
    do {
      auto it = graph.find(node);
      assert(it != graph.end() && "the graph is not strongly connected");

      // Find the best local parent, taking into account both the depth and the
      // tie breaking rules.
      Value &bestSource = parents[node];
      std::pair<unsigned, unsigned> bestCost;
      for (const auto &inner : it->second) {
        const RootOrderingEntry &entry = inner.second;
        if (!bestSource /* initial */ || bestCost > entry.cost) {
          bestSource = inner.first;
          bestCost = entry.cost;
        }
      }
      assert(bestSource && "the graph is not strongly connected");
      parentDepths[node] = bestCost.first;
      node = bestSource;
      totalCost += bestCost.first;
    } while (!parents.count(node));

    // If we reached a non-root node, we have a cycle.
    if (parentDepths.count(node)) {
      // Determine the cycle starting at the representative node.
      SmallVector<Value> cycle = getCycle(parents, node);

      // The following maps disambiguate the source / target of the edges
      // going out of / into the cycle.
      DenseMap<Value, Value> actualSource, actualTarget;

      // Contract the cycle and recurse.
      contract(graph, cycle, parentDepths, actualSource, actualTarget);
      totalCost = solve();

      // Redirect the going-away edges.
      for (auto &p : parents)
        if (p.second == node)
          // The parent is the node representating the cycle; replace it
          // with the actual (best) source in the cycle.
          p.second = actualSource.lookup(p.first);

      // Redirect the unique incoming edge and copy the cycle.
      Value parent = parents.lookup(node);
      Value entry = actualTarget.lookup(parent);
      cycle.push_back(node); // complete the cycle
      for (size_t i = 0, e = cycle.size() - 1; i < e; ++i) {
        totalCost += parentDepths.lookup(cycle[i]);
        if (cycle[i] == entry)
          parents[cycle[i]] = parent; // break the cycle
        else
          parents[cycle[i]] = cycle[i + 1];
      }

      // `parents` has a complete solution.
      break;
    }
  }

  return totalCost;
}

OptimalBranching::EdgeList
OptimalBranching::preOrderTraversal(ArrayRef<Value> nodes) const {
  // Invert the parent mapping.
  DenseMap<Value, std::vector<Value>> children;
  for (Value node : nodes) {
    if (node != root) {
      Value parent = parents.lookup(node);
      assert(parent && "invalid parent");
      children[parent].push_back(node);
    }
  }

  // The result which simultaneously acts as a queue.
  EdgeList result;
  result.reserve(nodes.size());
  result.emplace_back(root, Value());

  // Perform a BFS, pushing into the queue.
  for (size_t i = 0; i < result.size(); ++i) {
    Value node = result[i].first;
    for (Value child : children[node])
      result.emplace_back(child, node);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Shared PDL pattern helpers
//===----------------------------------------------------------------------===//

unsigned pdl_to_pdl_interp::getNumNonRangeValues(ValueRange values) {
  return llvm::count_if(values.getTypes(),
                        [](Type type) { return !isa<pdl::RangeType>(type); });
}

bool pdl_to_pdl_interp::useOperandGroup(pdl::OperationOp op, unsigned index) {
  OperandRange operands = op.getOperandValues();
  assert(index < operands.size() && "operand index out of range");
  for (unsigned i = 0; i <= index; ++i)
    if (isa<pdl::RangeType>(operands[i].getType()))
      return true;
  return false;
}

SmallVector<Value> pdl_to_pdl_interp::detectRoots(pdl::PatternOp pattern) {
  // First, collect all the operations that are used as operands
  // to other operations. These are not roots by default.
  DenseSet<Value> used;
  for (auto operationOp : pattern.getBodyRegion().getOps<pdl::OperationOp>()) {
    for (Value operand : operationOp.getOperandValues())
      TypeSwitch<Operation *>(operand.getDefiningOp())
          .Case<pdl::ResultOp, pdl::ResultsOp>(
              [&used](auto resultOp) { used.insert(resultOp.getParent()); });
  }

  // Remove the specified root from the use set, so that we can
  // always select it as a root, even if it is used by other operations.
  if (Value root = pattern.getRewriter().getRoot())
    used.erase(root);

  // Finally, collect all the unused operations.
  SmallVector<Value> roots;
  for (Value operationOp : pattern.getBodyRegion().getOps<pdl::OperationOp>())
    if (!used.contains(operationOp))
      roots.push_back(operationOp);

  return roots;
}

void pdl_to_pdl_interp::buildCostGraph(ArrayRef<Value> roots,
                                       RootOrderingGraph &graph,
                                       ParentMaps &parentMaps) {

  // The entry of a queue. The entry consists of the following items:
  // * the value in the DAG underneath the root;
  // * the parent of the value;
  // * the operand index of the value in its parent;
  // * the depth of the visited value.
  struct Entry {
    Entry(Value value, Value parent, std::optional<unsigned> index,
          unsigned depth)
        : value(value), parent(parent), index(index), depth(depth) {}

    Value value;
    Value parent;
    std::optional<unsigned> index;
    unsigned depth;
  };

  // A root of a value and its depth (distance from root to the value).
  struct RootDepth {
    Value root;
    unsigned depth = 0;
  };

  // Map from candidate connector values to their roots and depths. Using a
  // small vector with 1 entry because most values belong to a single root.
  llvm::MapVector<Value, SmallVector<RootDepth, 1>> connectorsRootsDepths;

  // Perform a breadth-first traversal of the op DAG rooted at each root.
  for (Value root : roots) {
    // The queue of visited values. A value may be present multiple times in
    // the queue, for multiple parents. We only accept the first occurrence,
    // which is guaranteed to have the lowest depth.
    std::queue<Entry> toVisit;
    toVisit.emplace(root, Value(), 0, 0);

    // The map from value to its parent for the current root.
    DenseMap<Value, OpIndex> &parentMap = parentMaps[root];

    while (!toVisit.empty()) {
      Entry entry = toVisit.front();
      toVisit.pop();
      // Skip if already visited.
      if (!parentMap.insert({entry.value, {entry.parent, entry.index}}).second)
        continue;

      // Mark the root and depth of the value.
      connectorsRootsDepths[entry.value].push_back({root, entry.depth});

      // Traverse the operands of an operation and result ops.
      // We intentionally do not traverse attributes and types, because those
      // are expensive to join on.
      TypeSwitch<Operation *>(entry.value.getDefiningOp())
          .Case([&](pdl::OperationOp operationOp) {
            OperandRange operands = operationOp.getOperandValues();
            // Special case when we pass all the operands in one range.
            // For those, the index is empty.
            if (operands.size() == 1 &&
                isa<pdl::RangeType>(operands[0].getType())) {
              toVisit.emplace(operands[0], entry.value, std::nullopt,
                              entry.depth + 1);
              return;
            }

            // Default case: visit all the operands.
            for (const auto &p :
                 llvm::enumerate(operationOp.getOperandValues()))
              toVisit.emplace(p.value(), entry.value, p.index(),
                              entry.depth + 1);
          })
          .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto resultOp) {
            toVisit.emplace(resultOp.getParent(), entry.value,
                            resultOp.getIndex(), entry.depth);
          });
    }
  }

  // Now build the cost graph.
  // This is simply a minimum over all depths for the target root.
  unsigned nextID = 0;
  for (const auto &connectorRootsDepths : connectorsRootsDepths) {
    Value value = connectorRootsDepths.first;
    ArrayRef<RootDepth> rootsDepths = connectorRootsDepths.second;
    // If there is only one root for this value, this will not trigger
    // any edges in the cost graph (a perf optimization).
    if (rootsDepths.size() == 1)
      continue;

    for (const RootDepth &p : rootsDepths) {
      for (const RootDepth &q : rootsDepths) {
        if (&p == &q)
          continue;
        // Insert or retrieve the property of edge from p to q.
        RootOrderingEntry &entry = graph[q.root][p.root];
        if (!entry.connector /* new edge */ || entry.cost.first > q.depth) {
          if (!entry.connector)
            entry.cost.second = nextID++;
          entry.cost.first = q.depth;
          entry.connector = value;
        }
      }
    }
  }

  assert((llvm::hasSingleElement(roots) || graph.size() == roots.size()) &&
         "the pattern contains a candidate root disconnected from the others");
}
