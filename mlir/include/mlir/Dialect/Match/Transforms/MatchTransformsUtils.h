//===- MatchTransformsUtils.h - Shared match transform helpers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared between the `match` dialect transforms (combine-matchers and
// regroup-operation-accesses). These encode the SSA-native reasoning both
// passes lean on: dependency edges are def-use edges, and pure navigation ops
// can be freely re-placed as long as they still dominate their uses.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATCH_TRANSFORMS_MATCHTRANSFORMSUTILS_H
#define MLIR_DIALECT_MATCH_TRANSFORMS_MATCHTRANSFORMSUTILS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <iterator>

namespace mlir {
class Region;

namespace match {

/// Pure navigation ops: side-effect-free value producers that can be freely
/// re-placed as long as they still dominate their uses. `get_each` is
/// deliberately excluded: it lowers to a `foreach` loop and therefore carries
/// control flow.
bool isPureNavigationOp(Operation *op);

/// SSA-native dependency: `a` must precede `b` iff `b` consumes any result of
/// `a`. No separate operand-ID / result-ID bookkeeping needed.
inline bool dependsOn(Operation *a, Operation *b) {
  for (Value operand : b->getOperands())
    if (operand.getDefiningOp() == a)
      return true;
  return false;
}

/// Stable topological sort of the range `[begin, end)` under the partial order
/// `cmp` (`cmp(a, b)` == "a must sort before b"). Preserves the incoming order
/// among elements not otherwise constrained, so it can be used as a
/// dominance-preserving backstop after an unrelated stable sort.
template <typename Iterator, typename Compare>
void stableTopologicalSort(Iterator begin, Iterator end, Compare cmp) {
  while (begin != end) {
    llvm::SmallPtrSet<typename std::iterator_traits<Iterator>::value_type, 16>
        sortBeforeOthers;
    for (auto i = begin; i != end; ++i) {
      if (std::none_of(begin, end, [&](auto const &b) { return cmp(b, *i); }))
        sortBeforeOthers.insert(*i);
    }
    auto const next = std::stable_partition(begin, end, [&](auto const &a) {
      return sortBeforeOthers.contains(a);
    });
    assert(next != begin && "not a partial ordering");
    begin = next;
  }
}

/// Sink pure navigation ops toward their first use within `region` (recursing
/// into nested regions), and drop dead ones. After this runs, every pure
/// navigation op sits just before the test that first consumes it, so the
/// `match -> pdl_interp` lowering becomes a straightforward program-order
/// translation.
void sinkNavigationOps(Region &region);

} // namespace match
} // namespace mlir

#endif // MLIR_DIALECT_MATCH_TRANSFORMS_MATCHTRANSFORMSUTILS_H
