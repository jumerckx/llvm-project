//===- RegroupOperationAccesses.cpp - Spread apart op accesses ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reorders the tests in each `match.matcher` so that no `match.get_defining_op`
// is emitted before every test that can be performed without it. This ports the
// eqsat `OperationPositionTree` / `ChooseNode` grouping into the `match`
// dialect.
//
// The idea:
//
//  * Which operation a predicate belongs to is encoded implicitly by the
//    `!pdl.operation` SSA value it consumes. Its *depth* is the length of the
//    `get_defining_op` chain from the matcher root (root = depth 0).
//  * Crossing to a deeper operation (a `get_defining_op`) is expensive in an
//    eqsat lowering (a `foreach` over an e-class), so every shallow test must
//    run first. We therefore assign each op a sort key equal to the maximum
//    depth over its operands and results, and stably reorder each straight-line
//    failure scope by ascending key.
//  * Reordering is only sound *within* a single failure scope. Failure scopes
//    are delimited by `match.try` / `match.switch_*`; `match.get_each`
//    (a `foreach` barrier) and `match.success` are anchors too. We segment each
//    block at those anchors and reorder only within a segment.
//
// After reordering, `sinkNavigationOps` re-places the pure navigation ops just
// before their first use, so the lowering to `pdl_interp` stays a
// straightforward program-order translation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Match/Transforms/Passes.h"

#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/Match/IR/MatchOps.h"
#include "mlir/Dialect/Match/Transforms/MatchTransformsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "match-regroup-operation-accesses"

namespace mlir {
namespace match {
#define GEN_PASS_DEF_MATCHREGROUPOPERATIONACCESSESPASS
#include "mlir/Dialect/Match/Transforms/Passes.h.inc"
} // namespace match
} // namespace mlir

using namespace mlir;
using namespace mlir::match;

namespace {

/// An op is a failure-scope / control-flow anchor: reordering never crosses it.
///
///  * `try` / `switch_op_name` / `switch_type` open (or dispatch into) new
///    failure scopes.
///  * `get_each` lowers to a `foreach` loop, so hoisting an element constraint
///    above it would move it out of the loop.
///  * `success` records a match; the tests above it are its precondition.
static bool isScopeAnchor(Operation *op) {
  return isa<TryOp, SwitchOpNameOp, SwitchTypeOp, GetEachOp, SuccessOp>(op);
}

//===----------------------------------------------------------------------===//
// Depth analysis
//===----------------------------------------------------------------------===//

/// Assigns every SSA value in `region` an operation depth: the matcher root is
/// depth 0, and each `get_defining_op` result is one deeper than the value it
/// navigates from. Depths flow through the navigation / test ops unchanged
/// otherwise. Computed once per matcher because reordering never changes the
/// values or their depths.
class DepthAnalysis {
public:
  explicit DepthAnalysis(Region &region) { compute(region); }

  /// Sort key for `op`: the deepest operation it touches. A `get_defining_op`
  /// (or the `is_not_null` that unwraps it) therefore keys to the *deeper*
  /// operation it produces, while a test / navigation rooted at op@d keys to d.
  /// A `match.equal` / `apply_native_constraint` spanning several depths keys
  /// to the maximum, so it stays after all of its producers.
  unsigned sortKey(Operation *op) const {
    unsigned key = 0;
    for (Value operand : op->getOperands())
      key = std::max(key, depthOf(operand));
    for (Value result : op->getResults())
      key = std::max(key, depthOf(result));
    return key;
  }

private:
  unsigned depthOf(Value v) const {
    auto it = depths.find(v);
    return it == depths.end() ? 0 : it->second;
  }

  void compute(Region &region) {
    // Seed the matcher root (block argument) at depth 0.
    for (BlockArgument arg : region.front().getArguments())
      depths[arg] = 0;

    // Pre-order walk visits producers before consumers (SSA dominance holds
    // across the structured region tree), so a single pass suffices.
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      unsigned operandDepth = 0;
      for (Value operand : op->getOperands())
        operandDepth = std::max(operandDepth, depthOf(operand));

      // Descending to the defining op crosses into a deeper operation.
      unsigned resultDepth =
          isa<GetDefiningOpOp>(op) ? operandDepth + 1 : operandDepth;
      for (Value result : op->getResults())
        depths[result] = resultDepth;
    });
  }

  DenseMap<Value, unsigned> depths;
};

//===----------------------------------------------------------------------===//
// Reordering
//===----------------------------------------------------------------------===//

class Regrouper {
public:
  explicit Regrouper(const DepthAnalysis &depths) : depths(depths) {}

  /// Reorder every straight-line segment of `region`'s block, then recurse into
  /// the case regions of nested `try` / `switch_*` ops.
  void run(Region &region) {
    Block &block = region.front();

    // Collect segment boundaries: the ops between two consecutive anchors form
    // one reorderable segment.
    SmallVector<Operation *> segment;
    auto flush = [&](Operation *insertBefore) {
      if (segment.size() > 1)
        reorderSegment(block, segment, insertBefore);
      segment.clear();
    };
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (isScopeAnchor(&op)) {
        flush(&op);
        continue;
      }
      segment.push_back(&op);
    }
    flush(/*insertBefore=*/nullptr);

    // Recurse into nested failure scopes.
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        if (!nested.empty())
          run(nested);
  }

private:
  /// Stably reorder `segment` (all ops of a single failure scope, in current
  /// program order) by ascending depth key, then move them into place just
  /// before `insertBefore` (or the block end when null).
  void reorderSegment(Block &block, ArrayRef<Operation *> segment,
                      Operation *insertBefore) {
    SmallVector<Operation *> sorted(segment.begin(), segment.end());

    // Primary: ascending depth key. `stable_sort` keeps the existing
    // (cost-model) order as the tiebreaker among equal-depth ops.
    llvm::stable_sort(sorted, [&](Operation *a, Operation *b) {
      return depths.sortKey(a) < depths.sortKey(b);
    });

    // Depth-ascending order already respects dominance (deeper ops depend on
    // shallower producers), but run the topological backstop so any same-key
    // producer/consumer pair is never inverted.
    stableTopologicalSort(sorted.begin(), sorted.end(), dependsOn);

    // Nothing to do if the order is unchanged.
    if (std::equal(sorted.begin(), sorted.end(), segment.begin()))
      return;

    Block::iterator pos =
        insertBefore ? Block::iterator(insertBefore) : block.end();
    for (Operation *op : sorted)
      op->moveBefore(&block, pos);

    LDBG() << "regrouped a segment of " << sorted.size() << " ops";
  }

  const DepthAnalysis &depths;
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct RegroupOperationAccessesPass
    : public mlir::match::impl::MatchRegroupOperationAccessesPassBase<
          RegroupOperationAccessesPass> {
  void runOnOperation() final {
    getOperation().walk([&](MatcherOp matcher) {
      Region &body = matcher.getBodyRegion();
      if (body.empty())
        return;
      DepthAnalysis depths(body);
      Regrouper(depths).run(body);
      // Re-place pure navigation ops just before their first use so the
      // lowering to pdl_interp stays a program-order translation.
      match::sinkNavigationOps(body);
    });
  }
};

} // namespace
