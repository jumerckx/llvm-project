//===- MatchTransformsUtils.cpp - Shared match transform helpers ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Match/Transforms/MatchTransformsUtils.h"

#include "mlir/Dialect/Match/IR/MatchOps.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::match;

bool mlir::match::isPureNavigationOp(Operation *op) {
  return isa<GetOperandOp, GetOperandsOp, GetResultOp, GetResultsOp,
             GetAttributeOp, GetDefiningOpOp, GetValueTypeOp,
             GetAttributeTypeOp, GetUsersOp, ExtractOp>(op);
}

namespace {

/// Walk up from `u` until reaching the operation that sits directly in
/// `block`. `block` must be an ancestor block of `u` (guaranteed here because
/// the navigation op in `block` dominates `u`).
static Operation *ancestorInBlock(Operation *u, Block *block) {
  Operation *cur = u;
  while (cur->getBlock() != block) {
    cur = cur->getBlock()->getParentOp();
    assert(cur && "expected block to be an ancestor of the use");
  }
  return cur;
}

/// Given that `anchor` is an ancestor op of `u`, return the region of `anchor`
/// that (transitively) contains `u`.
static Region *childRegionContaining(Operation *anchor, Operation *u) {
  Operation *cur = u;
  while (cur->getParentOp() != anchor)
    cur = cur->getParentOp();
  return cur->getParentRegion();
}

/// Compute the deepest valid placement for navigation op `n`: the block that
/// dominates all uses of its result, and the op within that block that the
/// navigation should be inserted right before. Returns {nullptr, nullptr} if
/// `n` has no uses (dead). Descends into a `try` / `switch_*` case region only
/// when every use is contained in that one region.
static std::pair<Block *, Operation *> computeSinkTarget(Operation *n) {
  Value v = n->getResult(0);
  if (v.use_empty())
    return {nullptr, nullptr};

  SmallVector<Operation *, 4> users(v.getUsers());
  Block *block = n->getBlock();
  while (true) {
    // Anchor each use to the op that sits directly in `block`, and find the
    // earliest such anchor.
    Operation *earliest = nullptr;
    Operation *commonAnchor = nullptr;
    bool allSameAnchor = true;
    for (Operation *u : users) {
      Operation *anchor = ancestorInBlock(u, block);
      if (!earliest || anchor->isBeforeInBlock(earliest))
        earliest = anchor;
      if (!commonAnchor)
        commonAnchor = anchor;
      else if (commonAnchor != anchor)
        allSameAnchor = false;
    }

    // Try to descend into a region of the common anchor: only possible when
    // all uses funnel through a single region-carrying op that is not itself
    // a direct user of the value.
    if (allSameAnchor && commonAnchor->getNumRegions() > 0 &&
        !llvm::is_contained(users, commonAnchor)) {
      Region *target = nullptr;
      bool single = true;
      for (Operation *u : users) {
        Region *r = childRegionContaining(commonAnchor, u);
        if (!target)
          target = r;
        else if (target != r) {
          single = false;
          break;
        }
      }
      if (single && target && target->hasOneBlock()) {
        block = &target->front();
        continue;
      }
    }

    return {block, earliest};
  }
}

} // namespace

void mlir::match::sinkNavigationOps(Region &region) {
  // Collect all pure navigation ops in the matcher (across nested regions).
  SmallVector<Operation *> navOps;
  region.walk([&](Operation *op) {
    if (isPureNavigationOp(op))
      navOps.push_back(op);
  });

  // Drop dead navigation ops first (unused values are never emitted by the
  // original lowering). Iterate to a fixpoint since erasing one may strand
  // another that only fed it.
  bool erased = true;
  while (erased) {
    erased = false;
    for (Operation *&op : navOps) {
      if (op && op->use_empty()) {
        op->erase();
        op = nullptr;
        erased = true;
      }
    }
  }
  llvm::erase(navOps, nullptr);

  // Sink each navigation op toward its first use. Iterate to a fixpoint:
  // sinking a consumer can let its producer sink further. Moves are monotone
  // (always later / deeper), so this converges.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : navOps) {
      auto [block, before] = computeSinkTarget(op);
      if (!before)
        continue;
      // The op is already sunk far enough when it is in the target block and
      // only other navigation ops lie between it and its first use: the real
      // (control-flow-bearing) tests are all above it, and the relative order
      // among sibling navigation ops that share a use is irrelevant. Checking
      // for an exact "immediately before" position instead would make two
      // navigation ops that feed the same use leapfrog each other forever.
      if (op->getBlock() == block) {
        bool needMove = false;
        for (Operation *cur = op->getNextNode(); cur != before;
             cur = cur->getNextNode()) {
          assert(cur && "first use must come after the navigation op");
          if (!isPureNavigationOp(cur)) {
            needMove = true;
            break;
          }
        }
        if (!needMove)
          continue;
      }
      op->moveBefore(before);
      changed = true;
    }
  }
}
