//===- CombinePatterns.cpp - Combine pdl_constr.pattern ops ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass combines all `pdl_constr.pattern` ops in a module into a single
// `pdl_constr.pattern`. It walks each input pattern in order, cloning its
// body into the combined pattern and deduplicating navigation / constraint
// ops via structural equivalence (`OperationEquivalence`). `pdl_constr.success`
// ops are always preserved and never deduplicated — each one becomes an
// independent success record inside the combined body.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLConstr/Transforms/Passes.h"

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pdl_constr {
#define GEN_PASS_DEF_PDLCONSTRCOMBINEPATTERNSPASS
#include "mlir/Dialect/PDLConstr/Transforms/Passes.h.inc"
} // namespace pdl_constr
} // namespace mlir

using namespace mlir;
using namespace mlir::pdl_constr;

namespace {

/// Hash / equality keyed on structural operation equivalence — same op kind,
/// same attributes, same operand SSA values, same result types. Modelled
/// after the CSE pass's `SimpleOperationInfo`.
struct OpEquivInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    if (opC == getEmptyKey() || opC == getTombstoneKey())
      return DenseMapInfo<Operation *>::getHashValue(opC);
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    if (lhsC == rhsC)
      return true;
    if (lhsC == getEmptyKey() || lhsC == getTombstoneKey() ||
        rhsC == getEmptyKey() || rhsC == getTombstoneKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        OperationEquivalence::IgnoreLocations);
  }
};

struct PDLConstrCombinePatternsPass
    : public pdl_constr::impl::PDLConstrCombinePatternsPassBase<
          PDLConstrCombinePatternsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void PDLConstrCombinePatternsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Snapshot the input patterns before mutating the module.
  SmallVector<PatternOp, 4> patterns(module.getOps<PatternOp>());
  if (patterns.size() <= 1)
    return;

  // Pick the maximum benefit across the inputs. Per-success benefits are not
  // yet represented; when inputs disagree on benefit this is a known
  // information loss.
  unsigned bestBenefit = 0;
  for (PatternOp p : patterns)
    bestBenefit = std::max(bestBenefit, (unsigned)p.getBenefit());

  // Create the combined pattern as a sibling, immediately before the first
  // input pattern so that the iteration order is preserved.
  OpBuilder b(patterns.front());
  Location combinedLoc = patterns.front().getLoc();
  auto combined = PatternOp::create(
      b, combinedLoc,
      /*benefit=*/b.getIntegerAttr(b.getIntegerType(16), bestBenefit),
      /*sym_name=*/StringAttr());

  Block *combinedBlock = &combined.getBodyRegion().emplaceBlock();
  Value combinedRoot =
      combinedBlock->addArgument(b.getType<pdl::OperationType>(), combinedLoc);

  // Dedup set: navigation / constraint / combinator ops that are structurally
  // identical share a single canonical representative inside the combined
  // body.
  DenseSet<Operation *, OpEquivInfo> dedup;

  OpBuilder bodyBuilder(combinedBlock, combinedBlock->end());

  for (PatternOp pattern : patterns) {
    Block &origBlock = pattern.getBodyRegion().front();

    // Map the original pattern's root block argument onto the combined root.
    IRMapping mapping;
    mapping.map(origBlock.getArgument(0), combinedRoot);

    for (Operation &orig : origBlock) {
      // `pdl_constr.success` ops are never deduplicated: each marks a
      // distinct successful match.
      if (isa<SuccessOp>(orig)) {
        bodyBuilder.clone(orig, mapping);
        continue;
      }

      // Clone into the combined block, then look for an equivalent existing
      // op. The clone is needed because `OperationEquivalence` expects two
      // concrete `Operation*` arguments (it does not provide a key-style
      // lookup from raw operand/attribute tuples).
      Operation *cloned = bodyBuilder.clone(orig, mapping);
      auto [it, inserted] = dedup.insert(cloned);
      if (inserted)
        continue;

      // An equivalent op already exists. Redirect the mapping from the
      // original results to the canonical results and discard the clone.
      Operation *canonical = *it;
      for (auto [origRes, canonRes] :
           llvm::zip(orig.getResults(), canonical->getResults()))
        mapping.map(origRes, canonRes);
      cloned->erase();
    }
  }

  // Erase the now-empty input patterns.
  for (PatternOp p : patterns)
    p.erase();
}
