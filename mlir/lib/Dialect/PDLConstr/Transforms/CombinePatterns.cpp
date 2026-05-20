//===- CombinePatterns.cpp - Combine pdl_constr.matcher ops ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: This pass is being re-implemented for the redesigned `pdl_constr`
// dialect (`pdl_constr.matcher` with ordered IR, `pdl_constr.try` /
// `pdl_constr.switch_*` regions). The previous implementation operated on
// the unordered predicate-set design and is incompatible with the new ops.
//
// Stub implementation: signals failure if invoked.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLConstr/Transforms/Passes.h"

#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pdl_constr {
#define GEN_PASS_DEF_PDLCONSTRCOMBINEPATTERNSPASS
#include "mlir/Dialect/PDLConstr/Transforms/Passes.h.inc"
} // namespace pdl_constr
} // namespace mlir

using namespace mlir;
using namespace mlir::pdl_constr;

namespace {
struct CombinePatternsPass : public mlir::pdl_constr::impl::
                                 PDLConstrCombinePatternsPassBase<
                                     CombinePatternsPass> {
  void runOnOperation() final {
    getOperation()->emitError(
        "pdl-constr-combine-patterns: pass is not yet implemented for the "
        "redesigned pdl_constr dialect");
    signalPassFailure();
  }
};
} // namespace
