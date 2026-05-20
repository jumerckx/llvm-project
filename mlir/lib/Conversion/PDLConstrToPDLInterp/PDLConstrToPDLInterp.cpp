//===- PDLConstrToPDLInterp.cpp - Lower pdl_constr to pdl_interp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO: This pass is being re-implemented for the redesigned `pdl_constr`
// dialect (ordered IR with explicit `try` / `switch_*` regions, no SSA
// predicates). The old implementation assumed the previous predicate-set
// design and is incompatible with the new ops. A new mechanical region
// walker (see the redesign spec) will be implemented in a follow-up.
//
// For now this file provides a stub pass that fails when invoked so that
// the rest of the codebase continues to build.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLConstrToPDLInterp/PDLConstrToPDLInterp.h"

#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLCONSTRTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct PDLConstrToPDLInterpPass
    : public impl::ConvertPDLConstrToPDLInterpPassBase<
          PDLConstrToPDLInterpPass> {
  void runOnOperation() final {
    getOperation()->emitError(
        "convert-pdl-constr-to-pdl-interp: pass is not yet implemented for "
        "the redesigned pdl_constr dialect");
    signalPassFailure();
  }
};
} // namespace
