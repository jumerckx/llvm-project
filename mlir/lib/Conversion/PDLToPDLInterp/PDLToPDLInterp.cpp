//===- PDLToPDLInterp.cpp - Lower a PDL module to the interpreter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"

#include "mlir/Conversion/MatchToPDLInterp/MatchToPDLInterp.h"
#include "mlir/Conversion/PDLToMatch/PDLToMatch.h"
#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/Match/Transforms/Passes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct PDLToPDLInterpPass
    : public impl::ConvertPDLToPDLInterpPassBase<PDLToPDLInterpPass> {
  PDLToPDLInterpPass() = default;
  PDLToPDLInterpPass(const PDLToPDLInterpPass &rhs) = default;
  PDLToPDLInterpPass(DenseMap<Operation *, PDLPatternConfigSet *> &configMap)
      : configMap(&configMap) {}
  void runOnOperation() final;

  DenseMap<Operation *, PDLPatternConfigSet *> *configMap = nullptr;
};
} // namespace

void PDLToPDLInterpPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (failed(convertPDLToMatch(module, configMap)) ||
      failed(match::combineMatchers(module)) ||
      failed(convertMatchToPDLInterp(module, configMap)))
    return signalPassFailure();

  if (!configMap)
    return;
  SmallVector<Operation *> rewriters;
  module.walk([&](pdl_interp::RecordMatchOp op) {
    Operation *rewriter =
        SymbolTable::lookupNearestSymbolFrom(op, op.getRewriterAttr());
    if (PDLPatternConfigSet *config = configMap->lookup(rewriter)) {
      (*configMap)[op] = config;
      rewriters.push_back(rewriter);
    }
  });
  for (Operation *rewriter : rewriters)
    configMap->erase(rewriter);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertPDLToPDLInterpPass(
    DenseMap<Operation *, PDLPatternConfigSet *> &configMap) {
  return std::make_unique<PDLToPDLInterpPass>(configMap);
}
