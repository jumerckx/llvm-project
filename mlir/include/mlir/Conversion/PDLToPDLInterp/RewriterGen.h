//===- RewriterGen.h - PDL -> PDL Interp rewriter generation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helper used to lower the `pdl.rewrite` region of a `pdl.pattern` into
// a `pdl_interp.func` inside a rewriter module. Used by both the PDL ->
// PDLInterp and PDL -> PDLConstr conversions so that both pipelines emit
// identical rewriter functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PDLTOPDLINTERP_REWRITERGEN_H
#define MLIR_CONVERSION_PDLTOPDLINTERP_REWRITERGEN_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pdl {
class PatternOp;
} // namespace pdl

namespace pdl_to_pdl_interp {

/// Generate a `pdl_interp.func` rewriter for the given pattern. The function is
/// inserted into `rewriterModule` using `rewriterSymbolTable` to ensure a
/// unique symbol name. The returned `SymbolRefAttr` references the new
/// function under the rewriter module symbol (e.g.
/// `@rewriters::@some_pattern`).
///
/// `usedMatchValues` is populated, in argument order, with the pdl values from
/// the pattern's match region that became arguments of the rewriter function.
/// Callers can use this list to forward the appropriate matched values at the
/// invocation site (`pdl_interp.record_match` or `pdl_constr.success`).
SymbolRefAttr generatePatternRewriter(pdl::PatternOp pattern,
                                      ModuleOp rewriterModule,
                                      SymbolTable &rewriterSymbolTable,
                                      OpBuilder &builder,
                                      SmallVectorImpl<Value> &usedMatchValues);

} // namespace pdl_to_pdl_interp
} // namespace mlir

#endif // MLIR_CONVERSION_PDLTOPDLINTERP_REWRITERGEN_H
