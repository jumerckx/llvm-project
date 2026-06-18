//===- MatchToPDLInterp.h - match to pdl_interp lowering *- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass that combines multiple `match.pattern` ops
// into a single `pdl_interp` matcher function. The combination logic mirrors
// the predicate-tree merging performed by `convert-pdl-to-pdl-interp`, but
// the predicate set is reconstructed from `match` IR instead of being
// derived from a `pdl.pattern` op directly.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATCHTOPDLINTERP_MATCHTOPDLINTERP_H
#define MLIR_CONVERSION_MATCHTOPDLINTERP_MATCHTOPDLINTERP_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
template <typename OpT>
class OperationPass;

#define GEN_PASS_DECL_CONVERTMATCHTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_MATCHTOPDLINTERP_MATCHTOPDLINTERP_H
