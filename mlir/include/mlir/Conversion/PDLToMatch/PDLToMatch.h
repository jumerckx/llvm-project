//===- PDLToMatch.h - pdl to match conversion ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass for PDL to PDL Constraint dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PDLTOMATCH_PDLTOMATCH_H
#define MLIR_CONVERSION_PDLTOMATCH_PDLTOMATCH_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
template <typename OpT>
class OperationPass;

#define GEN_PASS_DECL_CONVERTPDLTOMATCHPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_PDLTOMATCH_PDLTOMATCH_H
