//===- Passes.h - PDLConstr transform passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace pdl_constr {

#define GEN_PASS_DECL
#include "mlir/Dialect/PDLConstr/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/PDLConstr/Transforms/Passes.h.inc"

} // namespace pdl_constr
} // namespace mlir

#endif // MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES_H
