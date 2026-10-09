//===- Passes.td - Match transform passes ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATCH_TRANSFORMS_PASSES
#define MLIR_DIALECT_MATCH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MatchCombineMatchersPass
    : Pass<"match-combine-matchers", "ModuleOp"> {
  let summary = "Combine multiple match.matcher ops into a single one";
  let dependentDialects = ["match::MatchDialect"];
}

#endif // MLIR_DIALECT_MATCH_TRANSFORMS_PASSES
