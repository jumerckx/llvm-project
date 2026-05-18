//===-- Passes.td - PDLConstr transform passes -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES
#define MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def PDLConstrCombinePatternsPass
    : Pass<"pdl-constr-combine-patterns", "ModuleOp"> {
  let summary = "Combine multiple pdl_constr.pattern ops into a single one";
  let description = [{
    Merges all `pdl_constr.pattern` ops within a module into a single
    `pdl_constr.pattern`. Structurally-equivalent navigation and constraint
    ops (same op kind, same attributes, same operand SSA values, same result
    types) are deduplicated, so the combined pattern shares its constraint
    work across the originating patterns. Each input pattern's
    `pdl_constr.success` op is preserved as an independent success record
    inside the combined body.

    The combined `pdl_constr.pattern` carries the maximum benefit among its
    inputs. Per-success benefits are not yet represented — when input
    patterns disagree on benefit, that information is lost at this step.
  }];
  let dependentDialects = ["pdl_constr::PDLConstrDialect"];
}

#endif // MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES
