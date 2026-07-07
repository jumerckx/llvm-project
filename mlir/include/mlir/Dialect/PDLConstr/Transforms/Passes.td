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

def PDLConstrCombineMatchersPass
    : Pass<"pdl-constr-combine-matchers", "ModuleOp"> {
  let summary = "Combine multiple pdl_constr.matcher ops into a single one";
  let description = [{
    Merges all `pdl_constr.matcher` ops within a module into one combined
    matcher whose body materializes the matcher tree as IR.

    The combining algorithm mirrors the cost-based merge that the original
    `PDLToPDLInterp` pass performed in C++ on its `MatcherNode` tree. Each
    constraint op in an input matcher is treated as one entry of a globally
    sorted predicate list; entries shared across matchers are emitted once,
    and divergent suffixes appear as sibling `pdl_constr.try` regions or as
    cases of `pdl_constr.switch_op_name` / `pdl_constr.switch_type`. The
    `pdl_constr.success` ops of the input matchers are preserved verbatim
    (with their rewriter symbols and forwarded match values) at the leaves
    of the resulting tree.
  }];
  let dependentDialects = ["pdl_constr::PDLConstrDialect"];
}

#endif // MLIR_DIALECT_PDLCONSTR_TRANSFORMS_PASSES
