//===-- Passes.td - Match transform passes -------------*- tablegen -*-===//
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
  let description = [{
    Merges all `match.matcher` ops within a module into one combined
    matcher whose body materializes the matcher tree as IR.

    The combining algorithm mirrors the cost-based merge that the original
    `PDLToPDLInterp` pass performed in C++ on its `MatcherNode` tree. Each
    constraint op in an input matcher is treated as one entry of a globally
    sorted predicate list; entries shared across matchers are emitted once,
    and divergent suffixes appear as sibling `match.try` regions or as
    cases of `match.switch_op_name` / `match.switch_type`. The
    `match.success` ops of the input matchers are preserved verbatim
    (with their rewriter symbols and forwarded match values) at the leaves
    of the resulting tree.
  }];
  let dependentDialects = ["match::MatchDialect"];
}

#endif // MLIR_DIALECT_MATCH_TRANSFORMS_PASSES
