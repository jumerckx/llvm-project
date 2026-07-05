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

def MatchRegroupOperationAccessesPass
    : Pass<"match-regroup-operation-accesses", "ModuleOp"> {
  let summary = "Spread apart operation accesses in combined match.matcher ops";
  let description = [{
    Reorders the tests within each `match.matcher` so that every test that can
    be performed on the current operation is exhausted *before* descending into
    another operation via `match.get_defining_op`.

    In an eqsat-style lowering, crossing from a value to its defining op is not
    an O(1) step: it fans out into a `foreach` over an entire e-class. A matcher
    must therefore exhaust every cheap test it can perform on the current
    operation before descending, otherwise those shallow checks get needlessly
    re-evaluated inside every iteration of a deeper loop. This ports the eqsat
    `OperationPositionTree` / `ChooseNode` grouping into the `match` dialect.

    The pass assigns each `!pdl.operation` value a depth (the matcher root is
    depth 0, each `get_defining_op` adds one) and stably reorders the ops within
    each straight-line failure scope by ascending depth. This is sound because,
    within a single failure scope delimited by `match.try` / `match.switch_*` /
    `match.get_each`, all tests share the same failure target, so reordering
    them preserves semantics as long as SSA dominance is respected. It is meant
    to run *after* `match-combine-matchers`.
  }];
  let dependentDialects = ["match::MatchDialect"];
}

#endif // MLIR_DIALECT_MATCH_TRANSFORMS_PASSES
