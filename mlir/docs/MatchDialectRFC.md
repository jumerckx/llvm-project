# RFC: The `match` dialect — a new IR between `pdl` and `pdl_interp`

## Summary

This RFC proposes adding `match`, a new dialect that sits between `pdl` and
`pdl_interp`. It makes the *matcher* half of pattern compilation — navigation,
constraints, and the matcher-tree structure — explicit as MLIR IR, where today
that information only exists transiently inside the `PDLToPDLInterp` C++ pass.
The lowering `pdl → pdl_interp` is split into three composable passes:

```
pdl  --convert-pdl-to-match-->  match  --match-combine-matchers-->  match  --convert-match-to-pdl-interp-->  pdl_interp
```

A `match`-to-C++ tool (`mlir-match-to-cpp`) is also implemented as an
alternative backend that emits `RewritePattern`s directly.

There is a [WASM web demo](https://jumerckx.github.io/mlir-opt-wasm/pdl.html) illustrating the pdl lowering flow with the `match`
dialect.
The branch with current implementation lives [here](https://github.com/jumerckx/llvm-project/tree/jm/pdl_constr_cpp).

## Motivation

`PDLToPDLInterp` lowers every `pdl.pattern` in a module into a single shared
`pdl_interp` matcher function. To do so it builds a *predicate tree*: each
pattern is reduced to a list of positions and questions (`Predicate.h`), those
lists are merged with a cost model (`PredicateTree.cpp`), and the merged
`MatcherNode` tree is finally emitted as `pdl_interp` control flow. All of this
is in-memory C++ that is never expressed as IR.

Downsides of this are:

- **Not inspectable.** There is no textual form for "the matcher after root
  ordering" or "the combined predicate tree". Debugging an unexpected match
  order means reading C++ data structures in a debugger.
- **Not transformable.** Useful operations on the matcher — sharing a sub-tree,
  reordering predicates, swapping in a different combine strategy, or targeting
  a backend other than `pdl_interp` — have no IR to operate on.
- **Monolithic.** Root selection, predicate ordering, tree combination, and
  `pdl_interp` emission are a single pass; the stages cannot be tested or reused
  independently.

`match` exists to expose this intermediate state. The cost-based combine that
was previously private to `PDLToPDLInterp` becomes a standalone pass operating
on `match` IR, and `pdl_interp` emission becomes a separate, mechanical walk.

Concretely, the match dialect is a nicer format to generate C++ from because, in
contrast to `pdl`, the operations are imperative. And in contrast to `pdl_interp`,
the IR does not contain unstructured control flow.

The match dialect would also be used by other projects such as
[Tamagoyaki](https://github.com/jumerckx/Tamagoyaki) to otherwise
transform patterns.
There is also interest to write rewrite pattern frontends (e.g. in xDSL), that
generate `match` IR directly instead of forcing `pdl`'s declarative format onto
frontends.

## Design

- **Position encodes AND.** Ops in a matcher body run top to bottom. Reaching
  any op means every test above it has already succeeded.
- **Regions scope matching failure (OR).** Test ops are *not* `Pure`: each one
  carries an implicit control effect — *on failure, transfer to the enclosing
  failure scope*. `match.try` opens a new failure scope, so a run of sibling
  `match.try` regions is exactly an OR of alternatives. This is the direct IR
  analogue of `MatcherNode::failureNode`.
- **`match.success`** marks a succesful pattern match: every test on the path
  from the root to it has held. It carries the `benefit`, the rewriter symbol,
  and the forwarded match values — its signature deliberately mirrors
  `pdl_interp.record_match` so the metadata passes through the final lowering
  unchanged. One `match.matcher` may hold several `success` ops.

### Nullability

Navigation that can fail (`get_operand`, `get_result(s)`, `get_attribute`,
`get_defining_op`, ...) returns `!match.optional<T>`. The *only* way to obtain
the bare `T` is `match.is_not_null`, which has the implicit failure-control
effect. Because the unwrapped value is a new SSA result, dominance guarantees it
can only be consumed *below* the null-check — which is precisely the
control-flow guarantee the lowered `pdl_interp` form relies on. Navigation that
cannot fail (`get_value_type`, `get_users`, ...) returns the bare PDL type
directly. This split keeps "can this step fail?" visible in the type system.

Navigation ops are `Pure`; test ops are not. That distinction drives the
combine pass: pure navigation can be freely re-placed (deduplicated, sunk),
tests cannot.

The reason to have unwrapping of optional values explicitly is because native
constraints could be allowed to consume an optionally-null value.

### Types

The `match` dialect reuses `!pdl.operation`, `!pdl.value`, `!pdl.type`,
`!pdl.attribute` and `!pdl.range<...>`. The only new type is
`!match.optional<T>` above.

### Multi-way dispatch and ranges

- `match.switch_op_name` / `match.switch_type` represent `SwitchNode` directly:
  one case region per name/type, with the enclosing failure scope as the
  implicit default. Without them, opcode dispatch would degrade to a chain of
  equality `try`s.
- `match.get_each` declares *existential* iteration over a range (constraints
  must hold for at least one element); it is the one navigation op that carries
  control flow and lowers to a `pdl_interp.foreach` loop. `match.extract` is its
  index-based, non-looping counterpart.

### Operation summary

| Op | Role | Can fail? |
|---|---|---|
| `match.matcher` | `IsolatedFromAbove` container, root op as block arg | — |
| `match.try` | open a failure scope (OR alternative) | — |
| `match.switch_op_name` / `match.switch_type` | multi-way dispatch | default → enclosing scope |
| `match.get_operand(s)` / `get_result(s)` / `get_attribute` / `get_defining_op` | nullable navigation → `optional<T>` | via `is_not_null` |
| `match.get_value_type` / `get_attribute_type` / `get_users` | non-nullable navigation | no |
| `match.extract` / `get_each` | range element access / existential iteration | `extract` may be null |
| `match.is_not_null` | unwrap `optional<T>` | yes |
| `match.has_name` / `has_type(s)` / `has_attr_value` / `equal` / `check_operand_count` / `check_result_count` / `apply_native_constraint` | tests | yes |
| `match.success` | matched-pattern leaf | — |

## The passes

### `convert-pdl-to-match`

Lowers each `pdl.pattern` into its own `match.matcher`, with no `try`/`switch`
nesting beyond what the single pattern needs. Root selection and multi-root
ordering reuse the existing `PDLToPDLInterp` helpers (`detectRoots`,
`buildCostGraph`, `OptimalBranching`, ...).

### `match-combine-matchers`

Merges all per-pattern matchers in a module into one combined matcher tree. This
is roughly the same cost-based merge `PredicateTree.cpp` performed in C++,
re-expressed on IR.

| `PredicateTree` concept | `match` representation |
|---|---|
| Predicate identity / `OpKey` | `Operation *` in a canonical "pool" block |
| Canonical position/value ID | the `Value` a pool op produces |
| Dependency edge | SSA def-use in the pool |
| Per-matcher value map | `IRMapping` (matcher value → pool value) |
| Predicate ordering | block order of pool ops |

Predicates are deduplicated into the pool by `(op name, attributes, canonical
operand Values)`, sorted with the **same** cost model as before (the
position/question-kind tie-breakers mirror `Predicates::Kind` integer-for-integer
so output matches the original pass), then a failure tree is built and
materialized as nested `try`/`switch_*`. Two cleanups run on the result: sibling
`try`s starting with the same `has_name`/`has_type` are folded into a
`switch_*`, and pure navigation ops are sunk to just before their first use so
the final lowering is a straight program-order walk.

### `convert-match-to-pdl-interp`

A mechanical region walker carrying two pieces of state — `currentBlock` (where
to emit) and `failureBlock` (where failures branch). Tests become
`pdl_interp.check_*` / `is_not_null` with the failure branch at `failureBlock`;
`try` allocates a fresh "after" block and lowers its body against it;
`switch_*`, `get_each` (→ `foreach`), and `success` (→ `record_match`) follow
correspondingly. Because the tree shape is already in the IR, there is no
remaining cost model or tree-building here.

### `mlir-match-to-cpp`

An alternative backend that emits a `RewritePattern` subclass per matcher
directly from `match` IR, demonstrating that the explicit matcher IR is reusable
for targets other than the `pdl_interp` interpreter.
In contrast to DRR, this can generate matcher code for unregistered dialects
as well. It can optionally take ODS definitions, similar to DRR, to generate
matchers that use higher level C++ dialect APIs (see demo).

## Example

A single pattern after `convert-pdl-to-match`:

```mlir
match.matcher @addf_matcher root(%root : !pdl.operation) {
  match.has_name %root, "arith.addf"
  match.check_operand_count %root is 2
  %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
  %1 = match.is_not_null %0 : !match.optional<!pdl.value> -> !pdl.value
  match.success @rewriters::@addf benefit(1) (%root, %1 : !pdl.operation, !pdl.value)
}
```

Two patterns sharing an opcode prefix, after `match-combine-matchers`:

```mlir
match.matcher @combined root(%root : !pdl.operation) {
  match.has_name %root, "arith.addi"
  match.try {            // alternative 1
    match.success @rewriters::@rewrite_a benefit(2)
  }
  match.try {            // alternative 2
    match.success @rewriters::@rewrite_b benefit(1)
  }
}
```
