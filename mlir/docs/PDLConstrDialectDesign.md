# `pdl_constr` Dialect Design

## Motivation

The current PDL → PDL_Interp lowering hides all constraint and navigation logic
in C++ data structures (`Position*`, `Qualifier*`, `MatcherNode`). This design
introduces an intermediate dialect (`pdl_constr`) that makes these constraints
explicit in MLIR IR, enabling inspection, transformation, and composition before
the final lowering to `pdl_interp`.

### Current Pipeline

```
pdl.pattern → [C++ MatcherNode tree] → pdl_interp
```

### Proposed Pipeline

```
pdl.pattern → [Pass 1] → pdl_constr IR → [Pass 2] → pdl_interp
```

---

## Dialect Definition

**Namespace:** `pdl_constr`
**Dependent dialects:** `pdl::PDLDialect`, `pdl_interp::PDLInterpDialect`
**Location:** `mlir/include/mlir/Dialect/PDLConstr/IR/`

Reuse existing PDL types (`!pdl.operation`, `!pdl.value`, `!pdl.attribute`,
`!pdl.type`, `!pdl.range<...>`) throughout.

---

## Types

### `!pdl_constr.pred`

A boolean-like predicate result type. Every constraint op returns a
`!pdl_constr.pred` value. These are combined with `all` / `any` and consumed by
`success`.

### `!pdl_constr.optional<T>`

A wrapper type indicating that the contained value may be null at runtime.
Navigation ops that can fail (e.g., getting an operand that may not exist,
getting the defining op of a block argument) return
`!pdl_constr.optional<!pdl.operation>` etc. instead of the bare PDL type.

This enforces at the type level that no constraint can consume a
potentially-null value without first unwrapping it via `pdl_constr.is_not_null`.
The `is_not_null` op is the **only** way to convert from
`!pdl_constr.optional<T>` to `T`.

`T` may be any PDL type: `!pdl.operation`, `!pdl.value`, `!pdl.attribute`,
`!pdl.type`, `!pdl.range<...>`.

---

## Operations

### Top-Level: `pdl_constr.pattern`

Wraps a single pattern's navigation and constraints in one op with a single
block region. The block argument is the root operation.

```mlir
pdl_constr.pattern @name benefit(N) root(%root : !pdl.operation) {
  // ... navigation ops, constraint ops, combinators ...
  pdl_constr.success %pred
}
```

- One region, one block.
- The block has a single argument: the root `!pdl.operation`.
- Terminated by `pdl_constr.success`.
- The rewrite region is kept separate (same approach as today: either attached
  or in a rewriter module).

### Navigation Ops

These extract sub-values from the matched IR. They correspond to Position kinds
in the current `Predicate.h`. The `pdl_constr` dialect defines its own
navigation ops rather than reusing `pdl_interp` ops, because **navigations that
can fail return `!pdl_constr.optional<T>`** instead of bare PDL types. This
enforces null-safety at the type level. Navigations that cannot fail (e.g.,
getting the type of an already-unwrapped value) return bare PDL types directly.

#### Nullable navigations (return `!pdl_constr.optional<T>`)

These ops navigate to values that may not exist at runtime. Downstream
constraint ops cannot consume the result directly — it must first be unwrapped
via `pdl_constr.is_not_null`.

| Current Position Kind | `pdl_constr` op | Signature |
|---|---|---|
| `OperandPos` | `pdl_constr.get_operand` | `(op, index) → !pdl_constr.optional<!pdl.value>` |
| `OperandGroupPos` | `pdl_constr.get_operands` | `(op, opt<index>) → !pdl_constr.optional<!pdl.range<value>>` |
| `ResultPos` | `pdl_constr.get_result` | `(op, index) → !pdl_constr.optional<!pdl.value>` |
| `ResultGroupPos` | `pdl_constr.get_results` | `(op, opt<index>) → !pdl_constr.optional<!pdl.range<value>>` |
| `AttributePos` | `pdl_constr.get_attribute` | `(op, name) → !pdl_constr.optional<!pdl.attribute>` |
| `OperationPos` (def) | `pdl_constr.get_defining_op` | `(!pdl.value) → !pdl_constr.optional<!pdl.operation>` |

#### Non-nullable navigations (return bare PDL types)

These ops navigate from an already-validated (unwrapped) value to a property
that is guaranteed to exist, so no optional wrapper is needed.

| Current Position Kind | `pdl_constr` op | Signature |
|---|---|---|
| `TypePos` (value) | `pdl_constr.get_value_type` | `(!pdl.value) → !pdl.type` |
| `TypePos` (attr) | `pdl_constr.get_attribute_type` | `(!pdl.attribute) → !pdl.type` |
| `UsersPos` | `pdl_constr.get_users` | `(!pdl.value) → !pdl.range<operation>` |

`get_value_type` and `get_attribute_type` always succeed because the input
value/attribute has already been null-checked. `get_users` always returns a
range (possibly empty), so it cannot fail.

#### `pdl_constr.get_each`

Extracts a single element from a range for constraint purposes. This replaces
the `ForEachPosition` + `pdl_interp.foreach` loop structure. At this level,
there is **no structured loop**; `get_each` declares that the pattern applies
constraints to each element of the range. The `pdl_constr → pdl_interp`
lowering is responsible for generating the actual `pdl_interp.foreach` loop.

```
%elem = pdl_constr.get_each %range : !pdl.range<operation> -> !pdl.operation
```

**Signature:**
- Input: `!pdl.range<T>` (e.g., `!pdl.range<operation>`, `!pdl.range<value>`)
- Result: `T` (the element type, e.g., `!pdl.operation`, `!pdl.value`)

Semantics: any constraints applied to `%elem` must hold for *at least one*
element of the range (existential/"there exists" semantics, matching the current
`ForEachPosition` behavior). If the downstream constraints need to hold for
*all* elements, a separate `pdl_constr.get_all` could be introduced later, but
is not needed for the current PDL semantics.

### Constraint Ops

Every constraint op returns a `!pdl_constr.pred` result. They are pure
assertions — they do not branch. The lowering to `pdl_interp` turns them into
branching predicate ops.

All constraint ops take **unwrapped** PDL types as operands (not optional).
The only way to obtain an unwrapped value from a nullable navigation is via
`pdl_constr.is_not_null`.

#### `pdl_constr.is_not_null` (unwrap + check)

This op serves a dual role: it checks that an optional value is non-null **and**
unwraps it, producing both a `!pdl_constr.pred` and the bare inner value.

```mlir
%pred, %val = pdl_constr.is_not_null %opt : !pdl_constr.optional<!pdl.value>
    -> !pdl_constr.pred, !pdl.value
```

**Operands:** `!pdl_constr.optional<T>`
**Results:** `!pdl_constr.pred`, `T`

The unwrapped result `%val` may only be used by ops that are dominated by a
successful check of `%pred` (i.e., `%pred` must be included in the
`pdl_constr.all` that gates `pdl_constr.success`). The lowering to `pdl_interp`
ensures this by placing downstream ops in the success branch.

#### Other constraint ops

| Current Question | `pdl_constr` op | Operands | Attributes |
|---|---|---|---|
| `OperationNameQuestion` | `pdl_constr.has_name` | `(!pdl.operation)` | `name : StrAttr` |
| `EqualToQuestion` | `pdl_constr.equal` | `(lhs, rhs)` | — (`SameTypeOperands`) |
| `TypeQuestion` | `pdl_constr.has_type` | `(!pdl.type)` | `type : TypeAttr` (or `ArrayAttr` for ranges) |
| `AttributeQuestion` | `pdl_constr.has_attr_value` | `(!pdl.attribute)` | `value : AnyAttr` |
| `OperandCountQuestion` | `pdl_constr.check_operand_count` | `(!pdl.operation)` | `count : I32Attr`, opt `at_least : UnitAttr` |
| `ResultCountQuestion` | `pdl_constr.check_result_count` | `(!pdl.operation)` | `count : I32Attr`, opt `at_least : UnitAttr` |
| `ConstraintQuestion` | `pdl_constr.apply_native_constraint` | `(args...)` | `name : StrAttr`, opt `is_negated : BoolAttr` |

These ops all take **unwrapped** PDL types and return `!pdl_constr.pred`.

`apply_native_constraint` may also return additional `!pdl.*` results (for
native constraints that produce values), in addition to the `!pdl_constr.pred`.
The `!pdl_constr.pred` is always the **first** result.

### Combinators

#### `pdl_constr.all`

Logical AND over predicate values. Returns `!pdl_constr.pred`.

```mlir
%ok = pdl_constr.all %p1, %p2, %p3 : !pdl_constr.pred
```

**Operands:** variadic `!pdl_constr.pred`
**Result:** `!pdl_constr.pred`

#### `pdl_constr.any`

Logical OR over predicate values. Returns `!pdl_constr.pred`.

```mlir
%ok = pdl_constr.any %p1, %p2 : !pdl_constr.pred
```

**Operands:** variadic `!pdl_constr.pred`
**Result:** `!pdl_constr.pred`

### Terminator

#### `pdl_constr.success`

Marks a successful pattern match. Takes a single `!pdl_constr.pred` operand
(typically the output of an `all` combining all constraints).

```mlir
pdl_constr.success %pred
```

---

## Full Example

A pattern matching `arith.addi` where both operands are equal and have type
`i32`, with the LHS produced by an `arith.constant`:

```mlir
pdl_constr.pattern @addi_equal_operands benefit(1) root(%root : !pdl.operation) {
  // Constraint: operation name
  %c_name = pdl_constr.has_name %root, "arith.addi"

  // Constraint: exactly 2 operands
  %c_opcnt = pdl_constr.check_operand_count %root is 2

  // Navigate to operands (returns optional — may be null)
  %lhs_opt = pdl_constr.get_operand 0 of %root
      : !pdl_constr.optional<!pdl.value>
  %rhs_opt = pdl_constr.get_operand 1 of %root
      : !pdl_constr.optional<!pdl.value>

  // Unwrap operands: is_not_null checks AND produces the bare value
  %c_lhs_nn, %lhs = pdl_constr.is_not_null %lhs_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl_constr.pred, !pdl.value
  %c_rhs_nn, %rhs = pdl_constr.is_not_null %rhs_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl_constr.pred, !pdl.value

  // Constraint: operands are equal (uses unwrapped values)
  %c_eq = pdl_constr.equal %lhs, %rhs

  // Constraint: LHS has type i32 (get_value_type takes unwrapped !pdl.value)
  %lhs_type = pdl_constr.get_value_type of %lhs : !pdl.type
  %c_type = pdl_constr.has_type %lhs_type, i32

  // Navigate to defining op of LHS (returns optional)
  %def_op_opt = pdl_constr.get_defining_op of %lhs
      : !pdl_constr.optional<!pdl.operation>
  %c_def_nn, %def_op = pdl_constr.is_not_null %def_op_opt
      : !pdl_constr.optional<!pdl.operation> -> !pdl_constr.pred, !pdl.operation
  %c_def_name = pdl_constr.has_name %def_op, "arith.constant"

  // Combine all constraints
  %all = pdl_constr.all %c_name, %c_opcnt, %c_lhs_nn, %c_rhs_nn,
                        %c_eq, %c_type, %c_def_nn, %c_def_name
  pdl_constr.success %all
}
```

## Multi-Root Example with `get_each`

A pattern connecting two roots through users (upward traversal). Currently this
generates a `pdl_interp.foreach` loop; at the `pdl_constr` level the loop is
implicit:

```mlir
pdl_constr.pattern @multi_root benefit(1) root(%root : !pdl.operation) {
  %c_name = pdl_constr.has_name %root, "foo.producer"

  // Navigate downward to result (nullable), then unwrap
  %res_opt = pdl_constr.get_result 0 of %root
      : !pdl_constr.optional<!pdl.value>
  %c_res_nn, %res = pdl_constr.is_not_null %res_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl_constr.pred, !pdl.value

  // Navigate upward through users (non-nullable, returns range)
  %users = pdl_constr.get_users of %res : !pdl.range<operation>

  // get_each: extract one element (existential semantics)
  %user = pdl_constr.get_each %users : !pdl.range<operation> -> !pdl.operation

  // Constrain the user
  %c_user_name = pdl_constr.has_name %user, "foo.consumer"
  %user_operand_opt = pdl_constr.get_operand 0 of %user
      : !pdl_constr.optional<!pdl.value>
  %c_uo_nn, %user_operand = pdl_constr.is_not_null %user_operand_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl_constr.pred, !pdl.value
  %c_connected = pdl_constr.equal %user_operand, %res

  %all = pdl_constr.all %c_name, %c_res_nn, %c_user_name, %c_uo_nn, %c_connected
  pdl_constr.success %all
}
```

The `pdl_constr → pdl_interp` lowering sees `get_each` and generates the
`pdl_interp.foreach` loop with proper continue/failure blocks.

---

## Implementation Plan

### Step 1: Dialect and Ops (TableGen + C++)

**Files to create:**

```
mlir/include/mlir/Dialect/PDLConstr/IR/
  CMakeLists.txt
  PDLConstrDialect.td      # Dialect def, dependent on pdl + pdl_interp
  PDLConstrOps.td           # All ops defined above
  PDLConstr.h               # Include header

mlir/lib/Dialect/PDLConstr/IR/
  CMakeLists.txt
  PDLConstr.cpp             # Dialect + op implementations, verifiers
```

**Key implementation notes:**

- Define `PDLConstr_PredType` as an MLIR type (`pdl_constr.pred`).
- Define `PDLConstr_OptionalType<T>` as a parametric MLIR type
  (`pdl_constr.optional<T>`), where `T` is any PDL type.
- All constraint ops inherit from a common `PDLConstr_ConstraintOp` base class
  that always produces at least one `!pdl_constr.pred` result.
- `pdl_constr.is_not_null` additionally produces a second result of type `T`
  (the unwrapped inner type).
- Navigation ops that can fail (`get_operand`, `get_operands`, `get_result`,
  `get_results`, `get_attribute`, `get_defining_op`) return
  `!pdl_constr.optional<...>`. Non-nullable navigations (`get_value_type`,
  `get_attribute_type`, `get_users`) return bare PDL types.
- `pdl_constr.pattern` has an `IsolatedFromAbove` + `SingleBlock` region. The
  block has one `!pdl.operation` argument.
- Register the dialect in `mlir/include/mlir/InitAllDialects.h`.

### Step 2: PDL → pdl_constr Lowering

**Files to create:**

```
mlir/include/mlir/Conversion/PDLToPDLConstr/
  PDLToPDLConstr.h

mlir/lib/Conversion/PDLToPDLConstr/
  CMakeLists.txt
  PDLToPDLConstr.cpp
```

**This pass reuses existing infrastructure:**

- `detectRoots()`, `buildCostGraph()`, `OptimalBranching` from
  `RootOrdering.h` — root selection and multi-root ordering.
- The tree-walk logic from `getTreePredicates()` — but instead of building
  `PositionalPredicate` lists, emit `pdl_constr`/`pdl_interp` ops directly.

**Mapping from current code to ops emitted:**

| Current C++ call | Op emitted |
|---|---|
| `builder.getIsNotNull()` | `pdl_constr.is_not_null` (returns pred + unwrapped value) |
| `builder.getOperationName(name)` | `pdl_constr.has_name` |
| `builder.getEqualTo(pos)` | `pdl_constr.equal` |
| `builder.getOperandCount(n)` | `pdl_constr.check_operand_count` |
| `builder.getResultCount(n)` | `pdl_constr.check_result_count` |
| `builder.getTypeConstraint(t)` | `pdl_constr.has_type` |
| `builder.getAttributeConstraint(a)` | `pdl_constr.has_attr_value` |
| `builder.getConstraint(...)` | `pdl_constr.apply_native_constraint` |
| `builder.getForEach(usersPos, id)` | `pdl_constr.get_each` |
| Navigation positions | Corresponding `pdl_constr.get_*` ops |

**Navigation op emission:** the pass emits `pdl_constr.get_operand`,
`pdl_constr.get_result`, etc. (returning `!pdl_constr.optional<...>`).
Immediately after each nullable navigation, emit `pdl_constr.is_not_null` to
unwrap the optional into a bare PDL value + predicate. Downstream ops use the
unwrapped value. For non-nullable navigations (`get_value_type`,
`get_attribute_type`, `get_users`), emit the op directly — no unwrap needed.

**Upward traversals (multi-root):** instead of emitting `ForEachPosition`,
emit `pdl_constr.get_users` + `pdl_constr.get_each` + constraints on the
extracted element.

**End of pattern:** collect all `!pdl_constr.pred` values, emit
`pdl_constr.all`, emit `pdl_constr.success`.

**Rewrite handling:** the rewrite region from `pdl::PatternOp` is either:
- Preserved as an attribute/region on `pdl_constr.pattern`, or
- Lowered to the rewriter module at this stage (reuse existing rewriter
  codegen from `PatternLowering::generateRewriter`).

The simpler option is to preserve the rewrite region and lower it in Step 3.

### Step 3: pdl_constr → pdl_interp Lowering

**Files to create:**

```
mlir/include/mlir/Conversion/PDLConstrToPDLInterp/
  PDLConstrToPDLInterp.h

mlir/lib/Conversion/PDLConstrToPDLInterp/
  CMakeLists.txt
  PDLConstrToPDLInterp.cpp
```

**This pass does two things:**

1. **Extract predicates from IR:** Walk each `pdl_constr.pattern`, reconstruct
   `PositionalPredicate` triples from the ops. Each constraint op trivially
   maps back to a Question+Answer pair. Navigation ops map to Positions.
   `pdl_constr.is_not_null` maps to an `IsNotNullQuestion` on the position of
   its optional input; the unwrapped result maps to the same position (bare).

2. **Build matcher tree + emit pdl_interp:** Reuse the existing:
   - `OrderedPredicate` sorting / cost computation
   - `propagatePattern()` to build the merged `MatcherNode` tree
   - `foldSwitchToBool()`, `insertExitNode()`
   - `PatternLowering::generateMatcher()`, `generate(BoolNode*)`, etc.

**`pdl_constr.get_each` lowering:** When this op is encountered, reconstruct
a `ForEachPosition`+`UsersPosition` and generate the `pdl_interp.foreach` op
with continue/failure blocks, exactly as the current `getValueAt` case for
`Predicates::ForEachPos` does.

**`pdl_constr.all` / `pdl_constr.any` lowering:**
- `all`: the contained predicates are ordered and checked sequentially (failure
  on any → branch to failure). This is the default behavior today.
- `any`: generates a switch-like structure trying each alternative.

### Step 4: Registration and Testing

- Add pass declarations to `mlir/include/mlir/Conversion/Passes.td`.
- Register dialect in `mlir/include/mlir/InitAllDialects.h`.
- Add `CMakeLists.txt` entries in `mlir/lib/Dialect/CMakeLists.txt` and
  `mlir/lib/Conversion/CMakeLists.txt`.
- Convert existing tests in `mlir/test/Conversion/PDLToPDLInterp/` to also
  test the intermediate `pdl_constr` form.
- Add new tests for `pdl_constr` → `pdl_interp` directly.
- Optionally keep the old monolithic `PDL → PDL_Interp` pass as a composition
  of the two new passes.

---

## Design Decisions Summary

1. **Constraint ops return `!pdl_constr.pred`** — constraints are values, not
   control flow. This makes them composable via `all`/`any`.

2. **`all` and `any` combinators** — explicit aggregation of predicates.
   `success` takes a single pred (usually the result of `all`).

3. **No structured loops at this level** — `pdl_constr.get_each` replaces
   `ForEachPosition` / `pdl_interp.foreach`. Loop generation is deferred to
   the `pdl_constr → pdl_interp` lowering.

4. **`!pdl_constr.optional<T>` enforces null-safety** — nullable navigation
   ops return `!pdl_constr.optional<T>`. The only way to unwrap is
   `pdl_constr.is_not_null`, which produces both a `!pdl_constr.pred` and the
   bare value. This makes it a type error to use a potentially-null value
   without a null check, preventing constraint-before-check bugs.

5. **Own navigation ops instead of reusing `pdl_interp`** — because nullable
   navigations return `!pdl_constr.optional<T>`, the dialect defines its own
   `pdl_constr.get_operand`, `pdl_constr.get_result`, etc. Non-nullable
   navigations (`get_value_type`, `get_attribute_type`, `get_users`) return
   bare PDL types.

6. **One pattern = one `pdl_constr.pattern`** — cross-pattern merging into a
   shared matcher tree happens in the second lowering pass.

7. **Rewrite region is orthogonal** — preserved through `pdl_constr` and
   lowered to the rewriter module in the second pass.
