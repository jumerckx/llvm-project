# `pdl_constr` Dialect Design

## Motivation

The current PDL → PDL_Interp lowering hides all constraint, navigation, and
matcher-tree construction in C++ data structures (`Position*`, `Qualifier*`,
`MatcherNode`). This design introduces an intermediate dialect (`pdl_constr`)
that makes these constraints **and the matcher-tree shape** explicit in MLIR
IR, enabling inspection, transformation, and composition before the final
lowering to `pdl_interp`.

### Current Pipeline

```
pdl.pattern → [C++ MatcherNode tree] → pdl_interp
```

### Proposed Pipeline

```
pdl.pattern → [Pass 1: heavy] → pdl_constr IR → [Pass 2: mechanical] → pdl_interp
```

The first pass does the work — root ordering, predicate selection, cross-pattern
merging, and matcher-tree construction — and materializes the result as
`pdl_constr` IR. The second pass is a recursive region walker that mechanically
emits `pdl_interp` control flow.

---

## Conceptual Model

`pdl_constr` is an **explicitly ordered IR**. Position in the region determines
materialization order, and the matcher-tree shape is exposed via nested
regions.

| Aspect | Choice |
| :--- | :--- |
| **IR ordering** | Linear schedule; position matters |
| **AND of tests** | Implicit by linear sequence |
| **OR of alternatives** | Nested `pdl_constr.try` regions (`MatcherNode::failureNode` spine) |
| **Multi-way dispatch** | `pdl_constr.switch_op_name` / `pdl_constr.switch_type` (`SwitchNode`) |
| **Control flow on failure** | Implicit transfer to the enclosing failure scope |

There are no boolean SSA values. Test ops have an implicit control effect:
on failure, control transfers to the enclosing failure scope.

---

## Dialect Definition

**Namespace:** `pdl_constr`
**Dependent dialects:** `pdl::PDLDialect`, `pdl_interp::PDLInterpDialect`
**Location:** `mlir/include/mlir/Dialect/PDLConstr/IR/`

Reuses existing PDL types (`!pdl.operation`, `!pdl.value`, `!pdl.attribute`,
`!pdl.type`, `!pdl.range<...>`) throughout.

---

## Types

### `!pdl_constr.optional<T>`

A wrapper type indicating that the contained value may be null at runtime.
Nullable navigation ops (e.g. `get_operand`, `get_defining_op`) return
`!pdl_constr.optional<T>` instead of the bare PDL type.

The only way to convert from `!pdl_constr.optional<T>` to `T` is via
`pdl_constr.is_not_null`, which additionally has the implicit control effect
of failing the surrounding scope when the optional is null. This enforces
null-safety at the type level: no constraint can consume a potentially-null
value without a null check first.

`T` may be any PDL type: `!pdl.operation`, `!pdl.value`, `!pdl.attribute`,
`!pdl.type`, `!pdl.range<...>`.

> **Note**: there is no `!pdl_constr.pred` type. Tests do not produce SSA
> values; they have implicit failure control flow.

---

## Operations

### Structural

#### `pdl_constr.matcher`

Container for the navigation/constraint IR of one **matcher**. A matcher may
aggregate multiple logical sub-patterns that share a common navigation/test
prefix; cross-pattern merging is performed in Pass 1.

* `IsolatedFromAbove`, single-block region.
* Block argument is the root `!pdl.operation`.
* Optional symbol (so `pdl_constr.success` can be looked up if needed).
* No terminator — execution falls off the end if no `success` op is reached
  along the current path.

```mlir
pdl_constr.matcher @combined root(%root : !pdl.operation) {
  pdl_constr.has_name %root, "arith.addi"
  pdl_constr.try {
    // alternative 1
    pdl_constr.success @rewriters::@rewrite_a benefit(2)
  }
  pdl_constr.try {
    // alternative 2
    pdl_constr.success @rewriters::@rewrite_b benefit(1)
  }
}
```

#### `pdl_constr.try { ... }`

Introduces a new **failure scope**. Any test op that fails inside the region,
as well as fall-off-end of the region, transfers control to the op immediately
following the `pdl_constr.try` in its parent region.

This is the direct IR analogue of `MatcherNode::failureNode`: a chain of
sibling `pdl_constr.try` ops at the same region level encodes the OR of
alternative match attempts.

Single block, no block arguments. Values defined in enclosing regions are
visible inside.

#### `pdl_constr.switch_op_name` / `pdl_constr.switch_type`

Multi-way dispatches representing `SwitchNode`. Each carries one case region
per case key (`caseNames : StrArrayAttr` / `caseTypes : TypeArrayAttr`). When
the runtime value matches case `i`, control enters case region `i`. If no case
matches, control transfers to the enclosing failure scope (implicit default).
Case regions inherit the enclosing failure scope.

Without these ops, opcode/type dispatch would have to be expressed as a chain
of equality `try` blocks; making the switch first-class lets the lowering emit
a real `pdl_interp.switch_*`.

### Navigation Ops

Navigation ops extract sub-values from the matched IR. They correspond to
Position kinds in the current `Predicate.h`.

#### Nullable navigations (return `!pdl_constr.optional<T>`)

The value may not exist at runtime. Downstream constraint ops cannot consume
the result directly — it must first be unwrapped via `pdl_constr.is_not_null`.

| Current Position Kind | `pdl_constr` op | Signature |
|---|---|---|
| `OperandPos` | `pdl_constr.get_operand` | `(op, index) → optional<!pdl.value>` |
| `OperandGroupPos` | `pdl_constr.get_operands` | `(op, opt<index>) → optional<!pdl.range<value>>` |
| `ResultPos` | `pdl_constr.get_result` | `(op, index) → optional<!pdl.value>` |
| `ResultGroupPos` | `pdl_constr.get_results` | `(op, opt<index>) → optional<!pdl.range<value>>` |
| `AttributePos` | `pdl_constr.get_attribute` | `(op, name) → optional<!pdl.attribute>` |
| `OperationPos` (def) | `pdl_constr.get_defining_op` | `(!pdl.value or range) → optional<!pdl.operation>` |

#### Non-nullable navigations (return bare PDL types)

These navigate from an already-validated value to a property guaranteed to
exist.

| Current Position Kind | `pdl_constr` op | Signature |
|---|---|---|
| `TypePos` (value) | `pdl_constr.get_value_type` | `(!pdl.value or range) → !pdl.type or range` |
| `TypePos` (attr) | `pdl_constr.get_attribute_type` | `(!pdl.attribute) → !pdl.type` |
| `UsersPos` | `pdl_constr.get_users` | `(!pdl.value) → !pdl.range<operation>` |

#### `pdl_constr.get_each`

Extracts a single element from a range (existential semantics). At this level
there is no structured loop; `get_each` declares that the pattern applies
constraints to each element of the range. The `pdl_constr → pdl_interp`
lowering generates the actual `pdl_interp.foreach` loop, replacing
`ForEachPosition` + `pdl_interp.foreach`.

```mlir
%elem = pdl_constr.get_each %range : !pdl.range<operation> -> !pdl.operation
```

### Constraint / Test Ops

Test ops have an **implicit control effect**: on failure, control transfers
to the enclosing failure scope. They are not `Pure`; they must not be DCE'd
even though they typically produce no SSA results. They take **unwrapped**
PDL types as operands (never `optional<T>`).

#### `pdl_constr.is_not_null` (unwrap + check)

Checks that an optional value is non-null and produces the bare inner value.
If the optional is null, control transfers to the enclosing failure scope.

```mlir
%val = pdl_constr.is_not_null %opt : !pdl_constr.optional<!pdl.value> -> !pdl.value
```

By SSA dominance the unwrapped result can only be used by ops below this one,
mirroring the control-flow guarantee in the lowered form. A verifier ensures
the result type matches the optional's inner type.

#### Other constraint ops

| Current Question | `pdl_constr` op | Operands | Attributes |
|---|---|---|---|
| `OperationNameQuestion` | `pdl_constr.has_name` | `(!pdl.operation)` | `name : StrAttr` |
| `EqualToQuestion` | `pdl_constr.equal` | `(lhs, rhs)` | — (`SameTypeOperands`) |
| `TypeQuestion` | `pdl_constr.has_type` | `(!pdl.type)` | `constantType : TypeAttr` |
| `TypeQuestion` (range) | `pdl_constr.has_types` | `(!pdl.range<type>)` | `constantTypes : TypeArrayAttr` |
| `AttributeQuestion` | `pdl_constr.has_attr_value` | `(!pdl.attribute)` | `value : AnyAttr` |
| `OperandCountQuestion` | `pdl_constr.check_operand_count` | `(!pdl.operation)` | `count : I32Attr`, opt `atLeast : UnitAttr` |
| `ResultCountQuestion` | `pdl_constr.check_result_count` | `(!pdl.operation)` | `count : I32Attr`, opt `atLeast : UnitAttr` |
| `ConstraintQuestion` | `pdl_constr.apply_native_constraint` | `(args...)` | `name : StrAttr`, opt `isNegated : BoolAttr` |

`apply_native_constraint` may also produce additional `!pdl.*` results that
downstream ops consume; the test itself is the implicit failure effect.

### `pdl_constr.success`

Marks a successful pattern match. Carries:

* a symbol reference to a rewriter (`@rewriters::@my_rewriter`),
* a `benefit` attribute (a single matcher may hold sub-patterns of different
  benefits), and
* the variadic list of match values to forward to the rewriter.

```mlir
pdl_constr.success @rewriters::@rewriter
    benefit(1) (%root, %lhs : !pdl.operation, !pdl.value)
```

The signature mirrors `pdl_interp.record_match` so the rewriter metadata
passes through unchanged. The precondition is implicit: every test op above
this point on the path from the matcher root must have succeeded.

A `pdl_constr.success` must be enclosed by a `pdl_constr.matcher` (verified).

---

## Full Example

A pattern matching `arith.addi` where both operands are equal and have type
`i32`, with the LHS produced by an `arith.constant`:

```mlir
pdl_constr.matcher @addi_equal_operands root(%root : !pdl.operation) {
  pdl_constr.has_name %root, "arith.addi"
  pdl_constr.check_operand_count %root is 2

  %lhs_opt = pdl_constr.get_operand 0 of %root
      : !pdl_constr.optional<!pdl.value>
  %rhs_opt = pdl_constr.get_operand 1 of %root
      : !pdl_constr.optional<!pdl.value>
  %lhs = pdl_constr.is_not_null %lhs_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl.value
  %rhs = pdl_constr.is_not_null %rhs_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl.value

  pdl_constr.equal %lhs, %rhs : !pdl.value

  %lhs_type = pdl_constr.get_value_type of %lhs : !pdl.value : !pdl.type
  pdl_constr.has_type %lhs_type, i32

  %def_opt = pdl_constr.get_defining_op of %lhs
      : !pdl.value -> !pdl_constr.optional<!pdl.operation>
  %def = pdl_constr.is_not_null %def_opt
      : !pdl_constr.optional<!pdl.operation> -> !pdl.operation
  pdl_constr.has_name %def, "arith.constant"

  pdl_constr.success @rewriters::@addi_equal_operands
      benefit(1) (%root : !pdl.operation)
}
```

Every test is implicitly ANDed by linear order; any failure transfers control
to the enclosing failure scope (here, fall-off-end of the matcher).

## Multi-Root Example with `get_each`

```mlir
pdl_constr.matcher @multi_root root(%root : !pdl.operation) {
  pdl_constr.has_name %root, "foo.producer"

  %res_opt = pdl_constr.get_result 0 of %root
      : !pdl_constr.optional<!pdl.value>
  %res = pdl_constr.is_not_null %res_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl.value

  %users = pdl_constr.get_users of %res : !pdl.range<operation>
  %user = pdl_constr.get_each %users
      : !pdl.range<operation> -> !pdl.operation

  pdl_constr.has_name %user, "foo.consumer"
  %uo_opt = pdl_constr.get_operand 0 of %user
      : !pdl_constr.optional<!pdl.value>
  %uo = pdl_constr.is_not_null %uo_opt
      : !pdl_constr.optional<!pdl.value> -> !pdl.value
  pdl_constr.equal %uo, %res : !pdl.value

  pdl_constr.success @rewriters::@multi_root
      benefit(1) (%root, %user : !pdl.operation, !pdl.operation)
}
```

The `pdl_constr → pdl_interp` lowering sees `get_each` and generates the
`pdl_interp.foreach` loop with proper continue/failure blocks.

---

## Pass Structure

### Step 1: `pdl` → `pdl_constr` (heavy)

This pass absorbs the matcher-tree construction logic previously in the
monolithic lowering. It:

1. Uses `RootOrdering.h` (`detectRoots`, `buildCostGraph`, `OptimalBranching`)
   for root selection and multi-root ordering.
2. Walks the resulting `MatcherNode` tree and serializes it into `pdl_constr`
   IR using nested `try` and `switch_*` regions for the failure spine and
   multi-way dispatch.
3. Performs cross-pattern merging at this stage: multiple logical patterns
   sharing a navigation/test prefix become sub-paths in the same
   `pdl_constr.matcher`, each terminated by its own `pdl_constr.success`.
4. Lowers the `pdl.rewrite` region eagerly into a `pdl_interp.func` placed in
   a nested `@rewriters` symbol module. `pdl_constr.success` carries a symbol
   reference to that func plus the set of match values it consumes (mirroring
   `pdl_interp.record_match`).

**Op emission map:**

| Current C++ call | Op emitted |
|---|---|
| `builder.getIsNotNull()` | `pdl_constr.is_not_null` (unwrap, fails on null) |
| `builder.getOperationName(name)` | `pdl_constr.has_name` |
| `builder.getEqualTo(pos)` | `pdl_constr.equal` |
| `builder.getOperandCount(n)` | `pdl_constr.check_operand_count` |
| `builder.getResultCount(n)` | `pdl_constr.check_result_count` |
| `builder.getTypeConstraint(t)` | `pdl_constr.has_type` / `has_types` |
| `builder.getAttributeConstraint(a)` | `pdl_constr.has_attr_value` |
| `builder.getConstraint(...)` | `pdl_constr.apply_native_constraint` |
| `builder.getForEach(usersPos, id)` | `pdl_constr.get_each` |
| Navigation positions | Corresponding `pdl_constr.get_*` ops |
| `MatcherNode::failureNode` chain | Sibling `pdl_constr.try` regions |
| `SwitchNode` | `pdl_constr.switch_op_name` / `switch_type` |

### Step 2: `pdl_constr` → `pdl_interp` (mechanical)

A simple recursive region walker. It maintains two block pointers as it
descends:

* `currentBlock` — where the next `pdl_interp` op is emitted.
* `failureBlock` — the destination for failure branches in the current scope.

Lowering rules:

* **Test ops** (`has_name`, `equal`, `is_not_null`, …) → corresponding
  `pdl_interp.check_*` / `pdl_interp.are_equal` / `pdl_interp.is_not_null`
  with the failure branch pointing at `failureBlock`.
* **Navigation ops** → corresponding `pdl_interp.get_*` ops.
* **`pdl_constr.try`** → create a fresh "after-try" block; lower the body
  with `failureBlock` set to that block; on completion, branch to that block
  and continue lowering siblings there.
* **`pdl_constr.switch_*`** → `pdl_interp.switch_*` with case successors
  lowered recursively and the default successor set to `failureBlock`.
* **`pdl_constr.get_each`** → reconstruct the `ForEachPosition` and emit
  `pdl_interp.foreach` with proper continue/failure wiring.
* **`pdl_constr.success`** → `pdl_interp.record_match` referencing the same
  symbol and forwarding the same inputs.

No re-derivation of `OrderedPredicate` cost or `MatcherNode` construction is
needed here: the IR already encodes those decisions.

### Step 3: Registration and Testing

* Pass declarations in `mlir/include/mlir/Conversion/Passes.td`.
* Dialect registration in `mlir/include/mlir/InitAllDialects.h`.
* CMake glue in `mlir/lib/Dialect/CMakeLists.txt` and
  `mlir/lib/Conversion/CMakeLists.txt`.
* Convert existing `mlir/test/Conversion/PDLToPDLInterp/` tests to also pin
  the intermediate `pdl_constr` form; add direct
  `pdl_constr` → `pdl_interp` tests.
* Optionally keep the monolithic `PDL → PDL_Interp` pass as a composition of
  the two new passes.

---

## Verification

* **`MatcherOp` / `TryOp`**: must contain at least one transitive
  `pdl_constr.success` (otherwise the region is dead and would silently fall
  through).
* **`IsNotNullOp`**: the unwrapped result type must match the inner type of
  the optional operand.
* **`SwitchOpNameOp` / `SwitchTypeOp`**: number of case regions must equal
  the size of the `caseNames` / `caseTypes` array.
* **`SuccessOp`**: must be enclosed by a `pdl_constr.matcher`, and the
  rewriter symbol must resolve.

---

## Design Decisions Summary

1. **Explicitly ordered IR; no boolean SSA.** AND is implicit by linear
   order; OR is expressed by sibling `pdl_constr.try` regions; multi-way
   dispatch by `switch_*`. Test ops have an implicit failure control effect.
   The previous design's `!pdl_constr.pred` type, `pdl_constr.all`, and
   `pdl_constr.any` are not needed and were dropped.

2. **Matcher-tree shape is first-class.** `try` mirrors
   `MatcherNode::failureNode`; `switch_*` mirrors `SwitchNode`. This moves
   the "heavy" decision-making into Pass 1 and makes Pass 2 a mechanical
   region walker.

3. **`!pdl_constr.optional<T>` enforces null-safety.** Nullable navigations
   return `optional<T>`; the only way to unwrap is `pdl_constr.is_not_null`,
   which both fails the surrounding scope on null and produces the bare
   value. SSA dominance then guarantees that consumers run only when the
   value is known non-null.

4. **Own navigation ops instead of reusing `pdl_interp`** — because nullable
   navigations return `optional<T>` and there is no concept of branching at
   this level. Non-nullable navigations (`get_value_type`,
   `get_attribute_type`, `get_users`) return bare PDL types directly.

5. **`get_each` defers loop generation.** No structured loops at this level;
   the `pdl_interp.foreach` is created in Pass 2.

6. **One `pdl_constr.matcher` can hold multiple logical patterns** that share
   a navigation/test prefix. Each terminates at its own `pdl_constr.success`
   carrying its own `benefit` and rewriter symbol. Cross-pattern merging
   happens in Pass 1.

7. **Rewriters live in a symbol module.** Pass 1 emits a `pdl_interp.func`
   per rewriter in a nested `@rewriters` module; `pdl_constr.success` refers
   to it by symbol and forwards the same match values that
   `pdl_interp.record_match` will. The rewriter metadata is preserved
   unchanged through Pass 2.
