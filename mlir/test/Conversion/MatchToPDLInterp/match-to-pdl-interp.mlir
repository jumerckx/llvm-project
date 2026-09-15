// RUN: mlir-opt %s -split-input-file -convert-match-to-pdl-interp | FileCheck %s

// Lowering of match IR to a `pdl_interp.func @matcher`. Inputs are written
// directly in match IR because `-convert-pdl-to-match` cannot produce most of
// these shapes: it never emits `try`, `switch_*`, or more than one
// `match.success` per matcher.
//
// The invariant under test throughout is the failure edge. Every test op becomes
// a two-successor `pdl_interp` op whose second successor is the enclosing
// failure block, and each construct decides what that block is:
//
//   matcher body   -> the finalize block
//   try            -> the block following the try ("after-try")
//   switch case    -> the enclosing failure block (cases do not catch)
//   foreach body   -> the loop's continue block
//
// Block numbering is incidental but asserted anyway: these inputs are small and
// hand-written, so the numbering is stable, and getting the predecessor sets
// right is most of what this pass does.

//===----------------------------------------------------------------------===//
// An empty module still gets a matcher function.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.branch ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
module {
  module @rewriters { module @r {} }
}

// -----

//===----------------------------------------------------------------------===//
// A flat matcher: a linear block chain whose every failure edge targets the
// finalize block, and a trailing continuation block after the record_match.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.check_operation_name of %arg0 is "foo.op" -> ^bb2, ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    pdl_interp.check_operand_count of %arg0 is 1 -> ^bb3, ^bb1
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r(%arg0 : !pdl.operation) : benefit(1), loc([%arg0]), root("foo.op") -> ^bb4
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    pdl_interp.branch ^bb1
module {
  module @rewriters { module @r {} }
  match.matcher @flat root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(1) (%root : !pdl.operation)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Two matchers in one module are *chained*: the first matcher's failure block is
// the second matcher's entry, and only the last one falls through to finalize.
// Nothing else covers this — `-match-combine-matchers` normally collapses a
// module to a single matcher first.
//
// The `root(...)` hint is also per-matcher: "foo.op" does not leak into the
// second matcher's record_match (`opNameMap` is cleared per matcher).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.check_operation_name of %arg0 is "foo.op" -> ^bb3, ^bb2
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    pdl_interp.check_operation_name of %arg0 is "bar.op" -> ^bb5, ^bb1
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]), root("foo.op") -> ^bb4
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    pdl_interp.branch ^bb2
// CHECK-NEXT:  ^bb5:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(2), loc([%arg0]), root("bar.op") -> ^bb6
// CHECK-NEXT:  ^bb6:
// CHECK-NEXT:    pdl_interp.branch ^bb1
module {
  module @rewriters { module @r {} }
  match.matcher @first root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @second root(%root : !pdl.operation) {
    match.has_name %root, "bar.op"
    match.success @rewriters::@r benefit(2)
  }
}

// -----

//===----------------------------------------------------------------------===//
// match.try: failures inside the body branch to the after-try block, and ops
// following the try are lowered *into* that block.
//
// The `root(...)` hint is path-local: the first record_match is inside the try,
// where `has_name` has been checked, so it gets `root("foo.op")`. The second is
// in the after-try block, reached precisely when the try *failed* -- possibly
// because the name did not match -- so it must have no `root(...)`. Recording
// it under `root("foo.op")` would key the pattern to `foo.op` in the bytecode
// and it would never be attempted on any other operation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.check_operation_name of %arg0 is "foo.op" -> ^bb3, ^bb2
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// The after-try block: reached from the entry (name check failed) and from ^bb4
// (the try's success fell through).
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    pdl_interp.check_operand_count of %arg0 is 2 -> ^bb5, ^bb1
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]), root("foo.op") -> ^bb4
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    pdl_interp.branch ^bb2
// CHECK-NEXT:  ^bb5:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(2), loc([%arg0]) -> ^bb6
module {
  module @rewriters { module @r {} }
  match.matcher @with_try root(%root : !pdl.operation) {
    match.try {
      match.has_name %root, "foo.op"
      match.success @rewriters::@r benefit(1)
    }
    match.check_operand_count %root is 2
    match.success @rewriters::@r benefit(2)
  }
}

// -----

// A nested try is a second level of failure scope: the inner body's failures go
// to the inner after-try (^bb3), whose own failures go to the outer after-try
// (^bb2), and only that falls through to finalize.
//
// It also shows the `root(...)` scoping at two levels: only benefit(1), emitted
// inside both trys, carries `root("inner.op")`. benefit(2) is one level out and
// benefit(3) two, and each is reached by a path on which the name check may
// have been what failed, so neither gets a root.
// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.check_operation_name of %arg0 is "inner.op" -> ^bb4, ^bb3
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    pdl_interp.check_result_count of %arg0 is 3 -> ^bb8, ^bb1
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.check_operand_count of %arg0 is 1 -> ^bb6, ^bb2
// Innermost success keeps the root; the two outer ones must not have it.
// CHECK:         pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]), root("inner.op")
// CHECK:         pdl_interp.record_match @rewriters::@r : benefit(2), loc([%arg0]) ->
// CHECK:         pdl_interp.record_match @rewriters::@r : benefit(3), loc([%arg0]) ->
module {
  module @rewriters { module @r {} }
  match.matcher @nested_try root(%root : !pdl.operation) {
    match.try {
      match.try {
        match.has_name %root, "inner.op"
        match.success @rewriters::@r benefit(1)
      }
      match.check_operand_count %root is 1
      match.success @rewriters::@r benefit(2)
    }
    match.check_result_count %root is 3
    match.success @rewriters::@r benefit(3)
  }
}

// -----

//===----------------------------------------------------------------------===//
// A switch that is *not* last in its region: the default edge targets a fresh
// fall-through block, each case body falls through to it as well, and the ops
// after the switch are lowered into it.
//
// Note the trailing record_match has no `root(...)`: the case names are scoped
// to their case regions and correctly do not escape the switch.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.switch_operation_name of %arg0 to ["a.op", "b.op"](^bb3, ^bb4) -> ^bb2
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    pdl_interp.check_operand_count of %arg0 is 9 -> ^bb7, ^bb1
// Each case records its match under its own name, then falls through.
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]), root("a.op") -> ^bb5
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(2), loc([%arg0]), root("b.op") -> ^bb6
// CHECK-NEXT:  ^bb5:
// CHECK-NEXT:    pdl_interp.branch ^bb2
// CHECK-NEXT:  ^bb6:
// CHECK-NEXT:    pdl_interp.branch ^bb2
// CHECK-NEXT:  ^bb7:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(3), loc([%arg0]) -> ^bb8
module {
  module @rewriters { module @r {} }
  match.matcher @sw_mid root(%root : !pdl.operation) {
    match.switch_op_name %root
    case "a.op" { match.success @rewriters::@r benefit(1) }
    case "b.op" { match.success @rewriters::@r benefit(2) }
    match.check_operand_count %root is 9
    match.success @rewriters::@r benefit(3)
  }
}

// -----

// A switch that *is* last in its region needs no fall-through block: the default
// edge is the enclosing failure block directly.
// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.switch_operation_name of %arg0 to ["a.op", "b.op"](^bb2, ^bb3) -> ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
module {
  module @rewriters { module @r {} }
  match.matcher @sw_last root(%root : !pdl.operation) {
    match.switch_op_name %root
    case "a.op" { match.success @rewriters::@r benefit(1) }
    case "b.op" { match.success @rewriters::@r benefit(2) }
  }
}

// -----

// The same shape for switch_type.
// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK:         pdl_interp.switch_type %{{.*}} to [i32, i64]
module {
  module @rewriters { module @r {} }
  match.matcher @sw_type root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    match.switch_type %t
    case i32 { match.success @rewriters::@r benefit(1) }
    case i64 { match.success @rewriters::@r benefit(2) }
  }
}

// -----

// A `switch_type` case body is a failure scope too, so a `has_name` inside one
// does not reach the fall-through: only benefit(1) is keyed to `foo.op`.
// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-DAG:     pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]), root("foo.op")
// CHECK-DAG:     pdl_interp.record_match @rewriters::@r0 : benefit(2), loc([%arg0]) ->
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @sw_type_name_scope root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    match.switch_type %t
    case i32 {
      match.has_name %root, "foo.op"
      match.success @rewriters::@r benefit(1)
    }
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

//===----------------------------------------------------------------------===//
// `match.foreach` becomes a `pdl_interp.foreach`. Blocks for its body ops are
// created *inside* the loop region, failures inside the body branch to the
// loop's `continue` block, and -- because the loop is the last op in its
// region -- loop exhaustion branches straight to the outer failure block.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    %[[RS:.*]] = pdl_interp.get_results of %arg0 : !pdl.range<value>
// CHECK-NEXT:    pdl_interp.is_not_null %[[RS]] : !pdl.range<value> -> ^bb2, ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// CHECK-NEXT:  ^bb2:
// The loop variable is a new block argument of the foreach region.
// CHECK-NEXT:    pdl_interp.foreach %[[E:.*]] : !pdl.value in %[[RS]] {
// CHECK-NEXT:      %[[D:.*]] = pdl_interp.get_defining_op of %[[E]] : !pdl.value
// CHECK-NEXT:      pdl_interp.is_not_null %[[D]] : !pdl.operation -> ^bb2, ^bb1
// Failures inside the body go to `continue`, not to the function's finalize.
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      pdl_interp.continue
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:      pdl_interp.check_operation_name of %[[D]] is "inner.op" -> ^bb3, ^bb1
// The loop variable's defining op joins the loc list.
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:      pdl_interp.record_match @rewriters::@r(%arg0, %[[D]] : !pdl.operation, !pdl.operation) : benefit(1), loc([%arg0, %[[D]]]) -> ^bb4
// CHECK-NEXT:    ^bb4:
// CHECK-NEXT:      pdl_interp.branch ^bb1
// Exhausting the loop falls through to the enclosing failure block.
// CHECK-NEXT:    } -> ^bb1
module {
  module @rewriters { module @r {} }
  match.matcher @each root(%root : !pdl.operation) {
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    match.foreach %e in %vs : !pdl.range<value> {
      %d = match.get_defining_op of %e : !pdl.value -> !match.optional<!pdl.operation>
      %dop = match.is_not_null %d : !match.optional<!pdl.operation> -> !pdl.operation
      match.has_name %dop, "inner.op"
      match.success @rewriters::@r benefit(1) (%root, %dop : !pdl.operation, !pdl.operation)
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// A `match.foreach` that is *not* last in its region: exhausting the range
// falls through to a fresh after-loop block into which the following siblings
// are lowered. This is the "no element matched, try the next alternative"
// shape, which the region form makes expressible.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    %[[RS:.*]] = pdl_interp.get_results of %arg0 : !pdl.range<value>
// CHECK-NEXT:    pdl_interp.is_not_null %[[RS]] : !pdl.range<value> -> ^bb2, ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    pdl_interp.finalize
// CHECK-NEXT:  ^bb2:
// CHECK-NEXT:    pdl_interp.foreach %[[E:.*]] : !pdl.value in %[[RS]] {
// CHECK-NEXT:      %[[T:.*]] = pdl_interp.get_value_type of %[[E]] : !pdl.type
// CHECK-NEXT:      pdl_interp.check_type %[[T]] is i32 -> ^bb2, ^bb1
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:      pdl_interp.continue
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:      pdl_interp.record_match @rewriters::@r{{.*}}benefit(2){{.*}} -> ^bb3
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:      pdl_interp.branch ^bb1
// Loop exhaustion lands in the after-loop block, not the finalize block.
// CHECK-NEXT:    } -> ^bb3
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.check_operation_name of %arg0 is "fallback.op" -> ^bb4, ^bb1
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r0{{.*}}benefit(1){{.*}} -> ^bb5
// CHECK-NEXT:  ^bb5:
// CHECK-NEXT:    pdl_interp.branch ^bb1
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @each_then_alternative root(%root : !pdl.operation) {
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    match.foreach %e in %vs : !pdl.range<value> {
      %t = match.get_value_type of %e : !pdl.value : !pdl.type
      match.has_type %t, i32
      match.success @rewriters::@r benefit(2)
    }
    match.has_name %root, "fallback.op"
    match.success @rewriters::@r0 benefit(1)
  }
}

// -----

// A name checked *inside* the loop body holds only for that iteration: the
// after-loop block is reached once every element has failed. So benefit(1),
// recorded in the body, is keyed to `foo.op`, while benefit(2), recorded after
// the loop, must not be. The loop variable is likewise dropped from `loc()`.
// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-DAG:     pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0, %{{.*}}]), root("foo.op")
// CHECK-DAG:     pdl_interp.record_match @rewriters::@r0 : benefit(2), loc([%arg0]) ->
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @foreach_name_scope root(%root : !pdl.operation) {
    %os = match.get_operands of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %os : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    match.foreach %e in %vs : !pdl.range<value> {
      %users = match.get_users of %e : !pdl.range<operation>
      match.foreach %u in %users : !pdl.range<operation> {
        match.has_name %root, "foo.op"
        match.success @rewriters::@r benefit(1)
      }
    }
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Several successes in one region chain through continuation blocks; the last
// continuation falls through to the failure block.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK:       ^bb2:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]), root("foo.op") -> ^bb3
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    pdl_interp.record_match @rewriters::@r0 : benefit(2), loc([%arg0]), root("foo.op") -> ^bb4
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    pdl_interp.branch ^bb1
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @multi root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.success @rewriters::@r benefit(1)
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

//===----------------------------------------------------------------------===//
// The constraint / rewrite asymmetry: `apply_constraint` is a test and gets two
// successors; `apply_rewrite` cannot fail, so it stays in the current block with
// no successors at all.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    %[[A:.*]] = pdl_interp.apply_rewrite "mk"(%arg0 : !pdl.operation) : !pdl.attribute
// CHECK-NEXT:    pdl_interp.apply_constraint "c"(%[[A]] : !pdl.attribute) -> ^bb2, ^bb1
module {
  module @rewriters { module @r {} }
  match.matcher @nat root(%root : !pdl.operation) {
    %a = match.apply_native_rewrite "mk"(%root : !pdl.operation) : !pdl.attribute
    match.apply_native_constraint "c"(%a : !pdl.attribute)
    match.success @rewriters::@r benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// `isNegated` is forwarded, and `is_not_null` maps its operand and its unwrapped
// result to the *same* `pdl_interp` value: %0 is produced once by `get_operand`
// and used directly by the downstream `get_value_type`.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK-NEXT:    pdl_interp.apply_constraint "c"(%arg0 : !pdl.operation) is_negated = true -> ^bb2, ^bb1
// CHECK:       ^bb2:
// CHECK-NEXT:    %[[O:.*]] = pdl_interp.get_operand 0 of %arg0
// CHECK-NEXT:    pdl_interp.is_not_null %[[O]] : !pdl.value -> ^bb3, ^bb1
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    %[[T:.*]] = pdl_interp.get_value_type of %[[O]] : !pdl.type
// CHECK-NEXT:    pdl_interp.check_type %[[T]] is i32 -> ^bb4, ^bb1
module {
  module @rewriters { module @r {} }
  match.matcher @neg root(%root : !pdl.operation) {
    match.apply_native_constraint "c"(%root : !pdl.operation) is_negated = true
    %o = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %o : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    match.has_type %t, i32
    match.success @rewriters::@r benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// No `has_name` at all, and `has_name` on a non-root op: neither produces a
// `root(...)` hint. The second also shows the loc list growing with the ops
// discovered by navigation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK:         pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0]) -> ^bb3
// CHECK-NOT:     root(
module {
  module @rewriters { module @r {} }
  match.matcher @noname root(%root : !pdl.operation) {
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(1)
  }
}

// -----

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK:         pdl_interp.check_operation_name of %[[D:.*]] is "other.op"
// CHECK:         pdl_interp.record_match @rewriters::@r : benefit(1), loc([%arg0, %[[D]]]) -> ^bb5
module {
  module @rewriters { module @r {} }
  match.matcher @nonroot root(%root : !pdl.operation) {
    %o = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %o : !match.optional<!pdl.value> -> !pdl.value
    %d = match.get_defining_op of %v : !pdl.value -> !match.optional<!pdl.operation>
    %dop = match.is_not_null %d : !match.optional<!pdl.operation> -> !pdl.operation
    match.has_name %dop, "other.op"
    match.success @rewriters::@r benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Remaining test ops, one lowering each.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK:         pdl_interp.check_operand_count of %arg0 is at_least 1
// CHECK:         pdl_interp.check_result_count of %arg0 is at_least 1
// CHECK:         %[[A:.*]] = pdl_interp.get_attribute "attr" of %arg0
// CHECK:         pdl_interp.is_not_null %[[A]] : !pdl.attribute
// CHECK:         pdl_interp.check_attribute %[[A]] is 10 : i64
// CHECK:         %[[AT:.*]] = pdl_interp.get_attribute_type of %[[A]]
// CHECK:         pdl_interp.check_type %[[AT]] is i64
// CHECK:         %[[RS:.*]] = pdl_interp.get_results of %arg0 : !pdl.range<value>
// CHECK:         %[[TS:.*]] = pdl_interp.get_value_type of %[[RS]] : !pdl.range<type>
// CHECK:         pdl_interp.check_types %[[TS]] are [i32]
// CHECK:         %[[EX:.*]] = pdl_interp.extract 0 of %[[RS]] : !pdl.value
// CHECK:         pdl_interp.are_equal
module {
  module @rewriters { module @r {} }
  match.matcher @tests root(%root : !pdl.operation) {
    match.check_operand_count %root is at_least 1
    match.check_result_count %root is at_least 1

    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>
    %av = match.is_not_null %a : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %av is 10 : i64
    %at = match.get_attribute_type of %av : !pdl.type
    match.has_type %at, i64

    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    %ts = match.get_value_type of %vs : !pdl.range<value> : !pdl.range<type>
    match.has_types %ts, [i32]

    %ex = match.extract 0 of %vs : !pdl.value
    %o = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %ov = match.is_not_null %o : !match.optional<!pdl.value> -> !pdl.value
    match.equal %ex, %ov : !pdl.value
    match.success @rewriters::@r benefit(1)
  }
}

// -----

// `get_users` feeding a `match.foreach` becomes a `pdl_interp.foreach` over
// the users range.
// CHECK-LABEL: pdl_interp.func @matcher(%arg0: !pdl.operation) {
// CHECK:         %[[U:.*]] = pdl_interp.get_users of %{{.*}} : !pdl.value
// CHECK:         pdl_interp.foreach %{{.*}} : !pdl.operation in %[[U]]
module {
  module @rewriters { module @r {} }
  match.matcher @users root(%root : !pdl.operation) {
    %o = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %o : !match.optional<!pdl.value> -> !pdl.value
    %u = match.get_users of %v : !pdl.range<operation>
    match.foreach %e in %u : !pdl.range<operation> {
      match.has_name %e, "user.op"
      match.success @rewriters::@r benefit(1) (%root, %e : !pdl.operation, !pdl.operation)
    }
  }
}

// -----

// The `constant_*` ops are value producers with no failure edge: they lower to
// `pdl_interp.create_*` in the current block, without the extra success block
// that a test op needs. The `CHECK-NEXT` chain below asserts exactly that (no
// intervening block label).

// CHECK-LABEL: pdl_interp.func @matcher
// CHECK: %[[ATTR:.*]] = pdl_interp.create_attribute 10 : i64
// CHECK-NEXT: %[[TYPE:.*]] = pdl_interp.create_type i32
// CHECK-NEXT: %[[TYPES:.*]] = pdl_interp.create_types [i32, i64]
// CHECK-NEXT: pdl_interp.apply_constraint "c"(%[[ATTR]], %[[TYPE]], %[[TYPES]] : !pdl.attribute, !pdl.type, !pdl.range<type>)
module {
  module @rewriters {
    module @rewriter {}
  }

  match.matcher @constants root(%root : !pdl.operation) {
    %attr = match.constant_attribute 10 : i64
    %type = match.constant_type i32
    %types = match.constant_types [i32, i64]
    match.apply_native_constraint "c"(%attr, %type, %types : !pdl.attribute, !pdl.type, !pdl.range<type>)
    match.success @rewriters::@rewriter benefit(1) (%root : !pdl.operation)
  }
}
