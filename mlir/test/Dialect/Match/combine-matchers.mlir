// RUN: mlir-opt %s -split-input-file -match-combine-matchers | FileCheck %s

// `-match-combine-matchers` merges every `match.matcher` in a module into one:
// predicates shared by several input matchers are emitted once in a shared
// prefix, and each divergent suffix becomes a sibling `match.try`. A contiguous
// run of alternatives that differ only in `has_name` / `has_type` on the same
// value folds into a `switch_op_name` / `switch_type`.
//
// Inputs here are written directly in match IR rather than piped through
// `-convert-pdl-to-match`: they stay small and each isolates a single decision
// of the pass.
//
// The pass is deliberately not idempotent — it refuses input matchers that
// already contain a `try` or `switch_*`; see combine-matchers-invalid.mlir.
//
// Not yet covered here: the guard that blocks a switch fold when the tested
// value is defined inside the `try` body itself, and topological stabilisation
// of a high-cost op that consumes a low-cost op's result. Both need a more
// contrived input than the cases below.

//===----------------------------------------------------------------------===//
// Degenerate inputs
//===----------------------------------------------------------------------===//

// A module with no matchers is left alone.
// CHECK-LABEL: module
// CHECK-NOT:   match.matcher
module {
  module @rewriters { module @r {} }
}

// -----

// A single matcher passes through unchanged: no `try` is introduced, and the
// symbol name, benefit and forwarded inputs survive verbatim.
// CHECK-LABEL: match.matcher @only root(%arg0: !pdl.operation) {
// CHECK-NEXT:    has_name %arg0, "foo.op"
// CHECK-NEXT:    check_operand_count %arg0 is 1
// CHECK-NEXT:    success @rewriters::@r benefit(3) (%arg0 : !pdl.operation)
// CHECK-NOT:     try
module {
  module @rewriters { module @r {} }
  match.matcher @only root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(3) (%root : !pdl.operation)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Two matchers with nothing in common become two sibling `try`s. The combined
// matcher takes the symbol name of the *first* input matcher.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: match.matcher @disjoint_a root(%arg0: !pdl.operation) {
// CHECK-NEXT:    try {
// CHECK-NEXT:      check_operand_count %arg0 is 1
// CHECK-NEXT:      success @rewriters::@r benefit(1)
// CHECK-NEXT:    }
// CHECK-NEXT:    try {
// CHECK-NEXT:      check_result_count %arg0 is 2
// CHECK-NEXT:      success @rewriters::@r benefit(2)
// CHECK-NEXT:    }
module {
  module @rewriters { module @r {} }
  match.matcher @disjoint_a root(%root : !pdl.operation) {
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @disjoint_b root(%root : !pdl.operation) {
    match.check_result_count %root is 2
    match.success @rewriters::@r benefit(2)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Switch folding — the payoff of the pass.
//
// Three alternatives differing only in the root's name fold into one
// `switch_op_name`, and the predicates they share are hoisted *above* it, so
// they are tested once instead of once per arm.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: match.matcher @sw_a root(%arg0: !pdl.operation) {
// CHECK-NEXT:    check_operand_count %arg0 is 0
// CHECK-NEXT:    check_result_count %arg0 is 0
// CHECK-NEXT:    switch_op_name %arg0
// CHECK-NEXT:    case "foo.op" {
// CHECK-NEXT:      success @rewriters::@r benefit(1)
// CHECK-NEXT:    }
// CHECK-NEXT:    case "bar.op" {
// CHECK-NEXT:      success @rewriters::@r0 benefit(2)
// CHECK-NEXT:    }
// CHECK-NEXT:    case "baz.op" {
// CHECK-NEXT:      success @rewriters::@r1 benefit(3)
// CHECK-NEXT:    }
// CHECK-NOT:     try
module {
  module @rewriters { module @r {} module @r0 {} module @r1 {} }
  match.matcher @sw_a root(%root : !pdl.operation) {
    match.check_operand_count %root is 0
    match.check_result_count %root is 0
    match.has_name %root, "foo.op"
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @sw_b root(%root : !pdl.operation) {
    match.check_operand_count %root is 0
    match.check_result_count %root is 0
    match.has_name %root, "bar.op"
    match.success @rewriters::@r0 benefit(2)
  }
  match.matcher @sw_c root(%root : !pdl.operation) {
    match.check_operand_count %root is 0
    match.check_result_count %root is 0
    match.has_name %root, "baz.op"
    match.success @rewriters::@r1 benefit(3)
  }
}

// -----

// The same folding for types: the navigation chain producing the tested type is
// shared, so it is hoisted above the switch.
// CHECK-LABEL: match.matcher @st_a root(%arg0: !pdl.operation) {
// CHECK-NEXT:    %[[R:.*]] = get_result 0 of %arg0 : <!pdl.value>
// CHECK-NEXT:    %[[V:.*]] = is_not_null %[[R]]
// CHECK-NEXT:    %[[T:.*]] = get_value_type of %[[V]] : !pdl.value : !pdl.type
// CHECK-NEXT:    switch_type %[[T]]
// CHECK-NEXT:    case i32 {
// CHECK-NEXT:      success @rewriters::@r benefit(1)
// CHECK-NEXT:    }
// CHECK-NEXT:    case i64 {
// CHECK-NEXT:      success @rewriters::@r0 benefit(2)
// CHECK-NEXT:    }
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @st_a root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    match.has_type %t, i32
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @st_b root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    match.has_type %t, i64
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

// A switch needs at least two cases: a lone `has_name` alternative stays a
// plain `try` rather than becoming a one-case switch.
// CHECK-LABEL: match.matcher @one_case_a
// CHECK-NOT:     switch_op_name
// CHECK:         try {
// CHECK-NEXT:      has_name %{{.*}}, "foo.op"
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @one_case_a root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @one_case_b root(%root : !pdl.operation) {
    match.check_operand_count %root is 3
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

// Folding requires a *contiguous* run. Here two `has_name` alternatives are
// separated by one that tests something else, so no switch is formed and all
// three stay sibling `try`s.
// CHECK-LABEL: match.matcher @nc_a
// CHECK-NOT:     switch_op_name
// CHECK:         try {
// CHECK-NEXT:      has_name %{{.*}}, "foo.op"
// CHECK:         try {
// CHECK-NEXT:      check_operand_count %{{.*}} is 3
// CHECK:         try {
// CHECK-NEXT:      has_name %{{.*}}, "bar.op"
module {
  module @rewriters { module @r {} module @r0 {} module @r1 {} }
  match.matcher @nc_a root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @nc_b root(%root : !pdl.operation) {
    match.check_operand_count %root is 3
    match.success @rewriters::@r0 benefit(2)
  }
  match.matcher @nc_c root(%root : !pdl.operation) {
    match.has_name %root, "bar.op"
    match.success @rewriters::@r1 benefit(3)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Cost ordering: the predicate shared by three of the four alternatives is
// tested first, in an outer `try`, with those three nested inside it. Naive
// insertion order would have tested `check_operand_count is 7` first.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: match.matcher @cost_a root(%arg0: !pdl.operation) {
// CHECK-NEXT:    try {
// CHECK-NEXT:      check_result_count %arg0 is 1
// CHECK-NEXT:      try {
// CHECK-NEXT:        check_operand_count %arg0 is 7
// CHECK:           try {
// CHECK-NEXT:        check_operand_count %arg0 is 8
// CHECK:           try {
// CHECK-NEXT:        check_operand_count %arg0 is 9
// CHECK:         try {
// CHECK-NEXT:      check_operand_count %arg0 is 10
module {
  module @rewriters { module @r {} module @r0 {} module @r1 {} module @r2 {} }
  match.matcher @cost_a root(%root : !pdl.operation) {
    match.check_operand_count %root is 7
    match.check_result_count %root is 1
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @cost_b root(%root : !pdl.operation) {
    match.check_result_count %root is 1
    match.check_operand_count %root is 8
    match.success @rewriters::@r0 benefit(2)
  }
  match.matcher @cost_c root(%root : !pdl.operation) {
    match.check_result_count %root is 1
    match.check_operand_count %root is 9
    match.success @rewriters::@r1 benefit(3)
  }
  match.matcher @cost_d root(%root : !pdl.operation) {
    match.check_operand_count %root is 10
    match.success @rewriters::@r2 benefit(4)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Dead pure navigation ops are erased. Note what survives: `is_not_null` is a
// *test* (it can fail and transfer control), so neither it nor the
// `get_operand` feeding it is dead, even though nothing consumes the unwrapped
// value. Only the genuinely unused `get_value_type` goes.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: match.matcher @dead_nav root(%arg0: !pdl.operation) {
// CHECK-NEXT:    has_name %arg0, "foo.op"
// CHECK-NEXT:    %[[O:.*]] = get_operand 0 of %arg0 : <!pdl.value>
// CHECK-NEXT:    %{{.*}} = is_not_null %[[O]]
// CHECK-NEXT:    success @rewriters::@r benefit(1)
// CHECK-NOT:     get_value_type
module {
  module @rewriters { module @r {} }
  match.matcher @dead_nav root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %o = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %o : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    match.success @rewriters::@r benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Navigation sinking: a chain used by only one alternative moves *into* that
// alternative's `try` body, so the other alternative does not pay for it.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: match.matcher @sink_a root(%arg0: !pdl.operation) {
// CHECK-NEXT:    has_name %arg0, "foo.op"
// CHECK-NEXT:    try {
// CHECK-NEXT:      %[[A:.*]] = get_attribute "attr" of %arg0 : <!pdl.attribute>
// CHECK-NEXT:      %[[AV:.*]] = is_not_null %[[A]]
// CHECK-NEXT:      has_attr_value %[[AV]] is 10 : i64
// CHECK:         try {
// CHECK-NEXT:      check_operand_count %arg0 is 4
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @sink_a root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>
    %av = match.is_not_null %a : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %av is 10 : i64
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @sink_b root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.check_operand_count %root is 4
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

// The mirror case: a chain used by *both* alternatives stays in the shared
// prefix and is navigated once.
// CHECK-LABEL: match.matcher @keep_a root(%arg0: !pdl.operation) {
// CHECK-NEXT:    has_name %arg0, "foo.op"
// CHECK-NEXT:    %[[A:.*]] = get_attribute "attr" of %arg0 : <!pdl.attribute>
// CHECK-NEXT:    %[[AV:.*]] = is_not_null %[[A]]
// CHECK-NEXT:    try {
// CHECK-NEXT:      has_attr_value %[[AV]] is 10 : i64
// CHECK:         try {
// CHECK-NEXT:      has_attr_value %[[AV]] is 20 : i64
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @keep_a root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>
    %av = match.is_not_null %a : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %av is 10 : i64
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @keep_b root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>
    %av = match.is_not_null %a : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %av is 20 : i64
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

// `get_each` carries control flow (it becomes a loop in `pdl_interp`) and is
// excluded from sinking; here its whole chain belongs to one alternative, so it
// lives in that alternative's body.
// CHECK-LABEL: match.matcher @each_a
// CHECK:         try {
// CHECK:           %[[E:.*]] = get_each %{{.*}} : !pdl.range<value> -> !pdl.value
// CHECK-NEXT:      %[[T:.*]] = get_value_type of %[[E]]
// CHECK-NEXT:      has_type %[[T]], i32
module {
  module @rewriters { module @r {} module @r0 {} }
  match.matcher @each_a root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    %e = match.get_each %vs : !pdl.range<value> -> !pdl.value
    %t = match.get_value_type of %e : !pdl.value : !pdl.type
    match.has_type %t, i32
    match.success @rewriters::@r benefit(1)
  }
  match.matcher @each_b root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.check_operand_count %root is 5
    match.success @rewriters::@r0 benefit(2)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Constant pooling
//===----------------------------------------------------------------------===//

// The same literal in two input matchers collapses to a single pool op: the
// pool key is `(op name, attributes, operands)` and a constant has no operands.
// CHECK-LABEL: match.matcher @shared
module {
  module @rewriters {
    module @r {}
  }

  // CHECK: has_name %{{.*}}, "foo.op"
  // CHECK: %[[ATTR:.*]] = constant_attribute 10 : i64
  // CHECK: try {
  // CHECK: apply_native_constraint "c1"(%[[ATTR]] : !pdl.attribute)
  // CHECK: try {
  // CHECK: apply_native_constraint "c2"(%[[ATTR]] : !pdl.attribute)
  // CHECK-NOT: constant_attribute

  match.matcher @shared root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %attr = match.constant_attribute 10 : i64
    match.apply_native_constraint "c1"(%attr : !pdl.attribute)
    match.success @rewriters::@r benefit(1)
  }

  match.matcher @shared_other root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %attr = match.constant_attribute 10 : i64
    match.apply_native_constraint "c2"(%attr : !pdl.attribute)
    match.success @rewriters::@r benefit(1)
  }
}

// -----

// A literal used by only one alternative lives inside that alternative's `try`
// body, not in the shared prefix.
// CHECK-LABEL: match.matcher @sunk
// CHECK-NOT: constant_attribute
// CHECK: try {
// CHECK: constant_attribute 10 : i64
module {
  module @rewriters {
    module @r {}
  }

  match.matcher @sunk root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %attr = match.constant_attribute 10 : i64
    match.apply_native_constraint "c1"(%attr : !pdl.attribute)
    match.success @rewriters::@r benefit(1)
  }

  match.matcher @sunk_other root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.apply_native_constraint "c2"(%root : !pdl.operation)
    match.success @rewriters::@r benefit(1)
  }
}

// -----

// A dead literal is erased along with the dead navigation ops.
// CHECK-LABEL: match.matcher @dead
// CHECK-NOT: constant_
module {
  module @rewriters {
    module @r {}
  }

  match.matcher @dead root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    %attr = match.constant_attribute 10 : i64
    %type = match.constant_type i32
    match.success @rewriters::@r benefit(1)
  }
}
