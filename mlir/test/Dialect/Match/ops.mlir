// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Round-trip of every `match` op. Two printing behaviours are asserted
// explicitly because they are easy to break and nothing else covers them:
//
//  * ops inside a `match.matcher` / `match.try` / switch case body print
//    *without* the `match.` prefix (those regions set a default dialect), so
//    the CHECK lines below assert the elided form;
//  * `!match.optional<T>` and `!pdl.range<T>` print abbreviated when they
//    appear in a type position owned by the dialect (`: <!pdl.value>`,
//    `: <operation>`).
//
// Piping through `mlir-opt` twice also catches a printer that is not a fixed
// point.

module {
  module @rewriters {
    module @foo {}
    module @bar {}
  }

  //===--------------------------------------------------------------------===//
  // match.matcher
  //===--------------------------------------------------------------------===//

  // A matcher symbol name is optional.
  // CHECK-LABEL: match.matcher @named root(%arg0: !pdl.operation) {
  // CHECK-NEXT:    success @rewriters::@foo benefit(1)
  match.matcher @named root(%root : !pdl.operation) {
    match.success @rewriters::@foo benefit(1)
  }

  // CHECK-LABEL: match.matcher root(%arg0: !pdl.operation) {
  // CHECK-NEXT:    success @rewriters::@foo benefit(0)
  match.matcher root(%root : !pdl.operation) {
    match.success @rewriters::@foo benefit(0)
  }

  //===--------------------------------------------------------------------===//
  // match.try — plain, nested, and a sibling chain
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @tries
  match.matcher @tries root(%root : !pdl.operation) {
    // CHECK:      try {
    // CHECK-NEXT:   has_name %arg0, "foo.op"
    // CHECK-NEXT:   success @rewriters::@foo benefit(1)
    // CHECK-NEXT: }
    match.try {
      match.has_name %root, "foo.op"
      match.success @rewriters::@foo benefit(1)
    }
    // A nested try is a nested failure scope.
    // CHECK:      try {
    // CHECK-NEXT:   try {
    // CHECK-NEXT:     has_name %arg0, "bar.op"
    // CHECK-NEXT:     success @rewriters::@bar benefit(2)
    // CHECK-NEXT:   }
    // CHECK-NEXT:   success @rewriters::@foo benefit(3)
    // CHECK-NEXT: }
    match.try {
      match.try {
        match.has_name %root, "bar.op"
        match.success @rewriters::@bar benefit(2)
      }
      match.success @rewriters::@foo benefit(3)
    }
    // CHECK: success @rewriters::@foo benefit(4)
    match.success @rewriters::@foo benefit(4)
  }

  //===--------------------------------------------------------------------===//
  // match.switch_op_name — custom parser/printer; `caseNames` is elided in
  // favour of the `case` keywords.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @switch_names
  match.matcher @switch_names root(%root : !pdl.operation) {
    // A single case is legal.
    // CHECK:      switch_op_name %arg0
    // CHECK-NEXT: case "foo.op" {
    // CHECK-NEXT:   success @rewriters::@foo benefit(1)
    // CHECK-NEXT: }
    match.switch_op_name %root
    case "foo.op" {
      match.success @rewriters::@foo benefit(1)
    }

    // Three cases, one of which contains a nested switch.
    // CHECK:      switch_op_name %arg0
    // CHECK-NEXT: case "a.op" {
    // CHECK-NEXT:   success @rewriters::@foo benefit(1)
    // CHECK-NEXT: }
    // CHECK-NEXT: case "b.op" {
    // CHECK-NEXT:   switch_op_name %arg0
    // CHECK-NEXT:   case "c.op" {
    // CHECK-NEXT:     success @rewriters::@bar benefit(2)
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    // CHECK-NEXT: case "d.op" {
    // CHECK-NEXT:   success @rewriters::@foo benefit(3)
    // CHECK-NEXT: }
    match.switch_op_name %root
    case "a.op" {
      match.success @rewriters::@foo benefit(1)
    }
    case "b.op" {
      match.switch_op_name %root
      case "c.op" {
        match.success @rewriters::@bar benefit(2)
      }
    }
    case "d.op" {
      match.success @rewriters::@foo benefit(3)
    }
    match.success @rewriters::@foo benefit(0)
  }

  //===--------------------------------------------------------------------===//
  // match.switch_type
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @switch_types
  match.matcher @switch_types root(%root : !pdl.operation) {
    %r0 = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v0 = match.is_not_null %r0 : !match.optional<!pdl.value> -> !pdl.value
    // CHECK: %[[T:.*]] = get_value_type of %{{.*}} : !pdl.value : !pdl.type
    %t0 = match.get_value_type of %v0 : !pdl.value : !pdl.type
    // CHECK:      switch_type %[[T]]
    // CHECK-NEXT: case i32 {
    // CHECK-NEXT:   success @rewriters::@foo benefit(1)
    // CHECK-NEXT: }
    // CHECK-NEXT: case i64 {
    // CHECK-NEXT:   success @rewriters::@bar benefit(2)
    // CHECK-NEXT: }
    match.switch_type %t0
    case i32 {
      match.success @rewriters::@foo benefit(1)
    }
    case i64 {
      match.success @rewriters::@bar benefit(2)
    }
    match.success @rewriters::@foo benefit(0)
  }

  //===--------------------------------------------------------------------===//
  // Constant ops — a literal materialized as a value in its own right.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @constants
  match.matcher @constants root(%root : !pdl.operation) {
    // CHECK: %{{.*}} = constant_attribute 10 : i64
    %attr = match.constant_attribute 10 : i64
    // CHECK: %{{.*}} = constant_attribute "hello"
    %str = match.constant_attribute "hello"
    // CHECK: %{{.*}} = constant_attribute unit
    %unit = match.constant_attribute unit
    // CHECK: %{{.*}} = constant_attribute i32
    %typeAttr = match.constant_attribute i32
    // CHECK: %{{.*}} = constant_attribute dense<[1, 2]> : tensor<2xi32>
    %dense = match.constant_attribute dense<[1, 2]> : tensor<2xi32>
    // CHECK: %{{.*}} = constant_type i32
    %type = match.constant_type i32
    // CHECK: %{{.*}} = constant_types [i32, i64]
    %types = match.constant_types [i32, i64]
    // An empty type list is legal.
    // CHECK: %{{.*}} = constant_types []
    %empty = match.constant_types []

    match.success @rewriters::@foo benefit(1)
  }

  // A constant feeding a native constraint — the reason these ops exist.
  // CHECK-LABEL: match.matcher @constant_use
  match.matcher @constant_use root(%root : !pdl.operation) {
    // CHECK: %[[ATTR:.*]] = constant_attribute 10 : i64
    %attr = match.constant_attribute 10 : i64
    // CHECK: %[[TYPES:.*]] = constant_types [i32]
    %types = match.constant_types [i32]
    // CHECK: apply_native_constraint "c"(%[[ATTR]], %[[TYPES]] : !pdl.attribute, !pdl.range<type>)
    match.apply_native_constraint "c"(%attr, %types : !pdl.attribute, !pdl.range<type>)
    match.success @rewriters::@foo benefit(1)
  }

  //===--------------------------------------------------------------------===//
  // Nullable navigation — every op, including the optional-index forms.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @nav_nullable
  match.matcher @nav_nullable root(%root : !pdl.operation) {
    // CHECK: %[[O0:.*]] = get_operand 0 of %arg0 : <!pdl.value>
    %o0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    // `get_operands` / `get_results` print the index only when present.
    // CHECK: %[[OS:.*]] = get_operands of %arg0 : <!pdl.range<value>>
    %os = match.get_operands of %root : !match.optional<!pdl.range<value>>
    // CHECK: %{{.*}} = get_operands 1 of %arg0 : <!pdl.range<value>>
    %os1 = match.get_operands 1 of %root : !match.optional<!pdl.range<value>>
    // CHECK: %{{.*}} = get_result 0 of %arg0 : <!pdl.value>
    %r0 = match.get_result 0 of %root : !match.optional<!pdl.value>
    // CHECK: %{{.*}} = get_results of %arg0 : <!pdl.range<value>>
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    // CHECK: %{{.*}} = get_results 1 of %arg0 : <!pdl.range<value>>
    %rs1 = match.get_results 1 of %root : !match.optional<!pdl.range<value>>
    // CHECK: %{{.*}} = get_attribute "attr" of %arg0 : <!pdl.attribute>
    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>

    // `get_defining_op` on a value and on a value range.
    // CHECK: %[[V0:.*]] = is_not_null %[[O0]] : <!pdl.value> -> !pdl.value
    %v0 = match.is_not_null %o0 : !match.optional<!pdl.value> -> !pdl.value
    // CHECK: %{{.*}} = get_defining_op of %[[V0]] : !pdl.value -> <!pdl.operation>
    %d = match.get_defining_op of %v0 : !pdl.value -> !match.optional<!pdl.operation>
    // CHECK: %[[VS:.*]] = is_not_null %[[OS]] : <!pdl.range<value>> -> !pdl.range<value>
    %vs = match.is_not_null %os : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    // CHECK: %{{.*}} = get_defining_op of %[[VS]] : !pdl.range<value> -> <!pdl.operation>
    %dr = match.get_defining_op of %vs : !pdl.range<value> -> !match.optional<!pdl.operation>
    match.success @rewriters::@foo benefit(1)
  }

  //===--------------------------------------------------------------------===//
  // Non-nullable navigation, `extract` and `get_each`.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @nav_plain
  match.matcher @nav_plain root(%root : !pdl.operation) {
    %o0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    // CHECK: %[[V0:.*]] = is_not_null
    %v0 = match.is_not_null %o0 : !match.optional<!pdl.value> -> !pdl.value
    %os = match.get_operands of %root : !match.optional<!pdl.range<value>>
    // CHECK: %[[VS:.*]] = is_not_null
    %vs = match.is_not_null %os : !match.optional<!pdl.range<value>> -> !pdl.range<value>

    // `get_value_type` has a scalar and a range form.
    // CHECK: %{{.*}} = get_value_type of %[[V0]] : !pdl.value : !pdl.type
    %t0 = match.get_value_type of %v0 : !pdl.value : !pdl.type
    // CHECK: %{{.*}} = get_value_type of %[[VS]] : !pdl.range<value> : !pdl.range<type>
    %ts = match.get_value_type of %vs : !pdl.range<value> : !pdl.range<type>

    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>
    // CHECK: %[[AV:.*]] = is_not_null
    %av = match.is_not_null %a : !match.optional<!pdl.attribute> -> !pdl.attribute
    // CHECK: %{{.*}} = get_attribute_type of %[[AV]] : !pdl.type
    %at = match.get_attribute_type of %av : !pdl.type
    // A range of operations prints abbreviated as `<operation>`.
    // CHECK: %[[USERS:.*]] = get_users of %[[V0]] : <operation>
    %users = match.get_users of %v0 : !pdl.range<operation>

    // `extract` yields the range's element type.
    // CHECK: %{{.*}} = extract 0 of %[[VS]] : !pdl.value
    %ex = match.extract 0 of %vs : !pdl.value
    // CHECK: %{{.*}} = extract 1 of %[[USERS]] : !pdl.operation
    %exo = match.extract 1 of %users : !pdl.operation
    // CHECK: %{{.*}} = get_each %[[VS]] : !pdl.range<value> -> !pdl.value
    %each = match.get_each %vs : !pdl.range<value> -> !pdl.value
    // CHECK: %{{.*}} = get_each %[[USERS]] : !pdl.range<operation> -> !pdl.operation
    %eachop = match.get_each %users : !pdl.range<operation> -> !pdl.operation
    match.success @rewriters::@foo benefit(1)
  }

  //===--------------------------------------------------------------------===//
  // Test ops, including the `at_least` forms of both count checks.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @tests
  match.matcher @tests root(%root : !pdl.operation) {
    // CHECK: has_name %arg0, "foo.op"
    match.has_name %root, "foo.op"
    // CHECK: check_operand_count %arg0 is 2
    match.check_operand_count %root is 2
    // CHECK: check_operand_count %arg0 is at_least 2
    match.check_operand_count %root is at_least 2
    // CHECK: check_result_count %arg0 is 1
    match.check_result_count %root is 1
    // CHECK: check_result_count %arg0 is at_least 1
    match.check_result_count %root is at_least 1

    %o0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    // CHECK: %[[V0:.*]] = is_not_null
    %v0 = match.is_not_null %o0 : !match.optional<!pdl.value> -> !pdl.value
    %o1 = match.get_operand 1 of %root : !match.optional<!pdl.value>
    // CHECK: %[[V1:.*]] = is_not_null
    %v1 = match.is_not_null %o1 : !match.optional<!pdl.value> -> !pdl.value
    // CHECK: equal %[[V0]], %[[V1]] : !pdl.value
    match.equal %v0, %v1 : !pdl.value

    // CHECK: %[[T0:.*]] = get_value_type of %[[V0]]
    %t0 = match.get_value_type of %v0 : !pdl.value : !pdl.type
    // CHECK: has_type %[[T0]], i32
    match.has_type %t0, i32
    %os = match.get_operands of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %os : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    // CHECK: %[[TS:.*]] = get_value_type of %{{.*}} : !pdl.range<value> : !pdl.range<type>
    %ts = match.get_value_type of %vs : !pdl.range<value> : !pdl.range<type>
    // CHECK: has_types %[[TS]], [i32, i64]
    match.has_types %ts, [i32, i64]

    %a = match.get_attribute "attr" of %root : !match.optional<!pdl.attribute>
    // CHECK: %[[AV:.*]] = is_not_null
    %av = match.is_not_null %a : !match.optional<!pdl.attribute> -> !pdl.attribute
    // CHECK: has_attr_value %[[AV]] is 10 : i64
    match.has_attr_value %av is 10 : i64
    match.success @rewriters::@foo benefit(1)
  }

  //===--------------------------------------------------------------------===//
  // Native constraints and rewrites: 0 / 1 / N results, negation, and the
  // args-only and results-only forms of `apply_native_rewrite`.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @natives
  match.matcher @natives root(%root : !pdl.operation) {
    // CHECK: apply_native_constraint "c0"(%arg0 : !pdl.operation)
    match.apply_native_constraint "c0"(%root : !pdl.operation)
    // CHECK: %{{.*}} = apply_native_constraint "c1"(%arg0 : !pdl.operation) : !pdl.attribute
    %c1 = match.apply_native_constraint "c1"(%root : !pdl.operation) : !pdl.attribute
    // CHECK: %{{.*}}:2 = apply_native_constraint "c2"(%arg0 : !pdl.operation) : !pdl.attribute, !pdl.type
    %c2:2 = match.apply_native_constraint "c2"(%root : !pdl.operation) : !pdl.attribute, !pdl.type
    // CHECK: apply_native_constraint "c3"(%arg0 : !pdl.operation) {isNegated = true}
    match.apply_native_constraint "c3"(%root : !pdl.operation) {isNegated = true}

    // CHECK: %{{.*}} = apply_native_rewrite "r1"(%arg0 : !pdl.operation) : !pdl.attribute
    %r1 = match.apply_native_rewrite "r1"(%root : !pdl.operation) : !pdl.attribute
    // Results with no arguments.
    // CHECK: %{{.*}} = apply_native_rewrite "r2" : !pdl.attribute
    %r2 = match.apply_native_rewrite "r2" : !pdl.attribute
    // Arguments with no results.
    // CHECK: apply_native_rewrite "r3"(%arg0 : !pdl.operation)
    match.apply_native_rewrite "r3"(%root : !pdl.operation)
    match.success @rewriters::@foo benefit(1)
  }

  //===--------------------------------------------------------------------===//
  // match.success with forwarded inputs.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: match.matcher @successes
  match.matcher @successes root(%root : !pdl.operation) {
    %o0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    // CHECK: %[[V0:.*]] = is_not_null
    %v0 = match.is_not_null %o0 : !match.optional<!pdl.value> -> !pdl.value
    // CHECK: success @rewriters::@foo benefit(3) (%arg0, %[[V0]] : !pdl.operation, !pdl.value)
    match.success @rewriters::@foo benefit(3) (%root, %v0 : !pdl.operation, !pdl.value)
  }

  //===--------------------------------------------------------------------===//
  // !match.optional over every PDL type. Only four of the six are produced by
  // a navigation op, so a function signature round-trips the type syntax.
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: func private @optional_types
  // CHECK-SAME:    !match.optional<!pdl.attribute>
  // CHECK-SAME:    !match.optional<!pdl.operation>
  // CHECK-SAME:    !match.optional<!pdl.type>
  // CHECK-SAME:    !match.optional<!pdl.value>
  // CHECK-SAME:    !match.optional<!pdl.range<value>>
  // CHECK-SAME:    !match.optional<!pdl.range<type>>
  func.func private @optional_types(
    !match.optional<!pdl.attribute>,
    !match.optional<!pdl.operation>,
    !match.optional<!pdl.type>,
    !match.optional<!pdl.value>,
    !match.optional<!pdl.range<value>>,
    !match.optional<!pdl.range<type>>)
}
