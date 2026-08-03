// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Every verifier in the `match` dialect, one per split.
//
// A few checks are only reachable through the generic operation syntax: the
// custom assembly formats cannot express a region/case-count mismatch, a
// two-argument matcher body, or a zero-argument native constraint. Those are
// still reachable from a C++ builder, so they are tested in generic form and
// marked below.

// expected-error @below {{expected inner type of match.optional to be a PDL type}}
func.func private @optional_inner_type(!match.optional<i32>)

// -----

//===----------------------------------------------------------------------===//
// match.matcher
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @foo {} }
  // expected-error @below {{expected body block argument to be of type !pdl.operation}}
  match.matcher @root_type root(%root : i32) {
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

// Generic syntax: the custom format always parses exactly one block argument.
module {
  module @rewriters { module @foo {} }
  // expected-error @below {{expected body block to have exactly one argument}}
  "match.matcher"() ({
  ^bb0(%a: !pdl.operation, %b: !pdl.operation):
    "match.success"() {rewriter = @rewriters::@foo, benefit = 1 : i16} : () -> ()
  }) {sym_name = "two_args"} : () -> ()
}

// -----

// A matcher that can never report a match is dead.
module {
  // expected-error @below {{expected matcher to contain at least one `match.success`}}
  match.matcher @no_success root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
  }
}

// -----

//===----------------------------------------------------------------------===//
// match.try
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @foo {} }
  match.matcher @try_no_success root(%root : !pdl.operation) {
    // expected-error @below {{region contains no `match.success`}}
    match.try {
      match.has_name %root, "foo.op"
    }
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// match.switch_op_name / match.switch_type — one region per case
//
// Generic syntax: the custom format derives the region count from the `case`
// keywords, so the two can never disagree when parsed from the pretty form.
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @foo {} }
  match.matcher @switch_name_count root(%root : !pdl.operation) {
    // expected-error @below {{expected one region per case name (2 names vs 1 regions)}}
    "match.switch_op_name"(%root) ({
      "match.success"() {rewriter = @rewriters::@foo, benefit = 1 : i16} : () -> ()
    }) {caseNames = ["a.op", "b.op"]} : (!pdl.operation) -> ()
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @switch_type_count root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.value
    %t = match.get_value_type of %v : !pdl.value : !pdl.type
    // expected-error @below {{expected one region per case type (2 types vs 1 regions)}}
    "match.switch_type"(%t) ({
      "match.success"() {rewriter = @rewriters::@foo, benefit = 1 : i16} : () -> ()
    }) {caseTypes = [i32, i64]} : (!pdl.type) -> ()
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Navigation verifiers
//===----------------------------------------------------------------------===//

// A scalar value cannot produce a range of types.
module {
  module @rewriters { module @foo {} }
  match.matcher @value_type_range root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.value
    // expected-error @below {{expected result to be a range iff the value is a range}}
    %t = match.get_value_type of %v : !pdl.value : !pdl.range<type>
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @each_not_range root(%root : !pdl.operation) {
    // expected-error @below {{expected input to be a !pdl.range type}}
    %e = match.get_each %root : !pdl.operation -> !pdl.operation
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @each_elem_type root(%root : !pdl.operation) {
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    // expected-error @below {{expected result type to match the element type of the input range}}
    %e = match.get_each %vs : !pdl.range<value> -> !pdl.type
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @unwrap_type root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    // expected-error @below {{expected unwrapped result type '!pdl.type' to match inner type of optional '!pdl.value'}}
    %v = match.is_not_null %r : !match.optional<!pdl.value> -> !pdl.type
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Native constraint / rewrite
//===----------------------------------------------------------------------===//

// A constraint may produce attributes, types and values, but not operations:
// there is no position to bind a newly produced operation to.
module {
  module @rewriters { module @foo {} }
  match.matcher @constraint_returns_op root(%root : !pdl.operation) {
    // expected-error @below {{returning an operation from a constraint is not supported}}
    %c = match.apply_native_constraint "c"(%root : !pdl.operation) : !pdl.operation
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

// Generic syntax: `"c"( : )` does not parse in the custom format.
module {
  module @rewriters { module @foo {} }
  match.matcher @constraint_no_args root(%root : !pdl.operation) {
    // expected-error @below {{expected at least one argument}}
    "match.apply_native_constraint"() {name = "c"} : () -> ()
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @rewrite_no_args_or_results root(%root : !pdl.operation) {
    // expected-error @below {{expected at least one argument or result}}
    match.apply_native_rewrite "r"
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// match.success
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @foo {} }
  // expected-error @below {{`match.success` must be enclosed by a `match.matcher`}}
  match.success @rewriters::@foo benefit(1)
}

// -----

module {
  module @rewriters {}
  match.matcher @unknown_rewriter root(%root : !pdl.operation) {
    // expected-error @below {{references an unknown rewriter symbol: @rewriters::@nope}}
    match.success @rewriters::@nope benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Non-negative integer attribute constraints
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @foo {} }
  match.matcher @negative_benefit root(%root : !pdl.operation) {
    // expected-error @below {{'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative}}
    match.success @rewriters::@foo benefit(-1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @negative_count root(%root : !pdl.operation) {
    // expected-error @below {{'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
    match.check_operand_count %root is -1
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

module {
  module @rewriters { module @foo {} }
  match.matcher @negative_index root(%root : !pdl.operation) {
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    // expected-error @below {{'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
    %e = match.extract -1 of %vs : !pdl.value
    match.success @rewriters::@foo benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// ODS attribute constraints on the constant ops
//===----------------------------------------------------------------------===//

// `constant_type` takes a type attribute. The custom assembly format cannot
// express a non-`TypeAttr` value, so use the generic form.
// expected-error @below {{failed to satisfy constraint: any type attribute}}
%0 = "match.constant_type"() {value = 10 : i64} : () -> !pdl.type

// -----

// `constant_types` takes an array of type attributes.
// expected-error @below {{failed to satisfy constraint: type array attribute}}
%0 = match.constant_types [i32, 10 : i64]
