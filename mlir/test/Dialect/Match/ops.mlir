// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Round-trip of `match` ops. Ops inside a `match.matcher` / `match.try` body
// print without the `match.` prefix (the default dialect of those regions), so
// the CHECK lines assert the elided form.

module {
  module @rewriters {
    module @foo {}
  }

  // Constant ops: a literal materialized as a value in its own right.
  // CHECK: match.matcher @constants root(
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
    // CHECK: %{{.*}} = constant_types []
    %empty = match.constant_types []

    match.success @rewriters::@foo benefit(1)
  }

  // A constant feeding a native constraint — the reason these ops exist.
  // CHECK: match.matcher @constant_use root(
  match.matcher @constant_use root(%root : !pdl.operation) {
    // CHECK: %[[ATTR:.*]] = constant_attribute 10 : i64
    %attr = match.constant_attribute 10 : i64
    // CHECK: %[[TYPES:.*]] = constant_types [i32]
    %types = match.constant_types [i32]
    // CHECK: apply_native_constraint "c"(%[[ATTR]], %[[TYPES]] : !pdl.attribute, !pdl.range<type>)
    match.apply_native_constraint "c"(%attr, %types : !pdl.attribute, !pdl.range<type>)
    match.success @rewriters::@foo benefit(1)
  }
}
