// RUN: mlir-opt %s -split-input-file -match-combine-matchers | FileCheck %s

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
