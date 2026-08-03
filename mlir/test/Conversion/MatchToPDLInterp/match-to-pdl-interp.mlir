// RUN: mlir-opt %s -split-input-file -convert-match-to-pdl-interp | FileCheck %s

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
