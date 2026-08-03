// RUN: mlir-match-to-cpp %s | FileCheck %s

// `switch_type` emission. `switch.mlir` and `switch-concrete.mlir` cover
// `switch_op_name`; this is the type counterpart, which lowers to an
// if / else-if chain over the reconstructed type literals rather than to a
// dyn_cast on the operation class.

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

// The switched-on type is bound to a local so the cases can compare against it
// without recomputing it.
// CHECK: ::mlir::Type [[V:v[0-9]+]] = {{.*}}.getType();
// CHECK: ::mlir::Type [[TY:ty[0-9]+]] = [[V]];
// CHECK: if ([[TY]] == ::mlir::IntegerType::get(op->getContext(), 32)) {
// CHECK: rewrite_r(rewriter)
// A second case becomes an `else if`, not a fresh `if`: the cases are mutually
// exclusive and only one may run.
// CHECK: else if ([[TY]] == ::mlir::IntegerType::get(op->getContext(), 64)) {
// CHECK: rewrite_r(rewriter)
// Falling off the end of the switch is a match failure.
// CHECK: return ::mlir::failure();
