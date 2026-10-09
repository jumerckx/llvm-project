// RUN: mlir-match-to-cpp %s | FileCheck %s

module {
  module @rewriters { module @r {} }

  match.matcher @sw_type root(%root : !pdl.operation) {
    %r = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %r : !pdl.value
    %t = match.get_value_type of %v : !pdl.type
    match.switch_type %t
    case i32 { match.success @rewriters::@r benefit(1) }
    case i64 { match.success @rewriters::@r benefit(2) }
  }
}

// CHECK: ::mlir::Type [[V:v[0-9]+]] = {{.*}}.getType();
// CHECK: ::mlir::Type [[TY:ty[0-9]+]] = [[V]];
// CHECK: if ([[TY]] == ::mlir::IntegerType::get(op->getContext(), 32)) {
// CHECK: rewrite_r(rewriter)
// CHECK: else if ([[TY]] == ::mlir::IntegerType::get(op->getContext(), 64)) {
// CHECK: rewrite_r(rewriter)
// CHECK: return ::mlir::failure();
