// RUN: mlir-opt %s --test-match-to-cpp --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @rename
// CHECK-SAME:    (%[[ARG:.*]]: i32)
// CHECK:         %[[R:.*]] = "test.renamed"(%[[ARG]]) : (i32) -> i32
// CHECK-NOT:     test.original
// CHECK:         return %[[R]]
func.func @rename(%arg0: i32) -> i32 {
  %0 = "test.original"(%arg0) : (i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @swap
// CHECK-SAME:    (%[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK:         %[[R:.*]] = "test.swapped"(%[[B]], %[[A]]) : (i32, i32) -> i32
// CHECK-NOT:     test.swap"
// CHECK:         return %[[R]]
func.func @swap(%a: i32, %b: i32) -> i32 {
  %0 = "test.swap"(%a, %b) : (i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @untouched
// CHECK:         "test.keep"
func.func @untouched(%arg0: i32) -> i32 {
  %0 = "test.keep"(%arg0) : (i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @lower_addf
// CHECK-SAME:    (%[[A:.*]]: f32, %[[B:.*]]: f32)
// CHECK:         %[[R:.*]] = "test.lowered_addf"(%[[A]], %[[B]]) : (f32, f32) -> f32
// CHECK-NOT:     arith.addf
// CHECK:         return %[[R]]
func.func @lower_addf(%a: f32, %b: f32) -> f32 {
  %0 = arith.addf %a, %b : f32
  return %0 : f32
}
