// RUN: mlir-opt %s --test-match-to-cpp --allow-unregistered-dialect | FileCheck %s

// End-to-end check of the match-to-cpp flow: the matchers in
// test/lib/Target/MatchToCpp/TestMatchToCppPatterns.mlir are translated to C++
// at build time and applied here by `--test-match-to-cpp`. This validates the
// generated patterns actually *match and rewrite*, not just that the emitted
// text looks right (covered by the sibling mlir-translate tests).

// The `rename` matcher fires on `test.original` and rebuilds it as
// `test.renamed`, forwarding operands and result types.
// CHECK-LABEL: func @rename
// CHECK-SAME:    (%[[ARG:.*]]: i32)
// CHECK:         %[[R:.*]] = "test.renamed"(%[[ARG]]) : (i32) -> i32
// CHECK-NOT:     test.original
// CHECK:         return %[[R]]
func.func @rename(%arg0: i32) -> i32 {
  %0 = "test.original"(%arg0) : (i32) -> i32
  return %0 : i32
}

// The `swap` matcher navigates to both operands and rebuilds the op as
// `test.swapped` with the operands reversed.
// CHECK-LABEL: func @swap
// CHECK-SAME:    (%[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK:         %[[R:.*]] = "test.swapped"(%[[B]], %[[A]]) : (i32, i32) -> i32
// CHECK-NOT:     test.swap"
// CHECK:         return %[[R]]
func.func @swap(%a: i32, %b: i32) -> i32 {
  %0 = "test.swap"(%a, %b) : (i32, i32) -> i32
  return %0 : i32
}

// An op that matches no pattern is left untouched.
// CHECK-LABEL: func @untouched
// CHECK:         "test.keep"
func.func @untouched(%arg0: i32) -> i32 {
  %0 = "test.keep"(%arg0) : (i32) -> i32
  return %0 : i32
}
