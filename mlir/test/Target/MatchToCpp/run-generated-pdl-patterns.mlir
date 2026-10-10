// RUN: mlir-opt %s --test-match-to-cpp-pdl --allow-unregistered-dialect | FileCheck %s

// CHECK: module attributes {test.rewrites_begun = 6 : i64, test.rewrites_ended = 6 : i64}

// CHECK-LABEL: func @fold_add
// CHECK-SAME:    (%[[ARG:.*]]: i32)
// CHECK:         %[[R:.*]] = "test.folded"(%[[ARG]]) {value = 5 : i32} : (i32) -> i32
// CHECK:         return %[[R]]
func.func @fold_add(%arg0: i32) -> i32 {
  %0 = "test.fold_add"(%arg0) {value = 4 : i32} : (i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @fold_add_constraint_fails
// CHECK:         "test.fold_add"(%{{.*}}) {value = 3 : i32}
// CHECK:         "test.fold_add"(%{{.*}}) {value = 12 : i32}
func.func @fold_add_constraint_fails(%arg0: i32) -> (i32, i32) {
  %0 = "test.fold_add"(%arg0) {value = 3 : i32} : (i32) -> i32
  %1 = "test.fold_add"(%arg0) {value = 12 : i32} : (i32) -> i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func @to_addi
// CHECK-SAME:    (%[[A:.*]]: i32, %[[B:.*]]: i32)
// CHECK-DAG:     %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : i32
// CHECK-DAG:     %[[C:.*]] = arith.constant 1 : i32
// CHECK:         %[[R:.*]] = "test.pair"(%[[SUM]], %[[C]]) : (i32, i32) -> i32
// CHECK:         return %[[R]]
func.func @to_addi(%a: i32, %b: i32) -> i32 {
  %0 = "test.add"(%a, %b) : (i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @benefit
// CHECK:         %[[SRC:.*]] = "test.src"() : () -> i32
// CHECK:         "test.picked_specific"(%[[SRC]])
// CHECK:         "test.picked_generic"(%{{.*}})
func.func @benefit(%arg0: i32) -> (i32, i32) {
  %src = "test.src"() : () -> i32
  %0 = "test.pick"(%src) : (i32) -> i32
  %1 = "test.pick"(%arg0) : (i32) -> i32
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func @ranges
// CHECK-SAME:    (%[[A:.*]]: i32, %[[B:.*]]: f32)
// CHECK:         %[[R:.*]]:2 = "test.first"(%[[A]], %[[B]]) : (i32, f32) -> (f32, i32)
// CHECK:         return %[[R]]#0, %[[R]]#1
func.func @ranges(%a: i32, %b: f32) -> (f32, i32) {
  %0:2 = "test.split"(%a, %b) : (i32, f32) -> (f32, i32)
  return %0#0, %0#1 : f32, i32
}

// CHECK-LABEL: func @external
// CHECK-NOT:     test.external
// CHECK:         return
func.func @external() {
  "test.external"() : () -> ()
  return
}
