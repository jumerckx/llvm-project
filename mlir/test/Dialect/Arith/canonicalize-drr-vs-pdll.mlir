// Check that the PDLL port of the DRR arith canonicalization patterns
// (test/lib/Dialect/Arith/ArithCanonicalization.pdll) behaves the same as the
// DRR patterns (lib/Dialect/Arith/IR/ArithCanonicalization.td), both when run
// by the PDL bytecode and when compiled to C++ by mlir-match-to-cpp.

// RUN: mlir-opt %s -test-arith-canonicalize-drr | FileCheck %s
// RUN: mlir-opt %s -test-arith-canonicalize-pdll | FileCheck %s
// RUN: mlir-opt %s -test-arith-canonicalize-match-cpp | FileCheck %s
// RUN: mlir-opt %s -test-arith-canonicalize-match-cpp-split | FileCheck %s
// RUN: mlir-opt %s -test-arith-canonicalize-drr -o %t.drr
// RUN: mlir-opt %s -test-arith-canonicalize-pdll -o %t.pdll
// RUN: mlir-opt %s -test-arith-canonicalize-match-cpp -o %t.match-cpp
// RUN: mlir-opt %s -test-arith-canonicalize-match-cpp-split -o %t.match-cpp-split
// RUN: diff %t.drr %t.pdll
// RUN: diff %t.drr %t.match-cpp
// RUN: diff %t.drr %t.match-cpp-split

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @addi_addi_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 5 : i32
//       CHECK:   %[[R:.*]] = arith.addi %[[X]], %[[C]] overflow<nsw> : i32
//       CHECK:   return %[[R]]
func.func @addi_addi_constant(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.addi %x, %c2 overflow<nsw, nuw> : i32
  %1 = arith.addi %0, %c3 overflow<nsw> : i32
  return %1 : i32
}

// CHECK-LABEL: func @addi_addi_constant_no_flags
//       CHECK:   arith.addi %{{.*}}, %{{.*}} : i32
func.func @addi_addi_constant_no_flags(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.addi %x, %c2 overflow<nsw> : i32
  %1 = arith.addi %0, %c3 : i32
  return %1 : i32
}

// CHECK-LABEL: func @addi_subi_constant_rhs
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 1 : i32
//       CHECK:   %[[R:.*]] = arith.addi %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @addi_subi_constant_rhs(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.subi %x, %c2 : i32
  %1 = arith.addi %0, %c3 : i32
  return %1 : i32
}

// CHECK-LABEL: func @addi_subi_constant_lhs
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 5 : i32
//       CHECK:   %[[R:.*]] = arith.subi %[[C]], %[[X]] : i32
//       CHECK:   return %[[R]]
func.func @addi_subi_constant_lhs(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.subi %c2, %x : i32
  %1 = arith.addi %0, %c3 : i32
  return %1 : i32
}

// CHECK-LABEL: func @addi_muli_negative_one_rhs
//  CHECK-SAME:   (%[[X:.*]]: i32, %[[Y:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.subi %[[X]], %[[Y]] : i32
//       CHECK:   return %[[R]]
func.func @addi_muli_negative_one_rhs(%x: i32, %y: i32) -> i32 {
  %cm1 = arith.constant -1 : i32
  %0 = arith.muli %y, %cm1 : i32
  %1 = arith.addi %x, %0 : i32
  return %1 : i32
}

// CHECK-LABEL: func @addi_muli_negative_one_lhs_splat
//  CHECK-SAME:   (%[[X:.*]]: vector<4xi32>, %[[Y:.*]]: vector<4xi32>)
//       CHECK:   %[[R:.*]] = arith.subi %[[Y]], %[[X]] : vector<4xi32>
//       CHECK:   return %[[R]]
func.func @addi_muli_negative_one_lhs_splat(%x: vector<4xi32>, %y: vector<4xi32>) -> vector<4xi32> {
  %cm1 = arith.constant dense<-1> : vector<4xi32>
  %0 = arith.muli %x, %cm1 : vector<4xi32>
  %1 = arith.addi %0, %y : vector<4xi32>
  return %1 : vector<4xi32>
}

// CHECK-LABEL: func @muli_muli_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 6 : i32
//       CHECK:   %[[R:.*]] = arith.muli %[[X]], %[[C]] overflow<nuw> : i32
//       CHECK:   return %[[R]]
func.func @muli_muli_constant(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.muli %x, %c2 overflow<nuw> : i32
  %1 = arith.muli %0, %c3 overflow<nsw, nuw> : i32
  return %1 : i32
}

// Vector constants are not IntegerAttrs, the constant-combining patterns don't
// apply.
// CHECK-LABEL: func @addi_addi_constant_vector
//       CHECK:   arith.addi
//       CHECK:   arith.addi
func.func @addi_addi_constant_vector(%x: vector<4xi32>) -> vector<4xi32> {
  %c2 = arith.constant dense<2> : vector<4xi32>
  %c3 = arith.constant dense<3> : vector<4xi32>
  %0 = arith.addi %x, %c2 : vector<4xi32>
  %1 = arith.addi %0, %c3 : vector<4xi32>
  return %1 : vector<4xi32>
}

//===----------------------------------------------------------------------===//
// AddUIExtendedOp / SubUIExtendedOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @addui_extended_to_addi
//  CHECK-SAME:   (%[[X:.*]]: i32, %[[Y:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.addi %[[X]], %[[Y]] : i32
//       CHECK:   return %[[R]]
func.func @addui_extended_to_addi(%x: i32, %y: i32) -> i32 {
  %sum, %overflow = arith.addui_extended %x, %y : i32, i1
  return %sum : i32
}

// CHECK-LABEL: func @addui_extended_overflow_used
//       CHECK:   arith.addui_extended
func.func @addui_extended_overflow_used(%x: i32, %y: i32) -> (i32, i1) {
  %sum, %overflow = arith.addui_extended %x, %y : i32, i1
  return %sum, %overflow : i32, i1
}

// CHECK-LABEL: func @subui_extended_to_subi
//  CHECK-SAME:   (%[[X:.*]]: i32, %[[Y:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.subi %[[X]], %[[Y]] : i32
//       CHECK:   return %[[R]]
func.func @subui_extended_to_subi(%x: i32, %y: i32) -> i32 {
  %diff, %borrow = arith.subui_extended %x, %y : i32, i1
  return %diff : i32
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @subi_rhs_add_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant -1 : i32
//       CHECK:   %[[R:.*]] = arith.addi %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @subi_rhs_add_constant(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.addi %x, %c2 overflow<nsw> : i32
  %1 = arith.subi %0, %c3 overflow<nsw> : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_lhs_add_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 1 : i32
//       CHECK:   %[[R:.*]] = arith.subi %[[C]], %[[X]] overflow<nsw> : i32
//       CHECK:   return %[[R]]
func.func @subi_lhs_add_constant(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.addi %x, %c2 overflow<nsw> : i32
  %1 = arith.subi %c3, %0 overflow<nsw> : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_rhs_sub_constant_rhs
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 5 : i32
//       CHECK:   %[[R:.*]] = arith.subi %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @subi_rhs_sub_constant_rhs(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.subi %x, %c2 : i32
  %1 = arith.subi %0, %c3 : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_rhs_sub_constant_lhs
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant -1 : i32
//       CHECK:   %[[R:.*]] = arith.subi %[[C]], %[[X]] : i32
//       CHECK:   return %[[R]]
func.func @subi_rhs_sub_constant_lhs(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.subi %c2, %x : i32
  %1 = arith.subi %0, %c3 : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_lhs_sub_constant_rhs
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 5 : i32
//       CHECK:   %[[R:.*]] = arith.subi %[[C]], %[[X]] : i32
//       CHECK:   return %[[R]]
func.func @subi_lhs_sub_constant_rhs(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.subi %x, %c2 : i32
  %1 = arith.subi %c3, %0 : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_lhs_sub_constant_lhs
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 1 : i32
//       CHECK:   %[[R:.*]] = arith.addi %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @subi_lhs_sub_constant_lhs(%x: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %0 = arith.subi %c2, %x : i32
  %1 = arith.subi %c3, %0 : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_subi_lhs_rhs_lhs
//  CHECK-SAME:   (%[[A:.*]]: i32, %[[B:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 0 : i32
//       CHECK:   %[[R:.*]] = arith.subi %[[C]], %[[B]] overflow<nuw> : i32
//       CHECK:   return %[[R]]
func.func @subi_subi_lhs_rhs_lhs(%a: i32, %b: i32) -> i32 {
  %0 = arith.subi %a, %b overflow<nuw> : i32
  %1 = arith.subi %0, %a overflow<nuw> : i32
  return %1 : i32
}

// CHECK-LABEL: func @subi_subi_lhs_rhs_lhs_dynamic
//       CHECK:   arith.subi
//       CHECK:   arith.subi
func.func @subi_subi_lhs_rhs_lhs_dynamic(%a: tensor<?xi32>, %b: tensor<?xi32>) -> tensor<?xi32> {
  %0 = arith.subi %a, %b : tensor<?xi32>
  %1 = arith.subi %0, %a : tensor<?xi32>
  return %1 : tensor<?xi32>
}

//===----------------------------------------------------------------------===//
// MulSIExtendedOp / MulUIExtendedOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @mulsi_extended_to_muli
//  CHECK-SAME:   (%[[X:.*]]: i32, %[[Y:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.muli %[[X]], %[[Y]] : i32
//       CHECK:   return %[[R]]
func.func @mulsi_extended_to_muli(%x: i32, %y: i32) -> i32 {
  %low, %high = arith.mulsi_extended %x, %y : i32
  return %low : i32
}

// CHECK-LABEL: func @mulsi_extended_rhs_one
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:   %[[CMP:.*]] = arith.cmpi slt, %[[X]], %[[C0]] : i32
//       CHECK:   %[[EXT:.*]] = arith.extsi %[[CMP]] : i1 to i32
//       CHECK:   return %[[X]], %[[EXT]]
func.func @mulsi_extended_rhs_one(%x: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : i32
  %low, %high = arith.mulsi_extended %x, %c1 : i32
  return %low, %high : i32, i32
}

// CHECK-LABEL: func @mului_extended_to_muli
//  CHECK-SAME:   (%[[X:.*]]: i32, %[[Y:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.muli %[[X]], %[[Y]] : i32
//       CHECK:   return %[[R]]
func.func @mului_extended_to_muli(%x: i32, %y: i32) -> i32 {
  %low, %high = arith.mului_extended %x, %y : i32
  return %low : i32
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @xori_xori_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 6 : i32
//       CHECK:   %[[R:.*]] = arith.xori %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @xori_xori_constant(%x: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %0 = arith.xori %x, %c3 : i32
  %1 = arith.xori %0, %c5 : i32
  return %1 : i32
}

// CHECK-LABEL: func @xori_not_cmpi
//  CHECK-SAME:   (%[[A:.*]]: i32, %[[B:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.cmpi sge, %[[A]], %[[B]] : i32
//       CHECK:   return %[[R]]
func.func @xori_not_cmpi(%a: i32, %b: i32) -> i1 {
  %true = arith.constant true
  %0 = arith.cmpi slt, %a, %b : i32
  %1 = arith.xori %0, %true : i1
  return %1 : i1
}

// CHECK-LABEL: func @xori_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8, %[[Y:.*]]: i8)
//       CHECK:   %[[XOR:.*]] = arith.xori %[[X]], %[[Y]] : i8
//       CHECK:   %[[R:.*]] = arith.extui %[[XOR]] nneg : i8 to i32
//       CHECK:   return %[[R]]
func.func @xori_of_extui(%x: i8, %y: i8) -> i32 {
  %0 = arith.extui %x nneg : i8 to i32
  %1 = arith.extui %y nneg : i8 to i32
  %2 = arith.xori %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @xori_of_extui_one_nneg
//       CHECK:   %[[XOR:.*]] = arith.xori
//       CHECK:   arith.extui %[[XOR]] : i8 to i32
func.func @xori_of_extui_one_nneg(%x: i8, %y: i8) -> i32 {
  %0 = arith.extui %x nneg : i8 to i32
  %1 = arith.extui %y : i8 to i32
  %2 = arith.xori %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @xori_of_extsi
//  CHECK-SAME:   (%[[X:.*]]: i8, %[[Y:.*]]: i8)
//       CHECK:   %[[XOR:.*]] = arith.xori %[[X]], %[[Y]] : i8
//       CHECK:   %[[R:.*]] = arith.extsi %[[XOR]] : i8 to i32
//       CHECK:   return %[[R]]
func.func @xori_of_extsi(%x: i8, %y: i8) -> i32 {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.extsi %y : i8 to i32
  %2 = arith.xori %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @xori_of_extsi_type_mismatch
//       CHECK:   arith.extsi
//       CHECK:   arith.extsi
//       CHECK:   arith.xori
func.func @xori_of_extsi_type_mismatch(%x: i8, %y: i16) -> i32 {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.extsi %y : i16 to i32
  %2 = arith.xori %0, %1 : i32
  return %2 : i32
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @cmpi_extsi
//  CHECK-SAME:   (%[[A:.*]]: i8, %[[B:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.cmpi eq, %[[A]], %[[B]] : i8
//       CHECK:   return %[[R]]
func.func @cmpi_extsi(%a: i8, %b: i8) -> i1 {
  %0 = arith.extsi %a : i8 to i32
  %1 = arith.extsi %b : i8 to i32
  %2 = arith.cmpi eq, %0, %1 : i32
  return %2 : i1
}

// CHECK-LABEL: func @cmpi_extui
//  CHECK-SAME:   (%[[A:.*]]: i8, %[[B:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.cmpi ne, %[[A]], %[[B]] : i8
//       CHECK:   return %[[R]]
func.func @cmpi_extui(%a: i8, %b: i8) -> i1 {
  %0 = arith.extui %a : i8 to i32
  %1 = arith.extui %b : i8 to i32
  %2 = arith.cmpi ne, %0, %1 : i32
  return %2 : i1
}

// CHECK-LABEL: func @cmpi_extui_not_equality
//       CHECK:   arith.cmpi ult, %{{.*}}, %{{.*}} : i32
func.func @cmpi_extui_not_equality(%a: i8, %b: i8) -> i1 {
  %0 = arith.extui %a : i8 to i32
  %1 = arith.extui %b : i8 to i32
  %2 = arith.cmpi ult, %0, %1 : i32
  return %2 : i1
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @select_not_cond
//  CHECK-SAME:   (%[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.select %[[P]], %[[B]], %[[A]] : i32
//       CHECK:   return %[[R]]
func.func @select_not_cond(%p: i1, %a: i32, %b: i32) -> i32 {
  %true = arith.constant true
  %0 = arith.xori %p, %true : i1
  %1 = arith.select %0, %a, %b : i32
  return %1 : i32
}

// CHECK-LABEL: func @redundant_select_true
//  CHECK-SAME:   (%[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.select %[[P]], %[[A]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @redundant_select_true(%p: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %0 = arith.select %p, %a, %b : i32
  %1 = arith.select %p, %0, %c : i32
  return %1 : i32
}

// CHECK-LABEL: func @redundant_select_false
//  CHECK-SAME:   (%[[P:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.select %[[P]], %[[A]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @redundant_select_false(%p: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %0 = arith.select %p, %b, %c : i32
  %1 = arith.select %p, %a, %0 : i32
  return %1 : i32
}

// CHECK-LABEL: func @redundant_select_different_pred
//       CHECK:   arith.select
//       CHECK:   arith.select
func.func @redundant_select_different_pred(%p: i1, %q: i1, %a: i32, %b: i32, %c: i32) -> i32 {
  %0 = arith.select %q, %a, %b : i32
  %1 = arith.select %p, %0, %c : i32
  return %1 : i32
}

// CHECK-LABEL: func @select_i1_to_not
//  CHECK-SAME:   (%[[P:.*]]: i1)
//       CHECK:   %[[T:.*]] = arith.constant true
//       CHECK:   %[[R:.*]] = arith.xori %[[P]], %[[T]] : i1
//       CHECK:   return %[[R]]
func.func @select_i1_to_not(%p: i1) -> i1 {
  %false = arith.constant false
  %true = arith.constant true
  %0 = arith.select %p, %false, %true : i1
  return %0 : i1
}

// CHECK-LABEL: func @select_cmpi_to_min_max
//  CHECK-SAME:   (%[[A:.*]]: i32, %[[B:.*]]: i32)
//       CHECK:   %[[R0:.*]] = arith.minsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R1:.*]] = arith.minsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R2:.*]] = arith.maxsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R3:.*]] = arith.maxsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R4:.*]] = arith.minui %[[A]], %[[B]] : i32
//       CHECK:   %[[R5:.*]] = arith.minui %[[A]], %[[B]] : i32
//       CHECK:   %[[R6:.*]] = arith.maxui %[[A]], %[[B]] : i32
//       CHECK:   %[[R7:.*]] = arith.maxui %[[A]], %[[B]] : i32
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]]
func.func @select_cmpi_to_min_max(%a: i32, %b: i32) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
  %c0 = arith.cmpi slt, %a, %b : i32
  %s0 = arith.select %c0, %a, %b : i32
  %c1 = arith.cmpi sle, %a, %b : i32
  %s1 = arith.select %c1, %a, %b : i32
  %c2 = arith.cmpi sgt, %a, %b : i32
  %s2 = arith.select %c2, %a, %b : i32
  %c3 = arith.cmpi sge, %a, %b : i32
  %s3 = arith.select %c3, %a, %b : i32
  %c4 = arith.cmpi ult, %a, %b : i32
  %s4 = arith.select %c4, %a, %b : i32
  %c5 = arith.cmpi ule, %a, %b : i32
  %s5 = arith.select %c5, %a, %b : i32
  %c6 = arith.cmpi ugt, %a, %b : i32
  %s6 = arith.select %c6, %a, %b : i32
  %c7 = arith.cmpi uge, %a, %b : i32
  %s7 = arith.select %c7, %a, %b : i32
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 : i32, i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: func @select_cmpi_to_inverse_min_max
//  CHECK-SAME:   (%[[A:.*]]: i32, %[[B:.*]]: i32)
//       CHECK:   %[[R0:.*]] = arith.maxsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R1:.*]] = arith.maxsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R2:.*]] = arith.minsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R3:.*]] = arith.minsi %[[A]], %[[B]] : i32
//       CHECK:   %[[R4:.*]] = arith.maxui %[[A]], %[[B]] : i32
//       CHECK:   %[[R5:.*]] = arith.maxui %[[A]], %[[B]] : i32
//       CHECK:   %[[R6:.*]] = arith.minui %[[A]], %[[B]] : i32
//       CHECK:   %[[R7:.*]] = arith.minui %[[A]], %[[B]] : i32
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[R3]], %[[R4]], %[[R5]], %[[R6]], %[[R7]]
func.func @select_cmpi_to_inverse_min_max(%a: i32, %b: i32) -> (i32, i32, i32, i32, i32, i32, i32, i32) {
  %c0 = arith.cmpi slt, %a, %b : i32
  %s0 = arith.select %c0, %b, %a : i32
  %c1 = arith.cmpi sle, %a, %b : i32
  %s1 = arith.select %c1, %b, %a : i32
  %c2 = arith.cmpi sgt, %a, %b : i32
  %s2 = arith.select %c2, %b, %a : i32
  %c3 = arith.cmpi sge, %a, %b : i32
  %s3 = arith.select %c3, %b, %a : i32
  %c4 = arith.cmpi ult, %a, %b : i32
  %s4 = arith.select %c4, %b, %a : i32
  %c5 = arith.cmpi ule, %a, %b : i32
  %s5 = arith.select %c5, %b, %a : i32
  %c6 = arith.cmpi ugt, %a, %b : i32
  %s6 = arith.select %c6, %b, %a : i32
  %c7 = arith.cmpi uge, %a, %b : i32
  %s7 = arith.select %c7, %b, %a : i32
  return %s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7 : i32, i32, i32, i32, i32, i32, i32, i32
}

// CHECK-LABEL: func @select_cmpi_operand_mismatch
//       CHECK:   arith.select
func.func @select_cmpi_operand_mismatch(%a: i32, %b: i32, %c: i32) -> i32 {
  %cmp = arith.cmpi slt, %a, %b : i32
  %s = arith.select %cmp, %a, %c : i32
  return %s : i32
}

//===----------------------------------------------------------------------===//
// Casts
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @index_cast_of_extsi
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.index_cast %[[X]] : i8 to index
//       CHECK:   return %[[R]]
func.func @index_cast_of_extsi(%x: i8) -> index {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.index_cast %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func @index_castui_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.index_castui %[[X]] nneg : i8 to index
//       CHECK:   return %[[R]]
func.func @index_castui_of_extui(%x: i8) -> index {
  %0 = arith.extui %x nneg : i8 to i32
  %1 = arith.index_castui %0 : i32 to index
  return %1 : index
}

// CHECK-LABEL: func @bitcast_of_bitcast
//  CHECK-SAME:   (%[[X:.*]]: i16)
//       CHECK:   %[[R:.*]] = arith.bitcast %[[X]] : i16 to bf16
//       CHECK:   return %[[R]]
func.func @bitcast_of_bitcast(%x: i16) -> bf16 {
  %0 = arith.bitcast %x : i16 to f16
  %1 = arith.bitcast %0 : f16 to bf16
  return %1 : bf16
}

// CHECK-LABEL: func @extsi_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.extui %[[X]] : i8 to i64
//       CHECK:   return %[[R]]
func.func @extsi_of_extui(%x: i8) -> i64 {
  %0 = arith.extui %x : i8 to i32
  %1 = arith.extsi %0 : i32 to i64
  return %1 : i64
}

//===----------------------------------------------------------------------===//
// AndIOp / OrIOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @andi_andi_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 4 : i32
//       CHECK:   %[[R:.*]] = arith.andi %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @andi_andi_constant(%x: i32) -> i32 {
  %c6 = arith.constant 6 : i32
  %c12 = arith.constant 12 : i32
  %0 = arith.andi %x, %c6 : i32
  %1 = arith.andi %0, %c12 : i32
  return %1 : i32
}

// CHECK-LABEL: func @andi_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8, %[[Y:.*]]: i8)
//       CHECK:   %[[AND:.*]] = arith.andi %[[X]], %[[Y]] : i8
//       CHECK:   %[[R:.*]] = arith.extui %[[AND]] nneg : i8 to i32
//       CHECK:   return %[[R]]
func.func @andi_of_extui(%x: i8, %y: i8) -> i32 {
  %0 = arith.extui %x : i8 to i32
  %1 = arith.extui %y nneg : i8 to i32
  %2 = arith.andi %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @andi_of_extsi
//  CHECK-SAME:   (%[[X:.*]]: i8, %[[Y:.*]]: i8)
//       CHECK:   %[[AND:.*]] = arith.andi %[[X]], %[[Y]] : i8
//       CHECK:   %[[R:.*]] = arith.extsi %[[AND]] : i8 to i32
//       CHECK:   return %[[R]]
func.func @andi_of_extsi(%x: i8, %y: i8) -> i32 {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.extsi %y : i8 to i32
  %2 = arith.andi %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @ori_ori_constant
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 7 : i32
//       CHECK:   %[[R:.*]] = arith.ori %[[X]], %[[C]] : i32
//       CHECK:   return %[[R]]
func.func @ori_ori_constant(%x: i32) -> i32 {
  %c3 = arith.constant 3 : i32
  %c6 = arith.constant 6 : i32
  %0 = arith.ori %x, %c3 : i32
  %1 = arith.ori %0, %c6 : i32
  return %1 : i32
}

// CHECK-LABEL: func @ori_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8, %[[Y:.*]]: i8)
//       CHECK:   %[[OR:.*]] = arith.ori %[[X]], %[[Y]] : i8
//       CHECK:   %[[R:.*]] = arith.extui %[[OR]] : i8 to i32
//       CHECK:   return %[[R]]
func.func @ori_of_extui(%x: i8, %y: i8) -> i32 {
  %0 = arith.extui %x : i8 to i32
  %1 = arith.extui %y nneg : i8 to i32
  %2 = arith.ori %0, %1 : i32
  return %2 : i32
}

// CHECK-LABEL: func @ori_of_extsi
//  CHECK-SAME:   (%[[X:.*]]: i8, %[[Y:.*]]: i8)
//       CHECK:   %[[OR:.*]] = arith.ori %[[X]], %[[Y]] : i8
//       CHECK:   %[[R:.*]] = arith.extsi %[[OR]] : i8 to i32
//       CHECK:   return %[[R]]
func.func @ori_of_extsi(%x: i8, %y: i8) -> i32 {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.extsi %y : i8 to i32
  %2 = arith.ori %0, %1 : i32
  return %2 : i32
}

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @trunci_extsi_to_extsi
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.extsi %[[X]] : i8 to i16
//       CHECK:   return %[[R]]
func.func @trunci_extsi_to_extsi(%x: i8) -> i16 {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.trunci %0 : i32 to i16
  return %1 : i16
}

// CHECK-LABEL: func @trunci_extui_to_extui
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.extui %[[X]] nneg : i8 to i16
//       CHECK:   return %[[R]]
func.func @trunci_extui_to_extui(%x: i8) -> i16 {
  %0 = arith.extui %x nneg : i8 to i32
  %1 = arith.trunci %0 : i32 to i16
  return %1 : i16
}

// CHECK-LABEL: func @trunci_shrsi_to_trunci_shrui
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[C:.*]] = arith.constant 16 : i32
//       CHECK:   %[[SHR:.*]] = arith.shrui %[[X]], %[[C]] exact : i32
//       CHECK:   %[[R:.*]] = arith.trunci %[[SHR]] overflow<nuw> : i32 to i16
//       CHECK:   return %[[R]]
func.func @trunci_shrsi_to_trunci_shrui(%x: i32) -> i16 {
  %c16 = arith.constant 16 : i32
  %0 = arith.shrsi %x, %c16 exact : i32
  %1 = arith.trunci %0 overflow<nuw> : i32 to i16
  return %1 : i16
}

// CHECK-LABEL: func @trunci_shrsi_wrong_amount
//       CHECK:   arith.shrsi
func.func @trunci_shrsi_wrong_amount(%x: i32) -> i16 {
  %c8 = arith.constant 8 : i32
  %0 = arith.shrsi %x, %c8 : i32
  %1 = arith.trunci %0 : i32 to i16
  return %1 : i16
}

//===----------------------------------------------------------------------===//
// TruncFOp / SIToFPOp / UIToFPOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @truncf_sitofp
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.sitofp %[[X]] : i32 to f32
//       CHECK:   return %[[R]]
func.func @truncf_sitofp(%x: i32) -> f32 {
  %0 = arith.sitofp %x : i32 to f64
  %1 = arith.truncf %0 : f64 to f32
  return %1 : f32
}

// CHECK-LABEL: func @truncf_uitofp
//  CHECK-SAME:   (%[[X:.*]]: i32)
//       CHECK:   %[[R:.*]] = arith.uitofp %[[X]] nneg : i32 to f32
//       CHECK:   return %[[R]]
func.func @truncf_uitofp(%x: i32) -> f32 {
  %0 = arith.uitofp %x nneg : i32 to f64
  %1 = arith.truncf %0 : f64 to f32
  return %1 : f32
}

// CHECK-LABEL: func @truncf_sitofp_rounding_mode
//       CHECK:   arith.truncf
func.func @truncf_sitofp_rounding_mode(%x: i32) -> f32 {
  %0 = arith.sitofp %x : i32 to f64
  %1 = arith.truncf %0 downward : f64 to f32
  return %1 : f32
}

// CHECK-LABEL: func @sitofp_of_extsi
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.sitofp %[[X]] : i8 to f32
//       CHECK:   return %[[R]]
func.func @sitofp_of_extsi(%x: i8) -> f32 {
  %0 = arith.extsi %x : i8 to i32
  %1 = arith.sitofp %0 : i32 to f32
  return %1 : f32
}

// CHECK-LABEL: func @sitofp_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.uitofp %[[X]] nneg : i8 to f32
//       CHECK:   return %[[R]]
func.func @sitofp_of_extui(%x: i8) -> f32 {
  %0 = arith.extui %x nneg : i8 to i32
  %1 = arith.sitofp %0 : i32 to f32
  return %1 : f32
}

// CHECK-LABEL: func @uitofp_of_extui
//  CHECK-SAME:   (%[[X:.*]]: i8)
//       CHECK:   %[[R:.*]] = arith.uitofp %[[X]] : i8 to f32
//       CHECK:   return %[[R]]
func.func @uitofp_of_extui(%x: i8) -> f32 {
  %0 = arith.extui %x : i8 to i32
  %1 = arith.uitofp %0 nneg : i32 to f32
  return %1 : f32
}

//===----------------------------------------------------------------------===//
// Floating point
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @addf_of_negf_lhs
//  CHECK-SAME:   (%[[X:.*]]: f32, %[[Y:.*]]: f32)
//       CHECK:   %[[R:.*]] = arith.subf %[[Y]], %[[X]] upward fastmath<nnan> : f32
//       CHECK:   return %[[R]]
func.func @addf_of_negf_lhs(%x: f32, %y: f32) -> f32 {
  %0 = arith.negf %x : f32
  %1 = arith.addf %0, %y upward fastmath<nnan> : f32
  return %1 : f32
}

// CHECK-LABEL: func @addf_of_negf_rhs
//  CHECK-SAME:   (%[[X:.*]]: f32, %[[Y:.*]]: f32)
//       CHECK:   %[[R:.*]] = arith.subf %[[X]], %[[Y]] : f32
//       CHECK:   return %[[R]]
func.func @addf_of_negf_rhs(%x: f32, %y: f32) -> f32 {
  %0 = arith.negf %y : f32
  %1 = arith.addf %x, %0 : f32
  return %1 : f32
}

// CHECK-LABEL: func @subf_of_neg_zero
//  CHECK-SAME:   (%[[X:.*]]: f32)
//       CHECK:   %[[R:.*]] = arith.negf %[[X]] fastmath<nsz> : f32
//       CHECK:   return %[[R]]
func.func @subf_of_neg_zero(%x: f32) -> f32 {
  %c = arith.constant -0.0 : f32
  %0 = arith.subf %c, %x fastmath<nsz> : f32
  return %0 : f32
}

// CHECK-LABEL: func @subf_of_neg_zero_rounding_mode
//       CHECK:   arith.subf
func.func @subf_of_neg_zero_rounding_mode(%x: f32) -> f32 {
  %c = arith.constant -0.0 : f32
  %0 = arith.subf %c, %x downward : f32
  return %0 : f32
}

// CHECK-LABEL: func @mulf_of_negf
//  CHECK-SAME:   (%[[X:.*]]: f32, %[[Y:.*]]: f32)
//       CHECK:   %[[R:.*]] = arith.mulf %[[X]], %[[Y]] fastmath<fast> : f32
//       CHECK:   return %[[R]]
func.func @mulf_of_negf(%x: f32, %y: f32) -> f32 {
  %0 = arith.negf %x : f32
  %1 = arith.negf %y : f32
  %2 = arith.mulf %0, %1 fastmath<fast> : f32
  return %2 : f32
}

// CHECK-LABEL: func @divf_of_negf
//  CHECK-SAME:   (%[[X:.*]]: f32, %[[Y:.*]]: f32)
//       CHECK:   %[[R:.*]] = arith.divf %[[X]], %[[Y]] : f32
//       CHECK:   return %[[R]]
func.func @divf_of_negf(%x: f32, %y: f32) -> f32 {
  %0 = arith.negf %x : f32
  %1 = arith.negf %y : f32
  %2 = arith.divf %0, %1 : f32
  return %2 : f32
}
