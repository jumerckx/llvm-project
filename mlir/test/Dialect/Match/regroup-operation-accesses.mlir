// RUN: mlir-opt %s --match-regroup-operation-accesses | FileCheck %s
// Idempotence: running the pass twice yields the same result.
// RUN: mlir-opt %s --match-regroup-operation-accesses --match-regroup-operation-accesses | FileCheck %s

module {
  module @rewriters {
    module @r {}
    module @r0 {}
    module @r1 {}
  }

  // A shallow test on the root (`check_operand_count`) is deliberately written
  // *after* the descent into a deeper operation. The pass must pull it above
  // the `get_defining_op` so every root-level test runs before we cross into
  // the defining op.
  //
  // CHECK-LABEL: match.matcher @shallow_before_deep
  // CHECK: match.has_name %[[ROOT:[^ ,]+]], "arith.addi"
  // CHECK: match.check_operand_count %[[ROOT]] is 2
  // CHECK: match.get_defining_op
  // CHECK: match.has_name %{{.+}}, "arith.muli"
  match.matcher @shallow_before_deep root(%root : !pdl.operation) {
    match.has_name %root, "arith.addi"
    %op0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v0 = match.is_not_null %op0 : !match.optional<!pdl.value> -> !pdl.value
    %defA = match.get_defining_op of %v0 : !pdl.value -> !match.optional<!pdl.operation>
    %A = match.is_not_null %defA : !match.optional<!pdl.operation> -> !pdl.operation
    match.has_name %A, "arith.muli"
    match.check_operand_count %root is 2
    match.success @rewriters::@r benefit(1) (%root : !pdl.operation)
  }

  // A `match.get_each` is a `foreach` barrier: a shallow root test written after
  // it must NOT be hoisted above it (that would move it out of the loop). The
  // pass segments at `get_each`, so `check_operand_count` stays after it.
  //
  // CHECK-LABEL: match.matcher @get_each_barrier
  // CHECK: match.get_each
  // CHECK: match.check_operand_count
  match.matcher @get_each_barrier root(%root : !pdl.operation) {
    match.has_name %root, "arith.addi"
    %res = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %res : !match.optional<!pdl.value> -> !pdl.value
    %users = match.get_users of %v : !pdl.range<operation>
    %u = match.get_each %users : !pdl.range<operation> -> !pdl.operation
    match.has_name %u, "arith.muli"
    match.check_operand_count %root is 2
    match.success @rewriters::@r0 benefit(1) (%root : !pdl.operation)
  }

  // Reordering must not cross a `match.try` failure scope. The shallow root
  // test after the `try` stays after it even though the `try` descends into a
  // deeper operation.
  //
  // CHECK-LABEL: match.matcher @try_scope
  // CHECK: match.has_name %[[R:[^ ,]+]], "arith.addi"
  // CHECK: match.try {
  // CHECK:   match.get_defining_op
  // CHECK:   match.has_name %{{.+}}, "arith.muli"
  // CHECK: }
  // CHECK: match.check_operand_count %[[R]] is 2
  match.matcher @try_scope root(%root : !pdl.operation) {
    match.has_name %root, "arith.addi"
    match.try {
      %op0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
      %v0 = match.is_not_null %op0 : !match.optional<!pdl.value> -> !pdl.value
      %defA = match.get_defining_op of %v0 : !pdl.value -> !match.optional<!pdl.operation>
      %A = match.is_not_null %defA : !match.optional<!pdl.operation> -> !pdl.operation
      match.has_name %A, "arith.muli"
      match.success @rewriters::@r0 benefit(2) (%root : !pdl.operation)
    }
    match.check_operand_count %root is 2
    match.success @rewriters::@r1 benefit(1) (%root : !pdl.operation)
  }
}
