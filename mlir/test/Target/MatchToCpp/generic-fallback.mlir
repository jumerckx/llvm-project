// RUN: mlir-match-to-cpp %s | FileCheck %s

// An op whose name is not in the op-info registry falls back to the generic
// `Operation *` emission: a string name test, a runtime operand-count check,
// bounds-checked nullable operand navigation, and an explicit null check.

module {
  module @rewriters {
    module @foo {}
  }

  match.matcher @foo_matcher root(%root : !pdl.operation) {
    match.has_name %root, "test.foo"
    match.check_operand_count %root is 2
    %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %1 = match.is_not_null %0 : !match.optional<!pdl.value> -> !pdl.value
    match.success @rewriters::@foo benefit(1) (%root, %1 : !pdl.operation, !pdl.value)
  }
}

// No concrete op header / cast for an unregistered op.
// CHECK-NOT: dyn_cast
// CHECK:   if (op->getName().getStringRef() != "test.foo")
// CHECK:     return ::mlir::failure();
// CHECK:   if (op->getNumOperands() != 2)
// CHECK:     return ::mlir::failure();
// CHECK:   ::mlir::Value v0 = (0 < op->getNumOperands()) ? op->getOperand(0) : ::mlir::Value();
// CHECK:   if (!v0)
// CHECK:     return ::mlir::failure();
// CHECK:   if (::mlir::succeeded(rewrite_foo(rewriter, op, v0)))
