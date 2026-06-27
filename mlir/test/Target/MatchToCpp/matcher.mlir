// RUN: mlir-translate --match-to-cpp %s | FileCheck %s

// A flat matcher on a *registered* op (arith.addf): because the concrete C++
// class is known, the op-name test becomes a `dyn_cast`, the operand-count test
// is elided (AddFOp always has two operands), operand navigation drops its
// bounds check, and `is_not_null` collapses to an alias.

module {
  // The rewriter symbols referenced by `success` only need to resolve; here we
  // use nested modules so the test does not depend on pdl_interp.
  module @rewriters {
    module @addf {}
  }

  match.matcher @addf_matcher root(%root : !pdl.operation) {
    match.has_name %root, "arith.addf"
    match.check_operand_count %root is 2
    %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %1 = match.is_not_null %0 : !match.optional<!pdl.value> -> !pdl.value
    match.success @rewriters::@addf benefit(1) (%root, %1 : !pdl.operation, !pdl.value)
  }
}

// The generated matcher must pull in the op's header.
// CHECK: #include "mlir/Dialect/Arith/IR/Arith.h"
// CHECK: ::llvm::LogicalResult rewrite_addf(::mlir::PatternRewriter &rewriter, ::mlir::Operation *, ::mlir::Value);
// CHECK: struct GeneratedMatcher_0 : public ::mlir::RewritePattern
// CHECK: matchAndRewrite(::mlir::Operation *op,
// dyn_cast subsumes the name test.
// CHECK:   ::mlir::arith::AddFOp castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddFOp>(op);
// CHECK:   if (!castedOp0)
// CHECK:     return ::mlir::failure();
// No operand-count check is emitted (AddFOp has a fixed operand count of 2).
// CHECK-NOT: getNumOperands()
// In-range operand navigation needs no bounds check and is never null.
// CHECK:   ::mlir::Value v1 = castedOp0.getOperation()->getOperand(0);
// CHECK-NOT: if (!v1)
// CHECK:   rewriter.setInsertionPoint(op);
// CHECK:   if (::mlir::succeeded(rewrite_addf(rewriter, op, v1)))
// CHECK:     return ::mlir::success();
