// RUN: mlir-match-to-cpp %s --ods=%mlir_src_root/include/mlir/Dialect/Arith/IR/ArithOps.td \
// RUN:   -I %mlir_src_root/include | FileCheck %s

module {
  module @rewriters {
    module @addf {}
  }

  match.matcher @addf_matcher root(%root : !pdl.operation) {
    match.has_name %root, "arith.addf"
    match.check_operand_count %root is 2
    %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %1 = match.is_not_null %0 : !pdl.value
    match.success @rewriters::@addf benefit(1) (%root, %1 : !pdl.operation, !pdl.value)
  }
}

// CHECK-NOT: #include "mlir/Dialect/Arith/IR/Arith.h"
// CHECK: ::llvm::LogicalResult rewrite_addf(::mlir::PatternRewriter &rewriter, ::mlir::Operation *, ::mlir::Value);
// CHECK: struct GeneratedMatcher_0 : public ::mlir::RewritePattern
// CHECK: matchAndRewrite(::mlir::Operation *op,
// CHECK:   ::mlir::arith::AddFOp castedOp0 = ::llvm::dyn_cast<::mlir::arith::AddFOp>(op);
// CHECK:   if (!castedOp0)
// CHECK:     return ::mlir::failure();
// CHECK-NOT: getNumOperands()
// CHECK:   ::mlir::Value v1 = castedOp0.getOperation()->getOperand(0);
// CHECK-NOT: if (!v1)
// CHECK:   rewriter.setInsertionPoint(op);
// CHECK:   if (::mlir::succeeded(rewrite_addf(rewriter, op, v1)))
// CHECK:     return ::mlir::success();
