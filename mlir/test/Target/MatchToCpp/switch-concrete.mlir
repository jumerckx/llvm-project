// RUN: mlir-match-to-cpp %s --ods=%mlir_src_root/include/mlir/Dialect/Arith/IR/ArithOps.td \
// RUN:   -I %mlir_src_root/include | FileCheck %s

module {
  module @rewriters {
    module @addi {}
    module @subi {}
  }

  match.matcher @m root(%root : !pdl.operation) {
    match.switch_op_name %root case "arith.addi" {
      match.success @rewriters::@addi benefit(1) (%root : !pdl.operation)
    } case "arith.subi" {
      match.success @rewriters::@subi benefit(1) (%root : !pdl.operation)
    }
  }
}

// CHECK-NOT: getName().getStringRef()
// CHECK:   if (::llvm::isa<::mlir::arith::AddIOp>(op)) {
// CHECK:     ::mlir::arith::AddIOp castedOp0 = ::llvm::cast<::mlir::arith::AddIOp>(op);
// CHECK:     if (::mlir::succeeded(rewrite_addi(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:   }
// CHECK:   else if (::llvm::isa<::mlir::arith::SubIOp>(op)) {
// CHECK:     ::mlir::arith::SubIOp castedOp1 = ::llvm::cast<::mlir::arith::SubIOp>(op);
// CHECK:     if (::mlir::succeeded(rewrite_subi(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:   }
// CHECK:   return ::mlir::failure();
