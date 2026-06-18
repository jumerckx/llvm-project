// RUN: mlir-translate --match-to-cpp %s | FileCheck %s

// `switch_op_name` -> an if/else-if chain on the runtime op name.

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

// CHECK:   ::llvm::StringRef name0 = op->getName().getStringRef();
// CHECK:   if (name0 == "arith.addi") {
// CHECK:     if (::mlir::succeeded(rewrite_addi(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:   }
// CHECK:   else if (name0 == "arith.subi") {
// CHECK:     if (::mlir::succeeded(rewrite_subi(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:   }
// CHECK:   return ::mlir::failure();
