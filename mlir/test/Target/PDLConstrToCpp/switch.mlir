// RUN: mlir-translate --pdl-constr-to-cpp %s | FileCheck %s

// `switch_op_name` -> an if/else-if chain on the runtime op name.

module {
  module @rewriters {
    module @addi {}
    module @subi {}
  }

  pdl_constr.matcher @m root(%root : !pdl.operation) {
    pdl_constr.switch_op_name %root case "arith.addi" {
      pdl_constr.success @rewriters::@addi benefit(1) (%root : !pdl.operation)
    } case "arith.subi" {
      pdl_constr.success @rewriters::@subi benefit(1) (%root : !pdl.operation)
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
