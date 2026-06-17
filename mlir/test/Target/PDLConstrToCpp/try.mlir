// RUN: mlir-translate --pdl-constr-to-cpp %s | FileCheck %s

// Two sibling `try` alternatives -> first-success-wins lambdas. The pattern's
// static benefit is the max over the success ops (2).

module {
  module @rewriters {
    module @a {}
    module @b {}
  }

  pdl_constr.matcher @m root(%root : !pdl.operation) {
    pdl_constr.has_name %root, "test.foo"
    pdl_constr.try {
      pdl_constr.check_operand_count %root is 2
      pdl_constr.success @rewriters::@a benefit(2) (%root : !pdl.operation)
    }
    pdl_constr.try {
      pdl_constr.success @rewriters::@b benefit(1) (%root : !pdl.operation)
    }
  }
}

// CHECK: ::mlir::RewritePattern(::mlir::Pattern::MatchAnyOpTypeTag(), 2, context)
// CHECK:   if (op->getName().getStringRef() != "test.foo")
// CHECK:   auto attempt0 = [&]() -> ::llvm::LogicalResult {
// CHECK:     if (op->getNumOperands() != 2)
// CHECK:       return ::mlir::failure();
// CHECK:     if (::mlir::succeeded(rewrite_a(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:   };
// CHECK:   if (::mlir::succeeded(attempt0())) return ::mlir::success();
// CHECK:   auto attempt1 = [&]() -> ::llvm::LogicalResult {
// CHECK:     if (::mlir::succeeded(rewrite_b(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:   };
// CHECK:   if (::mlir::succeeded(attempt1())) return ::mlir::success();
// CHECK:   return ::mlir::failure();
