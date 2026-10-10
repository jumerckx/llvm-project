// RUN: mlir-match-to-cpp %s | FileCheck %s

module {
  module @rewriters {
    module @a {}
    module @b {}
  }

  match.matcher @m root(%root : !pdl.operation) {
    match.has_name %root, "test.foo"
    match.try {
      match.check_operand_count %root is 2
      match.success @rewriters::@a benefit(2) (%root : !pdl.operation)
    }
    match.try {
      match.success @rewriters::@b benefit(1) (%root : !pdl.operation)
    }
  }

  match.matcher @deferred root(%root : !pdl.operation) {
    match.try {
      match.success @rewriters::@b benefit(1) (%root : !pdl.operation)
      match.check_result_count %root is 1
      match.success @rewriters::@b benefit(1) (%root : !pdl.operation)
    }
    match.try {
      match.check_operand_count %root is 2
      match.success @rewriters::@a benefit(2) (%root : !pdl.operation)
    }
  }
}

// CHECK-LABEL: struct GeneratedMatcher_0
// CHECK: ::mlir::RewritePattern("test.foo", 2, context)
// CHECK-NOT: deferred
// CHECK:   auto attempt0 = [&]() -> ::llvm::LogicalResult {
// CHECK:     if (op->getNumOperands() != 2)
// CHECK:       return ::mlir::failure();
// CHECK:     if (::mlir::succeeded(rewrite_a(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:     return ::mlir::failure();
// CHECK:   };
// CHECK:   if (::mlir::succeeded(attempt0())) return ::mlir::success();
// CHECK:   auto attempt1 = [&]() -> ::llvm::LogicalResult {
// CHECK:     if (::mlir::succeeded(rewrite_b(rewriter, op)))
// CHECK:       return ::mlir::success();
// CHECK:     return ::mlir::failure();
// CHECK:   };
// CHECK:   if (::mlir::succeeded(attempt1())) return ::mlir::success();
// CHECK:   return ::mlir::failure();

// CHECK-LABEL: struct GeneratedMatcher_1
// CHECK: ::mlir::RewritePattern(::mlir::Pattern::MatchAnyOpTypeTag(), 2, context)
// CHECK:   unsigned deferredBenefit = 0;
// CHECK:   ::llvm::unique_function<::llvm::LogicalResult()> deferred;
// CHECK:   auto match = [&]() -> ::llvm::LogicalResult {
// CHECK:     auto attempt0 = [&]() -> ::llvm::LogicalResult {
// CHECK:       if (!deferred || deferredBenefit < 1) {
// CHECK:         deferredBenefit = 1;
// CHECK:         deferred = [=, &rewriter]() -> ::llvm::LogicalResult {
// CHECK:           rewriter.setInsertionPoint(op);
// CHECK:           return rewrite_b(rewriter, op);
// CHECK:         };
// CHECK:       }
// CHECK:       if (op->getNumResults() != 1)
// CHECK:         return ::mlir::failure();
// CHECK:       if (!deferred || deferredBenefit < 1) {
// CHECK:       return ::mlir::failure();
// CHECK:     };
// CHECK:     auto attempt1 = [&]() -> ::llvm::LogicalResult {
// CHECK:       if (op->getNumOperands() != 2)
// CHECK:         return ::mlir::failure();
// CHECK-NOT:   applyDeferred
// CHECK:       if (::mlir::succeeded(rewrite_a(rewriter, op)))
// CHECK:         return ::mlir::success();
// CHECK:   if (::mlir::succeeded(match()))
// CHECK-NEXT:  return ::mlir::success();
// CHECK-NEXT: if (deferred)
// CHECK-NEXT:  return applyDeferred();
// CHECK-NEXT: return ::mlir::failure();
