// RUN: mlir-translate --match-to-cpp %s | FileCheck %s

// A flat matcher: op-name + operand-count tests, one navigation, one success.

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

// CHECK: ::llvm::LogicalResult rewrite_addf(::mlir::PatternRewriter &rewriter, ::mlir::Operation *, ::mlir::Value);
// CHECK: struct GeneratedMatcher_0 : public ::mlir::RewritePattern
// CHECK: ::mlir::RewritePattern(::mlir::Pattern::MatchAnyOpTypeTag(), 1, context)
// CHECK: matchAndRewrite(::mlir::Operation *op,
// CHECK:   if (op->getName().getStringRef() != "arith.addf")
// CHECK:     return ::mlir::failure();
// CHECK:   if (op->getNumOperands() != 2)
// CHECK:     return ::mlir::failure();
// CHECK:   ::mlir::Value v0 = (0 < op->getNumOperands()) ? op->getOperand(0) : ::mlir::Value();
// CHECK:   if (!v0)
// CHECK:     return ::mlir::failure();
// CHECK:   rewriter.setInsertionPoint(op);
// CHECK:   if (::mlir::succeeded(rewrite_addf(rewriter, op, v0)))
// CHECK:     return ::mlir::success();
// CHECK: void populateGeneratedPatterns(::mlir::RewritePatternSet &set) {
// CHECK:   set.add<GeneratedMatcher_0>(set.getContext());
