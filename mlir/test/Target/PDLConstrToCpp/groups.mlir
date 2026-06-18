// RUN: mlir-translate --pdl-constr-to-cpp %s | FileCheck %s

// Indexed result group -> std::optional<ValueRange> via the emitted
// __pdl_get_group helper, unwrapped by is_not_null into a bare ValueRange.

module {
  module @rewriters {
    module @r {}
  }

  pdl_constr.matcher @m root(%root : !pdl.operation) {
    %0 = pdl_constr.get_results 1 of %root : !pdl_constr.optional<!pdl.range<value>>
    %1 = pdl_constr.is_not_null %0 : !pdl_constr.optional<!pdl.range<value>> -> !pdl.range<value>
    pdl_constr.success @rewriters::@r benefit(1) (%root, %1 : !pdl.operation, !pdl.range<value>)
  }
}

// CHECK: rewrite_r(::mlir::PatternRewriter &rewriter, ::mlir::Operation *, ::mlir::ValueRange);
// CHECK: static ::std::optional<::mlir::ValueRange>
// CHECK: __pdl_get_group(::mlir::Operation *op, unsigned index,
// CHECK:   ::std::optional<::mlir::ValueRange> v0 = __pdl_get_group(op, 1, ::mlir::ValueRange(op->getResults()), "resultSegmentSizes", op->hasTrait<::mlir::OpTrait::AttrSizedResultSegments>());
// CHECK:   if (!v0)
// CHECK:     return ::mlir::failure();
// CHECK:   ::mlir::ValueRange v1 = *v0;
// CHECK:   if (::mlir::succeeded(rewrite_r(rewriter, op, v1)))
