// RUN: mlir-match-to-cpp %s | FileCheck %s

module {
  module @rewriters {
    module @r {}
  }

  match.matcher @m root(%root : !pdl.operation) {
    %0 = match.get_results 1 of %root : !match.optional<!pdl.range<value>>
    %1 = match.is_not_null %0 : !pdl.range<value>
    %2 = match.get_operands 0 of %root : !match.optional<!pdl.value>
    %3 = match.is_not_null %2 : !pdl.value
    match.success @rewriters::@r benefit(1) (%root, %1, %3 : !pdl.operation, !pdl.range<value>, !pdl.value)
  }
}

// CHECK: #include "mlir/Target/MatchToCpp/Runtime.h"
// CHECK: rewrite_r(::mlir::PatternRewriter &rewriter, ::mlir::Operation *, ::mlir::ValueRange, ::mlir::Value);
// CHECK:   ::std::optional<::mlir::ValueRange> v0 = ::mlir::match::runtime::getResultGroup(op, 1);
// CHECK:   if (!v0)
// CHECK:     return ::mlir::failure();
// CHECK:   ::mlir::ValueRange v1 = *v0;
// CHECK:   ::mlir::Value v2 = ::mlir::match::runtime::getSingleValue(::mlir::match::runtime::getOperandGroup(op, 0));
// CHECK:   if (!v2)
// CHECK:     return ::mlir::failure();
// CHECK:   if (::mlir::succeeded(rewrite_r(rewriter, op, v1, v2)))
