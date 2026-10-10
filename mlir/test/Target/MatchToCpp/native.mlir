// RUN: mlir-match-to-cpp %s | FileCheck %s

module {
  module @rewriters { module @r {} }

  match.matcher @m root(%root : !pdl.operation) {
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !pdl.range<value>
    %attr = match.apply_native_constraint "attr_of"(%root : !pdl.operation) : !pdl.attribute
    %ts = match.apply_native_constraint "types_of"(%vs : !pdl.range<value>) : !pdl.range<type>
    match.apply_native_constraint "check"(%attr, %ts : !pdl.attribute, !pdl.range<type>) is_negated = true
    match.success @rewriters::@r benefit(1) (%root, %attr : !pdl.operation, !pdl.attribute)
  }
}

// CHECK:   ::mlir::ValueRange v0 = op->getResults();
// CHECK:   ::mlir::Attribute v1;
// CHECK:   if (::mlir::failed(::mlir::match::runtime::callNative(attr_ofPDLFn, rewriter, ::std::tie(v1), op)))
// CHECK:     return ::mlir::failure();
// CHECK:   ::llvm::SmallVector<::mlir::Type> v2;
// CHECK:   if (::mlir::failed(::mlir::match::runtime::callNative(types_ofPDLFn, rewriter, ::std::tie(v2), v0)))
// CHECK:     return ::mlir::failure();
// CHECK:   if (::mlir::succeeded(::mlir::match::runtime::callNative(checkPDLFn, rewriter, ::std::tie(), v1, v2)))
// CHECK:     return ::mlir::failure();
// CHECK:   if (::mlir::succeeded(rewrite_r(rewriter, op, v1)))
