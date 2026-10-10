// RUN: mlir-match-to-cpp %s | FileCheck %s

module {
  module @rewriters {
    module @r {}
    module @r0 {}
  }

  match.matcher @m root(%root : !pdl.operation) {
    %res = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %res : !pdl.value
    %users = match.get_users of %v
    match.foreach %u in %users : !pdl.range<operation> {
      match.has_name %u, "test.use"
      match.success @rewriters::@r benefit(1) (%root, %u : !pdl.operation, !pdl.operation)
    }
  }

  match.matcher @fallback root(%root : !pdl.operation) {
    %res = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %res : !pdl.value
    %users = match.get_users of %v
    match.foreach %u in %users : !pdl.range<operation> {
      match.has_name %u, "test.use"
      match.success @rewriters::@r benefit(2) (%root, %u : !pdl.operation, !pdl.operation)
    }
    match.has_name %root, "test.fallback"
    match.success @rewriters::@r0 benefit(1) (%root : !pdl.operation)
  }
}

// CHECK-LABEL: struct GeneratedMatcher_0
// CHECK:   ::mlir::Value v0 = (0 < op->getNumResults()) ? op->getResult(0) : ::mlir::Value();
// CHECK:   if (!v0)
// CHECK:     return ::mlir::failure();
// CHECK:   auto v1 = v0.getUsers();
// CHECK:   for (::mlir::Operation * v2 : v1) {
// CHECK:     if (v2->getName().getStringRef() != "test.use")
// CHECK:       continue;
// CHECK:     if (::mlir::succeeded(::mlir::match::runtime::applyRewrite(configs, rewriter, [&] { return rewrite_r(rewriter, op, v2); })))
// CHECK:       return ::mlir::success();
// CHECK:     continue;
// CHECK:   }
// CHECK:   return ::mlir::failure();

// CHECK-LABEL: struct GeneratedMatcher_1
// CHECK:   for (::mlir::Operation * v2 : v1) {
// CHECK:     if (v2->getName().getStringRef() != "test.use")
// CHECK:       continue;
// CHECK:     if (::mlir::succeeded(::mlir::match::runtime::applyRewrite(configs, rewriter, [&] { return rewrite_r(rewriter, op, v2); })))
// CHECK:       return ::mlir::success();
// CHECK:     continue;
// CHECK:   }
// CHECK:   if (op->getName().getStringRef() != "test.fallback")
// CHECK:     return ::mlir::failure();
// CHECK:   if (::mlir::succeeded(::mlir::match::runtime::applyRewrite(configs, rewriter, [&] { return rewrite_r0(rewriter, op); })))
// CHECK:     return ::mlir::success();
// CHECK:   return ::mlir::failure();
