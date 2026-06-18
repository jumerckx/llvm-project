// RUN: mlir-translate --match-to-cpp %s | FileCheck %s

// Upward traversal: get_users + get_each -> an existential for-loop whose body
// fails with `continue` and whose first matching element rewrites & returns.

module {
  module @rewriters {
    module @r {}
  }

  match.matcher @m root(%root : !pdl.operation) {
    %res = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %res : !match.optional<!pdl.value> -> !pdl.value
    %users = match.get_users of %v : !pdl.range<operation>
    %u = match.get_each %users : !pdl.range<operation> -> !pdl.operation
    match.has_name %u, "test.use"
    match.success @rewriters::@r benefit(1) (%root, %u : !pdl.operation, !pdl.operation)
  }
}

// CHECK:   ::mlir::Value v0 = (0 < op->getNumResults()) ? op->getResult(0) : ::mlir::Value();
// CHECK:   if (!v0)
// CHECK:     return ::mlir::failure();
// CHECK:   auto v1 = v0.getUsers();
// CHECK:   for (::mlir::Operation * v2 : v1) {
// CHECK:     if (v2->getName().getStringRef() != "test.use")
// CHECK:       continue;
// CHECK:     if (::mlir::succeeded(rewrite_r(rewriter, op, v2)))
// CHECK:       return ::mlir::success();
// CHECK:     continue;
// CHECK:   }
// CHECK:   return ::mlir::failure();
