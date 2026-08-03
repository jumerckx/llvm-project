// RUN: mlir-match-to-cpp %s | FileCheck %s

// Upward traversal: get_users + foreach -> an existential for-loop whose body
// fails with `continue` and whose first matching element rewrites & returns.

module {
  module @rewriters {
    module @r {}
    module @r0 {}
  }

  match.matcher @m root(%root : !pdl.operation) {
    %res = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %res : !match.optional<!pdl.value> -> !pdl.value
    %users = match.get_users of %v : !pdl.range<operation>
    match.foreach %u in %users : !pdl.range<operation> {
      match.has_name %u, "test.use"
      match.success @rewriters::@r benefit(1) (%root, %u : !pdl.operation, !pdl.operation)
    }
  }

  // Ops after the loop run when it is exhausted: the loop falls out into the
  // enclosing scope rather than swallowing everything below it.
  match.matcher @fallback root(%root : !pdl.operation) {
    %res = match.get_result 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %res : !match.optional<!pdl.value> -> !pdl.value
    %users = match.get_users of %v : !pdl.range<operation>
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
// CHECK:     if (::mlir::succeeded(rewrite_r(rewriter, op, v2)))
// CHECK:       return ::mlir::success();
// CHECK:     continue;
// CHECK:   }
// CHECK:   return ::mlir::failure();

// CHECK-LABEL: struct GeneratedMatcher_1
// CHECK:   for (::mlir::Operation * v2 : v1) {
// CHECK:     if (v2->getName().getStringRef() != "test.use")
// CHECK:       continue;
// CHECK:     if (::mlir::succeeded(rewrite_r(rewriter, op, v2)))
// CHECK:       return ::mlir::success();
// CHECK:     continue;
// CHECK:   }
// The loop exhausted; the alternative below it still runs.
// CHECK:   if (op->getName().getStringRef() != "test.fallback")
// CHECK:     return ::mlir::failure();
// CHECK:   if (::mlir::succeeded(rewrite_r0(rewriter, op)))
// CHECK:     return ::mlir::success();
// CHECK:   return ::mlir::failure();
