// RUN: mlir-match-to-cpp %s --ods=%mlir_src_root/include/mlir/Dialect/Arith/IR/ArithOps.td \
// RUN:   -I %mlir_src_root/include --namespace=ns | FileCheck %s
// RUN: mlir-match-to-cpp %s --ods=%mlir_src_root/include/mlir/Dialect/Arith/IR/ArithOps.td \
// RUN:   -I %mlir_src_root/include --combine-matchers=none | FileCheck %s --check-prefix=SPLIT
// RUN: mlir-match-to-cpp %s --ods=%mlir_src_root/include/mlir/Dialect/Arith/IR/ArithOps.td \
// RUN:   -I %mlir_src_root/include --combine-matchers=all | FileCheck %s --check-prefix=ALL
// RUN: mlir-match-to-cpp %s | FileCheck %s --check-prefix=NO-ODS

module {
  pdl.pattern @sub_to_add : benefit(1) {
    %lhs = pdl.operand
    %rhs = pdl.operand
    %type = pdl.type
    %root = pdl.operation "arith.subi"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %root {
      %add = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value)
      pdl.replace %root with %add
    }
  }

  pdl.pattern @mul_to_add : benefit(2) {
    %lhs = pdl.operand
    %rhs = pdl.operand
    %type = pdl.type
    %root = pdl.operation "arith.muli"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %root {
      %add = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value)
      pdl.replace %root with %add
    }
  }

  pdl.pattern @sub_self : benefit(3) {
    %x = pdl.operand
    %type = pdl.type
    %root = pdl.operation "arith.subi"(%x, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %root {
      %add = pdl.operation "arith.addi"(%x, %x : !pdl.value, !pdl.value)
      pdl.replace %root with %add
    }
  }
}

// CHECK: namespace ns {
// CHECK: static ::llvm::LogicalResult rewrite_sub_to_add(::mlir::PatternRewriter &rewriter, ::mlir::Location loc,
// CHECK:   ::mlir::arith::AddIOp::inferReturnTypes(
// CHECK: static ::llvm::LogicalResult rewrite_sub_self(
// CHECK: static ::llvm::LogicalResult rewrite_mul_to_add(
// CHECK: struct GeneratedMatcher_0
// CHECK:   ::mlir::RewritePattern("arith.subi", 3, context)
// CHECK:   ::mlir::arith::SubIOp castedOp0 = ::llvm::cast<::mlir::arith::SubIOp>(op);
// CHECK:   deferred = [=, &rewriter]
// CHECK-NEXT: rewriter.setInsertionPoint(op);
// CHECK-NEXT: rewrite_sub_to_add(
// CHECK:   rewrite_sub_self(
// CHECK: struct GeneratedMatcher_1
// CHECK:   ::mlir::RewritePattern("arith.muli", 2, context)
// CHECK-NOT: struct GeneratedMatcher_2
// CHECK: set.add<GeneratedMatcher_0, GeneratedMatcher_1>(set.getContext(), configSet);
// CHECK: } // namespace ns

// SPLIT-NOT: namespace ns
// SPLIT: struct GeneratedMatcher_0
// SPLIT:   ::mlir::RewritePattern("arith.subi", 1, context)
// SPLIT:   ::mlir::arith::SubIOp castedOp0 = ::llvm::cast<::mlir::arith::SubIOp>(op);
// SPLIT: struct GeneratedMatcher_1
// SPLIT:   ::mlir::RewritePattern("arith.muli", 2, context)
// SPLIT: struct GeneratedMatcher_2
// SPLIT:   ::mlir::RewritePattern("arith.subi", 3, context)

// ALL: struct GeneratedMatcher_0
// ALL:   ::mlir::RewritePattern(::mlir::Pattern::MatchAnyOpTypeTag(), 3, context)
// ALL-NOT: struct GeneratedMatcher_1

// NO-ODS: ::mlir::OperationState state{{[0-9]+}}(loc, "arith.addi");
// NO-ODS-NOT: inferReturnTypes
// NO-ODS: state{{[0-9]+}}.addTypes(v{{[0-9]+}});
