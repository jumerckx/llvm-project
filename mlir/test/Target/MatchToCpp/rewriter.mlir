// RUN: mlir-match-to-cpp %s --ods=%mlir_src_root/include/mlir/Dialect/Arith/IR/ArithOps.td \
// RUN:   -I %mlir_src_root/include | FileCheck %s

module {
  module @rewriters {
    match.rewriter @lower(%root: !pdl.operation, %lhs: !pdl.value, %attr: !pdl.attribute) {
      %flags = apply_native_rewrite "flags"(%root : !pdl.operation) : !pdl.attribute
      %lhsType = get_value_type of %lhs : !pdl.type
      %add = create_operation "arith.addi"(%lhs, %lhs : !pdl.value, !pdl.value) {"overflowFlags" = %flags} -> (%lhsType : !pdl.type)
      %sum = pdl.results 0 of %add -> !pdl.value
      %type = constant_type i32
      %types = constant_types [i32, i64]
      %all = pdl.results of %root
      %range = create_range %sum, %all : !pdl.value, !pdl.range<value>
      %generic = create_operation "test.op"(%range : !pdl.range<value>) {"attr" = %attr} -> (%type, %types : !pdl.type, !pdl.range<type>)
      %res = pdl.result 0 of %generic
      %rest = pdl.results 1 of %generic -> !pdl.range<value>
      %restTypes = get_value_type of %rest : !pdl.range<type>
      replace %root with (%res, %rest : !pdl.value, !pdl.range<value>)
      erase %generic
    }
  }

  match.matcher @m root(%root : !pdl.operation) {
    %o = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v = match.is_not_null %o : !pdl.value
    %d = match.get_defining_op of %v : !pdl.value -> !match.optional<!pdl.operation>
    %def = match.is_not_null %d : !pdl.operation
    %a = match.get_attribute "value" of %def : !match.optional<!pdl.attribute>
    %attr = match.is_not_null %a : !pdl.attribute
    match.success @rewriters::@lower benefit(1) (%root, %v, %attr : !pdl.operation, !pdl.value, !pdl.attribute)
  }
}

// CHECK-NOT: Hooks the user must define
// CHECK-LABEL: static ::llvm::LogicalResult rewrite_lower(::mlir::PatternRewriter &rewriter, ::mlir::Location loc, ::mlir::Operation * v0, ::mlir::Value v1, ::mlir::Attribute v2) {
// CHECK:   ::mlir::Attribute v3;
// CHECK:   if (::mlir::failed(::mlir::match::runtime::callNative(flagsPDLFn, rewriter, ::std::tie(v3), v0)))
// CHECK:     return ::mlir::failure();
// CHECK:   ::mlir::Type v4 = v1.getType();
// CHECK:   ::mlir::OperationState state5(loc, ::mlir::match::runtime::getOperationName<::mlir::arith::AddIOp>(rewriter.getContext()));
// CHECK:   state5.operands.push_back(v1);
// CHECK:   state5.operands.push_back(v1);
// CHECK:   if (v3)
// CHECK:     state5.addAttribute(::mlir::arith::AddIOp::getOverflowFlagsAttrName(state5.name), v3);
// CHECK:   state5.types.push_back(v4);
// CHECK:   ::mlir::Operation *v6 = rewriter.create(state5);
// CHECK:   ::mlir::arith::AddIOp castedOp7 = ::llvm::cast<::mlir::arith::AddIOp>(v6);
// CHECK:   ::mlir::Value v8 = castedOp7.getOperation()->getResult(0);
// CHECK:   ::mlir::Type v9 = ::mlir::IntegerType::get(rewriter.getContext(), 32);
// CHECK:   ::llvm::SmallVector<::mlir::Type> v10{::mlir::IntegerType::get(rewriter.getContext(), 32), ::mlir::IntegerType::get(rewriter.getContext(), 64)};
// CHECK:   ::mlir::ValueRange v11 = v0->getResults();
// CHECK:   ::llvm::SmallVector<::mlir::Value> v12;
// CHECK:   v12.push_back(v8);
// CHECK:   ::llvm::append_range(v12, v11);
// CHECK:   ::mlir::OperationState state13(loc, "test.op");
// CHECK:   state13.addOperands(v12);
// CHECK:   if (v2)
// CHECK:     state13.addAttribute(rewriter.getStringAttr("attr"), v2);
// CHECK:   state13.types.push_back(v9);
// CHECK:   state13.addTypes(v10);
// CHECK:   ::mlir::Operation *v14 = rewriter.create(state13);
// CHECK:   ::mlir::Value v15 = v14->getResult(0);
// CHECK:   ::mlir::ValueRange v16 = ::mlir::match::runtime::getResultGroup(v14, 1).value_or(::mlir::ValueRange());
// CHECK:   ::mlir::TypeRange v17 = ::mlir::ValueRange(v16).getTypes();
// CHECK:   ::llvm::SmallVector<::mlir::Value> repl18;
// CHECK:   repl18.push_back(v15);
// CHECK:   ::llvm::append_range(repl18, v16);
// CHECK:   rewriter.replaceOp(v0, repl18);
// CHECK:   rewriter.eraseOp(v14);
// CHECK:   return ::mlir::success();
// CHECK: }

// CHECK-LABEL: struct GeneratedMatcher_0
// CHECK:   ::mlir::Attribute v{{[0-9]+}} = v{{[0-9]+}}->getAttr("value");
// CHECK:   rewriter.setInsertionPoint(op);
// CHECK:   if (::mlir::succeeded(rewrite_lower(rewriter, rewriter.getFusedLoc({op->getLoc(), v{{[0-9]+}}->getLoc()}), op, v{{[0-9]+}}, v{{[0-9]+}})))
