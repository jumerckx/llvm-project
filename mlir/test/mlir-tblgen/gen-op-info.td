// RUN: mlir-tblgen -gen-op-info -I %S/../../include \
// RUN:   -op-info-func-name=populateTestOpInfo \
// RUN:   -op-info-include=mlir/Dialect/Test/IR/Test.h %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::mlir::test";
}
class NS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

// A fixed-arity op: two operands, one result, no variadics -> fixed counts.
def BinaryOp : NS_Op<"binary"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

// A variadic op: the flat operand count is not statically fixed (-1).
def VariadicOp : NS_Op<"variadic"> {
  let arguments = (ins Variadic<AnyType>:$ins);
  let results = (outs AnyType:$result);
}

// An op with AttrSizedOperandSegments: flat operand count not fixed either.
def SegmentedOp : NS_Op<"segmented", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$a, Variadic<AnyType>:$b);
  let results = (outs);
}

// CHECK: void populateTestOpInfo(::mlir::match::OpInfoRegistry &registry) {
// CHECK:   registry.insert("test.binary", ::mlir::match::OpInfo{"::mlir::test::BinaryOp", "mlir/Dialect/Test/IR/Test.h", 2, 1, false, false, 2, 1});
// CHECK:   registry.insert("test.variadic", ::mlir::match::OpInfo{"::mlir::test::VariadicOp", "mlir/Dialect/Test/IR/Test.h", 1, 1, false, false, -1, 1});
// CHECK:   registry.insert("test.segmented", ::mlir::match::OpInfo{"::mlir::test::SegmentedOp", "mlir/Dialect/Test/IR/Test.h", 2, 0, true, false, -1, 0});
// CHECK: }
