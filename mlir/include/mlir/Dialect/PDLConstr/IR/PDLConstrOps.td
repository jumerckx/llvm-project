//===- PDLConstrOps.td - PDL Constraint ops ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTROPS
#define MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTROPS

include "mlir/Dialect/PDLConstr/IR/PDLConstrTypes.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class
//===----------------------------------------------------------------------===//

class PDLConstr_Op<string mnemonic, list<Trait> traits = []>
    : Op<PDLConstr_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// pdl_constr::PatternOp
//===----------------------------------------------------------------------===//

def PDLConstr_PatternOp : PDLConstr_Op<"pattern", [
    IsolatedFromAbove, SingleBlock,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    Symbol
  ]> {
  let summary = "Wraps a single pattern's navigation and constraints";
  let description = [{
    `pdl_constr.pattern` wraps a single pattern's navigation and constraints
    in one op with a single block region. The block argument is the root
    operation. Terminated by `pdl_constr.success`.

    Example:

    ```mlir
    pdl_constr.pattern @name benefit(1) root(%root : !pdl.operation) {
      // ... navigation ops, constraint ops, combinators ...
      pdl_constr.success %pred
    }
    ```
  }];

  let arguments = (ins ConfinedAttr<I16Attr, [IntNonNegative]>:$benefit,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    ($sym_name^)? `:` `benefit` `(` $benefit `)` `root`
    custom<PatternRegion>($bodyRegion)
    attr-dict-with-keyword
  }];
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Navigation Ops - Nullable (return !pdl_constr.optional<T>)
//===----------------------------------------------------------------------===//

def PDLConstr_GetOperandOp : PDLConstr_Op<"get_operand", [Pure]> {
  let summary = "Get an operand of an operation (nullable)";
  let description = [{
    Navigates to a specific operand of an operation. Returns
    `!pdl_constr.optional<!pdl.value>` because the operand may not exist.
  }];
  let arguments = (ins PDL_Operation:$op, I32Attr:$index);
  let results = (outs PDLConstr_OptionalType:$result);
  let assemblyFormat = "$index `of` $op attr-dict `:` type($result)";
}

def PDLConstr_GetOperandsOp : PDLConstr_Op<"get_operands", [Pure]> {
  let summary = "Get a group of operands of an operation (nullable)";
  let description = [{
    Navigates to a group of operands of an operation. Returns
    `!pdl_constr.optional<!pdl.range<value>>`. If no index is provided,
    returns all operands.
  }];
  let arguments = (ins PDL_Operation:$op, OptionalAttr<I32Attr>:$index);
  let results = (outs PDLConstr_OptionalType:$result);
  let assemblyFormat = "($index^)? `of` $op attr-dict `:` type($result)";
}

def PDLConstr_GetResultOp : PDLConstr_Op<"get_result", [Pure]> {
  let summary = "Get a result of an operation (nullable)";
  let description = [{
    Navigates to a specific result of an operation. Returns
    `!pdl_constr.optional<!pdl.value>` because the result may not exist.
  }];
  let arguments = (ins PDL_Operation:$op, I32Attr:$index);
  let results = (outs PDLConstr_OptionalType:$result);
  let assemblyFormat = "$index `of` $op attr-dict `:` type($result)";
}

def PDLConstr_GetResultsOp : PDLConstr_Op<"get_results", [Pure]> {
  let summary = "Get a group of results of an operation (nullable)";
  let description = [{
    Navigates to a group of results of an operation. Returns
    `!pdl_constr.optional<!pdl.range<value>>`. If no index is provided,
    returns all results.
  }];
  let arguments = (ins PDL_Operation:$op, OptionalAttr<I32Attr>:$index);
  let results = (outs PDLConstr_OptionalType:$result);
  let assemblyFormat = "($index^)? `of` $op attr-dict `:` type($result)";
}

def PDLConstr_GetAttributeOp : PDLConstr_Op<"get_attribute", [Pure]> {
  let summary = "Get a named attribute of an operation (nullable)";
  let description = [{
    Navigates to a named attribute of an operation. Returns
    `!pdl_constr.optional<!pdl.attribute>` because the attribute may not exist.
  }];
  let arguments = (ins PDL_Operation:$op, StrAttr:$name);
  let results = (outs PDLConstr_OptionalType:$result);
  let assemblyFormat = "$name `of` $op attr-dict `:` type($result)";
}

def PDLConstr_GetDefiningOpOp : PDLConstr_Op<"get_defining_op", [Pure]> {
  let summary = "Get the defining operation of a value (nullable)";
  let description = [{
    Navigates to the defining operation of a value. Returns
    `!pdl_constr.optional<!pdl.operation>` because the value may be a block
    argument with no defining op.
  }];
  let arguments = (ins PDL_Value:$value);
  let results = (outs PDLConstr_OptionalType:$result);
  let assemblyFormat = "`of` $value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Navigation Ops - Non-nullable (return bare PDL types)
//===----------------------------------------------------------------------===//

def PDLConstr_GetValueTypeOp : PDLConstr_Op<"get_value_type", [Pure]> {
  let summary = "Get the type of a value (non-nullable)";
  let description = [{
    Gets the type of a value. Always succeeds because the input value
    has already been null-checked.
  }];
  let arguments = (ins PDL_Value:$value);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "`of` $value attr-dict `:` type($result)";
}

def PDLConstr_GetAttributeTypeOp : PDLConstr_Op<"get_attribute_type", [Pure]> {
  let summary = "Get the type of an attribute (non-nullable)";
  let description = [{
    Gets the type of an attribute. Always succeeds because the input attribute
    has already been null-checked.
  }];
  let arguments = (ins PDL_Attribute:$attribute);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "`of` $attribute attr-dict `:` type($result)";
}

def PDLConstr_GetUsersOp : PDLConstr_Op<"get_users", [Pure]> {
  let summary = "Get all users of a value (non-nullable)";
  let description = [{
    Gets all operations that use a value. Always returns a range
    (possibly empty), so it cannot fail.
  }];
  let arguments = (ins PDL_Value:$value);
  let results = (outs PDL_RangeOf<PDL_Operation>:$result);
  let assemblyFormat = "`of` $value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// pdl_constr::GetEachOp
//===----------------------------------------------------------------------===//

def PDLConstr_GetEachOp : PDLConstr_Op<"get_each", [Pure]> {
  let summary = "Extract elements from a range for constraint purposes";
  let description = [{
    Extracts a single element from a range for constraint purposes. At this
    level there is no structured loop; `get_each` declares that the pattern
    applies constraints to each element of the range. The lowering to
    `pdl_interp` generates the actual `pdl_interp.foreach` loop.

    Semantics: any constraints applied to the result must hold for at least
    one element of the range (existential semantics).
  }];
  let arguments = (ins PDL_AnyType:$range);
  let results = (outs PDL_AnyType:$result);
  let assemblyFormat = "$range `:` type($range) `->` type($result) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Constraint Ops
//===----------------------------------------------------------------------===//

def PDLConstr_IsNotNullOp : PDLConstr_Op<"is_not_null", [Pure]> {
  let summary = "Check that an optional value is non-null and unwrap it";
  let description = [{
    Checks that an optional value is non-null and unwraps it, producing both
    a `!pdl_constr.pred` and the bare inner value.

    The unwrapped result may only be used by ops that are dominated by a
    successful check of the predicate.
  }];
  let arguments = (ins PDLConstr_OptionalType:$optionalValue);
  let results = (outs PDLConstr_PredType:$pred, PDL_AnyType:$unwrapped);
  let assemblyFormat = [{
    $optionalValue `:` type($optionalValue) `->` type($pred) `,` type($unwrapped)
    attr-dict
  }];
  let hasVerifier = 1;
}

def PDLConstr_HasNameOp : PDLConstr_Op<"has_name", [Pure]> {
  let summary = "Check the operation name";
  let arguments = (ins PDL_Operation:$op, StrAttr:$name);
  let results = (outs PDLConstr_PredType:$pred);
  let assemblyFormat = "$op `,` $name attr-dict";
}

def PDLConstr_EqualOp : PDLConstr_Op<"equal", [Pure, SameTypeOperands]> {
  let summary = "Check equality of two values";
  let arguments = (ins PDL_AnyType:$lhs, PDL_AnyType:$rhs);
  let results = (outs PDLConstr_PredType:$pred);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) attr-dict";
}

def PDLConstr_HasTypeOp : PDLConstr_Op<"has_type", [Pure]> {
  let summary = "Check that a type matches a constant type";
  let arguments = (ins PDL_Type:$type, TypeAttr:$constantType);
  let results = (outs PDLConstr_PredType:$pred);
  let assemblyFormat = "$type `,` $constantType attr-dict";
}

def PDLConstr_HasAttrValueOp : PDLConstr_Op<"has_attr_value", [Pure]> {
  let summary = "Check that an attribute matches a constant value";
  let arguments = (ins PDL_Attribute:$attribute, AnyAttr:$value);
  let results = (outs PDLConstr_PredType:$pred);
  let assemblyFormat = "$attribute `is` $value attr-dict";
}

def PDLConstr_CheckOperandCountOp : PDLConstr_Op<"check_operand_count", [Pure]> {
  let summary = "Check the operand count of an operation";
  let arguments = (ins PDL_Operation:$op,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$count,
                       UnitAttr:$atLeast);
  let results = (outs PDLConstr_PredType:$pred);
  let assemblyFormat = [{
    $op `is` (`at_least` $atLeast^)? $count attr-dict
  }];
}

def PDLConstr_CheckResultCountOp : PDLConstr_Op<"check_result_count", [Pure]> {
  let summary = "Check the result count of an operation";
  let arguments = (ins PDL_Operation:$op,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$count,
                       UnitAttr:$atLeast);
  let results = (outs PDLConstr_PredType:$pred);
  let assemblyFormat = [{
    $op `is` (`at_least` $atLeast^)? $count attr-dict
  }];
}

def PDLConstr_ApplyNativeConstraintOp : PDLConstr_Op<"apply_native_constraint", [Pure]> {
  let summary = "Apply a native constraint";
  let description = [{
    Applies an externally registered native constraint. Returns a
    `!pdl_constr.pred` as the first result, and may return additional
    `!pdl.*` results.
  }];
  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_AnyType>:$args,
                       DefaultValuedAttr<BoolAttr, "false">:$isNegated);
  let results = (outs PDLConstr_PredType:$pred, Variadic<PDL_AnyType>:$constraintResults);
  let assemblyFormat = [{
    $name `(` $args `:` type($args) `)` (`:` type($constraintResults)^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Combinators
//===----------------------------------------------------------------------===//

def PDLConstr_AllOp : PDLConstr_Op<"all", [Pure]> {
  let summary = "Logical AND over predicate values";
  let arguments = (ins Variadic<PDLConstr_PredType>:$preds);
  let results = (outs PDLConstr_PredType:$result);
  let assemblyFormat = "$preds attr-dict";
}

def PDLConstr_AnyOp : PDLConstr_Op<"any", [Pure]> {
  let summary = "Logical OR over predicate values";
  let arguments = (ins Variadic<PDLConstr_PredType>:$preds);
  let results = (outs PDLConstr_PredType:$result);
  let assemblyFormat = "$preds attr-dict";
}

//===----------------------------------------------------------------------===//
// Terminator
//===----------------------------------------------------------------------===//

def PDLConstr_SuccessOp : PDLConstr_Op<"success", [Pure, Terminator,
    HasParent<"pdl_constr::PatternOp">]> {
  let summary = "Marks a successful pattern match";
  let arguments = (ins PDLConstr_PredType:$pred);
  let assemblyFormat = "$pred attr-dict";
}

#endif // MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTROPS
