//===- MatchOps.td - PDL Constraint ops ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATCH_IR_MATCHOPS
#define MLIR_DIALECT_MATCH_IR_MATCHOPS

include "mlir/Dialect/Match/IR/MatchTypes.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class
//===----------------------------------------------------------------------===//

class Match_Op<string mnemonic, list<Trait> traits = []>
    : Op<Match_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// match::MatcherOp
//===----------------------------------------------------------------------===//

def Match_MatcherOp : Match_Op<"matcher", [
    IsolatedFromAbove, SingleBlock, NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    Symbol
  ]> {
  let summary = "Container for the navigation/constraint IR of one matcher";
  let description = [{
    `match.matcher` wraps the navigation, constraint, and matcher-tree
    structure for one or more sub-patterns sharing a common matcher graph.
    Its region is a single block whose argument is the root operation.

    The body is a linear, ordered sequence of ops. IR position implicitly
    encodes AND: every test op above a given point must have succeeded for
    control to reach that point. The matcher-tree shape (the failure spine)
    is expressed by nested regions:

    * `match.try` opens a new failure scope. Failures inside (or
      fall-off-end) transfer control to the op immediately following the
      `match.try` in its parent region.
    * `match.switch_op_name` / `match.switch_type` are
      multi-way dispatches; each case region inherits the enclosing failure
      scope as its default.

    A successful sub-pattern match is marked by `match.success`, which
    may appear at any depth and carries its own benefit. A matcher therefore
    aggregates multiple logical patterns that share a common navigation /
    test prefix.

    Example:

    ```mlir
    match.matcher @combined root(%root : !pdl.operation) {
      match.has_name %root, "arith.addi"
      match.try {
        // alternative 1
        match.success @rewriters::@rewrite_a benefit(2)
      }
      match.try {
        // alternative 2
        match.success @rewriters::@rewrite_b benefit(1)
      }
    }
    ```
  }];

  let arguments = (ins OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    ($sym_name^)? `root` custom<MatcherRegion>($bodyRegion)
    attr-dict-with-keyword
  }];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A MatcherOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }
  }];
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// match::TryOp
//===----------------------------------------------------------------------===//

def Match_TryOp : Match_Op<"try", [
    SingleBlock, NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>
  ]> {
  let summary = "Open a new failure scope";
  let description = [{
    `match.try` introduces a new failure scope. Any test op that fails
    inside the region, as well as fall-off-end of the region, transfers
    control to the op immediately following the `match.try` in its
    parent region.

    This is the direct IR analogue of `MatcherNode::failureNode`: a chain of
    sibling `match.try` ops at the same region level encodes the OR of
    alternative match attempts.

    The region is a single block with no block arguments. Values defined in
    enclosing regions are visible inside; values defined inside the region
    are scoped to it.
  }];

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$body attr-dict";
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// match::SwitchOpNameOp
//===----------------------------------------------------------------------===//

def Match_SwitchOpNameOp : Match_Op<"switch_op_name", [
    NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>
  ]> {
  let summary = "Multi-way dispatch on operation name";
  let description = [{
    `match.switch_op_name` dispatches on the name of `op`. The op
    carries one case-region per name in `caseNames`. When the runtime name
    matches `caseNames[i]`, control enters the i-th case region. If no case
    matches, control transfers to the enclosing failure scope (the implicit
    default). Each case region is single-block and inherits the enclosing
    failure scope: failures inside also transfer to the enclosing failure
    scope.

    This represents `SwitchNode` directly. Without it, opcode dispatch must
    be expressed as a chain of equality `try` ops.

    Example:

    ```mlir
    match.switch_op_name %op
    case "arith.addi" {
      // ...
    }
    case "arith.subi" {
      // ...
    }
    ```
  }];

  let arguments = (ins PDL_Operation:$op, StrArrayAttr:$caseNames);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// match::SwitchTypeOp
//===----------------------------------------------------------------------===//

def Match_SwitchTypeOp : Match_Op<"switch_type", [
    NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>
  ]> {
  let summary = "Multi-way dispatch on a type value";
  let description = [{
    `match.switch_type` dispatches on the runtime value of a `!pdl.type`.
    The op carries one case-region per type in `caseTypes`. When the runtime
    type matches `caseTypes[i]`, control enters the i-th case region. If no
    case matches, control transfers to the enclosing failure scope. Each
    case region is single-block and inherits the enclosing failure scope.

    Example:

    ```mlir
    match.switch_type %t
    case i32 {
      // ...
    }
    case i64 {
      // ...
    }
    ```
  }];

  let arguments = (ins PDL_Type:$typeValue, TypeArrayAttr:$caseTypes);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Navigation Ops - Nullable (return !match.optional<T>)
//===----------------------------------------------------------------------===//

def Match_GetOperandOp : Match_Op<"get_operand", [Pure]> {
  let summary = "Get an operand of an operation (nullable)";
  let description = [{
    Navigates to a specific operand of an operation. Returns
    `!match.optional<!pdl.value>` because the operand may not exist.
  }];
  let arguments = (ins PDL_Operation:$op, I32Attr:$index);
  let results = (outs Match_OptionalType:$result);
  let assemblyFormat = "$index `of` $op attr-dict `:` type($result)";
}

def Match_GetOperandsOp : Match_Op<"get_operands", [Pure]> {
  let summary = "Get a group of operands of an operation (nullable)";
  let description = [{
    Navigates to a group of operands of an operation. Returns
    `!match.optional<!pdl.range<value>>`. If no index is provided,
    returns all operands.
  }];
  let arguments = (ins PDL_Operation:$op, OptionalAttr<I32Attr>:$index);
  let results = (outs Match_OptionalType:$result);
  let assemblyFormat = "($index^)? `of` $op attr-dict `:` type($result)";
}

def Match_GetResultOp : Match_Op<"get_result", [Pure]> {
  let summary = "Get a result of an operation (nullable)";
  let description = [{
    Navigates to a specific result of an operation. Returns
    `!match.optional<!pdl.value>` because the result may not exist.
  }];
  let arguments = (ins PDL_Operation:$op, I32Attr:$index);
  let results = (outs Match_OptionalType:$result);
  let assemblyFormat = "$index `of` $op attr-dict `:` type($result)";
}

def Match_GetResultsOp : Match_Op<"get_results", [Pure]> {
  let summary = "Get a group of results of an operation (nullable)";
  let description = [{
    Navigates to a group of results of an operation. Returns
    `!match.optional<!pdl.range<value>>`. If no index is provided,
    returns all results.
  }];
  let arguments = (ins PDL_Operation:$op, OptionalAttr<I32Attr>:$index);
  let results = (outs Match_OptionalType:$result);
  let assemblyFormat = "($index^)? `of` $op attr-dict `:` type($result)";
}

def Match_GetAttributeOp : Match_Op<"get_attribute", [Pure]> {
  let summary = "Get a named attribute of an operation (nullable)";
  let description = [{
    Navigates to a named attribute of an operation. Returns
    `!match.optional<!pdl.attribute>` because the attribute may not exist.
  }];
  let arguments = (ins PDL_Operation:$op, StrAttr:$name);
  let results = (outs Match_OptionalType:$result);
  let assemblyFormat = "$name `of` $op attr-dict `:` type($result)";
}

def Match_GetDefiningOpOp : Match_Op<"get_defining_op", [Pure]> {
  let summary = "Get the defining operation of a value or value range (nullable)";
  let description = [{
    Navigates to the defining operation of a value. If the input is a
    `!pdl.range<value>`, the defining op of the first value is used. Returns
    `!match.optional<!pdl.operation>` because the value may be a block
    argument with no defining op.
  }];
  let arguments = (ins PDL_InstOrRangeOf<PDL_Value>:$value);
  let results = (outs Match_OptionalType:$result);
  let assemblyFormat =
      "`of` $value `:` type($value) attr-dict `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Navigation Ops - Non-nullable (return bare PDL types)
//===----------------------------------------------------------------------===//

def Match_GetValueTypeOp : Match_Op<"get_value_type", [Pure]> {
  let summary = "Get the type of a value or value range (non-nullable)";
  let description = [{
    Gets the type of a value (or the types of a value range). Always succeeds
    because the input value has already been null-checked. If the input is a
    `!pdl.range<value>`, the result is a `!pdl.range<type>`.
  }];
  let arguments = (ins PDL_InstOrRangeOf<PDL_Value>:$value);
  let results = (outs PDL_InstOrRangeOf<PDL_Type>:$result);
  let assemblyFormat =
      "`of` $value `:` type($value) attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Match_GetAttributeTypeOp : Match_Op<"get_attribute_type", [Pure]> {
  let summary = "Get the type of an attribute (non-nullable)";
  let description = [{
    Gets the type of an attribute. Always succeeds because the input attribute
    has already been null-checked.
  }];
  let arguments = (ins PDL_Attribute:$attribute);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "`of` $attribute attr-dict `:` type($result)";
}

def Match_GetUsersOp : Match_Op<"get_users", [Pure]> {
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
// match::ExtractOp
//===----------------------------------------------------------------------===//

def Match_ExtractOp
    : Match_Op<"extract", [Pure,
     TypesMatchWith<
        "`range` is a PDL range whose element type matches type of `result`",
        "result", "range", "pdl::RangeType::get($_self)">]> {
  let summary = "Extract the item at the specified index in a range";
  let description = [{
    `match.extract` extracts an item from a range at the specified index,
    mirroring `pdl_interp.extract`. If the index is out of range, the result is
    null. It lowers directly to `pdl_interp.extract`.

    Example:

    ```mlir
    // Extract the value at index 0 from a range of values.
    %v = match.extract 0 of %values : !pdl.value
    ```
  }];

  let arguments = (ins PDL_RangeOf<PDL_AnyType>:$range,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$index);
  let results = (outs PDL_AnyType:$result);
  let assemblyFormat = "$index `of` $range `:` type($result) attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$range, "unsigned":$index), [{
      build($_builder, $_state,
            ::llvm::cast<pdl::RangeType>(range.getType()).getElementType(),
            range, index);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// match::GetEachOp
//===----------------------------------------------------------------------===//

def Match_GetEachOp : Match_Op<"get_each", [Pure]> {
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
// Constraint / Test Ops
//
// These ops have an implicit control effect: on failure, control transfers
// to the enclosing failure scope. They are not `Pure`; they must not be DCE'd
// even when they produce no SSA results.
//===----------------------------------------------------------------------===//

def Match_IsNotNullOp : Match_Op<"is_not_null"> {
  let summary = "Unwrap an optional value, failing if it is null";
  let description = [{
    Checks that an optional value is non-null and produces the bare inner
    value. If the optional is null, control transfers to the enclosing
    failure scope (the implicit predicate). Otherwise execution continues
    and the unwrapped value is in scope.

    By SSA dominance, the unwrapped result can only be used by ops below this
    one, which mirrors the control-flow guarantee in the lowered form.
  }];
  let arguments = (ins Match_OptionalType:$optionalValue);
  let results = (outs PDL_AnyType:$unwrapped);
  let assemblyFormat = [{
    $optionalValue `:` type($optionalValue) `->` type($unwrapped) attr-dict
  }];
  let hasVerifier = 1;
}

def Match_HasNameOp : Match_Op<"has_name"> {
  let summary = "Test the operation name";
  let description = [{
    Tests that `op` has the given name. On failure transfers control to the
    enclosing failure scope.
  }];
  let arguments = (ins PDL_Operation:$op, StrAttr:$name);
  let assemblyFormat = "$op `,` $name attr-dict";
}

def Match_EqualOp : Match_Op<"equal", [SameTypeOperands]> {
  let summary = "Test equality of two values";
  let description = [{
    Tests that `lhs` and `rhs` are equal at runtime. On failure transfers
    control to the enclosing failure scope.
  }];
  let arguments = (ins PDL_AnyType:$lhs, PDL_AnyType:$rhs);
  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) attr-dict";
}

def Match_HasTypeOp : Match_Op<"has_type"> {
  let summary = "Test that a type matches a constant type";
  let arguments = (ins PDL_Type:$typeValue, TypeAttr:$constantType);
  let assemblyFormat = "$typeValue `,` $constantType attr-dict";
}

def Match_HasTypesOp : Match_Op<"has_types"> {
  let summary = "Test that a range of types matches a constant array of types";
  let description = [{
    Tests that a `!pdl.range<type>` matches a statically known array of types.
  }];
  let arguments = (ins PDL_RangeOf<PDL_Type>:$types,
                       TypeArrayAttr:$constantTypes);
  let assemblyFormat = "$types `,` $constantTypes attr-dict";
}

def Match_HasAttrValueOp : Match_Op<"has_attr_value"> {
  let summary = "Test that an attribute matches a constant value";
  let arguments = (ins PDL_Attribute:$attribute, AnyAttr:$value);
  let assemblyFormat = "$attribute `is` $value attr-dict";
}

def Match_CheckOperandCountOp : Match_Op<"check_operand_count"> {
  let summary = "Test the operand count of an operation";
  let arguments = (ins PDL_Operation:$op,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$count,
                       UnitAttr:$atLeast);
  let assemblyFormat = [{
    $op `is` (`at_least` $atLeast^)? $count attr-dict
  }];
}

def Match_CheckResultCountOp : Match_Op<"check_result_count"> {
  let summary = "Test the result count of an operation";
  let arguments = (ins PDL_Operation:$op,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$count,
                       UnitAttr:$atLeast);
  let assemblyFormat = [{
    $op `is` (`at_least` $atLeast^)? $count attr-dict
  }];
}

def Match_ApplyNativeConstraintOp : Match_Op<"apply_native_constraint"> {
  let summary = "Apply a native constraint";
  let description = [{
    Applies an externally registered native constraint. On failure transfers
    control to the enclosing failure scope. The constraint may produce
    additional `!pdl.*` results that downstream ops consume.
  }];
  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_AnyType>:$args,
                       DefaultValuedAttr<BoolAttr, "false">:$isNegated);
  let results = (outs Variadic<PDL_AnyType>:$constraintResults);
  let assemblyFormat = [{
    $name `(` $args `:` type($args) `)` (`:` type($constraintResults)^)? attr-dict
  }];
  let hasVerifier = 1;
}
}

//===----------------------------------------------------------------------===//
// match::SuccessOp
//===----------------------------------------------------------------------===//

def Match_SuccessOp : Match_Op<"success", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Marks a successful pattern match";
  let description = [{
    `match.success` marks a successful pattern match. Its precondition
    is implicit: every test op above it in the same region, plus every test
    op in enclosing regions on the path from the matcher root to this op,
    must have succeeded.

    Carries a `benefit` (one matcher may hold sub-patterns of different
    benefits), a symbol reference to the rewriter, and the list of match
    values to forward. The signature mirrors `pdl_interp.record_match` so
    that the rewriter metadata passes through unchanged to the
    `match → pdl_interp` lowering, which emits the matching
    `pdl_interp.record_match`.

    Example:

    ```mlir
    match.success @rewriters::@rewriter
        benefit(1) (%root, %lhs : !pdl.operation, !pdl.value)
    ```
  }];

  let arguments = (ins SymbolRefAttr:$rewriter,
                       ConfinedAttr<I16Attr, [IntNonNegative]>:$benefit,
                       Variadic<PDL_AnyType>:$inputs);
  let assemblyFormat = [{
    $rewriter `benefit` `(` $benefit `)`
    (`(` $inputs^ `:` type($inputs) `)`)? attr-dict
  }];
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_MATCH_IR_MATCHOPS
