//===- PDLConstrTypes.td - PDL Constraint types ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTRTYPES
#define MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTRTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/PDLConstr/IR/PDLConstrDialect.td"

class PDLConstr_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PDLConstr_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// pdl_constr::OptionalType
//===----------------------------------------------------------------------===//

def PDLConstr_Optional : PDLConstr_Type<"Optional", "optional"> {
  let summary = "Optional wrapper for a PDL type";
  let description = [{
    A wrapper type indicating that the contained value may be null at runtime.
    Navigation ops that can fail return `!pdl_constr.optional<!pdl.operation>`
    etc. The only way to convert from `!pdl_constr.optional<T>` to `T` is via
    `pdl_constr.is_not_null`, which additionally has the implicit control
    effect of failing the surrounding scope when the optional is null.
  }];
  let parameters = (ins "Type":$innerType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$innerType), [{
      return $_get(innerType.getContext(), innerType);
    }]>,
  ];
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $innerType `>`";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def PDLConstr_OptionalType : Type<
  CPred<"::llvm::isa<::mlir::pdl_constr::OptionalType>($_self)">,
  "pdl_constr optional type",
  "::mlir::pdl_constr::OptionalType">;

#endif // MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTRTYPES
