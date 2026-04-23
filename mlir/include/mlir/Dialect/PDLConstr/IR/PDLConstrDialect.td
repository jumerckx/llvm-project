//===- PDLConstrDialect.td - PDL Constraint dialect --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTRDIALECT
#define MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTRDIALECT

include "mlir/IR/OpBase.td"

def PDLConstr_Dialect : Dialect {
  let summary = "Intermediate constraint dialect between PDL and PDL_Interp";
  let description = [{
    The `pdl_constr` dialect provides an intermediate representation between
    `pdl` and `pdl_interp`. It makes pattern constraints, navigation, and
    predicate logic explicit in MLIR IR, enabling inspection, transformation,
    and composition before the final lowering to `pdl_interp`.

    Constraint ops return `!pdl_constr.pred` values (a boolean-like predicate
    result type). These are combined with `all` / `any` combinators and
    consumed by `success`. Navigation ops that can fail return
    `!pdl_constr.optional<T>` which must be unwrapped via `is_not_null`.
  }];

  let name = "pdl_constr";
  let cppNamespace = "::mlir::pdl_constr";
  let dependentDialects = ["pdl::PDLDialect"];

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // MLIR_DIALECT_PDLCONSTR_IR_PDLCONSTRDIALECT
