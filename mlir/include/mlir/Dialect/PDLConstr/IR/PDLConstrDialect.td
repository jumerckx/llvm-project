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
    `pdl` and `pdl_interp`. It makes pattern constraints, navigation, and the
    matcher-tree structure explicit in MLIR IR, enabling inspection,
    transformation, and composition before the final lowering to `pdl_interp`.

    The dialect is **ordered**: IR position determines materialization order,
    and the matcher-tree structure (the failure spine) is expressed directly
    via nested regions (`pdl_constr.try`, `pdl_constr.switch_*`). Test ops
    have an implicit control effect — on failure, control transfers to the
    enclosing failure scope. There is no SSA predicate type; the predicate is
    the control-flow position.

    Navigation ops that can fail return `!pdl_constr.optional<T>`. The only
    way to obtain the bare value is via `pdl_constr.is_not_null`, which
    additionally has the implicit control effect of branching to the failure
    scope when the optional is null.
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
