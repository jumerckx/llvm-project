//===- TranslateRegistration.cpp - Register pdl-constr-to-cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Target/PDLConstrToCpp/PDLConstrToCpp.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerPDLConstrToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "pdl-constr-to-cpp",
      "translate pdl_constr matchers to C++ RewritePatterns",
      [](Operation *op, raw_ostream &output) {
        return pdl_constr::translateToCpp(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<pdl::PDLDialect, pdl_constr::PDLConstrDialect,
                        pdl_interp::PDLInterpDialect>();
      });
}

} // namespace mlir
