//===- TranslateRegistration.cpp - Register match-to-cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Target/MatchToCpp/MatchToCpp.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerMatchToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "match-to-cpp",
      "translate match matchers to C++ RewritePatterns",
      [](Operation *op, raw_ostream &output) {
        return match::translateToCpp(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<pdl::PDLDialect, match::MatchDialect,
                        pdl_interp::PDLInterpDialect>();
      });
}

} // namespace mlir
