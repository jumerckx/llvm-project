//===- MatchToCpp.h - Emit C++ matchers from match --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MATCHTOCPP_MATCHTOCPP_H
#define MLIR_TARGET_MATCHTOCPP_MATCHTOCPP_H

#include "mlir/Support/LLVM.h"
#include "mlir/Target/MatchToCpp/OpInfoRegistry.h"

namespace mlir {
class Operation;
namespace match {
LogicalResult translateToCpp(Operation *op, raw_ostream &os,
                             const OpInfoRegistry &registry);
LogicalResult translateToCpp(Operation *op, raw_ostream &os);

} // namespace match
} // namespace mlir

#endif // MLIR_TARGET_MATCHTOCPP_MATCHTOCPP_H
