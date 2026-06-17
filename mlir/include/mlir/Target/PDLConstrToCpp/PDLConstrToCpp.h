//===- PDLConstrToCpp.h - Emit C++ matchers from pdl_constr -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to emit readable C++ `RewritePattern` matchers
// from the `pdl_constr` dialect. Only the matcher side is emitted; each
// `pdl_constr.success` lowers to a call to an `extern` rewrite function that the
// user supplies. The generated code integrates with `applyPatternsGreedily` via
// a generated `populateGeneratedPatterns(RewritePatternSet &)` entry point.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_PDLCONSTRTOCPP_PDLCONSTRTOCPP_H
#define MLIR_TARGET_PDLCONSTRTOCPP_PDLCONSTRTOCPP_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
namespace pdl_constr {

/// Translate every `pdl_constr.matcher` nested under `op` into a C++
/// `RewritePattern` and emit the source to `os`. Returns failure if any matcher
/// uses a construct that cannot yet be emitted.
LogicalResult translateToCpp(Operation *op, raw_ostream &os);

} // namespace pdl_constr
} // namespace mlir

#endif // MLIR_TARGET_PDLCONSTRTOCPP_PDLCONSTRTOCPP_H
