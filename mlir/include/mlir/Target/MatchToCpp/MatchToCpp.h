//===- MatchToCpp.h - Emit C++ matchers from match -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to emit readable C++ `RewritePattern` matchers
// from the `match` dialect. Only the matcher side is emitted; each
// `match.success` lowers to a call to an `extern` rewrite function that the
// user supplies. The generated code integrates with `applyPatternsGreedily` via
// a generated `populateGeneratedPatterns(RewritePatternSet &)` entry point.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MATCHTOCPP_MATCHTOCPP_H
#define MLIR_TARGET_MATCHTOCPP_MATCHTOCPP_H

#include "mlir/Support/LLVM.h"
#include "mlir/Target/MatchToCpp/OpInfoRegistry.h"

namespace mlir {
class Operation;
namespace match {

/// Translate every `match.matcher` nested under `op` into a C++
/// `RewritePattern` and emit the source to `os`. Returns failure if any matcher
/// uses a construct that cannot yet be emitted.
///
/// `registry` supplies static ODS metadata for ops whose name is statically
/// known in the matcher; those ops are emitted as concrete-typed (`dyn_cast`)
/// matches. Ops absent from the registry use the generic `Operation *`
/// emission. The default registry is empty (fully generic output).
LogicalResult translateToCpp(Operation *op, raw_ostream &os,
                             const OpInfoRegistry &registry);
LogicalResult translateToCpp(Operation *op, raw_ostream &os);

} // namespace match
} // namespace mlir

#endif // MLIR_TARGET_MATCHTOCPP_MATCHTOCPP_H
