//===- OpInfoRegistry.h - Static op metadata for match-to-cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The `match` IR only carries operation *names* (`"arith.addf"`), never the
// concrete C++ op class. To emit DRR-style matchers -- a `dyn_cast` to the
// concrete op followed by typed, null-check-free navigation -- the emitter
// needs the static structure that lives only in ODS. `OpInfoRegistry` makes
// that ODS metadata available at translate time, keyed by op name.
//
// The registry is populated by the `mlir-match-to-cpp` tool directly from a
// dialect's ODS `.td` definitions (via `mlir::tblgen::Operator`), the same data
// DRR consumes at TableGen time. Op names absent from the registry fall back to
// the generic `Operation *` emission, so the registry is purely additive: an
// empty registry reproduces the fully generic output.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MATCHTOCPP_OPINFOREGISTRY_H
#define MLIR_TARGET_MATCHTOCPP_OPINFOREGISTRY_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir {
namespace match {

/// Static, ODS-derived metadata for a single operation, enough to emit a
/// concrete-typed matcher for it. Owns its strings, so an `OpInfo` may be
/// copied and stored freely.
struct OpInfo {
  /// Fully-qualified C++ class name, e.g. "::mlir::arith::AddFOp".
  std::string cppClassName;
  /// Number of ODS operand / result *groups* (variadic-aware).
  unsigned numOperandGroups = 0;
  unsigned numResultGroups = 0;
  /// Whether the op carries the corresponding segment-size attribute.
  bool attrSizedOperandSegments = false;
  bool attrSizedResultSegments = false;
  /// Statically fixed flat operand / result count, or -1 when the count is not
  /// fixed (any variadic/optional group, or AttrSized*Segments). When fixed,
  /// flat `get_operand`/`get_result` navigation below the bound is provably
  /// in-range and the matching `check_*_count` test is provably satisfied.
  int fixedNumOperands = -1;
  int fixedNumResults = -1;

  std::optional<unsigned> getFixedNumOperands() const {
    if (fixedNumOperands < 0)
      return std::nullopt;
    return static_cast<unsigned>(fixedNumOperands);
  }
  std::optional<unsigned> getFixedNumResults() const {
    if (fixedNumResults < 0)
      return std::nullopt;
    return static_cast<unsigned>(fixedNumResults);
  }
};

/// A name-keyed table of `OpInfo`. An empty registry reproduces the historical
/// generic emission exactly.
class OpInfoRegistry {
public:
  /// Returns the metadata for `opName`, or null if the op is unknown.
  const OpInfo *lookup(llvm::StringRef opName) const {
    auto it = table.find(opName);
    return it == table.end() ? nullptr : &it->second;
  }

  /// Registers metadata for `opName` (last writer wins).
  void insert(llvm::StringRef opName, OpInfo info) {
    table[opName] = std::move(info);
  }

  bool empty() const { return table.empty(); }

private:
  llvm::StringMap<OpInfo> table;
};

} // namespace match
} // namespace mlir

#endif // MLIR_TARGET_MATCHTOCPP_OPINFOREGISTRY_H
