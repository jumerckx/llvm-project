//===- OpInfoRegistry.h - Static op metadata for match-to-cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
struct OpInfo {
  std::string cppClassName;
  unsigned numOperandGroups = 0;
  unsigned numResultGroups = 0;
  bool attrSizedOperandSegments = false;
  bool attrSizedResultSegments = false;
  int fixedNumOperands = -1;
  int fixedNumResults = -1;
  bool inferResultTypes = false;
  llvm::StringMap<std::string> attrGetters;

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
  const std::string *lookupAttrGetter(llvm::StringRef attrName) const {
    auto it = attrGetters.find(attrName);
    return it == attrGetters.end() ? nullptr : &it->second;
  }
};

class OpInfoRegistry {
public:
  const OpInfo *lookup(llvm::StringRef opName) const {
    auto it = table.find(opName);
    return it == table.end() ? nullptr : &it->second;
  }

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
