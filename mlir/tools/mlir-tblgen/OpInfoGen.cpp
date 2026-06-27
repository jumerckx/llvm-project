//===- OpInfoGen.cpp - Op metadata table generator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits a `populate<Name>OpInfo(::mlir::match::OpInfoRegistry &)` function that
// records, for each op in a dialect, the static ODS structure the
// `match-to-cpp` translator needs to emit concrete-typed (`dyn_cast`) matchers.
// See mlir/include/mlir/Target/MatchToCpp/OpInfoRegistry.h.
//
//===----------------------------------------------------------------------===//

#include "OpGenHelpers.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

static llvm::cl::OptionCategory opInfoCat("Options for -gen-op-info");

static llvm::cl::opt<std::string> funcName(
    "op-info-func-name",
    llvm::cl::desc("Name of the generated populate function"),
    llvm::cl::init("populateOpInfo"), llvm::cl::cat(opInfoCat));

static llvm::cl::opt<std::string> headerInclude(
    "op-info-include",
    llvm::cl::desc("Header the generated matcher must #include to see the op "
                   "classes (e.g. mlir/Dialect/Arith/IR/Arith.h)"),
    llvm::cl::init(""), llvm::cl::cat(opInfoCat));

/// Returns the statically fixed flat count for a list of `count` operand/result
/// groups, or -1 when any group is variable-length or the op carries the given
/// segment-size trait (in which case the flat count is only known at runtime).
template <typename RangeT>
static int fixedFlatCount(const Operator &op, RangeT &&groups, int count,
                          bool attrSizedSegments) {
  if (attrSizedSegments)
    return -1;
  for (const auto &group : groups)
    if (group.isVariableLength())
      return -1;
  return count;
}

static bool emitOpInfo(const RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Op metadata table for match-to-cpp", os, records);

  os << "void " << funcName
     << "(::mlir::match::OpInfoRegistry &registry) {\n";

  for (const Record *def : getRequestedOpDefinitions(records)) {
    Operator op(def);

    bool attrSizedOperands =
        op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
    bool attrSizedResults =
        op.getTrait("::mlir::OpTrait::AttrSizedResultSegments");

    int fixedOperands = fixedFlatCount(op, op.getOperands(), op.getNumOperands(),
                                       attrSizedOperands);
    int fixedResults = fixedFlatCount(op, op.getResults(), op.getNumResults(),
                                      attrSizedResults);

    // Spell the class fully-qualified from the global namespace. Some dialects
    // already prefix their cppNamespace with "::", so only add it if missing.
    std::string qualClass = op.getQualCppClassName();
    if (!StringRef(qualClass).starts_with("::"))
      qualClass = "::" + qualClass;

    os << "  registry.insert(\"" << op.getOperationName()
       << "\", ::mlir::match::OpInfo{";
    os << "\"" << qualClass << "\", ";
    os << "\"" << headerInclude << "\", ";
    os << op.getNumOperands() << ", ";
    os << op.getNumResults() << ", ";
    os << (attrSizedOperands ? "true" : "false") << ", ";
    os << (attrSizedResults ? "true" : "false") << ", ";
    os << fixedOperands << ", ";
    os << fixedResults << "});\n";
  }

  os << "}\n";
  return false;
}

static mlir::GenRegistration
    genOpInfo("gen-op-info",
              "Generate an op metadata table for match-to-cpp",
              [](const RecordKeeper &records, raw_ostream &os) {
                return emitOpInfo(records, os);
              });
