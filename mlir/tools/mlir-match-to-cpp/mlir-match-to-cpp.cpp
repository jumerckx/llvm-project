//===- mlir-match-to-cpp.cpp - Emit C++ matchers from `match` IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Translates `match` IR into C++ `RewritePattern`s. The tool can be pointed at
// a dialect's ODS `.td` definitions (`--ods`); for every op whose name appears
// in those definitions it emits a concrete-typed, DRR-style matcher (`dyn_cast`
// to the op class, with provably-redundant null / bounds / count checks
// elided). The op-name -> C++-class mapping and the structural metadata both
// come straight from `mlir::tblgen::Operator`, the same data DRR consumes at
// TableGen time. Without `--ods`, every op is matched generically (`Operation
// *`, name comparisons, runtime checks).
//
// The generated file references the concrete op classes by name but does not
// `#include` their dialect headers -- as with DRR-generated `.inc` files, the
// includer is responsible for providing them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Target/MatchToCpp/MatchToCpp.h"
#include "mlir/Target/MatchToCpp/OpInfoRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TableGen/Parser.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

/// Returns the statically fixed flat count for a list of operand/result
/// *groups*, or -1 when any group is variable-length or the op carries the
/// matching segment-size trait (in which case the flat count is only known at
/// runtime). When all groups are single and there is no segment attribute, the
/// flat count equals the number of groups.
template <typename RangeT>
static int fixedFlatCount(RangeT &&groups, int count, bool attrSizedSegments) {
  if (attrSizedSegments)
    return -1;
  for (const auto &group : groups)
    if (group.isVariableLength())
      return -1;
  return count;
}

/// Parses each `--ods` `.td` file and records, for every op it defines, the
/// static ODS structure the emitter needs to produce concrete-typed matchers.
static LogicalResult buildRegistry(ArrayRef<std::string> odsFiles,
                                   const std::vector<std::string> &includeDirs,
                                   match::OpInfoRegistry &registry) {
  for (StringRef odsFile : odsFiles) {
    std::string errorMessage;
    std::unique_ptr<llvm::MemoryBuffer> buffer =
        openInputFile(odsFile, &errorMessage);
    if (!buffer) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }

    llvm::SourceMgr tdSrcMgr;
    tdSrcMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());
    tdSrcMgr.setIncludeDirs(includeDirs);
    tdSrcMgr.setVirtualFileSystem(llvm::vfs::getRealFileSystem());

    llvm::RecordKeeper records;
    if (llvm::TableGenParseFile(tdSrcMgr, records)) {
      llvm::errs() << "error: failed to parse ODS file '" << odsFile << "'\n";
      return failure();
    }

    for (const llvm::Record *def : records.getAllDerivedDefinitions("Op")) {
      tblgen::Operator op(def);

      bool attrSizedOperands =
          op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments");
      bool attrSizedResults =
          op.getTrait("::mlir::OpTrait::AttrSizedResultSegments");

      match::OpInfo info;
      // Spell the class fully-qualified from the global namespace. Some
      // dialects already prefix their cppNamespace with "::".
      std::string qualClass = op.getQualCppClassName();
      if (!StringRef(qualClass).starts_with("::"))
        qualClass = "::" + qualClass;
      info.cppClassName = std::move(qualClass);
      info.numOperandGroups = op.getNumOperands();
      info.numResultGroups = op.getNumResults();
      info.attrSizedOperandSegments = attrSizedOperands;
      info.attrSizedResultSegments = attrSizedResults;
      info.fixedNumOperands = fixedFlatCount(op.getOperands(),
                                             op.getNumOperands(),
                                             attrSizedOperands);
      info.fixedNumResults = fixedFlatCount(op.getResults(),
                                            op.getNumResults(),
                                            attrSizedResults);

      registry.insert(op.getOperationName(), std::move(info));
    }
  }
  return success();
}

int main(int argc, char **argv) {
  // FIXME: Necessary because we link in TableGen, which defines its options as
  // static variables -- some of which overlap with ours. Mirrors mlir-pdll.
  llvm::cl::ResetCommandLineParser();

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input match file>"),
      llvm::cl::init("-"), llvm::cl::value_desc("filename"));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::list<std::string> includeDirs(
      "I", llvm::cl::desc("Directory of include files for the ODS `.td`"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  llvm::cl::list<std::string> odsFiles(
      "ods",
      llvm::cl::desc("ODS `.td` file(s) whose ops should be matched with their "
                     "concrete C++ class (DRR-style). May be repeated."),
      llvm::cl::value_desc("filename"));

  // `ResetCommandLineParser` above unregistered TableGen's `-D` option, which
  // otherwise makes `.td` parsing fail on a macro define. Re-register it.
  llvm::cl::list<std::string> macroNames(
      "D",
      llvm::cl::desc("Name of the macro to be defined for ODS `.td` parsing"),
      llvm::cl::value_desc("macro name"), llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Translate `match` IR to C++ RewritePatterns\n");

  // Build the concrete-op registry from the supplied ODS definitions. With no
  // `--ods`, the registry is empty and the output is fully generic.
  match::OpInfoRegistry registry;
  if (failed(buildRegistry(odsFiles, includeDirs, registry)))
    return 1;

  // Parse the matcher IR.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<pdl::PDLDialect, match::MatchDialect,
                      pdl_interp::PDLInterpDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, ParserConfig(&context));
  if (!module)
    return 1;

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(match::translateToCpp(module->getOperation(), output->os(),
                                   registry)))
    return 1;

  output->keep();
  return 0;
}
