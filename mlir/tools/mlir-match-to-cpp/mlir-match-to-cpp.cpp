//===- mlir-match-to-cpp.cpp - Emit C++ matchers from `match` IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToMatch/PDLToMatch.h"
#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/Match/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
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

enum class Combine { None, Root, All };

template <typename RangeT>
static int fixedFlatCount(RangeT &&groups, int count, bool attrSizedSegments) {
  if (attrSizedSegments)
    return -1;
  for (const auto &group : groups)
    if (group.isVariableLength())
      return -1;
  return count;
}

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
      std::string qualClass = op.getQualCppClassName();
      if (!StringRef(qualClass).starts_with("::"))
        qualClass = "::" + qualClass;
      info.cppClassName = std::move(qualClass);
      info.numOperandGroups = op.getNumOperands();
      info.numResultGroups = op.getNumResults();
      info.attrSizedOperandSegments = attrSizedOperands;
      info.attrSizedResultSegments = attrSizedResults;
      info.fixedNumOperands = fixedFlatCount(
          op.getOperands(), op.getNumOperands(), attrSizedOperands);
      info.fixedNumResults =
          fixedFlatCount(op.getResults(), op.getNumResults(), attrSizedResults);
      info.inferResultTypes =
          op.getTrait("::mlir::InferTypeOpInterface::Trait") != nullptr;
      for (const tblgen::NamedAttribute &attr : op.getAttributes())
        if (!attr.attr.isDerivedAttr())
          info.attrGetters[attr.name] = op.getGetterName(attr.name);

      registry.insert(op.getOperationName(), std::move(info));
    }
  }
  return success();
}

int main(int argc, char **argv) {
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

  llvm::cl::opt<std::string> cppNamespace(
      "namespace",
      llvm::cl::desc("C++ namespace to emit the patterns and rewriters in"),
      llvm::cl::init(""));

  llvm::cl::opt<Combine> combine(
      "combine-matchers",
      llvm::cl::desc("How to combine the matchers of PDL input"),
      llvm::cl::values(
          clEnumValN(Combine::None, "none", "Emit one pattern per PDL pattern"),
          clEnumValN(Combine::Root, "root",
                     "Emit one pattern per root operation name"),
          clEnumValN(Combine::All, "all", "Emit a single pattern")),
      llvm::cl::init(Combine::Root));

  llvm::cl::list<std::string> macroNames(
      "D",
      llvm::cl::desc("Name of the macro to be defined for ODS `.td` parsing"),
      llvm::cl::value_desc("macro name"), llvm::cl::Prefix);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Translate `match` IR to C++ RewritePatterns\n");

  match::OpInfoRegistry registry;
  if (failed(buildRegistry(odsFiles, includeDirs, registry)))
    return 1;

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<pdl::PDLDialect, match::MatchDialect>();

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), SMLoc());
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, ParserConfig(&context));
  if (!module)
    return 1;

  if (!module->getOps<pdl::PatternOp>().empty()) {
    auto canInferResultTypes = [&](StringRef name) {
      const match::OpInfo *info = registry.lookup(name);
      return info && info->inferResultTypes;
    };
    if (failed(convertPDLToMatch(*module, /*configMap=*/nullptr,
                                 canInferResultTypes)))
      return 1;
    if (combine != Combine::None &&
        failed(
            match::combineMatchers(*module,
                                   /*groupByRoot=*/combine == Combine::Root)))
      return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> output =
      openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(match::translateToCpp(module->getOperation(), output->os(),
                                   registry, cppNamespace)))
    return 1;

  output->keep();
  return 0;
}
