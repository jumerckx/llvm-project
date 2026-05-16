//===- PDLConstr.cpp - PDL Constraint Dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrTypes.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pdl_constr;

#include "mlir/Dialect/PDLConstr/IR/PDLConstrOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PDLConstrDialect
//===----------------------------------------------------------------------===//

void PDLConstrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOpsTypes.cpp.inc"

void PDLConstrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// OptionalType
//===----------------------------------------------------------------------===//

LogicalResult
OptionalType::verify(function_ref<InFlightDiagnostic()> emitError,
                     Type innerType) {
  if (!llvm::isa<pdl::PDLType>(innerType)) {
    return emitError()
           << "expected inner type of pdl_constr.optional to be a PDL type "
              "(one of [!pdl.attribute, !pdl.operation, !pdl.type, "
              "!pdl.value, !pdl.range<...>]), but got "
           << innerType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PatternOp
//===----------------------------------------------------------------------===//

void PatternOp::build(OpBuilder &builder, OperationState &state,
                      IntegerAttr benefit, StringAttr symName) {
  state.addAttribute("benefit", benefit);
  if (symName)
    state.addAttribute(SymbolTable::getSymbolAttrName(), symName);
  state.addRegion(); // bodyRegion
  state.addRegion(); // rewriteRegion
}

SuccessOp PatternOp::getSuccessOp() {
  return cast<SuccessOp>(getBodyRegion().front().getTerminator());
}

/// Parse: `(%arg : type) { body }`
static ParseResult parsePatternBodyRegion(OpAsmParser &parser,
                                          Region &region) {
  OpAsmParser::Argument arg;
  if (parser.parseLParen() || parser.parseArgument(arg, /*allowType=*/true) ||
      parser.parseRParen())
    return failure();
  return parser.parseRegion(region, {arg});
}

/// Parse: `(%arg0 : type0, %arg1 : type1, ...) { body }`
/// or an empty rewrite region if not present.
static ParseResult parseRewriteRegion(OpAsmParser &parser, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseLParen())
    return failure();
  if (parser.parseOptionalRParen()) {
    // Parse argument list.
    do {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg, /*allowType=*/true))
        return failure();
      args.push_back(arg);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  return parser.parseRegion(region, args);
}

/// Custom assembly format:
///   @sym_name? `:` `benefit` `(` $benefit `)` `root` body_region
///   (`rewrite` rewrite_region)?
///   (`with` $externalRewriteName
///     (`(` $externalRewriteArgTypes `)`)? )?
///   attr-dict-with-keyword
ParseResult PatternOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symName;
  (void)parser.parseOptionalSymbolName(symName, SymbolTable::getSymbolAttrName(),
                                       result.attributes);

  if (parser.parseColon() || parser.parseKeyword("benefit") ||
      parser.parseLParen())
    return failure();

  IntegerAttr benefitAttr;
  if (parser.parseAttribute(benefitAttr, "benefit", result.attributes))
    return failure();
  if (parser.parseRParen() || parser.parseKeyword("root"))
    return failure();

  // Parse body region.
  Region *bodyRegion = result.addRegion();
  if (parsePatternBodyRegion(parser, *bodyRegion))
    return failure();

  // Parse optional rewrite region.
  Region *rewriteRegion = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("rewrite"))) {
    if (parseRewriteRegion(parser, *rewriteRegion))
      return failure();
  }

  // Parse optional external rewrite.
  if (succeeded(parser.parseOptionalKeyword("with"))) {
    StringAttr rewriteName;
    if (parser.parseAttribute(rewriteName, "externalRewriteName",
                              result.attributes))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void PatternOp::print(OpAsmPrinter &p) {
  if (auto sym = getSymNameAttr()) {
    p << ' ';
    p.printSymbolName(sym);
  }
  p << " : benefit(" << getBenefit() << ") root";

  // Print body region.
  Region &body = getBodyRegion();
  p << "(";
  if (!body.empty() && body.front().getNumArguments() > 0) {
    BlockArgument arg = body.front().getArgument(0);
    p.printRegionArgument(arg);
  }
  p << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false);

  // Print rewrite region if non-empty.
  Region &rewrite = getRewriteRegion();
  if (!rewrite.empty()) {
    p << " rewrite(";
    Block &rewriteBlock = rewrite.front();
    llvm::interleaveComma(rewriteBlock.getArguments(), p,
                          [&](BlockArgument arg) {
                            p.printRegionArgument(arg);
                          });
    p << ") ";
    p.printRegion(rewrite, /*printEntryBlockArgs=*/false);
  }

  // Print external rewrite name.
  if (auto name = getExternalRewriteNameAttr()) {
    p << " with ";
    p.printAttribute(name);
  }

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"benefit", "sym_name", "externalRewriteName",
                       "externalRewriteArgTypes"});
}

LogicalResult PatternOp::verifyRegions() {
  Region &body = getBodyRegion();
  if (body.empty())
    return emitOpError("expected non-empty body region");

  Block &block = body.front();

  // Verify the block has exactly one argument of type !pdl.operation.
  if (block.getNumArguments() != 1)
    return emitOpError("expected body block to have exactly one argument");

  if (!llvm::isa<pdl::OperationType>(block.getArgument(0).getType()))
    return emitOpError(
        "expected body block argument to be of type !pdl.operation");

  // Verify the block is terminated by pdl_constr.success.
  if (block.empty() || !llvm::isa<SuccessOp>(block.back()))
    return emitOpError("expected body to terminate with `pdl_constr.success`");

  // Verify rewrite region if present.
  Region &rewrite = getRewriteRegion();
  if (!rewrite.empty()) {
    // The rewrite region must have a single block.
    if (!rewrite.hasOneBlock())
      return emitOpError("expected rewrite region to have a single block");

    // Verify that the number of rewrite_values on the success op matches the
    // number of rewrite region block arguments.
    auto successOp = getSuccessOp();
    unsigned numRewriteValues = successOp.getRewriteValues().size();
    unsigned numRewriteArgs = rewrite.front().getNumArguments();
    if (numRewriteValues != numRewriteArgs)
      return emitOpError("expected ")
             << numRewriteArgs
             << " rewrite_values on success op to match rewrite region block "
                "arguments, but got "
             << numRewriteValues;
  }

  return success();
}

StringRef PatternOp::getDefaultDialect() {
  return PDLConstrDialect::getDialectNamespace();
}

//===----------------------------------------------------------------------===//
// GetEachOp
//===----------------------------------------------------------------------===//

LogicalResult GetEachOp::verify() {
  auto rangeType = llvm::dyn_cast<pdl::RangeType>(getRange().getType());
  if (!rangeType)
    return emitOpError("expected input to be a !pdl.range type");

  if (rangeType.getElementType() != getResult().getType())
    return emitOpError("expected result type to match the element type of the "
                       "input range, got ")
           << getResult().getType() << " but expected "
           << rangeType.getElementType();

  return success();
}

//===----------------------------------------------------------------------===//
// IsNotNullOp
//===----------------------------------------------------------------------===//

LogicalResult IsNotNullOp::verify() {
  auto optType =
      llvm::cast<OptionalType>(getOptionalValue().getType());
  if (optType.getInnerType() != getUnwrapped().getType())
    return emitOpError("expected unwrapped result type ")
           << getUnwrapped().getType() << " to match inner type of optional "
           << optType.getInnerType();
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.cpp.inc"
