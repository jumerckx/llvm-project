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

/// Custom parser for the pattern region: parses `(%arg : type) { body }`.
static ParseResult parsePatternRegion(OpAsmParser &parser, Region &region) {
  OpAsmParser::Argument arg;
  if (parser.parseLParen() || parser.parseArgument(arg, /*allowType=*/true) ||
      parser.parseRParen())
    return failure();

  return parser.parseRegion(region, {arg});
}

/// Custom printer for the pattern region.
static void printPatternRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p << "(";
  if (!region.empty() && region.front().getNumArguments() > 0) {
    BlockArgument arg = region.front().getArgument(0);
    p.printRegionArgument(arg);
  }
  p << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
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
