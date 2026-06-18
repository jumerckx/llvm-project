//===- Match.cpp - PDL Constraint Dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Match/IR/MatchOps.h"
#include "mlir/Dialect/Match/IR/MatchTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::match;

#include "mlir/Dialect/Match/IR/MatchOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MatchDialect
//===----------------------------------------------------------------------===//

void MatchDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Match/IR/MatchOps.cpp.inc"
      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Match/IR/MatchOpsTypes.cpp.inc"

void MatchDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Match/IR/MatchOpsTypes.cpp.inc"
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
           << "expected inner type of match.optional to be a PDL type "
              "(one of [!pdl.attribute, !pdl.operation, !pdl.type, "
              "!pdl.value, !pdl.range<...>]), but got "
           << innerType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatcherOp
//===----------------------------------------------------------------------===//

/// Custom parser for the matcher region: parses `(%arg : type) { body }`.
static ParseResult parseMatcherRegion(OpAsmParser &parser, Region &region) {
  OpAsmParser::Argument arg;
  if (parser.parseLParen() || parser.parseArgument(arg, /*allowType=*/true) ||
      parser.parseRParen())
    return failure();

  return parser.parseRegion(region, {arg});
}

/// Custom printer for the matcher region.
static void printMatcherRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p << "(";
  if (!region.empty() && region.front().getNumArguments() > 0) {
    BlockArgument arg = region.front().getArgument(0);
    p.printRegionArgument(arg);
  }
  p << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

/// Returns true if `region` (or any nested region) contains a SuccessOp.
static bool regionContainsSuccess(Region &region) {
  WalkResult result = region.walk([&](SuccessOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

LogicalResult MatcherOp::verifyRegions() {
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

  // Verify the matcher contains at least one `match.success` op
  // somewhere in its region tree.
  if (!regionContainsSuccess(body))
    return emitOpError("expected matcher to contain at least one "
                       "`match.success` (directly or transitively)");

  return success();
}

StringRef MatcherOp::getDefaultDialect() {
  return MatchDialect::getDialectNamespace();
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

LogicalResult TryOp::verifyRegions() {
  // A `try` region with no transitive success op is dead; reject it so that
  // the IR is meaningful.
  if (!regionContainsSuccess(getBody()))
    return emitOpError("`match.try` region contains no "
                       "`match.success` (directly or transitively); the "
                       "region is dead");
  return success();
}

StringRef TryOp::getDefaultDialect() {
  return MatchDialect::getDialectNamespace();
}

//===----------------------------------------------------------------------===//
// SwitchOpNameOp / SwitchTypeOp helpers
//===----------------------------------------------------------------------===//

namespace {

/// Parse a sequence of `case <case-spec> { region }` clauses. `parseCase` is
/// invoked once per clause to parse the case-specific bit (a string for
/// op-name switches, a type for type switches). Returns the parsed cases as
/// an array attribute (built from the per-case parsed attrs) along with the
/// per-case regions.
template <typename ParseCaseFn>
ParseResult
parseSwitchCases(OpAsmParser &parser, ArrayAttr &caseAttr,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions,
                 ParseCaseFn parseCase) {
  SmallVector<Attribute> caseAttrs;
  while (succeeded(parser.parseOptionalKeyword("case"))) {
    Attribute caseValue;
    if (failed(parseCase(parser, caseValue)))
      return failure();
    caseAttrs.push_back(caseValue);

    auto region = std::make_unique<Region>();
    if (parser.parseRegion(*region, /*arguments=*/{}))
      return failure();
    caseRegions.push_back(std::move(region));
  }
  caseAttr = parser.getBuilder().getArrayAttr(caseAttrs);
  return success();
}

/// Print a sequence of `case <case-spec> { region }` clauses.
template <typename PrintCaseFn>
void printSwitchCases(OpAsmPrinter &p, ArrayAttr caseAttr,
                      MutableArrayRef<Region> caseRegions,
                      PrintCaseFn printCase) {
  for (auto [attr, region] : llvm::zip(caseAttr, caseRegions)) {
    p.printNewline();
    p << "case ";
    printCase(p, attr);
    p << ' ';
    p.printRegion(region, /*printEntryBlockArgs=*/false);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// SwitchOpNameOp
//===----------------------------------------------------------------------===//

ParseResult SwitchOpNameOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand op;
  Type opType;
  if (parser.parseOperand(op))
    return failure();

  ArrayAttr cases;
  SmallVector<std::unique_ptr<Region>> caseRegions;
  if (parseSwitchCases(
          parser, cases, caseRegions,
          [](OpAsmParser &p, Attribute &out) -> ParseResult {
            std::string name;
            if (p.parseString(&name))
              return failure();
            out = p.getBuilder().getStringAttr(name);
            return success();
          }))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("caseNames", cases);

  opType = parser.getBuilder().getType<pdl::OperationType>();
  if (parser.resolveOperand(op, opType, result.operands))
    return failure();

  for (auto &region : caseRegions)
    result.addRegion(std::move(region));
  return success();
}

void SwitchOpNameOp::print(OpAsmPrinter &p) {
  p << ' ' << getOp();
  printSwitchCases(p, getCaseNames(), getCaseRegions(),
                   [](OpAsmPrinter &p, Attribute attr) {
                     p.printAttributeWithoutType(attr);
                   });
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseNames"});
}

LogicalResult SwitchOpNameOp::verify() {
  if (getCaseNames().size() != getCaseRegions().size())
    return emitOpError("expected one region per case name (")
           << getCaseNames().size() << " names vs "
           << getCaseRegions().size() << " regions)";
  for (Attribute attr : getCaseNames()) {
    if (!llvm::isa<StringAttr>(attr))
      return emitOpError("case names must be string attributes");
  }
  return success();
}

StringRef SwitchOpNameOp::getDefaultDialect() {
  return MatchDialect::getDialectNamespace();
}

//===----------------------------------------------------------------------===//
// SwitchTypeOp
//===----------------------------------------------------------------------===//

ParseResult SwitchTypeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand type;
  if (parser.parseOperand(type))
    return failure();

  ArrayAttr cases;
  SmallVector<std::unique_ptr<Region>> caseRegions;
  if (parseSwitchCases(
          parser, cases, caseRegions,
          [](OpAsmParser &p, Attribute &out) -> ParseResult {
            Type t;
            if (p.parseType(t))
              return failure();
            out = TypeAttr::get(t);
            return success();
          }))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("caseTypes", cases);

  Type pdlType = parser.getBuilder().getType<pdl::TypeType>();
  if (parser.resolveOperand(type, pdlType, result.operands))
    return failure();

  for (auto &region : caseRegions)
    result.addRegion(std::move(region));
  return success();
}

void SwitchTypeOp::print(OpAsmPrinter &p) {
  p << ' ' << getTypeValue();
  printSwitchCases(p, getCaseTypes(), getCaseRegions(),
                   [](OpAsmPrinter &p, Attribute attr) {
                     p.printType(llvm::cast<TypeAttr>(attr).getValue());
                   });
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseTypes"});
}

LogicalResult SwitchTypeOp::verify() {
  if (getCaseTypes().size() != getCaseRegions().size())
    return emitOpError("expected one region per case type (")
           << getCaseTypes().size() << " types vs "
           << getCaseRegions().size() << " regions)";
  for (Attribute attr : getCaseTypes()) {
    if (!llvm::isa<TypeAttr>(attr))
      return emitOpError("case types must be type attributes");
  }
  return success();
}

StringRef SwitchTypeOp::getDefaultDialect() {
  return MatchDialect::getDialectNamespace();
}

//===----------------------------------------------------------------------===//
// GetValueTypeOp
//===----------------------------------------------------------------------===//

LogicalResult GetValueTypeOp::verify() {
  bool valueIsRange = llvm::isa<pdl::RangeType>(getValue().getType());
  bool resultIsRange = llvm::isa<pdl::RangeType>(getResult().getType());
  if (valueIsRange != resultIsRange)
    return emitOpError(
        "expected result to be a range iff the value is a range");
  return success();
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
  auto optType = llvm::cast<OptionalType>(getOptionalValue().getType());
  if (optType.getInnerType() != getUnwrapped().getType())
    return emitOpError("expected unwrapped result type ")
           << getUnwrapped().getType() << " to match inner type of optional "
           << optType.getInnerType();
  return success();
}

//===----------------------------------------------------------------------===//
// SuccessOp
//===----------------------------------------------------------------------===//

LogicalResult SuccessOp::verify() {
  // Must be inside a matcher (possibly via nested try / switch regions).
  if (!(*this)->getParentOfType<MatcherOp>())
    return emitOpError(
        "`match.success` must be enclosed by a `match.matcher`");
  return success();
}

LogicalResult
SuccessOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // The rewriter symbol must resolve to some symbol op. We don't constrain it
  // to a specific op kind here, mirroring `pdl_interp.record_match` which also
  // references an arbitrary symbol (typically a `pdl_interp.func` in a
  // rewriter module).
  if (!symbolTable.lookupNearestSymbolFrom(*this, getRewriterAttr()))
    return emitOpError("references an unknown rewriter symbol: ")
           << getRewriterAttr();
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Match/IR/MatchOps.cpp.inc"
