//===- PDLConstrToPDLInterp.cpp - Lower pdl_constr to pdl_interp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mechanical lowering of `pdl_constr` IR to `pdl_interp` IR.
//
// The `pdl_constr` dialect is an explicitly ordered IR: tests have an
// implicit failure control effect, OR-of-alternatives is expressed by
// nested `pdl_constr.try` regions, and multi-way dispatch by
// `pdl_constr.switch_op_name` / `pdl_constr.switch_type`. Because the
// matcher-tree shape is already encoded in the IR, lowering to `pdl_interp`
// is a mechanical region walker that maintains two pieces of state:
//
//   * `currentBlock`  — where the next `pdl_interp` op is emitted.
//   * `failureBlock`  — the destination for failure branches in the current
//                       failure scope (also where `pdl_interp.record_match`
//                       branches after a successful match, since after a
//                       match we continue looking for more matches).
//
// Lowering rules:
//   * Test ops (`has_name`, `equal`, `is_not_null`, ...) lower to the
//     corresponding `pdl_interp.check_*` / `are_equal` / `is_not_null`
//     with the failure branch pointing at `failureBlock`.
//   * Navigation ops lower to the corresponding `pdl_interp.get_*` ops.
//   * `pdl_constr.try { ... }` creates a fresh "after-try" block; the body
//     is lowered with `failureBlock` set to that block; subsequent siblings
//     in the parent region are lowered into that block (which is also where
//     fall-off-end of the body branches).
//   * `pdl_constr.switch_*` lowers to `pdl_interp.switch_*` with the case
//     regions lowered into separate blocks and the default set to the
//     enclosing `failureBlock`. The switch is treated as a terminator: ops
//     in the parent region after the switch are unreachable and are not
//     emitted.
//   * `pdl_constr.get_each` lowers to `pdl_interp.foreach`; subsequent ops
//     in the same region emit inside the loop body, and their failure
//     branches go to a `pdl_interp.continue` (next iteration). When the
//     loop exhausts, control transfers to the original `failureBlock`.
//   * `pdl_constr.success` lowers to `pdl_interp.record_match` whose
//     successor is the current `failureBlock`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLConstrToPDLInterp/PDLConstrToPDLInterp.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLCONSTRTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pdl_constr;

namespace {

//===----------------------------------------------------------------------===//
// Lowerer
//===----------------------------------------------------------------------===//

class Lowerer {
public:
  Lowerer(pdl_interp::FuncOp matcherFunc)
      : builder(matcherFunc.getContext()), matcherFunc(matcherFunc) {}

  /// Lower the given matcher ops into the matcher function. Each matcher's
  /// failure is chained to the entry block of the next matcher; the last
  /// matcher's failure block is the finalize block.
  LogicalResult lower(ArrayRef<MatcherOp> matchers);

private:
  /// Lower a single matcher op into the function, starting at `entry` with
  /// failure destination `failure`.
  LogicalResult lowerMatcher(MatcherOp matcher, Block *entry, Block *failure);

  /// Lower the ops of a single-block region into the current `currentBlock`
  /// with the current `failureBlock`. On fall-off-end of the region the
  /// terminal block is branched to `failureBlock`. Mutates `currentBlock`
  /// and `failureBlock` as a side effect.
  LogicalResult lowerRegion(Region &region);

  /// Lower a single op, dispatching on op kind. May update `currentBlock`
  /// and / or `failureBlock`.
  LogicalResult lowerOp(Operation *op);

  /// Append a new empty block to the matcher function.
  Block *newBlock() {
    Block *block = new Block();
    matcherFunc.getBody().push_back(block);
    return block;
  }

  /// Lookup `v` (a `pdl_constr` SSA value) in the value mapping. Asserts
  /// when not found, since well-formed `pdl_constr` IR must define a value
  /// before use.
  Value lookup(Value v) const {
    Value mapped = valueMap.lookup(v);
    assert(mapped && "pdl_constr value not mapped");
    return mapped;
  }

  /// Map a `pdl_constr` SSA value to a `pdl_interp` SSA value.
  void map(Value from, Value to) { valueMap.map(from, to); }

  /// Record `val` as a possible "location" operation for subsequent
  /// `record_match` ops (only if it is an `!pdl.operation` value).
  void recordLocOp(Value val) {
    if (val && llvm::isa<pdl::OperationType>(val.getType()))
      locOps.insert(val);
  }

  //===--------------------------------------------------------------------===//
  // Per-op lowering helpers
  //===--------------------------------------------------------------------===//

  // Structural
  LogicalResult lowerTry(TryOp op);
  LogicalResult lowerSwitchOpName(SwitchOpNameOp op);
  LogicalResult lowerSwitchType(SwitchTypeOp op);

  // Navigation (nullable, return wrapped optional<T>)
  LogicalResult lowerGetOperand(GetOperandOp op);
  LogicalResult lowerGetOperands(GetOperandsOp op);
  LogicalResult lowerGetResult(GetResultOp op);
  LogicalResult lowerGetResults(GetResultsOp op);
  LogicalResult lowerGetAttribute(GetAttributeOp op);
  LogicalResult lowerGetDefiningOp(GetDefiningOpOp op);

  // Navigation (non-nullable)
  LogicalResult lowerGetValueType(GetValueTypeOp op);
  LogicalResult lowerGetAttributeType(GetAttributeTypeOp op);
  LogicalResult lowerGetUsers(GetUsersOp op);

  // get_each → foreach loop
  LogicalResult lowerGetEach(GetEachOp op);

  // Tests
  LogicalResult lowerIsNotNull(IsNotNullOp op);
  LogicalResult lowerHasName(HasNameOp op);
  LogicalResult lowerEqual(EqualOp op);
  LogicalResult lowerHasType(HasTypeOp op);
  LogicalResult lowerHasTypes(HasTypesOp op);
  LogicalResult lowerHasAttrValue(HasAttrValueOp op);
  LogicalResult lowerCheckOperandCount(CheckOperandCountOp op);
  LogicalResult lowerCheckResultCount(CheckResultCountOp op);
  LogicalResult lowerApplyNativeConstraint(ApplyNativeConstraintOp op);

  // Success
  LogicalResult lowerSuccess(SuccessOp op);

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  OpBuilder builder;
  pdl_interp::FuncOp matcherFunc;

  /// SSA value mapping from `pdl_constr` values to their `pdl_interp`
  /// counterparts. For `pdl_constr.is_not_null`, both the optional operand
  /// and the unwrapped result map to the same bare `pdl_interp` value.
  IRMapping valueMap;

  /// Current emission point.
  Block *currentBlock = nullptr;

  /// Current failure scope. Test failures and `record_match` continuations
  /// branch here.
  Block *failureBlock = nullptr;

  /// The set of `!pdl.operation` values defined on the current path; used
  /// as the `loc()` operand list of `pdl_interp.record_match`.
  llvm::SmallSetVector<Value, 4> locOps;

  /// Static operation names established on the current path, keyed by the
  /// mapped `pdl_interp` `!pdl.operation` value. Populated by `has_name`
  /// (and by `switch_op_name` case regions). Used to recover the optional
  /// `rootKind` of `pdl_interp.record_match`, mirroring the original
  /// `pdl -> pdl_interp` lowering.
  llvm::DenseMap<Value, StringAttr> opNameMap;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the inner type of an `!pdl_constr.optional<T>`.
static Type unwrapOptional(Type type) {
  auto opt = llvm::cast<OptionalType>(type);
  return opt.getInnerType();
}

//===----------------------------------------------------------------------===//
// Lowerer: top-level driver
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lower(ArrayRef<MatcherOp> matchers) {
  // Create the finalize block at the end of the function.
  Block *finalize = newBlock();
  builder.setInsertionPointToEnd(finalize);
  pdl_interp::FinalizeOp::create(builder, matcherFunc.getLoc());

  // The matcher function already has an entry block with the root operation
  // as its sole argument. We chain matchers: the first matcher emits into
  // the entry block, and each matcher's failure becomes the next matcher's
  // entry block. The last matcher's failure is the finalize block.
  Block *currentEntry = &matcherFunc.front();
  for (size_t i = 0, e = matchers.size(); i < e; ++i) {
    Block *failureEntry = (i + 1 == e) ? finalize : newBlock();
    if (failed(lowerMatcher(matchers[i], currentEntry, failureEntry)))
      return failure();
    currentEntry = failureEntry;
  }

  // If there were no matcher ops, we still need the entry block to branch
  // to the finalize block.
  if (matchers.empty()) {
    builder.setInsertionPointToEnd(currentEntry);
    pdl_interp::BranchOp::create(builder, matcherFunc.getLoc(), finalize);
  }

  return success();
}

LogicalResult Lowerer::lowerMatcher(MatcherOp matcher, Block *entry,
                                    Block *failureBB) {
  // Map the matcher's root block argument to the matcher function's root
  // argument (which is identical across all matchers).
  Block &matcherBody = matcher.getBodyRegion().front();
  Value funcRoot = matcherFunc.front().getArgument(0);
  map(matcherBody.getArgument(0), funcRoot);

  // Track the root as a location operation.
  size_t savedLocSize = locOps.size();
  recordLocOp(funcRoot);

  // Set up emission state for this matcher. Op names are path-local; the
  // shared root value would otherwise carry a name across matchers.
  opNameMap.clear();
  currentBlock = entry;
  failureBlock = failureBB;

  if (failed(lowerRegion(matcher.getBodyRegion())))
    return failure();

  // Restore locOps to the state before this matcher.
  while (locOps.size() > savedLocSize)
    locOps.pop_back();

  return success();
}

LogicalResult Lowerer::lowerRegion(Region &region) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block &block = region.front();
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (!currentBlock) {
      // The region has been terminated (e.g. by a switch or a success op
      // followed by no more relevant ops). Any subsequent ops are dead.
      break;
    }
    if (failed(lowerOp(&op)))
      return failure();
  }

  // Fall-off-end of the region: branch to the current failure scope.
  if (currentBlock) {
    builder.setInsertionPointToEnd(currentBlock);
    pdl_interp::BranchOp::create(builder, region.getLoc(), failureBlock);
    currentBlock = nullptr;
  }
  return success();
}

LogicalResult Lowerer::lowerOp(Operation *op) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<TryOp>([&](auto o) { return lowerTry(o); })
      .Case<SwitchOpNameOp>([&](auto o) { return lowerSwitchOpName(o); })
      .Case<SwitchTypeOp>([&](auto o) { return lowerSwitchType(o); })
      .Case<GetOperandOp>([&](auto o) { return lowerGetOperand(o); })
      .Case<GetOperandsOp>([&](auto o) { return lowerGetOperands(o); })
      .Case<GetResultOp>([&](auto o) { return lowerGetResult(o); })
      .Case<GetResultsOp>([&](auto o) { return lowerGetResults(o); })
      .Case<GetAttributeOp>([&](auto o) { return lowerGetAttribute(o); })
      .Case<GetDefiningOpOp>([&](auto o) { return lowerGetDefiningOp(o); })
      .Case<GetValueTypeOp>([&](auto o) { return lowerGetValueType(o); })
      .Case<GetAttributeTypeOp>([&](auto o) { return lowerGetAttributeType(o); })
      .Case<GetUsersOp>([&](auto o) { return lowerGetUsers(o); })
      .Case<GetEachOp>([&](auto o) { return lowerGetEach(o); })
      .Case<IsNotNullOp>([&](auto o) { return lowerIsNotNull(o); })
      .Case<HasNameOp>([&](auto o) { return lowerHasName(o); })
      .Case<EqualOp>([&](auto o) { return lowerEqual(o); })
      .Case<HasTypeOp>([&](auto o) { return lowerHasType(o); })
      .Case<HasTypesOp>([&](auto o) { return lowerHasTypes(o); })
      .Case<HasAttrValueOp>([&](auto o) { return lowerHasAttrValue(o); })
      .Case<CheckOperandCountOp>(
          [&](auto o) { return lowerCheckOperandCount(o); })
      .Case<CheckResultCountOp>(
          [&](auto o) { return lowerCheckResultCount(o); })
      .Case<ApplyNativeConstraintOp>(
          [&](auto o) { return lowerApplyNativeConstraint(o); })
      .Case<SuccessOp>([&](auto o) { return lowerSuccess(o); })
      .Default([&](Operation *o) {
        return o->emitOpError("unsupported pdl_constr op in lowering");
      });
}

//===----------------------------------------------------------------------===//
// Structural
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerTry(TryOp op) {
  Block *afterTry = newBlock();

  // Branch from the current block into the body (the body is lowered into
  // the current block; we don't need a separate body block).
  Block *savedFailure = failureBlock;
  failureBlock = afterTry;

  // Track locOps stack so anything created inside the try is dropped after.
  size_t savedLocSize = locOps.size();

  if (failed(lowerRegion(op.getBody())))
    return failure();

  while (locOps.size() > savedLocSize)
    locOps.pop_back();

  failureBlock = savedFailure;
  currentBlock = afterTry;
  return success();
}

LogicalResult Lowerer::lowerSwitchOpName(SwitchOpNameOp op) {
  Value mappedOp = lookup(op.getOp());

  // Lower each case region into its own block. Each case body inherits the
  // current failure scope.
  SmallVector<Block *, 4> caseBlocks;
  caseBlocks.reserve(op.getCaseRegions().size());

  Block *outerCurrent = currentBlock;
  Block *outerFailure = failureBlock;
  for (auto [caseRegion, caseName] :
       llvm::zip(op.getCaseRegions(), op.getCaseNames())) {
    Block *caseBlock = newBlock();
    caseBlocks.push_back(caseBlock);

    size_t savedLocSize = locOps.size();
    currentBlock = caseBlock;
    failureBlock = outerFailure;
    // Within this case region the switched op is known to have `caseName`.
    StringAttr savedName = opNameMap.lookup(mappedOp);
    opNameMap[mappedOp] = llvm::cast<StringAttr>(caseName);
    if (failed(lowerRegion(caseRegion)))
      return failure();
    if (savedName)
      opNameMap[mappedOp] = savedName;
    else
      opNameMap.erase(mappedOp);
    while (locOps.size() > savedLocSize)
      locOps.pop_back();
  }

  // Emit the switch terminator in the outer current block.
  currentBlock = outerCurrent;
  failureBlock = outerFailure;
  builder.setInsertionPointToEnd(currentBlock);

  SmallVector<OperationName, 4> names;
  names.reserve(op.getCaseNames().size());
  for (Attribute a : op.getCaseNames())
    names.push_back(
        OperationName(llvm::cast<StringAttr>(a).getValue(), op.getContext()));

  pdl_interp::SwitchOperationNameOp::create(
      builder, op.getLoc(), mappedOp, names, outerFailure, caseBlocks);

  // Switch is a terminator; no ops after it should be lowered in the parent.
  currentBlock = nullptr;
  return success();
}

LogicalResult Lowerer::lowerSwitchType(SwitchTypeOp op) {
  Value mappedTy = lookup(op.getTypeValue());

  SmallVector<Block *, 4> caseBlocks;
  caseBlocks.reserve(op.getCaseRegions().size());

  Block *outerCurrent = currentBlock;
  Block *outerFailure = failureBlock;
  for (Region &caseRegion : op.getCaseRegions()) {
    Block *caseBlock = newBlock();
    caseBlocks.push_back(caseBlock);

    size_t savedLocSize = locOps.size();
    currentBlock = caseBlock;
    failureBlock = outerFailure;
    if (failed(lowerRegion(caseRegion)))
      return failure();
    while (locOps.size() > savedLocSize)
      locOps.pop_back();
  }

  currentBlock = outerCurrent;
  failureBlock = outerFailure;
  builder.setInsertionPointToEnd(currentBlock);

  SmallVector<Attribute, 4> caseTypeAttrs(op.getCaseTypes().begin(),
                                          op.getCaseTypes().end());

  pdl_interp::SwitchTypeOp::create(builder, op.getLoc(), mappedTy,
                                   caseTypeAttrs, outerFailure, caseBlocks);

  currentBlock = nullptr;
  return success();
}

//===----------------------------------------------------------------------===//
// Navigation: nullable
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerGetOperand(GetOperandOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Type resultTy = unwrapOptional(op.getResult().getType());
  Value v = pdl_interp::GetOperandOp::create(
      builder, op.getLoc(), resultTy, lookup(op.getOp()), op.getIndexAttr());
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetOperands(GetOperandsOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Type resultTy = unwrapOptional(op.getResult().getType());
  Value v = pdl_interp::GetOperandsOp::create(
      builder, op.getLoc(), resultTy, lookup(op.getOp()), op.getIndexAttr());
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetResult(GetResultOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Type resultTy = unwrapOptional(op.getResult().getType());
  Value v = pdl_interp::GetResultOp::create(
      builder, op.getLoc(), resultTy, lookup(op.getOp()), op.getIndexAttr());
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetResults(GetResultsOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Type resultTy = unwrapOptional(op.getResult().getType());
  Value v = pdl_interp::GetResultsOp::create(
      builder, op.getLoc(), resultTy, lookup(op.getOp()), op.getIndexAttr());
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetAttribute(GetAttributeOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Value v = pdl_interp::GetAttributeOp::create(
      builder, op.getLoc(), builder.getType<pdl::AttributeType>(),
      lookup(op.getOp()), op.getNameAttr());
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetDefiningOp(GetDefiningOpOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Value v = pdl_interp::GetDefiningOpOp::create(
      builder, op.getLoc(), builder.getType<pdl::OperationType>(),
      lookup(op.getValue()));
  map(op.getResult(), v);
  recordLocOp(v);
  return success();
}

//===----------------------------------------------------------------------===//
// Navigation: non-nullable
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerGetValueType(GetValueTypeOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Value v = pdl_interp::GetValueTypeOp::create(
      builder, op.getLoc(), op.getResult().getType(), lookup(op.getValue()));
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetAttributeType(GetAttributeTypeOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Value v = pdl_interp::GetAttributeTypeOp::create(builder, op.getLoc(),
                                                   lookup(op.getAttribute()));
  map(op.getResult(), v);
  return success();
}

LogicalResult Lowerer::lowerGetUsers(GetUsersOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Value v = pdl_interp::GetUsersOp::create(builder, op.getLoc(),
                                           lookup(op.getValue()));
  map(op.getResult(), v);
  return success();
}

//===----------------------------------------------------------------------===//
// get_each
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerGetEach(GetEachOp op) {
  builder.setInsertionPointToEnd(currentBlock);
  Value range = lookup(op.getRange());

  // foreach's successor is the current failureBlock: when the loop
  // exhausts (no more elements), control transfers to the original
  // failure scope.
  auto foreach = pdl_interp::ForEachOp::create(builder, op.getLoc(), range,
                                               failureBlock, /*initLoop=*/true);

  // Loop variable is the result of get_each.
  map(op.getResult(), foreach.getLoopVariable());
  recordLocOp(foreach.getLoopVariable());

  // Append a continue block at the end of the foreach region.
  Block *continueBlock = new Block();
  foreach.getRegion().push_back(continueBlock);
  builder.setInsertionPointToEnd(continueBlock);
  pdl_interp::ContinueOp::create(builder, op.getLoc());

  // Subsequent ops in the enclosing region must emit inside the loop body
  // (the foreach region's first block), with failure -> continue (next
  // iteration). Fall-off-end of the enclosing region naturally branches to
  // the continue block (the next iteration), and when the iteration ends
  // foreach proceeds to the original failureBlock.
  currentBlock = &foreach.getRegion().front();
  failureBlock = continueBlock;
  return success();
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerIsNotNull(IsNotNullOp op) {
  Value mapped = lookup(op.getOptionalValue());

  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::IsNotNullOp::create(builder, op.getLoc(), mapped, successBB,
                                  failureBlock);

  // The unwrapped result is the same SSA value as the mapped operand —
  // SSA dominance guarantees consumers run only after the null check.
  map(op.getUnwrapped(), mapped);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerHasName(HasNameOp op) {
  Block *successBB = newBlock();
  Value mappedOp = lookup(op.getOp());
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::CheckOperationNameOp::create(builder, op.getLoc(), mappedOp,
                                           op.getNameAttr(), successBB,
                                           failureBlock);
  // Remember the static name established for this op so that a subsequent
  // `success` on the same path can recover the `record_match` root kind.
  opNameMap[mappedOp] = op.getNameAttr();
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerEqual(EqualOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::AreEqualOp::create(builder, op.getLoc(), lookup(op.getLhs()),
                                 lookup(op.getRhs()), successBB, failureBlock);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerHasType(HasTypeOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::CheckTypeOp::create(builder, op.getLoc(),
                                  lookup(op.getTypeValue()),
                                  op.getConstantTypeAttr(), successBB,
                                  failureBlock);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerHasTypes(HasTypesOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::CheckTypesOp::create(builder, op.getLoc(), lookup(op.getTypes()),
                                   op.getConstantTypesAttr(), successBB,
                                   failureBlock);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerHasAttrValue(HasAttrValueOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::CheckAttributeOp::create(builder, op.getLoc(),
                                       lookup(op.getAttribute()),
                                       op.getValueAttr(), successBB,
                                       failureBlock);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerCheckOperandCount(CheckOperandCountOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::CheckOperandCountOp::create(
      builder, op.getLoc(), lookup(op.getOp()), op.getCount(),
      /*compareAtLeast=*/op.getAtLeast(), successBB, failureBlock);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerCheckResultCount(CheckResultCountOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::CheckResultCountOp::create(
      builder, op.getLoc(), lookup(op.getOp()), op.getCount(),
      /*compareAtLeast=*/op.getAtLeast(), successBB, failureBlock);
  currentBlock = successBB;
  return success();
}

LogicalResult Lowerer::lowerApplyNativeConstraint(ApplyNativeConstraintOp op) {
  Block *successBB = newBlock();
  builder.setInsertionPointToEnd(currentBlock);

  SmallVector<Value, 4> args;
  args.reserve(op.getArgs().size());
  for (Value v : op.getArgs())
    args.push_back(lookup(v));

  SmallVector<Type, 2> resultTypes(op.getConstraintResults().getTypes());

  auto applied = pdl_interp::ApplyConstraintOp::create(
      builder, op.getLoc(), resultTypes, op.getNameAttr(), args,
      op.getIsNegatedAttr(), successBB, failureBlock);

  // Map the produced results.
  for (auto [orig, mapped] :
       llvm::zip(op.getConstraintResults(), applied.getResults())) {
    map(orig, mapped);
    recordLocOp(mapped);
  }

  currentBlock = successBB;
  return success();
}

//===----------------------------------------------------------------------===//
// Success
//===----------------------------------------------------------------------===//

LogicalResult Lowerer::lowerSuccess(SuccessOp op) {
  builder.setInsertionPointToEnd(currentBlock);

  SmallVector<Value, 4> inputs;
  inputs.reserve(op.getInputs().size());
  for (Value v : op.getInputs())
    inputs.push_back(lookup(v));

  ArrayAttr generatedOpsAttr;

  // Recover the optional root kind: the matcher's root is the function's
  // sole block argument; if a `has_name` (or `switch_op_name` case) on the
  // current path established its static op name, use that for the
  // `record_match` root kind, mirroring the original `pdl -> pdl_interp`
  // lowering.
  Value funcRoot = matcherFunc.front().getArgument(0);
  StringAttr rootKindAttr = opNameMap.lookup(funcRoot);

  SmallVector<Value, 4> matchedOps(locOps.begin(), locOps.end());

  pdl_interp::RecordMatchOp::create(
      builder, op.getLoc(), inputs, matchedOps, op.getRewriterAttr(),
      rootKindAttr, generatedOpsAttr, op.getBenefitAttr(), failureBlock);

  // record_match is a terminator; no ops after it should be emitted in
  // this scope (subsequent siblings in the parent region are unreachable
  // from this success op).
  currentBlock = nullptr;
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct PDLConstrToPDLInterpPass
    : public impl::ConvertPDLConstrToPDLInterpPassBase<
          PDLConstrToPDLInterpPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Collect all top-level `pdl_constr.matcher` ops in source order.
    SmallVector<MatcherOp, 4> matchers;
    for (MatcherOp m : module.getOps<MatcherOp>())
      matchers.push_back(m);

    // Create the matcher function at the top of the module.
    OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
    auto matcherFunc = pdl_interp::FuncOp::create(
        builder, module.getLoc(),
        pdl_interp::PDLInterpDialect::getMatcherFunctionName(),
        builder.getFunctionType(builder.getType<pdl::OperationType>(),
                                /*results=*/{}),
        /*attrs=*/ArrayRef<NamedAttribute>());

    // Lower the matchers into the matcher function.
    Lowerer lowerer(matcherFunc);
    if (failed(lowerer.lower(matchers))) {
      signalPassFailure();
      return;
    }

    // Erase the lowered `pdl_constr.matcher` ops.
    for (MatcherOp m : matchers)
      m.erase();
  }
};
} // namespace
