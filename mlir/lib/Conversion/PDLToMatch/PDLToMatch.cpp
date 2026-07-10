//===- PDLToMatch.cpp - Lower PDL to PDL Constraint dialect ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers each `pdl.pattern` in a module into its own
// `match.matcher`. The lowering performs root selection / multi-root
// ordering using the shared helpers from the `PDLToPDLInterp` library and
// emits an ordered linear sequence of navigation and constraint ops inside
// the matcher body. Implicit failure semantics of the constraint ops
// (failure transfers control to the enclosing failure scope, which for an
// individual matcher is "leave the matcher") replace the SSA predicate
// values of the old design.
//
// A subsequent pass is expected to combine multiple matchers into a single
// shared matcher tree (using `match.try` / `match.switch_*`); this
// pass deliberately emits one matcher per pattern with no `try` / `switch`
// nesting beyond what is necessary to express the pattern itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToMatch/PDLToMatch.h"

#include "mlir/Conversion/PDLToPDLInterp/RewriterGen.h"
#include "mlir/Conversion/PDLToPDLInterp/RootOrdering.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/Match/IR/MatchOps.h"
#include "mlir/Dialect/Match/IR/MatchTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLTOMATCHPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "convert-pdl-to-match"

//===----------------------------------------------------------------------===//
// Shared helpers
//
// Root detection and cost-graph construction are shared with the
// PDL -> PDLInterp lowering and live in the `PDLToPDLInterp` library (see
// `RootOrdering.h`); only the emission backend below differs.
//===----------------------------------------------------------------------===//

using pdl_to_pdl_interp::buildCostGraph;
using pdl_to_pdl_interp::detectRoots;
using pdl_to_pdl_interp::getNumNonRangeValues;
using pdl_to_pdl_interp::OpIndex;
using pdl_to_pdl_interp::OptimalBranching;
using pdl_to_pdl_interp::ParentMaps;
using pdl_to_pdl_interp::RootOrderingEntry;
using pdl_to_pdl_interp::RootOrderingGraph;
using pdl_to_pdl_interp::useOperandGroup;

//===----------------------------------------------------------------------===//
// PDL → match Emitter
//===----------------------------------------------------------------------===//

namespace {

/// Emits an ordered linear sequence of `match` ops for a single
/// `pdl::PatternOp`.
class MatchEmitter {
public:
  MatchEmitter(OpBuilder &builder, pdl::PatternOp pattern,
                   ModuleOp rewriterModule,
                   SymbolTable &rewriterSymbolTable)
      : builder(builder), pattern(pattern), ctx(pattern.getContext()),
        loc(pattern.getLoc()), rewriterModule(rewriterModule),
        rewriterSymbolTable(rewriterSymbolTable) {}

  /// Emit a `match.matcher` for the wrapped `pdl.pattern`.
  match::MatcherOp emit();

private:
  /// Emit a nullable navigation op and an `is_not_null` unwrap, returning the
  /// bare-typed unwrapped value.
  Value emitNavAndUnwrap(Value navResult, Type innerType);

  /// Emit navigation + constraints for a single `pdl::OperationOp` (downward
  /// tree walk). `opVal` is the `match` SSA value representing the op.
  /// If `ignoreOperand` is set, skip that operand (used during upward
  /// traversals).
  void emitOperationConstraints(pdl::OperationOp op, Value opVal,
                                std::optional<unsigned> ignoreOperand = {});

  /// Emit constraints for an operand value (`pdl::OperandOp`,
  /// `pdl::OperandsOp`, `pdl::ResultOp`, `pdl::ResultsOp`).
  void emitOperandConstraints(Value pdlVal, Value constrVal);

  /// Emit constraints for an attribute value (`pdl::AttributeOp`).
  void emitAttributeConstraints(pdl::AttributeOp attrOp, Value constrVal);

  /// Emit constraints for a type value (`pdl::TypeOp`, `pdl::TypesOp`).
  void emitTypeConstraints(Value pdlVal, Value constrVal);

  /// Emit upward traversals for multi-root patterns.
  void emitUpwardTraversal(OpIndex opIndex, Value &pos, unsigned rootID);

  /// Emit non-tree predicates (standalone results, native constraints).
  void emitNonTreePredicates();

  /// Get or create the match value corresponding to a pdl value. If the
  /// value has already been visited, returns the existing value and emits an
  /// equality constraint. Returns nullptr if the value is new (and registers
  /// it).
  Value getOrRegister(Value pdlVal, Value constrVal);

  OpBuilder &builder;
  pdl::PatternOp pattern;
  MLIRContext *ctx;
  Location loc;

  /// The nested module that holds the lowered rewriter functions and the
  /// symbol table used to insert into it.
  ModuleOp rewriterModule;
  SymbolTable &rewriterSymbolTable;

  /// Mapping from `pdl` SSA values to their `match` SSA values.
  DenseMap<Value, Value> valueMap;
};

} // namespace

Value MatchEmitter::getOrRegister(Value pdlVal, Value constrVal) {
  auto it = valueMap.try_emplace(pdlVal, constrVal);
  if (!it.second) {
    // Already visited — emit equality constraint.
    Value existing = it.first->second;
    if (isa_and_nonnull<pdl::AttributeOp, pdl::OperandOp, pdl::OperandsOp,
                        pdl::OperationOp, pdl::TypeOp, pdl::TypesOp>(
            pdlVal.getDefiningOp())) {
      match::EqualOp::create(builder, loc, constrVal, existing);
    }
    return existing;
  }
  return Value(); // new registration
}

Value MatchEmitter::emitNavAndUnwrap(Value navResult, Type innerType) {
  return match::IsNotNullOp::create(builder, loc, innerType, navResult)
      .getUnwrapped();
}

void MatchEmitter::emitTypeConstraints(Value pdlVal, Value constrVal) {
  if (auto typeOp = pdlVal.getDefiningOp<pdl::TypeOp>()) {
    if (Attribute type = typeOp.getConstantTypeAttr())
      match::HasTypeOp::create(builder, loc, constrVal,
                                    cast<TypeAttr>(type));
  } else if (auto typesOp = pdlVal.getDefiningOp<pdl::TypesOp>()) {
    if (Attribute types = typesOp.getConstantTypesAttr())
      match::HasTypesOp::create(builder, loc, constrVal,
                                     cast<ArrayAttr>(types));
  }
}

void MatchEmitter::emitAttributeConstraints(pdl::AttributeOp attrOp,
                                                Value constrVal) {
  // If the attribute has a type, constrain the type.
  if (Value type = attrOp.getValueType()) {
    Value typeVal = match::GetAttributeTypeOp::create(
        builder, loc, builder.getType<pdl::TypeType>(), constrVal);
    Value existing = getOrRegister(type, typeVal);
    if (!existing)
      emitTypeConstraints(type, typeVal);
  } else if (Attribute value = attrOp.getValueAttr()) {
    // If the attribute has a constant value, constrain it.
    match::HasAttrValueOp::create(builder, loc, constrVal, value);
  }
}

void MatchEmitter::emitOperandConstraints(Value pdlVal, Value constrVal) {
  TypeSwitch<Operation *>(pdlVal.getDefiningOp())
      .Case<pdl::OperandOp, pdl::OperandsOp>([&](auto op) {
        if (Value type = op.getValueType()) {
          Type typeResultType =
              isa<pdl::RangeType>(constrVal.getType())
                  ? (Type)pdl::RangeType::get(builder.getType<pdl::TypeType>())
                  : (Type)builder.getType<pdl::TypeType>();
          Value typeVal = match::GetValueTypeOp::create(
              builder, loc, typeResultType, constrVal);
          Value existing = getOrRegister(type, typeVal);
          if (!existing)
            emitTypeConstraints(type, typeVal);
        }
      })
      .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto op) {
        // Navigate to the defining op of the operand.
        Value defOpOpt = match::GetDefiningOpOp::create(
            builder, loc,
            match::OptionalType::get(
                builder.getType<pdl::OperationType>()),
            constrVal);
        Value defOp =
            emitNavAndUnwrap(defOpOpt, builder.getType<pdl::OperationType>());

        // Now check the result connects back.
        std::optional<unsigned> index = op.getIndex();
        bool isVariadic = isa<pdl::RangeType>(pdlVal.getType());
        Value resultVal;
        if (isa<pdl::ResultOp>(pdlVal.getDefiningOp())) {
          Value resOpt = match::GetResultOp::create(
              builder, loc,
              match::OptionalType::get(builder.getType<pdl::ValueType>()),
              defOp, builder.getI32IntegerAttr(*index));
          resultVal =
              emitNavAndUnwrap(resOpt, builder.getType<pdl::ValueType>());
        } else {
          Type innerType = isVariadic
                               ? (Type)pdl::RangeType::get(
                                     builder.getType<pdl::ValueType>())
                               : (Type)builder.getType<pdl::ValueType>();
          Value resOpt = match::GetResultsOp::create(
              builder, loc, match::OptionalType::get(innerType), defOp,
              index ? builder.getI32IntegerAttr(*index) : IntegerAttr());
          resultVal = emitNavAndUnwrap(resOpt, innerType);
        }
        // Equality constraint: result == operand.
        match::EqualOp::create(builder, loc, resultVal, constrVal);

        // Recursively process the defining operation.
        Value parentPdlVal = op.getParent();
        Value existing = getOrRegister(parentPdlVal, defOp);
        if (!existing) {
          auto opOp = cast<pdl::OperationOp>(parentPdlVal.getDefiningOp());
          emitOperationConstraints(opOp, defOp);
        }
      });
}

void MatchEmitter::emitOperationConstraints(
    pdl::OperationOp op, Value opVal, std::optional<unsigned> ignoreOperand) {
  // Operation name constraint.
  if (std::optional<StringRef> opName = op.getOpName())
    match::HasNameOp::create(builder, loc, opVal,
                                  builder.getStringAttr(*opName));

  // Operand count constraint.
  OperandRange operands = op.getOperandValues();
  unsigned minOperands = getNumNonRangeValues(operands);
  if (minOperands != operands.size()) {
    if (minOperands)
      match::CheckOperandCountOp::create(
          builder, loc, opVal, builder.getI32IntegerAttr(minOperands),
          /*atLeast=*/builder.getUnitAttr());
  } else {
    match::CheckOperandCountOp::create(
        builder, loc, opVal, builder.getI32IntegerAttr(minOperands),
        /*atLeast=*/UnitAttr());
  }

  // Result count constraint.
  OperandRange types = op.getTypeValues();
  unsigned minResults = getNumNonRangeValues(types);
  if (minResults == types.size()) {
    match::CheckResultCountOp::create(
        builder, loc, opVal, builder.getI32IntegerAttr(types.size()),
        /*atLeast=*/UnitAttr());
  } else if (minResults) {
    match::CheckResultCountOp::create(
        builder, loc, opVal, builder.getI32IntegerAttr(minResults),
        /*atLeast=*/builder.getUnitAttr());
  }

  // Attributes.
  for (auto [attrName, attr] :
       llvm::zip(op.getAttributeValueNames(), op.getAttributeValues())) {
    StringRef name = cast<StringAttr>(attrName).getValue();
    Value attrOpt = match::GetAttributeOp::create(
        builder, loc,
        match::OptionalType::get(builder.getType<pdl::AttributeType>()),
        opVal, builder.getStringAttr(name));
    Value attrVal =
        emitNavAndUnwrap(attrOpt, builder.getType<pdl::AttributeType>());

    Value existing = getOrRegister(attr, attrVal);
    if (!existing) {
      auto attrOp = cast<pdl::AttributeOp>(attr.getDefiningOp());
      emitAttributeConstraints(attrOp, attrVal);
    }
  }

  // Operands.
  if (operands.size() == 1 && isa<pdl::RangeType>(operands[0].getType())) {
    // Single variadic operand covering all operands.
    Type rangeValType = pdl::RangeType::get(builder.getType<pdl::ValueType>());
    Value opsOpt = match::GetOperandsOp::create(
        builder, loc, match::OptionalType::get(rangeValType), opVal,
        /*index=*/IntegerAttr());
    // Unwrap to a `range<value>`. The all-operands group can never be null at
    // runtime, but downstream ops expect bare PDL types so we still unwrap
    // (the null check will be a no-op).
    Value opsVal = emitNavAndUnwrap(opsOpt, rangeValType);

    Value existing = getOrRegister(operands[0], opsVal);
    if (!existing)
      emitOperandConstraints(operands[0], opsVal);
  } else {
    bool foundVariableLength = false;
    for (const auto &operandIt : llvm::enumerate(operands)) {
      bool isVariadic = isa<pdl::RangeType>(operandIt.value().getType());
      foundVariableLength |= isVariadic;

      if (ignoreOperand == operandIt.index())
        continue;

      Value operandVal;
      if (foundVariableLength) {
        Type innerType = isVariadic
                             ? (Type)pdl::RangeType::get(
                                   builder.getType<pdl::ValueType>())
                             : (Type)builder.getType<pdl::ValueType>();
        Value opsOpt = match::GetOperandsOp::create(
            builder, loc, match::OptionalType::get(innerType), opVal,
            builder.getI32IntegerAttr(operandIt.index()));
        operandVal = emitNavAndUnwrap(opsOpt, innerType);
      } else {
        Value opOpt = match::GetOperandOp::create(
            builder, loc,
            match::OptionalType::get(builder.getType<pdl::ValueType>()),
            opVal, builder.getI32IntegerAttr(operandIt.index()));
        operandVal =
            emitNavAndUnwrap(opOpt, builder.getType<pdl::ValueType>());
      }

      Value existing = getOrRegister(operandIt.value(), operandVal);
      if (!existing)
        emitOperandConstraints(operandIt.value(), operandVal);
    }
  }

  // Results.
  if (types.size() == 1 && isa<pdl::RangeType>(types[0].getType())) {
    // Single variadic result covering all results.
    Type rangeValType = pdl::RangeType::get(builder.getType<pdl::ValueType>());
    Value resOpt = match::GetResultsOp::create(
        builder, loc, match::OptionalType::get(rangeValType), opVal,
        /*index=*/IntegerAttr());
    Value resVal = emitNavAndUnwrap(resOpt, rangeValType);

    Type rangeTypeType = pdl::RangeType::get(builder.getType<pdl::TypeType>());
    Value typeVal = match::GetValueTypeOp::create(builder, loc,
                                                       rangeTypeType, resVal);

    Value existing = getOrRegister(types[0], typeVal);
    if (!existing)
      emitTypeConstraints(types[0], typeVal);
    return;
  }

  bool foundVariableLength = false;
  for (auto [idx, typeValue] : llvm::enumerate(types)) {
    bool isVariadic = isa<pdl::RangeType>(typeValue.getType());
    foundVariableLength |= isVariadic;

    Value resultVal;
    if (foundVariableLength) {
      Type innerType = isVariadic
                           ? (Type)pdl::RangeType::get(
                                 builder.getType<pdl::ValueType>())
                           : (Type)builder.getType<pdl::ValueType>();
      Value resOpt = match::GetResultsOp::create(
          builder, loc, match::OptionalType::get(innerType), opVal,
          builder.getI32IntegerAttr(idx));
      resultVal = emitNavAndUnwrap(resOpt, innerType);
    } else {
      Value resOpt = match::GetResultOp::create(
          builder, loc,
          match::OptionalType::get(builder.getType<pdl::ValueType>()),
          opVal, builder.getI32IntegerAttr(idx));
      resultVal = emitNavAndUnwrap(resOpt, builder.getType<pdl::ValueType>());
    }

    // Get the type of this result.
    Type typeResultType =
        isa<pdl::RangeType>(resultVal.getType())
            ? (Type)pdl::RangeType::get(builder.getType<pdl::TypeType>())
            : (Type)builder.getType<pdl::TypeType>();
    Value typeVal = match::GetValueTypeOp::create(builder, loc,
                                                       typeResultType, resultVal);

    Value existing = getOrRegister(typeValue, typeVal);
    if (!existing)
      emitTypeConstraints(typeValue, typeVal);
  }
}

void MatchEmitter::emitUpwardTraversal(OpIndex opIndex, Value &pos,
                                           unsigned rootID) {
  Value value = opIndex.parent;
  TypeSwitch<Operation *>(value.getDefiningOp())
      .Case([&](pdl::OperationOp operationOp) {
        // `get_users` requires a single `!pdl.value`. If `pos` is a range,
        // extract a representative element first using `match.extract`;
        // the `match -> pdl_interp` lowering emits the matching
        // `pdl_interp.extract`.
        Value userPos = pos;
        if (isa<pdl::RangeType>(pos.getType()))
          userPos = match::ExtractOp::create(builder, loc, pos, 0);

        // Get users and iterate.
        Value usersVal = match::GetUsersOp::create(
            builder, loc,
            pdl::RangeType::get(builder.getType<pdl::OperationType>()),
            userPos);
        Value opVal = match::GetEachOp::create(
            builder, loc, builder.getType<pdl::OperationType>(), usersVal);

        // Compare the operand(s) of the user against the input value(s).
        Value operandVal;
        if (!opIndex.index) {
          // All operands.
          Type rangeValType =
              pdl::RangeType::get(builder.getType<pdl::ValueType>());
          Value opsOpt = match::GetOperandsOp::create(
              builder, loc, match::OptionalType::get(rangeValType), opVal,
              /*index=*/IntegerAttr());
          operandVal = emitNavAndUnwrap(opsOpt, rangeValType);
        } else if (useOperandGroup(operationOp, *opIndex.index)) {
          Type type = operationOp.getOperandValues()[*opIndex.index].getType();
          bool variadic = isa<pdl::RangeType>(type);
          Type innerType = variadic
                               ? (Type)pdl::RangeType::get(
                                     builder.getType<pdl::ValueType>())
                               : (Type)builder.getType<pdl::ValueType>();
          Value opsOpt = match::GetOperandsOp::create(
              builder, loc, match::OptionalType::get(innerType), opVal,
              builder.getI32IntegerAttr(*opIndex.index));
          operandVal = emitNavAndUnwrap(opsOpt, innerType);
        } else {
          Value opOpt = match::GetOperandOp::create(
              builder, loc,
              match::OptionalType::get(builder.getType<pdl::ValueType>()),
              opVal, builder.getI32IntegerAttr(*opIndex.index));
          operandVal =
              emitNavAndUnwrap(opOpt, builder.getType<pdl::ValueType>());
        }
        // Equality constraint: operand == pos (the value we're traversing
        // from).
        match::EqualOp::create(builder, loc, operandVal, pos);

        // Register this operation.
        bool inserted = valueMap.try_emplace(value, opVal).second;
        (void)inserted;
        assert(inserted && "duplicate upward visit");

        // Recurse into the operation's other constraints.
        emitOperationConstraints(operationOp, opVal, opIndex.index);

        pos = opVal;
      })
      .Case([&](pdl::ResultOp resultOp) {
        // Individual result.
        Value resOpt = match::GetResultOp::create(
            builder, loc,
            match::OptionalType::get(builder.getType<pdl::ValueType>()),
            pos, builder.getI32IntegerAttr(*opIndex.index));
        pos = emitNavAndUnwrap(resOpt, builder.getType<pdl::ValueType>());
        valueMap.try_emplace(value, pos);
      })
      .Case([&](pdl::ResultsOp resultOp) {
        // Group of results.
        bool isVariadic = isa<pdl::RangeType>(value.getType());
        Type innerType = isVariadic
                             ? (Type)pdl::RangeType::get(
                                   builder.getType<pdl::ValueType>())
                             : (Type)builder.getType<pdl::ValueType>();
        if (opIndex.index) {
          Value resOpt = match::GetResultsOp::create(
              builder, loc, match::OptionalType::get(innerType), pos,
              builder.getI32IntegerAttr(*opIndex.index));
          pos = emitNavAndUnwrap(resOpt, innerType);
        } else {
          Value resOpt = match::GetResultsOp::create(
              builder, loc, match::OptionalType::get(innerType), pos,
              /*index=*/IntegerAttr());
          pos = emitNavAndUnwrap(resOpt, innerType);
        }
        valueMap.try_emplace(value, pos);
      });
}

void MatchEmitter::emitNonTreePredicates() {
  for (Operation &op : pattern.getBodyRegion().getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case([&](pdl::ApplyNativeConstraintOp constraintOp) {
          // Collect arguments. If any argument is not yet mapped (e.g. a
          // standalone literal), skip the constraint. (Well-formed PDL
          // shouldn't trigger this.)
          SmallVector<Value> args;
          args.reserve(constraintOp.getArgs().size());
          for (Value arg : constraintOp.getArgs()) {
            Value mapped = valueMap.lookup(arg);
            if (!mapped)
              return;
            args.push_back(mapped);
          }

          SmallVector<Type> resultTypes(constraintOp.getResultTypes().begin(),
                                        constraintOp.getResultTypes().end());

          auto nativeConstr = match::ApplyNativeConstraintOp::create(
              builder, loc, resultTypes, constraintOp.getNameAttr(), args,
              constraintOp.getIsNegatedAttr());

          // Register constraint results.
          for (auto [i, result] : llvm::enumerate(constraintOp.getResults())) {
            Value constrResult = nativeConstr.getConstraintResults()[i];
            (void)getOrRegister(result, constrResult);
          }
        })
        .Case([&](pdl::ApplyNativeRewriteOp rewriteOp) {
          // A native rewrite used during matching is a value producer (it
          // cannot fail). `pdl.apply_native_rewrite` is normally confined to
          // `pdl.rewrite` regions, but `match.apply_native_rewrite` is not, so
          // lower it here for completeness when it does appear in the matcher.
          // As with constraints, skip if any argument is not yet mapped.
          SmallVector<Value> args;
          args.reserve(rewriteOp.getArgs().size());
          for (Value arg : rewriteOp.getArgs()) {
            Value mapped = valueMap.lookup(arg);
            if (!mapped)
              return;
            args.push_back(mapped);
          }

          SmallVector<Type> resultTypes(rewriteOp.getResultTypes().begin(),
                                        rewriteOp.getResultTypes().end());

          auto nativeRewrite = match::ApplyNativeRewriteOp::create(
              builder, loc, resultTypes, rewriteOp.getNameAttr(), args);

          // Register rewrite results.
          for (auto [i, result] : llvm::enumerate(rewriteOp.getResults())) {
            Value rewriteResult = nativeRewrite.getResults()[i];
            (void)getOrRegister(result, rewriteResult);
          }
        })
        .Case([&](pdl::ResultOp resultOp) {
          if (valueMap.count(resultOp))
            return;
          Value parentVal = valueMap.lookup(resultOp.getParent());
          if (!parentVal)
            return;
          Value resOpt = match::GetResultOp::create(
              builder, loc,
              match::OptionalType::get(builder.getType<pdl::ValueType>()),
              parentVal, builder.getI32IntegerAttr(resultOp.getIndex()));
          Value resultVal =
              emitNavAndUnwrap(resOpt, builder.getType<pdl::ValueType>());
          valueMap[resultOp] = resultVal;
        })
        .Case([&](pdl::ResultsOp resultOp) {
          if (valueMap.count(resultOp))
            return;
          Value parentVal = valueMap.lookup(resultOp.getParent());
          if (!parentVal)
            return;
          bool isVariadic = isa<pdl::RangeType>(resultOp.getType());
          std::optional<unsigned> index = resultOp.getIndex();
          Type innerType = isVariadic
                               ? (Type)pdl::RangeType::get(
                                     builder.getType<pdl::ValueType>())
                               : (Type)builder.getType<pdl::ValueType>();
          Value resOpt = match::GetResultsOp::create(
              builder, loc, match::OptionalType::get(innerType), parentVal,
              index ? builder.getI32IntegerAttr(*index) : IntegerAttr());
          Value resultVal = emitNavAndUnwrap(resOpt, innerType);
          valueMap[resultOp] = resultVal;
        });
  }
}

match::MatcherOp MatchEmitter::emit() {
  SmallVector<Value> roots = detectRoots(pattern);

  // Build root ordering for multi-root patterns.
  RootOrderingGraph graph;
  ParentMaps parentMaps;
  buildCostGraph(roots, graph, parentMaps);

  // Solve optimal branching.
  Value bestRoot = pattern.getRewriter().getRoot();
  OptimalBranching::EdgeList bestEdges;
  if (!bestRoot) {
    unsigned bestCost = UINT_MAX;
    for (Value root : roots) {
      OptimalBranching solver(graph, root);
      unsigned cost = solver.solve();
      if (!bestRoot || cost < bestCost) {
        bestCost = cost;
        bestRoot = root;
        bestEdges = solver.preOrderTraversal(roots);
      }
    }
  } else {
    OptimalBranching solver(graph, bestRoot);
    solver.solve();
    bestEdges = solver.preOrderTraversal(roots);
  }

  // Create the `match.matcher` op with the pattern's symbol name.
  auto matcher =
      match::MatcherOp::create(builder, loc, pattern.getSymNameAttr());

  // Create the body block with a single `!pdl.operation` argument (the root).
  Block *body = &matcher.getBodyRegion().emplaceBlock();
  Value rootArg = body->addArgument(builder.getType<pdl::OperationType>(), loc);

  // Save insertion point and set to the body block.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);

  // Register the best root.
  valueMap[bestRoot] = rootArg;

  // Phase 1: downward tree predicates from root.
  auto rootOp = cast<pdl::OperationOp>(bestRoot.getDefiningOp());
  emitOperationConstraints(rootOp, rootArg);

  // Phase 2: upward traversals for multi-root.
  for (const auto &it : llvm::enumerate(bestEdges)) {
    Value target = it.value().first;
    Value source = it.value().second;

    if (valueMap.count(target))
      continue;

    Value connector = graph[target][source].connector;
    assert(connector && "invalid edge");
    Value pos = valueMap.lookup(connector);
    assert(pos && "connector has not been traversed yet");

    DenseMap<Value, OpIndex> parentMap = parentMaps.lookup(target);
    for (Value value = connector; value != target;) {
      OpIndex opIndex = parentMap.lookup(value);
      assert(opIndex.parent && "missing parent");
      emitUpwardTraversal(opIndex, pos, it.index());
      value = opIndex.parent;
    }
  }

  // Phase 3: non-tree predicates (standalone results, native constraints).
  emitNonTreePredicates();

  // Lower the pattern's rewrite region into a `pdl_interp.func` in the
  // rewriter module, reusing the shared rewriter generator. This mirrors what
  // the `pdl -> pdl_interp` pass would emit, so the success op can reference
  // it directly with the correct arguments.
  SmallVector<Value, 8> usedMatchValues;
  SymbolRefAttr rewriterRef = pdl_to_pdl_interp::generatePatternRewriter(
      pattern, rewriterModule, rewriterSymbolTable, builder, usedMatchValues);

  // Translate the pdl values used by the rewriter (from the match region)
  // into the corresponding match SSA values.
  SmallVector<Value, 8> mappedInputs;
  mappedInputs.reserve(usedMatchValues.size());
  for (Value matchValue : usedMatchValues) {
    Value mapped = valueMap.lookup(matchValue);
    assert(mapped && "rewriter uses a match value not produced by the pattern");
    mappedInputs.push_back(mapped);
  }

  // Emit the success terminator. The benefit comes from the original pattern.
  IntegerAttr benefitAttr =
      builder.getIntegerAttr(builder.getIntegerType(16), pattern.getBenefit());
  match::SuccessOp::create(builder, loc, rewriterRef, benefitAttr,
                                mappedInputs);

  return matcher;
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PDLToMatchPass
    : public impl::ConvertPDLToMatchPassBase<PDLToMatchPass> {
  void runOnOperation() final;
};
} // namespace

void PDLToMatchPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Create a nested module to hold the rewriter functions invoked after a
  // successful match. The same naming convention as the pdl_interp pipeline
  // is used so that downstream consumers (and the eventual `match ->
  // pdl_interp` lowering) can find them under `@rewriters::@<pattern>`.
  builder.setInsertionPointToStart(module.getBody());
  ModuleOp rewriterModule =
      ModuleOp::create(builder, module.getLoc(),
                       pdl_interp::PDLInterpDialect::getRewriterModuleName());
  SymbolTable rewriterSymbolTable(rewriterModule);

  for (pdl::PatternOp pattern :
       llvm::make_early_inc_range(module.getOps<pdl::PatternOp>())) {
    builder.setInsertionPoint(pattern);
    MatchEmitter emitter(builder, pattern, rewriterModule,
                             rewriterSymbolTable);
    emitter.emit();
    pattern.erase();
  }
}
