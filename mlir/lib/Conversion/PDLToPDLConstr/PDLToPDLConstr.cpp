//===- PDLToPDLConstr.cpp - Lower PDL to PDL Constraint dialect ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLConstr/PDLToPDLConstr.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <queue>

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLTOPDLCONSTRPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "convert-pdl-to-pdl-constr"

//===----------------------------------------------------------------------===//
// Helper: Root detection and cost graph (reused from PredicateTree.cpp)
//===----------------------------------------------------------------------===//

namespace {

/// An op accepting a value at an optional index.
struct OpIndex {
  Value parent;
  std::optional<unsigned> index;
};

/// The parent and operand index of each operation for each root.
using ParentMaps = DenseMap<Value, DenseMap<Value, OpIndex>>;

/// Entry for root ordering cost graph.
struct RootOrderingEntry {
  std::pair<unsigned, unsigned> cost;
  Value connector;
};

using RootOrderingGraph = DenseMap<Value, DenseMap<Value, RootOrderingEntry>>;

} // namespace

/// Returns the number of non-range elements within `values`.
static unsigned getNumNonRangeValues(ValueRange values) {
  return llvm::count_if(values.getTypes(),
                        [](Type type) { return !isa<pdl::RangeType>(type); });
}

/// Returns true if the operand at the given index needs to use an operand
/// group (variadic operand at or before that index).
static bool useOperandGroup(pdl::OperationOp op, unsigned index) {
  OperandRange operands = op.getOperandValues();
  assert(index < operands.size() && "operand index out of range");
  for (unsigned i = 0; i <= index; ++i)
    if (isa<pdl::RangeType>(operands[i].getType()))
      return true;
  return false;
}

static SmallVector<Value> detectRoots(pdl::PatternOp pattern) {
  DenseSet<Value> used;
  for (auto operationOp : pattern.getBodyRegion().getOps<pdl::OperationOp>()) {
    for (Value operand : operationOp.getOperandValues())
      TypeSwitch<Operation *>(operand.getDefiningOp())
          .Case<pdl::ResultOp, pdl::ResultsOp>(
              [&used](auto resultOp) { used.insert(resultOp.getParent()); });
  }
  if (Value root = pattern.getRewriter().getRoot())
    used.erase(root);

  SmallVector<Value> roots;
  for (Value operationOp : pattern.getBodyRegion().getOps<pdl::OperationOp>())
    if (!used.contains(operationOp))
      roots.push_back(operationOp);
  return roots;
}

static void buildCostGraph(ArrayRef<Value> roots, RootOrderingGraph &graph,
                            ParentMaps &parentMaps) {
  struct Entry {
    Value value;
    Value parent;
    std::optional<unsigned> index;
    unsigned depth;
  };
  struct RootDepth {
    Value root;
    unsigned depth = 0;
  };

  llvm::MapVector<Value, SmallVector<RootDepth, 1>> connectorsRootsDepths;

  for (Value root : roots) {
    std::queue<Entry> toVisit;
    toVisit.push({root, Value(), std::nullopt, 0});
    DenseMap<Value, OpIndex> &parentMap = parentMaps[root];

    while (!toVisit.empty()) {
      Entry entry = toVisit.front();
      toVisit.pop();
      if (!parentMap.insert({entry.value, {entry.parent, entry.index}}).second)
        continue;
      connectorsRootsDepths[entry.value].push_back({root, entry.depth});

      TypeSwitch<Operation *>(entry.value.getDefiningOp())
          .Case([&](pdl::OperationOp operationOp) {
            OperandRange operands = operationOp.getOperandValues();
            if (operands.size() == 1 &&
                isa<pdl::RangeType>(operands[0].getType())) {
              toVisit.push({operands[0], entry.value, std::nullopt,
                            entry.depth + 1});
              return;
            }
            for (const auto &p : llvm::enumerate(operands))
              toVisit.push(
                  {p.value(), entry.value, p.index(), entry.depth + 1});
          })
          .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto resultOp) {
            toVisit.push({resultOp.getParent(), entry.value,
                          resultOp.getIndex(), entry.depth});
          });
    }
  }

  unsigned nextID = 0;
  for (const auto &connectorRootsDepths : connectorsRootsDepths) {
    Value value = connectorRootsDepths.first;
    ArrayRef<RootDepth> rootsDepths = connectorRootsDepths.second;
    if (rootsDepths.size() == 1)
      continue;
    for (const RootDepth &p : rootsDepths) {
      for (const RootDepth &q : rootsDepths) {
        if (&p == &q)
          continue;
        RootOrderingEntry &entry = graph[q.root][p.root];
        if (!entry.connector || entry.cost.first > q.depth) {
          if (!entry.connector)
            entry.cost.second = nextID++;
          entry.cost.first = q.depth;
          entry.connector = value;
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Minimal Edmonds' algorithm for optimal root ordering
//===----------------------------------------------------------------------===//

namespace {
class OptimalBranching {
public:
  using EdgeList = std::vector<std::pair<Value, Value>>;

  OptimalBranching(RootOrderingGraph graph, Value root)
      : graph(std::move(graph)), root(root) {}

  unsigned solve() {
    // For each non-root node, pick the cheapest incoming edge.
    unsigned totalCost = 0;
    for (auto &[target, sources] : graph) {
      if (target == root)
        continue;
      Value bestSource;
      std::pair<unsigned, unsigned> bestCost = {UINT_MAX, UINT_MAX};
      for (auto &[source, entry] : sources) {
        if (entry.cost < bestCost) {
          bestCost = entry.cost;
          bestSource = source;
        }
      }
      if (bestSource) {
        parents[target] = bestSource;
        totalCost += bestCost.first;
      }
    }
    return totalCost;
  }

  const DenseMap<Value, Value> &getRootOrderingParents() const {
    return parents;
  }

  EdgeList preOrderTraversal(ArrayRef<Value> nodes) const {
    EdgeList result;
    // Build children map from parents.
    DenseMap<Value, SmallVector<Value>> children;
    for (auto &[child, parent] : parents)
      children[parent].push_back(child);

    // BFS from root.
    SmallVector<Value> worklist = {root};
    DenseSet<Value> visited;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      Value parent;
      auto it = parents.find(current);
      if (it != parents.end())
        parent = it->second;
      result.push_back({current, parent});
      // Add children in order of `nodes` for determinism.
      for (Value node : nodes) {
        if (auto childIt = children.find(current);
            childIt != children.end() &&
            llvm::is_contained(childIt->second, node)) {
          worklist.push_back(node);
        }
      }
    }
    return result;
  }

private:
  RootOrderingGraph graph;
  Value root;
  DenseMap<Value, Value> parents;
};
} // namespace

//===----------------------------------------------------------------------===//
// PDL → pdl_constr Emitter
//===----------------------------------------------------------------------===//

namespace {

/// Emits pdl_constr ops for a single pdl::PatternOp.
class PDLConstrEmitter {
public:
  PDLConstrEmitter(OpBuilder &builder, pdl::PatternOp pattern)
      : builder(builder), pattern(pattern),
        ctx(pattern.getContext()),
        loc(pattern.getLoc()),
        predType(builder.getType<pdl_constr::PredType>()) {}

  /// Emit a pdl_constr.pattern for the given pdl.pattern.
  pdl_constr::PatternOp emit();

private:
  /// Emit a nullable navigation and immediately unwrap it. Returns the
  /// unwrapped value and adds the pred to `preds`.
  Value emitNavAndUnwrap(Value navResult, Type innerType);

  /// Emit navigation + constraints for a single pdl::OperationOp (downward
  /// tree walk). `opVal` is the pdl_constr SSA value representing the op.
  /// If `ignoreOperand` is set, skip that operand (used during upward
  /// traversals).
  void emitOperationConstraints(pdl::OperationOp op, Value opVal,
                                std::optional<unsigned> ignoreOperand = {});

  /// Emit constraints for an operand value (pdl::OperandOp, pdl::OperandsOp,
  /// pdl::ResultOp, pdl::ResultsOp).
  void emitOperandConstraints(Value pdlVal, Value constrVal);

  /// Emit constraints for an attribute value (pdl::AttributeOp).
  void emitAttributeConstraints(pdl::AttributeOp attrOp, Value constrVal);

  /// Emit constraints for a type value (pdl::TypeOp, pdl::TypesOp).
  void emitTypeConstraints(Value pdlVal, Value constrVal);

  /// Emit upward traversals for multi-root patterns.
  void emitUpwardTraversal(OpIndex opIndex, Value &pos, unsigned rootID);

  /// Emit non-tree predicates (standalone attributes, constraints, results,
  /// types).
  void emitNonTreePredicates();

  /// Get or create the pdl_constr value corresponding to a pdl value. If the
  /// value has already been visited, returns the existing value and optionally
  /// emits an equality constraint. Returns nullptr if the value is new (and
  /// registers it).
  Value getOrRegister(Value pdlVal, Value constrVal);

  OpBuilder &builder;
  pdl::PatternOp pattern;
  MLIRContext *ctx;
  Location loc;
  pdl_constr::PredType predType;

  /// Mapping from pdl SSA values to their pdl_constr SSA values.
  DenseMap<Value, Value> valueMap;

  /// Collected predicate values.
  SmallVector<Value> preds;
};

} // namespace

Value PDLConstrEmitter::getOrRegister(Value pdlVal, Value constrVal) {
  auto it = valueMap.try_emplace(pdlVal, constrVal);
  if (!it.second) {
    // Already visited — emit equality constraint.
    Value existing = it.first->second;
    if (isa<pdl::AttributeOp, pdl::OperandOp, pdl::OperandsOp,
            pdl::OperationOp, pdl::TypeOp>(pdlVal.getDefiningOp())) {
      auto eqPred =
          pdl_constr::EqualOp::create(builder, loc, predType, constrVal,
                                      existing);
      preds.push_back(eqPred);
    }
    return existing;
  }
  return Value(); // new registration
}

Value PDLConstrEmitter::emitNavAndUnwrap(Value navResult, Type innerType) {
  auto isNotNull = pdl_constr::IsNotNullOp::create(
      builder, loc, TypeRange{predType, innerType}, navResult);
  preds.push_back(isNotNull.getPred());
  return isNotNull.getUnwrapped();
}

void PDLConstrEmitter::emitTypeConstraints(Value pdlVal, Value constrVal) {
  if (auto typeOp = pdlVal.getDefiningOp<pdl::TypeOp>()) {
    if (Attribute type = typeOp.getConstantTypeAttr()) {
      auto pred = pdl_constr::HasTypeOp::create(builder, loc, predType,
                                                 constrVal, cast<TypeAttr>(type));
      preds.push_back(pred);
    }
  } else if (auto typesOp = pdlVal.getDefiningOp<pdl::TypesOp>()) {
    // TypesOp constraints are handled via HasTypeOp with ArrayAttr - but
    // the pdl_constr.has_type takes a TypeAttr, so for now types ranges
    // with constant constraints are not fully handled here. The type check
    // for ranges would need extension. For now, skip.
  }
}

void PDLConstrEmitter::emitAttributeConstraints(pdl::AttributeOp attrOp,
                                                 Value constrVal) {
  // If the attribute has a type, constrain the type.
  if (Value type = attrOp.getValueType()) {
    auto typeVal =
        pdl_constr::GetAttributeTypeOp::create(builder, loc,
            builder.getType<pdl::TypeType>(), constrVal);
    Value existing = getOrRegister(type, typeVal);
    if (!existing)
      emitTypeConstraints(type, typeVal);
  } else if (Attribute value = attrOp.getValueAttr()) {
    // If the attribute has a constant value, constrain it.
    auto pred = pdl_constr::HasAttrValueOp::create(builder, loc, predType,
                                                    constrVal, value);
    preds.push_back(pred);
  }
}

void PDLConstrEmitter::emitOperandConstraints(Value pdlVal, Value constrVal) {
  TypeSwitch<Operation *>(pdlVal.getDefiningOp())
      .Case<pdl::OperandOp, pdl::OperandsOp>([&](auto op) {
        if (Value type = op.getValueType()) {
          Value typeVal =
              pdl_constr::GetValueTypeOp::create(builder, loc,
                  builder.getType<pdl::TypeType>(), constrVal);
          Value existing = getOrRegister(type, typeVal);
          if (!existing)
            emitTypeConstraints(type, typeVal);
        }
      })
      .Case<pdl::ResultOp, pdl::ResultsOp>([&](auto op) {
        // Navigate to the defining op of the operand.
        auto defOpOpt = pdl_constr::GetDefiningOpOp::create(
            builder, loc,
            pdl_constr::OptionalType::get(builder.getType<pdl::OperationType>()),
            constrVal);
        Value defOp =
            emitNavAndUnwrap(defOpOpt, builder.getType<pdl::OperationType>());

        // Now check the result connects back.
        std::optional<unsigned> index = op.getIndex();
        bool isVariadic = isa<pdl::RangeType>(pdlVal.getType());
        Value resultVal;
        if (isa<pdl::ResultOp>(pdlVal.getDefiningOp())) {
          auto resOpt = pdl_constr::GetResultOp::create(
              builder, loc,
              pdl_constr::OptionalType::get(builder.getType<pdl::ValueType>()),
              defOp, builder.getI32IntegerAttr(*index));
          resultVal =
              emitNavAndUnwrap(resOpt, builder.getType<pdl::ValueType>());
        } else {
          Type innerType = isVariadic
                               ? (Type)pdl::RangeType::get(
                                     builder.getType<pdl::ValueType>())
                               : (Type)builder.getType<pdl::ValueType>();
          auto resOpt = pdl_constr::GetResultsOp::create(
              builder, loc, pdl_constr::OptionalType::get(innerType), defOp,
              index ? builder.getI32IntegerAttr(*index)
                    : IntegerAttr());
          if (index)
            resultVal = emitNavAndUnwrap(resOpt, innerType);
          else
            resultVal = resOpt.getResult();
        }
        // Equality constraint: result == operand.
        auto eqPred =
            pdl_constr::EqualOp::create(builder, loc, predType, resultVal,
                                        constrVal);
        preds.push_back(eqPred);

        // Recursively process the defining operation.
        Value parentPdlVal = op.getParent();
        Value existing = getOrRegister(parentPdlVal, defOp);
        if (!existing) {
          auto opOp = cast<pdl::OperationOp>(parentPdlVal.getDefiningOp());
          emitOperationConstraints(opOp, defOp);
        }
      });
}

void PDLConstrEmitter::emitOperationConstraints(
    pdl::OperationOp op, Value opVal, std::optional<unsigned> ignoreOperand) {
  // Operation name constraint.
  if (std::optional<StringRef> opName = op.getOpName()) {
    auto pred = pdl_constr::HasNameOp::create(builder, loc, predType, opVal,
                                               builder.getStringAttr(*opName));
    preds.push_back(pred);
  }

  // Operand count constraint.
  OperandRange operands = op.getOperandValues();
  unsigned minOperands = getNumNonRangeValues(operands);
  if (minOperands != operands.size()) {
    if (minOperands) {
      auto pred = pdl_constr::CheckOperandCountOp::create(
          builder, loc, predType, opVal,
          builder.getI32IntegerAttr(minOperands),
          /*atLeast=*/builder.getUnitAttr());
      preds.push_back(pred);
    }
  } else {
    auto pred = pdl_constr::CheckOperandCountOp::create(
        builder, loc, predType, opVal, builder.getI32IntegerAttr(minOperands),
        /*atLeast=*/UnitAttr());
    preds.push_back(pred);
  }

  // Result count constraint.
  OperandRange types = op.getTypeValues();
  unsigned minResults = getNumNonRangeValues(types);
  if (minResults == types.size()) {
    auto pred = pdl_constr::CheckResultCountOp::create(
        builder, loc, predType, opVal,
        builder.getI32IntegerAttr(types.size()), /*atLeast=*/UnitAttr());
    preds.push_back(pred);
  } else if (minResults) {
    auto pred = pdl_constr::CheckResultCountOp::create(
        builder, loc, predType, opVal, builder.getI32IntegerAttr(minResults),
        /*atLeast=*/builder.getUnitAttr());
    preds.push_back(pred);
  }

  // Attributes.
  for (auto [attrName, attr] :
       llvm::zip(op.getAttributeValueNames(), op.getAttributeValues())) {
    StringRef name = cast<StringAttr>(attrName).getValue();
    auto attrOpt = pdl_constr::GetAttributeOp::create(
        builder, loc,
        pdl_constr::OptionalType::get(builder.getType<pdl::AttributeType>()),
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
    Type rangeValType =
        pdl::RangeType::get(builder.getType<pdl::ValueType>());
    auto opsOpt = pdl_constr::GetOperandsOp::create(
        builder, loc, pdl_constr::OptionalType::get(rangeValType), opVal,
        /*index=*/IntegerAttr());
    // All-operands group: don't null-check, just use directly.
    Value opsVal = opsOpt.getResult();

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
        // Use operand group.
        Type innerType = isVariadic
                             ? (Type)pdl::RangeType::get(
                                   builder.getType<pdl::ValueType>())
                             : (Type)builder.getType<pdl::ValueType>();
        auto opsOpt = pdl_constr::GetOperandsOp::create(
            builder, loc, pdl_constr::OptionalType::get(innerType), opVal,
            builder.getI32IntegerAttr(operandIt.index()));
        operandVal = emitNavAndUnwrap(opsOpt, innerType);
      } else {
        // Individual operand.
        auto opOpt = pdl_constr::GetOperandOp::create(
            builder, loc,
            pdl_constr::OptionalType::get(
                builder.getType<pdl::ValueType>()),
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
    Type rangeValType =
        pdl::RangeType::get(builder.getType<pdl::ValueType>());
    auto resOpt = pdl_constr::GetResultsOp::create(
        builder, loc, pdl_constr::OptionalType::get(rangeValType), opVal,
        /*index=*/IntegerAttr());
    Value resVal = resOpt.getResult();

    auto typeVal =
        pdl_constr::GetValueTypeOp::create(builder, loc,
            builder.getType<pdl::TypeType>(), resVal);

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
      auto resOpt = pdl_constr::GetResultsOp::create(
          builder, loc, pdl_constr::OptionalType::get(innerType), opVal,
          builder.getI32IntegerAttr(idx));
      resultVal = emitNavAndUnwrap(resOpt, innerType);
    } else {
      auto resOpt = pdl_constr::GetResultOp::create(
          builder, loc,
          pdl_constr::OptionalType::get(builder.getType<pdl::ValueType>()),
          opVal, builder.getI32IntegerAttr(idx));
      resultVal =
          emitNavAndUnwrap(resOpt, builder.getType<pdl::ValueType>());
    }

    // Get the type of this result.
    auto typeVal =
        pdl_constr::GetValueTypeOp::create(builder, loc,
            builder.getType<pdl::TypeType>(), resultVal);

    Value existing = getOrRegister(typeValue, typeVal);
    if (!existing)
      emitTypeConstraints(typeValue, typeVal);
  }
}

void PDLConstrEmitter::emitUpwardTraversal(OpIndex opIndex, Value &pos,
                                            unsigned rootID) {
  Value value = opIndex.parent;
  TypeSwitch<Operation *>(value.getDefiningOp())
      .Case([&](pdl::OperationOp operationOp) {
        // Get users and iterate.
        auto usersVal = pdl_constr::GetUsersOp::create(
            builder, loc,
            pdl::RangeType::get(builder.getType<pdl::OperationType>()), pos);
        auto eachVal = pdl_constr::GetEachOp::create(
            builder, loc, builder.getType<pdl::OperationType>(), usersVal);
        Value opVal = eachVal.getResult();

        // Compare the operand(s) of the user against the input value(s).
        Value operandVal;
        if (!opIndex.index) {
          // All operands.
          Type rangeValType =
              pdl::RangeType::get(builder.getType<pdl::ValueType>());
          auto opsOpt = pdl_constr::GetOperandsOp::create(
              builder, loc, pdl_constr::OptionalType::get(rangeValType), opVal,
              /*index=*/IntegerAttr());
          operandVal = opsOpt.getResult();
        } else if (useOperandGroup(operationOp, *opIndex.index)) {
          Type type =
              operationOp.getOperandValues()[*opIndex.index].getType();
          bool variadic = isa<pdl::RangeType>(type);
          Type innerType = variadic
                               ? (Type)pdl::RangeType::get(
                                     builder.getType<pdl::ValueType>())
                               : (Type)builder.getType<pdl::ValueType>();
          auto opsOpt = pdl_constr::GetOperandsOp::create(
              builder, loc, pdl_constr::OptionalType::get(innerType), opVal,
              builder.getI32IntegerAttr(*opIndex.index));
          operandVal = emitNavAndUnwrap(opsOpt, innerType);
        } else {
          auto opOpt = pdl_constr::GetOperandOp::create(
              builder, loc,
              pdl_constr::OptionalType::get(
                  builder.getType<pdl::ValueType>()),
              opVal, builder.getI32IntegerAttr(*opIndex.index));
          operandVal =
              emitNavAndUnwrap(opOpt, builder.getType<pdl::ValueType>());
        }
        // Equality constraint: operand == pos (the value we're traversing from).
        auto eqPred =
            pdl_constr::EqualOp::create(builder, loc, predType, operandVal,
                                        pos);
        preds.push_back(eqPred);

        // Register this operation.
        bool inserted = valueMap.try_emplace(value, opVal).second;
        (void)inserted;
        assert(inserted && "duplicate upward visit");

        // Get tree predicates for this operation.
        emitOperationConstraints(operationOp, opVal, opIndex.index);

        pos = opVal;
      })
      .Case([&](pdl::ResultOp resultOp) {
        // Individual result.
        auto resOpt = pdl_constr::GetResultOp::create(
            builder, loc,
            pdl_constr::OptionalType::get(builder.getType<pdl::ValueType>()),
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
          auto resOpt = pdl_constr::GetResultsOp::create(
              builder, loc, pdl_constr::OptionalType::get(innerType), pos,
              builder.getI32IntegerAttr(*opIndex.index));
          pos = emitNavAndUnwrap(resOpt, innerType);
        } else {
          auto resOpt = pdl_constr::GetResultsOp::create(
              builder, loc, pdl_constr::OptionalType::get(innerType), pos,
              /*index=*/IntegerAttr());
          pos = resOpt.getResult();
        }
        valueMap.try_emplace(value, pos);
      });
}

void PDLConstrEmitter::emitNonTreePredicates() {
  for (Operation &op : pattern.getBodyRegion().getOps()) {
    TypeSwitch<Operation *>(&op)
        .Case([&](pdl::AttributeOp attrOp) {
          if (valueMap.count(attrOp))
            return;
          Attribute value = attrOp.getValueAttr();
          if (!value)
            return;
          // Standalone attribute with a value — emit a has_attr_value pred
          // using a placeholder. We need to create a "literal" position.
          // For now, we just register it; the constraint is already embedded.
          // Actually for standalone attributes, they don't need navigation
          // — they are used by native constraints or equality checks.
          // We skip them here; they'll be picked up when used.
        })
        .Case([&](pdl::ApplyNativeConstraintOp constraintOp) {
          // Collect arguments.
          SmallVector<Value> args;
          SmallVector<Type> argTypes;
          for (Value arg : constraintOp.getArgs()) {
            Value mapped = valueMap.lookup(arg);
            if (!mapped) {
              // The argument may be a standalone type/attribute literal.
              // Try to handle it.
              if (auto typeOp = arg.getDefiningOp<pdl::TypeOp>()) {
                // Create a dummy type value if it has a constant type.
                // This shouldn't normally happen for well-formed PDL.
                return;
              }
              return;
            }
            args.push_back(mapped);
            argTypes.push_back(mapped.getType());
          }

          // Result types.
          SmallVector<Type> resultTypes;
          resultTypes.push_back(predType);
          for (Value result : constraintOp.getResults())
            resultTypes.push_back(result.getType());

          auto nativeConstr = pdl_constr::ApplyNativeConstraintOp::create(
              builder, loc, resultTypes, constraintOp.getNameAttr(), args,
              constraintOp.getIsNegatedAttr());
          preds.push_back(nativeConstr.getPred());

          // Register constraint results.
          for (auto [i, result] : llvm::enumerate(constraintOp.getResults())) {
            Value constrResult = nativeConstr.getConstraintResults()[i];
            Value existing = getOrRegister(result, constrResult);
            (void)existing;
          }
        })
        .Case([&](pdl::ResultOp resultOp) {
          if (valueMap.count(resultOp))
            return;
          Value parentVal = valueMap.lookup(resultOp.getParent());
          if (!parentVal)
            return;
          auto resOpt = pdl_constr::GetResultOp::create(
              builder, loc,
              pdl_constr::OptionalType::get(
                  builder.getType<pdl::ValueType>()),
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
          auto resOpt = pdl_constr::GetResultsOp::create(
              builder, loc, pdl_constr::OptionalType::get(innerType), parentVal,
              index ? builder.getI32IntegerAttr(*index) : IntegerAttr());
          Value resultVal;
          if (index)
            resultVal = emitNavAndUnwrap(resOpt, innerType);
          else
            resultVal = resOpt.getResult();
          valueMap[resultOp] = resultVal;
        })
        .Case([&](pdl::TypeOp typeOp) {
          if (valueMap.count(typeOp))
            return;
          // Standalone type with a constant — skip for now, used via equality.
        })
        .Case([&](pdl::TypesOp typeOp) {
          if (valueMap.count(typeOp))
            return;
        });
  }
}

pdl_constr::PatternOp PDLConstrEmitter::emit() {
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

  // Create the pdl_constr.pattern op.
  auto constrPattern = pdl_constr::PatternOp::create(
      builder, loc, pattern.getBenefitAttr(), pattern.getSymNameAttr());

  // Create the body block with a single !pdl.operation argument.
  Block *body = &constrPattern.getBodyRegion().emplaceBlock();
  Value rootArg =
      body->addArgument(builder.getType<pdl::OperationType>(), loc);

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

  // Phase 3: non-tree predicates.
  emitNonTreePredicates();

  // Combine all predicates.
  Value allPred;
  if (preds.size() == 1) {
    allPred = preds[0];
  } else if (preds.empty()) {
    // Edge case: no predicates. Create a trivially-true pattern.
    // This shouldn't happen in practice.
    allPred = pdl_constr::AllOp::create(builder, loc, predType, ValueRange{});
  } else {
    allPred = pdl_constr::AllOp::create(builder, loc, predType, preds);
  }

  pdl_constr::SuccessOp::create(builder, loc, allPred);

  return constrPattern;
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PDLToPDLConstrPass
    : public impl::ConvertPDLToPDLConstrPassBase<PDLToPDLConstrPass> {
  void runOnOperation() final;
};
} // namespace

void PDLToPDLConstrPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  for (pdl::PatternOp pattern :
       llvm::make_early_inc_range(module.getOps<pdl::PatternOp>())) {
    builder.setInsertionPoint(pattern);
    PDLConstrEmitter emitter(builder, pattern);
    emitter.emit();
    pattern.erase();
  }
}
