//===- PDLConstrToPDLInterp.cpp - Lower pdl_constr to pdl_interp ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass combines multiple `pdl_constr.pattern` ops into a single
// `pdl_interp` matcher function. The merging algorithm reuses the existing
// `PositionalPredicate` machinery developed for the `pdl -> pdl_interp`
// pipeline: we reconstruct the predicate triples (Position, Question, Answer)
// from the `pdl_constr` IR, run them through the shared `OrderedPredicate`
// sort + `propagatePattern` tree construction, and then emit `pdl_interp`
// ops by walking the resulting matcher tree.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLConstrToPDLInterp/PDLConstrToPDLInterp.h"

// Predicate / matcher tree infrastructure shared with the
// `pdl -> pdl_interp` pipeline.
#include "../PDLToPDLInterp/Predicate.h"
#include "../PDLToPDLInterp/PredicateTree.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLCONSTRTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

#define DEBUG_TYPE "convert-pdl-constr-to-pdl-interp"

namespace {

//===----------------------------------------------------------------------===//
// ConstrPatternInfo
//===----------------------------------------------------------------------===//

/// All information extracted from a single `pdl_constr.pattern` that is
/// needed to build the merged matcher tree and to emit the final
/// `pdl_interp.record_match` op at the success site.
struct ConstrPatternInfo {
  /// The originating pattern op.
  pdl_constr::PatternOp pattern;

  /// The block argument representing the matched root operation. Stored only
  /// so we can fall back on its name attribute for the record_match
  /// `rootKind` field.
  Value rootBlockArg;

  /// The set of positional predicates extracted from the pattern body.
  std::vector<PositionalPredicate> predicates;

  /// The rewriter symbol referenced by `pdl_constr.success`.
  SymbolRefAttr rewriterRef;

  /// The positions of the values forwarded to the rewriter (the `inputs`
  /// operand list of `pdl_constr.success`), in order.
  SmallVector<Position *, 4> inputPositions;

  /// Operation-name constraint applied to the root, if any. Used for the
  /// `rootKind` attribute of `pdl_interp.record_match`.
  StringAttr rootKindAttr;
};

//===----------------------------------------------------------------------===//
// ConstrSuccessNode
//===----------------------------------------------------------------------===//

/// A matcher node that records a successful pdl_constr match. Carries a
/// pointer to the per-pattern info captured during predicate extraction.
struct ConstrSuccessNode : public MatcherNode {
  ConstrSuccessNode(ConstrPatternInfo *info,
                    std::unique_ptr<MatcherNode> failureNode)
      : MatcherNode(TypeID::get<ConstrSuccessNode>(), /*position=*/nullptr,
                    /*question=*/nullptr, std::move(failureNode)),
        info(info) {}

  static bool classof(const MatcherNode *node) {
    return node->getMatcherTypeID() == TypeID::get<ConstrSuccessNode>();
  }

  ConstrPatternInfo *info;
};

} // namespace

//===----------------------------------------------------------------------===//
// Predicate extraction
//===----------------------------------------------------------------------===//

namespace {

/// Walks the body of a single `pdl_constr.pattern`, building up the list of
/// positional predicates and the mapping from pdl_constr SSA values to their
/// matching `Position*`.
class PredicateExtractor {
public:
  PredicateExtractor(PredicateBuilder &builder, pdl_constr::PatternOp pattern,
                     ConstrPatternInfo &info,
                     DenseMap<Value, Position *> &valueToPosition)
      : builder(builder), pattern(pattern), info(info),
        valueToPosition(valueToPosition) {}

  LogicalResult extract();

private:
  /// Register a position for a pdl_constr SSA value. If the value is already
  /// mapped, emit an EqualToQuestion predicate so the two positions are
  /// constrained to be equal at runtime.
  void registerPosition(Value v, Position *pos);

  /// Look up the position previously registered for `v`. Returns null if not
  /// registered (which is a bug in the IR or the extractor).
  Position *lookupPosition(Value v) const {
    return valueToPosition.lookup(v);
  }

  /// Helpers for individual op kinds.
  void handleGetOperand(pdl_constr::GetOperandOp op);
  void handleGetOperands(pdl_constr::GetOperandsOp op);
  void handleGetResult(pdl_constr::GetResultOp op);
  void handleGetResults(pdl_constr::GetResultsOp op);
  void handleGetAttribute(pdl_constr::GetAttributeOp op);
  void handleGetDefiningOp(pdl_constr::GetDefiningOpOp op);
  void handleGetValueType(pdl_constr::GetValueTypeOp op);
  void handleGetAttributeType(pdl_constr::GetAttributeTypeOp op);
  void handleGetUsers(pdl_constr::GetUsersOp op);
  void handleGetEach(pdl_constr::GetEachOp op);
  void handleIsNotNull(pdl_constr::IsNotNullOp op);
  void handleHasName(pdl_constr::HasNameOp op);
  void handleEqual(pdl_constr::EqualOp op);
  void handleHasType(pdl_constr::HasTypeOp op);
  void handleHasAttrValue(pdl_constr::HasAttrValueOp op);
  void handleCheckOperandCount(pdl_constr::CheckOperandCountOp op);
  void handleCheckResultCount(pdl_constr::CheckResultCountOp op);
  void handleApplyNativeConstraint(pdl_constr::ApplyNativeConstraintOp op);
  void handleSuccess(pdl_constr::SuccessOp op);

  PredicateBuilder &builder;
  pdl_constr::PatternOp pattern;
  ConstrPatternInfo &info;

  /// Map from pdl_constr SSA value to its matching position. This is shared
  /// across patterns: the same position may be referenced by multiple
  /// pdl_constr SSA values across different patterns, but each pattern only
  /// inserts its own values.
  DenseMap<Value, Position *> &valueToPosition;

  /// Counter used to assign a unique id to every `get_each` inside a pattern,
  /// matching the role that `rootID` plays in upward traversals.
  unsigned foreachId = 0;
};

} // namespace

void PredicateExtractor::registerPosition(Value v, Position *pos) {
  auto it = valueToPosition.try_emplace(v, pos);
  if (it.second)
    return;
  // The value was already registered. Emit an equality predicate. We anchor
  // the predicate on the deeper of the two positions, mirroring the convention
  // used by the pdl -> pdl_interp extractor.
  Position *existing = it.first->second;
  Position *deeper = existing;
  Position *shallower = pos;
  if (pos->getOperationDepth() > existing->getOperationDepth()) {
    deeper = pos;
    shallower = existing;
  }
  info.predicates.emplace_back(deeper, builder.getEqualTo(shallower));
}

void PredicateExtractor::handleGetOperand(pdl_constr::GetOperandOp op) {
  auto *parentPos = dyn_cast_or_null<OperationPosition>(lookupPosition(op.getOp()));
  assert(parentPos && "operand parent must be an operation position");
  Position *pos = builder.getOperand(parentPos, op.getIndex());
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetOperands(pdl_constr::GetOperandsOp op) {
  auto *parentPos = dyn_cast_or_null<OperationPosition>(lookupPosition(op.getOp()));
  assert(parentPos && "operands parent must be an operation position");
  auto optType = cast<pdl_constr::OptionalType>(op.getResult().getType());
  bool isVariadic = isa<pdl::RangeType>(optType.getInnerType());
  std::optional<unsigned> index;
  if (auto idx = op.getIndex())
    index = *idx;
  Position *pos = builder.getOperandGroup(parentPos, index, isVariadic);
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetResult(pdl_constr::GetResultOp op) {
  auto *parentPos = dyn_cast_or_null<OperationPosition>(lookupPosition(op.getOp()));
  assert(parentPos && "result parent must be an operation position");
  Position *pos = builder.getResult(parentPos, op.getIndex());
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetResults(pdl_constr::GetResultsOp op) {
  auto *parentPos = dyn_cast_or_null<OperationPosition>(lookupPosition(op.getOp()));
  assert(parentPos && "results parent must be an operation position");
  auto optType = cast<pdl_constr::OptionalType>(op.getResult().getType());
  bool isVariadic = isa<pdl::RangeType>(optType.getInnerType());
  std::optional<unsigned> index;
  if (auto idx = op.getIndex())
    index = *idx;
  Position *pos = builder.getResultGroup(parentPos, index, isVariadic);
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetAttribute(pdl_constr::GetAttributeOp op) {
  auto *parentPos = dyn_cast_or_null<OperationPosition>(lookupPosition(op.getOp()));
  assert(parentPos && "attribute parent must be an operation position");
  Position *pos = builder.getAttribute(parentPos, op.getName());
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetDefiningOp(pdl_constr::GetDefiningOpOp op) {
  Position *valuePos = lookupPosition(op.getValue());
  assert(valuePos && "value must already be positioned");
  assert((isa<OperandPosition, OperandGroupPosition>(valuePos)) &&
         "get_defining_op only applies to operand positions");
  Position *pos = builder.getOperandDefiningOp(valuePos);
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetValueType(pdl_constr::GetValueTypeOp op) {
  Position *valuePos = lookupPosition(op.getValue());
  assert(valuePos && "value must already be positioned");
  Position *pos = builder.getType(valuePos);
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetAttributeType(
    pdl_constr::GetAttributeTypeOp op) {
  Position *attrPos = lookupPosition(op.getAttribute());
  assert(attrPos && "attribute must already be positioned");
  Position *pos = builder.getType(attrPos);
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetUsers(pdl_constr::GetUsersOp op) {
  Position *valuePos = lookupPosition(op.getValue());
  assert(valuePos && "value must already be positioned");
  // Upward traversal: always use a representative for ranges.
  Position *pos = builder.getUsers(valuePos, /*useRepresentative=*/true);
  registerPosition(op.getResult(), pos);
}

void PredicateExtractor::handleGetEach(pdl_constr::GetEachOp op) {
  Position *rangePos = lookupPosition(op.getRange());
  assert(rangePos && "range must already be positioned");
  Position *foreachPos = builder.getForEach(rangePos, foreachId++);
  OperationPosition *opPos = builder.getPassthroughOp(foreachPos);
  registerPosition(op.getResult(), opPos);
}

void PredicateExtractor::handleIsNotNull(pdl_constr::IsNotNullOp op) {
  Position *optPos = lookupPosition(op.getOptionalValue());
  assert(optPos && "optional must already be positioned");
  // Emit an IsNotNullQuestion predicate at the optional's position.
  info.predicates.emplace_back(optPos, builder.getIsNotNull());
  // The unwrapped value shares the same position (the optional and the
  // unwrapped value refer to the same runtime entity).
  registerPosition(op.getUnwrapped(), optPos);
}

void PredicateExtractor::handleHasName(pdl_constr::HasNameOp op) {
  Position *opPos = lookupPosition(op.getOp());
  assert(opPos && "operand must already be positioned");
  info.predicates.emplace_back(opPos, builder.getOperationName(op.getName()));

  // Capture the root kind for the eventual record_match attribute, if this
  // predicate targets the pattern's root.
  if (op.getOp() == info.rootBlockArg)
    info.rootKindAttr = op.getNameAttr();
}

void PredicateExtractor::handleEqual(pdl_constr::EqualOp op) {
  Position *lhs = lookupPosition(op.getLhs());
  Position *rhs = lookupPosition(op.getRhs());
  assert(lhs && rhs && "equality operands must be positioned");
  // Anchor the predicate at the deeper position.
  Position *deeper = lhs;
  Position *shallower = rhs;
  if (rhs->getOperationDepth() > lhs->getOperationDepth()) {
    deeper = rhs;
    shallower = lhs;
  }
  info.predicates.emplace_back(deeper, builder.getEqualTo(shallower));
}

void PredicateExtractor::handleHasType(pdl_constr::HasTypeOp op) {
  Position *typePos = lookupPosition(op.getType());
  assert(typePos && "type must already be positioned");
  info.predicates.emplace_back(
      typePos, builder.getTypeConstraint(op.getConstantTypeAttr()));
}

void PredicateExtractor::handleHasAttrValue(pdl_constr::HasAttrValueOp op) {
  Position *attrPos = lookupPosition(op.getAttribute());
  assert(attrPos && "attribute must already be positioned");
  info.predicates.emplace_back(
      attrPos, builder.getAttributeConstraint(op.getValue()));
}

void PredicateExtractor::handleCheckOperandCount(
    pdl_constr::CheckOperandCountOp op) {
  Position *opPos = lookupPosition(op.getOp());
  assert(opPos && "operand must already be positioned");
  if (op.getAtLeast())
    info.predicates.emplace_back(
        opPos, builder.getOperandCountAtLeast(op.getCount()));
  else
    info.predicates.emplace_back(opPos, builder.getOperandCount(op.getCount()));
}

void PredicateExtractor::handleCheckResultCount(
    pdl_constr::CheckResultCountOp op) {
  Position *opPos = lookupPosition(op.getOp());
  assert(opPos && "operand must already be positioned");
  if (op.getAtLeast())
    info.predicates.emplace_back(opPos,
                                 builder.getResultCountAtLeast(op.getCount()));
  else
    info.predicates.emplace_back(opPos, builder.getResultCount(op.getCount()));
}

void PredicateExtractor::handleApplyNativeConstraint(
    pdl_constr::ApplyNativeConstraintOp op) {
  SmallVector<Position *> argPositions;
  argPositions.reserve(op.getArgs().size());
  for (Value arg : op.getArgs()) {
    Position *p = lookupPosition(arg);
    assert(p && "constraint argument must be positioned");
    argPositions.push_back(p);
  }

  SmallVector<Type> resultTypes(op.getConstraintResults().getTypes());

  PredicateBuilder::Predicate pred = builder.getConstraint(
      op.getName(), argPositions, resultTypes, op.getIsNegated());

  // Anchor the constraint at the deepest argument position (matches the
  // convention used by the pdl -> pdl_interp extractor).
  Position *anchor = argPositions.empty()
                         ? builder.getRoot()
                         : *llvm::max_element(argPositions,
                                              [](Position *a, Position *b) {
                                                return a->getOperationDepth() <
                                                       b->getOperationDepth();
                                              });

  auto *cq = cast<ConstraintQuestion>(pred.first);
  for (auto [i, result] : llvm::enumerate(op.getConstraintResults())) {
    Position *cpos = builder.getConstraintPosition(cq, i);
    registerPosition(result, cpos);
  }
  info.predicates.emplace_back(anchor, pred);
}

void PredicateExtractor::handleSuccess(pdl_constr::SuccessOp op) {
  info.rewriterRef = op.getRewriter();
  info.inputPositions.clear();
  for (Value v : op.getInputs()) {
    Position *p = lookupPosition(v);
    assert(p && "success input must be positioned");
    info.inputPositions.push_back(p);
  }
}

LogicalResult PredicateExtractor::extract() {
  Block &block = pattern.getBodyRegion().front();
  Value rootArg = block.getArgument(0);
  info.rootBlockArg = rootArg;

  // The root block argument lives at the root operation position.
  registerPosition(rootArg, builder.getRoot());

  // Walk the body in program order. Navigation ops are pure and ordered
  // before any of their uses, so a single linear pass is sufficient.
  for (Operation &op : block) {
    TypeSwitch<Operation *>(&op)
        .Case<pdl_constr::GetOperandOp>(
            [&](auto o) { handleGetOperand(o); })
        .Case<pdl_constr::GetOperandsOp>(
            [&](auto o) { handleGetOperands(o); })
        .Case<pdl_constr::GetResultOp>([&](auto o) { handleGetResult(o); })
        .Case<pdl_constr::GetResultsOp>([&](auto o) { handleGetResults(o); })
        .Case<pdl_constr::GetAttributeOp>(
            [&](auto o) { handleGetAttribute(o); })
        .Case<pdl_constr::GetDefiningOpOp>(
            [&](auto o) { handleGetDefiningOp(o); })
        .Case<pdl_constr::GetValueTypeOp>(
            [&](auto o) { handleGetValueType(o); })
        .Case<pdl_constr::GetAttributeTypeOp>(
            [&](auto o) { handleGetAttributeType(o); })
        .Case<pdl_constr::GetUsersOp>([&](auto o) { handleGetUsers(o); })
        .Case<pdl_constr::GetEachOp>([&](auto o) { handleGetEach(o); })
        .Case<pdl_constr::IsNotNullOp>([&](auto o) { handleIsNotNull(o); })
        .Case<pdl_constr::HasNameOp>([&](auto o) { handleHasName(o); })
        .Case<pdl_constr::EqualOp>([&](auto o) { handleEqual(o); })
        .Case<pdl_constr::HasTypeOp>([&](auto o) { handleHasType(o); })
        .Case<pdl_constr::HasAttrValueOp>(
            [&](auto o) { handleHasAttrValue(o); })
        .Case<pdl_constr::CheckOperandCountOp>(
            [&](auto o) { handleCheckOperandCount(o); })
        .Case<pdl_constr::CheckResultCountOp>(
            [&](auto o) { handleCheckResultCount(o); })
        .Case<pdl_constr::ApplyNativeConstraintOp>(
            [&](auto o) { handleApplyNativeConstraint(o); })
        .Case<pdl_constr::SuccessOp>([&](auto o) { handleSuccess(o); })
        .Case<pdl_constr::AllOp, pdl_constr::AnyOp>([&](auto) {
          // Combinators have no effect on the matcher: every leaf predicate
          // is already captured. (See the pass-level comment on `any`.)
        })
        .Default([](Operation *unhandled) {
          // Unknown ops are passed through silently — useful while the
          // dialect is still evolving.
          (void)unhandled;
        });
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Matcher tree merging (mirrors PredicateTree.cpp)
//===----------------------------------------------------------------------===//

namespace {

struct OrderedPredicate {
  OrderedPredicate(const std::pair<Position *, Qualifier *> &ip)
      : position(ip.first), question(ip.second) {}
  OrderedPredicate(const PositionalPredicate &ip)
      : position(ip.position), question(ip.question) {}

  Position *position;
  Qualifier *question;

  unsigned primary = 0;
  unsigned secondary = 0;
  unsigned id = 0;

  DenseMap<ConstrPatternInfo *, Qualifier *> patternToAnswer;

  bool operator<(const OrderedPredicate &rhs) const {
    auto *rhsPos = rhs.position;
    return std::make_tuple(primary, secondary, rhsPos->getOperationDepth(),
                           rhsPos->getKind(), rhs.question->getKind(), rhs.id) >
           std::make_tuple(rhs.primary, rhs.secondary,
                           position->getOperationDepth(), position->getKind(),
                           question->getKind(), id);
  }
};

struct OrderedPredicateDenseInfo {
  using Base = DenseMapInfo<std::pair<Position *, Qualifier *>>;

  static OrderedPredicate getEmptyKey() { return Base::getEmptyKey(); }
  static OrderedPredicate getTombstoneKey() { return Base::getTombstoneKey(); }
  static bool isEqual(const OrderedPredicate &lhs,
                      const OrderedPredicate &rhs) {
    return lhs.position == rhs.position && lhs.question == rhs.question;
  }
  static unsigned getHashValue(const OrderedPredicate &p) {
    return llvm::hash_combine(p.position, p.question);
  }
};

struct OrderedPredicateList {
  OrderedPredicateList(ConstrPatternInfo *info) : info(info) {}
  ConstrPatternInfo *info;
  DenseSet<OrderedPredicate *> predicates;
};

} // namespace

static bool isSamePredicate(MatcherNode *node, OrderedPredicate *predicate) {
  return node->getPosition() == predicate->position &&
         node->getQuestion() == predicate->question;
}

static std::unique_ptr<MatcherNode> &
getOrCreateChild(SwitchNode *node, OrderedPredicate *predicate,
                 ConstrPatternInfo *info) {
  auto it = predicate->patternToAnswer.find(info);
  assert(it != predicate->patternToAnswer.end() &&
         "expected pattern to exist in predicate");
  return node->getChildren()[it->second];
}

static void propagatePattern(std::unique_ptr<MatcherNode> &node,
                             OrderedPredicateList &list,
                             std::vector<OrderedPredicate *>::iterator current,
                             std::vector<OrderedPredicate *>::iterator end) {
  if (current == end) {
    node = std::make_unique<ConstrSuccessNode>(list.info, std::move(node));
  } else if (!list.predicates.contains(*current)) {
    propagatePattern(node, list, std::next(current), end);
  } else if (!node) {
    node = std::make_unique<SwitchNode>((*current)->position,
                                        (*current)->question);
    propagatePattern(
        getOrCreateChild(cast<SwitchNode>(&*node), *current, list.info), list,
        std::next(current), end);
  } else if (isSamePredicate(node.get(), *current)) {
    propagatePattern(
        getOrCreateChild(cast<SwitchNode>(&*node), *current, list.info), list,
        std::next(current), end);
  } else {
    propagatePattern(node->getFailureNode(), list, current, end);
  }
}

static void foldSwitchToBool(std::unique_ptr<MatcherNode> &node) {
  if (!node)
    return;
  if (auto *switchNode = dyn_cast<SwitchNode>(&*node)) {
    auto &children = switchNode->getChildren();
    for (auto &it : children)
      foldSwitchToBool(it.second);
    if (children.size() == 1) {
      auto *childIt = children.begin();
      node = std::make_unique<BoolNode>(
          node->getPosition(), node->getQuestion(), childIt->first,
          std::move(childIt->second), std::move(node->getFailureNode()));
    }
  } else if (auto *boolNode = dyn_cast<BoolNode>(&*node)) {
    foldSwitchToBool(boolNode->getSuccessNode());
  }
  foldSwitchToBool(node->getFailureNode());
}

static void insertExitNode(std::unique_ptr<MatcherNode> *root) {
  while (*root)
    root = &(*root)->getFailureNode();
  *root = std::make_unique<ExitNode>();
}

template <typename Iterator, typename Compare>
static void stableTopologicalSort(Iterator begin, Iterator end, Compare cmp) {
  while (begin != end) {
    llvm::SmallPtrSet<typename Iterator::value_type, 16> sortBeforeOthers;
    for (auto i = begin; i != end; ++i) {
      if (std::none_of(begin, end, [&](auto const &b) { return cmp(b, *i); }))
        sortBeforeOthers.insert(*i);
    }
    auto const next = std::stable_partition(begin, end, [&](auto const &a) {
      return sortBeforeOthers.contains(a);
    });
    assert(next != begin && "not a partial ordering");
    begin = next;
  }
}

static bool dependsOn(OrderedPredicate *a, OrderedPredicate *b) {
  auto *cqa = dyn_cast<ConstraintQuestion>(a->question);
  if (!cqa)
    return false;
  auto positionDependsOnA = [&](Position *p) {
    auto *cp = dyn_cast<ConstraintPosition>(p);
    return cp && cp->getQuestion() == cqa;
  };
  if (auto *cqb = dyn_cast<ConstraintQuestion>(b->question))
    return llvm::any_of(cqb->getArgs(), positionDependsOnA);
  if (auto *equalTo = dyn_cast<EqualToQuestion>(b->question))
    return positionDependsOnA(b->position) ||
           positionDependsOnA(equalTo->getValue());
  return positionDependsOnA(b->position);
}

//===----------------------------------------------------------------------===//
// Combine + matcher generation
//===----------------------------------------------------------------------===//

namespace {

class PDLConstrPatternLowering {
public:
  PDLConstrPatternLowering(pdl_interp::FuncOp matcherFunc,
                           ModuleOp rewriterModule);

  void lower(ModuleOp module);

private:
  using ValueMap = llvm::ScopedHashTable<Position *, Value>;
  using ValueMapScope = llvm::ScopedHashTableScope<Position *, Value>;

  /// Build the merged matcher tree across all pdl_constr.pattern ops.
  std::unique_ptr<MatcherNode> buildMatcherTree(ModuleOp module,
                                                PredicateBuilder &builder);

  Block *generateMatcher(MatcherNode &node, Region &region,
                         Block *block = nullptr);
  Value getValueAt(Block *&currentBlock, Position *pos);
  void generate(BoolNode *boolNode, Block *&currentBlock, Value val);
  void generate(SwitchNode *switchNode, Block *currentBlock, Value val);
  void generate(ConstrSuccessNode *successNode, Block *&currentBlock);

  /// Collect the names of operations created by the rewriter referenced from
  /// the given symbol.
  ArrayAttr collectGeneratedOps(SymbolRefAttr rewriterRef);

  OpBuilder builder;
  pdl_interp::FuncOp matcherFunc;
  ModuleOp rewriterModule;
  SymbolTable rewriterSymbolTable;
  ValueMap values;
  SmallVector<Block *, 8> failureBlockStack;
  SetVector<Value> locOps;
  DenseMap<ConstraintQuestion *, pdl_interp::ApplyConstraintOp> constraintOpMap;

  /// Owned storage for the per-pattern info structures. Held as a vector of
  /// unique_ptrs so that we can take stable pointers into the elements.
  std::vector<std::unique_ptr<ConstrPatternInfo>> patternInfos;
};

} // namespace

PDLConstrPatternLowering::PDLConstrPatternLowering(
    pdl_interp::FuncOp matcherFunc, ModuleOp rewriterModule)
    : builder(matcherFunc.getContext()), matcherFunc(matcherFunc),
      rewriterModule(rewriterModule), rewriterSymbolTable(rewriterModule) {}

void PDLConstrPatternLowering::lower(ModuleOp module) {
  PredicateUniquer predicateUniquer;
  PredicateBuilder predicateBuilder(predicateUniquer, module.getContext());

  ValueMapScope topLevelValueScope(values);

  Block *matcherEntryBlock = &matcherFunc.front();
  values.insert(predicateBuilder.getRoot(), matcherEntryBlock->getArgument(0));

  std::unique_ptr<MatcherNode> root = buildMatcherTree(module, predicateBuilder);
  Block *firstMatcherBlock = generateMatcher(*root, matcherFunc.getBody());
  assert(failureBlockStack.empty() && "failed to empty the stack");

  matcherEntryBlock->getOperations().splice(matcherEntryBlock->end(),
                                            firstMatcherBlock->getOperations());
  firstMatcherBlock->erase();
}

std::unique_ptr<MatcherNode>
PDLConstrPatternLowering::buildMatcherTree(ModuleOp module,
                                           PredicateBuilder &builder) {
  DenseMap<Value, Position *> valueToPosition;

  // Step 1: extract per-pattern positional predicates.
  for (pdl_constr::PatternOp pattern : module.getOps<pdl_constr::PatternOp>()) {
    auto info = std::make_unique<ConstrPatternInfo>();
    info->pattern = pattern;
    PredicateExtractor extractor(builder, pattern, *info, valueToPosition);
    if (failed(extractor.extract()))
      continue;
    patternInfos.push_back(std::move(info));
  }

  // Step 2: unique predicates across all patterns.
  DenseSet<OrderedPredicate, OrderedPredicateDenseInfo> uniqued;
  for (auto &info : patternInfos) {
    for (auto &predicate : info->predicates) {
      auto it = uniqued.insert(predicate);
      it.first->patternToAnswer.try_emplace(info.get(), predicate.answer);
      if (it.second)
        it.first->id = uniqued.size() - 1;
    }
  }

  // Step 3: build per-pattern ordered predicate sets and fill in the cost
  // model's primary / secondary sums.
  std::vector<OrderedPredicateList> lists;
  lists.reserve(patternInfos.size());
  for (auto &info : patternInfos) {
    OrderedPredicateList list(info.get());
    for (auto &predicate : info->predicates) {
      OrderedPredicate *op = &*uniqued.find(predicate);
      list.predicates.insert(op);
      ++op->primary;
    }
    lists.push_back(std::move(list));
  }
  for (auto &list : lists) {
    unsigned total = 0;
    for (auto *p : list.predicates)
      total += p->primary * p->primary;
    for (auto *p : list.predicates)
      p->secondary += total;
  }

  // Step 4: sort and propagate.
  std::vector<OrderedPredicate *> ordered;
  ordered.reserve(uniqued.size());
  for (auto &ip : uniqued)
    ordered.push_back(&ip);
  llvm::sort(ordered, [](OrderedPredicate *lhs, OrderedPredicate *rhs) {
    return *lhs < *rhs;
  });
  stableTopologicalSort(ordered.begin(), ordered.end(), dependsOn);

  std::unique_ptr<MatcherNode> root;
  for (OrderedPredicateList &list : lists)
    propagatePattern(root, list, ordered.begin(), ordered.end());

  foldSwitchToBool(root);
  insertExitNode(&root);
  return root;
}

//===----------------------------------------------------------------------===//
// Matcher code generation (mirrors PDLToPDLInterp.cpp::PatternLowering)
//===----------------------------------------------------------------------===//

Block *PDLConstrPatternLowering::generateMatcher(MatcherNode &node,
                                                 Region &region, Block *block) {
  if (!block)
    block = &region.emplaceBlock();
  ValueMapScope scope(values);

  if (isa<ExitNode>(node)) {
    builder.setInsertionPointToEnd(block);
    pdl_interp::FinalizeOp::create(builder, matcherFunc.getLoc());
    return block;
  }

  std::unique_ptr<MatcherNode> &failureNode = node.getFailureNode();
  Block *failureBlock;
  if (failureNode) {
    failureBlock = generateMatcher(*failureNode, region);
    failureBlockStack.push_back(failureBlock);
  } else {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    failureBlock = failureBlockStack.back();
  }

  Block *currentBlock = block;
  Position *position = node.getPosition();
  Value val = position ? getValueAt(currentBlock, position) : Value();

  bool isOperationValue = val && isa<pdl::OperationType>(val.getType());
  if (isOperationValue)
    locOps.insert(val);

  TypeSwitch<MatcherNode *>(&node)
      .Case<BoolNode, SwitchNode>([&](auto *derivedNode) {
        this->generate(derivedNode, currentBlock, val);
      })
      .Case([&](ConstrSuccessNode *successNode) {
        generate(successNode, currentBlock);
      });

  while (failureBlockStack.back() != failureBlock) {
    failureBlockStack.pop_back();
    assert(!failureBlockStack.empty() && "unable to locate failure block");
  }
  if (failureNode)
    failureBlockStack.pop_back();
  if (isOperationValue)
    locOps.remove(val);

  return block;
}

Value PDLConstrPatternLowering::getValueAt(Block *&currentBlock, Position *pos) {
  if (Value val = values.lookup(pos))
    return val;

  Value parentVal;
  if (Position *parent = pos->getParent())
    parentVal = getValueAt(currentBlock, parent);

  Location loc = parentVal ? parentVal.getLoc() : builder.getUnknownLoc();
  builder.setInsertionPointToEnd(currentBlock);
  Value value;
  switch (pos->getKind()) {
  case Predicates::OperationPos: {
    auto *operationPos = cast<OperationPosition>(pos);
    if (operationPos->isOperandDefiningOp())
      value = pdl_interp::GetDefiningOpOp::create(
          builder, loc, builder.getType<pdl::OperationType>(), parentVal);
    else
      value = parentVal;
    break;
  }
  case Predicates::UsersPos: {
    auto *usersPos = cast<UsersPosition>(pos);
    if (isa<pdl::RangeType>(parentVal.getType()) &&
        usersPos->useRepresentative())
      value = pdl_interp::ExtractOp::create(builder, loc, parentVal, 0);
    else
      value = parentVal;
    value = pdl_interp::GetUsersOp::create(builder, loc, value);
    break;
  }
  case Predicates::ForEachPos: {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    auto foreach = pdl_interp::ForEachOp::create(
        builder, loc, parentVal, failureBlockStack.back(), /*initLoop=*/true);
    value = foreach.getLoopVariable();
    Block *continueBlock = builder.createBlock(&foreach.getRegion());
    pdl_interp::ContinueOp::create(builder, loc);
    failureBlockStack.push_back(continueBlock);
    currentBlock = &foreach.getRegion().front();
    break;
  }
  case Predicates::OperandPos: {
    auto *operandPos = cast<OperandPosition>(pos);
    value = pdl_interp::GetOperandOp::create(
        builder, loc, builder.getType<pdl::ValueType>(), parentVal,
        operandPos->getOperandNumber());
    break;
  }
  case Predicates::OperandGroupPos: {
    auto *operandPos = cast<OperandGroupPosition>(pos);
    Type valueTy = builder.getType<pdl::ValueType>();
    value = pdl_interp::GetOperandsOp::create(
        builder, loc,
        operandPos->isVariadic() ? pdl::RangeType::get(valueTy) : valueTy,
        parentVal, operandPos->getOperandGroupNumber());
    break;
  }
  case Predicates::AttributePos: {
    auto *attrPos = cast<AttributePosition>(pos);
    value = pdl_interp::GetAttributeOp::create(
        builder, loc, builder.getType<pdl::AttributeType>(), parentVal,
        attrPos->getName().strref());
    break;
  }
  case Predicates::TypePos: {
    if (isa<pdl::AttributeType>(parentVal.getType()))
      value = pdl_interp::GetAttributeTypeOp::create(builder, loc, parentVal);
    else
      value = pdl_interp::GetValueTypeOp::create(builder, loc, parentVal);
    break;
  }
  case Predicates::ResultPos: {
    auto *resPos = cast<ResultPosition>(pos);
    value = pdl_interp::GetResultOp::create(
        builder, loc, builder.getType<pdl::ValueType>(), parentVal,
        resPos->getResultNumber());
    break;
  }
  case Predicates::ResultGroupPos: {
    auto *resPos = cast<ResultGroupPosition>(pos);
    Type valueTy = builder.getType<pdl::ValueType>();
    value = pdl_interp::GetResultsOp::create(
        builder, loc,
        resPos->isVariadic() ? pdl::RangeType::get(valueTy) : valueTy,
        parentVal, resPos->getResultGroupNumber());
    break;
  }
  case Predicates::AttributeLiteralPos: {
    auto *attrPos = cast<AttributeLiteralPosition>(pos);
    value = pdl_interp::CreateAttributeOp::create(builder, loc,
                                                  attrPos->getValue());
    break;
  }
  case Predicates::TypeLiteralPos: {
    auto *typePos = cast<TypeLiteralPosition>(pos);
    Attribute rawTypeAttr = typePos->getValue();
    if (TypeAttr typeAttr = dyn_cast<TypeAttr>(rawTypeAttr))
      value = pdl_interp::CreateTypeOp::create(builder, loc, typeAttr);
    else
      value = pdl_interp::CreateTypesOp::create(builder, loc,
                                                cast<ArrayAttr>(rawTypeAttr));
    break;
  }
  case Predicates::ConstraintResultPos: {
    auto *constrResPos = cast<ConstraintPosition>(pos);
    auto i = constraintOpMap.find(constrResPos->getQuestion());
    assert(i != constraintOpMap.end());
    value = i->second->getResult(constrResPos->getIndex());
    break;
  }
  default:
    llvm_unreachable("Generating unknown Position getter");
    break;
  }
  values.insert(pos, value);
  return value;
}

void PDLConstrPatternLowering::generate(BoolNode *boolNode, Block *&currentBlock,
                                        Value val) {
  Location loc = val.getLoc();
  Qualifier *question = boolNode->getQuestion();
  Qualifier *answer = boolNode->getAnswer();
  Region *region = currentBlock->getParent();

  SmallVector<Value> args;
  if (auto *equalToQuestion = dyn_cast<EqualToQuestion>(question)) {
    args = {getValueAt(currentBlock, equalToQuestion->getValue())};
  } else if (auto *cstQuestion = dyn_cast<ConstraintQuestion>(question)) {
    for (Position *position : cstQuestion->getArgs())
      args.push_back(getValueAt(currentBlock, position));
  }

  Block *success = &region->emplaceBlock();
  Block *failure = failureBlockStack.back();

  builder.setInsertionPointToEnd(currentBlock);
  Predicates::Kind kind = question->getKind();
  switch (kind) {
  case Predicates::IsNotNullQuestion:
    pdl_interp::IsNotNullOp::create(builder, loc, val, success, failure);
    break;
  case Predicates::OperationNameQuestion: {
    auto *opNameAnswer = cast<OperationNameAnswer>(answer);
    pdl_interp::CheckOperationNameOp::create(
        builder, loc, val, opNameAnswer->getValue().getStringRef(), success,
        failure);
    break;
  }
  case Predicates::TypeQuestion: {
    auto *ans = cast<TypeAnswer>(answer);
    if (isa<pdl::RangeType>(val.getType()))
      pdl_interp::CheckTypesOp::create(builder, loc, val,
                                       llvm::cast<ArrayAttr>(ans->getValue()),
                                       success, failure);
    else
      pdl_interp::CheckTypeOp::create(builder, loc, val,
                                      llvm::cast<TypeAttr>(ans->getValue()),
                                      success, failure);
    break;
  }
  case Predicates::AttributeQuestion: {
    auto *ans = cast<AttributeAnswer>(answer);
    pdl_interp::CheckAttributeOp::create(builder, loc, val, ans->getValue(),
                                         success, failure);
    break;
  }
  case Predicates::OperandCountAtLeastQuestion:
  case Predicates::OperandCountQuestion:
    pdl_interp::CheckOperandCountOp::create(
        builder, loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::OperandCountAtLeastQuestion,
        success, failure);
    break;
  case Predicates::ResultCountAtLeastQuestion:
  case Predicates::ResultCountQuestion:
    pdl_interp::CheckResultCountOp::create(
        builder, loc, val, cast<UnsignedAnswer>(answer)->getValue(),
        /*compareAtLeast=*/kind == Predicates::ResultCountAtLeastQuestion,
        success, failure);
    break;
  case Predicates::EqualToQuestion: {
    bool trueAnswer = isa<TrueAnswer>(answer);
    pdl_interp::AreEqualOp::create(builder, loc, val, args.front(),
                                   trueAnswer ? success : failure,
                                   trueAnswer ? failure : success);
    break;
  }
  case Predicates::ConstraintQuestion: {
    auto *cstQuestion = cast<ConstraintQuestion>(question);
    auto applyConstraintOp = pdl_interp::ApplyConstraintOp::create(
        builder, loc, cstQuestion->getResultTypes(), cstQuestion->getName(),
        args, cstQuestion->getIsNegated(), success, failure);
    constraintOpMap.insert({cstQuestion, applyConstraintOp});
    break;
  }
  default:
    llvm_unreachable("Generating unknown Predicate operation");
  }

  generateMatcher(*boolNode->getSuccessNode(), *region, success);
}

template <typename OpT, typename PredT, typename ValT = typename PredT::KeyTy>
static void createSwitchOp(Value val, Block *defaultDest, OpBuilder &builder,
                           llvm::MapVector<Qualifier *, Block *> &dests) {
  std::vector<ValT> values;
  std::vector<Block *> blocks;
  values.reserve(dests.size());
  blocks.reserve(dests.size());
  for (const auto &it : dests) {
    blocks.push_back(it.second);
    values.push_back(cast<PredT>(it.first)->getValue());
  }
  OpT::create(builder, val.getLoc(), val, values, defaultDest, blocks);
}

void PDLConstrPatternLowering::generate(SwitchNode *switchNode,
                                        Block *currentBlock, Value val) {
  Qualifier *question = switchNode->getQuestion();
  Region *region = currentBlock->getParent();
  Block *defaultDest = failureBlockStack.back();

  Predicates::Kind kind = question->getKind();
  if (kind == Predicates::OperandCountAtLeastQuestion ||
      kind == Predicates::ResultCountAtLeastQuestion) {
    SmallVector<unsigned> sortedChildren = llvm::to_vector<16>(
        llvm::seq<unsigned>(0, switchNode->getChildren().size()));
    llvm::sort(sortedChildren, [&](unsigned lhs, unsigned rhs) {
      return cast<UnsignedAnswer>(switchNode->getChild(lhs).first)->getValue() >
             cast<UnsignedAnswer>(switchNode->getChild(rhs).first)->getValue();
    });

    failureBlockStack.push_back(defaultDest);
    Location loc = val.getLoc();
    for (unsigned idx : sortedChildren) {
      auto &child = switchNode->getChild(idx);
      Block *childBlock = generateMatcher(*child.second, *region);
      Block *predicateBlock = builder.createBlock(childBlock);
      builder.setInsertionPointToEnd(predicateBlock);
      unsigned ans = cast<UnsignedAnswer>(child.first)->getValue();
      switch (kind) {
      case Predicates::OperandCountAtLeastQuestion:
        pdl_interp::CheckOperandCountOp::create(builder, loc, val, ans,
                                                /*compareAtLeast=*/true,
                                                childBlock, defaultDest);
        break;
      case Predicates::ResultCountAtLeastQuestion:
        pdl_interp::CheckResultCountOp::create(builder, loc, val, ans,
                                               /*compareAtLeast=*/true,
                                               childBlock, defaultDest);
        break;
      default:
        llvm_unreachable("Generating invalid AtLeast operation");
      }
      failureBlockStack.back() = predicateBlock;
    }
    Block *firstPredicateBlock = failureBlockStack.pop_back_val();
    currentBlock->getOperations().splice(currentBlock->end(),
                                         firstPredicateBlock->getOperations());
    firstPredicateBlock->erase();
    return;
  }

  llvm::MapVector<Qualifier *, Block *> children;
  for (auto &it : switchNode->getChildren())
    children.insert({it.first, generateMatcher(*it.second, *region)});
  builder.setInsertionPointToEnd(currentBlock);

  switch (question->getKind()) {
  case Predicates::OperandCountQuestion:
    return createSwitchOp<pdl_interp::SwitchOperandCountOp, UnsignedAnswer,
                          int32_t>(val, defaultDest, builder, children);
  case Predicates::ResultCountQuestion:
    return createSwitchOp<pdl_interp::SwitchResultCountOp, UnsignedAnswer,
                          int32_t>(val, defaultDest, builder, children);
  case Predicates::OperationNameQuestion:
    return createSwitchOp<pdl_interp::SwitchOperationNameOp,
                          OperationNameAnswer>(val, defaultDest, builder,
                                               children);
  case Predicates::TypeQuestion:
    if (isa<pdl::RangeType>(val.getType()))
      return createSwitchOp<pdl_interp::SwitchTypesOp, TypeAnswer>(
          val, defaultDest, builder, children);
    return createSwitchOp<pdl_interp::SwitchTypeOp, TypeAnswer>(
        val, defaultDest, builder, children);
  case Predicates::AttributeQuestion:
    return createSwitchOp<pdl_interp::SwitchAttributeOp, AttributeAnswer>(
        val, defaultDest, builder, children);
  default:
    llvm_unreachable("Generating unknown switch predicate.");
  }
}

ArrayAttr PDLConstrPatternLowering::collectGeneratedOps(
    SymbolRefAttr rewriterRef) {
  // Resolve the rewriter function inside the rewriter module.
  Operation *symbol = SymbolTable::lookupSymbolIn(rewriterModule, rewriterRef);
  if (!symbol)
    return {};

  SmallVector<StringRef, 4> generatedOps;
  symbol->walk([&](pdl_interp::CreateOperationOp createOp) {
    generatedOps.push_back(createOp.getName());
  });
  if (generatedOps.empty())
    return {};
  return builder.getStrArrayAttr(generatedOps);
}

void PDLConstrPatternLowering::generate(ConstrSuccessNode *successNode,
                                        Block *&currentBlock) {
  ConstrPatternInfo *info = successNode->info;

  // Resolve each input position to its corresponding pdl_interp value.
  std::vector<Value> mappedMatchValues;
  mappedMatchValues.reserve(info->inputPositions.size());
  for (Position *pos : info->inputPositions)
    mappedMatchValues.push_back(getValueAt(currentBlock, pos));

  ArrayAttr generatedOpsAttr = collectGeneratedOps(info->rewriterRef);

  builder.setInsertionPointToEnd(currentBlock);
  pdl_interp::RecordMatchOp::create(
      builder, info->pattern.getLoc(), mappedMatchValues, locOps.getArrayRef(),
      info->rewriterRef, info->rootKindAttr, generatedOpsAttr,
      info->pattern.getBenefitAttr(), failureBlockStack.back());
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {

struct PDLConstrToPDLInterpPass
    : public impl::ConvertPDLConstrToPDLInterpPassBase<
          PDLConstrToPDLInterpPass> {
  void runOnOperation() final;
};

} // namespace

void PDLConstrToPDLInterpPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Locate the rewriter module produced by the pdl -> pdl_constr pass. It is
  // a nested module named after `getRewriterModuleName()` and holds the
  // `pdl_interp.func` rewriter functions referenced from `pdl_constr.success`.
  ModuleOp rewriterModule;
  StringRef rewriterModuleName =
      pdl_interp::PDLInterpDialect::getRewriterModuleName();
  for (ModuleOp nested : module.getOps<ModuleOp>()) {
    if (nested.getSymName() == rewriterModuleName) {
      rewriterModule = nested;
      break;
    }
  }
  if (!rewriterModule) {
    // No rewriter module — synthesize an empty one so that we can still
    // produce a well-formed matcher (with no successful matches).
    OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
    rewriterModule = ModuleOp::create(b, module.getLoc(), rewriterModuleName);
  }

  // Create the matcher function alongside the rewriter module.
  OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
  auto matcherFunc = pdl_interp::FuncOp::create(
      b, module.getLoc(),
      pdl_interp::PDLInterpDialect::getMatcherFunctionName(),
      b.getFunctionType(b.getType<pdl::OperationType>(), /*results=*/{}),
      /*attrs=*/ArrayRef<NamedAttribute>());

  PDLConstrPatternLowering generator(matcherFunc, rewriterModule);
  generator.lower(module);

  // Erase the now-lowered pdl_constr patterns.
  for (pdl_constr::PatternOp pattern :
       llvm::make_early_inc_range(module.getOps<pdl_constr::PatternOp>()))
    pattern.erase();
}
