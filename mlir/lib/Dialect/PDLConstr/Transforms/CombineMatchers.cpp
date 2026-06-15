//===- CombineMatchers.cpp - Combine pdl_constr.matcher ops ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Combines every `pdl_constr.matcher` in a module into one combined matcher
// that materializes the matcher tree directly in IR (`pdl_constr.try`,
// `pdl_constr.switch_op_name`, `pdl_constr.switch_type`). The combining
// algorithm mirrors `MatcherNode::generateMatcherTree` in
// `PDLToPDLInterp/PredicateTree.cpp`.
//
// Canonical predicates are represented as actual `pdl_constr` ops in a
// standalone "pool" block: exactly one pool op per equivalence class of
// `(op name, attribute dictionary, canonical-operand-Values)`. The pool's
// block argument is the canonical root. The whole point of `pdl_constr` is
// that predicates and their dependency edges live in SSA; this pass leans on
// that directly:
//
//  * Predicate identity     := `Operation *` into the pool.
//  * Canonical value ID     := `Value` produced by a pool op (or `poolRoot`).
//  * Dependency edges       := SSA def-use in the pool.
//  * Per-matcher value map  := `IRMapping` (input matcher Value -> pool Value).
//  * Predicate ordering     := block order of pool ops.
//
// There is no parallel `OrderedPredicate` / `OpKey` / integer-ID datastructure
// — those would just be a shadow of what the IR already encodes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDLConstr/Transforms/Passes.h"

#include "mlir/Dialect/PDLConstr/IR/PDLConstr.h"
#include "mlir/Dialect/PDLConstr/IR/PDLConstrOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "pdl-constr-combine-matchers"

namespace mlir {
namespace pdl_constr {
#define GEN_PASS_DEF_PDLCONSTRCOMBINEMATCHERSPASS
#include "mlir/Dialect/PDLConstr/Transforms/Passes.h.inc"
} // namespace pdl_constr
} // namespace mlir

using namespace mlir;
using namespace mlir::pdl_constr;

namespace {

//===----------------------------------------------------------------------===//
// Pool key
//
// Hash key used to dedup canonical pool ops. Two input-matcher ops collapse
// to the same pool op iff they have the same op name, same raw attribute
// dictionary, and their operands map to the same pool SSA values. Identity
// of the operand `Value`s does all the work that the old integer-ID scheme
// did manually.
//===----------------------------------------------------------------------===//

struct PoolKey {
  OperationName name;
  DictionaryAttr attrs;
  SmallVector<Value, 4> operands;

  bool operator==(const PoolKey &o) const {
    return name == o.name && attrs == o.attrs && operands == o.operands;
  }
};

} // namespace

namespace llvm {
template <>
struct DenseMapInfo<PoolKey> {
  static PoolKey getEmptyKey() {
    return {DenseMapInfo<OperationName>::getEmptyKey(), nullptr, {}};
  }
  static PoolKey getTombstoneKey() {
    return {DenseMapInfo<OperationName>::getTombstoneKey(), nullptr, {}};
  }
  static unsigned getHashValue(const PoolKey &k) {
    return llvm::hash_combine(
        k.name, k.attrs,
        llvm::hash_combine_range(k.operands.begin(), k.operands.end()));
  }
  static bool isEqual(const PoolKey &a, const PoolKey &b) { return a == b; }
};
} // namespace llvm

namespace {

//===----------------------------------------------------------------------===//
// MatcherInfo: per-input-matcher state.
//===----------------------------------------------------------------------===//

struct MatcherInfo {
  MatcherOp matcher;

  /// The (unique) `pdl_constr.success` op in this matcher.
  SuccessOp success;

  /// Set of canonical pool ops this matcher exercises.
  DenseSet<Operation *> preds;

  /// Map from this matcher's SSA values to their canonical pool values.
  IRMapping toCanonical;
};

//===----------------------------------------------------------------------===//
// TreeNode: in-memory failure spine, materialized into IR at the end.
//
// The failure spine is genuine transient algorithm state (it represents the
// shape of `try` alternatives we are *going to* emit), not a shadow of
// existing IR, so it stays in memory. Its contents are SSA-native: each
// node references the canonical pool op / pool Values directly.
//===----------------------------------------------------------------------===//

struct TreeNode {
  enum class Kind { Test, Success };
  Kind kind;

  // ---- Test node ----
  Operation *pred = nullptr; ///< canonical pool op
  std::unique_ptr<TreeNode> successPath;
  std::unique_ptr<TreeNode> failurePath;

  // ---- Success node ----
  SuccessOp originalSuccess;
  SmallVector<Value, 4> successInputs; ///< canonical pool values
};

//===----------------------------------------------------------------------===//
// Combiner
//===----------------------------------------------------------------------===//

class Combiner {
public:
  Combiner(ModuleOp module) : module(module) {}
  LogicalResult run();

private:
  LogicalResult buildCanonicalPool();
  void sortCanonicalPool();
  void buildTree();
  void emitCombinedMatcher();

  Operation *getOrCreatePoolOp(Operation *modelOp,
                               ArrayRef<Value> canonicalOperands);

  void propagate(std::unique_ptr<TreeNode> &node, MatcherInfo &info,
                 unsigned predIdx);

  void emitNode(OpBuilder &builder, Location loc, TreeNode *node,
                IRMapping &mapping);

  void foldSwitches(Region &region);

  ModuleOp module;

  /// Canonical pool: a standalone block whose ops are the unique predicates.
  /// The block argument is the canonical root operation value. Pool ops'
  /// SSA results are the canonical "IDs" for every constraint-produced value.
  /// The block has no parent region — we never need to traverse upward from
  /// it, and side-stepping the region keeps `Region::getContext()` (which
  /// asserts on detached regions) out of the picture.
  std::unique_ptr<Block> poolBody;
  Value poolRoot;

  /// Hash-based dedup for canonical pool ops.
  DenseMap<PoolKey, Operation *> poolDedup;

  /// Cost-based ordering side tables, keyed by pool op (mirrors
  /// OrderedPredicate::primary / secondary / id).
  DenseMap<Operation *, unsigned> primary;
  DenseMap<Operation *, unsigned> secondary;
  DenseMap<Operation *, unsigned> insertionIndex;

  /// Pool ops in cost-sorted order. Their order in the pool block is kept
  /// in sync (so SSA dominance matches the predicate ordering).
  std::vector<Operation *> sortedPoolOps;

  std::vector<MatcherInfo> matcherInfos;

  std::unique_ptr<TreeNode> treeRoot;
};

} // namespace

//===----------------------------------------------------------------------===//
// Phase 1: build the canonical pool by walking input matchers.
//===----------------------------------------------------------------------===//

Operation *Combiner::getOrCreatePoolOp(Operation *modelOp,
                                      ArrayRef<Value> canonicalOperands) {
  PoolKey key{modelOp->getName(), modelOp->getAttrDictionary(),
              SmallVector<Value, 4>(canonicalOperands)};
  if (auto it = poolDedup.find(key); it != poolDedup.end())
    return it->second;

  // Clone modelOp into the pool with operands rewired to canonical values.
  // `Operation::clone` doesn't need a builder, which is the point: the pool
  // block is standalone (no parent region), so `OpBuilder(poolBody, ...)`
  // would assert when fetching the context via the (nonexistent) region.
  IRMapping mapping;
  for (auto [orig, canonical] :
       llvm::zip(modelOp->getOperands(), canonicalOperands))
    mapping.map(orig, canonical);
  Operation *poolOp = modelOp->cloneWithoutRegions(mapping);
  poolBody->push_back(poolOp);

  insertionIndex[poolOp] = poolDedup.size();
  poolDedup[key] = poolOp;
  return poolOp;
}

LogicalResult Combiner::buildCanonicalPool() {
  // Snapshot input matchers before we touch the module.
  SmallVector<MatcherOp> inputMatchers(module.getOps<MatcherOp>());

  poolBody = std::make_unique<Block>();
  poolRoot = poolBody->addArgument(
      pdl::OperationType::get(module.getContext()), module.getLoc());

  for (MatcherOp matcher : inputMatchers) {
    MatcherInfo info;
    info.matcher = matcher;

    Block &body = matcher.getBodyRegion().front();
    info.toCanonical.map(body.getArgument(0), poolRoot);

    for (Operation &op : body) {
      if (auto succ = dyn_cast<SuccessOp>(&op)) {
        if (info.success)
          return op.emitOpError("combine-matchers expects exactly one "
                                "`pdl_constr.success` per input matcher");
        info.success = succ;
        continue;
      }

      if (isa<TryOp, SwitchOpNameOp, SwitchTypeOp>(&op))
        return op.emitOpError(
            "combine-matchers does not support nested `try`/`switch_*` in "
            "input matchers");

      // Operand canonicalization is just SSA value lookup: each input-matcher
      // operand is either the matcher root or the result of a predicate we
      // have already mapped, so its canonical counterpart is in `toCanonical`.
      SmallVector<Value, 4> canonicalOperands;
      canonicalOperands.reserve(op.getNumOperands());
      for (Value operand : op.getOperands())
        canonicalOperands.push_back(info.toCanonical.lookup(operand));

      Operation *poolOp = getOrCreatePoolOp(&op, canonicalOperands);

      // Bind this op's results to the canonical pool results. From this point
      // on, downstream uses in this matcher resolve directly to pool values.
      for (auto [origRes, poolRes] :
           llvm::zip(op.getResults(), poolOp->getResults()))
        info.toCanonical.map(origRes, poolRes);

      if (info.preds.insert(poolOp).second)
        ++primary[poolOp];
    }

    if (!info.success)
      return matcher.emitOpError(
          "combine-matchers expects a `pdl_constr.success` in every input "
          "matcher");

    matcherInfos.push_back(std::move(info));
  }

  // Secondary sums: for each matcher, sum primary^2 over its predicates and
  // add that to each contained predicate's secondary (same formula as
  // OrderedPredicate).
  for (MatcherInfo &info : matcherInfos) {
    unsigned total = 0;
    for (Operation *p : info.preds)
      total += primary[p] * primary[p];
    for (Operation *p : info.preds)
      secondary[p] += total;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Phase 2: sort canonical pool ops by cost, then by topological order.
//===----------------------------------------------------------------------===//

namespace {

/// SSA-native dependency: `a` must precede `b` iff `b` consumes any result
/// of `a`. No separate operand-ID / result-ID bookkeeping needed.
static bool dependsOn(Operation *a, Operation *b) {
  for (Value operand : b->getOperands())
    if (operand.getDefiningOp() == a)
      return true;
  return false;
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

/// Infers the equivalent PDL Position::Kind integer of the given value.
/// Numeric values match `Predicates::Kind` in
/// mlir/lib/Conversion/PDLToPDLInterp/Predicate.h so that tie-breaking in the
/// cost comparator matches the original pass byte-for-byte.
static unsigned getPosKind(Value val) {
  // The matcher root block argument is the canonical root operation.
  if (isa<BlockArgument>(val))
    return 0u; // OperationPos

  Operation *def = val.getDefiningOp();
  if (!def)
    return 12u; // sentinel above all real kinds

  return TypeSwitch<Operation *, unsigned>(def)
      .Case<GetDefiningOpOp>([](auto) { return 0u; /* OperationPos */ })
      .Case<GetOperandOp>([](auto) { return 1u; /* OperandPos */ })
      .Case<GetOperandsOp>([](auto) { return 2u; /* OperandGroupPos */ })
      .Case<GetAttributeOp>([](auto) { return 3u; /* AttributePos */ })
      .Case<ApplyNativeConstraintOp>(
          [](auto) { return 4u; /* ConstraintResultPos */ })
      .Case<GetResultOp>([](auto) { return 5u; /* ResultPos */ })
      .Case<GetResultsOp>([](auto) { return 6u; /* ResultGroupPos */ })
      .Case<GetValueTypeOp, GetAttributeTypeOp>(
          [](auto) { return 7u; /* TypePos */ })
      // AttributeLiteralPos = 8 / TypeLiteralPos = 9 not represented in
      // pdl_constr yet — extend here if/when literal accessor ops are added.
      .Case<GetUsersOp>([](auto) { return 10u; /* UsersPos */ })
      .Case<GetEachOp>([](auto) { return 11u; /* ForEachPos */ })
      .Default([](Operation *) { return 12u; /* unknown */ });
}

/// Infers the equivalent PDL Qualifier::Kind integer of the given op.
/// Numeric values match the question half of `Predicates::Kind` in
/// mlir/lib/Conversion/PDLToPDLInterp/Predicate.h. Accessor ops return 0 so
/// they naturally sort before any real predicate (all of which are >= 12).
static unsigned getQuestKind(Operation *op) {
  return TypeSwitch<Operation *, unsigned>(op)
      .Case<IsNotNullOp>([](auto) { return 12u; /* IsNotNullQuestion */ })
      .Case<HasNameOp>([](auto) { return 13u; /* OperationNameQuestion */ })
      .Case<HasTypeOp, HasTypesOp>([](auto) { return 14u; /* TypeQuestion */ })
      .Case<HasAttrValueOp>([](auto) { return 15u; /* AttributeQuestion */ })
      .Case<CheckOperandCountOp>([](auto countOp) {
        // OperandCountAtLeastQuestion = 16, OperandCountQuestion = 17.
        return countOp.getAtLeast() ? 16u : 17u;
      })
      .Case<CheckResultCountOp>([](auto countOp) {
        // ResultCountAtLeastQuestion = 18, ResultCountQuestion = 19.
        return countOp.getAtLeast() ? 18u : 19u;
      })
      .Case<EqualOp>([](auto) { return 20u; /* EqualToQuestion */ })
      .Case<ApplyNativeConstraintOp>(
          [](auto) { return 21u; /* ConstraintQuestion */ })
      .Default([](Operation *) { return 0u; /* accessor — sorts first */ });
}

/// Recursively computes the OperationDepth by walking the SSA use-def chain.
static unsigned getOperationDepth(Value val, DenseMap<Value, unsigned> &cache) {
  if (auto it = cache.find(val); it != cache.end())
    return it->second;

  if (isa<BlockArgument>(val))
    return cache[val] = 0u;

  Operation *def = val.getDefiningOp();
  if (!def)
    return cache[val] = 0u;

  return TypeSwitch<Operation *, unsigned>(def)
      .Case<ApplyNativeConstraintOp>([&](auto constraintOp) {
        // Constraints anchor their depth to the maximum depth of their arguments.
        unsigned maxDepth = 0u;
        for (Value operand : constraintOp.getArgs())
          maxDepth = std::max(maxDepth, getOperationDepth(operand, cache));
        return cache[val] = maxDepth;
      })
      .Default([&](Operation *op) {
        if (op->getNumOperands() == 0)
          return cache[val] = 0u;

        unsigned parentDepth = getOperationDepth(op->getOperand(0), cache);

        // Reaching a new structural Operation boundary increments the depth.
        if (getPosKind(val) == 0 /* OperationPos */)
          return cache[val] = parentDepth + 1;

        return cache[val] = parentDepth;
      });
}

/// Gets the depth of the position this predicate acts upon.
static unsigned getPredicateDepth(Operation *op, DenseMap<Value, unsigned> &cache) {
  return TypeSwitch<Operation *, unsigned>(op)
      .Case<ApplyNativeConstraintOp>([&](auto constraintOp) {
        unsigned maxDepth = 0;
        for (Value operand : constraintOp.getArgs())
          maxDepth = std::max(maxDepth, getOperationDepth(operand, cache));
        return maxDepth;
      })
      .Default([&](Operation *defaultOp) {
        if (defaultOp->getNumOperands() > 0)
          return getOperationDepth(defaultOp->getOperand(0), cache);
        return 0u;
      });
}

/// Gets the equivalent Position::Kind this predicate acts upon.
static unsigned getPredicatePosKind(Operation *op, DenseMap<Value, unsigned> &cache) {
  return TypeSwitch<Operation *, unsigned>(op)
      .Case<ApplyNativeConstraintOp>([&](auto constraintOp) {
        Value maxVal = nullptr;
        unsigned maxD = 0;
        for (Value operand : constraintOp.getArgs()) {
          unsigned d = getOperationDepth(operand, cache);
          if (!maxVal || d > maxD) {
            maxVal = operand;
            maxD = d;
          }
        }
        if (maxVal) return getPosKind(maxVal);
        return 12u; // sentinel past all position kinds
      })
      .Default([&](Operation *defaultOp) {
        if (defaultOp->getNumOperands() > 0)
          return getPosKind(defaultOp->getOperand(0));
        return 12u; // sentinel past all position kinds
      });
}

void Combiner::sortCanonicalPool() {
  sortedPoolOps.reserve(poolDedup.size());
  for (Operation &op : *poolBody)
    sortedPoolOps.push_back(&op);

  DenseMap<Value, unsigned> depthsCache;

  // Cost-based sort matching PredicateTree's OrderedPredicate cost model exactly.
  // Higher frequency wins. When tied, smaller depth/position/question/ID wins.
  llvm::sort(sortedPoolOps, [&](Operation *a, Operation *b) {
    unsigned depthA = getPredicateDepth(a, depthsCache);
    unsigned depthB = getPredicateDepth(b, depthsCache);
    unsigned posKA = getPredicatePosKind(a, depthsCache);
    unsigned posKB = getPredicatePosKind(b, depthsCache);
    unsigned questKA = getQuestKind(a);
    unsigned questKB = getQuestKind(b);

    return std::make_tuple(primary[a], secondary[a],
                           depthB, posKB, questKB, insertionIndex[b]) >
           std::make_tuple(primary[b], secondary[b],
                           depthA, posKA, questKA, insertionIndex[a]);
  });

  LDBG() << "Sorted predicates (after cost sort):";
  for (Operation *op : sortedPoolOps) {
    LDBG() << "  * primary=" << primary[op] << " secondary=" << secondary[op]
           << " depth=" << getPredicateDepth(op, depthsCache)
           << " posKind=" << getPredicatePosKind(op, depthsCache)
           << " questKind=" << getQuestKind(op)
           << " id=" << insertionIndex[op] << " op=" << op->getName();
  }

  // Stabilize so operand-producers precede their users (preserves SSA
  // dominance once we reflect the order back into the pool block).
  stableTopologicalSort(sortedPoolOps.begin(), sortedPoolOps.end(), dependsOn);

  LDBG() << "Sorted predicates (after topological sort):";
  for (Operation *op : sortedPoolOps) {
    LDBG() << "  * primary=" << primary[op] << " secondary=" << secondary[op]
           << " depth=" << getPredicateDepth(op, depthsCache)
           << " posKind=" << getPredicatePosKind(op, depthsCache)
           << " questKind=" << getQuestKind(op)
           << " id=" << insertionIndex[op] << " op=" << op->getName();
  }

  for (Operation *op : sortedPoolOps)
    op->moveBefore(poolBody.get(), poolBody->end());
}
} // namespace

//===----------------------------------------------------------------------===//
// Phase 3: build the in-memory failure tree.
//===----------------------------------------------------------------------===//

void Combiner::propagate(std::unique_ptr<TreeNode> &node, MatcherInfo &info,
                         unsigned predIdx) {
  if (predIdx == sortedPoolOps.size()) {
    auto leaf = std::make_unique<TreeNode>();
    leaf->kind = TreeNode::Kind::Success;
    leaf->originalSuccess = info.success;
    leaf->successInputs.reserve(info.success.getInputs().size());
    for (Value input : info.success.getInputs())
      leaf->successInputs.push_back(info.toCanonical.lookup(input));
    leaf->failurePath = std::move(node);
    node = std::move(leaf);
    return;
  }

  Operation *current = sortedPoolOps[predIdx];
  if (!info.preds.contains(current)) {
    propagate(node, info, predIdx + 1);
    return;
  }

  if (!node) {
    auto fresh = std::make_unique<TreeNode>();
    fresh->kind = TreeNode::Kind::Test;
    fresh->pred = current;
    node = std::move(fresh);
    propagate(node->successPath, info, predIdx + 1);
    return;
  }

  if (node->kind == TreeNode::Kind::Test && node->pred == current) {
    propagate(node->successPath, info, predIdx + 1);
    return;
  }

  // Different predicate already at this position. Recurse into the failure
  // path; this builds a chain of `try` alternatives.
  propagate(node->failurePath, info, predIdx);
}

void Combiner::buildTree() {
  for (MatcherInfo &info : matcherInfos)
    propagate(treeRoot, info, /*predIdx=*/0);
}

//===----------------------------------------------------------------------===//
// Phase 4: emit IR by cloning canonical pool ops into the combined matcher.
//===----------------------------------------------------------------------===//

void Combiner::emitNode(OpBuilder &builder, Location loc, TreeNode *node,
                        IRMapping &mapping) {
  // Walk down the failure-spine chain, emitting siblings sequentially in the
  // current scope. A Success node emits its success op and continues with
  // the next alternative; a Test node either wraps its test + success path
  // in a `pdl_constr.try` (so failure transfers to the next sibling) or
  // emits the test bare when it's the only alternative at this scope.
  // `builder.clone(*pred, mapping)` reuses the canonical pool op as a
  // template and records the new SSA Values in `mapping`.
  //
  // A Test is wrapped iff it has a sibling alternative at the same scope
  // (either a following failurePath or an already-emitted preceding wrapped
  // sibling, tracked by `hadTestSibling`). The trailing test in a multi-way
  // chain must wrap too, even though its failure already transfers to the
  // enclosing scope: without the wrapper `foldSwitchAt` cannot collapse it
  // into a sibling `switch_*` since that requires a contiguous run of
  // `TryOp`s. A *sole* test alternative still emits bare — wrapping it
  // would add empty scaffolding for no fold opportunity.
  bool hadTestSibling = false;
  while (node) {
    if (node->kind == TreeNode::Kind::Success) {
      SmallVector<Value, 4> inputs;
      inputs.reserve(node->successInputs.size());
      for (Value canonical : node->successInputs)
        inputs.push_back(mapping.lookup(canonical));
      SuccessOp orig = node->originalSuccess;
      SuccessOp::create(builder, loc, orig.getRewriterAttr(),
                        orig.getBenefitAttr(), inputs);
      node = node->failurePath.get();
      continue;
    }

    if (node->failurePath || hadTestSibling) {
      TryOp tryOp = TryOp::create(builder, loc);
      Block &tryBlock = tryOp.getBody().emplaceBlock();
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&tryBlock);
        IRMapping tryMapping = mapping;
        builder.clone(*node->pred, tryMapping);
        emitNode(builder, loc, node->successPath.get(), tryMapping);
      }
      hadTestSibling = true;
      node = node->failurePath.get();
    } else {
      // Sole test alternative at this scope: emit bare; failure transfers
      // to the enclosing failure scope.
      builder.clone(*node->pred, mapping);
      node = node->successPath.get();
    }
  }
}

//===----------------------------------------------------------------------===//
// Switch folding
//
// After tree emission, sibling `try` blocks at the same scope level that all
// start with a `pdl_constr.has_name` (or `pdl_constr.has_type`) on the same
// operand are folded into a single `pdl_constr.switch_op_name` (or
// `pdl_constr.switch_type`). Mirrors SwitchNode emission in PredicateTree.cpp.
//===----------------------------------------------------------------------===//

namespace {

template <typename TestOpT, typename GetTestedValueFn, typename GetCaseAttrFn>
static bool foldSwitchAt(Block &block, Block::iterator startIt,
                         GetTestedValueFn getTested, GetCaseAttrFn getCaseAttr,
                         SmallVectorImpl<TryOp> &cases, Value &operand,
                         SmallVectorImpl<Attribute> &caseAttrs) {
  operand = nullptr;
  cases.clear();
  caseAttrs.clear();
  for (auto it = startIt; it != block.end(); ++it) {
    auto tryOp = dyn_cast<TryOp>(*it);
    if (!tryOp)
      break;
    Block &tb = tryOp.getBody().front();
    if (tb.empty())
      break;
    auto testOp = dyn_cast<TestOpT>(&tb.front());
    if (!testOp)
      break;
    Value tested = getTested(testOp);
    // The tested value must come from the enclosing scope (not inside the
    // try body), otherwise we cannot hoist the test.
    if (tested.getParentBlock() == &tb)
      break;
    if (!operand)
      operand = tested;
    else if (operand != tested)
      break;
    cases.push_back(tryOp);
    caseAttrs.push_back(getCaseAttr(testOp));
  }
  return cases.size() >= 2;
}

} // namespace

void Combiner::foldSwitches(Region &region) {
  for (Block &block : region) {
    for (auto it = block.begin(); it != block.end();) {
      // --- has_name → switch_op_name ---
      {
        SmallVector<TryOp> cases;
        SmallVector<Attribute> attrs;
        Value operand;
        if (foldSwitchAt<HasNameOp>(
                block, it, [](HasNameOp h) { return h.getOp(); },
                [](HasNameOp h) -> Attribute { return h.getNameAttr(); }, cases,
                operand, attrs)) {
          OpBuilder builder(cases.front());
          Location loc = cases.front().getLoc();
          auto switchOp = SwitchOpNameOp::create(
              builder, loc, operand, builder.getArrayAttr(attrs),
              /*caseRegionsCount=*/cases.size());
          for (auto [i, caseTry] : llvm::enumerate(cases)) {
            Region &caseRegion = switchOp.getCaseRegions()[i];
            Block &newBlock = caseRegion.emplaceBlock();
            Block &oldBlock = caseTry.getBody().front();
            oldBlock.front().erase(); // drop the has_name test
            newBlock.getOperations().splice(newBlock.end(),
                                            oldBlock.getOperations());
          }
          // Erase the now-empty case tries before taking `next(switchOp)`:
          // switchOp was inserted just before cases.front(), so taking the
          // iterator first would leave it dangling once the trys are erased.
          for (TryOp t : cases)
            t.erase();
          it = std::next(Block::iterator(switchOp));
          continue;
        }
      }

      // --- has_type → switch_type ---
      {
        SmallVector<TryOp> cases;
        SmallVector<Attribute> attrs;
        Value operand;
        if (foldSwitchAt<HasTypeOp>(
                block, it, [](HasTypeOp h) { return h.getTypeValue(); },
                [](HasTypeOp h) -> Attribute { return h.getConstantTypeAttr(); },
                cases, operand, attrs)) {
          OpBuilder builder(cases.front());
          Location loc = cases.front().getLoc();
          auto switchOp = SwitchTypeOp::create(
              builder, loc, operand, builder.getArrayAttr(attrs),
              /*caseRegionsCount=*/cases.size());
          for (auto [i, caseTry] : llvm::enumerate(cases)) {
            Region &caseRegion = switchOp.getCaseRegions()[i];
            Block &newBlock = caseRegion.emplaceBlock();
            Block &oldBlock = caseTry.getBody().front();
            oldBlock.front().erase();
            newBlock.getOperations().splice(newBlock.end(),
                                            oldBlock.getOperations());
          }
          for (TryOp t : cases)
            t.erase();
          it = std::next(Block::iterator(switchOp));
          continue;
        }
      }

      ++it;
    }
  }

  // Recurse into nested regions.
  for (Block &block : region)
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        foldSwitches(nested);
}

//===----------------------------------------------------------------------===//
// Top-level emission
//===----------------------------------------------------------------------===//

void Combiner::emitCombinedMatcher() {
  if (matcherInfos.empty())
    return;

  OpBuilder builder(module.getContext());
  Location loc = module.getLoc();
  builder.setInsertionPointToStart(module.getBody());

  StringAttr symName;
  if (auto firstName = matcherInfos.front().matcher.getSymNameAttr())
    symName = firstName;

  auto combined = MatcherOp::create(builder, loc, symName);
  Block *body = &combined.getBodyRegion().emplaceBlock();
  Value rootArg =
      body->addArgument(builder.getType<pdl::OperationType>(), loc);

  // Seed the pool->combined mapping with the canonical root.
  IRMapping mapping;
  mapping.map(poolRoot, rootArg);

  builder.setInsertionPointToStart(body);
  emitNode(builder, loc, treeRoot.get(), mapping);

  foldSwitches(combined.getBodyRegion());

  for (MatcherInfo &info : matcherInfos)
    info.matcher.erase();
}

LogicalResult Combiner::run() {
  if (failed(buildCanonicalPool()))
    return failure();
  sortCanonicalPool();
  buildTree();
  emitCombinedMatcher();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

namespace {
struct CombineMatchersPass
    : public mlir::pdl_constr::impl::PDLConstrCombineMatchersPassBase<
          CombineMatchersPass> {
  void runOnOperation() final {
    Combiner combiner(getOperation());
    if (failed(combiner.run()))
      signalPassFailure();
  }
};
} // namespace
