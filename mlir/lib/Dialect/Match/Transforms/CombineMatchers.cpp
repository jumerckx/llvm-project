//===- CombineMatchers.cpp - Combine match.matcher ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Match/Transforms/Passes.h"

#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/Match/IR/MatchOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace match {
#define GEN_PASS_DEF_MATCHCOMBINEMATCHERSPASS
#include "mlir/Dialect/Match/Transforms/Passes.h.inc"
} // namespace match
} // namespace mlir

using namespace mlir;
using namespace mlir::match;

namespace {
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
  static unsigned getHashValue(const PoolKey &k) {
    return llvm::hash_combine(
        k.name, k.attrs,
        llvm::hash_combine_range(k.operands.begin(), k.operands.end()));
  }
  static bool isEqual(const PoolKey &a, const PoolKey &b) { return a == b; }
};
} // namespace llvm

namespace {
struct MatcherInfo {
  MatcherOp matcher;
  SuccessOp success;
  DenseSet<Operation *> preds;
  SmallVector<Operation *> guards;
  IRMapping toCanonical;
};

struct TreeNode {
  enum class Kind { Test, Success };
  Kind kind;
  Operation *pred = nullptr;
  std::unique_ptr<TreeNode> successPath;
  std::unique_ptr<TreeNode> failurePath;
  SuccessOp originalSuccess;
  SmallVector<Value, 4> successInputs;
};

class Combiner {
public:
  Combiner(ModuleOp module) : module(module) {}
  LogicalResult run();

private:
  LogicalResult buildCanonicalPool();
  LogicalResult collectPredicates(Block &block, MatcherInfo &info);
  void sortCanonicalPool();
  void emitCombinedMatcher();

  Operation *getOrCreatePoolOp(Operation *modelOp,
                               ArrayRef<Value> canonicalOperands,
                               const DenseSet<Operation *> &alreadyUsed);
  void propagate(std::unique_ptr<TreeNode> &node, MatcherInfo &info,
                 unsigned predIdx);
  void emitNode(OpBuilder &builder, Location loc, TreeNode *node,
                IRMapping &mapping);

  ModuleOp module;
  std::unique_ptr<Block> poolBody;
  Value poolRoot;
  DenseMap<PoolKey, Operation *> poolDedup;
  DenseMap<Operation *, unsigned> primary;
  DenseMap<Operation *, unsigned> secondary;
  DenseMap<Operation *, unsigned> insertionIndex;
  unsigned nextInsertionIndex = 0;
  std::vector<Operation *> sortedPoolOps;
  DenseSet<std::pair<Operation *, Operation *>> guardEdges;
  std::vector<MatcherInfo> matcherInfos;
  std::unique_ptr<TreeNode> treeRoot;
};
} // namespace

Operation *
Combiner::getOrCreatePoolOp(Operation *modelOp,
                            ArrayRef<Value> canonicalOperands,
                            const DenseSet<Operation *> &alreadyUsed) {
  PoolKey key{modelOp->getName(), modelOp->getAttrDictionary(),
              SmallVector<Value, 4>(canonicalOperands)};
  if (auto it = poolDedup.find(key); it != poolDedup.end()) {
    if (!isa<ForEachOp>(modelOp) || !alreadyUsed.contains(it->second))
      return it->second;
  }

  IRMapping mapping;
  for (auto [orig, canonical] :
       llvm::zip(modelOp->getOperands(), canonicalOperands))
    mapping.map(orig, canonical);
  Operation *poolOp = modelOp->cloneWithoutRegions(mapping);
  poolBody->push_back(poolOp);

  if (auto forEach = dyn_cast<ForEachOp>(poolOp)) {
    Block &block = forEach.getBody().emplaceBlock();
    block.addArgument(forEach.getElementType(), forEach.getLoc());
  }

  insertionIndex[poolOp] = nextInsertionIndex++;
  poolDedup[key] = poolOp;
  return poolOp;
}

LogicalResult Combiner::collectPredicates(Block &block, MatcherInfo &info) {
  for (Operation &op : block) {
    if (auto succ = dyn_cast<SuccessOp>(&op)) {
      if (info.success)
        return op.emitOpError("combine-matchers expects exactly one "
                              "`match.success` per input matcher");
      info.success = succ;
      continue;
    }

    if (isa<TryOp, SwitchOpNameOp, SwitchTypeOp>(&op))
      return op.emitOpError(
          "combine-matchers does not support nested `try`/`switch_*` in "
          "input matchers");

    if (isa<ForEachOp>(&op) && op.getNextNode())
      return op.emitOpError(
          "combine-matchers requires `match.foreach` to be the last op in its "
          "block; ops following the loop would be absorbed into its body");

    SmallVector<Value, 4> canonicalOperands;
    canonicalOperands.reserve(op.getNumOperands());
    for (Value operand : op.getOperands())
      canonicalOperands.push_back(info.toCanonical.lookup(operand));

    Operation *poolOp = getOrCreatePoolOp(&op, canonicalOperands, info.preds);

    for (auto [origRes, poolRes] :
         llvm::zip(op.getResults(), poolOp->getResults()))
      info.toCanonical.map(origRes, poolRes);

    if (info.preds.insert(poolOp).second)
      ++primary[poolOp];

    if (isa<ApplyNativeConstraintOp>(poolOp)) {
      for (Operation *guard : info.guards)
        guardEdges.insert({guard, poolOp});
    } else {
      info.guards.push_back(poolOp);
    }

    if (auto forEach = dyn_cast<ForEachOp>(&op)) {
      info.toCanonical.map(forEach.getLoopVariable(),
                           cast<ForEachOp>(poolOp).getLoopVariable());
      if (failed(collectPredicates(forEach.getBody().front(), info)))
        return failure();
    }
  }
  return success();
}

LogicalResult Combiner::buildCanonicalPool() {
  SmallVector<MatcherOp> inputMatchers(module.getOps<MatcherOp>());

  poolBody = std::make_unique<Block>();
  poolRoot = poolBody->addArgument(pdl::OperationType::get(module.getContext()),
                                   module.getLoc());

  for (MatcherOp matcher : inputMatchers) {
    MatcherInfo info;
    info.matcher = matcher;

    Block &body = matcher.getBodyRegion().front();
    info.toCanonical.map(body.getArgument(0), poolRoot);

    if (failed(collectPredicates(body, info)))
      return failure();

    if (!info.success)
      return matcher.emitOpError(
          "combine-matchers expects a `match.success` in every input "
          "matcher");

    matcherInfos.push_back(std::move(info));
  }

  for (MatcherInfo &info : matcherInfos) {
    unsigned total = 0;
    for (Operation *p : info.preds)
      total += primary[p] * primary[p];
    for (Operation *p : info.preds)
      secondary[p] += total;
  }

  return success();
}

static bool dependsOn(Operation *a, Operation *b) {
  for (Value operand : b->getOperands()) {
    if (operand.getDefiningOp() == a)
      return true;
    if (auto arg = dyn_cast<BlockArgument>(operand))
      if (arg.getOwner()->getParentOp() == a)
        return true;
  }
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

void Combiner::sortCanonicalPool() {
  for (Operation &op : *poolBody)
    sortedPoolOps.push_back(&op);
  llvm::sort(sortedPoolOps, [&](Operation *a, Operation *b) {
    return std::make_tuple(primary[a], secondary[a], insertionIndex[b]) >
           std::make_tuple(primary[b], secondary[b], insertionIndex[a]);
  });
  stableTopologicalSort(sortedPoolOps.begin(), sortedPoolOps.end(),
                        [&](Operation *a, Operation *b) {
                          return dependsOn(a, b) || guardEdges.contains({a, b});
                        });
  for (Operation *op : sortedPoolOps)
    op->moveBefore(poolBody.get(), poolBody->end());
}

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

  propagate(node->failurePath, info, predIdx);
}

void Combiner::emitNode(OpBuilder &builder, Location loc, TreeNode *node,
                        IRMapping &mapping) {
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

    if (auto poolLoop = dyn_cast<ForEachOp>(node->pred)) {
      auto loop =
          ForEachOp::create(builder, loc, mapping.lookup(poolLoop.getRange()));
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&loop.getBody().front());
        IRMapping bodyMapping = mapping;
        bodyMapping.map(poolLoop.getLoopVariable(), loop.getLoopVariable());
        emitNode(builder, loc, node->successPath.get(), bodyMapping);
      }
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
      builder.clone(*node->pred, mapping);
      node = node->successPath.get();
    }
  }
}

template <typename SwitchOpT, typename TestOpT, typename GetTestedFn,
          typename GetCaseFn>
static bool foldSwitchAt(Block &block, Block::iterator &it,
                         GetTestedFn getTested, GetCaseFn getCase) {
  SmallVector<TryOp> cases;
  SmallVector<Attribute> caseAttrs;
  Value operand;
  for (auto cur = it; cur != block.end(); ++cur) {
    auto tryOp = dyn_cast<TryOp>(*cur);
    if (!tryOp)
      break;
    Block &body = tryOp.getBody().front();
    auto testOp = body.empty() ? TestOpT() : dyn_cast<TestOpT>(&body.front());
    if (!testOp)
      break;
    Value tested = getTested(testOp);
    if (tested.getParentBlock() == &body || (operand && operand != tested))
      break;
    operand = tested;
    cases.push_back(tryOp);
    caseAttrs.push_back(getCase(testOp));
  }
  if (cases.size() < 2)
    return false;

  OpBuilder builder(cases.front());
  auto switchOp =
      SwitchOpT::create(builder, cases.front().getLoc(), operand,
                        builder.getArrayAttr(caseAttrs), cases.size());
  for (auto [region, caseTry] : llvm::zip(switchOp.getCaseRegions(), cases)) {
    Block &newBlock = region.emplaceBlock();
    Block &oldBlock = caseTry.getBody().front();
    oldBlock.front().erase();
    newBlock.getOperations().splice(newBlock.end(), oldBlock.getOperations());
    caseTry.erase();
  }
  it = std::next(Block::iterator(switchOp));
  return true;
}

static void foldSwitches(Region &region) {
  for (Block &block : region) {
    for (auto it = block.begin(); it != block.end();) {
      if (foldSwitchAt<SwitchOpNameOp, HasNameOp>(
              block, it, [](HasNameOp op) { return op.getOp(); },
              [](HasNameOp op) -> Attribute { return op.getNameAttr(); }))
        continue;
      if (foldSwitchAt<SwitchTypeOp, HasTypeOp>(
              block, it, [](HasTypeOp op) { return op.getTypeValue(); },
              [](HasTypeOp op) -> Attribute {
                return op.getConstantTypeAttr();
              }))
        continue;
      ++it;
    }
  }
  for (Block &block : region)
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        foldSwitches(nested);
}

static bool isPureNavigationOp(Operation *op) {
  return isa<ConstantAttributeOp, ConstantTypeOp, ConstantTypesOp, GetOperandOp,
             GetOperandsOp, GetResultOp, GetResultsOp, GetAttributeOp,
             GetDefiningOpOp, GetValueTypeOp, GetAttributeTypeOp, GetUsersOp,
             ExtractOp>(op);
}

static Operation *ancestorInBlock(Operation *u, Block *block) {
  Operation *cur = u;
  while (cur->getBlock() != block) {
    cur = cur->getBlock()->getParentOp();
    assert(cur && "expected block to be an ancestor of the use");
  }
  return cur;
}

static Region *childRegionContaining(Operation *anchor, Operation *u) {
  Operation *cur = u;
  while (cur->getParentOp() != anchor)
    cur = cur->getParentOp();
  return cur->getParentRegion();
}

static std::pair<Block *, Operation *> computeSinkTarget(Operation *n) {
  Value v = n->getResult(0);
  if (v.use_empty())
    return {nullptr, nullptr};

  SmallVector<Operation *, 4> users(v.getUsers());
  Block *block = n->getBlock();
  while (true) {
    Operation *earliest = nullptr;
    Operation *commonAnchor = nullptr;
    bool allSameAnchor = true;
    for (Operation *u : users) {
      Operation *anchor = ancestorInBlock(u, block);
      if (!earliest || anchor->isBeforeInBlock(earliest))
        earliest = anchor;
      if (!commonAnchor)
        commonAnchor = anchor;
      else if (commonAnchor != anchor)
        allSameAnchor = false;
    }

    if (allSameAnchor && commonAnchor->getNumRegions() > 0 &&
        !llvm::is_contained(users, commonAnchor)) {
      Region *target = nullptr;
      bool single = true;
      for (Operation *u : users) {
        Region *r = childRegionContaining(commonAnchor, u);
        if (!target)
          target = r;
        else if (target != r) {
          single = false;
          break;
        }
      }
      if (single && target && target->hasOneBlock()) {
        block = &target->front();
        continue;
      }
    }

    return {block, earliest};
  }
}

static void sinkNavigationOps(Region &region) {
  SmallVector<Operation *> navOps;
  region.walk([&](Operation *op) {
    if (isPureNavigationOp(op))
      navOps.push_back(op);
  });

  bool erased = true;
  while (erased) {
    erased = false;
    for (Operation *&op : navOps) {
      if (op && op->use_empty()) {
        op->erase();
        op = nullptr;
        erased = true;
      }
    }
  }
  llvm::erase(navOps, nullptr);

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : navOps) {
      auto [block, before] = computeSinkTarget(op);
      if (!before)
        continue;
      if (op->getBlock() == block) {
        bool needMove = false;
        for (Operation *cur = op->getNextNode(); cur != before;
             cur = cur->getNextNode()) {
          assert(cur && "first use must come after the navigation op");
          if (!isPureNavigationOp(cur)) {
            needMove = true;
            break;
          }
        }
        if (!needMove)
          continue;
      }
      op->moveBefore(before);
      changed = true;
    }
  }
}

void Combiner::emitCombinedMatcher() {
  if (matcherInfos.empty())
    return;

  OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
  Location loc = module.getLoc();
  StringAttr symName = matcherInfos.size() == 1
                           ? matcherInfos.front().matcher.getSymNameAttr()
                           : builder.getStringAttr("combined");
  auto combined = MatcherOp::create(builder, loc, symName);
  Block *body = &combined.getBodyRegion().emplaceBlock();
  IRMapping mapping;
  mapping.map(poolRoot,
              body->addArgument(builder.getType<pdl::OperationType>(), loc));

  builder.setInsertionPointToStart(body);
  emitNode(builder, loc, treeRoot.get(), mapping);
  foldSwitches(combined.getBodyRegion());
  sinkNavigationOps(combined.getBodyRegion());

  for (MatcherInfo &info : matcherInfos)
    info.matcher.erase();
}

LogicalResult Combiner::run() {
  if (failed(buildCanonicalPool()))
    return failure();
  sortCanonicalPool();
  for (MatcherInfo &info : matcherInfos)
    propagate(treeRoot, info, /*predIdx=*/0);
  emitCombinedMatcher();
  return success();
}

LogicalResult mlir::match::combineMatchers(ModuleOp module) {
  return Combiner(module).run();
}

namespace {
struct CombineMatchersPass
    : public match::impl::MatchCombineMatchersPassBase<CombineMatchersPass> {
  void runOnOperation() final {
    if (failed(combineMatchers(getOperation())))
      signalPassFailure();
  }
};
} // namespace
