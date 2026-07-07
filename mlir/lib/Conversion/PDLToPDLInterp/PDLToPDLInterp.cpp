//===- PDLToPDLInterp.cpp - Lower a PDL module to the interpreter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"

#include "PredicateTree.h"
#include "mlir/Conversion/PDLToPDLInterp/RewriterGen.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPDLTOPDLINTERPPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pdl_to_pdl_interp;

//===----------------------------------------------------------------------===//
// PatternLowering
//===----------------------------------------------------------------------===//

namespace {
/// This class generators operations within the PDL Interpreter dialect from a
/// given module containing PDL pattern operations.
struct PatternLowering {
public:
  PatternLowering(pdl_interp::FuncOp matcherFunc, ModuleOp rewriterModule,
                  DenseMap<Operation *, PDLPatternConfigSet *> *configMap);

  /// Generate code for matching and rewriting based on the pattern operations
  /// within the module.
  void lower(ModuleOp module);

private:
  using ValueMap = llvm::ScopedHashTable<Position *, Value>;
  using ValueMapScope = llvm::ScopedHashTableScope<Position *, Value>;

  /// Generate interpreter operations for the tree rooted at the given matcher
  /// node, in the specified region.
  Block *generateMatcher(MatcherNode &node, Region &region,
                         Block *block = nullptr);

  /// Get or create an access to the provided positional value in the current
  /// block. This operation may mutate the provided block pointer if nested
  /// regions (i.e., pdl_interp.iterate) are required.
  Value getValueAt(Block *&currentBlock, Position *pos);

  /// Create the interpreter predicate operations. This operation may mutate the
  /// provided current block pointer if nested regions (iterates) are required.
  void generate(BoolNode *boolNode, Block *&currentBlock, Value val);

  /// Create the interpreter switch / predicate operations, with several case
  /// destinations. This operation never mutates the provided current block
  /// pointer, because the switch operation does not need Values beyond `val`.
  void generate(SwitchNode *switchNode, Block *currentBlock, Value val);

  /// Create the interpreter operations to record a successful pattern match
  /// using the contained root operation. This operation may mutate the current
  /// block pointer if nested regions (i.e., pdl_interp.iterate) are required.
  void generate(SuccessNode *successNode, Block *&currentBlock);

  /// A builder to use when generating interpreter operations.
  OpBuilder builder;

  /// The matcher function used for all match related logic within PDL patterns.
  pdl_interp::FuncOp matcherFunc;

  /// The rewriter module containing the all rewrite related logic within PDL
  /// patterns.
  ModuleOp rewriterModule;

  /// The symbol table of the rewriter module used for insertion.
  SymbolTable rewriterSymbolTable;

  /// A scoped map connecting a position with the corresponding interpreter
  /// value.
  ValueMap values;

  /// A stack of blocks used as the failure destination for matcher nodes that
  /// don't have an explicit failure path.
  SmallVector<Block *, 8> failureBlockStack;

  /// A mapping between values defined in a pattern match, and the corresponding
  /// positional value.
  DenseMap<Value, Position *> valueToPosition;

  /// The set of operation values whose location will be used for newly
  /// generated operations.
  SetVector<Value> locOps;

  /// A mapping between pattern operations and the corresponding configuration
  /// set.
  DenseMap<Operation *, PDLPatternConfigSet *> *configMap;

  /// A mapping from a constraint question to the ApplyConstraintOp
  /// that implements it.
  DenseMap<ConstraintQuestion *, pdl_interp::ApplyConstraintOp> constraintOpMap;
};
} // namespace

PatternLowering::PatternLowering(
    pdl_interp::FuncOp matcherFunc, ModuleOp rewriterModule,
    DenseMap<Operation *, PDLPatternConfigSet *> *configMap)
    : builder(matcherFunc.getContext()), matcherFunc(matcherFunc),
      rewriterModule(rewriterModule), rewriterSymbolTable(rewriterModule),
      configMap(configMap) {}

void PatternLowering::lower(ModuleOp module) {
  PredicateUniquer predicateUniquer;
  PredicateBuilder predicateBuilder(predicateUniquer, module.getContext());

  // Define top-level scope for the arguments to the matcher function.
  ValueMapScope topLevelValueScope(values);

  // Insert the root operation, i.e. argument to the matcher, at the root
  // position.
  Block *matcherEntryBlock = &matcherFunc.front();
  values.insert(predicateBuilder.getRoot(), matcherEntryBlock->getArgument(0));

  // Generate a root matcher node from the provided PDL module.
  std::unique_ptr<MatcherNode> root = MatcherNode::generateMatcherTree(
      module, predicateBuilder, valueToPosition);
  Block *firstMatcherBlock = generateMatcher(*root, matcherFunc.getBody());
  assert(failureBlockStack.empty() && "failed to empty the stack");

  // After generation, merged the first matched block into the entry.
  matcherEntryBlock->getOperations().splice(matcherEntryBlock->end(),
                                            firstMatcherBlock->getOperations());
  firstMatcherBlock->erase();
}

Block *PatternLowering::generateMatcher(MatcherNode &node, Region &region,
                                        Block *block) {
  // Push a new scope for the values used by this matcher.
  if (!block)
    block = &region.emplaceBlock();
  ValueMapScope scope(values);

  // If this is the return node, simply insert the corresponding interpreter
  // finalize.
  if (isa<ExitNode>(node)) {
    builder.setInsertionPointToEnd(block);
    pdl_interp::FinalizeOp::create(builder, matcherFunc.getLoc());
    return block;
  }

  // Get the next block in the match sequence.
  // This is intentionally executed first, before we get the value for the
  // position associated with the node, so that we preserve an "there exist"
  // semantics: if getting a value requires an upward traversal (going from a
  // value to its consumers), we want to perform the check on all the consumers
  // before we pass control to the failure node.
  std::unique_ptr<MatcherNode> &failureNode = node.getFailureNode();
  Block *failureBlock;
  if (failureNode) {
    failureBlock = generateMatcher(*failureNode, region);
    failureBlockStack.push_back(failureBlock);
  } else {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    failureBlock = failureBlockStack.back();
  }

  // If this node contains a position, get the corresponding value for this
  // block.
  Block *currentBlock = block;
  Position *position = node.getPosition();
  Value val = position ? getValueAt(currentBlock, position) : Value();

  // If this value corresponds to an operation, record that we are going to use
  // its location as part of a fused location.
  bool isOperationValue = val && isa<pdl::OperationType>(val.getType());
  if (isOperationValue)
    locOps.insert(val);

  // Dispatch to the correct method based on derived node type.
  TypeSwitch<MatcherNode *>(&node)
      .Case<BoolNode, SwitchNode>([&](auto *derivedNode) {
        this->generate(derivedNode, currentBlock, val);
      })
      .Case([&](SuccessNode *successNode) {
        generate(successNode, currentBlock);
      });

  // Pop all the failure blocks that were inserted due to nesting of
  // pdl_interp.iterate.
  while (failureBlockStack.back() != failureBlock) {
    failureBlockStack.pop_back();
    assert(!failureBlockStack.empty() && "unable to locate failure block");
  }

  // Pop the new failure block.
  if (failureNode)
    failureBlockStack.pop_back();

  if (isOperationValue)
    locOps.remove(val);

  return block;
}

Value PatternLowering::getValueAt(Block *&currentBlock, Position *pos) {
  if (Value val = values.lookup(pos))
    return val;

  // Get the value for the parent position.
  Value parentVal;
  if (Position *parent = pos->getParent())
    parentVal = getValueAt(currentBlock, parent);

  // TODO: Use a location from the position.
  Location loc = parentVal ? parentVal.getLoc() : builder.getUnknownLoc();
  builder.setInsertionPointToEnd(currentBlock);
  Value value;
  switch (pos->getKind()) {
  case Predicates::OperationPos: {
    auto *operationPos = cast<OperationPosition>(pos);
    if (operationPos->isOperandDefiningOp())
      // Standard (downward) traversal which directly follows the defining op.
      value = pdl_interp::GetDefiningOpOp::create(
          builder, loc, builder.getType<pdl::OperationType>(), parentVal);
    else
      // A passthrough operation position.
      value = parentVal;
    break;
  }
  case Predicates::UsersPos: {
    auto *usersPos = cast<UsersPosition>(pos);

    // The first operation retrieves the representative value of a range.
    // This applies only when the parent is a range of values and we were
    // requested to use a representative value (e.g., upward traversal).
    if (isa<pdl::RangeType>(parentVal.getType()) &&
        usersPos->useRepresentative())
      value = pdl_interp::ExtractOp::create(builder, loc, parentVal, 0);
    else
      value = parentVal;

    // The second operation retrieves the users.
    value = pdl_interp::GetUsersOp::create(builder, loc, value);
    break;
  }
  case Predicates::ForEachPos: {
    assert(!failureBlockStack.empty() && "expected valid failure block");
    auto foreach = pdl_interp::ForEachOp::create(
        builder, loc, parentVal, failureBlockStack.back(), /*initLoop=*/true);
    value = foreach.getLoopVariable();

    // Create the continuation block.
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
    // Due to the order of traversal, the ApplyConstraintOp has already been
    // created and we can find it in constraintOpMap.
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

void PatternLowering::generate(BoolNode *boolNode, Block *&currentBlock,
                               Value val) {
  Location loc = val.getLoc();
  Qualifier *question = boolNode->getQuestion();
  Qualifier *answer = boolNode->getAnswer();
  Region *region = currentBlock->getParent();

  // Execute the getValue queries first, so that we create success
  // matcher in the correct (possibly nested) region.
  SmallVector<Value> args;
  if (auto *equalToQuestion = dyn_cast<EqualToQuestion>(question)) {
    args = {getValueAt(currentBlock, equalToQuestion->getValue())};
  } else if (auto *cstQuestion = dyn_cast<ConstraintQuestion>(question)) {
    for (Position *position : cstQuestion->getArgs())
      args.push_back(getValueAt(currentBlock, position));
  }

  // Generate a new block as success successor and get the failure successor.
  Block *success = &region->emplaceBlock();
  Block *failure = failureBlockStack.back();

  // Create the predicate.
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

  // Generate the matcher in the current (potentially nested) region.
  // This might use the results of the current predicate.
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

void PatternLowering::generate(SwitchNode *switchNode, Block *currentBlock,
                               Value val) {
  Qualifier *question = switchNode->getQuestion();
  Region *region = currentBlock->getParent();
  Block *defaultDest = failureBlockStack.back();

  // If the switch question is not an exact answer, i.e. for the `at_least`
  // cases, we generate a special block sequence.
  Predicates::Kind kind = question->getKind();
  if (kind == Predicates::OperandCountAtLeastQuestion ||
      kind == Predicates::ResultCountAtLeastQuestion) {
    // Order the children such that the cases are in reverse numerical order.
    SmallVector<unsigned> sortedChildren = llvm::to_vector<16>(
        llvm::seq<unsigned>(0, switchNode->getChildren().size()));
    llvm::sort(sortedChildren, [&](unsigned lhs, unsigned rhs) {
      return cast<UnsignedAnswer>(switchNode->getChild(lhs).first)->getValue() >
             cast<UnsignedAnswer>(switchNode->getChild(rhs).first)->getValue();
    });

    // Build the destination for each child using the next highest child as a
    // a failure destination. This essentially creates the following control
    // flow:
    //
    // if (operand_count < 1)
    //   goto failure
    // if (child1.match())
    //   ...
    //
    // if (operand_count < 2)
    //   goto failure
    // if (child2.match())
    //   ...
    //
    // failure:
    //   ...
    //
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

  // Otherwise, generate each of the children and generate an interpreter
  // switch.
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
    if (isa<pdl::RangeType>(val.getType())) {
      return createSwitchOp<pdl_interp::SwitchTypesOp, TypeAnswer>(
          val, defaultDest, builder, children);
    }
    return createSwitchOp<pdl_interp::SwitchTypeOp, TypeAnswer>(
        val, defaultDest, builder, children);
  case Predicates::AttributeQuestion:
    return createSwitchOp<pdl_interp::SwitchAttributeOp, AttributeAnswer>(
        val, defaultDest, builder, children);
  default:
    llvm_unreachable("Generating unknown switch predicate.");
  }
}

void PatternLowering::generate(SuccessNode *successNode, Block *&currentBlock) {
  pdl::PatternOp pattern = successNode->getPattern();
  Value root = successNode->getRoot();

  // Generate a rewriter for the pattern this success node represents, and track
  // any values used from the match region.
  SmallVector<Value, 8> usedMatchValues;
  SymbolRefAttr rewriterFuncRef = pdl_to_pdl_interp::generatePatternRewriter(
      pattern, rewriterModule, rewriterSymbolTable, builder, usedMatchValues);

  // Process any values used in the rewrite that are defined in the match.
  std::vector<Value> mappedMatchValues;
  mappedMatchValues.reserve(usedMatchValues.size());
  for (Value matchValue : usedMatchValues) {
    Position *inputPos = valueToPosition.lookup(matchValue);
    assert(inputPos && "expected value to be a pattern input");
    mappedMatchValues.push_back(getValueAt(currentBlock, inputPos));
  }

  // Collect the set of operations generated by the rewriter.
  SmallVector<StringRef, 4> generatedOps;
  for (auto op :
       pattern.getRewriter().getBodyRegion().getOps<pdl::OperationOp>())
    generatedOps.push_back(*op.getOpName());
  ArrayAttr generatedOpsAttr;
  if (!generatedOps.empty())
    generatedOpsAttr = builder.getStrArrayAttr(generatedOps);

  // Grab the root kind if present.
  StringAttr rootKindAttr;
  if (pdl::OperationOp rootOp = root.getDefiningOp<pdl::OperationOp>())
    if (std::optional<StringRef> rootKind = rootOp.getOpName())
      rootKindAttr = builder.getStringAttr(*rootKind);

  builder.setInsertionPointToEnd(currentBlock);
  auto matchOp = pdl_interp::RecordMatchOp::create(
      builder, pattern.getLoc(), mappedMatchValues, locOps.getArrayRef(),
      rewriterFuncRef, rootKindAttr, generatedOpsAttr, pattern.getBenefitAttr(),
      failureBlockStack.back());

  // Set the config of the lowered match to the parent pattern.
  if (configMap)
    configMap->try_emplace(matchOp, configMap->lookup(pattern));
}


//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct PDLToPDLInterpPass
    : public impl::ConvertPDLToPDLInterpPassBase<PDLToPDLInterpPass> {
  PDLToPDLInterpPass() = default;
  PDLToPDLInterpPass(const PDLToPDLInterpPass &rhs) = default;
  PDLToPDLInterpPass(DenseMap<Operation *, PDLPatternConfigSet *> &configMap)
      : configMap(&configMap) {}
  void runOnOperation() final;

  /// A map containing the configuration for each pattern.
  DenseMap<Operation *, PDLPatternConfigSet *> *configMap = nullptr;
};
} // namespace

/// Convert the given module containing PDL pattern operations into a PDL
/// Interpreter operations.
void PDLToPDLInterpPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Create the main matcher function This function contains all of the match
  // related functionality from patterns in the module.
  OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
  auto matcherFunc = pdl_interp::FuncOp::create(
      builder, module.getLoc(),
      pdl_interp::PDLInterpDialect::getMatcherFunctionName(),
      builder.getFunctionType(builder.getType<pdl::OperationType>(),
                              /*results=*/{}),
      /*attrs=*/ArrayRef<NamedAttribute>());

  // Create a nested module to hold the functions invoked for rewriting the IR
  // after a successful match.
  ModuleOp rewriterModule =
      ModuleOp::create(builder, module.getLoc(),
                       pdl_interp::PDLInterpDialect::getRewriterModuleName());

  // Generate the code for the patterns within the module.
  PatternLowering generator(matcherFunc, rewriterModule, configMap);
  generator.lower(module);

  // After generation, delete all of the pattern operations.
  for (pdl::PatternOp pattern :
       llvm::make_early_inc_range(module.getOps<pdl::PatternOp>())) {
    // Drop the now dead config mappings.
    if (configMap)
      configMap->erase(pattern);

    pattern.erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertPDLToPDLInterpPass(
    DenseMap<Operation *, PDLPatternConfigSet *> &configMap) {
  return std::make_unique<PDLToPDLInterpPass>(configMap);
}
