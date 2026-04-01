//===- ByteCode.cpp - Pattern ByteCode Interpreter ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR to byte-code generation and the interpreter.
//
//===----------------------------------------------------------------------===//

#include "ByteCode.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InterleavedRange.h"
#include <numeric>
#include <optional>

#define DEBUG_TYPE "pdl-bytecode"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// PDLByteCodePattern
//===----------------------------------------------------------------------===//

PDLByteCodePattern PDLByteCodePattern::create(pdl_interp::RecordMatchOp matchOp,
                                              PDLPatternConfigSet *configSet,
                                              ByteCodeAddr rewriterAddr) {
  PatternBenefit benefit = matchOp.getBenefit();
  MLIRContext *ctx = matchOp.getContext();

  // Collect the set of generated operations.
  SmallVector<StringRef, 8> generatedOps;
  if (ArrayAttr generatedOpsAttr = matchOp.getGeneratedOpsAttr())
    generatedOps =
        llvm::to_vector<8>(generatedOpsAttr.getAsValueRange<StringAttr>());

  // Check to see if this is pattern matches a specific operation type.
  if (std::optional<StringRef> rootKind = matchOp.getRootKind())
    return PDLByteCodePattern(rewriterAddr, configSet, *rootKind, benefit, ctx,
                              generatedOps);
  return PDLByteCodePattern(rewriterAddr, configSet, MatchAnyOpTypeTag(),
                            benefit, ctx, generatedOps);
}

//===----------------------------------------------------------------------===//
// PDLByteCodeMutableState
//===----------------------------------------------------------------------===//

/// Set the new benefit for a bytecode pattern. The `patternIndex` corresponds
/// to the position of the pattern within the range returned by
/// `PDLByteCode::getPatterns`.
void PDLByteCodeMutableState::updatePatternBenefit(unsigned patternIndex,
                                                   PatternBenefit benefit) {
  currentPatternBenefits[patternIndex] = benefit;
}

/// Cleanup any allocated state after a full match/rewrite has been completed.
/// This method should be called irregardless of whether the match+rewrite was a
/// success or not.
void PDLByteCodeMutableState::cleanupAfterMatchAndRewrite() {
  allocatedTypeRangeMemory.clear();
  allocatedValueRangeMemory.clear();
}

//===----------------------------------------------------------------------===//
// Bytecode OpCodes
//===----------------------------------------------------------------------===//

namespace {
enum OpCode : ByteCodeField {
  /// Apply an externally registered constraint.
  ApplyConstraint,
  /// Apply an externally registered rewrite.
  ApplyRewrite,
  /// Check if two generic values are equal.
  AreEqual,
  /// Check if two ranges are equal.
  AreRangesEqual,
  /// Unconditional branch.
  Branch,
  /// Compare the operand count of an operation with a constant.
  CheckOperandCount,
  /// Compare the name of an operation with a constant.
  CheckOperationName,
  /// Compare the result count of an operation with a constant.
  CheckResultCount,
  /// Compare a range of types to a constant range of types.
  CheckTypes,
  /// Continue to the next iteration of a loop.
  Continue,
  /// Create a type range from a list of constant types.
  CreateConstantTypeRange,
  /// Create an operation.
  CreateOperation,
  /// Create a type range from a list of dynamic types.
  CreateDynamicTypeRange,
  /// Create a value range.
  CreateDynamicValueRange,
  /// Erase an operation.
  EraseOp,
  /// Extract the op from a range at the specified index.
  ExtractOp,
  /// Extract the type from a range at the specified index.
  ExtractType,
  /// Extract the value from a range at the specified index.
  ExtractValue,
  /// Terminate a matcher or rewrite sequence.
  Finalize,
  /// Iterate over a range of values.
  ForEach,
  /// Get a specific attribute of an operation.
  GetAttribute,
  /// Get the type of an attribute.
  GetAttributeType,
  /// Get the defining operation of a value.
  GetDefiningOp,
  /// Get a specific operand of an operation.
  GetOperand0,
  GetOperand1,
  GetOperand2,
  GetOperand3,
  GetOperandN,
  /// Get a specific operand group of an operation.
  GetOperands,
  /// Get a specific result of an operation.
  GetResult0,
  GetResult1,
  GetResult2,
  GetResult3,
  GetResultN,
  /// Get a specific result group of an operation.
  GetResults,
  /// Get the users of a value or a range of values.
  GetUsers,
  /// Get the type of a value.
  GetValueType,
  /// Get the types of a value range.
  GetValueRangeTypes,
  /// Check if a generic value is not null.
  IsNotNull,
  /// Record a successful pattern match.
  RecordMatch,
  /// Replace an operation.
  ReplaceOp,
  /// Compare an attribute with a set of constants.
  SwitchAttribute,
  /// Compare the operand count of an operation with a set of constants.
  SwitchOperandCount,
  /// Compare the name of an operation with a set of constants.
  SwitchOperationName,
  /// Compare the result count of an operation with a set of constants.
  SwitchResultCount,
  /// Compare a type with a set of constants.
  SwitchType,
  /// Compare a range of types with a set of constants.
  SwitchTypes,
};
} // namespace

/// A marker used to indicate if an operation should infer types.
static constexpr ByteCodeField kInferTypesMarker =
    std::numeric_limits<ByteCodeField>::max();

//===----------------------------------------------------------------------===//
// ByteCode Generation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Generator
//===----------------------------------------------------------------------===//

namespace {
struct ByteCodeLiveRange;
struct ByteCodeWriter;

/// Check if the given class `T` can be converted to an opaque pointer.
template <typename T, typename... Args>
using has_pointer_traits = decltype(std::declval<T>().getAsOpaquePointer());

/// This class represents the main generator for the pattern bytecode.
class Generator {
public:
  Generator(MLIRContext *ctx, std::vector<const void *> &uniquedData,
            SmallVectorImpl<ByteCodeField> &matcherByteCode,
            SmallVectorImpl<ByteCodeField> &rewriterByteCode,
            SmallVectorImpl<PDLByteCodePattern> &patterns,
            ByteCodeField &maxValueMemoryIndex,
            ByteCodeField &maxOpRangeMemoryIndex,
            ByteCodeField &maxTypeRangeMemoryIndex,
            ByteCodeField &maxValueRangeMemoryIndex,
            ByteCodeField &maxLoopLevel,
            llvm::StringMap<PDLConstraintFunction> &constraintFns,
            llvm::StringMap<PDLRewriteFunction> &rewriteFns,
            const DenseMap<Operation *, PDLPatternConfigSet *> &configMap)
      : ctx(ctx), uniquedData(uniquedData), matcherByteCode(matcherByteCode),
        rewriterByteCode(rewriterByteCode), patterns(patterns),
        maxValueMemoryIndex(maxValueMemoryIndex),
        maxOpRangeMemoryIndex(maxOpRangeMemoryIndex),
        maxTypeRangeMemoryIndex(maxTypeRangeMemoryIndex),
        maxValueRangeMemoryIndex(maxValueRangeMemoryIndex),
        maxLoopLevel(maxLoopLevel), configMap(configMap) {
    for (const auto &it : llvm::enumerate(constraintFns))
      constraintToMemIndex.try_emplace(it.value().first(), it.index());
    for (const auto &it : llvm::enumerate(rewriteFns))
      externalRewriterToMemIndex.try_emplace(it.value().first(), it.index());
  }

  /// Generate the bytecode for the given PDL interpreter module.
  void generate(ModuleOp module);

  /// Return the memory index to use for the given value.
  ByteCodeField &getMemIndex(Value value) {
    assert(valueToMemIndex.count(value) &&
           "expected memory index to be assigned");
    return valueToMemIndex[value];
  }

  /// Return the range memory index used to store the given range value.
  ByteCodeField &getRangeStorageIndex(Value value) {
    assert(valueToRangeIndex.count(value) &&
           "expected range index to be assigned");
    return valueToRangeIndex[value];
  }

  /// Return an index to use when referring to the given data that is uniqued in
  /// the MLIR context.
  template <typename T>
  std::enable_if_t<!std::is_convertible<T, Value>::value, ByteCodeField &>
  getMemIndex(T val) {
    const void *opaqueVal = val.getAsOpaquePointer();

    // Get or insert a reference to this value.
    auto it = uniquedDataToMemIndex.try_emplace(
        opaqueVal, maxValueMemoryIndex + uniquedData.size());
    if (it.second)
      uniquedData.push_back(opaqueVal);
    return it.first->second;
  }

private:
  /// Allocate memory indices for the results of operations within the matcher
  /// and rewriters.
  void allocateMemoryIndices(pdl_interp::FuncOp matcherFunc,
                             ModuleOp rewriterModule);

  /// Generate the bytecode for the given operation.
  void generate(Region *region, ByteCodeWriter &writer);
  void generate(Operation *op, ByteCodeWriter &writer);
  void generate(pdl_interp::ApplyConstraintOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ApplyRewriteOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::AreEqualOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::BranchOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckOperandCountOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckOperationNameOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckResultCountOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CheckTypesOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ContinueOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateOperationOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateRangeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::CreateTypesOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::EraseOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ExtractOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::FinalizeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ForEachOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetAttributeTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetDefiningOpOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetOperandOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetOperandsOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetResultOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetResultsOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetUsersOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::GetValueTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::IsNotNullOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::RecordMatchOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::ReplaceOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchAttributeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchTypeOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchTypesOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchOperandCountOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchOperationNameOp op, ByteCodeWriter &writer);
  void generate(pdl_interp::SwitchResultCountOp op, ByteCodeWriter &writer);

  /// Mapping from value to its corresponding memory index.
  DenseMap<Value, ByteCodeField> valueToMemIndex;

  /// Mapping from a range value to its corresponding range storage index.
  DenseMap<Value, ByteCodeField> valueToRangeIndex;

  /// Mapping from the name of an externally registered rewrite to its index in
  /// the bytecode registry.
  llvm::StringMap<ByteCodeField> externalRewriterToMemIndex;

  /// Mapping from the name of an externally registered constraint to its index
  /// in the bytecode registry.
  llvm::StringMap<ByteCodeField> constraintToMemIndex;

  /// Mapping from rewriter function name to the bytecode address of the
  /// rewriter function in byte.
  llvm::StringMap<ByteCodeAddr> rewriterToAddr;

  /// Mapping from a uniqued storage object to its memory index within
  /// `uniquedData`.
  DenseMap<const void *, ByteCodeField> uniquedDataToMemIndex;

  /// The current level of the foreach loop.
  ByteCodeField curLoopLevel = 0;

  /// The current MLIR context.
  MLIRContext *ctx;

  /// Mapping from block to its address.
  DenseMap<Block *, ByteCodeAddr> blockToAddr;

  /// Data of the ByteCode class to be populated.
  std::vector<const void *> &uniquedData;
  SmallVectorImpl<ByteCodeField> &matcherByteCode;
  SmallVectorImpl<ByteCodeField> &rewriterByteCode;
  SmallVectorImpl<PDLByteCodePattern> &patterns;
  ByteCodeField &maxValueMemoryIndex;
  ByteCodeField &maxOpRangeMemoryIndex;
  ByteCodeField &maxTypeRangeMemoryIndex;
  ByteCodeField &maxValueRangeMemoryIndex;
  ByteCodeField &maxLoopLevel;

  /// A map of pattern configurations.
  const DenseMap<Operation *, PDLPatternConfigSet *> &configMap;
};

/// This class provides utilities for writing a bytecode stream.
struct ByteCodeWriter {
  ByteCodeWriter(SmallVectorImpl<ByteCodeField> &bytecode, Generator &generator)
      : bytecode(bytecode), generator(generator) {}

  /// Append a field to the bytecode.
  void append(ByteCodeField field) { bytecode.push_back(field); }
  void append(OpCode opCode) { bytecode.push_back(opCode); }

  /// Append an address to the bytecode.
  void append(ByteCodeAddr field) {
    static_assert((sizeof(ByteCodeAddr) / sizeof(ByteCodeField)) == 2,
                  "unexpected ByteCode address size");

    ByteCodeField fieldParts[2];
    std::memcpy(fieldParts, &field, sizeof(ByteCodeAddr));
    bytecode.append({fieldParts[0], fieldParts[1]});
  }

  /// Append a single successor to the bytecode, the exact address will need to
  /// be resolved later.
  void append(Block *successor) {
    // Add back a reference to the successor so that the address can be resolved
    // later.
    unresolvedSuccessorRefs[successor].push_back(bytecode.size());
    append(ByteCodeAddr(0));
  }

  /// Append a successor range to the bytecode, the exact address will need to
  /// be resolved later.
  void append(SuccessorRange successors) {
    for (Block *successor : successors)
      append(successor);
  }

  /// Append a range of values that will be read as generic PDLValues.
  void appendPDLValueList(OperandRange values) {
    bytecode.push_back(values.size());
    for (Value value : values)
      appendPDLValue(value);
  }

  /// Append a value as a PDLValue.
  void appendPDLValue(Value value) {
    appendPDLValueKind(value);
    append(value);
  }

  /// Append the PDLValue::Kind of the given value.
  void appendPDLValueKind(Value value) { appendPDLValueKind(value.getType()); }

  /// Append the PDLValue::Kind of the given type.
  void appendPDLValueKind(Type type) {
    PDLValue::Kind kind =
        TypeSwitch<Type, PDLValue::Kind>(type)
            .Case<pdl::AttributeType>(
                [](Type) { return PDLValue::Kind::Attribute; })
            .Case<pdl::OperationType>(
                [](Type) { return PDLValue::Kind::Operation; })
            .Case([](pdl::RangeType rangeTy) {
              if (isa<pdl::TypeType>(rangeTy.getElementType()))
                return PDLValue::Kind::TypeRange;
              return PDLValue::Kind::ValueRange;
            })
            .Case<pdl::TypeType>([](Type) { return PDLValue::Kind::Type; })
            .Case<pdl::ValueType>([](Type) { return PDLValue::Kind::Value; });
    bytecode.push_back(static_cast<ByteCodeField>(kind));
  }

  /// Append a value that will be stored in a memory slot and not inline within
  /// the bytecode.
  template <typename T>
  std::enable_if_t<llvm::is_detected<has_pointer_traits, T>::value ||
                   std::is_pointer<T>::value>
  append(T value) {
    bytecode.push_back(generator.getMemIndex(value));
  }

  /// Append a range of values.
  template <typename T, typename IteratorT = llvm::detail::IterOfRange<T>>
  std::enable_if_t<!llvm::is_detected<has_pointer_traits, T>::value>
  append(T range) {
    bytecode.push_back(llvm::size(range));
    for (auto it : range)
      append(it);
  }

  /// Append a variadic number of fields to the bytecode.
  template <typename FieldTy, typename Field2Ty, typename... FieldTys>
  void append(FieldTy field, Field2Ty field2, FieldTys... fields) {
    append(field);
    append(field2, fields...);
  }

  /// Appends a value as a pointer, stored inline within the bytecode.
  template <typename T>
  std::enable_if_t<llvm::is_detected<has_pointer_traits, T>::value>
  appendInline(T value) {
    constexpr size_t numParts = sizeof(const void *) / sizeof(ByteCodeField);
    const void *pointer = value.getAsOpaquePointer();
    ByteCodeField fieldParts[numParts];
    std::memcpy(fieldParts, &pointer, sizeof(const void *));
    bytecode.append(fieldParts, fieldParts + numParts);
  }

  /// Successor references in the bytecode that have yet to be resolved.
  DenseMap<Block *, SmallVector<unsigned, 4>> unresolvedSuccessorRefs;

  /// The underlying bytecode buffer.
  SmallVectorImpl<ByteCodeField> &bytecode;

  /// The main generator producing PDL.
  Generator &generator;
};

/// This class represents a live range of PDL Interpreter values, containing
/// information about when values are live within a match/rewrite.
struct ByteCodeLiveRange {
  using Set = llvm::IntervalMap<uint64_t, char, 16>;
  using Allocator = Set::Allocator;

  ByteCodeLiveRange(Allocator &alloc) : liveness(new Set(alloc)) {}

  /// Union this live range with the one provided.
  void unionWith(const ByteCodeLiveRange &rhs) {
    for (auto it = rhs.liveness->begin(), e = rhs.liveness->end(); it != e;
         ++it)
      liveness->insert(it.start(), it.stop(), /*dummyValue*/ 0);
  }

  /// Returns true if this range overlaps with the one provided.
  bool overlaps(const ByteCodeLiveRange &rhs) const {
    return llvm::IntervalMapOverlaps<Set, Set>(*liveness, *rhs.liveness)
        .valid();
  }

  /// A map representing the ranges of the match/rewrite that a value is live in
  /// the interpreter.
  ///
  /// We use std::unique_ptr here, because IntervalMap does not provide a
  /// correct copy or move constructor. We can eliminate the pointer once
  /// https://reviews.llvm.org/D113240 lands.
  std::unique_ptr<llvm::IntervalMap<uint64_t, char, 16>> liveness;

  /// The operation range storage index for this range.
  std::optional<unsigned> opRangeIndex;

  /// The type range storage index for this range.
  std::optional<unsigned> typeRangeIndex;

  /// The value range storage index for this range.
  std::optional<unsigned> valueRangeIndex;
};
} // namespace

void Generator::generate(ModuleOp module) {
  auto matcherFunc = module.lookupSymbol<pdl_interp::FuncOp>(
      pdl_interp::PDLInterpDialect::getMatcherFunctionName());
  ModuleOp rewriterModule = module.lookupSymbol<ModuleOp>(
      pdl_interp::PDLInterpDialect::getRewriterModuleName());
  assert(matcherFunc && rewriterModule && "invalid PDL Interpreter module");

  // Allocate memory indices for the results of operations within the matcher
  // and rewriters.
  allocateMemoryIndices(matcherFunc, rewriterModule);

  // Generate code for the rewriter functions.
  ByteCodeWriter rewriterByteCodeWriter(rewriterByteCode, *this);
  for (auto rewriterFunc : rewriterModule.getOps<pdl_interp::FuncOp>()) {
    rewriterToAddr.try_emplace(rewriterFunc.getName(), rewriterByteCode.size());
    for (Operation &op : rewriterFunc.getOps())
      generate(&op, rewriterByteCodeWriter);
  }
  assert(rewriterByteCodeWriter.unresolvedSuccessorRefs.empty() &&
         "unexpected branches in rewriter function");

  // Generate code for the matcher function.
  ByteCodeWriter matcherByteCodeWriter(matcherByteCode, *this);
  generate(&matcherFunc.getBody(), matcherByteCodeWriter);

  // Resolve successor references in the matcher.
  for (auto &it : matcherByteCodeWriter.unresolvedSuccessorRefs) {
    ByteCodeAddr addr = blockToAddr[it.first];
    for (unsigned offsetToFix : it.second)
      std::memcpy(&matcherByteCode[offsetToFix], &addr, sizeof(ByteCodeAddr));
  }
}

void Generator::allocateMemoryIndices(pdl_interp::FuncOp matcherFunc,
                                      ModuleOp rewriterModule) {
  // Rewriters use simplistic allocation scheme that simply assigns an index to
  // each result.
  for (auto rewriterFunc : rewriterModule.getOps<pdl_interp::FuncOp>()) {
    ByteCodeField index = 0, typeRangeIndex = 0, valueRangeIndex = 0;
    auto processRewriterValue = [&](Value val) {
      valueToMemIndex.try_emplace(val, index++);
      if (pdl::RangeType rangeType = dyn_cast<pdl::RangeType>(val.getType())) {
        Type elementTy = rangeType.getElementType();
        if (isa<pdl::TypeType>(elementTy))
          valueToRangeIndex.try_emplace(val, typeRangeIndex++);
        else if (isa<pdl::ValueType>(elementTy))
          valueToRangeIndex.try_emplace(val, valueRangeIndex++);
      }
    };

    for (BlockArgument arg : rewriterFunc.getArguments())
      processRewriterValue(arg);
    rewriterFunc.getBody().walk([&](Operation *op) {
      for (Value result : op->getResults())
        processRewriterValue(result);
    });
    if (index > maxValueMemoryIndex)
      maxValueMemoryIndex = index;
    if (typeRangeIndex > maxTypeRangeMemoryIndex)
      maxTypeRangeMemoryIndex = typeRangeIndex;
    if (valueRangeIndex > maxValueRangeMemoryIndex)
      maxValueRangeMemoryIndex = valueRangeIndex;
  }

  // The matcher function uses a more sophisticated numbering that tries to
  // minimize the number of memory indices assigned. This is done by determining
  // a live range of the values within the matcher, then the allocation is just
  // finding the minimal number of overlapping live ranges. This is essentially
  // a simplified form of register allocation where we don't necessarily have a
  // limited number of registers, but we still want to minimize the number used.
  DenseMap<Operation *, unsigned> opToFirstIndex;
  DenseMap<Operation *, unsigned> opToLastIndex;

  // A custom walk that marks the first and the last index of each operation.
  // The entry marks the beginning of the liveness range for this operation,
  // followed by nested operations, followed by the end of the liveness range.
  unsigned index = 0;
  llvm::unique_function<void(Operation *)> walk = [&](Operation *op) {
    opToFirstIndex.try_emplace(op, index++);
    for (Region &region : op->getRegions())
      for (Block &block : region.getBlocks())
        for (Operation &nested : block)
          walk(&nested);
    opToLastIndex.try_emplace(op, index++);
  };
  walk(matcherFunc);

  // Liveness info for each of the defs within the matcher.
  ByteCodeLiveRange::Allocator allocator;
  DenseMap<Value, ByteCodeLiveRange> valueDefRanges;

  // Assign the root operation being matched to slot 0.
  BlockArgument rootOpArg = matcherFunc.getArgument(0);
  valueToMemIndex[rootOpArg] = 0;

  // Walk each of the blocks, computing the def interval that the value is used.
  Liveness matcherLiveness(matcherFunc);
  matcherFunc->walk([&](Block *block) {
    const LivenessBlockInfo *info = matcherLiveness.getLiveness(block);
    assert(info && "expected liveness info for block");
    auto processValue = [&](Value value, Operation *firstUseOrDef) {
      // We don't need to process the root op argument, this value is always
      // assigned to the first memory slot.
      if (value == rootOpArg)
        return;

      // Set indices for the range of this block that the value is used.
      auto defRangeIt = valueDefRanges.try_emplace(value, allocator).first;
      defRangeIt->second.liveness->insert(
          opToFirstIndex[firstUseOrDef],
          opToLastIndex[info->getEndOperation(value, firstUseOrDef)],
          /*dummyValue*/ 0);

      // Check to see if this value is a range type.
      if (auto rangeTy = dyn_cast<pdl::RangeType>(value.getType())) {
        Type eleType = rangeTy.getElementType();
        if (isa<pdl::OperationType>(eleType))
          defRangeIt->second.opRangeIndex = 0;
        else if (isa<pdl::TypeType>(eleType))
          defRangeIt->second.typeRangeIndex = 0;
        else if (isa<pdl::ValueType>(eleType))
          defRangeIt->second.valueRangeIndex = 0;
      }
    };

    // Process the live-ins of this block.
    for (Value liveIn : info->in()) {
      // Only process the value if it has been defined in the current region.
      // Other values that span across pdl_interp.foreach will be added higher
      // up. This ensures that the we keep them alive for the entire duration
      // of the loop.
      if (liveIn.getParentRegion() == block->getParent())
        processValue(liveIn, &block->front());
    }

    // Process the block arguments for the entry block (those are not live-in).
    if (block->isEntryBlock()) {
      for (Value argument : block->getArguments())
        processValue(argument, &block->front());
    }

    // Process any new defs within this block.
    for (Operation &op : *block)
      for (Value result : op.getResults())
        processValue(result, &op);
  });

  // Greedily allocate memory slots using the computed def live ranges.
  std::vector<ByteCodeLiveRange> allocatedIndices;

  // The number of memory indices currently allocated (and its next value).
  // Recall that the root gets allocated memory index 0.
  ByteCodeField numIndices = 1;

  // The number of memory ranges of various types (and their next values).
  ByteCodeField numOpRanges = 0, numTypeRanges = 0, numValueRanges = 0;

  for (auto &defIt : valueDefRanges) {
    ByteCodeField &memIndex = valueToMemIndex[defIt.first];
    ByteCodeLiveRange &defRange = defIt.second;

    // Try to allocate to an existing index.
    for (const auto &existingIndexIt : llvm::enumerate(allocatedIndices)) {
      ByteCodeLiveRange &existingRange = existingIndexIt.value();
      if (!defRange.overlaps(existingRange)) {
        existingRange.unionWith(defRange);
        memIndex = existingIndexIt.index() + 1;

        if (defRange.opRangeIndex) {
          if (!existingRange.opRangeIndex)
            existingRange.opRangeIndex = numOpRanges++;
          valueToRangeIndex[defIt.first] = *existingRange.opRangeIndex;
        } else if (defRange.typeRangeIndex) {
          if (!existingRange.typeRangeIndex)
            existingRange.typeRangeIndex = numTypeRanges++;
          valueToRangeIndex[defIt.first] = *existingRange.typeRangeIndex;
        } else if (defRange.valueRangeIndex) {
          if (!existingRange.valueRangeIndex)
            existingRange.valueRangeIndex = numValueRanges++;
          valueToRangeIndex[defIt.first] = *existingRange.valueRangeIndex;
        }
        break;
      }
    }

    // If no existing index could be used, add a new one.
    if (memIndex == 0) {
      allocatedIndices.emplace_back(allocator);
      ByteCodeLiveRange &newRange = allocatedIndices.back();
      newRange.unionWith(defRange);

      // Allocate an index for op/type/value ranges.
      if (defRange.opRangeIndex) {
        newRange.opRangeIndex = numOpRanges;
        valueToRangeIndex[defIt.first] = numOpRanges++;
      } else if (defRange.typeRangeIndex) {
        newRange.typeRangeIndex = numTypeRanges;
        valueToRangeIndex[defIt.first] = numTypeRanges++;
      } else if (defRange.valueRangeIndex) {
        newRange.valueRangeIndex = numValueRanges;
        valueToRangeIndex[defIt.first] = numValueRanges++;
      }

      memIndex = allocatedIndices.size();
      ++numIndices;
    }
  }

  // Print the index usage and ensure that we did not run out of index space.
  LDBG() << "Allocated " << allocatedIndices.size() << " indices "
         << "(down from initial " << valueDefRanges.size() << ").";
  assert(allocatedIndices.size() <= std::numeric_limits<ByteCodeField>::max() &&
         "Ran out of memory for allocated indices");

  // Update the max number of indices.
  if (numIndices > maxValueMemoryIndex)
    maxValueMemoryIndex = numIndices;
  if (numOpRanges > maxOpRangeMemoryIndex)
    maxOpRangeMemoryIndex = numOpRanges;
  if (numTypeRanges > maxTypeRangeMemoryIndex)
    maxTypeRangeMemoryIndex = numTypeRanges;
  if (numValueRanges > maxValueRangeMemoryIndex)
    maxValueRangeMemoryIndex = numValueRanges;
}

void Generator::generate(Region *region, ByteCodeWriter &writer) {
  llvm::ReversePostOrderTraversal<Region *> rpot(region);
  for (Block *block : rpot) {
    // Keep track of where this block begins within the matcher function.
    blockToAddr.try_emplace(block, matcherByteCode.size());
    for (Operation &op : *block)
      generate(&op, writer);
  }
}

void Generator::generate(Operation *op, ByteCodeWriter &writer) {
  LDBG() << "Generating bytecode for operation: " << op->getName();
  LLVM_DEBUG({
    // The following list must contain all the operations that do not
    // produce any bytecode.
    if (!isa<pdl_interp::CreateAttributeOp, pdl_interp::CreateTypeOp>(op))
      writer.appendInline(op->getLoc());
  });
  TypeSwitch<Operation *>(op)
      .Case<pdl_interp::ApplyConstraintOp, pdl_interp::ApplyRewriteOp,
            pdl_interp::AreEqualOp, pdl_interp::BranchOp,
            pdl_interp::CheckAttributeOp, pdl_interp::CheckOperandCountOp,
            pdl_interp::CheckOperationNameOp, pdl_interp::CheckResultCountOp,
            pdl_interp::CheckTypeOp, pdl_interp::CheckTypesOp,
            pdl_interp::ContinueOp, pdl_interp::CreateAttributeOp,
            pdl_interp::CreateOperationOp, pdl_interp::CreateRangeOp,
            pdl_interp::CreateTypeOp, pdl_interp::CreateTypesOp,
            pdl_interp::EraseOp, pdl_interp::ExtractOp, pdl_interp::FinalizeOp,
            pdl_interp::ForEachOp, pdl_interp::GetAttributeOp,
            pdl_interp::GetAttributeTypeOp, pdl_interp::GetDefiningOpOp,
            pdl_interp::GetOperandOp, pdl_interp::GetOperandsOp,
            pdl_interp::GetResultOp, pdl_interp::GetResultsOp,
            pdl_interp::GetUsersOp, pdl_interp::GetValueTypeOp,
            pdl_interp::IsNotNullOp, pdl_interp::RecordMatchOp,
            pdl_interp::ReplaceOp, pdl_interp::SwitchAttributeOp,
            pdl_interp::SwitchTypeOp, pdl_interp::SwitchTypesOp,
            pdl_interp::SwitchOperandCountOp, pdl_interp::SwitchOperationNameOp,
            pdl_interp::SwitchResultCountOp>(
          [&](auto interpOp) { this->generate(interpOp, writer); })
      .DefaultUnreachable("unknown `pdl_interp` operation");
}

void Generator::generate(pdl_interp::ApplyConstraintOp op,
                         ByteCodeWriter &writer) {
  // Constraints that should return a value have to be registered as rewrites.
  // If a constraint and a rewrite of similar name are registered the
  // constraint takes precedence
  writer.append(OpCode::ApplyConstraint, constraintToMemIndex[op.getName()]);
  writer.appendPDLValueList(op.getArgs());
  writer.append(ByteCodeField(op.getIsNegated()));
  ResultRange results = op.getResults();
  writer.append(ByteCodeField(results.size()));
  for (Value result : results) {
    // We record the expected kind of the result, so that we can provide extra
    // verification of the native rewrite function and handle the failure case
    // of constraints accordingly.
    writer.appendPDLValueKind(result);

    // Range results also need to append the range storage index.
    if (isa<pdl::RangeType>(result.getType()))
      writer.append(getRangeStorageIndex(result));
    writer.append(result);
  }
  writer.append(op.getSuccessors());
}
void Generator::generate(pdl_interp::ApplyRewriteOp op,
                         ByteCodeWriter &writer) {
  assert(externalRewriterToMemIndex.count(op.getName()) &&
         "expected index for rewrite function");
  writer.append(OpCode::ApplyRewrite, externalRewriterToMemIndex[op.getName()]);
  writer.appendPDLValueList(op.getArgs());

  ResultRange results = op.getResults();
  writer.append(ByteCodeField(results.size()));
  for (Value result : results) {
    // We record the expected kind of the result, so that we
    // can provide extra verification of the native rewrite function.
    writer.appendPDLValueKind(result);

    // Range results also need to append the range storage index.
    if (isa<pdl::RangeType>(result.getType()))
      writer.append(getRangeStorageIndex(result));
    writer.append(result);
  }
}
void Generator::generate(pdl_interp::AreEqualOp op, ByteCodeWriter &writer) {
  Value lhs = op.getLhs();
  if (isa<pdl::RangeType>(lhs.getType())) {
    writer.append(OpCode::AreRangesEqual);
    writer.appendPDLValueKind(lhs);
    writer.append(op.getLhs(), op.getRhs(), op.getSuccessors());
    return;
  }

  writer.append(OpCode::AreEqual, lhs, op.getRhs(), op.getSuccessors());
}
void Generator::generate(pdl_interp::BranchOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::Branch, SuccessorRange(op.getOperation()));
}
void Generator::generate(pdl_interp::CheckAttributeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::AreEqual, op.getAttribute(), op.getConstantValue(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckOperandCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CheckOperandCount, op.getInputOp(), op.getCount(),
                static_cast<ByteCodeField>(op.getCompareAtLeast()),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckOperationNameOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CheckOperationName, op.getInputOp(),
                OperationName(op.getName(), ctx), op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckResultCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CheckResultCount, op.getInputOp(), op.getCount(),
                static_cast<ByteCodeField>(op.getCompareAtLeast()),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckTypeOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::AreEqual, op.getValue(), op.getType(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::CheckTypesOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::CheckTypes, op.getValue(), op.getTypes(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::ContinueOp op, ByteCodeWriter &writer) {
  assert(curLoopLevel > 0 && "encountered pdl_interp.continue at top level");
  writer.append(OpCode::Continue, ByteCodeField(curLoopLevel - 1));
}
void Generator::generate(pdl_interp::CreateAttributeOp op,
                         ByteCodeWriter &writer) {
  // Simply repoint the memory index of the result to the constant.
  getMemIndex(op.getAttribute()) = getMemIndex(op.getValue());
}
void Generator::generate(pdl_interp::CreateOperationOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::CreateOperation, op.getResultOp(),
                OperationName(op.getName(), ctx));
  writer.appendPDLValueList(op.getInputOperands());

  // Add the attributes.
  OperandRange attributes = op.getInputAttributes();
  writer.append(static_cast<ByteCodeField>(attributes.size()));
  for (auto it : llvm::zip(op.getInputAttributeNames(), attributes))
    writer.append(std::get<0>(it), std::get<1>(it));

  // Add the result types. If the operation has inferred results, we use a
  // marker "size" value. Otherwise, we add the list of explicit result types.
  if (op.getInferredResultTypes())
    writer.append(kInferTypesMarker);
  else
    writer.appendPDLValueList(op.getInputResultTypes());
}
void Generator::generate(pdl_interp::CreateRangeOp op, ByteCodeWriter &writer) {
  // Append the correct opcode for the range type.
  TypeSwitch<Type>(op.getType().getElementType())
      .Case(
          [&](pdl::TypeType) { writer.append(OpCode::CreateDynamicTypeRange); })
      .Case([&](pdl::ValueType) {
        writer.append(OpCode::CreateDynamicValueRange);
      });

  writer.append(op.getResult(), getRangeStorageIndex(op.getResult()));
  writer.appendPDLValueList(op->getOperands());
}
void Generator::generate(pdl_interp::CreateTypeOp op, ByteCodeWriter &writer) {
  // Simply repoint the memory index of the result to the constant.
  getMemIndex(op.getResult()) = getMemIndex(op.getValue());
}
void Generator::generate(pdl_interp::CreateTypesOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::CreateConstantTypeRange, op.getResult(),
                getRangeStorageIndex(op.getResult()), op.getValue());
}
void Generator::generate(pdl_interp::EraseOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::EraseOp, op.getInputOp());
}
void Generator::generate(pdl_interp::ExtractOp op, ByteCodeWriter &writer) {
  OpCode opCode =
      TypeSwitch<Type, OpCode>(op.getResult().getType())
          .Case([](pdl::OperationType) { return OpCode::ExtractOp; })
          .Case([](pdl::ValueType) { return OpCode::ExtractValue; })
          .Case([](pdl::TypeType) { return OpCode::ExtractType; })
          .DefaultUnreachable("unsupported element type");
  writer.append(opCode, op.getRange(), op.getIndex(), op.getResult());
}
void Generator::generate(pdl_interp::FinalizeOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::Finalize);
}
void Generator::generate(pdl_interp::ForEachOp op, ByteCodeWriter &writer) {
  BlockArgument arg = op.getLoopVariable();
  writer.append(OpCode::ForEach, getRangeStorageIndex(op.getValues()), arg);
  writer.appendPDLValueKind(arg.getType());
  writer.append(curLoopLevel, op.getSuccessor());
  ++curLoopLevel;
  if (curLoopLevel > maxLoopLevel)
    maxLoopLevel = curLoopLevel;
  generate(&op.getRegion(), writer);
  --curLoopLevel;
}
void Generator::generate(pdl_interp::GetAttributeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetAttribute, op.getAttribute(), op.getInputOp(),
                op.getNameAttr());
}
void Generator::generate(pdl_interp::GetAttributeTypeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetAttributeType, op.getResult(), op.getValue());
}
void Generator::generate(pdl_interp::GetDefiningOpOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::GetDefiningOp, op.getInputOp());
  writer.appendPDLValue(op.getValue());
}
void Generator::generate(pdl_interp::GetOperandOp op, ByteCodeWriter &writer) {
  uint32_t index = op.getIndex();
  if (index < 4)
    writer.append(static_cast<OpCode>(OpCode::GetOperand0 + index));
  else
    writer.append(OpCode::GetOperandN, index);
  writer.append(op.getInputOp(), op.getValue());
}
void Generator::generate(pdl_interp::GetOperandsOp op, ByteCodeWriter &writer) {
  Value result = op.getValue();
  std::optional<uint32_t> index = op.getIndex();
  writer.append(OpCode::GetOperands,
                index.value_or(std::numeric_limits<uint32_t>::max()),
                op.getInputOp());
  if (isa<pdl::RangeType>(result.getType()))
    writer.append(getRangeStorageIndex(result));
  else
    writer.append(std::numeric_limits<ByteCodeField>::max());
  writer.append(result);
}
void Generator::generate(pdl_interp::GetResultOp op, ByteCodeWriter &writer) {
  uint32_t index = op.getIndex();
  if (index < 4)
    writer.append(static_cast<OpCode>(OpCode::GetResult0 + index));
  else
    writer.append(OpCode::GetResultN, index);
  writer.append(op.getInputOp(), op.getValue());
}
void Generator::generate(pdl_interp::GetResultsOp op, ByteCodeWriter &writer) {
  Value result = op.getValue();
  std::optional<uint32_t> index = op.getIndex();
  writer.append(OpCode::GetResults,
                index.value_or(std::numeric_limits<uint32_t>::max()),
                op.getInputOp());
  if (isa<pdl::RangeType>(result.getType()))
    writer.append(getRangeStorageIndex(result));
  else
    writer.append(std::numeric_limits<ByteCodeField>::max());
  writer.append(result);
}
void Generator::generate(pdl_interp::GetUsersOp op, ByteCodeWriter &writer) {
  Value operations = op.getOperations();
  ByteCodeField rangeIndex = getRangeStorageIndex(operations);
  writer.append(OpCode::GetUsers, operations, rangeIndex);
  writer.appendPDLValue(op.getValue());
}
void Generator::generate(pdl_interp::GetValueTypeOp op,
                         ByteCodeWriter &writer) {
  if (isa<pdl::RangeType>(op.getType())) {
    Value result = op.getResult();
    writer.append(OpCode::GetValueRangeTypes, result,
                  getRangeStorageIndex(result), op.getValue());
  } else {
    writer.append(OpCode::GetValueType, op.getResult(), op.getValue());
  }
}
void Generator::generate(pdl_interp::IsNotNullOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::IsNotNull, op.getValue(), op.getSuccessors());
}
void Generator::generate(pdl_interp::RecordMatchOp op, ByteCodeWriter &writer) {
  ByteCodeField patternIndex = patterns.size();
  patterns.emplace_back(PDLByteCodePattern::create(
      op, configMap.lookup(op),
      rewriterToAddr[op.getRewriter().getLeafReference().getValue()]));
  writer.append(OpCode::RecordMatch, patternIndex,
                SuccessorRange(op.getOperation()), op.getMatchedOps());
  writer.appendPDLValueList(op.getInputs());
}
void Generator::generate(pdl_interp::ReplaceOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::ReplaceOp, op.getInputOp());
  writer.appendPDLValueList(op.getReplValues());
}
void Generator::generate(pdl_interp::SwitchAttributeOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchAttribute, op.getAttribute(),
                op.getCaseValuesAttr(), op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchOperandCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchOperandCount, op.getInputOp(),
                op.getCaseValuesAttr(), op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchOperationNameOp op,
                         ByteCodeWriter &writer) {
  auto cases = llvm::map_range(op.getCaseValuesAttr(), [&](Attribute attr) {
    return OperationName(cast<StringAttr>(attr).getValue(), ctx);
  });
  writer.append(OpCode::SwitchOperationName, op.getInputOp(), cases,
                op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchResultCountOp op,
                         ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchResultCount, op.getInputOp(),
                op.getCaseValuesAttr(), op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchTypeOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchType, op.getValue(), op.getCaseValuesAttr(),
                op.getSuccessors());
}
void Generator::generate(pdl_interp::SwitchTypesOp op, ByteCodeWriter &writer) {
  writer.append(OpCode::SwitchTypes, op.getValue(), op.getCaseValuesAttr(),
                op.getSuccessors());
}

//===----------------------------------------------------------------------===//
// PDLByteCode
//===----------------------------------------------------------------------===//

PDLByteCode::PDLByteCode(
    ModuleOp module, SmallVector<std::unique_ptr<PDLPatternConfigSet>> configs,
    const DenseMap<Operation *, PDLPatternConfigSet *> &configMap,
    llvm::StringMap<PDLConstraintFunction> constraintFns,
    llvm::StringMap<PDLRewriteFunction> rewriteFns)
    : configs(std::move(configs)) {
  Generator generator(module.getContext(), uniquedData, matcherByteCode,
                      rewriterByteCode, patterns, maxValueMemoryIndex,
                      maxOpRangeCount, maxTypeRangeCount, maxValueRangeCount,
                      maxLoopLevel, constraintFns, rewriteFns, configMap);
  generator.generate(module);

  // Initialize the external functions.
  for (auto &it : constraintFns)
    constraintFunctions.push_back(std::move(it.second));
  for (auto &it : rewriteFns)
    rewriteFunctions.push_back(std::move(it.second));
}

/// Initialize the given state such that it can be used to execute the current
/// bytecode.
void PDLByteCode::initializeMutableState(PDLByteCodeMutableState &state) const {
  state.memory.resize(maxValueMemoryIndex, nullptr);
  state.opRangeMemory.resize(maxOpRangeCount);
  state.typeRangeMemory.resize(maxTypeRangeCount, TypeRange());
  state.valueRangeMemory.resize(maxValueRangeCount, ValueRange());
  state.loopIndex.resize(maxLoopLevel, 0);
  state.currentPatternBenefits.reserve(patterns.size());
  for (const PDLByteCodePattern &pattern : patterns)
    state.currentPatternBenefits.push_back(pattern.getBenefit());
}

//===----------------------------------------------------------------------===//
// ByteCode Execution — Tail-Calling Dispatch
//===----------------------------------------------------------------------===//

namespace {

/// This class is an instantiation of the PDLResultList that provides access to
/// the returned results. This API is not on `PDLResultList` to avoid
/// overexposing access to information specific solely to the ByteCode.
class ByteCodeRewriteResultList : public PDLResultList {
public:
  ByteCodeRewriteResultList(unsigned maxNumResults)
      : PDLResultList(maxNumResults) {}

  /// Return the list of PDL results.
  MutableArrayRef<PDLValue> getResults() { return results; }

  /// Return the type ranges allocated by this list.
  MutableArrayRef<std::vector<Type>> getAllocatedTypeRanges() {
    return allocatedTypeRanges;
  }

  /// Return the value ranges allocated by this list.
  MutableArrayRef<std::vector<Value>> getAllocatedValueRanges() {
    return allocatedValueRanges;
  }
};

//===----------------------------------------------------------------------===//
// Forward declarations for tail-calling dispatch
//===----------------------------------------------------------------------===//

class ByteCodeExecutor;

/// Handler function type for tail-calling dispatch. Every opcode handler has
/// this signature, enabling [[clang::musttail]] calls between them.
using Handler = LogicalResult (*)(ByteCodeExecutor &);

// Forward declare all opcode handlers. No `static` needed — the enclosing
// anonymous namespace already gives them internal linkage.
LogicalResult handleApplyConstraint(ByteCodeExecutor &);
LogicalResult handleApplyRewrite(ByteCodeExecutor &);
LogicalResult handleAreEqual(ByteCodeExecutor &);
LogicalResult handleAreRangesEqual(ByteCodeExecutor &);
LogicalResult handleBranch(ByteCodeExecutor &);
LogicalResult handleCheckOperandCount(ByteCodeExecutor &);
LogicalResult handleCheckOperationName(ByteCodeExecutor &);
LogicalResult handleCheckResultCount(ByteCodeExecutor &);
LogicalResult handleCheckTypes(ByteCodeExecutor &);
LogicalResult handleContinue(ByteCodeExecutor &);
LogicalResult handleCreateConstantTypeRange(ByteCodeExecutor &);
LogicalResult handleCreateOperation(ByteCodeExecutor &);
LogicalResult handleCreateDynamicTypeRange(ByteCodeExecutor &);
LogicalResult handleCreateDynamicValueRange(ByteCodeExecutor &);
LogicalResult handleEraseOp(ByteCodeExecutor &);
LogicalResult handleExtractOp(ByteCodeExecutor &);
LogicalResult handleExtractType(ByteCodeExecutor &);
LogicalResult handleExtractValue(ByteCodeExecutor &);
LogicalResult handleFinalize(ByteCodeExecutor &);
LogicalResult handleForEach(ByteCodeExecutor &);
LogicalResult handleGetAttribute(ByteCodeExecutor &);
LogicalResult handleGetAttributeType(ByteCodeExecutor &);
LogicalResult handleGetDefiningOp(ByteCodeExecutor &);
LogicalResult handleGetOperand0(ByteCodeExecutor &);
LogicalResult handleGetOperand1(ByteCodeExecutor &);
LogicalResult handleGetOperand2(ByteCodeExecutor &);
LogicalResult handleGetOperand3(ByteCodeExecutor &);
LogicalResult handleGetOperandN(ByteCodeExecutor &);
LogicalResult handleGetOperands(ByteCodeExecutor &);
LogicalResult handleGetResult0(ByteCodeExecutor &);
LogicalResult handleGetResult1(ByteCodeExecutor &);
LogicalResult handleGetResult2(ByteCodeExecutor &);
LogicalResult handleGetResult3(ByteCodeExecutor &);
LogicalResult handleGetResultN(ByteCodeExecutor &);
LogicalResult handleGetResults(ByteCodeExecutor &);
LogicalResult handleGetUsers(ByteCodeExecutor &);
LogicalResult handleGetValueType(ByteCodeExecutor &);
LogicalResult handleGetValueRangeTypes(ByteCodeExecutor &);
LogicalResult handleIsNotNull(ByteCodeExecutor &);
LogicalResult handleRecordMatch(ByteCodeExecutor &);
LogicalResult handleReplaceOp(ByteCodeExecutor &);
LogicalResult handleSwitchAttribute(ByteCodeExecutor &);
LogicalResult handleSwitchOperandCount(ByteCodeExecutor &);
LogicalResult handleSwitchOperationName(ByteCodeExecutor &);
LogicalResult handleSwitchResultCount(ByteCodeExecutor &);
LogicalResult handleSwitchType(ByteCodeExecutor &);
LogicalResult handleSwitchTypes(ByteCodeExecutor &);

/// Dispatch table indexed by OpCode. Order must exactly match the OpCode enum.
const Handler dispatchTable[] = {
    /* ApplyConstraint */        handleApplyConstraint,
    /* ApplyRewrite */           handleApplyRewrite,
    /* AreEqual */               handleAreEqual,
    /* AreRangesEqual */         handleAreRangesEqual,
    /* Branch */                 handleBranch,
    /* CheckOperandCount */      handleCheckOperandCount,
    /* CheckOperationName */     handleCheckOperationName,
    /* CheckResultCount */       handleCheckResultCount,
    /* CheckTypes */             handleCheckTypes,
    /* Continue */               handleContinue,
    /* CreateConstantTypeRange */handleCreateConstantTypeRange,
    /* CreateOperation */        handleCreateOperation,
    /* CreateDynamicTypeRange */ handleCreateDynamicTypeRange,
    /* CreateDynamicValueRange */handleCreateDynamicValueRange,
    /* EraseOp */                handleEraseOp,
    /* ExtractOp */              handleExtractOp,
    /* ExtractType */            handleExtractType,
    /* ExtractValue */           handleExtractValue,
    /* Finalize */               handleFinalize,
    /* ForEach */                handleForEach,
    /* GetAttribute */           handleGetAttribute,
    /* GetAttributeType */       handleGetAttributeType,
    /* GetDefiningOp */          handleGetDefiningOp,
    /* GetOperand0 */            handleGetOperand0,
    /* GetOperand1 */            handleGetOperand1,
    /* GetOperand2 */            handleGetOperand2,
    /* GetOperand3 */            handleGetOperand3,
    /* GetOperandN */            handleGetOperandN,
    /* GetOperands */            handleGetOperands,
    /* GetResult0 */             handleGetResult0,
    /* GetResult1 */             handleGetResult1,
    /* GetResult2 */             handleGetResult2,
    /* GetResult3 */             handleGetResult3,
    /* GetResultN */             handleGetResultN,
    /* GetResults */             handleGetResults,
    /* GetUsers */               handleGetUsers,
    /* GetValueType */           handleGetValueType,
    /* GetValueRangeTypes */     handleGetValueRangeTypes,
    /* IsNotNull */              handleIsNotNull,
    /* RecordMatch */            handleRecordMatch,
    /* ReplaceOp */              handleReplaceOp,
    /* SwitchAttribute */        handleSwitchAttribute,
    /* SwitchOperandCount */     handleSwitchOperandCount,
    /* SwitchOperationName */    handleSwitchOperationName,
    /* SwitchResultCount */      handleSwitchResultCount,
    /* SwitchType */             handleSwitchType,
    /* SwitchTypes */            handleSwitchTypes,
};

/// Dispatch macro: reads the next opcode and tail-calls its handler.
/// In debug builds, also reads and prints the inline source Location.
/// No non-trivially-destructible temporaries are live at the musttail call.
#ifndef NDEBUG
#define DISPATCH(exec)                                                         \
  do {                                                                         \
    LDBG() << "";                                                              \
    LDBG() << (exec).readInline<Location>();                                   \
    OpCode _op = static_cast<OpCode>((exec).read());                           \
    [[clang::musttail]] return dispatchTable[_op]((exec));                      \
  } while (0)
#else
#define DISPATCH(exec)                                                         \
  do {                                                                         \
    OpCode _op = static_cast<OpCode>((exec).read());                           \
    [[clang::musttail]] return dispatchTable[_op]((exec));                      \
  } while (0)
#endif

//===----------------------------------------------------------------------===//
// ByteCodeExecutor
//===----------------------------------------------------------------------===//

/// This class provides support for executing a bytecode stream.
class ByteCodeExecutor {
public:
  ByteCodeExecutor(const ByteCodeField *curCodeIt,
                   MutableArrayRef<const void *> memory,
                   MutableArrayRef<std::vector<Operation *>> opRangeMemory,
                   MutableArrayRef<TypeRange> typeRangeMemory,
                   std::vector<std::vector<Type>> &allocatedTypeRangeMemory,
                   MutableArrayRef<ValueRange> valueRangeMemory,
                   std::vector<std::vector<Value>> &allocatedValueRangeMemory,
                   MutableArrayRef<unsigned> loopIndex,
                   ArrayRef<const void *> uniquedMemory,
                   ArrayRef<ByteCodeField> code,
                   ArrayRef<PatternBenefit> currentPatternBenefits,
                   ArrayRef<PDLByteCodePattern> patterns,
                   ArrayRef<PDLConstraintFunction> constraintFunctions,
                   ArrayRef<PDLRewriteFunction> rewriteFunctions)
      : curCodeIt(curCodeIt), memory(memory), opRangeMemory(opRangeMemory),
        typeRangeMemory(typeRangeMemory),
        allocatedTypeRangeMemory(allocatedTypeRangeMemory),
        valueRangeMemory(valueRangeMemory),
        allocatedValueRangeMemory(allocatedValueRangeMemory),
        loopIndex(loopIndex), uniquedMemory(uniquedMemory), code(code),
        currentPatternBenefits(currentPatternBenefits), patterns(patterns),
        constraintFunctions(constraintFunctions),
        rewriteFunctions(rewriteFunctions) {}

  /// Entry point: stores execution context and dispatches the first opcode.
  LogicalResult
  execute(PatternRewriter &rewriter,
          SmallVectorImpl<PDLByteCode::MatchResult> *matches = nullptr,
          std::optional<Location> mainRewriteLoc = {});

  //===--------------------------------------------------------------------===//
  // Opcode implementation methods — called by the handler functions.
  // Public so that the free-standing handler functions can call them.
  //===--------------------------------------------------------------------===//

  void executeApplyConstraint();
  LogicalResult executeApplyRewrite();
  void executeAreEqual();
  void executeAreRangesEqual();
  void executeBranch();
  void executeCheckOperandCount();
  void executeCheckOperationName();
  void executeCheckResultCount();
  void executeCheckTypes();
  void executeContinue();
  void executeCreateConstantTypeRange();
  void executeCreateOperation();
  template <typename T>
  void executeDynamicCreateRange(StringRef type);
  void executeEraseOp();
  template <typename T, typename Range, PDLValue::Kind kind>
  void executeExtract();
  void executeFinalize();
  void executeForEach();
  void executeGetAttribute();
  void executeGetAttributeType();
  void executeGetDefiningOp();
  void executeGetOperand(unsigned index);
  void executeGetOperands();
  void executeGetResult(unsigned index);
  void executeGetResults();
  void executeGetUsers();
  void executeGetValueType();
  void executeGetValueRangeTypes();
  void executeIsNotNull();
  void executeRecordMatch();
  void executeReplaceOp();
  void executeSwitchAttribute();
  void executeSwitchOperandCount();
  void executeSwitchOperationName();
  void executeSwitchResultCount();
  void executeSwitchType();
  void executeSwitchTypes();
  void processNativeFunResults(ByteCodeRewriteResultList &results,
                               unsigned numResults,
                               LogicalResult &rewriteResult);

  //===--------------------------------------------------------------------===//
  // Bytecode reading and control flow helpers
  //===--------------------------------------------------------------------===//

  /// Pushes a code iterator to the stack.
  void pushCodeIt(const ByteCodeField *it) { resumeCodeIt.push_back(it); }

  /// Pops a code iterator from the stack.
  void popCodeIt() {
    assert(!resumeCodeIt.empty() && "attempt to pop code off empty stack");
    curCodeIt = resumeCodeIt.pop_back_val();
  }

  /// Return the bytecode iterator at the start of the current op code.
  const ByteCodeField *getPrevCodeIt() const {
    LLVM_DEBUG({
      // Account for the op code and the Location stored inline.
      return curCodeIt - 1 - sizeof(const void *) / sizeof(ByteCodeField);
    });
    // Account for the op code only.
    return curCodeIt - 1;
  }

  /// Read a value from the bytecode buffer, optionally skipping a certain
  /// number of prefix values.
  template <typename T = ByteCodeField>
  T read(size_t skipN = 0) {
    curCodeIt += skipN;
    return readImpl<T>();
  }
  ByteCodeField read(size_t skipN = 0) { return read<ByteCodeField>(skipN); }

  /// Read a list of values from the bytecode buffer.
  template <typename ValueT, typename T>
  void readList(SmallVectorImpl<T> &list) {
    list.clear();
    for (unsigned i = 0, e = read(); i != e; ++i)
      list.push_back(read<ValueT>());
  }

  /// Read a list of values, handling single elements or ranges.
  void readList(SmallVectorImpl<Type> &list) {
    for (unsigned i = 0, e = read(); i != e; ++i) {
      if (read<PDLValue::Kind>() == PDLValue::Kind::Type) {
        list.push_back(read<Type>());
      } else {
        TypeRange *values = read<TypeRange *>();
        list.append(values->begin(), values->end());
      }
    }
  }
  void readList(SmallVectorImpl<Value> &list) {
    for (unsigned i = 0, e = read(); i != e; ++i) {
      if (read<PDLValue::Kind>() == PDLValue::Kind::Value) {
        list.push_back(read<Value>());
      } else {
        ValueRange *values = read<ValueRange *>();
        list.append(values->begin(), values->end());
      }
    }
  }

  /// Read a value stored inline as a pointer.
  template <typename T>
  std::enable_if_t<llvm::is_detected<has_pointer_traits, T>::value, T>
  readInline() {
    const void *pointer;
    std::memcpy(&pointer, curCodeIt, sizeof(const void *));
    curCodeIt += sizeof(const void *) / sizeof(ByteCodeField);
    return T::getFromOpaquePointer(pointer);
  }

  void skip(size_t skipN) { curCodeIt += skipN; }

  /// Jump to a specific successor based on a predicate value.
  void selectJump(bool isTrue) { selectJump(size_t(isTrue ? 0 : 1)); }
  /// Jump to a specific successor based on a destination index.
  void selectJump(size_t destIndex) {
    curCodeIt = &code[read<ByteCodeAddr>(destIndex * 2)];
  }

  /// Handle a switch operation with the provided value and cases.
  template <typename T, typename RangeT, typename Comparator = std::equal_to<T>>
  void handleSwitch(const T &value, RangeT &&cases, Comparator cmp = {}) {
    LDBG() << "Switch operation:\n  * Value: " << value
           << "\n  * Cases: " << llvm::interleaved(cases);
    for (auto it = cases.begin(), e = cases.end(); it != e; ++it)
      if (cmp(*it, value))
        return selectJump(size_t((it - cases.begin()) + 1));
    selectJump(size_t(0));
  }

  /// Store a pointer to memory.
  void storeToMemory(unsigned index, const void *value) {
    memory[index] = value;
  }

  /// Store a value to memory as an opaque pointer.
  template <typename T>
  std::enable_if_t<llvm::is_detected<has_pointer_traits, T>::value>
  storeToMemory(unsigned index, T value) {
    memory[index] = value.getAsOpaquePointer();
  }

  /// Assign a range to memory.
  template <typename RangeT, typename T = llvm::detail::ValueOfRange<RangeT>>
  void assignRangeToMemory(RangeT &&range, unsigned memIndex,
                           unsigned rangeIndex) {
    auto assignRange = [&](auto &allocatedRangeMemory, auto &rangeMemory) {
      if (range.empty()) {
        rangeMemory[rangeIndex] = {};
      } else {
        allocatedRangeMemory.emplace_back(range.begin(), range.end());
        rangeMemory[rangeIndex] = allocatedRangeMemory.back();
      }
      memory[memIndex] = &rangeMemory[rangeIndex];
    };
    if constexpr (std::is_same_v<T, Type>) {
      return assignRange(allocatedTypeRangeMemory, typeRangeMemory);
    } else if constexpr (std::is_same_v<T, Value>) {
      return assignRange(allocatedValueRangeMemory, valueRangeMemory);
    } else {
      llvm_unreachable("unhandled range type");
    }
  }

  //===--------------------------------------------------------------------===//
  // State
  //===--------------------------------------------------------------------===//

  /// The current bytecode instruction pointer.
  const ByteCodeField *curCodeIt;

  /// The stack of bytecode positions at which to resume operation.
  SmallVector<const ByteCodeField *> resumeCodeIt;

  /// Execution context — stored by execute() before first dispatch.
  PatternRewriter *rewriter = nullptr;
  SmallVectorImpl<PDLByteCode::MatchResult> *matches = nullptr;
  std::optional<Location> mainRewriteLoc;

  /// The current execution memory.
  MutableArrayRef<const void *> memory;
  MutableArrayRef<std::vector<Operation *>> opRangeMemory;
  MutableArrayRef<TypeRange> typeRangeMemory;
  std::vector<std::vector<Type>> &allocatedTypeRangeMemory;
  MutableArrayRef<ValueRange> valueRangeMemory;
  std::vector<std::vector<Value>> &allocatedValueRangeMemory;

  /// The current loop indices.
  MutableArrayRef<unsigned> loopIndex;

  /// References to ByteCode data necessary for execution.
  ArrayRef<const void *> uniquedMemory;
  ArrayRef<ByteCodeField> code;
  ArrayRef<PatternBenefit> currentPatternBenefits;
  ArrayRef<PDLByteCodePattern> patterns;
  ArrayRef<PDLConstraintFunction> constraintFunctions;
  ArrayRef<PDLRewriteFunction> rewriteFunctions;

private:
  /// Internal implementation of reading data types from the bytecode stream.
  template <typename T>
  const void *readFromMemory() {
    size_t index = *curCodeIt++;
    if (llvm::is_one_of<T, Operation *, TypeRange *, ValueRange *,
                        Value>::value ||
        index < memory.size())
      return memory[index];
    return uniquedMemory[index - memory.size()];
  }
  template <typename T>
  std::enable_if_t<std::is_pointer<T>::value, T> readImpl() {
    return reinterpret_cast<T>(const_cast<void *>(readFromMemory<T>()));
  }
  template <typename T>
  std::enable_if_t<std::is_class<T>::value && !std::is_same<PDLValue, T>::value,
                   T>
  readImpl() {
    return T(T::getFromOpaquePointer(readFromMemory<T>()));
  }
  template <typename T>
  std::enable_if_t<std::is_same<PDLValue, T>::value, T> readImpl() {
    switch (read<PDLValue::Kind>()) {
    case PDLValue::Kind::Attribute:
      return read<Attribute>();
    case PDLValue::Kind::Operation:
      return read<Operation *>();
    case PDLValue::Kind::Type:
      return read<Type>();
    case PDLValue::Kind::Value:
      return read<Value>();
    case PDLValue::Kind::TypeRange:
      return read<TypeRange *>();
    case PDLValue::Kind::ValueRange:
      return read<ValueRange *>();
    }
    llvm_unreachable("unhandled PDLValue::Kind");
  }
  template <typename T>
  std::enable_if_t<std::is_same<T, ByteCodeAddr>::value, T> readImpl() {
    static_assert((sizeof(ByteCodeAddr) / sizeof(ByteCodeField)) == 2,
                  "unexpected ByteCode address size");
    ByteCodeAddr result;
    std::memcpy(&result, curCodeIt, sizeof(ByteCodeAddr));
    curCodeIt += 2;
    return result;
  }
  template <typename T>
  std::enable_if_t<std::is_same<T, ByteCodeField>::value, T> readImpl() {
    return *curCodeIt++;
  }
  template <typename T>
  std::enable_if_t<std::is_same<T, PDLValue::Kind>::value, T> readImpl() {
    return static_cast<PDLValue::Kind>(readImpl<ByteCodeField>());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ByteCodeExecutor entry point
//===----------------------------------------------------------------------===//

LogicalResult
ByteCodeExecutor::execute(PatternRewriter &rewriter,
                          SmallVectorImpl<PDLByteCode::MatchResult> *matches,
                          std::optional<Location> mainRewriteLoc) {
  // Store execution context as members so handlers can access them.
  this->rewriter = &rewriter;
  this->matches = matches;
  this->mainRewriteLoc = mainRewriteLoc;

  // Read the first instruction's debug location and opcode, then dispatch.
  LDBG() << readInline<Location>();
  OpCode opCode = static_cast<OpCode>(read());
  return dispatchTable[opCode](*this);
}

//===----------------------------------------------------------------------===//
// Opcode implementation methods
//===----------------------------------------------------------------------===//

void ByteCodeExecutor::executeApplyConstraint() {
  LDBG() << "Executing ApplyConstraint:";
  ByteCodeField fun_idx = read();
  SmallVector<PDLValue, 16> args;
  readList<PDLValue>(args);

  LDBG() << "  * Arguments: " << llvm::interleaved(args);

  ByteCodeField isNegated = read();
  LDBG() << "  * isNegated: " << isNegated;

  ByteCodeField numResults = read();
  const PDLRewriteFunction &constraintFn = constraintFunctions[fun_idx];
  ByteCodeRewriteResultList results(numResults);
  LogicalResult rewriteResult = constraintFn(*rewriter, results, args);
  [[maybe_unused]] ArrayRef<PDLValue> constraintResults = results.getResults();
  if (succeeded(rewriteResult)) {
    LDBG() << "  * Constraint succeeded, results: "
           << llvm::interleaved(constraintResults);
  } else {
    LDBG() << "  * Constraint failed";
  }
  assert((failed(rewriteResult) || constraintResults.size() == numResults) &&
         "native PDL rewrite function succeeded but returned "
         "unexpected number of results");
  processNativeFunResults(results, numResults, rewriteResult);

  selectJump(isNegated != succeeded(rewriteResult));
}

LogicalResult ByteCodeExecutor::executeApplyRewrite() {
  LDBG() << "Executing ApplyRewrite:";
  const PDLRewriteFunction &rewriteFn = rewriteFunctions[read()];
  SmallVector<PDLValue, 16> args;
  readList<PDLValue>(args);

  LDBG() << "  * Arguments: " << llvm::interleaved(args);

  ByteCodeField numResults = read();
  ByteCodeRewriteResultList results(numResults);
  LogicalResult rewriteResult = rewriteFn(*rewriter, results, args);

  assert(results.getResults().size() == numResults &&
         "native PDL rewrite function returned unexpected number of results");

  processNativeFunResults(results, numResults, rewriteResult);

  if (failed(rewriteResult)) {
    LDBG() << "  - Failed";
    return failure();
  }
  return success();
}

void ByteCodeExecutor::processNativeFunResults(
    ByteCodeRewriteResultList &results, unsigned numResults,
    LogicalResult &rewriteResult) {
  if (failed(rewriteResult)) {
    for (unsigned resultIdx = 0; resultIdx < numResults; resultIdx++) {
      const PDLValue::Kind resultKind = read<PDLValue::Kind>();
      if (resultKind == PDLValue::Kind::TypeRange ||
          resultKind == PDLValue::Kind::ValueRange) {
        skip(2);
      } else {
        skip(1);
      }
    }
    return;
  }

  for (unsigned resultIdx = 0; resultIdx < numResults; resultIdx++) {
    PDLValue::Kind resultKind = read<PDLValue::Kind>();
    (void)resultKind;
    PDLValue result = results.getResults()[resultIdx];
    LDBG() << "  * Result: " << result;
    assert(result.getKind() == resultKind &&
           "native PDL rewrite function returned an unexpected type of "
           "result");
    if (std::optional<TypeRange> typeRange = result.dyn_cast<TypeRange>()) {
      unsigned rangeIndex = read();
      typeRangeMemory[rangeIndex] = *typeRange;
      memory[read()] = &typeRangeMemory[rangeIndex];
    } else if (std::optional<ValueRange> valueRange =
                   result.dyn_cast<ValueRange>()) {
      unsigned rangeIndex = read();
      valueRangeMemory[rangeIndex] = *valueRange;
      memory[read()] = &valueRangeMemory[rangeIndex];
    } else {
      memory[read()] = result.getAsOpaquePointer();
    }
  }

  for (auto &it : results.getAllocatedTypeRanges())
    allocatedTypeRangeMemory.push_back(std::move(it));
  for (auto &it : results.getAllocatedValueRanges())
    allocatedValueRangeMemory.push_back(std::move(it));
}

void ByteCodeExecutor::executeAreEqual() {
  LDBG() << "Executing AreEqual:";
  const void *lhs = read<const void *>();
  const void *rhs = read<const void *>();

  LDBG() << "  * " << lhs << " == " << rhs;
  selectJump(lhs == rhs);
}

void ByteCodeExecutor::executeAreRangesEqual() {
  LDBG() << "Executing AreRangesEqual:";
  PDLValue::Kind valueKind = read<PDLValue::Kind>();
  const void *lhs = read<const void *>();
  const void *rhs = read<const void *>();

  switch (valueKind) {
  case PDLValue::Kind::TypeRange: {
    const TypeRange *lhsRange = reinterpret_cast<const TypeRange *>(lhs);
    const TypeRange *rhsRange = reinterpret_cast<const TypeRange *>(rhs);
    LDBG() << "  * " << lhs << " == " << rhs;
    selectJump(*lhsRange == *rhsRange);
    break;
  }
  case PDLValue::Kind::ValueRange: {
    const auto *lhsRange = reinterpret_cast<const ValueRange *>(lhs);
    const auto *rhsRange = reinterpret_cast<const ValueRange *>(rhs);
    LDBG() << "  * " << lhs << " == " << rhs;
    selectJump(*lhsRange == *rhsRange);
    break;
  }
  default:
    llvm_unreachable("unexpected `AreRangesEqual` value kind");
  }
}

void ByteCodeExecutor::executeBranch() {
  LDBG() << "Executing Branch";
  curCodeIt = &code[read<ByteCodeAddr>()];
}

void ByteCodeExecutor::executeCheckOperandCount() {
  LDBG() << "Executing CheckOperandCount:";
  Operation *op = read<Operation *>();
  uint32_t expectedCount = read<uint32_t>();
  bool compareAtLeast = read();

  LDBG() << "  * Found: " << op->getNumOperands()
         << "\n  * Expected: " << expectedCount
         << "\n  * Comparator: " << (compareAtLeast ? ">=" : "==");
  if (compareAtLeast)
    selectJump(op->getNumOperands() >= expectedCount);
  else
    selectJump(op->getNumOperands() == expectedCount);
}

void ByteCodeExecutor::executeCheckOperationName() {
  LDBG() << "Executing CheckOperationName:";
  Operation *op = read<Operation *>();
  OperationName expectedName = read<OperationName>();

  LDBG() << "  * Found: \"" << op->getName() << "\"\n  * Expected: \""
         << expectedName << "\"";
  selectJump(op->getName() == expectedName);
}

void ByteCodeExecutor::executeCheckResultCount() {
  LDBG() << "Executing CheckResultCount:";
  Operation *op = read<Operation *>();
  uint32_t expectedCount = read<uint32_t>();
  bool compareAtLeast = read();

  LDBG() << "  * Found: " << op->getNumResults()
         << "\n  * Expected: " << expectedCount
         << "\n  * Comparator: " << (compareAtLeast ? ">=" : "==");
  if (compareAtLeast)
    selectJump(op->getNumResults() >= expectedCount);
  else
    selectJump(op->getNumResults() == expectedCount);
}

void ByteCodeExecutor::executeCheckTypes() {
  LDBG() << "Executing AreEqual:";
  TypeRange *lhs = read<TypeRange *>();
  Attribute rhs = read<Attribute>();
  LDBG() << "  * " << lhs << " == " << rhs;

  selectJump(*lhs == cast<ArrayAttr>(rhs).getAsValueRange<TypeAttr>());
}

void ByteCodeExecutor::executeContinue() {
  ByteCodeField level = read();
  LDBG() << "Executing Continue\n  * Level: " << level;
  ++loopIndex[level];
  popCodeIt();
}

void ByteCodeExecutor::executeCreateConstantTypeRange() {
  LDBG() << "Executing CreateConstantTypeRange:";
  unsigned memIndex = read();
  unsigned rangeIndex = read();
  ArrayAttr typesAttr = cast<ArrayAttr>(read<Attribute>());

  LDBG() << "  * Types: " << typesAttr;
  assignRangeToMemory(typesAttr.getAsValueRange<TypeAttr>(), memIndex,
                      rangeIndex);
}

void ByteCodeExecutor::executeCreateOperation() {
  LDBG() << "Executing CreateOperation:";

  unsigned memIndex = read();
  OperationState state(*mainRewriteLoc, read<OperationName>());
  readList(state.operands);
  for (unsigned i = 0, e = read(); i != e; ++i) {
    StringAttr name = read<StringAttr>();
    if (Attribute attr = read<Attribute>())
      state.addAttribute(name, attr);
  }

  unsigned numResults = read();
  if (numResults == kInferTypesMarker) {
    InferTypeOpInterface::Concept *inferInterface =
        state.name.getInterface<InferTypeOpInterface>();
    assert(inferInterface &&
           "expected operation to provide InferTypeOpInterface");

    if (failed(inferInterface->inferReturnTypes(
            state.getContext(), state.location, state.operands,
            state.attributes.getDictionary(state.getContext()),
            state.getRawProperties(), state.regions, state.types)))
      return;
  } else {
    for (unsigned i = 0; i != numResults; ++i) {
      if (read<PDLValue::Kind>() == PDLValue::Kind::Type) {
        state.types.push_back(read<Type>());
      } else {
        TypeRange *resultTypes = read<TypeRange *>();
        state.types.append(resultTypes->begin(), resultTypes->end());
      }
    }
  }

  Operation *resultOp = rewriter->create(state);
  memory[memIndex] = resultOp;

  LDBG() << "  * Attributes: "
         << state.attributes.getDictionary(state.getContext())
         << "\n  * Operands: " << llvm::interleaved(state.operands)
         << "\n  * Result Types: " << llvm::interleaved(state.types)
         << "\n  * Result: " << *resultOp;
}

template <typename T>
void ByteCodeExecutor::executeDynamicCreateRange(StringRef type) {
  LDBG() << "Executing CreateDynamic" << type << "Range:";
  unsigned memIndex = read();
  unsigned rangeIndex = read();
  SmallVector<T> values;
  readList(values);

  LDBG() << "  * " << type << "s: " << llvm::interleaved(values);

  assignRangeToMemory(values, memIndex, rangeIndex);
}

void ByteCodeExecutor::executeEraseOp() {
  LDBG() << "Executing EraseOp:";
  Operation *op = read<Operation *>();

  LDBG() << "  * Operation: " << *op;
  rewriter->eraseOp(op);
}

template <typename T, typename Range, PDLValue::Kind kind>
void ByteCodeExecutor::executeExtract() {
  LDBG() << "Executing Extract" << kind << ":";
  Range *range = read<Range *>();
  unsigned index = read<uint32_t>();
  unsigned memIndex = read();

  if (!range) {
    memory[memIndex] = nullptr;
    return;
  }

  T result = index < range->size() ? (*range)[index] : T();
  LDBG() << "  * " << kind << "s(" << range->size() << ")";
  LDBG() << "  * Index: " << index;
  LDBG() << "  * Result: " << result;
  storeToMemory(memIndex, result);
}

void ByteCodeExecutor::executeFinalize() { LDBG() << "Executing Finalize"; }

void ByteCodeExecutor::executeForEach() {
  LDBG() << "Executing ForEach:";
  const ByteCodeField *prevCodeIt = getPrevCodeIt();
  unsigned rangeIndex = read();
  unsigned memIndex = read();
  const void *value = nullptr;

  switch (read<PDLValue::Kind>()) {
  case PDLValue::Kind::Operation: {
    unsigned &index = loopIndex[read()];
    ArrayRef<Operation *> array = opRangeMemory[rangeIndex];
    assert(index <= array.size() && "iterated past the end");
    if (index < array.size()) {
      LDBG() << "  * Result: " << array[index];
      value = array[index];
      break;
    }
    LDBG() << "  * Done";
    index = 0;
    selectJump(size_t(0));
    return;
  }
  case PDLValue::Kind::Value: {
    unsigned &index = loopIndex[read()];
    ValueRange range = valueRangeMemory[rangeIndex];
    assert(index <= range.size() && "iterated past the end");
    if (index < range.size()) {
      LLVM_DEBUG(llvm::dbgs() << "  * Result: " << range[index] << "\n");
      value = range[index].getAsOpaquePointer();
      break;
    }
    LLVM_DEBUG(llvm::dbgs() << "  * Done\n");
    index = 0;
    selectJump(size_t(0));
    return;
  }
  case PDLValue::Kind::Type: {
    unsigned &index = loopIndex[read()];
    TypeRange range = typeRangeMemory[rangeIndex];
    assert(index <= range.size() && "iterated past the end");
    if (index < range.size()) {
      LLVM_DEBUG(llvm::dbgs() << "  * Result: " << range[index] << "\n");
      value = range[index].getAsOpaquePointer();
      break;
    }
    LLVM_DEBUG(llvm::dbgs() << "  * Done\n");
    index = 0;
    selectJump(size_t(0));
    return;
  }
  default:
    llvm_unreachable("unexpected `ForEach` value kind");
  }

  memory[memIndex] = value;
  pushCodeIt(prevCodeIt);
  read<ByteCodeAddr>();
}

void ByteCodeExecutor::executeGetAttribute() {
  LDBG() << "Executing GetAttribute:";
  unsigned memIndex = read();
  Operation *op = read<Operation *>();
  StringAttr attrName = read<StringAttr>();
  Attribute attr = op->getAttr(attrName);

  LDBG() << "  * Operation: " << *op << "\n  * Attribute: " << attrName
         << "\n  * Result: " << attr;
  memory[memIndex] = attr.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetAttributeType() {
  LDBG() << "Executing GetAttributeType:";
  unsigned memIndex = read();
  Attribute attr = read<Attribute>();
  Type type;
  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    type = typedAttr.getType();

  LDBG() << "  * Attribute: " << attr << "\n  * Result: " << type;
  memory[memIndex] = type.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetDefiningOp() {
  LDBG() << "Executing GetDefiningOp:";
  unsigned memIndex = read();
  Operation *op = nullptr;
  if (read<PDLValue::Kind>() == PDLValue::Kind::Value) {
    Value value = read<Value>();
    if (value)
      op = value.getDefiningOp();
    LDBG() << "  * Value: " << value;
  } else {
    ValueRange *values = read<ValueRange *>();
    if (values && !values->empty()) {
      op = values->front().getDefiningOp();
    }
    LDBG() << "  * Values: " << values;
  }

  LDBG() << "  * Result: " << op;
  memory[memIndex] = op;
}

void ByteCodeExecutor::executeGetOperand(unsigned index) {
  Operation *op = read<Operation *>();
  unsigned memIndex = read();
  Value operand =
      index < op->getNumOperands() ? op->getOperand(index) : Value();

  LDBG() << "  * Operation: " << *op << "\n  * Index: " << index
         << "\n  * Result: " << operand;
  memory[memIndex] = operand.getAsOpaquePointer();
}

/// This function is the internal implementation of `GetResults` and
/// `GetOperands` that provides support for extracting a value range from the
/// given operation.
template <template <typename> class AttrSizedSegmentsT, typename RangeT>
static void *
executeGetOperandsResults(RangeT values, Operation *op, unsigned index,
                          ByteCodeField rangeIndex, StringRef attrSizedSegments,
                          MutableArrayRef<ValueRange> valueRangeMemory) {
  if (index == std::numeric_limits<uint32_t>::max()) {
    LDBG() << "  * Getting all values";
  } else if (op->hasTrait<AttrSizedSegmentsT>()) {
    LDBG() << "  * Extracting values from `" << attrSizedSegments << "`";
    auto segmentAttr = op->getAttrOfType<DenseI32ArrayAttr>(attrSizedSegments);
    if (!segmentAttr || segmentAttr.asArrayRef().size() <= index)
      return nullptr;
    ArrayRef<int32_t> segments = segmentAttr;
    unsigned startIndex = llvm::sum_of(segments.take_front(index));
    values = values.slice(startIndex, *std::next(segments.begin(), index));
    LDBG() << "  * Extracting range[" << startIndex << ", "
           << *std::next(segments.begin(), index) << "]";
  } else if (values.size() >= index) {
    LDBG() << "  * Treating values as trailing variadic range";
    values = values.drop_front(index);
  } else {
    return nullptr;
  }

  if (rangeIndex != std::numeric_limits<ByteCodeField>::max()) {
    valueRangeMemory[rangeIndex] = values;
    return &valueRangeMemory[rangeIndex];
  }
  return values.size() != 1 ? nullptr : values.front().getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetOperands() {
  LDBG() << "Executing GetOperands:";
  unsigned index = read<uint32_t>();
  Operation *op = read<Operation *>();
  ByteCodeField rangeIndex = read();

  void *result = executeGetOperandsResults<OpTrait::AttrSizedOperandSegments>(
      op->getOperands(), op, index, rangeIndex, "operandSegmentSizes",
      valueRangeMemory);
  if (!result)
    LDBG() << "  * Invalid operand range";
  memory[read()] = result;
}

void ByteCodeExecutor::executeGetResult(unsigned index) {
  Operation *op = read<Operation *>();
  unsigned memIndex = read();
  OpResult result =
      index < op->getNumResults() ? op->getResult(index) : OpResult();

  LDBG() << "  * Operation: " << *op << "\n  * Index: " << index
         << "\n  * Result: " << result;
  memory[memIndex] = result.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetResults() {
  LDBG() << "Executing GetResults:";
  unsigned index = read<uint32_t>();
  Operation *op = read<Operation *>();
  ByteCodeField rangeIndex = read();

  void *result = executeGetOperandsResults<OpTrait::AttrSizedResultSegments>(
      op->getResults(), op, index, rangeIndex, "resultSegmentSizes",
      valueRangeMemory);
  if (!result)
    LDBG() << "  * Invalid result range";
  memory[read()] = result;
}

void ByteCodeExecutor::executeGetUsers() {
  LDBG() << "Executing GetUsers:";
  unsigned memIndex = read();
  unsigned rangeIndex = read();
  std::vector<Operation *> &range = opRangeMemory[rangeIndex];
  memory[memIndex] = &range;

  range.clear();
  if (read<PDLValue::Kind>() == PDLValue::Kind::Value) {
    Value value = read<Value>();
    if (!value)
      return;
    LDBG() << "  * Value: " << value;
    range.assign(value.user_begin(), value.user_end());
  } else {
    ValueRange *values = read<ValueRange *>();
    if (!values)
      return;
    LDBG() << "  * Values (" << values->size()
           << "): " << llvm::interleaved(*values);
    for (Value value : *values)
      range.insert(range.end(), value.user_begin(), value.user_end());
  }

  LDBG() << "  * Result: " << range.size() << " operations";
}

void ByteCodeExecutor::executeGetValueType() {
  LDBG() << "Executing GetValueType:";
  unsigned memIndex = read();
  Value value = read<Value>();
  Type type = value ? value.getType() : Type();

  LDBG() << "  * Value: " << value << "\n  * Result: " << type;
  memory[memIndex] = type.getAsOpaquePointer();
}

void ByteCodeExecutor::executeGetValueRangeTypes() {
  LDBG() << "Executing GetValueRangeTypes:";
  unsigned memIndex = read();
  unsigned rangeIndex = read();
  ValueRange *values = read<ValueRange *>();
  if (!values) {
    LDBG() << "  * Values: <NULL>";
    memory[memIndex] = nullptr;
    return;
  }

  LDBG() << "  * Values (" << values->size()
         << "): " << llvm::interleaved(*values)
         << "\n  * Result: " << llvm::interleaved(values->getType());
  typeRangeMemory[rangeIndex] = values->getType();
  memory[memIndex] = &typeRangeMemory[rangeIndex];
}

void ByteCodeExecutor::executeIsNotNull() {
  LDBG() << "Executing IsNotNull:";
  const void *value = read<const void *>();

  LDBG() << "  * Value: " << value;
  selectJump(value != nullptr);
}

void ByteCodeExecutor::executeRecordMatch() {
  LDBG() << "Executing RecordMatch:";
  unsigned patternIndex = read();
  PatternBenefit benefit = currentPatternBenefits[patternIndex];
  const ByteCodeField *dest = &code[read<ByteCodeAddr>()];

  if (benefit.isImpossibleToMatch()) {
    LDBG() << "  * Benefit: Impossible To Match";
    curCodeIt = dest;
    return;
  }

  unsigned numMatchLocs = read();
  SmallVector<Location, 4> matchLocs;
  matchLocs.reserve(numMatchLocs);
  for (unsigned i = 0; i != numMatchLocs; ++i)
    matchLocs.push_back(read<Operation *>()->getLoc());
  Location matchLoc = rewriter->getFusedLoc(matchLocs);

  LDBG() << "  * Benefit: " << benefit.getBenefit();
  LDBG() << "  * Location: " << matchLoc;
  matches->emplace_back(matchLoc, patterns[patternIndex], benefit);
  PDLByteCode::MatchResult &match = matches->back();

  unsigned numInputs = read();
  match.values.reserve(numInputs);
  match.typeRangeValues.reserve(numInputs);
  match.valueRangeValues.reserve(numInputs);
  for (unsigned i = 0; i < numInputs; ++i) {
    switch (read<PDLValue::Kind>()) {
    case PDLValue::Kind::TypeRange:
      match.typeRangeValues.push_back(*read<TypeRange *>());
      match.values.push_back(&match.typeRangeValues.back());
      break;
    case PDLValue::Kind::ValueRange:
      match.valueRangeValues.push_back(*read<ValueRange *>());
      match.values.push_back(&match.valueRangeValues.back());
      break;
    default:
      match.values.push_back(read<const void *>());
      break;
    }
  }
  curCodeIt = dest;
}

void ByteCodeExecutor::executeReplaceOp() {
  LDBG() << "Executing ReplaceOp:";
  Operation *op = read<Operation *>();
  SmallVector<Value, 16> args;
  readList(args);

  LDBG() << "  * Operation: " << *op
         << "\n  * Values: " << llvm::interleaved(args);
  rewriter->replaceOp(op, args);
}

void ByteCodeExecutor::executeSwitchAttribute() {
  LDBG() << "Executing SwitchAttribute:";
  Attribute value = read<Attribute>();
  ArrayAttr cases = read<ArrayAttr>();
  handleSwitch(value, cases);
}

void ByteCodeExecutor::executeSwitchOperandCount() {
  LDBG() << "Executing SwitchOperandCount:";
  Operation *op = read<Operation *>();
  auto cases = read<DenseTypedElementsAttr>().getValues<uint32_t>();

  LDBG() << "  * Operation: " << *op;
  handleSwitch(op->getNumOperands(), cases);
}

void ByteCodeExecutor::executeSwitchOperationName() {
  LDBG() << "Executing SwitchOperationName:";
  OperationName value = read<Operation *>()->getName();
  size_t caseCount = read();

  LLVM_DEBUG({
    const ByteCodeField *prevCodeIt = curCodeIt;
    LDBG() << "  * Value: " << value << "\n  * Cases: "
           << llvm::interleaved(
                  llvm::map_range(llvm::seq<size_t>(0, caseCount), [&](size_t) {
                    return read<OperationName>();
                  }));
    curCodeIt = prevCodeIt;
  });

  for (size_t i = 0; i != caseCount; ++i) {
    if (read<OperationName>() == value) {
      curCodeIt += (caseCount - i - 1);
      return selectJump(i + 1);
    }
  }
  selectJump(size_t(0));
}

void ByteCodeExecutor::executeSwitchResultCount() {
  LDBG() << "Executing SwitchResultCount:";
  Operation *op = read<Operation *>();
  auto cases = read<DenseTypedElementsAttr>().getValues<uint32_t>();

  LDBG() << "  * Operation: " << *op;
  handleSwitch(op->getNumResults(), cases);
}

void ByteCodeExecutor::executeSwitchType() {
  LDBG() << "Executing SwitchType:";
  Type value = read<Type>();
  auto cases = read<ArrayAttr>().getAsValueRange<TypeAttr>();
  handleSwitch(value, cases);
}

void ByteCodeExecutor::executeSwitchTypes() {
  LDBG() << "Executing SwitchTypes:";
  TypeRange *value = read<TypeRange *>();
  auto cases = read<ArrayAttr>().getAsRange<ArrayAttr>();
  if (!value) {
    LDBG() << "Types: <NULL>";
    return selectJump(size_t(0));
  }
  handleSwitch(*value, cases, [](ArrayAttr caseValue, const TypeRange &value) {
    return value == caseValue.getAsValueRange<TypeAttr>();
  });
}

//===----------------------------------------------------------------------===//
// Tail-calling handler functions
//
// Each handler calls the corresponding executeX method (which does the real
// work and may allocate locals with non-trivial destructors), then uses
// DISPATCH to tail-call the next handler. Because the executeX call returns
// before DISPATCH, all non-trivially-destructible locals are dead at the
// musttail call site, satisfying Clang's requirements.
//
// Reopened anonymous namespace to match the forward declarations above.
//===----------------------------------------------------------------------===//
namespace {

LogicalResult handleApplyConstraint(ByteCodeExecutor &exec) {
  exec.executeApplyConstraint();
  DISPATCH(exec);
}

LogicalResult handleApplyRewrite(ByteCodeExecutor &exec) {
  if (failed(exec.executeApplyRewrite()))
    return failure();
  DISPATCH(exec);
}

LogicalResult handleAreEqual(ByteCodeExecutor &exec) {
  exec.executeAreEqual();
  DISPATCH(exec);
}

LogicalResult handleAreRangesEqual(ByteCodeExecutor &exec) {
  exec.executeAreRangesEqual();
  DISPATCH(exec);
}

LogicalResult handleBranch(ByteCodeExecutor &exec) {
  exec.executeBranch();
  DISPATCH(exec);
}

LogicalResult handleCheckOperandCount(ByteCodeExecutor &exec) {
  exec.executeCheckOperandCount();
  DISPATCH(exec);
}

LogicalResult handleCheckOperationName(ByteCodeExecutor &exec) {
  exec.executeCheckOperationName();
  DISPATCH(exec);
}

LogicalResult handleCheckResultCount(ByteCodeExecutor &exec) {
  exec.executeCheckResultCount();
  DISPATCH(exec);
}

LogicalResult handleCheckTypes(ByteCodeExecutor &exec) {
  exec.executeCheckTypes();
  DISPATCH(exec);
}

LogicalResult handleContinue(ByteCodeExecutor &exec) {
  exec.executeContinue();
  DISPATCH(exec);
}

LogicalResult handleCreateConstantTypeRange(ByteCodeExecutor &exec) {
  exec.executeCreateConstantTypeRange();
  DISPATCH(exec);
}

LogicalResult handleCreateOperation(ByteCodeExecutor &exec) {
  exec.executeCreateOperation();
  DISPATCH(exec);
}

LogicalResult handleCreateDynamicTypeRange(ByteCodeExecutor &exec) {
  exec.executeDynamicCreateRange<Type>("Type");
  DISPATCH(exec);
}

LogicalResult handleCreateDynamicValueRange(ByteCodeExecutor &exec) {
  exec.executeDynamicCreateRange<Value>("Value");
  DISPATCH(exec);
}

LogicalResult handleEraseOp(ByteCodeExecutor &exec) {
  exec.executeEraseOp();
  DISPATCH(exec);
}

LogicalResult handleExtractOp(ByteCodeExecutor &exec) {
  exec.executeExtract<Operation *, std::vector<Operation *>,
                      PDLValue::Kind::Operation>();
  DISPATCH(exec);
}

LogicalResult handleExtractType(ByteCodeExecutor &exec) {
  exec.executeExtract<Type, TypeRange, PDLValue::Kind::Type>();
  DISPATCH(exec);
}

LogicalResult handleExtractValue(ByteCodeExecutor &exec) {
  exec.executeExtract<Value, ValueRange, PDLValue::Kind::Value>();
  DISPATCH(exec);
}

LogicalResult handleFinalize(ByteCodeExecutor &exec) {
  exec.executeFinalize();
  LDBG() << "";
  return success();
}

LogicalResult handleForEach(ByteCodeExecutor &exec) {
  exec.executeForEach();
  DISPATCH(exec);
}

LogicalResult handleGetAttribute(ByteCodeExecutor &exec) {
  exec.executeGetAttribute();
  DISPATCH(exec);
}

LogicalResult handleGetAttributeType(ByteCodeExecutor &exec) {
  exec.executeGetAttributeType();
  DISPATCH(exec);
}

LogicalResult handleGetDefiningOp(ByteCodeExecutor &exec) {
  exec.executeGetDefiningOp();
  DISPATCH(exec);
}

LogicalResult handleGetOperand0(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetOperand0:";
  exec.executeGetOperand(0);
  DISPATCH(exec);
}

LogicalResult handleGetOperand1(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetOperand1:";
  exec.executeGetOperand(1);
  DISPATCH(exec);
}

LogicalResult handleGetOperand2(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetOperand2:";
  exec.executeGetOperand(2);
  DISPATCH(exec);
}

LogicalResult handleGetOperand3(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetOperand3:";
  exec.executeGetOperand(3);
  DISPATCH(exec);
}

LogicalResult handleGetOperandN(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetOperandN:";
  exec.executeGetOperand(exec.read<uint32_t>());
  DISPATCH(exec);
}

LogicalResult handleGetOperands(ByteCodeExecutor &exec) {
  exec.executeGetOperands();
  DISPATCH(exec);
}

LogicalResult handleGetResult0(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetResult0:";
  exec.executeGetResult(0);
  DISPATCH(exec);
}

LogicalResult handleGetResult1(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetResult1:";
  exec.executeGetResult(1);
  DISPATCH(exec);
}

LogicalResult handleGetResult2(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetResult2:";
  exec.executeGetResult(2);
  DISPATCH(exec);
}

LogicalResult handleGetResult3(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetResult3:";
  exec.executeGetResult(3);
  DISPATCH(exec);
}

LogicalResult handleGetResultN(ByteCodeExecutor &exec) {
  LDBG() << "Executing GetResultN:";
  exec.executeGetResult(exec.read<uint32_t>());
  DISPATCH(exec);
}

LogicalResult handleGetResults(ByteCodeExecutor &exec) {
  exec.executeGetResults();
  DISPATCH(exec);
}

LogicalResult handleGetUsers(ByteCodeExecutor &exec) {
  exec.executeGetUsers();
  DISPATCH(exec);
}

LogicalResult handleGetValueType(ByteCodeExecutor &exec) {
  exec.executeGetValueType();
  DISPATCH(exec);
}

LogicalResult handleGetValueRangeTypes(ByteCodeExecutor &exec) {
  exec.executeGetValueRangeTypes();
  DISPATCH(exec);
}

LogicalResult handleIsNotNull(ByteCodeExecutor &exec) {
  exec.executeIsNotNull();
  DISPATCH(exec);
}

LogicalResult handleRecordMatch(ByteCodeExecutor &exec) {
  assert(exec.matches &&
         "expected matches to be provided when executing the matcher");
  exec.executeRecordMatch();
  DISPATCH(exec);
}

LogicalResult handleReplaceOp(ByteCodeExecutor &exec) {
  exec.executeReplaceOp();
  DISPATCH(exec);
}

LogicalResult handleSwitchAttribute(ByteCodeExecutor &exec) {
  exec.executeSwitchAttribute();
  DISPATCH(exec);
}

LogicalResult handleSwitchOperandCount(ByteCodeExecutor &exec) {
  exec.executeSwitchOperandCount();
  DISPATCH(exec);
}

LogicalResult handleSwitchOperationName(ByteCodeExecutor &exec) {
  exec.executeSwitchOperationName();
  DISPATCH(exec);
}

LogicalResult handleSwitchResultCount(ByteCodeExecutor &exec) {
  exec.executeSwitchResultCount();
  DISPATCH(exec);
}

LogicalResult handleSwitchType(ByteCodeExecutor &exec) {
  exec.executeSwitchType();
  DISPATCH(exec);
}

LogicalResult handleSwitchTypes(ByteCodeExecutor &exec) {
  exec.executeSwitchTypes();
  DISPATCH(exec);
}

} // namespace

#undef DISPATCH

//===----------------------------------------------------------------------===//
// PDLByteCode match/rewrite entry points
//===----------------------------------------------------------------------===//

void PDLByteCode::match(Operation *op, PatternRewriter &rewriter,
                        SmallVectorImpl<MatchResult> &matches,
                        PDLByteCodeMutableState &state) const {
  // The first memory slot is always the root operation.
  state.memory[0] = op;

  // The matcher function always starts at code address 0.
  ByteCodeExecutor executor(
      matcherByteCode.data(), state.memory, state.opRangeMemory,
      state.typeRangeMemory, state.allocatedTypeRangeMemory,
      state.valueRangeMemory, state.allocatedValueRangeMemory, state.loopIndex,
      uniquedData, matcherByteCode, state.currentPatternBenefits, patterns,
      constraintFunctions, rewriteFunctions);
  LogicalResult executeResult = executor.execute(rewriter, &matches);
  (void)executeResult;
  assert(succeeded(executeResult) && "unexpected matcher execution failure");

  // Order the found matches by benefit.
  llvm::stable_sort(matches,
                    [](const MatchResult &lhs, const MatchResult &rhs) {
                      return lhs.benefit > rhs.benefit;
                    });
}

LogicalResult PDLByteCode::rewrite(PatternRewriter &rewriter,
                                   const MatchResult &match,
                                   PDLByteCodeMutableState &state) const {
  auto *configSet = match.pattern->getConfigSet();
  if (configSet)
    configSet->notifyRewriteBegin(rewriter);

  // The arguments of the rewrite function are stored at the start of the
  // memory buffer.
  llvm::copy(match.values, state.memory.begin());

  ByteCodeExecutor executor(
      &rewriterByteCode[match.pattern->getRewriterAddr()], state.memory,
      state.opRangeMemory, state.typeRangeMemory,
      state.allocatedTypeRangeMemory, state.valueRangeMemory,
      state.allocatedValueRangeMemory, state.loopIndex, uniquedData,
      rewriterByteCode, state.currentPatternBenefits, patterns,
      constraintFunctions, rewriteFunctions);
  LogicalResult result =
      executor.execute(rewriter, /*matches=*/nullptr, match.location);

  if (configSet)
    configSet->notifyRewriteEnd(rewriter);

  if (failed(result) && !rewriter.canRecoverFromRewriteFailure()) {
    LDBG() << " and rollback is not supported - aborting";
    llvm::report_fatal_error(
        "Native PDL Rewrite failed, but the pattern "
        "rewriter doesn't support recovery. Failable pattern rewrites should "
        "not be used with pattern rewriters that do not support them.");
  }
  return result;
}
