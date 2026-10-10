//===- TestArithCanonicalizationPDLL.cpp - DRR vs. PDLL arith patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test passes that apply either the DRR patterns of
// lib/Dialect/Arith/IR/ArithCanonicalization.td, or their PDLL port in
// ArithCanonicalization.pdll, using the greedy pattern rewrite driver. Only
// those patterns are applied (no C++ canonicalization patterns), so the
// passes can be used to compare the performance of DRR and PDL rewrites. The
// PDLL patterns are either run by the PDL bytecode interpreter, or compiled to
// C++ through the match dialect by mlir-match-to-cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// Pattern helpers
//===----------------------------------------------------------------------===//

// These mirror the (file-local) helpers in lib/Dialect/Arith/IR/ArithOps.cpp
// that the DRR patterns reference, and are also used by the native constraints
// and rewrites of the PDLL patterns.

static IntegerAttr
applyToIntegerAttrs(PatternRewriter &builder, Value res, Attribute lhs,
                    Attribute rhs,
                    function_ref<APInt(const APInt &, const APInt &)> binFn) {
  const APInt &lhsVal = llvm::cast<IntegerAttr>(lhs).getValue();
  const APInt &rhsVal = llvm::cast<IntegerAttr>(rhs).getValue();
  APInt value = binFn(lhsVal, rhsVal);
  return IntegerAttr::get(res.getType(), value);
}

static IntegerAttr addIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::plus<APInt>());
}

static IntegerAttr subIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::minus<APInt>());
}

static IntegerAttr mulIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::multiplies<APInt>());
}

static IntegerAttr andIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::bit_and<APInt>());
}

static IntegerAttr orIntegerAttrs(PatternRewriter &builder, Value res,
                                  Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::bit_or<APInt>());
}

static IntegerAttr xorIntegerAttrs(PatternRewriter &builder, Value res,
                                   Attribute lhs, Attribute rhs) {
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::bit_xor<APInt>());
}

// Merge overflow flags from 2 ops, selecting the most conservative combination.
static IntegerOverflowFlagsAttr
mergeOverflowFlags(IntegerOverflowFlagsAttr val1,
                   IntegerOverflowFlagsAttr val2) {
  return IntegerOverflowFlagsAttr::get(val1.getContext(),
                                       val1.getValue() & val2.getValue());
}

// PDLL variant of the above that reads the flags off the ops. DRR substitutes
// the default value for an absent (null) flags attribute, do the same here.
static IntegerOverflowFlags getOverflowFlags(Operation *op) {
  IntegerOverflowFlagsAttr attr =
      cast<ArithIntegerOverflowFlagsInterface>(op).getOverflowAttr();
  return attr ? attr.getValue() : IntegerOverflowFlags::none;
}

static IntegerOverflowFlagsAttr mergeOverflowFlags(Operation *op1,
                                                   Operation *op2) {
  return IntegerOverflowFlagsAttr::get(op1->getContext(),
                                       getOverflowFlags(op1) &
                                           getOverflowFlags(op2));
}

static CmpIPredicateAttr invertPredicate(CmpIPredicateAttr pred) {
  return CmpIPredicateAttr::get(pred.getContext(),
                                arith::invertPredicate(pred.getValue()));
}

static int64_t getScalarOrElementWidth(Type type) {
  Type elemTy = getElementTypeOrSelf(type);
  if (elemTy.isIntOrFloat())
    return elemTy.getIntOrFloatBitWidth();

  return -1;
}

static int64_t getScalarOrElementWidth(Value value) {
  return getScalarOrElementWidth(value.getType());
}

static FailureOr<APInt> getIntOrSplatIntValue(Attribute attr) {
  APInt value;
  if (matchPattern(attr, m_ConstantInt(&value)))
    return value;

  return failure();
}

// Returns true if `value` is produced by a constant-like op whose value is the
// i1 attribute `expected` (DRR: ConstantLikeMatcher ConstantAttr<I1Attr, ..>).
static bool isConstantI1(Value value, bool expected) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return false;
  Builder builder(value.getContext());
  return attr == builder.getIntegerAttr(builder.getIntegerType(1), expected);
}

//===----------------------------------------------------------------------===//
// DRR patterns
//===----------------------------------------------------------------------===//

namespace drr {
#include "ArithCanonicalization.inc"
} // namespace drr

//===----------------------------------------------------------------------===//
// PDLL patterns
//===----------------------------------------------------------------------===//

#include "ArithCanonicalizationPDLL.h.inc"

/// Native constraint implementing DRR's `(ConstantLikeMatcher AttrT:$c)` on the
/// value `$c` is bound to: matches if the value is produced by a constant-like
/// op that folds to an attribute of kind `AttrT`, and returns that attribute.
template <typename AttrT>
static FailureOr<Attribute> constantValueOfKind(PatternRewriter &,
                                                Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)) || !isa<AttrT>(attr))
    return failure();
  return attr;
}

static FailureOr<Attribute> ConstantValuePDLFn(PatternRewriter &rewriter,
                                               Value value) {
  return constantValueOfKind<Attribute>(rewriter, value);
}
static FailureOr<Attribute> ConstantIntValuePDLFn(PatternRewriter &rewriter,
                                                  Value value) {
  return constantValueOfKind<IntegerAttr>(rewriter, value);
}
static FailureOr<Attribute> ConstantTypedValuePDLFn(PatternRewriter &rewriter,
                                                    Value value) {
  return constantValueOfKind<TypedAttr>(rewriter, value);
}

template <FailureOr<Attribute> (*ConstraintFn)(PatternRewriter &, Value)>
static LogicalResult withResultList(PatternRewriter &rewriter,
                                    PDLResultList &results,
                                    ArrayRef<PDLValue> args) {
  FailureOr<Attribute> attr = ConstraintFn(rewriter, args[0].cast<Value>());
  if (failed(attr))
    return failure();
  results.push_back(*attr);
  return success();
}

static void populatePDLLPatterns(RewritePatternSet &patterns) {
  populateGeneratedPDLLPatterns(patterns);

  // Native constraints with results can't be defined inline in PDLL.
  PDLPatternModule &pdlPatterns = patterns.getPDLPatterns();
  pdlPatterns.registerConstraintFunction("ConstantValue",
                                         withResultList<ConstantValuePDLFn>);
  pdlPatterns.registerConstraintFunction("ConstantIntValue",
                                         withResultList<ConstantIntValuePDLFn>);
  pdlPatterns.registerConstraintFunction(
      "ConstantTypedValue", withResultList<ConstantTypedValuePDLFn>);
}

//===----------------------------------------------------------------------===//
// PDLL patterns compiled to C++ through the match dialect
//===----------------------------------------------------------------------===//

#include "ArithCanonicalizationMatch.h.inc"
#include "ArithCanonicalizationMatchSplit.h.inc"

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

namespace {
template <typename DerivedT>
struct TestArithCanonicalizationBase
    : public PassWrapper<DerivedT, OperationPass<>> {
  TestArithCanonicalizationBase() = default;
  TestArithCanonicalizationBase(const TestArithCanonicalizationBase &other)
      : PassWrapper<DerivedT, OperationPass<>>(other) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, match::MatchDialect, pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect>();
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    // Build (and for PDL, compile) the pattern set once, outside of
    // `runOnOperation`.
    RewritePatternSet patternList(ctx);
    static_cast<DerivedT *>(this)->populatePatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.enableFolding(fold);
    (void)applyPatternsGreedily(this->getOperation(), patterns, config);
  }

  Pass::Option<bool> fold{*this, "fold",
                          llvm::cl::desc("Fold operations while rewriting"),
                          llvm::cl::init(true)};

  FrozenRewritePatternSet patterns;
};

struct TestArithCanonicalizationDRRPass
    : public TestArithCanonicalizationBase<TestArithCanonicalizationDRRPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestArithCanonicalizationDRRPass)

  StringRef getArgument() const final { return "test-arith-canonicalize-drr"; }
  StringRef getDescription() const final {
    return "Apply the DRR arith canonicalization patterns";
  }

  void populatePatterns(RewritePatternSet &patterns) {
    drr::populateWithGenerated(patterns);
  }
};

struct TestArithCanonicalizationPDLLPass
    : public TestArithCanonicalizationBase<TestArithCanonicalizationPDLLPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestArithCanonicalizationPDLLPass)

  StringRef getArgument() const final { return "test-arith-canonicalize-pdll"; }
  StringRef getDescription() const final {
    return "Apply the PDLL port of the DRR arith canonicalization patterns";
  }

  void populatePatterns(RewritePatternSet &patterns) {
    populatePDLLPatterns(patterns);
  }
};

struct TestArithCanonicalizationMatchCppPass
    : public TestArithCanonicalizationBase<
          TestArithCanonicalizationMatchCppPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestArithCanonicalizationMatchCppPass)

  StringRef getArgument() const final {
    return "test-arith-canonicalize-match-cpp";
  }
  StringRef getDescription() const final {
    return "Apply the PDLL arith canonicalization patterns compiled to a "
           "single combined C++ matcher";
  }

  void populatePatterns(RewritePatternSet &patterns) {
    match_cpp::populateGeneratedPatterns(patterns);
  }
};

struct TestArithCanonicalizationMatchCppSplitPass
    : public TestArithCanonicalizationBase<
          TestArithCanonicalizationMatchCppSplitPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestArithCanonicalizationMatchCppSplitPass)

  StringRef getArgument() const final {
    return "test-arith-canonicalize-match-cpp-split";
  }
  StringRef getDescription() const final {
    return "Apply the PDLL arith canonicalization patterns compiled to one "
           "C++ pattern per PDL pattern";
  }

  void populatePatterns(RewritePatternSet &patterns) {
    match_cpp_split::populateGeneratedPatterns(patterns);
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestArithCanonicalizationPasses() {
  PassRegistration<TestArithCanonicalizationDRRPass>();
  PassRegistration<TestArithCanonicalizationPDLLPass>();
  PassRegistration<TestArithCanonicalizationMatchCppPass>();
  PassRegistration<TestArithCanonicalizationMatchCppSplitPass>();
}
} // namespace test
} // namespace mlir
