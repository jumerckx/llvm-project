//===- RewriterGen.cpp - PDL -> PDL Interp rewriter generation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PDLToPDLInterp/RewriterGen.h"

#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Helper that performs the actual lowering of a single PDL rewriter region.
class RewriterGen {
public:
  RewriterGen(OpBuilder &builder, SmallVectorImpl<Value> &usedMatchValues)
      : builder(builder), usedMatchValues(usedMatchValues) {}

  /// Generate the body of `rewriterFunc` from `pattern`.
  void generate(pdl::PatternOp pattern, pdl_interp::FuncOp rewriterFunc);

private:
  // Per-op rewriter generators.
  void generateRewriter(pdl::ApplyNativeRewriteOp rewriteOp);
  void generateRewriter(pdl::AttributeOp attrOp);
  void generateRewriter(pdl::EraseOp eraseOp);
  void generateRewriter(pdl::OperationOp operationOp);
  void generateRewriter(pdl::RangeOp rangeOp);
  void generateRewriter(pdl::ReplaceOp replaceOp);
  void generateRewriter(pdl::ResultOp resultOp);
  void generateRewriter(pdl::ResultsOp resultOp);
  void generateRewriter(pdl::TypeOp typeOp);
  void generateRewriter(pdl::TypesOp typeOp);

  /// Compute the result-type operands for a created operation.
  void generateOperationResultTypeRewriter(pdl::OperationOp op,
                                           SmallVectorImpl<Value> &types,
                                           bool &hasInferredResultTypes);

  /// Translate a PDL match value into an SSA value usable inside the rewriter
  /// function, materializing constants in-place and adding match values as
  /// function arguments on demand.
  Value mapRewriteValue(Value oldValue);

  OpBuilder &builder;
  SmallVectorImpl<Value> &usedMatchValues;
  pdl_interp::FuncOp rewriterFunc;
  DenseMap<Value, Value> rewriteValues;
};

} // namespace

Value RewriterGen::mapRewriteValue(Value oldValue) {
  Value &newValue = rewriteValues[oldValue];
  if (newValue)
    return newValue;

  // Prefer materializing constants directly when possible.
  Operation *oldOp = oldValue.getDefiningOp();
  if (pdl::AttributeOp attrOp = dyn_cast<pdl::AttributeOp>(oldOp)) {
    if (Attribute value = attrOp.getValueAttr()) {
      return newValue = pdl_interp::CreateAttributeOp::create(
                 builder, attrOp.getLoc(), value);
    }
  } else if (pdl::TypeOp typeOp = dyn_cast<pdl::TypeOp>(oldOp)) {
    if (TypeAttr type = typeOp.getConstantTypeAttr()) {
      return newValue = pdl_interp::CreateTypeOp::create(
                 builder, typeOp.getLoc(), type);
    }
  } else if (pdl::TypesOp typeOp = dyn_cast<pdl::TypesOp>(oldOp)) {
    if (ArrayAttr type = typeOp.getConstantTypesAttr()) {
      return newValue = pdl_interp::CreateTypesOp::create(
                 builder, typeOp.getLoc(), typeOp.getType(), type);
    }
  }

  // Otherwise, add this as an input to the rewriter.
  usedMatchValues.push_back(oldValue);
  return newValue =
             rewriterFunc.front().addArgument(oldValue.getType(),
                                              oldValue.getLoc());
}

void RewriterGen::generate(pdl::PatternOp pattern,
                           pdl_interp::FuncOp rewriterFunc) {
  this->rewriterFunc = rewriterFunc;

  builder.setInsertionPointToEnd(&rewriterFunc.front());

  // If this is a custom rewriter, simply dispatch to the registered rewrite
  // method.
  pdl::RewriteOp rewriter = pattern.getRewriter();
  if (StringAttr rewriteName = rewriter.getNameAttr()) {
    SmallVector<Value> args;
    if (rewriter.getRoot())
      args.push_back(mapRewriteValue(rewriter.getRoot()));
    auto mappedArgs =
        llvm::map_range(rewriter.getExternalArgs(),
                        [&](Value v) { return mapRewriteValue(v); });
    args.append(mappedArgs.begin(), mappedArgs.end());
    pdl_interp::ApplyRewriteOp::create(builder, rewriter.getLoc(),
                                       /*results=*/TypeRange(), rewriteName,
                                       args);
  } else {
    // Otherwise this is a dag rewriter defined using PDL operations.
    for (Operation &rewriteOp : *rewriter.getBody()) {
      llvm::TypeSwitch<Operation *>(&rewriteOp)
          .Case<pdl::ApplyNativeRewriteOp, pdl::AttributeOp, pdl::EraseOp,
                pdl::OperationOp, pdl::RangeOp, pdl::ReplaceOp, pdl::ResultOp,
                pdl::ResultsOp, pdl::TypeOp, pdl::TypesOp>(
              [&](auto op) { this->generateRewriter(op); });
    }
  }

  // Update the signature of the rewrite function.
  rewriterFunc.setType(builder.getFunctionType(
      llvm::to_vector<8>(rewriterFunc.front().getArgumentTypes()),
      /*results=*/{}));

  pdl_interp::FinalizeOp::create(builder, rewriter.getLoc());
}

void RewriterGen::generateRewriter(pdl::ApplyNativeRewriteOp rewriteOp) {
  SmallVector<Value, 2> arguments;
  for (Value argument : rewriteOp.getArgs())
    arguments.push_back(mapRewriteValue(argument));
  auto interpOp = pdl_interp::ApplyRewriteOp::create(
      builder, rewriteOp.getLoc(), rewriteOp.getResultTypes(),
      rewriteOp.getNameAttr(), arguments);
  for (auto it : llvm::zip(rewriteOp.getResults(), interpOp.getResults()))
    rewriteValues[std::get<0>(it)] = std::get<1>(it);
}

void RewriterGen::generateRewriter(pdl::AttributeOp attrOp) {
  Value newAttr = pdl_interp::CreateAttributeOp::create(
      builder, attrOp.getLoc(), attrOp.getValueAttr());
  rewriteValues[attrOp] = newAttr;
}

void RewriterGen::generateRewriter(pdl::EraseOp eraseOp) {
  pdl_interp::EraseOp::create(builder, eraseOp.getLoc(),
                              mapRewriteValue(eraseOp.getOpValue()));
}

void RewriterGen::generateRewriter(pdl::OperationOp operationOp) {
  SmallVector<Value, 4> operands;
  for (Value operand : operationOp.getOperandValues())
    operands.push_back(mapRewriteValue(operand));

  SmallVector<Value, 4> attributes;
  for (Value attr : operationOp.getAttributeValues())
    attributes.push_back(mapRewriteValue(attr));

  bool hasInferredResultTypes = false;
  SmallVector<Value, 2> types;
  generateOperationResultTypeRewriter(operationOp, types,
                                      hasInferredResultTypes);

  // Create the new operation.
  Location loc = operationOp.getLoc();
  Value createdOp = pdl_interp::CreateOperationOp::create(
      builder, loc, *operationOp.getOpName(), types, hasInferredResultTypes,
      operands, attributes, operationOp.getAttributeValueNames());
  rewriteValues[operationOp.getOp()] = createdOp;

  // Generate accesses for any results that have their types constrained.
  // Handle the case where there is a single range representing all of the
  // result types.
  OperandRange resultTys = operationOp.getTypeValues();
  if (resultTys.size() == 1 && isa<pdl::RangeType>(resultTys[0].getType())) {
    Value &type = rewriteValues[resultTys[0]];
    if (!type) {
      auto results = pdl_interp::GetResultsOp::create(builder, loc, createdOp);
      type = pdl_interp::GetValueTypeOp::create(builder, loc, results);
    }
    return;
  }

  // Otherwise, populate the individual results.
  bool seenVariableLength = false;
  Type valueTy = builder.getType<pdl::ValueType>();
  Type valueRangeTy = pdl::RangeType::get(valueTy);
  for (const auto &it : llvm::enumerate(resultTys)) {
    Value &type = rewriteValues[it.value()];
    if (type)
      continue;
    bool isVariadic = isa<pdl::RangeType>(it.value().getType());
    seenVariableLength |= isVariadic;

    // After a variable length result has been seen, we need to use result
    // groups because the exact index of the result is not statically known.
    Value resultVal;
    if (seenVariableLength)
      resultVal = pdl_interp::GetResultsOp::create(
          builder, loc, isVariadic ? valueRangeTy : valueTy, createdOp,
          it.index());
    else
      resultVal = pdl_interp::GetResultOp::create(builder, loc, valueTy,
                                                  createdOp, it.index());
    type = pdl_interp::GetValueTypeOp::create(builder, loc, resultVal);
  }
}

void RewriterGen::generateRewriter(pdl::RangeOp rangeOp) {
  SmallVector<Value, 4> replOperands;
  for (Value operand : rangeOp.getArguments())
    replOperands.push_back(mapRewriteValue(operand));
  rewriteValues[rangeOp] = pdl_interp::CreateRangeOp::create(
      builder, rangeOp.getLoc(), rangeOp.getType(), replOperands);
}

void RewriterGen::generateRewriter(pdl::ReplaceOp replaceOp) {
  SmallVector<Value, 4> replOperands;

  // If the replacement was another operation, get its results. `pdl` allows
  // for using an operation for simplicitly, but the interpreter isn't as
  // user facing.
  if (Value replOp = replaceOp.getReplOperation()) {
    // Don't use replace if we know the replaced operation has no results.
    auto opOp = replaceOp.getOpValue().getDefiningOp<pdl::OperationOp>();
    if (!opOp || !opOp.getTypeValues().empty()) {
      replOperands.push_back(pdl_interp::GetResultsOp::create(
          builder, replOp.getLoc(), mapRewriteValue(replOp)));
    }
  } else {
    for (Value operand : replaceOp.getReplValues())
      replOperands.push_back(mapRewriteValue(operand));
  }

  // If there are no replacement values, just create an erase instead.
  if (replOperands.empty()) {
    pdl_interp::EraseOp::create(builder, replaceOp.getLoc(),
                                mapRewriteValue(replaceOp.getOpValue()));
    return;
  }

  pdl_interp::ReplaceOp::create(builder, replaceOp.getLoc(),
                                mapRewriteValue(replaceOp.getOpValue()),
                                replOperands);
}

void RewriterGen::generateRewriter(pdl::ResultOp resultOp) {
  rewriteValues[resultOp] = pdl_interp::GetResultOp::create(
      builder, resultOp.getLoc(), builder.getType<pdl::ValueType>(),
      mapRewriteValue(resultOp.getParent()), resultOp.getIndex());
}

void RewriterGen::generateRewriter(pdl::ResultsOp resultOp) {
  rewriteValues[resultOp] = pdl_interp::GetResultsOp::create(
      builder, resultOp.getLoc(), resultOp.getType(),
      mapRewriteValue(resultOp.getParent()), resultOp.getIndex());
}

void RewriterGen::generateRewriter(pdl::TypeOp typeOp) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (TypeAttr typeAttr = typeOp.getConstantTypeAttr()) {
    rewriteValues[typeOp] =
        pdl_interp::CreateTypeOp::create(builder, typeOp.getLoc(), typeAttr);
  }
}

void RewriterGen::generateRewriter(pdl::TypesOp typeOp) {
  // If the type isn't constant, the users (e.g. OperationOp) will resolve this
  // type.
  if (ArrayAttr typeAttr = typeOp.getConstantTypesAttr()) {
    rewriteValues[typeOp] = pdl_interp::CreateTypesOp::create(
        builder, typeOp.getLoc(), typeOp.getType(), typeAttr);
  }
}

void RewriterGen::generateOperationResultTypeRewriter(
    pdl::OperationOp op, SmallVectorImpl<Value> &types,
    bool &hasInferredResultTypes) {
  Block *rewriterBlock = op->getBlock();

  // Try to handle resolution for each of the result types individually. This is
  // preferred over type inferrence because it will allow for us to use existing
  // types directly, as opposed to trying to rebuild the type list.
  OperandRange resultTypeValues = op.getTypeValues();
  auto tryResolveResultTypes = [&] {
    types.reserve(resultTypeValues.size());
    for (const auto &it : llvm::enumerate(resultTypeValues)) {
      Value resultType = it.value();

      // Check for an already translated value.
      if (Value existingRewriteValue = rewriteValues.lookup(resultType)) {
        types.push_back(existingRewriteValue);
        continue;
      }

      // Check for an input from the matcher.
      if (resultType.getDefiningOp()->getBlock() != rewriterBlock) {
        types.push_back(mapRewriteValue(resultType));
        continue;
      }

      // Otherwise, we couldn't infer the result types. Bail out here to see if
      // we can infer the types for this operation from another way.
      types.clear();
      return failure();
    }
    return success();
  };
  if (!resultTypeValues.empty() && succeeded(tryResolveResultTypes()))
    return;

  // Otherwise, check if the operation has type inference support itself.
  if (op.hasTypeInference()) {
    hasInferredResultTypes = true;
    return;
  }

  // Look for an operation that was replaced by `op`. The result types will be
  // inferred from the results that were replaced.
  for (OpOperand &use : op.getOp().getUses()) {
    // Check that the use corresponds to a ReplaceOp and that it is the
    // replacement value, not the operation being replaced.
    pdl::ReplaceOp replOpUser = dyn_cast<pdl::ReplaceOp>(use.getOwner());
    if (!replOpUser || use.getOperandNumber() == 0)
      continue;
    // Make sure the replaced operation was defined before this one. PDL
    // rewrites only have single block regions, so if the op isn't in the
    // rewriter block (i.e. the current block of the operation) we already know
    // it dominates (i.e. it's in the matcher).
    Value replOpVal = replOpUser.getOpValue();
    Operation *replacedOp = replOpVal.getDefiningOp();
    if (replacedOp->getBlock() == rewriterBlock &&
        !replacedOp->isBeforeInBlock(op))
      continue;

    Value replacedOpResults = pdl_interp::GetResultsOp::create(
        builder, replacedOp->getLoc(), mapRewriteValue(replOpVal));
    types.push_back(pdl_interp::GetValueTypeOp::create(
        builder, replacedOp->getLoc(), replacedOpResults));
    return;
  }

  // If the types could not be inferred from any context and there weren't any
  // explicit result types, assume the user actually meant for the operation to
  // have no results.
  if (resultTypeValues.empty())
    return;

  // The verifier asserts that the result types of each pdl.getOperation can be
  // inferred. If we reach here, there is a bug either in the logic above or
  // in the verifier for pdl.getOperation.
  op->emitOpError() << "unable to infer result type for operation";
  llvm_unreachable("unable to infer result type for operation");
}

SymbolRefAttr mlir::pdl_to_pdl_interp::generatePatternRewriter(
    pdl::PatternOp pattern, ModuleOp rewriterModule,
    SymbolTable &rewriterSymbolTable, OpBuilder &builder,
    SmallVectorImpl<Value> &usedMatchValues) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(rewriterModule.getBody());

  // Get the pattern name if available, otherwise use default.
  StringRef rewriterName = "pdl_generated_rewriter";
  if (auto symName = pattern.getSymName())
    rewriterName = symName.value();
  auto rewriterFunc = pdl_interp::FuncOp::create(
      builder, pattern.getLoc(), rewriterName, builder.getFunctionType({}, {}));
  rewriterSymbolTable.insert(rewriterFunc);

  RewriterGen gen(builder, usedMatchValues);
  gen.generate(pattern, rewriterFunc);

  return SymbolRefAttr::get(
      builder.getContext(),
      pdl_interp::PDLInterpDialect::getRewriterModuleName(),
      SymbolRefAttr::get(rewriterFunc));
}
