//===- TranslateToCpp.cpp - Emit C++ matchers from match ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/MatchToCpp/MatchToCpp.h"

#include "mlir/Dialect/Match/IR/Match.h"
#include "mlir/Dialect/Match/IR/MatchOps.h"
#include "mlir/Dialect/Match/IR/MatchTypes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
using namespace mlir::match;

namespace {
class MatcherCppEmitter {
public:
  MatcherCppEmitter(raw_ostream &os, const OpInfoRegistry &registry)
      : os(os), registry(registry) {}

  LogicalResult emitModule(Operation *root);

private:
  std::string getName(Value v) {
    auto it = names.find(v);
    assert(it != names.end() && "value used before being named");
    return it->second;
  }
  std::string declareName(Value v) {
    std::string name = ("v" + Twine(counter++)).str();
    names[v] = name;
    return name;
  }
  void mapName(Value v, StringRef name) { names[v] = name.str(); }
  std::string freshId(StringRef prefix) {
    return (prefix + Twine(counter++)).str();
  }

  std::string cppType(Type t);
  std::string typeLiteral(Type t);
  std::string attrLiteral(Attribute a);
  void emitTypeLiterals(ArrayAttr types);

  raw_ostream &indent() { return os.indent(indentLevel * 2); }

  LogicalResult emitMatcher(MatcherOp matcher, unsigned idx);

  LogicalResult emitOpSequence(ArrayRef<Operation *> ops, StringRef failAction);

  LogicalResult emitSingle(Operation *op, StringRef failAction);

  void bindConcrete(Value op, const OpInfo &info, bool alreadyKnownName,
                    StringRef failAction);

  LogicalResult emitTry(TryOp op);
  LogicalResult emitForEach(ForEachOp op);
  LogicalResult emitSwitchOpName(SwitchOpNameOp op, StringRef failAction);
  LogicalResult emitSwitchType(SwitchTypeOp op, StringRef failAction);
  LogicalResult emitSuccess(SuccessOp op, StringRef failAction);

  static SmallVector<Operation *> opsOf(Region &region) {
    SmallVector<Operation *> ops;
    for (Operation &op : region.front())
      ops.push_back(&op);
    return ops;
  }

  struct ConcreteBinding {
    const OpInfo *info;
    std::string var;
  };

  const ConcreteBinding *concreteOf(Value v) const {
    auto it = concrete.find(v);
    return it == concrete.end() ? nullptr : &it->second;
  }

  raw_ostream &os;
  const OpInfoRegistry &registry;
  DenseMap<Value, std::string> names;
  DenseMap<Value, ConcreteBinding> concrete;
  DenseSet<Value> knownNonNull;
  unsigned counter = 0;
  unsigned indentLevel = 0;
  bool needsGroupHelper = false;
  std::map<std::string, std::string> rewriterDecls;
  SmallVector<std::string> structNames;
};

} // namespace

std::string MatcherCppEmitter::cppType(Type t) {
  if (auto opt = dyn_cast<OptionalType>(t))
    return cppType(opt.getInnerType());
  if (isa<pdl::OperationType>(t))
    return "::mlir::Operation *";
  if (isa<pdl::ValueType>(t))
    return "::mlir::Value";
  if (isa<pdl::TypeType>(t))
    return "::mlir::Type";
  if (isa<pdl::AttributeType>(t))
    return "::mlir::Attribute";
  if (auto r = dyn_cast<pdl::RangeType>(t)) {
    Type el = r.getElementType();
    if (isa<pdl::ValueType>(el))
      return "::mlir::ValueRange";
    if (isa<pdl::TypeType>(el))
      return "::mlir::TypeRange";
    if (isa<pdl::OperationType>(el))
      return "::llvm::SmallVector<::mlir::Operation *>";
  }
  return "auto";
}

std::string MatcherCppEmitter::typeLiteral(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << "::mlir::IntegerType::get(op->getContext(), " << it.getWidth();
    if (it.isSigned())
      ss << ", ::mlir::IntegerType::Signed";
    else if (it.isUnsigned())
      ss << ", ::mlir::IntegerType::Unsigned";
    ss << ")";
    return str;
  }
  if (isa<IndexType>(t))
    return "::mlir::IndexType::get(op->getContext())";
  if (isa<BFloat16Type>(t))
    return "::mlir::BFloat16Type::get(op->getContext())";
  if (isa<Float16Type>(t))
    return "::mlir::Float16Type::get(op->getContext())";
  if (isa<Float32Type>(t))
    return "::mlir::Float32Type::get(op->getContext())";
  if (isa<Float64Type>(t))
    return "::mlir::Float64Type::get(op->getContext())";

  std::string str;
  llvm::raw_string_ostream ss(str);
  t.print(ss);
  return ("::mlir::parseType(\"" + StringRef(ss.str()) +
          "\", op->getContext())")
      .str();
}

std::string MatcherCppEmitter::attrLiteral(Attribute a) {
  if (auto ia = dyn_cast<IntegerAttr>(a)) {
    APInt v = ia.getValue();
    if (v.getBitWidth() <= 64) {
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss << "::mlir::IntegerAttr::get(" << typeLiteral(ia.getType()) << ", ";
      if (ia.getType().isUnsignedInteger())
        ss << v.getZExtValue() << "u";
      else if (v.getBitWidth() == 1)
        ss << v.getZExtValue();
      else
        ss << v.getSExtValue();
      ss << ")";
      return str;
    }
  }
  if (auto fa = dyn_cast<FloatAttr>(a)) {
    if (fa.getValue().isFinite()) {
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss << "::mlir::FloatAttr::get(" << typeLiteral(fa.getType()) << ", "
         << llvm::format("%.17g", fa.getValueAsDouble()) << ")";
      return str;
    }
  }
  if (auto sa = dyn_cast<StringAttr>(a)) {
    if (isa<NoneType>(sa.getType())) {
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss << "::mlir::StringAttr::get(op->getContext(), \"";
      llvm::printEscapedString(sa.getValue(), ss);
      ss << "\")";
      return str;
    }
  }
  if (auto ta = dyn_cast<TypeAttr>(a))
    return (Twine("::mlir::TypeAttr::get(") + typeLiteral(ta.getValue()) + ")")
        .str();
  if (isa<UnitAttr>(a))
    return "::mlir::UnitAttr::get(op->getContext())";

  std::string str;
  llvm::raw_string_ostream ss(str);
  a.print(ss);
  return ("::mlir::parseAttribute(\"" + StringRef(ss.str()) +
          "\", op->getContext())")
      .str();
}

void MatcherCppEmitter::emitTypeLiterals(ArrayAttr types) {
  llvm::interleaveComma(types, os, [&](Attribute a) {
    os << typeLiteral(cast<TypeAttr>(a).getValue());
  });
}

void MatcherCppEmitter::bindConcrete(Value op, const OpInfo &info,
                                     bool alreadyKnownName,
                                     StringRef failAction) {
  std::string casted = freshId("castedOp");
  StringRef cls = info.cppClassName;
  if (alreadyKnownName) {
    indent() << cls << " " << casted << " = ::llvm::cast<" << cls << ">("
             << getName(op) << ");\n";
    indent() << "(void)" << casted << ";\n";
  } else {
    indent() << cls << " " << casted << " = ::llvm::dyn_cast<" << cls << ">("
             << getName(op) << ");\n";
    indent() << "if (!" << casted << ")\n";
    indent() << "  " << failAction << "\n";
  }
  concrete[op] = ConcreteBinding{&info, casted};
}

LogicalResult MatcherCppEmitter::emitSuccess(SuccessOp op,
                                             StringRef failAction) {
  StringRef leaf = op.getRewriter().getLeafReference().getValue();
  std::string fnName = ("rewrite_" + leaf).str();

  std::string decl;
  {
    llvm::raw_string_ostream ds(decl);
    ds << "::llvm::LogicalResult " << fnName
       << "(::mlir::PatternRewriter &rewriter";
    for (Value in : op.getInputs())
      ds << ", " << cppType(in.getType());
    ds << ");";
  }
  rewriterDecls[fnName] = decl;

  indent() << "// match.success @" << leaf << " benefit(" << op.getBenefit()
           << ")\n";
  indent() << "rewriter.setInsertionPoint(op);\n";
  indent() << "if (::mlir::succeeded(" << fnName << "(rewriter";
  for (Value in : op.getInputs())
    os << ", " << getName(in);
  os << ")))\n";
  indent() << "  return ::mlir::success();\n";
  indent() << failAction << "\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitTry(TryOp op) {
  std::string fn = freshId("attempt");
  indent() << "auto " << fn << " = [&]() -> ::llvm::LogicalResult {\n";
  ++indentLevel;
  if (failed(emitOpSequence(opsOf(op.getBody()), "return ::mlir::failure();")))
    return failure();
  --indentLevel;
  indent() << "};\n";
  indent() << "if (::mlir::succeeded(" << fn
           << "())) return ::mlir::success();\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitForEach(ForEachOp op) {
  std::string elem = declareName(op.getLoopVariable());
  indent() << "for (" << cppType(op.getLoopVariable().getType()) << " " << elem
           << " : " << getName(op.getRange()) << ") {\n";
  ++indentLevel;
  LogicalResult body = emitOpSequence(opsOf(op.getBody()), "continue;");
  --indentLevel;
  indent() << "}\n";
  return body;
}

LogicalResult MatcherCppEmitter::emitSwitchOpName(SwitchOpNameOp op,
                                                  StringRef failAction) {
  ArrayAttr cases = op.getCaseNames();
  SmallVector<const OpInfo *> caseInfos;
  bool anyGeneric = false;
  for (Attribute c : cases) {
    const OpInfo *info = registry.lookup(cast<StringAttr>(c).getValue());
    caseInfos.push_back(info);
    anyGeneric |= !info;
  }

  indent() << "{\n";
  ++indentLevel;
  std::string nameVar;
  if (anyGeneric) {
    nameVar = freshId("name");
    indent() << "::llvm::StringRef " << nameVar << " = " << getName(op.getOp())
             << "->getName().getStringRef();\n";
  }
  for (auto [j, region] : llvm::enumerate(op.getCaseRegions())) {
    auto caseName = cast<StringAttr>(cases[j]).getValue();
    indent() << (j == 0 ? "if" : "else if") << " (";
    if (const OpInfo *info = caseInfos[j])
      os << "::llvm::isa<" << info->cppClassName << ">(" << getName(op.getOp())
         << ")";
    else
      os << nameVar << " == \"" << caseName << "\"";
    os << ") {\n";
    ++indentLevel;
    bool bound = false;
    bool hadPrev = false;
    ConcreteBinding saved{};
    if (auto prev = concrete.find(op.getOp()); prev != concrete.end()) {
      hadPrev = true;
      saved = prev->second;
    }
    if (const OpInfo *info = caseInfos[j]) {
      bindConcrete(op.getOp(), *info, /*alreadyKnownName=*/true, failAction);
      bound = true;
    }
    if (failed(emitOpSequence(opsOf(region), failAction)))
      return failure();
    if (bound) {
      if (hadPrev)
        concrete[op.getOp()] = saved;
      else
        concrete.erase(op.getOp());
    }
    --indentLevel;
    indent() << "}\n";
  }
  --indentLevel;
  indent() << "}\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitSwitchType(SwitchTypeOp op,
                                                StringRef failAction) {
  std::string tyVar = freshId("ty");
  indent() << "{\n";
  ++indentLevel;
  indent() << "::mlir::Type " << tyVar << " = " << getName(op.getTypeValue())
           << ";\n";
  ArrayAttr cases = op.getCaseTypes();
  for (auto [j, region] : llvm::enumerate(op.getCaseRegions())) {
    Type caseType = cast<TypeAttr>(cases[j]).getValue();
    indent() << (j == 0 ? "if" : "else if") << " (" << tyVar
             << " == " << typeLiteral(caseType) << ") {\n";
    ++indentLevel;
    if (failed(emitOpSequence(opsOf(region), failAction)))
      return failure();
    --indentLevel;
    indent() << "}\n";
  }
  --indentLevel;
  indent() << "}\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitSingle(Operation *op,
                                            StringRef failAction) {
  auto fail = [&](const Twine &cond) {
    indent() << "if (" << cond << ")\n";
    indent() << "  " << failAction << "\n";
  };

  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<TryOp>([&](TryOp o) { return emitTry(o); })
      .Case<ForEachOp>([&](ForEachOp o) { return emitForEach(o); })
      .Case<SwitchOpNameOp>(
          [&](SwitchOpNameOp o) { return emitSwitchOpName(o, failAction); })
      .Case<SwitchTypeOp>(
          [&](SwitchTypeOp o) { return emitSwitchType(o, failAction); })
      .Case<SuccessOp>([&](SuccessOp o) { return emitSuccess(o, failAction); })
      .Case<ConstantAttributeOp>([&](ConstantAttributeOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::mlir::Attribute " << v << " = "
                 << attrLiteral(o.getValue()) << ";\n";
        return success();
      })
      .Case<ConstantTypeOp>([&](ConstantTypeOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::mlir::Type " << v << " = " << typeLiteral(o.getValue())
                 << ";\n";
        return success();
      })
      .Case<ConstantTypesOp>([&](ConstantTypesOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::llvm::SmallVector<::mlir::Type> " << v << "{";
        emitTypeLiterals(o.getValue());
        os << "};\n";
        return success();
      })
      .Case<GetOperandOp>([&](GetOperandOp o) {
        std::string opName = getName(o.getOp());
        std::string v = declareName(o.getResult());
        if (const ConcreteBinding *b = concreteOf(o.getOp())) {
          if (auto fixed = b->info->getFixedNumOperands();
              fixed && o.getIndex() < *fixed) {
            indent() << "::mlir::Value " << v << " = " << b->var
                     << ".getOperation()->getOperand(" << o.getIndex()
                     << ");\n";
            knownNonNull.insert(o.getResult());
            return success();
          }
        }
        indent() << "::mlir::Value " << v << " = (" << o.getIndex() << " < "
                 << opName << "->getNumOperands()) ? " << opName
                 << "->getOperand(" << o.getIndex() << ") : ::mlir::Value();\n";
        return success();
      })
      .Case<GetResultOp>([&](GetResultOp o) {
        std::string opName = getName(o.getOp());
        std::string v = declareName(o.getResult());
        if (const ConcreteBinding *b = concreteOf(o.getOp())) {
          if (auto fixed = b->info->getFixedNumResults();
              fixed && o.getIndex() < *fixed) {
            indent() << "::mlir::Value " << v << " = " << b->var
                     << ".getOperation()->getResult(" << o.getIndex() << ");\n";
            knownNonNull.insert(o.getResult());
            return success();
          }
        }
        indent() << "::mlir::Value " << v << " = (" << o.getIndex() << " < "
                 << opName << "->getNumResults()) ? " << opName
                 << "->getResult(" << o.getIndex() << ") : ::mlir::Value();\n";
        return success();
      })
      .Case<GetOperandsOp>([&](GetOperandsOp o) {
        std::string opn = getName(o.getOp());
        std::string v = declareName(o.getResult());
        if (const ConcreteBinding *b = concreteOf(o.getOp())) {
          if (o.getIndex() && *o.getIndex() < b->info->numOperandGroups) {
            indent() << "auto " << v << " = " << b->var << ".getODSOperands("
                     << *o.getIndex() << ");\n";
            knownNonNull.insert(o.getResult());
            return success();
          }
          if (!o.getIndex()) {
            indent() << "auto " << v << " = " << b->var
                     << ".getOperation()->getOperands();\n";
            knownNonNull.insert(o.getResult());
            return success();
          }
        }
        if (o.getIndex()) {
          needsGroupHelper = true;
          indent() << "::std::optional<::mlir::ValueRange> " << v
                   << " = __pdl_get_group(" << opn << ", " << *o.getIndex()
                   << ", ::mlir::ValueRange(" << opn
                   << "->getOperands()), \"operandSegmentSizes\", " << opn
                   << "->hasTrait<::mlir::OpTrait::AttrSizedOperandSegments>()"
                      ");\n";
        } else {
          indent() << "::std::optional<::mlir::ValueRange> " << v
                   << " = ::mlir::ValueRange(" << opn << "->getOperands());\n";
        }
        return success();
      })
      .Case<GetResultsOp>([&](GetResultsOp o) {
        std::string opn = getName(o.getOp());
        std::string v = declareName(o.getResult());
        if (const ConcreteBinding *b = concreteOf(o.getOp())) {
          if (o.getIndex() && *o.getIndex() < b->info->numResultGroups) {
            indent() << "auto " << v << " = " << b->var << ".getODSResults("
                     << *o.getIndex() << ");\n";
            knownNonNull.insert(o.getResult());
            return success();
          }
          if (!o.getIndex()) {
            indent() << "auto " << v << " = " << b->var
                     << ".getOperation()->getResults();\n";
            knownNonNull.insert(o.getResult());
            return success();
          }
        }
        if (o.getIndex()) {
          needsGroupHelper = true;
          indent() << "::std::optional<::mlir::ValueRange> " << v
                   << " = __pdl_get_group(" << opn << ", " << *o.getIndex()
                   << ", ::mlir::ValueRange(" << opn
                   << "->getResults()), \"resultSegmentSizes\", " << opn
                   << "->hasTrait<::mlir::OpTrait::AttrSizedResultSegments>()"
                      ");\n";
        } else {
          indent() << "::std::optional<::mlir::ValueRange> " << v
                   << " = ::mlir::ValueRange(" << opn << "->getResults());\n";
        }
        return success();
      })
      .Case<GetAttributeOp>([&](GetAttributeOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::mlir::Attribute " << v << " = " << getName(o.getOp())
                 << "->getAttr(\"" << o.getName() << "\");\n";
        return success();
      })
      .Case<GetDefiningOpOp>([&](GetDefiningOpOp o) {
        std::string v = declareName(o.getResult());
        if (isa<pdl::RangeType>(o.getValue().getType())) {
          std::string r = getName(o.getValue());
          indent() << "::mlir::Operation *" << v << " = " << r
                   << ".empty() ? nullptr : (*" << r
                   << ".begin()).getDefiningOp();\n";
        } else {
          indent() << "::mlir::Operation *" << v << " = "
                   << getName(o.getValue()) << ".getDefiningOp();\n";
        }
        return success();
      })
      .Case<GetValueTypeOp>([&](GetValueTypeOp o) {
        std::string v = declareName(o.getResult());
        if (isa<pdl::RangeType>(o.getValue().getType()))
          indent() << "auto " << v << " = " << getName(o.getValue())
                   << ".getTypes();\n";
        else
          indent() << "::mlir::Type " << v << " = " << getName(o.getValue())
                   << ".getType();\n";
        return success();
      })
      .Case<GetAttributeTypeOp>([&](GetAttributeTypeOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::mlir::Type " << v
                 << " = ::llvm::cast<::mlir::TypedAttr>("
                 << getName(o.getAttribute()) << ").getType();\n";
        return success();
      })
      .Case<GetUsersOp>([&](GetUsersOp o) {
        std::string v = declareName(o.getResult());
        indent() << "auto " << v << " = " << getName(o.getValue())
                 << ".getUsers();\n";
        return success();
      })
      .Case<ExtractOp>([&](ExtractOp o) {
        std::string r = getName(o.getRange());
        std::string v = declareName(o.getResult());
        Type elemTy = o.getResult().getType();
        std::string ty = cppType(elemTy);
        if (isa<pdl::OperationType>(elemTy))
          indent() << ty << " " << v << " = (" << o.getIndex() << " < " << r
                   << ".size()) ? " << r << "[" << o.getIndex()
                   << "] : nullptr;\n";
        else
          indent() << ty << " " << v << " = (" << o.getIndex() << " < " << r
                   << ".size()) ? " << ty << "(" << r << "[" << o.getIndex()
                   << "]) : " << ty << "();\n";
        return success();
      })
      .Case<IsNotNullOp>([&](IsNotNullOp o) {
        std::string name = getName(o.getOptionalValue());
        if (knownNonNull.contains(o.getOptionalValue())) {
          mapName(o.getUnwrapped(), name);
          return success();
        }
        Type inner =
            cast<OptionalType>(o.getOptionalValue().getType()).getInnerType();
        if (isa<pdl::RangeType>(inner)) {
          fail("!" + name);
          std::string u = declareName(o.getUnwrapped());
          indent() << cppType(inner) << " " << u << " = *" << name << ";\n";
        } else {
          mapName(o.getUnwrapped(), name);
          fail("!" + name);
        }
        return success();
      })
      .Case<HasNameOp>([&](HasNameOp o) {
        if (const OpInfo *info = registry.lookup(o.getName())) {
          bindConcrete(o.getOp(), *info, /*alreadyKnownName=*/false,
                       failAction);
          return success();
        }
        fail(getName(o.getOp()) + "->getName().getStringRef() != \"" +
             o.getName().str() + "\"");
        return success();
      })
      .Case<EqualOp>([&](EqualOp o) {
        if (isa<pdl::RangeType>(o.getLhs().getType()))
          fail("!::llvm::equal(" + getName(o.getLhs()) + ", " +
               getName(o.getRhs()) + ")");
        else
          fail(getName(o.getLhs()) + " != " + getName(o.getRhs()));
        return success();
      })
      .Case<HasTypeOp>([&](HasTypeOp o) {
        fail(getName(o.getTypeValue()) +
             " != " + typeLiteral(o.getConstantType()));
        return success();
      })
      .Case<HasTypesOp>([&](HasTypesOp o) {
        std::string arr = freshId("expected");
        indent() << "{\n";
        ++indentLevel;
        indent() << "::mlir::Type " << arr << "[] = {";
        emitTypeLiterals(o.getConstantTypes());
        os << "};\n";
        indent() << "if (" << getName(o.getTypes())
                 << ".size() != " << o.getConstantTypes().size()
                 << " || !::llvm::equal(" << getName(o.getTypes())
                 << ", ::llvm::ArrayRef<::mlir::Type>(" << arr << ")))\n";
        indent() << "  " << failAction << "\n";
        --indentLevel;
        indent() << "}\n";
        return success();
      })
      .Case<HasAttrValueOp>([&](HasAttrValueOp o) {
        fail(getName(o.getAttribute()) + " != " + attrLiteral(o.getValue()));
        return success();
      })
      .Case<CheckOperandCountOp>([&](CheckOperandCountOp o) {
        if (const ConcreteBinding *b = concreteOf(o.getOp()))
          if (auto fixed = b->info->getFixedNumOperands())
            if (o.getAtLeast() ? *fixed >= o.getCount()
                               : *fixed == o.getCount())
              return success();
        fail(Twine(getName(o.getOp())) + "->getNumOperands() " +
             (o.getAtLeast() ? "< " : "!= ") + Twine(o.getCount()));
        return success();
      })
      .Case<CheckResultCountOp>([&](CheckResultCountOp o) {
        if (const ConcreteBinding *b = concreteOf(o.getOp()))
          if (auto fixed = b->info->getFixedNumResults())
            if (o.getAtLeast() ? *fixed >= o.getCount()
                               : *fixed == o.getCount())
              return success();
        fail(Twine(getName(o.getOp())) + "->getNumResults() " +
             (o.getAtLeast() ? "< " : "!= ") + Twine(o.getCount()));
        return success();
      })
      .Case<ApplyNativeConstraintOp>([&](ApplyNativeConstraintOp o)
                                         -> LogicalResult {
        if (!o.getConstraintResults().empty())
          return o.emitError("result-producing native constraints are not yet "
                             "supported by the C++ matcher emitter");
        for (Value arg : o.getArgs())
          if (isa<pdl::RangeType>(arg.getType()))
            return o.emitError("range arguments to native constraints are not "
                               "yet supported by the C++ matcher emitter");
        std::string argsVar = freshId("cargs");
        indent() << "{\n";
        ++indentLevel;
        indent() << "::mlir::PDLValue " << argsVar << "[] = {";
        llvm::interleaveComma(o.getArgs(), os,
                              [&](Value v) { os << getName(v); });
        os << "};\n";
        indent() << "::mlir::PDLResultList " << argsVar << "Res(0);\n";
        indent() << "if ("
                 << (o.getIsNegated() ? "::mlir::succeeded" : "::mlir::failed")
                 << "(" << o.getName() << "(rewriter, " << argsVar << "Res, "
                 << argsVar << ")))\n";
        indent() << "  " << failAction << "\n";
        --indentLevel;
        indent() << "}\n";
        rewriterDecls[o.getName().str()] =
            "::llvm::LogicalResult " + o.getName().str() +
            "(::mlir::PatternRewriter &, ::mlir::PDLResultList &, "
            "::llvm::ArrayRef<::mlir::PDLValue>);";
        return success();
      })
      .Default([&](Operation *o) {
        return o->emitError("unsupported match op in C++ matcher emitter");
      });
}

LogicalResult MatcherCppEmitter::emitOpSequence(ArrayRef<Operation *> ops,
                                                StringRef failAction) {
  for (Operation *op : ops)
    if (failed(emitSingle(op, failAction)))
      return failure();

  if (!ops.empty() && isa<SuccessOp>(ops.back()))
    return success();
  indent() << failAction << "\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitMatcher(MatcherOp matcher, unsigned idx) {
  names.clear();
  concrete.clear();
  knownNonNull.clear();
  counter = 0;

  Block &body = matcher.getBodyRegion().front();
  mapName(body.getArgument(0), "op");

  uint64_t benefit = 1;
  matcher.walk([&](SuccessOp s) {
    benefit = std::max<uint64_t>(benefit, s.getBenefit());
  });

  std::string structName = ("GeneratedMatcher_" + Twine(idx)).str();
  structNames.push_back(structName);

  os << "namespace {\n";
  os << "struct " << structName << " : public ::mlir::RewritePattern {\n";
  os << "  " << structName << "(::mlir::MLIRContext *context)\n";
  os << "      : ::mlir::RewritePattern(::mlir::Pattern::MatchAnyOpTypeTag(), "
     << benefit << ", context) {}\n";
  os << "  ::llvm::LogicalResult\n";
  os << "  matchAndRewrite(::mlir::Operation *op,\n";
  os << "                  ::mlir::PatternRewriter &rewriter) const override "
        "{\n";
  indentLevel = 2;
  if (failed(emitOpSequence(opsOf(matcher.getBodyRegion()),
                            "return ::mlir::failure();")))
    return failure();
  indentLevel = 0;
  os << "  }\n";
  os << "};\n";
  os << "} // namespace\n\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitModule(Operation *root) {
  SmallVector<MatcherOp> matchers;
  root->walk([&](MatcherOp m) { matchers.push_back(m); });

  std::string structs;
  {
    llvm::raw_string_ostream structOs(structs);
    MatcherCppEmitter sub(structOs, registry);
    for (auto [idx, m] : llvm::enumerate(matchers))
      if (failed(sub.emitMatcher(m, idx)))
        return failure();
    rewriterDecls = std::move(sub.rewriterDecls);
    structNames = std::move(sub.structNames);
    needsGroupHelper = sub.needsGroupHelper;
  }

  os << "//===- Generated by mlir-match-to-cpp. DO NOT EDIT. -===//\n";
  os << "//\n";
  os << "// The user must provide the `rewrite_*` hooks declared below.\n";
  os << "//===--------------------------------------------------------------"
        "------===//\n\n";
  os << "#include \"mlir/IR/BuiltinAttributes.h\"\n";
  os << "#include \"mlir/IR/OpDefinition.h\"\n";
  os << "#include \"mlir/IR/Operation.h\"\n";
  os << "#include \"mlir/IR/PatternMatch.h\"\n";
  os << "#include \"mlir/Parser/Parser.h\"\n";
  os << "#include \"llvm/ADT/STLExtras.h\"\n";
  os << "#include <optional>\n";
  os << "\n";

  if (needsGroupHelper) {
    os << "static ::std::optional<::mlir::ValueRange>\n"
          "__pdl_get_group(::mlir::Operation *op, unsigned index,\n"
          "                ::mlir::ValueRange values, ::llvm::StringRef "
          "segmentAttr,\n"
          "                bool hasSegments) {\n"
          "  if (hasSegments) {\n"
          "    auto seg = "
          "op->getAttrOfType<::mlir::DenseI32ArrayAttr>(segmentAttr);\n"
          "    if (!seg || (unsigned)seg.asArrayRef().size() <= index)\n"
          "      return ::std::nullopt;\n"
          "    ::llvm::ArrayRef<int32_t> segs = seg.asArrayRef();\n"
          "    unsigned start = 0;\n"
          "    for (unsigned i = 0; i < index; ++i)\n"
          "      start += segs[i];\n"
          "    return values.slice(start, segs[index]);\n"
          "  }\n"
          "  if (values.size() >= index)\n"
          "    return values.drop_front(index);\n"
          "  return ::std::nullopt;\n"
          "}\n\n";
  }

  if (!rewriterDecls.empty()) {
    os << "// Hooks the user must define and link:\n";
    for (auto &kv : rewriterDecls)
      os << kv.second << "\n";
    os << "\n";
  }

  os << structs;

  os << "void populateGeneratedPatterns(::mlir::RewritePatternSet &set) {\n";
  if (!structNames.empty()) {
    os << "  set.add<";
    llvm::interleaveComma(structNames, os);
    os << ">(set.getContext());\n";
  }
  os << "}\n";
  return success();
}

LogicalResult mlir::match::translateToCpp(Operation *op, raw_ostream &os,
                                          const OpInfoRegistry &registry) {
  MatcherCppEmitter emitter(os, registry);
  return emitter.emitModule(op);
}

LogicalResult mlir::match::translateToCpp(Operation *op, raw_ostream &os) {
  static const OpInfoRegistry emptyRegistry;
  return translateToCpp(op, os, emptyRegistry);
}
