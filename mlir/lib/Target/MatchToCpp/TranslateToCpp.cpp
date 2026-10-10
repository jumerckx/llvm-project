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
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace mlir;
using namespace mlir::match;

static std::string sanitizeIdentifier(StringRef name) {
  std::string result;
  for (char c : name)
    result.push_back(llvm::isAlnum(c) ? c : '_');
  return result;
}

static bool isRangeType(Type type) { return isa<pdl::RangeType>(type); }

namespace {
class MatcherCppEmitter {
public:
  MatcherCppEmitter(raw_ostream &os, const OpInfoRegistry &registry)
      : os(os), registry(registry) {}

  LogicalResult emitModule(Operation *root, StringRef cppNamespace);

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
  void resetState() {
    names.clear();
    concrete.clear();
    knownNonNull.clear();
    locOps.clear();
    counter = 0;
  }

  std::string cppType(Type t);
  std::string cppStorageType(Type t);
  std::string typeLiteral(Type t);
  std::string attrLiteral(Attribute a);
  void emitTypeLiterals(ArrayAttr types);

  raw_ostream &indent() { return os.indent(indentLevel * 2); }

  LogicalResult emitMatcher(MatcherOp matcher, unsigned idx);
  LogicalResult emitRewriter(RewriterOp rewriter, StringRef fnName);
  LogicalResult emitRewriterOp(Operation *op);

  LogicalResult emitOpSequence(ArrayRef<Operation *> ops, StringRef failAction);

  LogicalResult emitSingle(Operation *op, StringRef failAction);

  void bindConcrete(Value op, const OpInfo &info, bool alreadyKnownName,
                    StringRef failAction);

  LogicalResult emitTry(TryOp op);
  LogicalResult emitForEach(ForEachOp op);
  LogicalResult emitSwitchOpName(SwitchOpNameOp op, StringRef failAction);
  LogicalResult emitSwitchType(SwitchTypeOp op, StringRef failAction);
  LogicalResult emitSuccess(SuccessOp op);
  LogicalResult emitNativeCall(Operation *op, StringRef name, ValueRange args,
                               ValueRange results, StringRef failCondition,
                               StringRef failAction);
  void emitGetValues(Value opVal, std::optional<unsigned> index, Value result,
                     bool operands, bool optional);
  std::string locExpr();

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

  struct Scope {
    Scope(MatcherCppEmitter &emitter)
        : emitter(emitter), numLocOps(emitter.locOps.size()),
          concrete(emitter.concrete) {}
    ~Scope() {
      emitter.locOps.resize(numLocOps);
      emitter.concrete = std::move(concrete);
    }
    MatcherCppEmitter &emitter;
    size_t numLocOps;
    DenseMap<Value, ConcreteBinding> concrete;
  };

  raw_ostream &os;
  const OpInfoRegistry &registry;
  DenseMap<Value, std::string> names;
  DenseMap<Value, ConcreteBinding> concrete;
  DenseSet<Value> knownNonNull;
  SmallVector<Value> locOps;
  unsigned counter = 0;
  unsigned indentLevel = 0;
  std::string ctxExpr = "op->getContext()";
  Value rootValue;
  StringRef rootKind;
  DenseMap<Operation *, uint64_t> reachableBenefit;
  bool anyDeferred = false;
  uint64_t maxDeferredBenefit = 0;
  std::map<std::string, std::string> rewriterDecls;
  llvm::MapVector<Operation *, std::string> generatedRewriters;
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

std::string MatcherCppEmitter::cppStorageType(Type t) {
  if (auto r = dyn_cast<pdl::RangeType>(t)) {
    Type el = r.getElementType();
    if (isa<pdl::ValueType>(el))
      return "::llvm::SmallVector<::mlir::Value>";
    if (isa<pdl::TypeType>(el))
      return "::llvm::SmallVector<::mlir::Type>";
  }
  return cppType(t);
}

std::string MatcherCppEmitter::typeLiteral(Type t) {
  if (auto it = dyn_cast<IntegerType>(t)) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << "::mlir::IntegerType::get(" << ctxExpr << ", " << it.getWidth();
    if (it.isSigned())
      ss << ", ::mlir::IntegerType::Signed";
    else if (it.isUnsigned())
      ss << ", ::mlir::IntegerType::Unsigned";
    ss << ")";
    return str;
  }
  if (isa<IndexType>(t))
    return "::mlir::IndexType::get(" + ctxExpr + ")";
  if (isa<BFloat16Type>(t))
    return "::mlir::BFloat16Type::get(" + ctxExpr + ")";
  if (isa<Float16Type>(t))
    return "::mlir::Float16Type::get(" + ctxExpr + ")";
  if (isa<Float32Type>(t))
    return "::mlir::Float32Type::get(" + ctxExpr + ")";
  if (isa<Float64Type>(t))
    return "::mlir::Float64Type::get(" + ctxExpr + ")";

  std::string str;
  llvm::raw_string_ostream ss(str);
  t.print(ss);
  std::string escaped;
  llvm::raw_string_ostream es(escaped);
  llvm::printEscapedString(str, es);
  return ("::mlir::parseType(\"" + StringRef(escaped) + "\", " + ctxExpr + ")")
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
      ss << "::mlir::StringAttr::get(" << ctxExpr << ", \"";
      llvm::printEscapedString(sa.getValue(), ss);
      ss << "\")";
      return str;
    }
  }
  if (auto ta = dyn_cast<TypeAttr>(a))
    return (Twine("::mlir::TypeAttr::get(") + typeLiteral(ta.getValue()) + ")")
        .str();
  if (isa<UnitAttr>(a))
    return "::mlir::UnitAttr::get(" + ctxExpr + ")";

  std::string str;
  llvm::raw_string_ostream ss(str);
  a.print(ss);
  std::string escaped;
  llvm::raw_string_ostream es(escaped);
  llvm::printEscapedString(str, es);
  return ("::mlir::parseAttribute(\"" + StringRef(escaped) + "\", " + ctxExpr +
          ")")
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

std::string MatcherCppEmitter::locExpr() {
  if (locOps.size() == 1)
    return getName(locOps.front()) + "->getLoc()";
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << "rewriter.getFusedLoc({";
  llvm::interleaveComma(locOps, ss,
                        [&](Value v) { ss << getName(v) << "->getLoc()"; });
  ss << "})";
  return str;
}

LogicalResult MatcherCppEmitter::emitSuccess(SuccessOp op) {
  StringRef leaf = op.getRewriter().getLeafReference().getValue();
  std::string fnName = "rewrite_" + sanitizeIdentifier(leaf);

  auto rewriterOp = dyn_cast_or_null<RewriterOp>(
      SymbolTable::lookupNearestSymbolFrom(op, op.getRewriter()));
  if (rewriterOp) {
    unsigned numArgs = rewriterOp.getBody().getNumArguments();
    if (numArgs != op.getInputs().size())
      return op.emitError("rewriter @")
             << leaf << " expects " << numArgs
             << " inputs, but the success provides " << op.getInputs().size();
    generatedRewriters.insert({rewriterOp, fnName});
  } else {
    std::string decl;
    llvm::raw_string_ostream ds(decl);
    ds << "::llvm::LogicalResult " << fnName
       << "(::mlir::PatternRewriter &rewriter";
    for (Value in : op.getInputs())
      ds << ", " << cppType(in.getType());
    ds << ");";
    rewriterDecls[fnName] = decl;
  }

  std::string call;
  {
    llvm::raw_string_ostream cs(call);
    cs << "::mlir::match::runtime::applyRewrite(configs, rewriter, [&] { "
          "return "
       << fnName << "(rewriter";
    if (rewriterOp)
      cs << ", " << locExpr();
    for (Value in : op.getInputs())
      cs << ", " << getName(in);
    cs << "); })";
  }

  uint64_t benefit = op.getBenefit();
  indent() << "// match.success @" << leaf << " benefit(" << benefit << ")\n";
  if (benefit < reachableBenefit.lookup(op)) {
    indent() << "if (!deferred || deferredBenefit < " << benefit << ") {\n";
    indent() << "  deferredBenefit = " << benefit << ";\n";
    indent() << "  deferred = [=, &rewriter]() -> ::llvm::LogicalResult {\n";
    indent() << "    rewriter.setInsertionPoint(op);\n";
    indent() << "    return " << call << ";\n";
    indent() << "  };\n";
    indent() << "}\n";
    return success();
  }
  if (anyDeferred && maxDeferredBenefit >= benefit) {
    indent() << "if (deferred && deferredBenefit >= " << benefit << ")\n";
    indent() << "  return applyDeferred();\n";
  }
  indent() << "rewriter.setInsertionPoint(op);\n";
  indent() << "if (::mlir::succeeded(" << call << "))\n";
  indent() << "  return ::mlir::success();\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitTry(TryOp op) {
  Scope scope(*this);
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
  Scope scope(*this);
  std::string elem = declareName(op.getLoopVariable());
  if (isa<pdl::OperationType>(op.getLoopVariable().getType()))
    locOps.push_back(op.getLoopVariable());
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
    Scope scope(*this);
    auto caseName = cast<StringAttr>(cases[j]).getValue();
    indent() << (j == 0 ? "if" : "else if") << " (";
    if (const OpInfo *info = caseInfos[j])
      os << "::llvm::isa<" << info->cppClassName << ">(" << getName(op.getOp())
         << ")";
    else
      os << nameVar << " == \"" << caseName << "\"";
    os << ") {\n";
    ++indentLevel;
    if (const OpInfo *info = caseInfos[j])
      bindConcrete(op.getOp(), *info, /*alreadyKnownName=*/true, failAction);
    if (failed(emitOpSequence(opsOf(region), failAction)))
      return failure();
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
    Scope scope(*this);
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

LogicalResult MatcherCppEmitter::emitNativeCall(Operation *op, StringRef name,
                                                ValueRange args,
                                                ValueRange results,
                                                StringRef failCondition,
                                                StringRef failAction) {
  for (Value result : results) {
    if (isRangeType(result.getType()) &&
        isa<pdl::OperationType>(
            cast<pdl::RangeType>(result.getType()).getElementType()))
      return op->emitError("operation range results of native functions are "
                           "not supported by the C++ emitter");
    std::string v = declareName(result);
    indent() << cppStorageType(result.getType()) << " " << v << ";\n";
  }
  indent() << "if (" << failCondition << "(::mlir::match::runtime::callNative("
           << name << "PDLFn, rewriter, ::std::tie(";
  llvm::interleaveComma(results, os, [&](Value v) { os << getName(v); });
  os << ")";
  for (Value arg : args)
    os << ", " << getName(arg);
  os << ")))\n";
  indent() << "  " << failAction << "\n";
  return success();
}

void MatcherCppEmitter::emitGetValues(Value opVal,
                                      std::optional<unsigned> index,
                                      Value result, bool operands,
                                      bool optional) {
  Type type = result.getType();
  if (auto opt = dyn_cast<OptionalType>(type))
    type = opt.getInnerType();
  bool range = isRangeType(type);
  std::string opn = getName(opVal);
  std::string v = declareName(result);
  StringRef kind = operands ? "Operand" : "Result";
  StringRef all = operands ? "getOperands()" : "getResults()";

  if (const ConcreteBinding *b = concreteOf(opVal)) {
    const OpInfo &info = *b->info;
    std::optional<unsigned> fixed =
        operands ? info.getFixedNumOperands() : info.getFixedNumResults();
    unsigned numGroups =
        operands ? info.numOperandGroups : info.numResultGroups;
    if (!index) {
      if (range) {
        indent() << "auto " << v << " = " << b->var << ".getOperation()->"
                 << all << ";\n";
        knownNonNull.insert(result);
        return;
      }
    } else if (fixed && *index < *fixed && !range) {
      indent() << "::mlir::Value " << v << " = " << b->var
               << ".getOperation()->get" << kind << "(" << *index << ");\n";
      knownNonNull.insert(result);
      return;
    } else if (*index < numGroups) {
      if (range) {
        indent() << "auto " << v << " = " << b->var << ".getODS" << kind << "s("
                 << *index << ");\n";
        knownNonNull.insert(result);
      } else {
        indent() << "::mlir::Value " << v
                 << " = ::mlir::match::runtime::getSingleValue(" << b->var
                 << ".getODS" << kind << "s(" << *index << "));\n";
      }
      return;
    }
  }

  if (!index) {
    if (range) {
      indent() << "::mlir::ValueRange " << v << " = " << opn << "->" << all
               << ";\n";
      knownNonNull.insert(result);
    } else {
      indent() << "::mlir::Value " << v
               << " = ::mlir::match::runtime::getSingleValue(" << opn << "->"
               << all << ");\n";
    }
    return;
  }

  std::string group = ("::mlir::match::runtime::get" + kind + "Group(" + opn +
                       ", " + Twine(*index) + ")")
                          .str();
  if (!range) {
    indent() << "::mlir::Value " << v
             << " = ::mlir::match::runtime::getSingleValue(" << group << ");\n";
  } else if (optional) {
    indent() << "::std::optional<::mlir::ValueRange> " << v << " = " << group
             << ";\n";
  } else {
    indent() << "::mlir::ValueRange " << v << " = " << group
             << ".value_or(::mlir::ValueRange());\n";
  }
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
      .Case<SuccessOp>([&](SuccessOp o) { return emitSuccess(o); })
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
        emitGetValues(o.getOp(), o.getIndex(), o.getResult(),
                      /*operands=*/true, /*optional=*/true);
        return success();
      })
      .Case<GetResultsOp>([&](GetResultsOp o) {
        emitGetValues(o.getOp(), o.getIndex(), o.getResult(),
                      /*operands=*/false, /*optional=*/true);
        return success();
      })
      .Case<GetAttributeOp>([&](GetAttributeOp o) {
        std::string v = declareName(o.getResult());
        if (const ConcreteBinding *b = concreteOf(o.getOp())) {
          if (const std::string *getter =
                  b->info->lookupAttrGetter(o.getName())) {
            indent() << "::mlir::Attribute " << v << " = " << b->var << "."
                     << *getter << "Attr();\n";
            return success();
          }
        }
        indent() << "::mlir::Attribute " << v << " = " << getName(o.getOp())
                 << "->getAttr(\"" << o.getName() << "\");\n";
        return success();
      })
      .Case<GetDefiningOpOp>([&](GetDefiningOpOp o) {
        std::string v = declareName(o.getResult());
        if (isRangeType(o.getValue().getType())) {
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
        if (isRangeType(o.getValue().getType()))
          indent() << "::mlir::TypeRange " << v << " = ::mlir::ValueRange("
                   << getName(o.getValue()) << ").getTypes();\n";
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
        if (isa<pdl::OperationType>(o.getUnwrapped().getType()))
          locOps.push_back(o.getUnwrapped());
        if (knownNonNull.contains(o.getOptionalValue())) {
          mapName(o.getUnwrapped(), name);
          return success();
        }
        Type inner =
            cast<OptionalType>(o.getOptionalValue().getType()).getInnerType();
        if (isRangeType(inner)) {
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
        bool isRootKind = o.getOp() == rootValue && o.getName() == rootKind;
        if (const OpInfo *info = registry.lookup(o.getName())) {
          bindConcrete(o.getOp(), *info, /*alreadyKnownName=*/isRootKind,
                       failAction);
          return success();
        }
        if (isRootKind)
          return success();
        fail(getName(o.getOp()) + "->getName().getStringRef() != \"" +
             o.getName().str() + "\"");
        return success();
      })
      .Case<EqualOp>([&](EqualOp o) {
        if (isRangeType(o.getLhs().getType()))
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
      .Case<ApplyNativeConstraintOp>([&](ApplyNativeConstraintOp o) {
        return emitNativeCall(
            o, o.getName(), o.getArgs(), o.getConstraintResults(),
            o.getIsNegated() ? "::mlir::succeeded" : "::mlir::failed",
            failAction);
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
  indent() << failAction << "\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitMatcher(MatcherOp matcher, unsigned idx) {
  resetState();
  ctxExpr = "op->getContext()";

  Block &body = matcher.getBodyRegion().front();
  rootValue = body.getArgument(0);
  mapName(rootValue, "op");
  locOps.push_back(rootValue);

  rootKind = StringRef();
  for (Operation &op : body) {
    if (isa<SuccessOp>(op) || op.getNumRegions())
      break;
    if (auto hasName = dyn_cast<HasNameOp>(op))
      if (hasName.getOp() == rootValue)
        rootKind = hasName.getName();
  }

  SmallVector<SuccessOp> successes;
  matcher.walk<WalkOrder::PreOrder>(
      [&](SuccessOp s) { successes.push_back(s); });
  DenseMap<Operation *, uint64_t> loopBenefit;
  matcher.walk([&](ForEachOp loop) {
    uint64_t benefit = 0;
    loop.walk([&](SuccessOp s) {
      benefit = std::max<uint64_t>(benefit, s.getBenefit());
    });
    loopBenefit[loop] = benefit;
  });
  uint64_t maxBenefit = 1;
  uint64_t laterBenefit = 0;
  reachableBenefit.clear();
  anyDeferred = false;
  maxDeferredBenefit = 0;
  for (SuccessOp s : llvm::reverse(successes)) {
    uint64_t reachable = laterBenefit;
    for (Operation *parent = s->getParentOp(); parent != matcher;
         parent = parent->getParentOp())
      if (isa<ForEachOp>(parent))
        reachable = std::max(reachable, loopBenefit.lookup(parent));
    reachableBenefit[s] = reachable;
    laterBenefit = std::max<uint64_t>(laterBenefit, s.getBenefit());
    maxBenefit = std::max<uint64_t>(maxBenefit, s.getBenefit());
    if (s.getBenefit() < reachable) {
      anyDeferred = true;
      maxDeferredBenefit =
          std::max<uint64_t>(maxDeferredBenefit, s.getBenefit());
    }
  }

  std::string structName = ("GeneratedMatcher_" + Twine(idx)).str();
  structNames.push_back(structName);

  os << "namespace {\n";
  os << "struct " << structName << " : public ::mlir::RewritePattern {\n";
  os << "  " << structName
     << "(::mlir::MLIRContext *context,\n"
        "      ::std::shared_ptr<::mlir::PDLPatternConfigSet> configSet)\n";
  os << "      : ::mlir::RewritePattern(";
  if (rootKind.empty())
    os << "::mlir::Pattern::MatchAnyOpTypeTag()";
  else
    os << "\"" << rootKind << "\"";
  os << ", " << maxBenefit << ", context),\n";
  os << "        configSet(::std::move(configSet)) {}\n";
  os << "  ::llvm::LogicalResult\n";
  os << "  matchAndRewrite(::mlir::Operation *op,\n";
  os << "                  ::mlir::PatternRewriter &rewriter) const override "
        "{\n";
  indentLevel = 2;
  indent() << "::mlir::PDLPatternConfigSet *configs = configSet.get();\n";
  if (anyDeferred) {
    indent() << "unsigned deferredBenefit = 0;\n";
    indent() << "::llvm::unique_function<::llvm::LogicalResult()> deferred;\n";
    indent() << "auto applyDeferred = [&]() -> ::llvm::LogicalResult {\n";
    indent() << "  auto fn = std::move(deferred);\n";
    indent() << "  deferred = nullptr;\n";
    indent() << "  return fn();\n";
    indent() << "};\n";
    indent() << "auto match = [&]() -> ::llvm::LogicalResult {\n";
    ++indentLevel;
  }
  if (failed(emitOpSequence(opsOf(matcher.getBodyRegion()),
                            "return ::mlir::failure();")))
    return failure();
  if (anyDeferred) {
    --indentLevel;
    indent() << "};\n";
    indent() << "if (::mlir::succeeded(match()))\n";
    indent() << "  return ::mlir::success();\n";
    indent() << "if (deferred)\n";
    indent() << "  return applyDeferred();\n";
    indent() << "return ::mlir::failure();\n";
  }
  indentLevel = 0;
  os << "  }\n";
  os << "  ::std::shared_ptr<::mlir::PDLPatternConfigSet> configSet;\n";
  os << "};\n";
  os << "} // namespace\n\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitRewriterOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](ConstantAttributeOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::mlir::Attribute " << v << " = "
                 << attrLiteral(o.getValue()) << ";\n";
        knownNonNull.insert(o.getResult());
        return success();
      })
      .Case([&](ConstantTypeOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::mlir::Type " << v << " = " << typeLiteral(o.getValue())
                 << ";\n";
        return success();
      })
      .Case([&](ConstantTypesOp o) {
        std::string v = declareName(o.getResult());
        indent() << "::llvm::SmallVector<::mlir::Type> " << v << "{";
        emitTypeLiterals(o.getValue());
        os << "};\n";
        return success();
      })
      .Case([&](ApplyNativeRewriteOp o) {
        return emitNativeCall(o, o.getName(), o.getArgs(), o.getResults(),
                              "::mlir::failed", "return ::mlir::failure();");
      })
      .Case([&](CreateOperationOp o) -> LogicalResult {
        const OpInfo *info = registry.lookup(o.getName());
        std::string state = freshId("state");
        indent() << "::mlir::OperationState " << state << "(loc, ";
        if (info)
          os << "::mlir::match::runtime::getOperationName<"
             << info->cppClassName << ">(rewriter.getContext())";
        else
          os << "\"" << o.getName() << "\"";
        os << ");\n";
        for (Value operand : o.getInputOperands()) {
          if (isRangeType(operand.getType()))
            indent() << state << ".addOperands(" << getName(operand) << ");\n";
          else
            indent() << state << ".operands.push_back(" << getName(operand)
                     << ");\n";
        }
        for (auto [nameAttr, attr] :
             llvm::zip(o.getInputAttributeNames(), o.getInputAttributes())) {
          StringRef attrName = cast<StringAttr>(nameAttr).getValue();
          std::string nameExpr;
          if (const std::string *getter =
                  info ? info->lookupAttrGetter(attrName) : nullptr)
            nameExpr = (info->cppClassName + "::" + *getter + "AttrName(" +
                        state + ".name)");
          else
            nameExpr = ("rewriter.getStringAttr(\"" + attrName + "\")").str();
          std::string attrVar = getName(attr);
          if (knownNonNull.contains(attr)) {
            indent() << state << ".addAttribute(" << nameExpr << ", " << attrVar
                     << ");\n";
          } else {
            indent() << "if (" << attrVar << ")\n";
            indent() << "  " << state << ".addAttribute(" << nameExpr << ", "
                     << attrVar << ");\n";
          }
        }
        if (o.getInferredResultTypes()) {
          indent() << "if (::mlir::failed(";
          if (info && info->inferResultTypes)
            os << info->cppClassName
               << "::inferReturnTypes(rewriter.getContext(), " << state
               << ".location, " << state << ".operands, " << state
               << ".attributes.getDictionary(rewriter.getContext()), " << state
               << ".getRawProperties(), " << state << ".regions, " << state
               << ".types)";
          else
            os << "::mlir::match::runtime::inferResultTypes(" << state << ")";
          os << "))\n";
          indent() << "  return ::mlir::failure();\n";
        } else {
          for (Value type : o.getInputResultTypes()) {
            if (isRangeType(type.getType()))
              indent() << state << ".addTypes(" << getName(type) << ");\n";
            else
              indent() << state << ".types.push_back(" << getName(type)
                       << ");\n";
          }
        }
        std::string v = declareName(o.getResultOp());
        indent() << "::mlir::Operation *" << v << " = rewriter.create(" << state
                 << ");\n";
        if (info)
          bindConcrete(o.getResultOp(), *info, /*alreadyKnownName=*/true, "");
        return success();
      })
      .Case([&](pdl::ResultOp o) {
        std::string opName = getName(o.getParent());
        std::string v = declareName(o.getVal());
        indent() << "::mlir::Value " << v << " = " << opName << "->getResult("
                 << o.getIndex() << ");\n";
        return success();
      })
      .Case([&](pdl::ResultsOp o) {
        emitGetValues(o.getParent(), o.getIndex(), o.getVal(),
                      /*operands=*/false, /*optional=*/false);
        return success();
      })
      .Case([&](GetValueTypeOp o) {
        std::string v = declareName(o.getResult());
        if (isRangeType(o.getValue().getType()))
          indent() << "::mlir::TypeRange " << v << " = ::mlir::ValueRange("
                   << getName(o.getValue()) << ").getTypes();\n";
        else
          indent() << "::mlir::Type " << v << " = " << getName(o.getValue())
                   << ".getType();\n";
        return success();
      })
      .Case([&](CreateRangeOp o) {
        std::string v = declareName(o.getResult());
        indent() << cppStorageType(o.getResult().getType()) << " " << v
                 << ";\n";
        for (Value arg : o.getArguments()) {
          if (isRangeType(arg.getType()))
            indent() << "::llvm::append_range(" << v << ", " << getName(arg)
                     << ");\n";
          else
            indent() << v << ".push_back(" << getName(arg) << ");\n";
        }
        return success();
      })
      .Case([&](ReplaceOp o) {
        std::string target = getName(o.getInputOp());
        OperandRange repl = o.getReplValues();
        if (repl.size() == 1 && isRangeType(repl[0].getType())) {
          indent() << "rewriter.replaceOp(" << target << ", "
                   << getName(repl[0]) << ");\n";
        } else if (llvm::none_of(repl, [](Value v) {
                     return isRangeType(v.getType());
                   })) {
          indent() << "rewriter.replaceOp(" << target
                   << ", ::mlir::ValueRange{";
          llvm::interleaveComma(repl, os, [&](Value v) { os << getName(v); });
          os << "});\n";
        } else {
          std::string vals = freshId("repl");
          indent() << "::llvm::SmallVector<::mlir::Value> " << vals << ";\n";
          for (Value v : repl) {
            if (isRangeType(v.getType()))
              indent() << "::llvm::append_range(" << vals << ", " << getName(v)
                       << ");\n";
            else
              indent() << vals << ".push_back(" << getName(v) << ");\n";
          }
          indent() << "rewriter.replaceOp(" << target << ", " << vals << ");\n";
        }
        return success();
      })
      .Case([&](EraseOp o) {
        indent() << "rewriter.eraseOp(" << getName(o.getInputOp()) << ");\n";
        return success();
      })
      .Default([&](Operation *o) {
        return o->emitError("unsupported op in C++ rewriter emitter");
      });
}

LogicalResult MatcherCppEmitter::emitRewriter(RewriterOp rewriter,
                                              StringRef fnName) {
  resetState();
  ctxExpr = "rewriter.getContext()";
  indentLevel = 0;

  os << "static ::llvm::LogicalResult " << fnName
     << "(::mlir::PatternRewriter &rewriter, ::mlir::Location loc";
  Block &body = rewriter.getBody().front();
  for (BlockArgument arg : body.getArguments()) {
    if (isRangeType(arg.getType()) &&
        isa<pdl::OperationType>(
            cast<pdl::RangeType>(arg.getType()).getElementType()))
      return rewriter.emitError(
          "operation range rewriter inputs are not supported "
          "by the C++ emitter");
    os << ", " << cppType(arg.getType()) << " " << declareName(arg);
  }
  os << ") {\n";
  indentLevel = 1;
  for (Operation &op : body)
    if (failed(emitRewriterOp(&op)))
      return failure();
  indent() << "return ::mlir::success();\n";
  indentLevel = 0;
  os << "}\n\n";
  return success();
}

LogicalResult MatcherCppEmitter::emitModule(Operation *root,
                                            StringRef cppNamespace) {
  SmallVector<MatcherOp> matchers;
  root->walk([&](MatcherOp m) { matchers.push_back(m); });

  std::string structs;
  std::string rewriters;
  {
    llvm::raw_string_ostream structOs(structs);
    MatcherCppEmitter sub(structOs, registry);
    for (auto [idx, m] : llvm::enumerate(matchers))
      if (failed(sub.emitMatcher(m, idx)))
        return failure();
    rewriterDecls = std::move(sub.rewriterDecls);
    structNames = std::move(sub.structNames);
    generatedRewriters = std::move(sub.generatedRewriters);
  }
  {
    llvm::raw_string_ostream rewriterOs(rewriters);
    MatcherCppEmitter sub(rewriterOs, registry);
    for (auto &[func, fnName] : generatedRewriters)
      if (failed(sub.emitRewriter(cast<RewriterOp>(func), fnName)))
        return failure();
  }

  os << "//===- Generated by mlir-match-to-cpp. DO NOT EDIT. -===//\n";
  os << "//\n";
  os << "// Native constraints and rewrites `Name` are called as "
        "`NamePDLFn`.\n";
  os << "// The user must provide them and the `rewrite_*` hooks declared "
        "below.\n";
  os << "//===--------------------------------------------------------------"
        "------===//\n\n";
  os << "#include \"mlir/IR/BuiltinAttributes.h\"\n";
  os << "#include \"mlir/IR/OpDefinition.h\"\n";
  os << "#include \"mlir/IR/Operation.h\"\n";
  os << "#include \"mlir/IR/PatternMatch.h\"\n";
  os << "#include \"mlir/Parser/Parser.h\"\n";
  os << "#include \"mlir/Target/MatchToCpp/Runtime.h\"\n";
  os << "#include \"llvm/ADT/FunctionExtras.h\"\n";
  os << "#include \"llvm/ADT/STLExtras.h\"\n";
  os << "#include <memory>\n";
  os << "#include <optional>\n";
  os << "\n";
  os << "#if defined(__GNUC__)\n";
  os << "#pragma GCC diagnostic push\n";
  os << "#pragma GCC diagnostic ignored \"-Wunused-variable\"\n";
  os << "#endif\n\n";

  if (!cppNamespace.empty())
    os << "namespace " << cppNamespace << " {\n\n";

  if (!rewriterDecls.empty()) {
    os << "// Hooks the user must define and link:\n";
    for (auto &kv : rewriterDecls)
      os << kv.second << "\n";
    os << "\n";
  }

  os << rewriters;
  os << structs;

  os << "template <typename... ConfigsT>\n"
        "[[maybe_unused]] static void\n"
        "populateGeneratedPatterns(::mlir::RewritePatternSet &set,\n"
        "                          ConfigsT &&...configs) {\n";
  os << "  ::std::shared_ptr<::mlir::PDLPatternConfigSet> configSet;\n";
  os << "  if constexpr (sizeof...(ConfigsT) != 0)\n";
  os << "    configSet = ::std::make_shared<::mlir::PDLPatternConfigSet>(\n";
  os << "        ::std::forward<ConfigsT>(configs)...);\n";
  if (!structNames.empty()) {
    os << "  set.add<";
    llvm::interleaveComma(structNames, os);
    os << ">(set.getContext(), configSet);\n";
  }
  os << "}\n";

  if (!cppNamespace.empty())
    os << "\n} // namespace " << cppNamespace << "\n";

  os << "\n#if defined(__GNUC__)\n";
  os << "#pragma GCC diagnostic pop\n";
  os << "#endif\n";
  return success();
}

LogicalResult mlir::match::translateToCpp(Operation *op, raw_ostream &os,
                                          const OpInfoRegistry &registry,
                                          StringRef cppNamespace) {
  MatcherCppEmitter emitter(os, registry);
  return emitter.emitModule(op, cppNamespace);
}

LogicalResult mlir::match::translateToCpp(Operation *op, raw_ostream &os) {
  static const OpInfoRegistry emptyRegistry;
  return translateToCpp(op, os, emptyRegistry);
}
