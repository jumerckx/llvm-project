//===- Runtime.h - Support code for C++ emitted from match ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_MATCHTOCPP_RUNTIME_H
#define MLIR_TARGET_MATCHTOCPP_RUNTIME_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mlir {
namespace match {
namespace runtime {

inline std::optional<ValueRange> getODSGroup(Operation *op, unsigned index,
                                             ValueRange values,
                                             StringRef segmentAttr,
                                             bool hasSegments) {
  if (hasSegments) {
    auto segments = op->getAttrOfType<DenseI32ArrayAttr>(segmentAttr);
    if (!segments || segments.asArrayRef().size() <= index)
      return std::nullopt;
    ArrayRef<int32_t> sizes = segments.asArrayRef();
    unsigned start = 0;
    for (unsigned i = 0; i < index; ++i)
      start += sizes[i];
    return values.slice(start, sizes[index]);
  }
  if (values.size() >= index)
    return values.drop_front(index);
  return std::nullopt;
}

inline std::optional<ValueRange> getOperandGroup(Operation *op,
                                                 unsigned index) {
  return getODSGroup(op, index, op->getOperands(), "operandSegmentSizes",
                     op->hasTrait<OpTrait::AttrSizedOperandSegments>());
}

inline std::optional<ValueRange> getResultGroup(Operation *op, unsigned index) {
  return getODSGroup(op, index, op->getResults(), "resultSegmentSizes",
                     op->hasTrait<OpTrait::AttrSizedResultSegments>());
}

template <typename RangeT>
Value getSingleValue(RangeT &&range) {
  return llvm::hasSingleElement(range) ? Value(*range.begin()) : Value();
}

inline Value getSingleValue(std::optional<ValueRange> range) {
  return range ? getSingleValue(*range) : Value();
}

template <typename OpT>
OperationName getOperationName(MLIRContext *context) {
  if (std::optional<RegisteredOperationName> name =
          RegisteredOperationName::lookup(TypeID::get<OpT>(), context))
    return *name;
  return OperationName(OpT::getOperationName(), context);
}

inline LogicalResult inferResultTypes(OperationState &state) {
  auto *inferInterface = state.name.getInterface<InferTypeOpInterface>();
  if (!inferInterface)
    return failure();
  MLIRContext *context = state.getContext();
  return inferInterface->inferReturnTypes(
      context, state.location, state.operands,
      state.attributes.getDictionary(context), state.getRawProperties(),
      state.regions, state.types);
}

namespace detail {
template <typename T>
struct IsSmallVector : std::false_type {};
template <typename T, unsigned N>
struct IsSmallVector<SmallVector<T, N>> : std::true_type {};

template <typename T>
struct IsFailureOr : std::false_type {};
template <typename T>
struct IsFailureOr<FailureOr<T>> : std::true_type {};

template <typename T>
struct IsTuple : std::false_type {};
template <typename... Ts>
struct IsTuple<std::tuple<Ts...>> : std::true_type {};
template <typename T1, typename T2>
struct IsTuple<std::pair<T1, T2>> : std::true_type {};

class ResultList : public PDLResultList {
public:
  explicit ResultList(unsigned maxNumResults) : PDLResultList(maxNumResults) {}
  ArrayRef<PDLValue> getResults() const { return results; }
};

template <typename T, typename From>
T castArg(From &&from) {
  if constexpr (std::is_convertible_v<From, T>)
    return std::forward<From>(from);
  else if constexpr (std::is_same_v<T, StringRef> ||
                     std::is_same_v<T, std::string>)
    return T(llvm::cast<StringAttr>(from).getValue());
  else
    return llvm::cast<T>(from);
}

template <typename To, typename From>
void assignResult(PatternRewriter &rewriter, To &to, From &&from) {
  using FromT = std::decay_t<From>;
  if constexpr (IsSmallVector<To>::value)
    to.assign(from.begin(), from.end());
  else if constexpr (std::is_same_v<To, Attribute> &&
                     !std::is_base_of_v<Attribute, FromT> &&
                     std::is_convertible_v<From, StringRef>)
    to = rewriter.getStringAttr(StringRef(from));
  else
    to = std::forward<From>(from);
}

template <typename To>
void assignPDLValue(To &to, PDLValue value) {
  if constexpr (IsSmallVector<To>::value) {
    if constexpr (std::is_same_v<typename To::value_type, Value>) {
      ValueRange range = value.cast<ValueRange>();
      to.assign(range.begin(), range.end());
    } else {
      TypeRange range = value.cast<TypeRange>();
      to.assign(range.begin(), range.end());
    }
  } else if constexpr (std::is_same_v<To, Operation *>) {
    to = static_cast<Operation *>(
        const_cast<void *>(value.getAsOpaquePointer()));
  } else {
    to = To::getFromOpaquePointer(value.getAsOpaquePointer());
  }
}

template <typename... Rs, typename R, size_t... I>
void assignTuple(PatternRewriter &rewriter, std::tuple<Rs &...> results,
                 R &&values, std::index_sequence<I...>) {
  (assignResult(rewriter, std::get<I>(results),
                std::get<I>(std::forward<R>(values))),
   ...);
}

template <typename... Rs, size_t... I>
void assignPDLValues(std::tuple<Rs &...> results, ArrayRef<PDLValue> values,
                     std::index_sequence<I...>) {
  (assignPDLValue(std::get<I>(results), values[I]), ...);
}

template <typename... Rs, typename R>
LogicalResult assignResults(PatternRewriter &rewriter,
                            std::tuple<Rs &...> results, R &&value) {
  using ValueT = std::decay_t<R>;
  if constexpr (std::is_same_v<ValueT, LogicalResult>) {
    static_assert(sizeof...(Rs) == 0, "unexpected native function results");
    return value;
  } else if constexpr (IsFailureOr<ValueT>::value) {
    if (failed(value))
      return failure();
    return assignResults(rewriter, results, std::move(*value));
  } else if constexpr (IsTuple<ValueT>::value) {
    static_assert(std::tuple_size_v<ValueT> == sizeof...(Rs),
                  "mismatched number of native function results");
    assignTuple(rewriter, results, std::forward<R>(value),
                std::index_sequence_for<Rs...>{});
    return success();
  } else {
    static_assert(sizeof...(Rs) == 1, "expected a single native result");
    assignResult(rewriter, std::get<0>(results), std::forward<R>(value));
    return success();
  }
}

template <typename T>
auto toPDLStorage(T &&arg) {
  using ArgT = std::decay_t<T>;
  if constexpr (std::is_convertible_v<ArgT, Value> ||
                std::is_convertible_v<ArgT, Attribute> ||
                std::is_convertible_v<ArgT, Type> ||
                std::is_convertible_v<ArgT, Operation *>)
    return arg;
  else if constexpr (std::is_convertible_v<T, ValueRange>)
    return ValueRange(arg);
  else
    return TypeRange(arg);
}

template <typename T>
PDLValue toPDLValue(T &stored) {
  if constexpr (std::is_same_v<T, ValueRange> || std::is_same_v<T, TypeRange>)
    return PDLValue(&stored);
  else
    return PDLValue(stored);
}

template <typename Fn, typename... Rs, typename... Args>
LogicalResult callPDLValueFn(Fn &fn, PatternRewriter &rewriter,
                             std::tuple<Rs &...> results, Args &&...args) {
  auto storage = std::make_tuple(toPDLStorage(std::forward<Args>(args))...);
  return std::apply(
      [&](auto &...stored) -> LogicalResult {
        SmallVector<PDLValue, 4> values{toPDLValue(stored)...};
        ResultList resultList(sizeof...(Rs));
        if (failed(fn(rewriter, resultList, values)))
          return failure();
        ArrayRef<PDLValue> out = resultList.getResults();
        assert(out.size() == sizeof...(Rs) &&
               "mismatched number of native function results");
        assignPDLValues(results, out, std::index_sequence_for<Rs...>{});
        return success();
      },
      storage);
}

template <typename Fn, typename... Rs, size_t... I, typename... Args>
LogicalResult callTypedFn(Fn &fn, PatternRewriter &rewriter,
                          std::tuple<Rs &...> results,
                          std::index_sequence<I...>, Args &&...args) {
  using Traits = llvm::function_traits<std::decay_t<Fn>>;
  static_assert(Traits::num_args == sizeof...(Args) + 1,
                "mismatched number of native function arguments");
  if constexpr (std::is_void_v<typename Traits::result_t>) {
    static_assert(sizeof...(Rs) == 0, "unexpected native function results");
    fn(rewriter, castArg<typename Traits::template arg_t<I + 1>>(
                     std::forward<Args>(args))...);
    return success();
  } else {
    return assignResults(
        rewriter, results,
        fn(rewriter, castArg<typename Traits::template arg_t<I + 1>>(
                         std::forward<Args>(args))...));
  }
}
} // namespace detail

template <typename Fn, typename... Rs, typename... Args>
LogicalResult callNative(Fn &&fn, PatternRewriter &rewriter,
                         std::tuple<Rs &...> results, Args &&...args) {
  if constexpr (std::is_invocable_r_v<LogicalResult, Fn &, PatternRewriter &,
                                      PDLResultList &, ArrayRef<PDLValue>>)
    return detail::callPDLValueFn(fn, rewriter, results,
                                  std::forward<Args>(args)...);
  else
    return detail::callTypedFn(fn, rewriter, results,
                               std::index_sequence_for<Args...>{},
                               std::forward<Args>(args)...);
}

} // namespace runtime
} // namespace match
} // namespace mlir

#endif // MLIR_TARGET_MATCHTOCPP_RUNTIME_H
