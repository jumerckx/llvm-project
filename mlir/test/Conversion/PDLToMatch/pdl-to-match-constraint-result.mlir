// RUN: mlir-opt -split-input-file -convert-pdl-to-match \
// RUN:   %S/../PDLToPDLInterp/use-constraint-result.mlir | FileCheck %s

// Native constraints that *produce* values, run over the reference suite's
// corpus (`test/Conversion/PDLToPDLInterp/use-constraint-result.mlir`).
//
// Every split here exercises the same pair of behaviours:
//
//  1. an unbound literal argument is materialized as a `match.constant_*` op
//     (it is not reachable by navigation from the root, so there is nothing to
//     navigate to);
//  2. when a constraint result corresponds to a value that *is* reachable, the
//     two are tied together with a `match.equal`. Without that link the
//     constraint result would float free and the pattern would match strictly
//     more than written.
//
// The splits are unnamed modules, so the CHECK lines below are anchored on each
// split's distinctive constraint name and appear in input order.

//===----------------------------------------------------------------------===//
// Split 1: two attribute literals feed a result-producing constraint, whose
// result feeds a second constraint together with a navigated attribute.
//===----------------------------------------------------------------------===//

// CHECK:         %[[SHIFT:.*]] = get_attribute "shift" of %{{.*}} : <!pdl.attribute>
// CHECK-NEXT:    %[[SHIFTV:.*]] = is_not_null %[[SHIFT]]
// CHECK:         %[[C0:.*]] = constant_attribute 0 : i32
// CHECK-NEXT:    %[[C1:.*]] = constant_attribute 1 : i32
// CHECK-NEXT:    %[[RES:.*]] = apply_native_constraint "return_attr_constraint"(%[[C0]], %[[C1]] : !pdl.attribute, !pdl.attribute) : !pdl.attribute
// CHECK-NEXT:    apply_native_constraint "use_attr_constraint"(%[[SHIFTV]], %[[RES]] : !pdl.attribute, !pdl.attribute)

//===----------------------------------------------------------------------===//
// Split 2: the constraint result is used as an *operation attribute*. The
// attribute is produced by the constraint rather than by a `pdl.attribute`, so
// it carries no attribute-level constraints of its own; the link to the
// navigated `get_attribute` is the `equal` below.
//
// This split used to abort the pass (`cast<pdl::AttributeOp>` on a
// constraint-produced value).
//===----------------------------------------------------------------------===//

// CHECK:         %[[ATTR:.*]] = get_attribute "attr" of %{{.*}} : <!pdl.attribute>
// CHECK-NEXT:    %[[ATTRV:.*]] = is_not_null %[[ATTR]]
// CHECK:         %[[DOP:.*]] = is_not_null %{{.*}} : <!pdl.operation> -> !pdl.operation
// CHECK:         %[[RES:.*]] = apply_native_constraint "return_attr_constraint"(%[[DOP]] : !pdl.operation) : !pdl.attribute
// CHECK-NEXT:    equal %[[RES]], %[[ATTRV]] : !pdl.attribute
// The rewriter receives the navigated attribute, not the constraint result.
// CHECK-NEXT:    success @rewriters::@pdl_generated_rewriter benefit(1) (%{{.*}}, %[[ATTRV]] : !pdl.operation, !pdl.attribute)

//===----------------------------------------------------------------------===//
// Split 3: a constraint returning a !pdl.value, tied to the root's operand.
//===----------------------------------------------------------------------===//

// CHECK:         %[[O0:.*]] = get_operand 0 of %{{.*}} : <!pdl.value>
// CHECK-NEXT:    %[[V0:.*]] = is_not_null %[[O0]]
// CHECK-NEXT:    %[[C:.*]] = constant_attribute 10 : i64
// CHECK-NEXT:    %[[RES:.*]] = apply_native_constraint "return_value_constr"(%[[C]] : !pdl.attribute) : !pdl.value
// CHECK-NEXT:    equal %[[RES]], %[[V0]] : !pdl.value

//===----------------------------------------------------------------------===//
// Split 4: a constraint returning a !pdl.type, tied to the result's type.
//===----------------------------------------------------------------------===//

// CHECK:         %[[T:.*]] = get_value_type of %{{.*}} : !pdl.value : !pdl.type
// CHECK-NEXT:    %[[C:.*]] = constant_attribute 10 : i64
// CHECK-NEXT:    %[[RES:.*]] = apply_native_constraint "return_type_constr"(%[[C]] : !pdl.attribute) : !pdl.type
// CHECK-NEXT:    equal %[[RES]], %[[T]] : !pdl.type

//===----------------------------------------------------------------------===//
// Split 5: the range form — a constraint returning !pdl.range<type>, tied to
// the root's result types. Note there is no `check_result_count`: a single
// result range constrains nothing about the count.
//===----------------------------------------------------------------------===//

// CHECK:         %[[RS:.*]] = get_results of %{{.*}} : <!pdl.range<value>>
// CHECK-NEXT:    %[[RVS:.*]] = is_not_null %[[RS]]
// CHECK-NEXT:    %[[TS:.*]] = get_value_type of %[[RVS]] : !pdl.range<value> : !pdl.range<type>
// CHECK-NEXT:    %[[C:.*]] = constant_attribute 10 : i64
// CHECK-NEXT:    %[[RES:.*]] = apply_native_constraint "return_type_range_constr"(%[[C]] : !pdl.attribute) : !pdl.range<type>
// CHECK-NEXT:    equal %[[RES]], %[[TS]] : !pdl.range<type>
