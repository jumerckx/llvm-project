// RUN: mlir-opt -split-input-file -convert-pdl-to-match \
// RUN:   %S/../PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir | FileCheck %s

// This test deliberately has no input of its own: it runs
// `-convert-pdl-to-match` over the *same* curated `pdl.pattern` corpus that
// `test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir` uses for the
// reference lowering, and only supplies CHECK lines. The two passes consume
// identical input, so keeping one corpus avoids two sets of pattern inputs
// drifting apart.
//
// If you add or reorder a split in that file, the CHECK-LABELs below must be
// updated to match; they are anchored on the split's `module @name` and appear
// in input order.
//
// What this pass emits, in general: one `match.matcher` per `pdl.pattern`, with
// a flat body (no `try` / `switch_*` — those only appear after
// `-match-combine-matchers`), every nullable navigation op immediately followed
// by an `is_not_null` unwrap, and a `match.success` carrying the rewriter
// symbol, benefit and forwarded values.

//===----------------------------------------------------------------------===//
// A pattern-free module yields an empty rewriter module and no matcher.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @empty_module
// CHECK:         module @rewriters {
// CHECK-NOT:     match.matcher

//===----------------------------------------------------------------------===//
// The minimal shape: name + operand/result counts, then success.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @simple
// CHECK:       match.matcher root(%[[ROOT:.*]]: !pdl.operation) {
// CHECK-NEXT:    has_name %[[ROOT]], "foo.op"
// CHECK-NEXT:    check_operand_count %[[ROOT]] is 0
// CHECK-NEXT:    check_result_count %[[ROOT]] is 0
// CHECK-NEXT:    success @rewriters::@pdl_generated_rewriter benefit(1) (%[[ROOT]] : !pdl.operation)

//===----------------------------------------------------------------------===//
// Attributes: a literal reached by navigation folds into `has_attr_value`
// rather than becoming a constant op, and an attribute's type is reached with
// `get_attribute_type`.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @attributes
// CHECK:         %[[A:.*]] = get_attribute "attr" of %{{.*}} : <!pdl.attribute>
// CHECK-NEXT:    %[[AV:.*]] = is_not_null %[[A]] : <!pdl.attribute> -> !pdl.attribute
// CHECK-NEXT:    has_attr_value %[[AV]] is 10 : i64
// CHECK:         %[[A1:.*]] = get_attribute "attr1" of %{{.*}} : <!pdl.attribute>
// CHECK-NEXT:    %[[AV1:.*]] = is_not_null %[[A1]]
// CHECK-NEXT:    %[[AT:.*]] = get_attribute_type of %[[AV1]] : !pdl.type
// CHECK-NEXT:    has_type %[[AT]], i64

//===----------------------------------------------------------------------===//
// Native constraints: multi-argument, with results, with unused results, with
// multiple patterns, and negated.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @constraints
// CHECK:         apply_native_constraint "multi_constraint"(%{{.*}}, %{{.*}}, %{{.*}} : !pdl.value, !pdl.value, !pdl.value)

// A constraint result is registered and forwarded to the rewriter.
// CHECK-LABEL: module @constraint_with_result
// CHECK:         %[[C:.*]] = apply_native_constraint "check_op_and_get_attr_constr"(%[[ROOT:.*]] : !pdl.operation) : !pdl.attribute
// CHECK-NEXT:    success @rewriters::@pdl_generated_rewriter benefit(1) (%[[ROOT]], %[[C]] : !pdl.operation, !pdl.attribute)

// An unused constraint result is still emitted — the constraint has to run.
// CHECK-LABEL: module @constraint_with_unused_result
// CHECK:         %{{.*}} = apply_native_constraint "check_op_and_get_attr_constr"(%{{.*}} : !pdl.operation) : !pdl.attribute
// CHECK-NEXT:    success @rewriters::@pdl_generated_rewriter benefit(1) (%{{.*}} : !pdl.operation)

// Two patterns produce two matchers and two distinct rewriter symbols.
// CHECK-LABEL: module @constraint_with_result_multiple
// CHECK:         match.matcher root
// CHECK:           success @rewriters::@pdl_generated_rewriter benefit(1)
// CHECK:         match.matcher root
// CHECK:           success @rewriters::@pdl_generated_rewriter_0 benefit(1)

// `isNegated` is carried through verbatim.
// CHECK-LABEL: module @negated_constraint
// CHECK:         apply_native_constraint "constraint"(%{{.*}} : !pdl.operation) {isNegated = true}

//===----------------------------------------------------------------------===//
// Operands. A revisited value produces a `match.equal` instead of a second
// navigation chain (the `getOrRegister` dedup).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @inputs
// CHECK:         check_operand_count %{{.*}} is 2
// CHECK:         %[[O0:.*]] = get_operand 0 of %{{.*}} : <!pdl.value>
// CHECK-NEXT:    %[[V0:.*]] = is_not_null %[[O0]]
// CHECK:         %[[O1:.*]] = get_operand 1 of %{{.*}} : <!pdl.value>
// CHECK-NEXT:    %[[V1:.*]] = is_not_null %[[O1]]
// CHECK-NEXT:    equal %[[V1]], %[[V0]] : !pdl.value

// A variadic operand switches the check to `at_least` and selects
// `get_operands` at and after the variadic position.
// CHECK-LABEL: module @variadic_inputs
// CHECK:         check_operand_count %{{.*}} is at_least 2
// CHECK:         %{{.*}} = get_operand 0 of %{{.*}} : <!pdl.value>
// CHECK:         %[[OS:.*]] = get_operands 1 of %{{.*}} : <!pdl.range<value>>
// CHECK-NEXT:    %[[VS:.*]] = is_not_null %[[OS]] : <!pdl.range<value>> -> !pdl.range<value>
// CHECK-NEXT:    %[[TS:.*]] = get_value_type of %[[VS]] : !pdl.range<value> : !pdl.range<type>
// CHECK-NEXT:    has_types %[[TS]], [i64]

// A single operand range needs no operand-count check at all.
// CHECK-LABEL: module @single_operand_range
// CHECK-NOT:     check_operand_count
// CHECK:         %[[OS:.*]] = get_operands of %{{.*}} : <!pdl.range<value>>
// CHECK-NEXT:    %[[VS:.*]] = is_not_null %[[OS]]
// CHECK-NEXT:    %[[TS:.*]] = get_value_type of %[[VS]]
// CHECK-NEXT:    has_types %[[TS]], [i64]

//===----------------------------------------------------------------------===//
// Results — the mirror image of the operand cases.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @results
// CHECK:         check_result_count %{{.*}} is 2
// CHECK:         %[[R0:.*]] = get_result 0 of %{{.*}} : <!pdl.value>
// CHECK-NEXT:    %[[RV0:.*]] = is_not_null %[[R0]]
// CHECK-NEXT:    %[[T0:.*]] = get_value_type of %[[RV0]]
// CHECK-NEXT:    has_type %[[T0]], i32

// CHECK-LABEL: module @variadic_results
// CHECK:         check_result_count %{{.*}} is at_least 2
// CHECK:         %[[RS:.*]] = get_results 1 of %{{.*}} : <!pdl.range<value>>
// CHECK-NEXT:    %[[RVS:.*]] = is_not_null %[[RS]]
// CHECK-NEXT:    %[[TS:.*]] = get_value_type of %[[RVS]]
// CHECK-NEXT:    has_types %[[TS]], [i64]
// A revisited *type* also dedups to an `equal`.
// CHECK:         equal %{{.*}}, %{{.*}} : !pdl.type

// CHECK-LABEL: module @single_result_range
// CHECK-NOT:     check_result_count
// CHECK:         %[[RS:.*]] = get_results of %{{.*}} : <!pdl.range<value>>

//===----------------------------------------------------------------------===//
// A result used as an operand: navigate to the defining op, then tie the
// result back to the operand with `equal`.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @results_as_operands
// CHECK:         %[[O0:.*]] = get_operand 0 of %{{.*}} : <!pdl.value>
// CHECK-NEXT:    %[[V0:.*]] = is_not_null %[[O0]]
// CHECK-NEXT:    %[[D:.*]] = get_defining_op of %[[V0]] : !pdl.value -> <!pdl.operation>
// CHECK-NEXT:    %[[DOP:.*]] = is_not_null %[[D]] : <!pdl.operation> -> !pdl.operation
// CHECK-NEXT:    %[[DR:.*]] = get_result 0 of %[[DOP]] : <!pdl.value>
// CHECK-NEXT:    %[[DRV:.*]] = is_not_null %[[DR]]
// CHECK-NEXT:    equal %[[DRV]], %[[V0]] : !pdl.value
// The two operands share a defining op, which is pinned by a final `equal`.
// CHECK:         equal %{{.*}}, %[[DOP]] : !pdl.operation

// The range form uses `get_operands` / `get_results` and a range-typed equal.
// CHECK-LABEL: module @single_result_range_as_operands
// CHECK:         %[[OS:.*]] = get_operands of %{{.*}} : <!pdl.range<value>>
// CHECK-NEXT:    %[[VS:.*]] = is_not_null %[[OS]]
// CHECK-NEXT:    %[[D:.*]] = get_defining_op of %[[VS]] : !pdl.range<value> -> <!pdl.operation>
// CHECK-NEXT:    %[[DOP:.*]] = is_not_null %[[D]]
// CHECK-NEXT:    %[[DRS:.*]] = get_results of %[[DOP]] : <!pdl.range<value>>
// CHECK-NEXT:    %[[DRVS:.*]] = is_not_null %[[DRS]]
// CHECK-NEXT:    equal %[[DRVS]], %[[VS]] : !pdl.range<value>

//===----------------------------------------------------------------------===//
// Patterns differing only in a type or count become separate matchers here —
// this pass never emits a switch. Folding them is `-match-combine-matchers`.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @switch_single_result_type
// CHECK:         match.matcher root
// CHECK:           has_type %{{.*}}, i32
// CHECK:         match.matcher root
// CHECK:           has_type %{{.*}}, i64

// CHECK-LABEL: module @switch_result_types
// CHECK:         match.matcher root
// CHECK:           has_types %{{.*}}, [i32]
// CHECK:         match.matcher root
// CHECK:           has_types %{{.*}}, [i64, i32]

// CHECK-LABEL: module @switch_operand_count_at_least
// CHECK:         match.matcher root
// CHECK:           check_operand_count %{{.*}} is at_least 1
// CHECK:         match.matcher root
// CHECK:           check_operand_count %{{.*}} is at_least 2

// CHECK-LABEL: module @switch_result_count_at_least
// CHECK:         match.matcher root
// CHECK:           check_result_count %{{.*}} is at_least 1
// CHECK:         match.matcher root
// CHECK:           check_result_count %{{.*}} is at_least 2

//===----------------------------------------------------------------------===//
// Predicate ordering: the constrained pattern and the unconstrained one are
// emitted independently, each in source order.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @predicate_ordering
// CHECK:         match.matcher root
// CHECK:           %[[T:.*]] = get_value_type of %{{.*}} : !pdl.value : !pdl.type
// CHECK-NEXT:      apply_native_constraint "typeConstraint"(%[[T]] : !pdl.type)
// CHECK:         match.matcher root
// CHECK-NOT:       apply_native_constraint

//===----------------------------------------------------------------------===//
// Multi-root patterns: the second root is reached by walking *up* from a shared
// value with `get_users` + `get_each`, then tied back with `equal`. The matcher
// takes the name of the pattern and forwards both roots.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @multi_root
// CHECK:       match.matcher @rewrite_multi_root root(%[[ROOT:.*]]: !pdl.operation) {
// CHECK:         %[[USERS:.*]] = get_users of %[[V:.*]] : <operation>
// CHECK-NEXT:    %[[EACH:.*]] = get_each %[[USERS]] : !pdl.range<operation> -> !pdl.operation
// CHECK-NEXT:    %[[EO:.*]] = get_operand 0 of %[[EACH]] : <!pdl.value>
// CHECK-NEXT:    %[[EOV:.*]] = is_not_null %[[EO]]
// CHECK-NEXT:    equal %[[EOV]], %[[V]] : !pdl.value
// CHECK:         success @rewriters::@rewrite_multi_root benefit(1) (%[[ROOT]], %[[EACH]] : !pdl.operation, !pdl.operation)

// Overlapping roots need *no* upward traversal: both roots are reachable by a
// single downward chain from the chosen root, so the pass emits neither
// `get_users` nor `get_each` and forwards one operation.
// CHECK-LABEL: module @overlapping_roots
// CHECK:       match.matcher @rewrite_overlapping_roots root(%[[ROOT:.*]]: !pdl.operation) {
// CHECK-NOT:     get_users
// CHECK-NOT:     get_each
// CHECK:         success @rewriters::@rewrite_overlapping_roots benefit(1) (%[[ROOT]] : !pdl.operation)

// A forced root still produces a single upward traversal.
// CHECK-LABEL: module @force_overlapped_root
// CHECK:       match.matcher @rewrite_forced_overlapped_root root(%[[ROOT:.*]]: !pdl.operation) {
// CHECK:         %[[USERS:.*]] = get_users of %{{.*}} : <operation>
// CHECK-NEXT:    %[[EACH:.*]] = get_each %[[USERS]]
// CHECK:         success @rewriters::@rewrite_forced_overlapped_root benefit(1) (%[[ROOT]], %[[EACH]] : !pdl.operation, !pdl.operation)

//===----------------------------------------------------------------------===//
// When the connector between two roots is a *range*, an `extract` is needed to
// get a single value to walk up from.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @variadic_results_all
// CHECK:         %[[EX:.*]] = extract 0 of %[[RVS:.*]] : !pdl.value
// CHECK-NEXT:    %[[USERS:.*]] = get_users of %[[EX]] : <operation>
// CHECK-NEXT:    %[[EACH:.*]] = get_each %[[USERS]] : !pdl.range<operation> -> !pdl.operation
// The extracted range is the one tied back to the second root's operands.
// CHECK:         equal %{{.*}}, %[[RVS]] : !pdl.range<value>

// CHECK-LABEL: module @variadic_results_at
// CHECK:         %[[EX:.*]] = extract 0 of %{{.*}} : !pdl.value
// CHECK-NEXT:    %[[USERS:.*]] = get_users of %[[EX]] : <operation>
// CHECK-NEXT:    %[[EACH:.*]] = get_each %[[USERS]]

//===----------------------------------------------------------------------===//
// Unbound literals. These are not reachable by navigation from the root, so
// they are materialized as `match.constant_*` ops — the counterpart of the
// reference lowering's `pdl_interp.create_*`. Without this the constraint would
// be dropped and the pattern would match strictly more than written.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @attribute_literal
// CHECK:         %[[A:.*]] = constant_attribute 10 : i64
// CHECK-NEXT:    apply_native_constraint "constraint"(%[[A]] : !pdl.attribute)

// CHECK-LABEL: module @type_literal
// CHECK:         %[[T:.*]] = constant_type i32
// CHECK-NEXT:    %[[TS:.*]] = constant_types [i32, i64]
// CHECK-NEXT:    apply_native_constraint "constraint"(%[[T]], %[[TS]] : !pdl.type, !pdl.range<type>)

//===----------------------------------------------------------------------===//
// A value shared by several roots is navigated to once and reused; each root
// gets its own upward traversal.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @common_connector
// CHECK:       match.matcher @common_connector root(%[[ROOT:.*]]: !pdl.operation) {
// CHECK:         %[[USERS:.*]] = get_users of %[[V:.*]] : <operation>
// CHECK-NEXT:    %[[E1:.*]] = get_each %[[USERS]]
// CHECK:         %[[USERS2:.*]] = get_users of %[[V]] : <operation>
// CHECK-NEXT:    %[[E2:.*]] = get_each %[[USERS2]]
// CHECK:         success @rewriters::@common_connector benefit(1) (%[[E1]], %[[E2]], %[[ROOT]] : !pdl.operation, !pdl.operation, !pdl.operation)

// CHECK-LABEL: module @common_connector_range
// CHECK:       match.matcher @common_connector_range root(%[[ROOT:.*]]: !pdl.operation) {
// CHECK:         %[[EX:.*]] = extract 0 of %{{.*}} : !pdl.value
// CHECK-NEXT:    %[[USERS:.*]] = get_users of %[[EX]] : <operation>
// CHECK-NEXT:    %[[E1:.*]] = get_each %[[USERS]]
// CHECK:         success @rewriters::@common_connector_range benefit(1) (%[[E1]], %{{.*}}, %[[ROOT]] : !pdl.operation, !pdl.operation, !pdl.operation)
