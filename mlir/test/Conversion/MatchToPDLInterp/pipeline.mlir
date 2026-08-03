// RUN: mlir-opt -split-input-file \
// RUN:   -convert-pdl-to-match -match-combine-matchers -convert-match-to-pdl-interp \
// RUN:   %S/../PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir | FileCheck %s

// The three passes composed, run over the same curated `pdl.pattern` corpus the
// reference lowering uses. This is a regression net rather than a unit test:
// its job is to prove that all 30 pattern shapes survive
// pdl -> match -> combine -> pdl_interp and produce a well-formed matcher.
//
// Checks are deliberately loose — the set and order of the interesting
// `pdl_interp` ops, and each `record_match` — but never block numbers. Tight
// block-level assertions belong in match-to-pdl-interp.mlir, where the input is
// hand-written and small. Here the numbering is incidental and shifts whenever
// predicate ordering in the combiner changes.
//
// The output is *semantically* equivalent to `-convert-pdl-to-pdl-interp` on the
// same input but not textually identical, so this cannot be a diff against the
// reference: predicate order differs (the combiner is cost-driven), every
// `record_match` is followed by a continuation block that branches to the
// failure block, and `loc([...])` list order differs. Do not try to make the two
// pipelines converge.

// An empty module still yields a matcher that immediately finalizes.
// CHECK-LABEL: module @empty_module
// CHECK:         pdl_interp.func @matcher
// CHECK:           pdl_interp.finalize

// CHECK-LABEL: module @simple
// CHECK:         pdl_interp.check_operation_name of %arg0 is "foo.op"
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%arg0 : !pdl.operation) : benefit(1), loc([%arg0]), root("foo.op")

// CHECK-LABEL: module @attributes
// CHECK:         pdl_interp.get_attribute "attr" of %arg0
// CHECK:         pdl_interp.check_attribute %{{.*}} is 10 : i64
// CHECK:         pdl_interp.get_attribute_type of
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter

// CHECK-LABEL: module @constraints
// CHECK:         pdl_interp.apply_constraint "multi_constraint"(%{{.*}}, %{{.*}}, %{{.*}} : !pdl.value, !pdl.value, !pdl.value)
// CHECK:         pdl_interp.record_match

// A constraint result is forwarded to the rewriter.
// CHECK-LABEL: module @constraint_with_result
// CHECK:         %[[C:.*]] = pdl_interp.apply_constraint "check_op_and_get_attr_constr"(%arg0 : !pdl.operation) : !pdl.attribute
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%arg0, %[[C]] : !pdl.operation, !pdl.attribute)

// CHECK-LABEL: module @constraint_with_unused_result
// CHECK:         pdl_interp.apply_constraint "check_op_and_get_attr_constr"
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%arg0 : !pdl.operation)

// Two patterns sharing a prefix are combined into one matcher with two
// record_matches, each keeping its own rewriter symbol.
// CHECK-LABEL: module @constraint_with_result_multiple
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter

// CHECK-LABEL: module @negated_constraint
// CHECK:         pdl_interp.apply_constraint "constraint"(%arg0 : !pdl.operation) {isNegated = true}

// CHECK-LABEL: module @inputs
// CHECK:         pdl_interp.check_operand_count of %arg0 is 2
// CHECK:         pdl_interp.are_equal
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @variadic_inputs
// CHECK:         pdl_interp.check_operand_count of %arg0 is at_least 2
// CHECK:         pdl_interp.check_types
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @single_operand_range
// CHECK:         pdl_interp.get_operands of %arg0 : !pdl.range<value>
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @results
// CHECK:         pdl_interp.check_result_count of %arg0 is 2
// CHECK:         pdl_interp.check_type
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @variadic_results
// CHECK:         pdl_interp.check_result_count of %arg0 is at_least 2
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @single_result_range
// CHECK:         pdl_interp.get_results of %arg0 : !pdl.range<value>
// CHECK:         pdl_interp.record_match

// A result used as an operand: navigate to the defining op and tie it back.
// CHECK-LABEL: module @results_as_operands
// CHECK:         pdl_interp.get_defining_op of
// CHECK:         pdl_interp.are_equal
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @single_result_range_as_operands
// CHECK:         pdl_interp.get_defining_op of
// CHECK:         pdl_interp.record_match

//===----------------------------------------------------------------------===//
// The payoff of the combiner: two patterns differing only in a result type
// become a single `pdl_interp.switch_type` instead of two independent chains.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @switch_single_result_type
// CHECK:         pdl_interp.switch_type %{{.*}} to [i32, i64]
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0

// CHECK-LABEL: module @switch_result_types
// CHECK:         pdl_interp.check_types
// CHECK:         pdl_interp.record_match

// Count checks do not fold into a switch; the two alternatives stay distinct.
// CHECK-LABEL: module @switch_operand_count_at_least
// CHECK:         pdl_interp.check_operand_count of %arg0 is at_least
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter
// CHECK:         pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0

// CHECK-LABEL: module @switch_result_count_at_least
// CHECK:         pdl_interp.check_result_count of %arg0 is at_least
// CHECK:         pdl_interp.record_match
// CHECK:         pdl_interp.record_match

// CHECK-LABEL: module @predicate_ordering
// CHECK:         pdl_interp.apply_constraint "typeConstraint"
// CHECK:         pdl_interp.record_match

//===----------------------------------------------------------------------===//
// Multi-root patterns become a `foreach` over the users of the shared value.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @multi_root
// CHECK:         pdl_interp.get_users of
// CHECK:         pdl_interp.foreach %[[E:.*]] : !pdl.operation in
// CHECK:         pdl_interp.record_match @rewriters::@rewrite_multi_root(%arg0, %[[E]] : !pdl.operation, !pdl.operation)

// Overlapping roots need no upward walk at all.
// CHECK-LABEL: module @overlapping_roots
// CHECK-NOT:     pdl_interp.foreach
// CHECK:         pdl_interp.record_match @rewriters::@rewrite_overlapping_roots

// CHECK-LABEL: module @force_overlapped_root
// CHECK:         pdl_interp.foreach
// CHECK:         pdl_interp.record_match @rewriters::@rewrite_forced_overlapped_root

// A range connector needs an `extract` to get a value to walk up from.
// CHECK-LABEL: module @variadic_results_all
// CHECK:         pdl_interp.extract 0 of
// CHECK:         pdl_interp.foreach
// CHECK:         pdl_interp.record_match @rewriters::@variadic_results_all

// CHECK-LABEL: module @variadic_results_at
// CHECK:         pdl_interp.extract 0 of
// CHECK:         pdl_interp.foreach
// CHECK:         pdl_interp.record_match @rewriters::@variadic_results_at

//===----------------------------------------------------------------------===//
// Unbound literals survive the whole pipeline: `pdl.attribute = 10` becomes a
// `match.constant_attribute` and then a `pdl_interp.create_attribute`, matching
// what the reference lowering emits directly.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @attribute_literal
// CHECK:         %[[A:.*]] = pdl_interp.create_attribute 10 : i64
// CHECK:         pdl_interp.apply_constraint "constraint"(%[[A]] : !pdl.attribute)

// CHECK-LABEL: module @type_literal
// CHECK:         %[[T:.*]] = pdl_interp.create_type i32
// CHECK:         %[[TS:.*]] = pdl_interp.create_types [i32, i64]
// CHECK:         pdl_interp.apply_constraint "constraint"(%[[T]], %[[TS]] : !pdl.type, !pdl.range<type>)

// CHECK-LABEL: module @common_connector
// CHECK:         pdl_interp.foreach
// CHECK:         pdl_interp.record_match @rewriters::@common_connector

// CHECK-LABEL: module @common_connector_range
// CHECK:         pdl_interp.extract 0 of
// CHECK:         pdl_interp.foreach
// CHECK:         pdl_interp.record_match @rewriters::@common_connector_range
