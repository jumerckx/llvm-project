// RUN: mlir-opt %s -split-input-file -convert-pdl-to-match | FileCheck %s

// An unbound literal (`pdl.attribute = <attr>`, `pdl.type : <type>`,
// `pdl.types : [<types>]`) is never reached by navigation from the root, so it
// has to be materialized as a value of its own before it can be passed to a
// native constraint. These are the `match` counterparts of the
// `pdl_interp.create_*` ops checked by
// `test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir` (splits
// `attribute_literal` / `type_literal`) and
// `test/Conversion/PDLToPDLInterp/use-constraint-result.mlir`.

// CHECK-LABEL: module @attribute_literal
module @attribute_literal {
  // CHECK: match.matcher root(
  // CHECK: %[[ATTR:.*]] = constant_attribute 10 : i64
  // CHECK: apply_native_constraint "constraint"(%[[ATTR]] : !pdl.attribute)

  pdl.pattern : benefit(1) {
    %attr = attribute = 10
    pdl.apply_native_constraint "constraint"(%attr : !pdl.attribute)

    %root = operation
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @type_literal
module @type_literal {
  // CHECK: %[[TYPE:.*]] = constant_type i32
  // CHECK: %[[TYPES:.*]] = constant_types [i32, i64]
  // CHECK: apply_native_constraint "constraint"(%[[TYPE]], %[[TYPES]] : !pdl.type, !pdl.range<type>)

  pdl.pattern : benefit(1) {
    %type = type : i32
    %types = types : [i32, i64]
    pdl.apply_native_constraint "constraint"(%type, %types : !pdl.type, !pdl.range<type>)

    %root = operation
    rewrite %root with "rewriter"
  }
}

// -----

// A literal shared by two constraints is materialized once.
// CHECK-LABEL: module @shared_literal
module @shared_literal {
  // CHECK: %[[ATTR:.*]] = constant_attribute 10 : i64
  // CHECK: apply_native_constraint "c1"(%[[ATTR]] : !pdl.attribute)
  // CHECK: apply_native_constraint "c2"(%[[ATTR]] : !pdl.attribute)
  // CHECK-NOT: constant_attribute

  pdl.pattern : benefit(1) {
    %attr = attribute = 10
    pdl.apply_native_constraint "c1"(%attr : !pdl.attribute)
    pdl.apply_native_constraint "c2"(%attr : !pdl.attribute)

    %root = operation
    rewrite %root with "rewriter"
  }
}

// -----

// Literal materialization and constraint-result forwarding in one pattern:
// `use-constraint-result.mlir` split 3.
// CHECK-LABEL: module @literal_into_value_result
module @literal_into_value_result {
  // CHECK: %[[OPERAND:.*]] = is_not_null
  // CHECK: %[[ATTR:.*]] = constant_attribute 10 : i64
  // CHECK: %[[RES:.*]] = apply_native_constraint "return_value_constr"(%[[ATTR]] : !pdl.attribute) : !pdl.value
  // CHECK: equal %[[RES]], %[[OPERAND]] : !pdl.value

  pdl.pattern : benefit(1) {
    %attr = attribute = 10
    %value = pdl.apply_native_constraint "return_value_constr"(%attr : !pdl.attribute) : !pdl.value

    %root = operation(%value : !pdl.value)
    rewrite %root with "rewriter"
  }
}

// -----

// A literal that reaches the constraint through the operation is *not*
// materialized: it is already bound by navigation, and the value constrained by
// `has_attr_value` is what the constraint receives.
// CHECK-LABEL: module @bound_literal
module @bound_literal {
  // CHECK: %[[ATTR:.*]] = is_not_null
  // CHECK: has_attr_value %[[ATTR]] is 10 : i64
  // CHECK: apply_native_constraint "constraint"(%[[ATTR]] : !pdl.attribute)
  // CHECK-NOT: constant_attribute

  pdl.pattern : benefit(1) {
    %attr = attribute = 10
    pdl.apply_native_constraint "constraint"(%attr : !pdl.attribute)

    %root = operation {"a" = %attr}
    rewrite %root with "rewriter"
  }
}
