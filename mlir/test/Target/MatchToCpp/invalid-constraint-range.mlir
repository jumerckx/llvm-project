// RUN: not mlir-match-to-cpp %s 2>&1 | FileCheck %s

// A range-typed constraint argument cannot be packed into the flat
// `ArrayRef<PDLValue>` the generated hook takes.

module {
  module @rewriters { module @r {} }

  match.matcher @constraint_range root(%root : !pdl.operation) {
    %rs = match.get_results of %root : !match.optional<!pdl.range<value>>
    %vs = match.is_not_null %rs : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    // CHECK: error: range arguments to native constraints are not yet supported by the C++ matcher emitter
    match.apply_native_constraint "c"(%vs : !pdl.range<value>)
    match.success @rewriters::@r benefit(1)
  }
}
