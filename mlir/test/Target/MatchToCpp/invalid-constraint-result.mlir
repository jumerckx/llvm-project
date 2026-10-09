// RUN: not mlir-match-to-cpp %s 2>&1 | FileCheck %s

module {
  module @rewriters { module @r {} }

  match.matcher @constraint_result root(%root : !pdl.operation) {
    // CHECK: error: result-producing native constraints are not yet supported by the C++ matcher emitter
    %a = match.apply_native_constraint "c"(%root : !pdl.operation) : !pdl.attribute
    match.success @rewriters::@r benefit(1)
  }
}
