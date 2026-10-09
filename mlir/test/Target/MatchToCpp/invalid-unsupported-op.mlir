// RUN: not mlir-match-to-cpp %s 2>&1 | FileCheck %s

module {
  module @rewriters { module @r {} }

  match.matcher @native_rewrite root(%root : !pdl.operation) {
    // CHECK: error: unsupported match op in C++ matcher emitter
    %a = match.apply_native_rewrite "mk"(%root : !pdl.operation) : !pdl.attribute
    match.has_attr_value %a is 10 : i64
    match.success @rewriters::@r benefit(1)
  }
}
