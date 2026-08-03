// RUN: not mlir-match-to-cpp %s 2>&1 | FileCheck %s

// `match.apply_native_rewrite` has no case in the C++ emitter and so hits the
// `Default` arm. It is supported by the `pdl_interp` lowering, so this asserts
// the emitter fails loudly rather than silently dropping the op — dropping it
// would produce a matcher that accepts more than the pattern describes.

module {
  module @rewriters { module @r {} }

  match.matcher @native_rewrite root(%root : !pdl.operation) {
    // CHECK: error: unsupported match op in C++ matcher emitter
    %a = match.apply_native_rewrite "mk"(%root : !pdl.operation) : !pdl.attribute
    match.has_attr_value %a is 10 : i64
    match.success @rewriters::@r benefit(1)
  }
}
