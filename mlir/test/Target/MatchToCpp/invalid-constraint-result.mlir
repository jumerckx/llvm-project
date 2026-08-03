// RUN: not mlir-match-to-cpp %s 2>&1 | FileCheck %s

// A native constraint that produces results has no C++ emission yet: the
// generated hook signature passes a `PDLResultList` sized 0. This is a
// documented limitation, not a bug — the `pdl_interp` path does support it.
//
// The tool has no -split-input-file, so each diagnostic gets its own file.

module {
  module @rewriters { module @r {} }

  match.matcher @constraint_result root(%root : !pdl.operation) {
    // CHECK: error: result-producing native constraints are not yet supported by the C++ matcher emitter
    %a = match.apply_native_constraint "c"(%root : !pdl.operation) : !pdl.attribute
    match.success @rewriters::@r benefit(1)
  }
}
