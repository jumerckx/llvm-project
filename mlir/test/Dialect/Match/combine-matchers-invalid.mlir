// RUN: mlir-opt %s -split-input-file -match-combine-matchers -verify-diagnostics

// `-match-combine-matchers` guards its input shape rather than trying to handle
// the general case. Both refusals below are deliberate.
//
// A consequence worth knowing: the pass is **not idempotent**. Its own output
// contains `try` / `switch_*` ops, so running it twice hits the first guard.

//===----------------------------------------------------------------------===//
// Input matchers must be flat: no `try` or `switch_*`.
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @r {} }
  match.matcher @nested root(%root : !pdl.operation) {
    // expected-error @below {{combine-matchers does not support nested `try`/`switch_*` in input matchers}}
    match.try {
      match.has_name %root, "foo.op"
      match.success @rewriters::@r benefit(1)
    }
    match.success @rewriters::@r benefit(2)
  }
  match.matcher @nested_other root(%root : !pdl.operation) {
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(1)
  }
}

// -----

module {
  module @rewriters { module @r {} }
  match.matcher @nested_switch root(%root : !pdl.operation) {
    // expected-error @below {{combine-matchers does not support nested `try`/`switch_*` in input matchers}}
    match.switch_op_name %root
    case "foo.op" {
      match.success @rewriters::@r benefit(1)
    }
    case "bar.op" {
      match.success @rewriters::@r benefit(2)
    }
  }
  match.matcher @nested_switch_other root(%root : !pdl.operation) {
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(1)
  }
}

// -----

//===----------------------------------------------------------------------===//
// Exactly one `match.success` per input matcher. (Zero is already rejected by
// `MatcherOp`'s own region verifier, so it cannot reach this pass.)
//===----------------------------------------------------------------------===//

module {
  module @rewriters { module @r {} }
  match.matcher @two_succ root(%root : !pdl.operation) {
    match.has_name %root, "foo.op"
    match.success @rewriters::@r benefit(1)
    // expected-error @below {{combine-matchers expects exactly one `match.success` per input matcher}}
    match.success @rewriters::@r benefit(2)
  }
  match.matcher @two_succ_other root(%root : !pdl.operation) {
    match.check_operand_count %root is 1
    match.success @rewriters::@r benefit(1)
  }
}
