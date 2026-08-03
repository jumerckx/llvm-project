// RUN: mlir-opt %s -split-input-file -convert-pdl-to-match -verify-diagnostics

// An unbound `pdl.operand` is neither reachable from the pattern root nor a
// constant, so the constraint on it cannot be expressed in the match dialect.
// Silently dropping it would make the pattern match strictly more than it was
// written to, so the conversion fails instead. (The reference
// `-convert-pdl-to-pdl-interp` crashes on this input; see match-test-plan.md
// §5.7.)
module {
  pdl.pattern : benefit(1) {
    %v = pdl.operand

    // expected-error @below {{constraint argument is not reachable from the pattern root and is not a constant}}
    pdl.apply_native_constraint "c"(%v : !pdl.value)

    %root = operation "foo.op"
    rewrite %root with "rewriter"
  }
}

// -----

// A `pdl.attribute` with only a type is not a literal either: it has to be
// bound by navigation.
module {
  pdl.pattern : benefit(1) {
    %type = type : i32
    %attr = attribute : %type

    // expected-error @below {{constraint argument is not reachable from the pattern root and is not a constant}}
    pdl.apply_native_constraint "c"(%attr : !pdl.attribute)

    %root = operation "foo.op"
    rewrite %root with "rewriter"
  }
}
