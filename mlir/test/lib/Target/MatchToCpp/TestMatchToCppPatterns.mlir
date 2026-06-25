// Matcher descriptions for the `test-match-to-cpp` pass. At build time these
// are translated to C++ `RewritePattern`s by `mlir-translate --match-to-cpp`;
// the generated `populateGeneratedPatterns` entry point and `rewrite_*` hooks
// are consumed by TestMatchToCpp.cpp.

module {
  // The `success` symbols only need to resolve to *something*; the rewrite side
  // is supplied as hand-written C++ hooks in the test pass, so empty nested
  // modules suffice here.
  module @rewriters {
    module @rename {}
    module @swap {}
  }

  // Rename any `test.original` to `test.renamed`, forwarding the matched op.
  match.matcher @rename_matcher root(%root : !pdl.operation) {
    match.has_name %root, "test.original"
    match.success @rewriters::@rename benefit(1) (%root : !pdl.operation)
  }

  // Match a binary `test.swap`, navigate to both operands, and forward them so
  // the hook can rebuild the op with the operands reversed.
  match.matcher @swap_matcher root(%root : !pdl.operation) {
    match.has_name %root, "test.swap"
    match.check_operand_count %root is 2
    %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %a = match.is_not_null %0 : !match.optional<!pdl.value> -> !pdl.value
    %1 = match.get_operand 1 of %root : !match.optional<!pdl.value>
    %b = match.is_not_null %1 : !match.optional<!pdl.value> -> !pdl.value
    match.success @rewriters::@swap benefit(1) (%root, %a, %b : !pdl.operation, !pdl.value, !pdl.value)
  }
}
