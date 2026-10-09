
module {
  module @rewriters {
    module @rename {}
    module @swap {}
    module @lower_addf {}
  }

  match.matcher @rename_matcher root(%root : !pdl.operation) {
    match.has_name %root, "test.original"
    match.success @rewriters::@rename benefit(1) (%root : !pdl.operation)
  }

  match.matcher @swap_matcher root(%root : !pdl.operation) {
    match.has_name %root, "test.swap"
    match.check_operand_count %root is 2
    %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %a = match.is_not_null %0 : !pdl.value
    %1 = match.get_operand 1 of %root : !match.optional<!pdl.value>
    %b = match.is_not_null %1 : !pdl.value
    match.success @rewriters::@swap benefit(1) (%root, %a, %b : !pdl.operation, !pdl.value, !pdl.value)
  }

  match.matcher @addf_matcher root(%root : !pdl.operation) {
    match.has_name %root, "arith.addf"
    match.check_operand_count %root is 2
    %0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %a = match.is_not_null %0 : !pdl.value
    %1 = match.get_operand 1 of %root : !match.optional<!pdl.value>
    %b = match.is_not_null %1 : !pdl.value
    match.success @rewriters::@lower_addf benefit(1) (%root, %a, %b : !pdl.operation, !pdl.value, !pdl.value)
  }
}
