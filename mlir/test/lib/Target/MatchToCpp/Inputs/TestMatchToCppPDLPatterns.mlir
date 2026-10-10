module {
  pdl.pattern @fold_add : benefit(1) {
    %attr = pdl.attribute
    %type = pdl.type
    %operand = pdl.operand
    %root = pdl.operation "test.fold_add"(%operand : !pdl.value) {"value" = %attr} -> (%type : !pdl.type)
    %step = pdl.apply_native_constraint "SmallInt"(%attr : !pdl.attribute) : !pdl.attribute
    pdl.apply_native_constraint "IsEven"(%attr : !pdl.attribute)
    pdl.rewrite %root {
      %sum = pdl.apply_native_rewrite "AddInts"(%attr, %step : !pdl.attribute, !pdl.attribute) : !pdl.attribute
      %none = pdl.apply_native_rewrite "NullAttr" : !pdl.attribute
      %new = pdl.operation "test.folded"(%operand : !pdl.value) {"value" = %sum, "dropped" = %none} -> (%type : !pdl.type)
      pdl.replace %root with %new
    }
  }

  pdl.pattern @to_addi : benefit(1) {
    %type = pdl.type
    %lhs = pdl.operand
    %rhs = pdl.operand
    %root = pdl.operation "test.add"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %root {
      %add = pdl.operation "arith.addi"(%lhs, %rhs : !pdl.value, !pdl.value)
      %one = pdl.attribute = 1 : i32
      %cst = pdl.operation "arith.constant" {"value" = %one}
      %sum = pdl.result 0 of %add
      %c = pdl.result 0 of %cst
      %pair = pdl.operation "test.pair"(%sum, %c : !pdl.value, !pdl.value) -> (%type : !pdl.type)
      pdl.replace %root with %pair
    }
  }

  pdl.pattern @generic : benefit(1) {
    %type = pdl.type
    %operand = pdl.operand
    %root = pdl.operation "test.pick"(%operand : !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %root {
      %new = pdl.operation "test.picked_generic"(%operand : !pdl.value) -> (%type : !pdl.type)
      pdl.replace %root with %new
    }
  }

  pdl.pattern @specific : benefit(2) {
    %type = pdl.type
    %src = pdl.operation "test.src" -> (%type : !pdl.type)
    %value = pdl.result 0 of %src
    %root = pdl.operation "test.pick"(%value : !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %root {
      %new = pdl.operation "test.picked_specific"(%value : !pdl.value) -> (%type : !pdl.type)
      pdl.replace %root with %new
    }
  }

  pdl.pattern @split : benefit(1) {
    %types = pdl.types
    %operands = pdl.operands
    %root = pdl.operation "test.split"(%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
    pdl.rewrite %root {
      %first = pdl.operation "test.first"(%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
      %results = pdl.results of %first
      pdl.replace %root with (%results : !pdl.range<value>)
    }
  }

  pdl.pattern @external : benefit(1) {
    %root = pdl.operation "test.external"
    pdl.rewrite %root with "EraseOp"
  }
}
