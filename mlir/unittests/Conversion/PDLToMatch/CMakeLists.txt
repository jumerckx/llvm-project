add_mlir_unittest(MLIRPDLToMatchTests
  RootOrderingTest.cpp
)
mlir_target_link_libraries(MLIRPDLToMatchTests
  PRIVATE
  MLIRArithDialect
  MLIRPDLToMatch
)
