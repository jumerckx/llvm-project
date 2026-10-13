add_mlir_pdll_library(MLIRTestArithCanonicalizationPDLLIncGen
  ArithCanonicalization.pdll
  ArithCanonicalizationPDLL.h.inc
  )

# Exclude tests from libMLIR.so
add_mlir_library(MLIRArithTestPasses
  TestArithCanonicalizationPDLL.cpp
  TestEmulateWideInt.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRArithCanonicalizationIncGen
  MLIRTestArithCanonicalizationPDLLIncGen
)
mlir_target_link_libraries(MLIRArithTestPasses PUBLIC
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRPass
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRTransformUtils
  MLIRVectorDialect
)

# For the DRR-generated ArithCanonicalization.inc.
target_include_directories(MLIRArithTestPasses
  PRIVATE
  ${MLIR_BINARY_DIR}/lib/Dialect/Arith/IR
  )
