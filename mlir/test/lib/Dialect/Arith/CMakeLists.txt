add_mlir_pdll_library(MLIRTestArithCanonicalizationPDLLIncGen
  ArithCanonicalization.pdll
  ArithCanonicalizationPDLL.h.inc
  )

set(ARITH_PDLL_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/ArithCanonicalization.pdll)
set(ARITH_PDL_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ArithCanonicalization.pdl.mlir)
set(ARITH_ODS ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/IR/ArithOps.td)
add_custom_command(
  OUTPUT ${ARITH_PDL_OUTPUT}
  COMMAND ${MLIR_PDLL_TABLEGEN_EXE} ${ARITH_PDLL_INPUT} -x=mlir
          -I ${MLIR_MAIN_INCLUDE_DIR} -I ${MLIR_INCLUDE_DIR}
          -o ${ARITH_PDL_OUTPUT}
  DEPENDS ${MLIR_PDLL_TABLEGEN_TARGET} ${ARITH_PDLL_INPUT}
  COMMENT "Generating PDL from ArithCanonicalization.pdll"
  )

set(ARITH_MATCH_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ArithCanonicalizationMatch.h.inc)
add_custom_command(
  OUTPUT ${ARITH_MATCH_OUTPUT}
  COMMAND $<TARGET_FILE:mlir-match-to-cpp> ${ARITH_PDL_OUTPUT}
          --ods=${ARITH_ODS} -I ${MLIR_MAIN_INCLUDE_DIR} -I ${MLIR_INCLUDE_DIR}
          --namespace=match_cpp -o ${ARITH_MATCH_OUTPUT}
  DEPENDS mlir-match-to-cpp ${ARITH_PDL_OUTPUT}
  COMMENT "Generating combined C++ patterns from ArithCanonicalization.pdll"
  )

set(ARITH_MATCH_SPLIT_OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/ArithCanonicalizationMatchSplit.h.inc)
add_custom_command(
  OUTPUT ${ARITH_MATCH_SPLIT_OUTPUT}
  COMMAND $<TARGET_FILE:mlir-match-to-cpp> ${ARITH_PDL_OUTPUT}
          --ods=${ARITH_ODS} -I ${MLIR_MAIN_INCLUDE_DIR} -I ${MLIR_INCLUDE_DIR}
          --combine-matchers=false --namespace=match_cpp_split
          -o ${ARITH_MATCH_SPLIT_OUTPUT}
  DEPENDS mlir-match-to-cpp ${ARITH_PDL_OUTPUT}
  COMMENT "Generating per-pattern C++ patterns from ArithCanonicalization.pdll"
  )
add_custom_target(MLIRTestArithCanonicalizationMatchIncGen
  DEPENDS ${ARITH_MATCH_OUTPUT} ${ARITH_MATCH_SPLIT_OUTPUT})

# Exclude tests from libMLIR.so
add_mlir_library(MLIRArithTestPasses
  TestArithCanonicalizationPDLL.cpp
  TestEmulateWideInt.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRArithCanonicalizationIncGen
  MLIRTestArithCanonicalizationPDLLIncGen
  MLIRTestArithCanonicalizationMatchIncGen
)
mlir_target_link_libraries(MLIRArithTestPasses PUBLIC
  MLIRArithDialect
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRLLVMDialect
  MLIRMatchDialect
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
  ${CMAKE_CURRENT_BINARY_DIR}
  )
