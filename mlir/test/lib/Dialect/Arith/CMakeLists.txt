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
set(ARITH_DRR_INPUT ${MLIR_BINARY_DIR}/lib/Dialect/Arith/IR/ArithCanonicalization.inc)
set(ARITH_DRR_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ArithCanonicalizationDRR.inc)
add_custom_command(
  OUTPUT ${ARITH_DRR_OUTPUT}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${ARITH_DRR_INPUT} -DOUTPUT=${ARITH_DRR_OUTPUT}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/InstrumentDRRRewrites.cmake
  DEPENDS MLIRArithCanonicalizationIncGen ${ARITH_DRR_INPUT}
          ${CMAKE_CURRENT_SOURCE_DIR}/InstrumentDRRRewrites.cmake
  COMMENT "Instrumenting the rewrites of the DRR arith canonicalization patterns"
  )
add_custom_target(MLIRTestArithCanonicalizationMatchIncGen
  DEPENDS ${ARITH_MATCH_OUTPUT} ${ARITH_MATCH_SPLIT_OUTPUT} ${ARITH_DRR_OUTPUT})

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

target_include_directories(MLIRArithTestPasses
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  )
