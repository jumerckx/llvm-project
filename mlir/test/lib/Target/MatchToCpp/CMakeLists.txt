set(MATCH_TO_CPP_INPUT
  ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/TestMatchToCppPatterns.mlir)
set(MATCH_TO_CPP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TestMatchToCppPatterns.h.inc)
add_custom_command(
  OUTPUT ${MATCH_TO_CPP_OUTPUT}
  COMMAND $<TARGET_FILE:mlir-match-to-cpp> ${MATCH_TO_CPP_INPUT}
          --ods=${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/IR/ArithOps.td
          -I ${MLIR_MAIN_INCLUDE_DIR} -I ${MLIR_INCLUDE_DIR}
          -o ${MATCH_TO_CPP_OUTPUT}
  DEPENDS mlir-match-to-cpp ${MATCH_TO_CPP_INPUT}
  COMMENT "Generating C++ matchers from TestMatchToCppPatterns.mlir"
  )
add_custom_target(MLIRTestMatchToCppPatternsIncGen DEPENDS ${MATCH_TO_CPP_OUTPUT})

add_mlir_library(MLIRTestMatchToCpp
  TestMatchToCpp.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRTestMatchToCppPatternsIncGen
  )
mlir_target_link_libraries(MLIRTestMatchToCpp PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )

target_include_directories(MLIRTestMatchToCpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
