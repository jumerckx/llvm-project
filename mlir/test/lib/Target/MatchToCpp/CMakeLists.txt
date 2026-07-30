# Translate the matcher descriptions to C++ at build time, mirroring how a real
# user would invoke `mlir-match-to-cpp` on their `match` IR. `--ods` points the
# tool at the Arith ODS definitions so `arith.addf` is matched via its concrete
# C++ class (exercising the DRR-style dyn_cast / check-elision path end to end);
# the dialect header is included by TestMatchToCpp.cpp, not the generated file.
#
# The input lives under Inputs/ because lit excludes that directory: it is a
# build-time input, not a standalone test, so it has no RUN line.
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

# Exclude tests from libMLIR.so
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
