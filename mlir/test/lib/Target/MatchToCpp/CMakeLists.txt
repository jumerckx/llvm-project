# Translate the matcher descriptions to C++ at build time, mirroring how a real
# user would invoke `mlir-translate --match-to-cpp` on their `match` IR.
set(MATCH_TO_CPP_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/TestMatchToCppPatterns.mlir)
set(MATCH_TO_CPP_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TestMatchToCppPatterns.h.inc)
add_custom_command(
  OUTPUT ${MATCH_TO_CPP_OUTPUT}
  COMMAND $<TARGET_FILE:mlir-translate> --match-to-cpp ${MATCH_TO_CPP_INPUT}
          -o ${MATCH_TO_CPP_OUTPUT}
  DEPENDS mlir-translate ${MATCH_TO_CPP_INPUT}
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
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )

target_include_directories(MLIRTestMatchToCpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
