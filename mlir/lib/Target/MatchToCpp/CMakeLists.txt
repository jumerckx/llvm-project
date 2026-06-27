# Generate the op-info table that lets the translator emit concrete-typed
# (`dyn_cast`) matchers for known dialects. The `-op-info-include` path is the
# header the *generated matcher* must include to see the op classes.
set(LLVM_TARGET_DEFINITIONS KnownOpInfo.td)
mlir_tablegen(ArithOpInfo.inc -gen-op-info
  -op-info-func-name=populateArithOpInfo
  -op-info-include=mlir/Dialect/Arith/IR/Arith.h)
add_public_tablegen_target(MLIRMatchToCppOpInfoIncGen)

add_mlir_translation_library(MLIRTargetMatchToCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  DEPENDS
  MLIRMatchToCppOpInfoIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRMatchDialect
  MLIRPDLInterpDialect
  MLIRSupport
  MLIRTranslateLib
  )

target_include_directories(MLIRTargetMatchToCpp PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR})
