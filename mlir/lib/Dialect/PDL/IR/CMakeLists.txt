set(LLVM_OPTIONAL_SOURCES
  PDL.cpp
  PDLTypes.cpp
  )

add_mlir_dialect_library(MLIRPDLTypes
  PDLTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PDL

  DEPENDS
  MLIRPDLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

add_mlir_dialect_library(MLIRPDLDialect
  PDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PDL

  DEPENDS
  MLIRPDLOpsIncGen
  MLIRMatchOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRMatchDialect
  MLIRPDLTypes
  MLIRSideEffectInterfaces
  )
