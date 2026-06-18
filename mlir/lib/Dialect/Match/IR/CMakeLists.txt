add_mlir_dialect_library(MLIRMatchDialect
  Match.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Match

  DEPENDS
  MLIRMatchOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLTypes
  MLIRSideEffectInterfaces
  )
