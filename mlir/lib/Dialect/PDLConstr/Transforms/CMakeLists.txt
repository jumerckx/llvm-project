add_mlir_dialect_library(MLIRPDLConstrTransforms
  CombinePatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PDLConstr/Transforms

  DEPENDS
  MLIRPDLConstrPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRPDLConstrDialect
  MLIRPass
  )
