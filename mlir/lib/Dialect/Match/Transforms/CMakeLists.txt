add_mlir_dialect_library(MLIRMatchTransforms
  CombineMatchers.cpp
  MatchTransformsUtils.cpp
  RegroupOperationAccesses.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Match/Transforms

  DEPENDS
  MLIRMatchPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRMatchDialect
  MLIRPass
  )
