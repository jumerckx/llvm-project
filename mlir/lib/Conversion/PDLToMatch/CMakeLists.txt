add_mlir_conversion_library(MLIRPDLToMatch
  PDLToMatch.cpp
  RewriterGen.cpp
  RootOrdering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PDLToMatch

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRMatchDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRPass
  )
