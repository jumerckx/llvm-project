add_mlir_conversion_library(MLIRPDLToMatch
  PDLToMatch.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PDLToMatch

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRMatchDialect
  MLIRPDLInterpDialect
  MLIRPDLToPDLInterp
  MLIRPass
  )
