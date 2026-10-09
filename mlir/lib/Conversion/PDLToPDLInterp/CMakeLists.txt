add_mlir_conversion_library(MLIRPDLToPDLInterp
  PDLToPDLInterp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PDLToPDLInterp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMatchDialect
  MLIRMatchToPDLInterp
  MLIRMatchTransforms
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRPDLToMatch
  MLIRPass
  )
