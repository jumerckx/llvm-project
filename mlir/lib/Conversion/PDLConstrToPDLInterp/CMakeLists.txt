add_mlir_conversion_library(MLIRPDLConstrToPDLInterp
  PDLConstrToPDLInterp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PDLConstrToPDLInterp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRPDLConstrDialect
  MLIRPDLInterpDialect
  MLIRPDLToPDLInterp
  MLIRPass
  )
