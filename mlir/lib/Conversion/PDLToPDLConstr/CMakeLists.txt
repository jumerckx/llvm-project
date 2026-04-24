add_mlir_conversion_library(MLIRPDLToPDLConstr
  PDLToPDLConstr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PDLToPDLConstr

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRPDLConstrDialect
  MLIRPass
  )
