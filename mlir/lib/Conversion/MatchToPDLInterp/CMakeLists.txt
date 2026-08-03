add_mlir_conversion_library(MLIRMatchToPDLInterp
  MatchToPDLInterp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MatchToPDLInterp

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
