add_mlir_library(MLIRTargetMatchToCpp
  TranslateToCpp.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRMatchDialect
  MLIRSupport
  )
