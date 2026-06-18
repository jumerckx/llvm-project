add_mlir_translation_library(MLIRTargetMatchToCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRMatchDialect
  MLIRPDLInterpDialect
  MLIRSupport
  MLIRTranslateLib
  )
