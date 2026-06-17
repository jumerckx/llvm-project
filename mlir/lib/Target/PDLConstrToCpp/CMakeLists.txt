add_mlir_translation_library(MLIRTargetPDLConstrToCpp
  TranslateRegistration.cpp
  TranslateToCpp.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRPDLConstrDialect
  MLIRPDLInterpDialect
  MLIRSupport
  MLIRTranslateLib
  )
