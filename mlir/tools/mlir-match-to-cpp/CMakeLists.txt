set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

set(LIBS
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPDLDialect
  MLIRPDLToMatch
  MLIRMatchDialect
  MLIRMatchTransforms
  MLIRTargetMatchToCpp
  MLIRTableGen
  )

add_tablegen(mlir-match-to-cpp MLIR_MATCH_TO_CPP
  DESTINATION "${MLIR_TOOLS_INSTALL_DIR}"
  EXPORT MLIR
  mlir-match-to-cpp.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-match-to-cpp PRIVATE ${LIBS})

mlir_check_all_link_libraries(mlir-match-to-cpp)
