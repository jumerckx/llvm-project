//===-- LLVMIR.h - C Interface for MLIR LLVMIR Target ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Target/LLVMIR.h"
#include "llvm-c/Support.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirOperation module,
                                          LLVMContextRef context) {
  Operation *moduleOp = unwrap(module);

  llvm::LLVMContext *ctx = llvm::unwrap(context);

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, *ctx);

  LLVMModuleRef moduleRef = llvm::wrap(llvmModule.release());

  return moduleRef;
}

MlirStringRef mlirSerializeGPUModuleOp(MlirOperation op) {
  
  auto op_unwrapped = dyn_cast<gpu::GPUModuleOp>(unwrap(op));
  auto targets = op_unwrapped.getTargetsAttr();
  auto target = dyn_cast<gpu::TargetAttrInterface>(targets.getValue()[0]);
  assert(target && "Target attribute doesn't implements `TargetAttrInterface`.");

  gpu::TargetOptions options("", {}, "", gpu::CompilationTarget::Assembly);

  std::optional<llvm::SmallVector<char, 0U>> serialized = target.serializeToObject(op_unwrapped, options);
  
  if (serialized.has_value()) {
    auto data = serialized->data();
    auto size = serialized->size();
    char* c_data = static_cast<char*>(std::malloc(size));
    std::memcpy(c_data, data, size);
    return mlirStringRefCreate(c_data, size);
  }
  else {
    return mlirStringRefCreate(nullptr, 0);
  }
}
