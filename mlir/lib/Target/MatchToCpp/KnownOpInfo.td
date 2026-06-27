//===- KnownOpInfo.td - Dialects with op-info for match-to-cpp -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Op definitions for which `mlir-translate --match-to-cpp` knows the concrete
// C++ op class and can therefore emit DRR-style `dyn_cast` matchers. The
// `-gen-op-info` backend turns these into a `populate*OpInfo` table; add more
// `include`s (and matching tablegen calls in CMakeLists.txt) to support more
// dialects.
//
//===----------------------------------------------------------------------===//

include "mlir/Dialect/Arith/IR/ArithOps.td"
