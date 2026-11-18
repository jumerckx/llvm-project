// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: declare i32 @foo()
llvm.func @foo() -> i32

// CHECK-LABEL: @test_none
llvm.func @test_none() -> i32 {
  // CHECK-NEXT: call i32 @foo()
  %0 = llvm.call none @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_default
llvm.func @test_default() -> i32 {
  // CHECK-NEXT: call i32 @foo()
  %0 = llvm.call @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_musttail
llvm.func @test_musttail() -> i32 {
  // CHECK-NEXT: musttail call i32 @foo()
  %0 = llvm.call musttail @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_tail
llvm.func @test_tail() -> i32 {
  // CHECK-NEXT: tail call i32 @foo()
  %0 = llvm.call tail @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_notail
llvm.func @test_notail() -> i32 {
  // CHECK-NEXT: notail call i32 @foo()
  %0 = llvm.call notail @foo() : () -> i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @test_intrinsic
llvm.func @test_intrinsic(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i64) -> i32 {
  // CHECK-NEXT: tail call i32 @llvm.bswap.i32(i32 %{{.*}})
  %0 = llvm.intr.bswap(%arg0) {llvm.tail_call_kind = #llvm.tailcallkind<tail>} : (i32) -> i32
  // CHECK-NEXT: notail call i32 @llvm.ctpop.i32(i32 %{{.*}})
  %1 = llvm.intr.ctpop(%0) {llvm.tail_call_kind = #llvm.tailcallkind<notail>} : (i32) -> i32
  // CHECK-NEXT: call i32 @llvm.bitreverse.i32(i32 %{{.*}})
  %2 = llvm.intr.bitreverse(%1) : (i32) -> i32
  // CHECK-NEXT: tail call void @llvm.memcpy.p0.p0.i64(ptr %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i1 false)
  "llvm.intr.memcpy"(%arg1, %arg2, %arg3) <{isVolatile = false}> {llvm.tail_call_kind = #llvm.tailcallkind<tail>} : (!llvm.ptr, !llvm.ptr, i64) -> ()
  // CHECK-NEXT: tail call i32 @llvm.bswap.i32(i32 %{{.*}})
  %3 = llvm.call_intrinsic "llvm.bswap.i32"(%2) {llvm.tail_call_kind = #llvm.tailcallkind<tail>} : (i32) -> i32
  llvm.return %3 : i32
}
