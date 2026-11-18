; RUN: mlir-translate -import-llvm -prefer-unregistered-intrinsics %s | FileCheck %s

; CHECK-LABEL: llvm.func @lifetime
define void @lifetime() {
  %a = alloca [16 x i8]
  ; CHECK: llvm.call_intrinsic "llvm.lifetime.start.p0"(%[[ptr:.*]]) : (!llvm.ptr {llvm.nonnull}) -> ()
  call void @llvm.lifetime.start.p0(ptr nonnull %a)
  ; CHECK: llvm.call_intrinsic "llvm.lifetime.end.p0"(%[[ptr]]) : (!llvm.ptr {llvm.nonnull}) -> ()
  call void @llvm.lifetime.end.p0(ptr nonnull %a)
  ret void
}

; CHECK-LABEL: llvm.func @tail_call_kind
define i32 @tail_call_kind(i32 %0) {
  ; CHECK: llvm.call_intrinsic "llvm.bswap.i32"(%{{.*}}) {llvm.tail_call_kind = #llvm.tailcallkind<tail>} : (i32) -> i32
  %2 = tail call i32 @llvm.bswap.i32(i32 %0)
  ret i32 %2
}
