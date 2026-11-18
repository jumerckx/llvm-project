; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.func @tailkind()
declare void @tailkind()

; CHECK-LABEL: @call_tailkind
define void @call_tailkind() {
  ; CHECK: llvm.call musttail @tailkind()
  musttail call void @tailkind()
  ret void
}

; // -----

; CHECK: llvm.func @tailkind()
declare void @tailkind()

; CHECK-LABEL: @call_tailkind
define void @call_tailkind() {
  ; CHECK: llvm.call tail @tailkind()
  tail call void @tailkind()
  ret void
}

; // -----

; CHECK: llvm.func @tailkind()
declare void @tailkind()

; CHECK-LABEL: @call_tailkind
define void @call_tailkind() {
  ; CHECK: llvm.call notail @tailkind()
  notail call void @tailkind()
  ret void
}

; // -----

; Intrinsic calls keep their tail call kind as a discardable attribute.

; CHECK-LABEL: @intrinsic_tailkind
define i32 @intrinsic_tailkind(i32 %a, ptr %p, ptr %q, i64 %n) {
  ; CHECK: llvm.intr.bswap(%{{.*}}) {llvm.tail_call_kind = #llvm.tailcallkind<tail>} : (i32) -> i32
  %1 = tail call i32 @llvm.bswap.i32(i32 %a)
  ; CHECK: llvm.intr.ctpop(%{{.*}}) {llvm.tail_call_kind = #llvm.tailcallkind<notail>} : (i32) -> i32
  %2 = notail call i32 @llvm.ctpop.i32(i32 %1)
  ; CHECK: llvm.intr.bitreverse(%{{.*}}) : (i32) -> i32
  %3 = call i32 @llvm.bitreverse.i32(i32 %2)
  ; CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %{{.*}}) <{isVolatile = false}> {llvm.tail_call_kind = #llvm.tailcallkind<tail>}
  tail call void @llvm.memcpy.p0.p0.i64(ptr %p, ptr %q, i64 %n, i1 false)
  ret i32 %3
}
