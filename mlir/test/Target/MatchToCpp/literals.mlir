// RUN: mlir-match-to-cpp %s | FileCheck %s

// Checks that common builtin attributes/types are emitted as direct
// construction calls instead of round-tripping through the parser.

module {
  module @rewriters {
    module @foo {}
  }

  match.matcher @lit_matcher root(%root : !pdl.operation) {
    // Integer attribute (signless).
    %a0 = match.get_attribute "i" of %root : !match.optional<!pdl.attribute>
    %a1 = match.is_not_null %a0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %a1 is 1 : i32

    // Unsigned + signed integer attributes.
    %b0 = match.get_attribute "u" of %root : !match.optional<!pdl.attribute>
    %b1 = match.is_not_null %b0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %b1 is 7 : ui8

    %c0 = match.get_attribute "s" of %root : !match.optional<!pdl.attribute>
    %c1 = match.is_not_null %c0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %c1 is -3 : si16

    // Index attribute.
    %d0 = match.get_attribute "idx" of %root : !match.optional<!pdl.attribute>
    %d1 = match.is_not_null %d0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %d1 is 5 : index

    // Float attribute.
    %e0 = match.get_attribute "f" of %root : !match.optional<!pdl.attribute>
    %e1 = match.is_not_null %e0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %e1 is 1.500000e+00 : f32

    // Boolean attribute (i1 integer attr).
    %g0 = match.get_attribute "b" of %root : !match.optional<!pdl.attribute>
    %g1 = match.is_not_null %g0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %g1 is true

    // String attribute.
    %h0 = match.get_attribute "name" of %root : !match.optional<!pdl.attribute>
    %h1 = match.is_not_null %h0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %h1 is "hello"

    // Unit attribute.
    %i0 = match.get_attribute "unit" of %root : !match.optional<!pdl.attribute>
    %i1 = match.is_not_null %i0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %i1 is unit

    // Type attribute.
    %j0 = match.get_attribute "t" of %root : !match.optional<!pdl.attribute>
    %j1 = match.is_not_null %j0 : !match.optional<!pdl.attribute> -> !pdl.attribute
    match.has_attr_value %j1 is i64

    // Type check.
    %v0 = match.get_operand 0 of %root : !match.optional<!pdl.value>
    %v1 = match.is_not_null %v0 : !match.optional<!pdl.value> -> !pdl.value
    %vt = match.get_value_type of %v1 : !pdl.value : !pdl.type
    match.has_type %vt, f64

    match.success @rewriters::@foo benefit(1) (%root : !pdl.operation)
  }

  // Standalone constants: the same literal expressions, but bound to a C++
  // variable instead of being folded into a test.
  match.matcher @const_matcher root(%root : !pdl.operation) {
    %attr = match.constant_attribute 10 : i64
    %type = match.constant_type i32
    %types = match.constant_types [i32, i64]

    // A constant range of types is compared against a real one, which is also
    // what pins down that the constant owns its storage.
    %r0 = match.get_results of %root : !match.optional<!pdl.range<value>>
    %r1 = match.is_not_null %r0 : !match.optional<!pdl.range<value>> -> !pdl.range<value>
    %rt = match.get_value_type of %r1 : !pdl.range<value> : !pdl.range<type>
    match.equal %rt, %types : !pdl.range<type>

    match.apply_native_constraint "native_c"(%attr, %type : !pdl.attribute, !pdl.type)
    match.success @rewriters::@foo benefit(1) (%root : !pdl.operation)
  }
}

// CHECK: if (v{{[0-9]+}} != ::mlir::IntegerAttr::get(::mlir::IntegerType::get(op->getContext(), 32), 1))
// CHECK: if (v{{[0-9]+}} != ::mlir::IntegerAttr::get(::mlir::IntegerType::get(op->getContext(), 8, ::mlir::IntegerType::Unsigned), 7u))
// CHECK: if (v{{[0-9]+}} != ::mlir::IntegerAttr::get(::mlir::IntegerType::get(op->getContext(), 16, ::mlir::IntegerType::Signed), -3))
// CHECK: if (v{{[0-9]+}} != ::mlir::IntegerAttr::get(::mlir::IndexType::get(op->getContext()), 5))
// CHECK: if (v{{[0-9]+}} != ::mlir::FloatAttr::get(::mlir::Float32Type::get(op->getContext()), 1.5{{[0-9]*}}))
// CHECK: if (v{{[0-9]+}} != ::mlir::IntegerAttr::get(::mlir::IntegerType::get(op->getContext(), 1), 1))
// CHECK: if (v{{[0-9]+}} != ::mlir::StringAttr::get(op->getContext(), "hello"))
// CHECK: if (v{{[0-9]+}} != ::mlir::UnitAttr::get(op->getContext()))
// CHECK: if (v{{[0-9]+}} != ::mlir::TypeAttr::get(::mlir::IntegerType::get(op->getContext(), 64)))
// CHECK: if (v{{[0-9]+}} != ::mlir::Float64Type::get(op->getContext()))

// The standalone constants of the second matcher.
// CHECK: ::mlir::Attribute v{{[0-9]+}} = ::mlir::IntegerAttr::get(::mlir::IntegerType::get(op->getContext(), 64), 10);
// CHECK: ::mlir::Type v{{[0-9]+}} = ::mlir::IntegerType::get(op->getContext(), 32);
// CHECK: ::llvm::SmallVector<::mlir::Type> v{{[0-9]+}}{::mlir::IntegerType::get(op->getContext(), 32), ::mlir::IntegerType::get(op->getContext(), 64)};
// CHECK: if (!::llvm::equal(v{{[0-9]+}}, v{{[0-9]+}}))
// CHECK: ::mlir::PDLValue {{.*}}[] = {v{{[0-9]+}}, v{{[0-9]+}}};
