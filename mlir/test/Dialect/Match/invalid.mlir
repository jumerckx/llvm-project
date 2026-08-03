// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Constant ops: the ODS attribute constraints. The messages come from
// generated code, so keep the expectations terse.

// `constant_type` takes a type attribute. The custom assembly format cannot
// express a non-`TypeAttr` value, so use the generic form.
// expected-error @below {{failed to satisfy constraint: any type attribute}}
%0 = "match.constant_type"() {value = 10 : i64} : () -> !pdl.type

// -----

// `constant_types` takes an array of type attributes.
// expected-error @below {{failed to satisfy constraint: type array attribute}}
%0 = match.constant_types [i32, 10 : i64]
