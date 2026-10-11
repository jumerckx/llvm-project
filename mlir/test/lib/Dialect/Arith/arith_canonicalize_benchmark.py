#!/usr/bin/env python3
# ===- arith_canonicalize_benchmark.py - Arith canonicalization benchmark --===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

import argparse
import random
import re
import statistics
import subprocess
import sys

INT_TYPES = ["i8", "i16", "i32", "i64"]
TYPES = ["i1", "i8", "i16", "i32", "i64", "index", "f32"]
ACCUMULATE = {
    "i1": ["arith.maxui", "arith.minui"],
    "index": ["arith.maxsi", "arith.minui", "arith.divui"],
    "f32": ["arith.maximumf", "arith.minimumf"],
}
for _t in INT_TYPES:
    ACCUMULATE[_t] = ["arith.maxsi", "arith.minui", "arith.remui", "arith.shrui"]

VARIANTS = ["drr", "pdll", "match-cpp", "match-cpp-split"]


class FunctionBuilder:
    def __init__(self, rng, name):
        self.rng = rng
        self.name = name
        self.lines = []
        self.counter = 0
        self.args = []
        self.pool = {}
        self.acc = {}
        for t in TYPES:
            self.pool[t] = []
            for _ in range(3):
                arg = f"%a{len(self.args)}"
                self.args.append((arg, t))
                self.pool[t].append(arg)
            self.acc[t] = self.pool[t][0]
        self.num_ops = 0

    def emit(self, text, results=1):
        names = [f"%v{self.counter + i}" for i in range(results)]
        self.counter += results
        self.lines.append(f"    {', '.join(names)} = {text}")
        self.num_ops += 1
        return names[0] if results == 1 else names

    def opaque(self, t, n=1):
        values = self.rng.sample(self.pool[t], n)
        return values[0] if n == 1 else values

    def const(self, value, t):
        if t == "i1":
            return self.emit(f"arith.constant {value}")
        return self.emit(f"arith.constant {value} : {t}")

    def sink(self, value, t):
        op = self.rng.choice(ACCUMULATE[t])
        acc = self.emit(f"{op} {value}, {self.acc[t]} : {t}")
        self.acc[t] = acc
        self.pool[t].append(acc)

    def finish(self):
        args = ", ".join(f"{a}: {t}" for a, t in self.args)
        results = ", ".join(TYPES)
        values = ", ".join(self.acc[t] for t in TYPES)
        out = [f"  func.func @{self.name}({args}) -> ({results}) {{"]
        out += self.lines
        out.append(f"    return {values} : {results}")
        out.append("  }")
        return "\n".join(out)


def int_type(b):
    return b.rng.choice(["i32", "i64"])


def match_addi_addi_constant(b):
    t = int_type(b)
    x = b.opaque(t)
    inner = b.emit(f"arith.addi {x}, {b.const(b.rng.randint(1, 99), t)} : {t}")
    return [(b.emit(f"arith.addi {inner}, {b.const(b.rng.randint(1, 99), t)} : {t}"), t)]


def match_subi_subi_constant(b):
    t = int_type(b)
    x = b.opaque(t)
    inner = b.emit(f"arith.subi {x}, {b.const(b.rng.randint(1, 99), t)} : {t}")
    return [(b.emit(f"arith.subi {inner}, {b.const(b.rng.randint(1, 99), t)} : {t}"), t)]


def match_select_cmpi_to_min_max(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    pred = b.rng.choice(["slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"])
    cmp = b.emit(f"arith.cmpi {pred}, {x}, {y} : {t}")
    return [(b.emit(f"arith.select {cmp}, {x}, {y} : {t}"), t)]


def match_xori_not_cmpi(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    cmp = b.emit(f"arith.cmpi slt, {x}, {y} : {t}")
    return [(b.emit(f"arith.xori {cmp}, {b.const('true', 'i1')} : i1"), "i1")]


def match_select_not_cond(b):
    t = int_type(b)
    c = b.opaque("i1")
    x, y = b.opaque(t, 2)
    n = b.emit(f"arith.xori {c}, {b.const('true', 'i1')} : i1")
    return [(b.emit(f"arith.select {n}, {x}, {y} : {t}"), t)]


def match_trunci_extsi(b):
    x = b.opaque("i16")
    ext = b.emit(f"arith.extsi {x} : i16 to i64")
    return [(b.emit(f"arith.trunci {ext} : i64 to i32"), "i32")]


def match_index_cast_extsi(b):
    x = b.opaque("i16")
    ext = b.emit(f"arith.extsi {x} : i16 to i32")
    return [(b.emit(f"arith.index_cast {ext} : i32 to index"), "index")]


def match_sitofp_extsi(b):
    x = b.opaque("i16")
    ext = b.emit(f"arith.extsi {x} : i16 to i32")
    return [(b.emit(f"arith.sitofp {ext} : i32 to f32"), "f32")]


def match_andi_extui(b):
    x, y = b.opaque("i16", 2)
    ex = b.emit(f"arith.extui {x} : i16 to i32")
    ey = b.emit(f"arith.extui {y} : i16 to i32")
    op = b.rng.choice(["arith.andi", "arith.ori", "arith.xori"])
    return [(b.emit(f"{op} {ex}, {ey} : i32"), "i32")]


def match_addf_negf(b):
    x, y = b.opaque("f32", 2)
    neg = b.emit(f"arith.negf {x} : f32")
    return [(b.emit(f"arith.addf {neg}, {y} : f32"), "f32")]


def match_mulsi_extended(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    low, _ = b.emit(f"arith.mulsi_extended {x}, {y} : {t}", results=2)
    return [(low, t)]


def miss_addi_addi(b):
    t = int_type(b)
    x, y, z = b.opaque(t, 3)
    inner = b.emit(f"arith.addi {x}, {y} : {t}")
    return [(b.emit(f"arith.addi {inner}, {z} : {t}"), t)]


def miss_addi_muli(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    mul = b.emit(f"arith.muli {y}, {b.const(b.rng.randint(2, 99), t)} : {t}")
    return [(b.emit(f"arith.addi {x}, {mul} : {t}"), t)]


def miss_subi_subi(b):
    t = int_type(b)
    x, y, z = b.opaque(t, 3)
    inner = b.emit(f"arith.subi {x}, {y} : {t}")
    return [(b.emit(f"arith.subi {inner}, {z} : {t}"), t)]


def miss_select_cmpi_operands(b):
    t = int_type(b)
    x, y, z = b.opaque(t, 3)
    cmp = b.emit(f"arith.cmpi slt, {x}, {y} : {t}")
    return [(b.emit(f"arith.select {cmp}, {x}, {z} : {t}"), t)]


def miss_select_cmpi_predicate(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    pred = b.rng.choice(["eq", "ne"])
    cmp = b.emit(f"arith.cmpi {pred}, {x}, {y} : {t}")
    return [(b.emit(f"arith.select {cmp}, {x}, {y} : {t}"), t)]


def miss_xori_cmpi(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    cmp = b.emit(f"arith.cmpi slt, {x}, {y} : {t}")
    return [(b.emit(f"arith.xori {cmp}, {b.opaque('i1')} : i1"), "i1")]


def miss_cmpi_ext_types(b):
    x = b.opaque("i8")
    y = b.opaque("i16")
    op = b.rng.choice(["arith.extsi", "arith.extui"])
    ex = b.emit(f"{op} {x} : i8 to i32")
    ey = b.emit(f"{op} {y} : i16 to i32")
    return [(b.emit(f"arith.cmpi eq, {ex}, {ey} : i32"), "i1")]


def miss_cmpi_ext_predicate(b):
    x, y = b.opaque("i16", 2)
    op = b.rng.choice(["arith.extsi", "arith.extui"])
    ex = b.emit(f"{op} {x} : i16 to i32")
    ey = b.emit(f"{op} {y} : i16 to i32")
    return [(b.emit(f"arith.cmpi slt, {ex}, {ey} : i32"), "i1")]


def miss_trunci_shrsi(b):
    x = b.opaque("i64")
    shr = b.emit(f"arith.shrsi {x}, {b.const(b.rng.randint(1, 31), 'i64')} : i64")
    return [(b.emit(f"arith.trunci {shr} : i64 to i32"), "i32")]


def miss_mulsi_extended(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    low, high = b.emit(f"arith.mulsi_extended {x}, {y} : {t}", results=2)
    return [(low, t), (high, t)]


def miss_mulf_negf(b):
    x, y = b.opaque("f32", 2)
    neg = b.emit(f"arith.negf {x} : f32")
    op = b.rng.choice(["arith.mulf", "arith.divf"])
    return [(b.emit(f"{op} {neg}, {y} : f32"), "f32")]


def miss_andi_extui_types(b):
    x = b.opaque("i8")
    y = b.opaque("i16")
    ex = b.emit(f"arith.extui {x} : i8 to i32")
    ey = b.emit(f"arith.extui {y} : i16 to i32")
    op = b.rng.choice(["arith.andi", "arith.ori", "arith.xori"])
    return [(b.emit(f"{op} {ex}, {ey} : i32"), "i32")]


def miss_index_cast_extui(b):
    x = b.opaque("i16")
    ext = b.emit(f"arith.extui {x} : i16 to i32")
    return [(b.emit(f"arith.index_cast {ext} : i32 to index"), "index")]


def shallow_binary(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    op = b.rng.choice(
        ["arith.addi", "arith.subi", "arith.muli", "arith.andi", "arith.ori",
         "arith.xori"]
    )
    return [(b.emit(f"{op} {x}, {y} : {t}"), t)]


def shallow_float(b):
    x, y = b.opaque("f32", 2)
    op = b.rng.choice(["arith.addf", "arith.subf", "arith.mulf", "arith.divf"])
    return [(b.emit(f"{op} {x}, {y} : f32"), "f32")]


def shallow_cmpi(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    pred = b.rng.choice(["eq", "ne", "slt", "ugt"])
    return [(b.emit(f"arith.cmpi {pred}, {x}, {y} : {t}"), "i1")]


def shallow_select(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    return [(b.emit(f"arith.select {b.opaque('i1')}, {x}, {y} : {t}"), t)]


def shallow_cast(b):
    kind = b.rng.randrange(6)
    if kind == 0:
        return [(b.emit(f"arith.extsi {b.opaque('i16')} : i16 to i32"), "i32")]
    if kind == 1:
        return [(b.emit(f"arith.trunci {b.opaque('i64')} : i64 to i32"), "i32")]
    if kind == 2:
        return [(b.emit(f"arith.sitofp {b.opaque('i32')} : i32 to f32"), "f32")]
    if kind == 3:
        return [(b.emit(f"arith.uitofp {b.opaque('i32')} : i32 to f32"), "f32")]
    if kind == 4:
        return [(b.emit(f"arith.index_cast {b.opaque('i32')} : i32 to index"),
                 "index")]
    return [(b.emit(f"arith.bitcast {b.opaque('f32')} : f32 to i32"), "i32")]


def shallow_extended(b):
    t = int_type(b)
    x, y = b.opaque(t, 2)
    op = b.rng.choice(["arith.addui_extended", "arith.mului_extended"])
    first, second = b.emit(
        f"{op} {x}, {y} : {t}" + (", i1" if op == "arith.addui_extended" else ""),
        results=2,
    )
    return [(first, t), (second, "i1" if op == "arith.addui_extended" else t)]


def unrelated(b):
    kind = b.rng.randrange(5)
    if kind == 0:
        t = b.rng.choice(INT_TYPES)
        x, y = b.opaque(t, 2)
        op = b.rng.choice(["arith.divsi", "arith.remsi", "arith.shli",
                           "arith.maxui", "arith.minsi"])
        return [(b.emit(f"{op} {x}, {y} : {t}"), t)]
    if kind == 1:
        x, y = b.opaque("f32", 2)
        pred = b.rng.choice(["olt", "oeq", "uge"])
        return [(b.emit(f"arith.cmpf {pred}, {x}, {y} : f32"), "i1")]
    if kind == 2:
        return [(b.emit(f"arith.negf {b.opaque('f32')} : f32"), "f32")]
    if kind == 3:
        return [(b.emit(f"arith.fptosi {b.opaque('f32')} : f32 to i32"), "i32")]
    x, y = b.opaque("f32", 2)
    return [(b.emit(f"arith.remf {x}, {y} : f32"), "f32")]


MATCHES = [
    match_addi_addi_constant,
    match_subi_subi_constant,
    match_select_cmpi_to_min_max,
    match_xori_not_cmpi,
    match_select_not_cond,
    match_trunci_extsi,
    match_index_cast_extsi,
    match_sitofp_extsi,
    match_andi_extui,
    match_addf_negf,
    match_mulsi_extended,
]
DEEP_MISSES = [
    miss_addi_addi,
    miss_addi_muli,
    miss_subi_subi,
    miss_select_cmpi_operands,
    miss_select_cmpi_predicate,
    miss_xori_cmpi,
    miss_cmpi_ext_types,
    miss_cmpi_ext_predicate,
    miss_trunci_shrsi,
    miss_mulsi_extended,
    miss_mulf_negf,
    miss_andi_extui_types,
    miss_index_cast_extui,
]
SHALLOW_MISSES = [
    shallow_binary,
    shallow_float,
    shallow_cmpi,
    shallow_select,
    shallow_cast,
    shallow_extended,
]


def generate(args, out):
    rng = random.Random(args.seed)
    weights = [
        args.match_rate,
        args.deep_miss_rate,
        args.shallow_miss_rate,
        max(0.0, 1.0 - args.match_rate - args.deep_miss_rate -
            args.shallow_miss_rate),
    ]
    groups = [MATCHES, DEEP_MISSES, SHALLOW_MISSES, [unrelated]]
    counts = [0, 0, 0, 0]
    num_ops = 0
    out.write("module {\n")
    for f in range(args.functions):
        b = FunctionBuilder(rng, f"f{f}")
        for _ in range(args.templates):
            group = rng.choices(range(4), weights)[0]
            counts[group] += 1
            for value, t in rng.choice(groups[group])(b):
                b.sink(value, t)
        num_ops += b.num_ops + 1
        out.write(b.finish())
        out.write("\n")
    out.write("}\n")
    print(
        f"generated {args.functions} functions, {num_ops} ops; templates: "
        f"{counts[0]} matches, {counts[1]} deep misses, {counts[2]} shallow "
        f"misses, {counts[3]} unrelated",
        file=sys.stderr,
    )


def run(args):
    phases = ["total", "driver", "match", "rewrite"]
    results = {}
    for variant in args.variants:
        samples = []
        counts = None
        for _ in range(args.repetitions):
            cmd = [
                args.mlir_opt,
                args.input,
                f"-test-arith-canonicalize-{variant}=phase-timing "
                f"fold={'true' if args.fold else 'false'}",
                "-mlir-disable-threading",
                "-o",
                "/dev/null",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            times = {}
            for line in proc.stderr.splitlines():
                m = re.match(r"\s*(\w+)\s+([0-9.]+) s$", line)
                if m:
                    times[m.group(1)] = float(m.group(2))
                m = re.match(r"\s*(\d+) pattern applications, (\d+) rewrites", line)
                if m:
                    counts = (int(m.group(1)), int(m.group(2)))
            samples.append(times)
        results[variant] = (
            {p: statistics.median(s[p] for s in samples) for p in phases},
            counts,
        )

    print(f"median of {args.repetitions} runs (seconds)")
    print(f"{'variant':<16}" + "".join(f"{p:>10}" for p in phases) +
          f"{'applications':>14}{'rewrites':>10}")
    for variant, (times, counts) in results.items():
        print(f"{variant:<16}" + "".join(f"{times[p]:>10.4f}" for p in phases) +
              f"{counts[0]:>14}{counts[1]:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and run the arith canonicalization benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a benchmark input")
    gen.add_argument("-o", "--output", default="-")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--functions", type=int, default=1000)
    gen.add_argument("--templates", type=int, default=200,
                     help="pattern templates per function")
    gen.add_argument("--match-rate", type=float, default=0.01,
                     help="fraction of templates that match a pattern")
    gen.add_argument("--deep-miss-rate", type=float, default=0.3,
                     help="fraction of templates that fail late in a pattern")
    gen.add_argument("--shallow-miss-rate", type=float, default=0.4,
                     help="fraction of templates rooted at a pattern root kind "
                          "that fail early")

    runp = sub.add_parser("run", help="time the test passes on an input")
    runp.add_argument("input")
    runp.add_argument("--mlir-opt", default="mlir-opt")
    runp.add_argument("--repetitions", type=int, default=5)
    runp.add_argument("--no-fold", dest="fold", action="store_false")
    runp.add_argument("--variants", nargs="+", default=VARIANTS,
                      choices=VARIANTS)

    args = parser.parse_args()
    if args.command == "generate":
        if args.output == "-":
            generate(args, sys.stdout)
        else:
            with open(args.output, "w") as out:
                generate(args, out)
    else:
        run(args)


if __name__ == "__main__":
    main()
