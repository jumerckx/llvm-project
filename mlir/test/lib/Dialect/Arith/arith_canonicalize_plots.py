#!/usr/bin/env python3
# ===- arith_canonicalize_plots.py - Plot the PDL benchmarks ---------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PDLL = os.path.join(HERE, "ArithCanonicalization.pdll")
GENERATOR = os.path.join(HERE, "arith_canonicalize_benchmark.py")

STAGES = [
    ("pdl-to-match", "convert-pdl-to-match", "ConvertPDLToMatchPass"),
    ("combine-matchers", "match-combine-matchers", "MatchCombineMatchersPass"),
    ("match-to-pdl-interp", "convert-match-to-pdl-interp",
     "ConvertMatchToPDLInterpPass"),
]
LEGACY = ("legacy pdl-to-pdl-interp", "convert-pdl-to-pdl-interp",
          "ConvertPDLToPDLInterpPass")

VARIANTS = [
    ("drr", "DRR (C++)"),
    ("pdll", "PDL bytecode"),
    ("match-cpp", "match-to-cpp, per root"),
    ("match-cpp-all", "match-to-cpp, one pattern"),
    ("match-cpp-split", "match-to-cpp, per PDL pattern"),
]
PHASES = ["driver", "match", "rewrite"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LEGACY_COLOR = "#898781"


def pdl_module(mlir_pdll, mlir_opt, includes):
    cmd = [mlir_pdll, PDLL, "-x=mlir"] + [f"-I{d}" for d in includes]
    pdl = subprocess.run(cmd, capture_output=True, text=True,
                         check=True).stdout
    return subprocess.run([mlir_opt, "-"], input=pdl, capture_output=True,
                          text=True, check=True).stdout


def split_patterns(module):
    body = module[module.index("{") + 1:module.rindex("}")]
    starts = [m.start() for m in re.finditer(r"^  pdl\.pattern ", body, re.M)]
    return [body[a:b].rstrip() + "\n"
            for a, b in zip(starts, starts[1:] + [len(body)])]


def build_module(patterns, count):
    out = []
    for i in range(count):
        copy, pattern = divmod(i, len(patterns))
        text = patterns[pattern]
        if copy:
            text = re.sub(r"pdl\.pattern @(\w+)",
                          lambda m: f"pdl.pattern @{m.group(1)}_{copy}", text,
                          count=1)
        out.append(text)
    return "module {\n" + "".join(out) + "}\n"


def pass_times(mlir_opt, path, pipeline):
    cmd = [mlir_opt, path, f"-pass-pipeline=builtin.module({pipeline})",
           "-mlir-disable-threading", "-verify-each=false", "-mlir-timing",
           "-mlir-timing-display=list", "-o", os.devnull]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    times = {}
    for line in proc.stderr.splitlines():
        m = re.match(r"\s*([0-9.]+) \(\s*[0-9.]+%\)\s+(\S+)$", line)
        if m:
            times[m.group(2)] = float(m.group(1))
    return times


def collect_conversion(args):
    patterns = split_patterns(
        pdl_module(args.mlir_pdll, args.mlir_opt, args.include))
    counts = sorted(set(args.counts + [len(patterns)]))
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for count in counts:
            path = os.path.join(tmp, f"patterns-{count}.mlir")
            with open(path, "w") as f:
                f.write(build_module(patterns, count))
            runs = [(args.legacy_mlir_opt, LEGACY[1], [LEGACY]),
                    (args.mlir_opt, ",".join(s[1] for s in STAGES), STAGES)]
            for mlir_opt, pipeline, stages in runs:
                samples = [pass_times(mlir_opt, path, pipeline)
                           for _ in range(args.repetitions)]
                for label, _, timer_name in stages:
                    values = [s[timer_name] for s in samples]
                    rows.append({"patterns": count,
                                 "source_patterns": len(patterns),
                                 "stage": label,
                                 "median_s": statistics.median(values),
                                 "min_s": min(values), "max_s": max(values)})
            print(f"converted {count} patterns", file=sys.stderr)
    write_csv(args.output, rows)


def collect_runtime(args):
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rate in args.match_rates:
            path = os.path.join(tmp, f"input-{rate}.mlir")
            subprocess.run(
                [sys.executable, GENERATOR, "generate", "--functions",
                 str(args.functions), "--templates", str(args.templates),
                 "--match-rate", str(rate), "-o", path],
                check=True, capture_output=True)
            for variant, _ in VARIANTS:
                samples = []
                for _ in range(args.repetitions):
                    cmd = [args.mlir_opt, path,
                           f"-test-arith-canonicalize-{variant}=phase-timing",
                           "-mlir-disable-threading", "-o", os.devnull]
                    proc = subprocess.run(cmd, capture_output=True, text=True,
                                          check=True)
                    times = {}
                    for line in proc.stderr.splitlines():
                        m = re.match(r"\s*(\w+)\s+([0-9.]+) s$", line)
                        if m:
                            times[m.group(1)] = float(m.group(2))
                        m = re.match(r"\s*\d+ pattern applications, (\d+) "
                                     r"rewrites", line)
                        if m:
                            times["rewrites"] = int(m.group(1))
                    samples.append(times)
                for phase in PHASES + ["total"]:
                    values = [s[phase] for s in samples]
                    rows.append({"match_rate": rate, "variant": variant,
                                 "phase": phase,
                                 "median_s": statistics.median(values),
                                 "min_s": min(values), "max_s": max(values),
                                 "rewrites": samples[0]["rewrites"]})
            print(f"ran match rate {rate}", file=sys.stderr)
    write_csv(args.output, rows)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "xtick.labelcolor": INK_2,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def fmt_ms(seconds):
    ms = seconds * 1e3
    return f"{ms:.0f} ms" if ms >= 100 else f"{ms:.1f} ms" if ms >= 1 else \
        f"{ms:.2f} ms"


def stacked_hbar(ax, y, parts, colors, height):
    left = 0.0
    for value, color in zip(parts, colors):
        ax.barh(y, value, left=left, height=height, color=color,
                edgecolor=SURFACE, linewidth=1.5)
        left += value
    return left


def plot_conversion(plt, rows, out, note=""):
    by = {}
    for r in rows:
        by[(int(r["patterns"]), r["stage"])] = float(r["median_s"])
    counts = sorted({c for c, _ in by})
    full = int(rows[0]["source_patterns"])

    fig, (bar_ax, line_ax) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})

    new_parts = [by[(full, s[0])] for s in STAGES]
    legacy = by[(full, LEGACY[0])]
    total = stacked_hbar(bar_ax, 1, new_parts, SERIES, 0.5)
    bar_ax.barh(0, legacy, height=0.5, color=LEGACY_COLOR, edgecolor=SURFACE)
    for y, value in [(1, total), (0, legacy)]:
        bar_ax.text(value, y, "  " + fmt_ms(value), va="center", ha="left",
                    color=INK, fontsize=9)
    bar_ax.set_yticks([1, 0], ["match pipeline", LEGACY[0]])
    bar_ax.tick_params(axis="y", length=0)
    bar_ax.set_xlim(0, max(total, legacy) * 1.25)
    bar_ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
    bar_ax.set_xlabel("wall time (ms)")
    bar_ax.set_title(f"Lowering {full} arith PDLL patterns to pdl_interp")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES]
    bar_ax.legend(handles, [s[0] for s in STAGES], loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1,
                  handleheight=1)

    xs = [c for c in counts if all((c, s[0]) in by for s in STAGES)]
    stacks = [[by[(c, s[0])] * 1e3 for c in xs] for s in STAGES]
    line_ax.stackplot(xs, stacks, colors=SERIES, edgecolor=SURFACE,
                      linewidth=1.0, alpha=0.9)
    legacy_ys = [by[(c, LEGACY[0])] * 1e3 for c in xs]
    line_ax.plot(xs, legacy_ys, color=INK_2, linewidth=2, marker="o",
                 markersize=4, markeredgecolor=SURFACE)
    line_ax.text(xs[-1], legacy_ys[-1], "  legacy", va="center", color=INK_2)
    new_total = [sum(col) for col in zip(*stacks)]
    line_ax.text(xs[-1], new_total[-1], "  match pipeline", va="center",
                 color=INK)
    line_ax.set_xlim(0, xs[-1] * 1.22)
    line_ax.set_ylim(0, max(max(legacy_ys), max(new_total)) * 1.1)
    line_ax.grid(axis="y")
    line_ax.set_xlabel("number of patterns")
    line_ax.set_ylabel("wall time (ms)")
    line_ax.set_title("Scaling with the number of patterns")
    fig.text(0.01, -0.04,
             f"Median of the -mlir-timing pass times. Pattern sets beyond {full} "
             "repeat the arith patterns under new names; smaller sets are "
             "prefixes.\nThe legacy lowering emits IR that fails verification "
             "(a use not dominated by its definition) for every set of two or "
             "more of these patterns." + note, color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


def plot_runtime(plt, rows, out, phases=PHASES, title=None, note=""):
    by = {}
    rewrites = {}
    for r in rows:
        by[(float(r["match_rate"]), r["variant"], r["phase"])] = \
            float(r["median_s"])
        rewrites[float(r["match_rate"])] = int(r["rewrites"])
    rates = sorted(rewrites)
    xmax = max(sum(by[(rate, v, p)] for p in phases) for rate in rates
               for v, _ in VARIANTS)
    colors = [SERIES[PHASES.index(p)] for p in phases]

    fig, axes = plt.subplots(1, len(rates), figsize=(4.2 * len(rates), 3.2),
                             sharey=True, squeeze=False)
    for ax, rate in zip(axes[0], rates):
        for i, (variant, _) in enumerate(VARIANTS):
            y = len(VARIANTS) - 1 - i
            parts = [by[(rate, variant, p)] for p in phases]
            total = stacked_hbar(ax, y, parts, colors, 0.6)
            ax.text(total, y, "  " + fmt_ms(total), va="center", ha="left",
                    color=INK, fontsize=8)
        ax.set_yticks(range(len(VARIANTS)),
                      [label for _, label in reversed(VARIANTS)])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, xmax * 1.3)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
        ax.set_xlabel("wall time (ms)")
        ax.set_title(f"match rate {rate:g}  ({rewrites[rate]} rewrites)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, phases, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), handlelength=1, handleheight=1)
    fig.suptitle(title or "Greedy arith canonicalization: time per phase",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.text(0.01, -0.12,
             "Median of the phase-timing runs of the arith canonicalization "
             "test passes on generated inputs, single-threaded." + note,
             color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    os.makedirs(args.output_dir, exist_ok=True)
    note = f" {args.note}" if args.note else ""
    if args.conversion:
        plot_conversion(plt, read_csv(args.conversion),
                        os.path.join(args.output_dir, "conversion.png"),
                        note)
    if args.runtime:
        rows = read_csv(args.runtime)
        plot_runtime(plt, rows, os.path.join(args.output_dir, "runtime.png"),
                     note=note)
        plot_runtime(plt, rows,
                     os.path.join(args.output_dir, "runtime-match-rewrite.png"),
                     phases=["match", "rewrite"],
                     title="Greedy arith canonicalization: time spent in "
                           "patterns", note=note)


def main():
    parser = argparse.ArgumentParser(
        description="Collect and plot the PDL conversion and runtime "
                    "benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("conversion", help="time the PDL to pdl_interp "
                                              "lowering")
    conv.add_argument("--mlir-opt", required=True)
    conv.add_argument("--legacy-mlir-opt", required=True)
    conv.add_argument("--mlir-pdll", required=True)
    conv.add_argument("-I", "--include", action="append", default=[])
    conv.add_argument("--counts", type=int, nargs="+",
                      default=[8, 16, 32, 134, 268, 536, 1072])
    conv.add_argument("--repetitions", type=int, default=10)
    conv.add_argument("-o", "--output", default="conversion.csv")

    rt = sub.add_parser("runtime", help="time the canonicalization passes")
    rt.add_argument("--mlir-opt", required=True)
    rt.add_argument("--match-rates", type=float, nargs="+",
                    default=[0.01, 0.1, 0.5])
    rt.add_argument("--functions", type=int, default=200)
    rt.add_argument("--templates", type=int, default=200)
    rt.add_argument("--repetitions", type=int, default=5)
    rt.add_argument("-o", "--output", default="runtime.csv")

    pl = sub.add_parser("plot", help="plot collected results")
    pl.add_argument("--conversion")
    pl.add_argument("--runtime")
    pl.add_argument("--output-dir", default=".")
    pl.add_argument("--note", default="",
                    help="text appended to the footnote of every figure")

    args = parser.parse_args()
    {"conversion": collect_conversion, "runtime": collect_runtime,
     "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
