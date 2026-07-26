#!/usr/bin/env python3
"""plot_bench.py — matplotlib chart of Throughput vs Concurrency.

Reads the aggregated CSV (one row per (mode, concurrency)) and writes a
PNG line chart with one line per mode (continuous, static-B=2, etc).

Usage:
    scripts/plot_bench.py <comparison.csv> --output <path.png>
"""
import argparse
import csv
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="aggregated comparison CSV")
    ap.add_argument("--output", "-o", default="throughput_vs_concurrency.png",
                    help="output PNG path")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        sys.exit(0)

    # Group: {mode: [(N, throughput_tps)]}
    series = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            tag = row["tag"]
            parts = {}
            for piece in tag.split(";"):
                if "=" in piece:
                    k, v = piece.split("=", 1)
                    parts[k] = v
            n = int(parts.get("N", "0"))
            tps = float(row["aggregate_tps"])
            mode = tag.split(";")[0]
            series[mode].append((n, tps))

    if not series:
        print("no rows in CSV", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(8, 5))
    style_map = {
        "continuous": ("-o",  "continuous batching"),
        "static":     ("--s", "static batching"),
    }
    for mode, pts in sorted(series.items()):
        pts.sort()
        style, label = style_map.get(mode, ("-x", mode))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, style, label=label, linewidth=2, markersize=8)

    plt.xlabel("Concurrency (number of concurrent requests)")
    plt.ylabel("Aggregate throughput (output tokens / sec)")
    plt.title("mini-infer: Continuous vs Static Batching")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()