#!/usr/bin/env python3
"""bench_compare.py — render the comparison CSV as a markdown table.

Reads the aggregated CSV (one row per (mode, concurrency)) and prints
markdown rows with throughput + speedup_vs_static.

Usage:
    scripts/bench_compare.py <comparison.csv>
"""
import csv
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("usage: bench_compare.py <comparison.csv>", file=sys.stderr)
        sys.exit(2)
    rows = []
    with open(sys.argv[1]) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print("(no rows)")
        return

    # Group static throughput by N for lookup.
    def parse_tag(tag):
        out = {}
        for part in tag.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k] = v
        return out

    static_by_n = {}
    for r in rows:
        if r["tag"].startswith("static;"):
            parts = parse_tag(r["tag"])
            n = int(parts.get("N", "0"))
            static_by_n[n] = float(r["aggregate_tps"])

    for r in rows:
        agg = float(r["aggregate_tps"])
        parts = parse_tag(r["tag"])
        n = int(parts.get("N", "0"))
        sp = ""
        if r["tag"].startswith("continuous"):
            base = static_by_n.get(n, 0.0)
            if base > 0.0:
                sp = f"{agg / base:.2f}x"
            else:
                sp = "—"
        else:
            sp = "1.00x (baseline)"
        mode = r["tag"].split(";")[0]
        print(f"| {mode} "
              f"| {r['tag']} "
              f"| {n} "
              f"| {int(float(r['wall_ms']))} "
              f"| {r['gen_tokens']} "
              f"| {agg:.1f} "
              f"| {sp} |")


if __name__ == "__main__":
    main()