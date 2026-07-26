#!/usr/bin/env bash
# bench.sh — one-click benchmark runner (Week 6).
#
# Runs bench_static + bench_continuous across a configurable matrix of
# (batch_size, num_prompts) points, then aggregates the CSV rows into a
# single comparison report and plots a Throughput-vs-Concurrency chart.
#
# Usage:
#   scripts/bench.sh [--model DIR] [--quick] [--sharegpt PATH]
#                    [--out-dir DIR] [--max-new-tokens N]
#
# Environment overrides:
#   MODEL_DIR   : default model directory (auto-detected)
#   QUICK       : 1 → use the small matrix below; 0 → full sweep
#   SHAREGPT    : path to a ShareGPT-style JSON file
#
# Outputs (default out-dir = benchmarks/results/):
#   bench_<tag>.csv            : one row per (mode, concurrency)
#   bench_<tag>.md             : per-mode markdown summary
#   comparison_<tag>.csv       : aggregated comparison (one row per mode)
#   comparison_<tag>.md        : side-by-side markdown report
#   throughput_vs_concurrency.png : matplotlib chart
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"

# Defaults
MODEL_DIR="${MODEL_DIR:-/data1/kdy/LLMs/Qwen2.5-Coder-1.5B-Instruct}"
QUICK="${QUICK:-1}"
SHAREGPT="${SHAREGPT:-$HERE/benchmarks/datasets/sharegpt_sample.json}"
OUT_DIR="${OUT_DIR:-$HERE/benchmarks/results}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-384}"

# Parse args (override env)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL_DIR="$2"; shift 2;;
        --quick)          QUICK=1; shift;;
        --full)           QUICK=0; shift;;
        --sharegpt)       SHAREGPT="$2"; shift 2;;
        --out-dir)        OUT_DIR="$2"; shift 2;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
        --max-seq-len)    MAX_SEQ_LEN="$2"; shift 2;;
        *)                echo "unknown arg: $1" >&2; exit 2;;
    esac
done

# Env setup (cuda + conda python with torch/transformers/matplotlib).
export PATH=/data1/kdy/anaconda3/bin:/data1/kdy/anaconda3/envs/vllm/bin:/data1/tyh/miniconda3/bin:/usr/local/cuda-12.1/bin:$PATH
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9

mkdir -p "$OUT_DIR"
TAG="$(date +%Y%m%d_%H%M%S)"

# Benchmark matrix:
#   - QUICK=1 (default):  4 points (B in {2,4,8,16} × N=B; plus continuous at N=8)
#   - QUICK=0:            full sweep  N in {8,16,32,64} × B in {2,4,8}
if [[ "$QUICK" == "1" ]]; then
    POINTS=( "2 8" "4 16" "8 32" "16 64" )
else
    POINTS=( "2 8" "4 16" "8 32" "16 64" "4 32" "8 64" "2 32" "4 64" )
fi

CSV_HEADER="tag,num_requests,num_completed,wall_ms,prompt_tokens,gen_tokens,ttft_avg_ms,ttft_p50_ms,ttft_p99_ms,tpot_avg_ms,tpot_p50_ms,tpot_p99_ms,aggregate_tps,per_req_tps,peak_blocks,total_blocks"

echo "[bench.sh] model  = $MODEL_DIR"
echo "[bench.sh] quick  = $QUICK"
echo "[bench.sh] sharegpt = $SHAREGPT"
echo "[bench.sh] out_dir = $OUT_DIR"
echo "[bench.sh] tag    = $TAG"
echo ""

# Write the consolidated CSV header.
COMBINED="$OUT_DIR/comparison_$TAG.csv"
echo "$CSV_HEADER" > "$COMBINED"

run_static() {
    local B="$1"; local N="$2"
    local PREFIX="$OUT_DIR/static_b${B}_n${N}_$TAG"
    local CSV="$PREFIX.csv"
    echo ">>> bench_static  B=$B N=$N  -> $CSV"
    if "$HERE/build/benchmarks/bench_static" \
            --model "$MODEL_DIR" \
            --dataset "$SHAREGPT" \
            --num-prompts "$N" \
            --batch-size  "$B" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --max-seq-len   "$MAX_SEQ_LEN" \
            --out-prefix "$PREFIX" \
            2>"$PREFIX.stderr"; then
        # Append row (skip header).
        tail -n +2 "$CSV" >> "$COMBINED"
    else
        echo "    FAILED (see $PREFIX.stderr)" >&2
    fi
}

run_continuous() {
    local N="$1"
    local PREFIX="$OUT_DIR/cont_n${N}_$TAG"
    local CSV="$PREFIX.csv"
    echo ">>> bench_continuous  N=$N  -> $CSV"
    # Bucket + max-prefill-batch chosen so logits tensor [B, bucket, vocab]
    # stays under ~6 GB: B * bucket * 152064 * 2 < 6 * 2^30.
    #   64 * 256 * 152064 * 2  ~  4.7 GB   (OK)
    #   64 * 512 * 152064 * 2  ~  9.5 GB   (too big)
    # Largest bucket must be >= max_prompt_len = max_seq_len - max_new_tokens.
    local MAX_PROMPT_LEN=$((MAX_SEQ_LEN - MAX_NEW_TOKENS))
    if [[ "$MAX_PROMPT_LEN" -le 128 ]]; then
        local BUCKETS="64,128"
    elif [[ "$MAX_PROMPT_LEN" -le 256 ]]; then
        local BUCKETS="128,256"
    elif [[ "$MAX_PROMPT_LEN" -le 512 ]]; then
        local BUCKETS="256,512"
    elif [[ "$MAX_PROMPT_LEN" -le 1024 ]]; then
        local BUCKETS="512,1024"
    else
        local BUCKETS="512,1024,2048"
    fi
    if "$HERE/build/benchmarks/bench_continuous" \
            --model "$MODEL_DIR" \
            --dataset "$SHAREGPT" \
            --num-prompts "$N" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --max-seq-len   "$MAX_SEQ_LEN" \
            --max-num-blocks 4096 \
            --max-prefill-batch 32 \
            --bucket "$BUCKETS" \
            --out-prefix "$PREFIX" \
            2>"$PREFIX.stderr"; then
        tail -n +2 "$CSV" >> "$COMBINED"
    else
        echo "    FAILED (see $PREFIX.stderr)" >&2
    fi
}

# Static baseline runs.
for pt in "${POINTS[@]}"; do
    B=$(echo "$pt" | cut -d' ' -f1)
    N=$(echo "$pt" | cut -d' ' -f2)
    run_static "$B" "$N"
done

# Continuous runs (one per unique N in the matrix).
NS=()
for pt in "${POINTS[@]}"; do
    N=$(echo "$pt" | cut -d' ' -f2)
    if [[ ! " ${NS[*]} " =~ " $N " ]]; then
        NS+=("$N")
    fi
done
for N in "${NS[@]}"; do
    run_continuous "$N"
done

# Build the markdown comparison.
MD="$OUT_DIR/comparison_$TAG.md"
{
    echo "# mini-infer benchmark report ($TAG)"
    echo ""
    echo "model: $MODEL_DIR"
    echo "max_new_tokens: $MAX_NEW_TOKENS"
    echo "max_seq_len:    $MAX_SEQ_LEN"
    echo ""
    echo "## Throughput comparison"
    echo ""
    echo "| mode | tag | N | wall_ms | gen_tokens | aggregate_tps | speedup_vs_static |"
    echo "|------|-----|---|---------|------------|---------------|-------------------|"
    python3 "$HERE/scripts/bench_compare.py" "$COMBINED"
} > "$MD"
cat "$MD"

# Plot.
echo ""
echo "[bench.sh] plotting throughput vs concurrency ..."
python3 "$HERE/scripts/plot_bench.py" "$COMBINED" \
        --output "$OUT_DIR/throughput_vs_concurrency_$TAG.png"

echo ""
echo "[bench.sh] done. Results in: $OUT_DIR"
echo "  CSV   : $COMBINED"
echo "  MD    : $MD"
echo "  PNG   : $OUT_DIR/throughput_vs_concurrency_$TAG.png"