#!/bin/bash
# TurboQuant quality + speed gate - run BEFORE shipping any perf-affecting change.
# Checks: (1) perplexity within 5% of a LIVE-measured q8_0 baseline,
#         (2) prefill throughput ratio vs q8_0 > 0.95 at 4k context.
#
# Usage: bash scripts/turbo-quality-gate.sh
# Exit 0 = PASS, 1 = FAIL, 77 = SKIP (missing binary / model / dataset)
#
# Env overrides:
#   LLAMA   - directory holding llama-perplexity (default: <repo>/build-perf/bin)
#   MODEL   - path to the reference GGUF (default: probes known locations)
#   WIKI    - path to wiki.test.raw (default: <repo>/wikitext-2-raw/wiki.test.raw)
#   CHUNKS  - perplexity chunks (default 8)

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

LLAMA=${LLAMA:-$REPO_ROOT/build-perf/bin}
WIKI=${WIKI:-$REPO_ROOT/wikitext-2-raw/wiki.test.raw}
CHUNKS=${CHUNKS:-8}

# Configs under test. Symmetric turbo plus the two production-relevant asymmetric pairs
# (precise K + compressed V). The old gate only ever exercised symmetric turbo3.
CONFIGS="turbo3:turbo3 q8_0:turbo3 f16:turbo3"

# --- resolve binary (oneAPI/Windows builds produce .exe) ---
PPL_BIN=""
for cand in "$LLAMA/llama-perplexity" "$LLAMA/llama-perplexity.exe"; do
    if [ -x "$cand" ]; then PPL_BIN=$cand; break; fi
done
if [ -z "$PPL_BIN" ]; then
    echo "SKIP: llama-perplexity not found under $LLAMA"
    echo "      build it with:"
    echo "      cmake --build build-perf --config Release -j 8 --target llama-perplexity"
    exit 77
fi

# --- resolve model: reference is Qwen3-4B Q4_K_M (head_dim 128, GQA 4:1) ---
if [ -z "${MODEL:-}" ]; then
    for cand in \
        "$REPO_ROOT/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
        "/g/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
        "G:/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf"; do
        if [ -f "$cand" ]; then MODEL=$cand; break; fi
    done
fi
if [ -z "${MODEL:-}" ] || [ ! -f "$MODEL" ]; then
    echo "SKIP: reference model not found. Set MODEL=/path/to/Qwen3-4B-Q4_K_M.gguf"
    exit 77
fi

# --- resolve dataset ---
if [ ! -f "$WIKI" ] && [ -f "$SCRIPT_DIR/get-wikitext-2.sh" ]; then
    echo "wikitext-2 missing, downloading..."
    (cd "$REPO_ROOT" && bash "$SCRIPT_DIR/get-wikitext-2.sh") || true
fi
if [ ! -f "$WIKI" ]; then
    echo "SKIP: wikitext-2 not available at $WIKI"
    echo "      fetch it with: bash scripts/get-wikitext-2.sh"
    exit 77
fi

# "Final estimate: PPL = <x>". Empty output means the run crashed or produced nothing.
run_ppl() {
    "$PPL_BIN" -m "$MODEL" -f "$WIKI" -c 512 -ctk "$1" -ctv "$2" \
        -fa on --chunks "$CHUNKS" -ngl 99 2>&1 |
        grep -E "Final estimate" | grep -oE 'PPL = [0-9.]+' | grep -oE '[0-9.]+$'
}

# Prefill throughput in tokens per second, from llama-bench.
# llama-perplexity reports "N seconds per pass" and never prints a tokens-per-second line, so the
# old grep for "prompt eval ... tokens per second" could not match anything and the speed half of
# this gate always reported a crash. llama-bench is the right instrument and prints a t/s column.
BENCH_BIN=""
for cand in "$LLAMA/llama-bench" "$LLAMA/llama-bench.exe"; do
    if [ -x "$cand" ]; then BENCH_BIN=$cand; break; fi
done

run_tps() {
    if [ -z "$BENCH_BIN" ]; then return 0; fi
    "$BENCH_BIN" -m "$MODEL" -ngl 99 -fa 1 -ctk "$1" -ctv "$2" -p 4096 -n 0 -r 2 2>/dev/null |
        awk -F'|' '/pp4096/ { if (match($(NF-1), /[0-9]+\.[0-9]+/)) print substr($(NF-1), RSTART, RLENGTH) }'
}

# awk rather than bc: bc is routinely absent from Git Bash on Windows, and the old script
# would have died on every comparison there.
fcmp() { awk -v a="$1" -v b="$2" "BEGIN{print (a $3 b) ? 1 : 0}"; }
fmul() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f", a*b}'; }
fdiv() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.4f", (b==0 ? 0 : a/b)}'; }

FAIL=0

echo "========================================"
echo "  TurboQuant Quality + Speed Gate"
echo "========================================"
echo "  model : $MODEL"
echo "  binary: $PPL_BIN"
echo "  chunks: $CHUNKS"
echo ""

# --- Test 1: perplexity vs a MEASURED q8_0 baseline ---
# The previous version hardcoded BASELINE_PPL=6.111, which silently decoupled the gate from
# whichever model was actually under test. Measure the baseline in the same session instead.
echo "[1/2] Measuring q8_0 baseline perplexity ($CHUNKS chunks)..."
BASELINE_PPL=$(run_ppl q8_0 q8_0)
if [ -z "$BASELINE_PPL" ]; then
    echo "  FAIL: could not measure q8_0 baseline (crash or timeout)"
    exit 1
fi
MAX_PPL=$(fmul "$BASELINE_PPL" 1.05)
echo "  baseline q8_0 PPL = $BASELINE_PPL (budget: < $MAX_PPL)"

for cfg in $CONFIGS; do
    CTK=${cfg%%:*}; CTV=${cfg##*:}
    PPL=$(run_ppl "$CTK" "$CTV")
    if [ -z "$PPL" ]; then
        echo "  FAIL: -ctk $CTK -ctv $CTV produced no perplexity (crash or timeout)"
        FAIL=1
    elif [ "$(fcmp "$PPL" "$MAX_PPL" '<')" -eq 1 ]; then
        echo "  PASS: -ctk $CTK -ctv $CTV PPL = $PPL (< $MAX_PPL)"
    else
        echo "  FAIL: -ctk $CTK -ctv $CTV PPL = $PPL (>= $MAX_PPL, over the 5% budget)"
        FAIL=1
    fi
done
echo ""

# --- Test 2: prefill throughput at 4k ---
echo "[2/2] Running context scaling check (4K prefill)..."
Q8_TPS=$(run_tps q8_0 q8_0)
if [ -z "$Q8_TPS" ]; then
    echo "  FAIL: could not measure q8_0 speed (crash or timeout)"
    exit 1
fi
echo "  baseline q8_0 = $Q8_TPS tok/s"

for cfg in $CONFIGS; do
    CTK=${cfg%%:*}; CTV=${cfg##*:}
    TPS=$(run_tps "$CTK" "$CTV")
    if [ -z "$TPS" ]; then
        echo "  FAIL: -ctk $CTK -ctv $CTV produced no timing (crash or timeout)"
        FAIL=1
        continue
    fi
    RATIO=$(fdiv "$TPS" "$Q8_TPS")
    if [ "$(fcmp "$RATIO" 0.95 '>')" -eq 1 ]; then
        echo "  PASS: -ctk $CTK -ctv $CTV = ${RATIO}x of q8_0 ($TPS tok/s)"
    else
        echo "  FAIL: -ctk $CTK -ctv $CTV = ${RATIO}x of q8_0 ($TPS tok/s, below 0.95)"
        echo "        context scaling regression"
        FAIL=1
    fi
done
echo ""

echo "========================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
    echo "========================================"
    exit 0
else
    echo "  CHECKS FAILED - DO NOT SHIP"
    echo "========================================"
    exit 1
fi
