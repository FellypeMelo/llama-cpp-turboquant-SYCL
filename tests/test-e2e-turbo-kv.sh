#!/bin/bash
# =============================================================================
# test-e2e-turbo-kv.sh  -  END-TO-END turbo KV-cache COHERENCE gate (SYCL/Arc)
# =============================================================================
# Proves the RECOMMENDED, SYCL-safe turbo KV-cache configs generate COHERENT
# text in REAL end-to-end llama-cli generation (NOT kernel numeric parity).
#
# This complements the other two turbo gates:
#   - tests/test-sycl-turbo.cpp    numeric golden (quant cosine ~1.0, FA parity)
#   - scripts/turbo-quality-gate.sh perplexity + speed vs q8_0 (needs wikitext)
# This one answers the owner's question directly: "with turbo KV really ON,
# does generation come out coherent?"  -> asserts coherence on the GPU.
#
# GATED CONFIGS (coherent + SYCL-safe, from the 2026-07-09 B580 sweep):
#   SYMMETRIC (uniform turbo K and V):
#     -ctk turbo3 -ctv turbo3   (uniform 3-bit; best balance, 5.1x KV saving)
#     -ctk turbo4 -ctv turbo4   (uniform 4-bit; safest turbo, 3.8x)
#     -ctk turbo2 -ctv turbo2   (2-bit + AUTO boundary-symmetric-q8_0 mode 8; 5.6x)
#   ASYMMETRIC (precise K + turbo V; "V is free, K is everything"):
#     -ctk q8_0 -ctv turbo{2,3,4}   (recommended prod default; K protected at 8-bit)
#     -ctk f16  -ctv turbo{2,3,4}   (max-precision K; Q stays un-rotated when K is f16)
# NOT gated (characterised in docs/BENCHMARKS.md):
#   turbo2 with TURBO_LAYER_ADAPTIVE=0  -> degenerate repetition (needs boundary)
#   TURBO_LAYER_ADAPTIVE=5/6/7          -> SYCL FA abort for turbo-K widths other than turbo4
#   mixed turbo-K / q8_0-V              -> runs but flagged unreliable in code
#   mixed turbo-K + turbo-V, K != turbo4 -> SYCL FA abort (no curated vec row)
#
# Mixed-width turbo4-K + cheaper-turbo-V IS gated below. Perplexity says turbo on V costs ~0.4%
# while turbo on K costs ~30%, and turbo4 is the quality knee on K, so turbo4-K x turbo3-V reaches
# ~4.3x KV saving at turbo4-K quality - a point symmetric turbo cannot hit. See ADR-0007.
#
# REQUIREMENTS: Intel Arc GPU + oneAPI runtime on PATH (source setvars.bat first)
#   + the pure-attention reference model (Qwen3-4B). Absent model -> SKIP (77).
#
# USAGE:
#   TURBO_E2E_MODEL=/path/Qwen3-4B-Q4_K_M.gguf bash tests/test-e2e-turbo-kv.sh
# ENV:
#   LLAMA_CLI        llama-cli path (auto-detected in build-perf/build/build-sync)
#   TURBO_E2E_MODEL  gguf model (default: G:/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf)
#   TURBO_E2E_NGEN   tokens to generate (default 150)
# EXIT: 0 = PASS, 1 = FAIL (incoherent / abort), 77 = SKIP (no model -> ctest SKIP)
# =============================================================================
set -u

# turbo requires flash-attention + these runtime knobs on Intel GPUs
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"
export SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# --- locate llama-cli ---------------------------------------------------------
find_cli() {
    if [ -n "${LLAMA_CLI:-}" ] && [ -x "${LLAMA_CLI}" ]; then echo "$LLAMA_CLI"; return; fi
    for d in build-perf build build-sync; do
        for e in llama-cli.exe llama-cli; do
            if [ -x "$ROOT/$d/bin/$e" ]; then echo "$ROOT/$d/bin/$e"; return; fi
        done
    done
}
CLI="$(find_cli)"

MODEL="${TURBO_E2E_MODEL:-G:/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
NGEN="${TURBO_E2E_NGEN:-150}"
PROMPT="Explain why the sky is blue in 3 sentences."

if [ -z "${CLI:-}" ]; then
    echo "SKIP: llama-cli not found (build it first: build-perf/bin/llama-cli)"; exit 77
fi
if [ ! -f "$MODEL" ]; then
    echo "SKIP: reference model not found: $MODEL"
    echo "      set TURBO_E2E_MODEL to a pure-attention gguf (e.g. Qwen3-4B) to run this gate."
    exit 77
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "========================================"
echo "  turbo KV-cache END-TO-END coherence gate"
echo "  cli   : $CLI"
echo "  model : $MODEL"
echo "  gen   : $NGEN tokens, greedy (--temp 0), -fa on, -ngl 99, -c 4096"
echo "========================================"

FAIL=0

run_one() {
    local name="$1" k="$2" v="$3"
    local out="$TMP/out_$name.txt" err="$TMP/err_$name.txt" gen="$TMP/gen_$name.txt"
    echo ""
    echo "[$name]  -ctk $k -ctv $v"

    "$CLI" -m "$MODEL" -ngl 99 -fa on -c 4096 --temp 0 -n "$NGEN" \
        -st --no-display-prompt -ctk "$k" -ctv "$v" -p "$PROMPT" \
        > "$out" 2> "$err"
    local rc=$?

    # (1) no abort / hard error
    if [ $rc -ne 0 ]; then
        echo "  FAIL: llama-cli exited $rc (abort/crash)"; FAIL=1; return
    fi
    if grep -qaE 'GGML_ASSERT|Not match KV type|SYCL error|failed to allocate|out of memory|-nan|[^a-z]nan[^a-z]' "$err"; then
        echo "  FAIL: error signature in stderr:"; grep -aE 'GGML_ASSERT|Not match KV type|SYCL error|failed to allocate|out of memory|nan' "$err" | head -3 | sed 's/^/    /'
        FAIL=1; return
    fi

    # isolate generated prose: drop UI banner + keep only ASCII printable
    grep -aiv -e 'Loading model' -e 'available commands' -e 'build *:' \
        -e 'model *:' -e 'ftype' -e 'modalities' -e 'Exiting' \
        -e 'regenerate' -e 'clear the chat' -e 'add a text' -e ' glob ' \
        -e '^\s*/' -e 'Prompt:' "$out" \
        | LC_ALL=C tr -cd '\11\12\15\40-\176' > "$gen"

    # (2) on-topic coherence: scatter/wavelength/rayleigh appear ONLY in real
    #     generation (never in the prompt) -> proves coherent physics output
    if ! grep -qiE 'scatter|wavelength|rayleigh' "$gen"; then
        echo "  FAIL: no on-topic keyword (scatter/wavelength/rayleigh) in output"
        echo "    got: $(tr '\n' ' ' < "$gen" | cut -c1-200)"
        FAIL=1; return
    fi

    # (2b) no '?'-corruption: a broken KV dispatch emits runs of '?' replacement glyphs
    local nq; nq=$(LC_ALL=C tr -cd '?' < "$gen" | wc -c)
    if [ "${nq:-0}" -ge 10 ]; then
        echo "  FAIL: output corruption ('?' appears ${nq}x)"; FAIL=1; return
    fi

    # (3) not degenerate: max 5-gram frequency and unique-word ratio
    read -r maxg nwords ratio < <(
        LC_ALL=C tr 'A-Z' 'a-z' < "$gen" | LC_ALL=C tr -cs 'a-z0-9' ' ' | awk '
        { for (i=1;i<=NF;i++) w[++n]=$i }
        END {
            max=0
            for (i=1; i+4<=n; i++) { g=w[i]" "w[i+1]" "w[i+2]" "w[i+3]" "w[i+4]; c[g]++; if (c[g]>max) max=c[g] }
            u=0; for (i=1;i<=n;i++) if (!(w[i] in s)) { s[w[i]]=1; u++ }
            printf "%d %d %.3f\n", max, n, (n>0? u/n : 1)
        }')
    echo "    words=$nwords  max-5gram-repeat=$maxg  unique-ratio=$ratio"
    if [ "${maxg:-0}" -ge 4 ]; then
        echo "  FAIL: degenerate repetition (a 5-word phrase repeats ${maxg}x)"; FAIL=1; return
    fi
    awk "BEGIN{ exit !($nwords>=40 && $ratio<0.42) }" && { echo "  FAIL: low lexical diversity (ratio $ratio < 0.42)"; FAIL=1; return; }

    echo "  PASS: coherent ($(tr '\n' ' ' < "$gen" | sed 's/  */ /g' | cut -c1-160)...)"
}

run_one "turbo3" turbo3 turbo3
run_one "turbo4" turbo4 turbo4
run_one "turbo2" turbo2 turbo2

# ASYMMETRIC precise-K + turbo-V ("V is free, K is everything"): K stays high-precision
# (q8_0 recommended prod default, or f16 max-precision) while V is turbo-compressed. All six
# {q8_0,f16} x {turbo2,turbo3,turbo4} combos dispatch on the SYCL vec kernel (head_dim=128).
run_one "q8_0-turbo3" q8_0 turbo3
run_one "q8_0-turbo2" q8_0 turbo2
run_one "q8_0-turbo4" q8_0 turbo4
run_one "f16-turbo3"  f16  turbo3
run_one "f16-turbo2"  f16  turbo2
run_one "f16-turbo4"  f16  turbo4
# Mixed-width turbo: turbo4 K (quality knee) + a cheaper turbo V. Both sides turbo, so the graph
# rotates Q forward AND inverse-rotates the output - the only gated configs where both happen with
# different block widths on each side.
run_one "turbo4-turbo3" turbo4 turbo3
run_one "turbo4-turbo2" turbo4 turbo2

echo ""
echo "========================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL turbo KV configs generated COHERENT text - PASS"
    echo "========================================"
    exit 0
else
    echo "  turbo KV coherence FAILED - DO NOT SHIP"
    echo "========================================"
    exit 1
fi
