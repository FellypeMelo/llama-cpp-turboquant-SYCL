#!/usr/bin/env python3
"""Mechanically enforce the invariants that turbo FA dispatch relies on comments for today.

Three separate places have to agree about which KV type pairs are dispatchable, and today the
only thing keeping them in sync is a comment asking the reader to keep them in sync:

  1. the curated FATTN_VEC_CASES_* table in ggml/src/ggml-sycl/fattn.cpp  (what decode can run)
  2. the mixed_width_dispatchable allowlist in the prefill shim, same file (what prefill accepts)
  3. the `mixed_ok` predicate in tests/test-sycl-turbo.cpp                (which tolerance the
     golden test applies, i.e. which device path it believes ran)

A drift between 1 and 2 is the dangerous one: prefill succeeds through the f16 TILE and the
GGML_ABORT lands on the first decode token, after the whole prompt was processed. That exact bug
was shipped and caught by review once already. A drift between 2 and 3 makes the golden test
assert against the wrong numeric bound.

Exit 0 = consistent, 1 = drift found.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FATTN = REPO / "ggml" / "src" / "ggml-sycl" / "fattn.cpp"
TEST = REPO / "tests" / "test-sycl-turbo.cpp"

TURBO = {"GGML_TYPE_TURBO2_0", "GGML_TYPE_TURBO3_0", "GGML_TYPE_TURBO4_0"}


def fail(msg):
    print("FAIL: " + msg)
    return False


def curated_rows(src):
    """(type_K, type_V) pairs from the curated dispatch table, ignoring the ALL_QUANTS branch.

    The fork keeps GGML_SYCL_FA_ALL_QUANTS off (ADR-0004), so only the '#else' arm is live. Take
    the text after the last '#else' that precedes the closing '#endif // GGML_SYCL_FA_ALL_QUANTS'.
    """
    end = src.find("#endif // GGML_SYCL_FA_ALL_QUANTS")
    if end < 0:
        raise SystemExit("FAIL: could not locate the GGML_SYCL_FA_ALL_QUANTS #endif in fattn.cpp")
    start = src.rfind("#else", 0, end)
    region = src[start:end]
    pairs = set()
    for m in re.finditer(r"FATTN_VEC_CASES(?:_TURBO_D|_ALL_D)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", region):
        pairs.add((m.group(1), m.group(2)))
    return pairs


def shim_allowlist(src):
    """(type_K, type_V) pairs the prefill shim admits despite differing turbo widths."""
    m = re.search(r"mixed_width_dispatchable\s*=\s*\[\]\s*\([^)]*\)\s*\{(.*?)\};", src, re.S)
    if not m:
        raise SystemExit("FAIL: mixed_width_dispatchable lambda not found in fattn.cpp")
    body = m.group(1)
    ks = re.findall(r"k\s*==\s*(GGML_TYPE_\w+)", body)
    vs = re.findall(r"v\s*==\s*(GGML_TYPE_\w+)", body)
    if not ks or not vs:
        raise SystemExit("FAIL: could not parse the k/v comparisons out of mixed_width_dispatchable")
    return {(k, v) for k in ks for v in vs}


def test_mixed_ok(src):
    """The same allowlist as the golden test believes it to be."""
    m = re.search(r"mixed_ok\s*=\s*(.*?);", src, re.S)
    if not m:
        raise SystemExit("FAIL: mixed_ok predicate not found in tests/test-sycl-turbo.cpp")
    body = m.group(1)
    ks = re.findall(r"type_K\s*==\s*(GGML_TYPE_\w+)", body)
    vs = re.findall(r"type_V\s*==\s*(GGML_TYPE_\w+)", body)
    if not ks or not vs:
        raise SystemExit("FAIL: could not parse type_K/type_V out of mixed_ok")
    return {(k, v) for k in ks for v in vs}


def main():
    fattn_src = FATTN.read_text(encoding="utf-8", errors="replace")
    test_src = TEST.read_text(encoding="utf-8", errors="replace")

    curated = curated_rows(fattn_src)
    shim = shim_allowlist(fattn_src)
    test = test_mixed_ok(test_src)

    ok = True

    # 1. Everything the shim lets through at prefill must have a decode row, or the abort just
    #    moves later. This is the invariant that actually protects users.
    orphans = sorted(p for p in shim if p not in curated)
    if orphans:
        ok = fail(
            "the prefill shim admits pairs with no curated vec row, so prefill would succeed and "
            "decode would abort:\n  " + "\n  ".join(f"{k} x {v}" for k, v in orphans)
        )

    # 2. The golden test must model the same allowlist, or it asserts against the wrong bound.
    if shim != test:
        only_shim = sorted(shim - test)
        only_test = sorted(test - shim)
        detail = []
        if only_shim:
            detail.append("in fattn.cpp but not in the test: " + ", ".join(f"{k} x {v}" for k, v in only_shim))
        if only_test:
            detail.append("in the test but not in fattn.cpp: " + ", ".join(f"{k} x {v}" for k, v in only_test))
        ok = fail("mixed-width allowlist drifted between fattn.cpp and test-sycl-turbo.cpp:\n  " +
                  "\n  ".join(detail))

    # 3. Every pair on the allowlist really is mixed-width turbo. A same-type or non-turbo entry
    #    means the allowlist is doing something its name does not say.
    bogus = sorted(p for p in shim if not (p[0] in TURBO and p[1] in TURBO and p[0] != p[1]))
    if bogus:
        ok = fail("mixed_width_dispatchable lists pairs that are not mixed-width turbo:\n  " +
                  "\n  ".join(f"{k} x {v}" for k, v in bogus))

    # 4. ADR-0004 guard: TURBO4_0 must never gain an extern template declaration, or every
    #    implicitly-instantiated TURBO4_0-K row loses its definition and the link breaks.
    vec_hpp = (REPO / "ggml" / "src" / "ggml-sycl" / "fattn-vec.hpp").read_text(
        encoding="utf-8", errors="replace")
    if re.search(r"EXTERN_DECL_FATTN_VEC_CASES\s*\([^)]*GGML_TYPE_TURBO4_0", vec_hpp):
        ok = fail(
            "EXTERN_DECL_FATTN_VEC_CASES was given GGML_TYPE_TURBO4_0 as type_K. That suppresses "
            "the implicit instantiation every TURBO4_0-K curated row depends on and will produce "
            "LNK2019 (see ADR-0004 and ADR-0007)."
        )

    if ok:
        print(f"OK: {len(curated)} curated rows, {len(shim)} mixed-width allowlist entries, "
              "test predicate in lockstep, no TURBO4_0 extern declaration.")
        for k, v in sorted(shim):
            print(f"  allowlisted: {k} x {v}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
