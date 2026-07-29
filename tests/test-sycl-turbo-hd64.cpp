// Turbo KV on a model whose head_dim is not a multiple of 128.
//
// Why this test exists
// --------------------
// The turbo formats have a block size of 128 (QK_TURBO), so a turbo cache row cannot be 64 wide -
// llama_kv_cache pads every turbo side up to the next multiple of 128. Only the turbo side was
// padded, so on a head_dim-64 model the recommended production config `-ctk q8_0 -ctv turbo3` left
// K at 64 while V became 128, and ggml_sycl_get_best_fattn_kernel refuses a pair whose head_dims
// disagree. The consequence is not an abort: ggml schedules the whole attention op on the CPU
// backend. Correct output, roughly 10x slower, and nothing in the log.
//
// That is why this test asserts on two different things:
//   1. the fallback warning does NOT fire, i.e. the dispatcher accepted the pair and attention
//      stayed on the GPU. This is the property the fix is about, and no numeric check can see it -
//      the CPU produces correct values, so logits match either way.
//   2. the logits still match an f16 reference. The fix pads a non-turbo side of the cache, so this
//      is the guard against that padding corrupting the values it was supposed to leave alone.
//
// The model
// ---------
// No head_dim-64 gguf was available, and every architecture in test-llama-archs yields 128 or more.
// The model is synthesized by that test, which builds gguf files with arbitrary hparams. It has
// random weights and no vocab, which is fine - this test feeds raw token ids and compares logits,
// it never tokenizes. Regenerate with:
//
//   cmake -B build-gen -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
//         -DGGML_SYCL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
//   cmake --build build-gen --target test-llama-archs
//   LLAMA_ARCHS_N_EMBD=128 LLAMA_ARCHS_N_HEAD=2 build-gen/bin/test-llama-archs -a llama -s 42 -o <dir>
//
// A separate build tree is needed because test-llama-archs uses llama internals that are not
// exported from the shared library on Windows. Running the result only needs the public API, which
// is why this test builds in the normal configuration.
//
// Point TURBO_HD64_MODEL at <dir>/llama-dense.gguf. Absent -> exit 77 (SKIP).

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string g_log;

static void capture_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        g_log += text;
    }
    // Without this the callback swallows everything else, including the ggml scheduler's own
    // per-node backend assignment (GGML_SCHED_DEBUG=2). That output is the only check on this test
    // that does not go through the warning this test is partly written to verify - use it to
    // confirm independently that FLASH_ATTN_EXT really landed on SYCL and not on the CPU.
    if (getenv("TURBO_HD64_VERBOSE")) {
        fputs(text, stderr);
    }
}

// normalized mean squared error: mse(a, b) / mse(a, 0). Scale-free, so it does not care that the
// synthetic model's logits are arbitrary in magnitude.
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        num += d*d;
        den += (double) a[i] * (double) a[i];
    }
    return den > 0.0 ? num/den : INFINITY;
}

struct run_result {
    std::vector<float> logits;
    std::string        warnings;
    bool               can_shift = false;
    bool               ok = false;
};

static run_result run(llama_model * model, ggml_type type_k, ggml_type type_v, int n_vocab) {
    run_result res;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 64;
    cp.type_k          = type_k;
    cp.type_v          = type_v;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    g_log.clear();
    llama_context * ctx = llama_init_from_model(model, cp);
    res.warnings = g_log;
    if (!ctx) {
        return res;
    }

    // Raw ids, no tokenizer: the synthetic model has none. Deterministic and inside the vocab.
    std::vector<llama_token> tokens;
    for (int i = 0; i < 32; i++) {
        tokens.push_back((llama_token) ((i*7 + 3) % n_vocab));
    }

    // Context shift derives its view strides from hparams.n_embd_k_gqa, so it is wrong on a padded
    // cache and llama_kv_cache::get_can_shift() must refuse. The old guard only tested for a turbo
    // K type, which stopped covering the padded case once a non-turbo K started being padded too.
    res.can_shift = llama_memory_can_shift(llama_get_memory(ctx));

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_free(ctx);
        return res;
    }

    const float * logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    if (!logits) {
        llama_free(ctx);
        return res;
    }
    res.logits.assign(logits, logits + n_vocab);
    res.ok = true;

    llama_free(ctx);
    return res;
}

int main() {
    const char * path = getenv("TURBO_HD64_MODEL");
    if (!path || !*path) {
        printf("SKIP: TURBO_HD64_MODEL is not set.\n");
        printf("      This test needs a head_dim-64 gguf; see the header of this file for how to\n");
        printf("      synthesize one with test-llama-archs.\n");
        return 77;
    }
    if (FILE * f = fopen(path, "rb")) {
        fclose(f);
    } else {
        printf("SKIP: TURBO_HD64_MODEL=%s cannot be opened.\n", path);
        return 77;
    }

    llama_log_set(capture_log, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        printf("FAIL: could not load %s\n", path);
        llama_backend_free();
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("model: %s (n_vocab %d)\n", path, n_vocab);

    int failures = 0;

    const run_result ref = run(model, GGML_TYPE_F16, GGML_TYPE_F16, n_vocab);
    if (!ref.ok) {
        printf("FAIL: f16 reference run did not produce logits\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    printf("f16 x f16 reference: %zu logits, shift %s\n",
           ref.logits.size(), ref.can_shift ? "allowed" : "refused");

    // Control for the shift assertions below. f16 x f16 is never padded, so context shift must still
    // be ALLOWED here. Without this, a get_can_shift() guard that is too broad - refusing on every
    // cache rather than only padded ones - would silently disable context shift for every user and
    // every config, and the per-config "shift refused" checks would all still pass.
    if (!ref.can_shift) {
        printf("FAIL: context shift refused on an unpadded f16 cache; the get_can_shift() guard is\n");
        printf("      too broad and has disabled context shift for configurations that never pad\n");
        failures++;
    }

    struct cfg {
        const char * name;
        ggml_type    k;
        ggml_type    v;
        // Whether the SYCL turbo dispatch is expected to accept this pair on a head_dim-64 model.
        // Symmetric turbo pads both sides and has always worked; the asymmetric pairs are what the
        // padding fix is for.
        bool         expect_gpu;
    };

    const cfg cfgs[] = {
        { "turbo3 x turbo3", GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, true },
        { "q8_0 x turbo3",   GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, true },
        { "f16 x turbo3",    GGML_TYPE_F16,      GGML_TYPE_TURBO3_0, true },
        { "q8_0 x turbo2",   GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0, true },
        { "q8_0 x turbo4",   GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0, true },
    };

    for (const cfg & c : cfgs) {
        const run_result r = run(model, c.k, c.v, n_vocab);
        if (!r.ok) {
            printf("  FAIL %-16s: no logits produced\n", c.name);
            failures++;
            continue;
        }

        // The dispatcher never announces itself, so the cache-construction warning is the only
        // signal available from outside that attention was pushed onto the CPU backend.
        const bool fell_back =
            r.warnings.find("attention will run on the CPU") != std::string::npos ||
            r.warnings.find("leaves K at")                   != std::string::npos;

        const double err = nmse(ref.logits, r.logits);

        // Loose on purpose: the weights are random, so this bound is a corruption check, not a
        // quality measurement. Real quality lives in scripts/turbo-quality-gate.sh.
        const bool numeric_ok = err < 5e-2;

        const bool dispatch_ok = (c.expect_gpu != fell_back);

        // Every config here pads the cache (head_dim 64 -> 128), so context shift must be refused.
        // Allowed, build_graph_shift would stride the padded rows as if they were narrow and corrupt
        // K silently - correct-looking output, wrong cache.
        const bool shift_ok = !r.can_shift;

        printf("  %-4s %-16s: nmse %.3e, %s, shift %s\n",
               (numeric_ok && dispatch_ok && shift_ok) ? "PASS" : "FAIL",
               c.name, err,
               fell_back ? "FELL BACK TO CPU" : "dispatched on GPU",
               r.can_shift ? "ALLOWED" : "refused");

        if (!numeric_ok) {
            printf("        numeric: nmse %.3e exceeds 5e-2\n", err);
        }
        if (!dispatch_ok) {
            printf("        dispatch: expected %s, the kv-cache warning says otherwise\n",
                   c.expect_gpu ? "GPU" : "CPU fallback");
            printf("        warnings: %s\n", r.warnings.c_str());
        }
        if (!shift_ok) {
            printf("        shift: allowed on a padded cache; get_can_shift() must refuse or\n");
            printf("               build_graph_shift corrupts K with unpadded strides\n");
        }
        if (!numeric_ok || !dispatch_ok || !shift_ok) {
            failures++;
        }
    }

    llama_model_free(model);
    llama_backend_free();

    printf("%s: %d failure(s)\n", failures == 0 ? "PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
