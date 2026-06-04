"""EAGLE-3 vs vanilla speculative decoding vs baseline.

EAGLE's benefit is a *single-layer* draft (reusing target features) plus *tree* drafting
verified in one target pass. Capturing the layer-count reduction requires depth-aware
costs, so this experiment uses realistic n_layers for draft and target (unlike the
single-block convention used in the scheduler/KV studies).

Methods (per-token latency/energy on the kv_buffer arch):
  * baseline     : full target autoregressive decode (Lt layers).
  * vanilla spec : Ld-layer draft model, gamma-token linear verify.
  * eagle        : 1-layer target-width draft, (depth x width) tree verify.

Sharded by context.
"""
import specdec as sd

ARCH = "kv_buffer"
CTXS = [512, 1024, 2048, 4096]
ALPHAS = sd.ALPHAS_DEFAULT
GAMMAS = [1, 2, 4, 6, 8, 10, 12]
TREES = [(4, 1), (6, 1), (8, 1), (4, 2), (6, 2), (8, 2), (6, 4), (8, 4)]
OUT = "results/eagle.csv"

DRAFT, TARGET = sd.DRAFT_MODEL, sd.TARGET_MODEL
LD = sd.MODELS[DRAFT]["n_layers"]
LT = sd.MODELS[TARGET]["n_layers"]


def make_rows(ctx):
    rows = []
    # baseline: independent of alpha
    be, bl = sd.baseline_step(ARCH, ctx, model=TARGET)
    be *= LT
    bl *= LT
    base = {"experiment": "eagle", "arch": ARCH, "ctx": ctx,
            "Ld": LD, "Lt": LT}

    # precompute vanilla round costs per gamma and eagle round costs per tree
    van = {g: sd.spec_round_cost(ARCH, ctx, g, draft=DRAFT, target=TARGET,
                                 draft_layers=LD, target_layers=LT) for g in GAMMAS}
    eag = {(d, w): sd.eagle_round_cost(ARCH, ctx, d, w, target=TARGET, target_layers=LT)
           for (d, w) in TREES}

    for a in ALPHAS:
        rows.append({**base, "method": "baseline", "config": "ar", "alpha": a,
                     "L_per_tok": bl, "E_per_tok": be,
                     "latency_speedup": 1.0, "energy_ratio": 1.0})
        for g in GAMMAS:
            re, rl = van[g]
            y = sd.expected_tokens_per_round(g, a)
            sl, se = rl / y, re / y
            rows.append({**base, "method": "vanilla", "config": f"g{g}", "alpha": a,
                         "L_per_tok": sl, "E_per_tok": se,
                         "latency_speedup": bl / sl, "energy_ratio": se / be})
        for (d, w) in TREES:
            re, rl = eag[(d, w)]
            y = sd.expected_tokens_tree(d, w, a)
            sl, se = rl / y, re / y
            rows.append({**base, "method": "eagle", "config": f"d{d}w{w}", "alpha": a,
                         "tree_depth": d, "tree_width": w,
                         "L_per_tok": sl, "E_per_tok": se,
                         "latency_speedup": bl / sl, "energy_ratio": se / be})
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, CTXS, OUT, checkpoint_every=1)
