"""EAGLE-3 vs vanilla speculative decoding vs baseline (calibrated acceptance length).

EAGLE's benefit is a single-layer draft (reusing target features) + tree drafting
verified in one target pass. Capturing the layer-count reduction needs depth-aware costs
(draft 32 layers, target 96).

Crucially, the per-token yield uses a CALIBRATED average acceptance length tau rather than
the optimistic independence tree model. The EAGLE-3 paper (Li et al. 2025, Table 1)
reports tau ~ 5.8-6.6 (mean 6.6 for Vicuna-13B, ~6.2 for LLaMA-3.1-8B) with batch-1
speedups of 4.1-5.5x; we sweep tau in {4,5,6,7} and mark tau=6.

Sharded by context.
"""
import specdec as sd

ARCH = "kv_buffer"
CTXS = [512, 1024, 2048, 4096]
ALPHAS = sd.ALPHAS_DEFAULT
GAMMAS = [1, 2, 4, 6, 8, 10, 12]
# Full-binary EAGLE trees: explicit per-layer frontiers (1, 2, 4, 8, ...).
# The model deduplicates explicit node IDs within each layer if a caller supplies them.
TREE_LAYERS = [sd.eagle_power2_layer_counts(d) for d in (5, 6, 7)]
TREES = {sd.eagle_tree_config_name(layers): layers for layers in TREE_LAYERS}
TAUS = [4, 5, 6, 7]            # paper-reported acceptance-length range
OUT = "results/eagle.csv"

DRAFT, TARGET = sd.DRAFT_MODEL, sd.TARGET_MODEL
LD = sd.MODELS[DRAFT]["n_layers"]
LT = sd.MODELS[TARGET]["n_layers"]


def make_rows(ctx):
    rows = []
    print(f"[eagle] ctx={ctx} baseline", flush=True)
    be, bl = sd.baseline_step(ARCH, ctx, model=TARGET)
    be, bl = be * LT, bl * LT
    base = {"experiment": "eagle", "arch": ARCH, "ctx": ctx, "Ld": LD, "Lt": LT}

    # baseline
    rows.append({**base, "method": "baseline", "config": "ar", "tau": 1.0,
                 "L_per_tok": bl, "E_per_tok": be, "latency_speedup": 1.0,
                 "energy_ratio": 1.0})

    # vanilla speculative: yield (tau) derived from (gamma, alpha)
    for g in GAMMAS:
        print(f"[eagle] ctx={ctx} vanilla g={g}", flush=True)
        re, rl = sd.spec_round_cost(ARCH, ctx, g, draft=DRAFT, target=TARGET,
                                    draft_layers=LD, target_layers=LT)
        for a in ALPHAS:
            tau = sd.expected_tokens_per_round(g, a)
            sl, se = rl / tau, re / tau
            rows.append({**base, "method": "vanilla", "config": f"g{g}", "alpha": a,
                         "tau": tau, "L_per_tok": sl, "E_per_tok": se,
                         "latency_speedup": bl / sl, "energy_ratio": se / be})

    # EAGLE-3: single-layer draft + explicit tree verify, calibrated tau
    for name, layer_counts in TREES.items():
        print(f"[eagle] ctx={ctx} {name}", flush=True)
        depth = len(layer_counts)
        nodes = sum(layer_counts)
        re, rl = sd.eagle_round_cost(ARCH, ctx, layer_counts, target=TARGET,
                                     target_layers=LT)
        for tau in TAUS:
            sl, se = rl / tau, re / tau
            rows.append({**base, "method": "eagle", "config": name, "tau": float(tau),
                         "tree_depth": depth, "tree_nodes": nodes,
                         "tree_layer_counts": " ".join(map(str, layer_counts)),
                         "L_per_tok": sl, "E_per_tok": se,
                         "latency_speedup": bl / sl, "energy_ratio": se / be})
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, CTXS, OUT, checkpoint_every=1)
