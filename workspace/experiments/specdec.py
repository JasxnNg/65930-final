"""Reusable speculative-decoding cost model on top of AccelForge.

Generalizes the proven cost model from milestone3_runs.ipynb so that architecture,
model scale (H = #heads), precision (BITS), depth (n_layers), and extra arch jinja
knobs all become parameters, and so we can swap the "vanilla" linear-chain cost model
for the "eagle" (cheap-draft + tree-verify) one.

Semantics preserved from the original paper:
  * draft step      : draft model, BATCH=1, N_NEW=1, decoding 1 token at ctx.
  * baseline step   : target model autoregressive decode, BATCH=1, N_NEW=1.
  * verify step     : target model, BATCH=1, N_NEW=gamma (parallel verify of gamma).
  * per-token spec  : (gamma draft steps + 1 verify) / expected_tokens_per_round.
  * expected_tokens_per_round(gamma, alpha) = (1 - alpha**(gamma+1)) / (1 - alpha).

All AccelForge map calls are memoized (the expensive part), exactly as milestone3 did.
"""
from __future__ import annotations

import os
import functools
from pathlib import Path

import accelforge as af

af.set_n_parallel_jobs(int(os.environ.get("AF_JOBS", os.cpu_count() or 1)),
                       print_message=True)

DESIGNS = Path("designs")
WORKLOAD = DESIGNS / "transformer_block.yaml"

# ---------------------------------------------------------------------------
# Architectures (jinja-parametrized YAMLs already in designs/)
# ---------------------------------------------------------------------------
ARCHS = {
    "baseline": DESIGNS / "arch.yaml",            # global buffer, KV in shared SRAM
    "kv_buffer": DESIGNS / "arch_kv_buffer.yaml",  # dedicated unified KV buffer
    "hier": DESIGNS / "arch_hier.yaml",            # memory-hierarchy parametrized
    "kv_split": DESIGNS / "arch_kv_split.yaml",    # separate K and V buffers
}

# ---------------------------------------------------------------------------
# Model configs.  E (head dim) = 128 throughout (as in the paper). A model is
# characterized by its width H (# heads); D = E*H, C = 4*D.  n_layers defaults to 1
# (single-block, matching the original paper's per-block methodology); set >1 only
# for explicit depth-aware robustness studies.
# ---------------------------------------------------------------------------
MODELS = {
    # name            H    n_layers (real depth, used only when depth-aware)
    "gpt3_1.3b":  dict(H=16,  n_layers=24),
    "gpt3_6.7b":  dict(H=32,  n_layers=32),   # paper's draft
    "gpt3_13b":   dict(H=40,  n_layers=40),
    "gpt3_30b":   dict(H=56,  n_layers=48),
    "llama_70b":  dict(H=64,  n_layers=80),
    "gpt3_175b":  dict(H=96,  n_layers=96),   # paper's target
}

DRAFT_MODEL = "gpt3_6.7b"
TARGET_MODEL = "gpt3_175b"


# ---------------------------------------------------------------------------
# Low-level: build + map a single workload instance.
# ---------------------------------------------------------------------------
def _keep_all(spec, only_main=False):
    """Force a memory level to keep all tensors (mirrors milestone3 helpers)."""
    for node in spec.arch.nodes:
        if not isinstance(node, af.arch.Memory):
            continue
        if only_main and node.name != "MainMemory":
            continue
        node.tensors.keep = "All"
        return


@functools.lru_cache(maxsize=None)
def _map(arch_name, model_name, tokens, n_new, batch, bits, keep_main, extra):
    """Map one workload instance; returns (energy, latency) as floats.

    `extra` is a tuple of (key, value) jinja overrides for the arch (hashable).
    """
    model = MODELS[model_name]
    jinja = {
        "BATCH_SIZE": batch,
        "N_TOKENS": tokens,
        "N_NEW_TOKENS": n_new,
        "H": model["H"],
        "BITS": bits,
    }
    jinja.update(dict(extra))
    spec = af.Spec.from_yaml(ARCHS[arch_name], WORKLOAD, jinja_parse_data=jinja)
    _keep_all(spec, only_main=keep_main)
    spec.mapper.metrics = af.mapper.Metrics.LATENCY
    res = spec.map_workload_to_arch(print_progress=False)
    return float(res.energy()), float(res.latency())


# ---------------------------------------------------------------------------
# Cost-model steps.
# ---------------------------------------------------------------------------
def draft_step(arch, ctx, *, model=DRAFT_MODEL, bits=8, extra=()):
    """One draft-model token at context length `ctx`."""
    return _map(arch, model, ctx, 1, 1, bits, False, extra)


def baseline_step(arch, ctx, *, model=TARGET_MODEL, bits=8, extra=()):
    """One target-model autoregressive token at context length `ctx`."""
    return _map(arch, model, ctx + 1, 1, 1, bits, False, extra)


def verify_step(arch, ctx, gamma, *, model=TARGET_MODEL, bits=8, extra=()):
    """One target-model parallel verification of `gamma` speculative tokens."""
    return _map(arch, model, ctx + gamma, gamma, 1, bits, True, extra)


def expected_tokens_per_round(gamma, alpha):
    if alpha >= 1.0:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


# ---------------------------------------------------------------------------
# Aggregated per-token costs.
# ---------------------------------------------------------------------------
def baseline_per_token(arch, ctx, gamma, *, target=TARGET_MODEL, bits=8,
                       target_layers=1, extra=()):
    e = l = 0.0
    for i in range(gamma):
        de, dl = baseline_step(arch, ctx + i, model=target, bits=bits, extra=extra)
        e += de
        l += dl
    return (e / gamma) * target_layers, (l / gamma) * target_layers


def spec_round_cost(arch, ctx, gamma, *, draft=DRAFT_MODEL, target=TARGET_MODEL,
                    bits=8, draft_layers=1, target_layers=1, extra=()):
    e = l = 0.0
    for i in range(gamma):
        de, dl = draft_step(arch, ctx + i, model=draft, bits=bits, extra=extra)
        e += de * draft_layers
        l += dl * draft_layers
    ve, vl = verify_step(arch, ctx, gamma, model=target, bits=bits, extra=extra)
    e += ve * target_layers
    l += vl * target_layers
    return e, l


def spec_per_token(arch, ctx, gamma, alpha, *, draft=DRAFT_MODEL,
                   target=TARGET_MODEL, bits=8, draft_layers=1, target_layers=1,
                   extra=()):
    e, l = spec_round_cost(arch, ctx, gamma, draft=draft, target=target, bits=bits,
                           draft_layers=draft_layers, target_layers=target_layers,
                           extra=extra)
    y = expected_tokens_per_round(gamma, alpha)
    return e / y, l / y


# ---------------------------------------------------------------------------
# EAGLE-3 cost model.
#   * Cheap draft: ONE transformer block at *target* hidden dims (reuses target
#     features), instead of a full draft model -> draft_step on target model with
#     a single layer.
#   * Tree drafting + verification: the target verifies a whole candidate tree in
#     one pass (N_NEW = tree size).  expected accepted length uses a tree model.
# ---------------------------------------------------------------------------
def eagle_draft_step(arch, ctx, *, target=TARGET_MODEL, bits=8, extra=()):
    """One EAGLE draft expansion: a single block at target width."""
    # single decode token through one target-width block
    return _map(arch, target, ctx, 1, 1, bits, False, extra)


def expected_tokens_tree(depth, width, alpha):
    """Expected number of accepted tokens for a (depth x width) candidate tree.

    At each depth level we keep `width` candidate continuations; the chain is
    accepted as long as at least one candidate at the next level matches. The
    probability that the deepest accepted position reaches level k is the prob that
    every level up to k has >=1 accepted candidate. With per-token accept prob alpha
    and `width` independent candidates, P(level ok) = 1 - (1-alpha)**width.
    Expected accepted length = sum_{k=1..depth} p**k  (+ the always-correct token).
    """
    if alpha >= 1.0:
        return depth + 1
    p = 1 - (1 - alpha) ** width
    if p >= 1.0:
        return depth + 1
    # sum_{k=1}^{depth} p^k  = p (1-p^depth)/(1-p)
    return 1 + p * (1 - p ** depth) / (1 - p)


def eagle_round_cost(arch, ctx, depth, width, *, target=TARGET_MODEL, bits=8,
                     target_layers=1, extra=()):
    """`depth` cheap draft expansions + 1 tree verify of (depth*width) nodes."""
    tree_nodes = depth * width
    e = l = 0.0
    for i in range(depth):
        de, dl = eagle_draft_step(arch, ctx + i, target=target, bits=bits, extra=extra)
        # one block, not the full target stack
        e += de
        l += dl
    ve, vl = verify_step(arch, ctx, tree_nodes, model=target, bits=bits, extra=extra)
    e += ve * target_layers
    l += vl * target_layers
    return e, l


def eagle_per_token(arch, ctx, depth, width, alpha, *, target=TARGET_MODEL, bits=8,
                    target_layers=1, extra=()):
    e, l = eagle_round_cost(arch, ctx, depth, width, target=target, bits=bits,
                            target_layers=target_layers, extra=extra)
    y = expected_tokens_tree(depth, width, alpha)
    return e / y, l / y


def clear_caches():
    _map.cache_clear()


# ---------------------------------------------------------------------------
# Sweep helpers shared by all experiment drivers.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ALPHAS_DEFAULT = [round(float(a), 3) for a in np.arange(0.2, 0.95 + 1e-9, 0.05)]
CTX_DEFAULT = list(range(512, 4096 + 1, 512))
GAMMA_DEFAULT = list(range(1, 13))


def get_shard(items):
    """Return this task's slice of `items` based on SHARD_ID / SHARD_COUNT env."""
    sid = int(os.environ.get("SHARD_ID", "0"))
    n = int(os.environ.get("SHARD_COUNT", "1"))
    return list(items)[sid::n]


def core_rows(arch, ctx, gamma, *, alphas=ALPHAS_DEFAULT, draft=DRAFT_MODEL,
              target=TARGET_MODEL, bits=8, draft_layers=1, target_layers=1,
              extra=(), meta=None):
    """One (arch, ctx, gamma) point -> rows over alpha (vanilla spec vs baseline).

    The expensive map calls are made once here; the alpha loop is pure arithmetic.
    """
    be, bl = baseline_step(arch, ctx, model=target, bits=bits, extra=extra)
    be *= target_layers
    bl *= target_layers
    re, rl = spec_round_cost(arch, ctx, gamma, draft=draft, target=target, bits=bits,
                             draft_layers=draft_layers, target_layers=target_layers,
                             extra=extra)
    meta = meta or {}
    rows = []
    for a in alphas:
        y = expected_tokens_per_round(gamma, a)
        se, sl = re / y, rl / y
        rows.append({**meta, "arch": arch, "ctx": ctx, "gamma": gamma, "alpha": a,
                     "draft": draft, "target": target, "bits": bits,
                     "base_E": be, "base_L": bl, "spec_E": se, "spec_L": sl,
                     "round_E": re, "round_L": rl, "yield": y,
                     "latency_speedup": bl / sl, "energy_ratio": se / be})
    return rows


def run_sweep(make_rows, tasks, out_csv, checkpoint_every=5):
    """Run `make_rows(task)` over (sharded) `tasks`, checkpointing to CSV.

    When run as part of a Slurm array (SHARD_COUNT>1) each task writes a separate
    `<stem>.shard<id>.csv`; merge_shards() recombines them afterwards.
    """
    sid = int(os.environ.get("SHARD_ID", "0"))
    n = int(os.environ.get("SHARD_COUNT", "1"))
    tasks = get_shard(tasks)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    target = out_csv if n == 1 else out_csv.with_suffix(f".shard{sid}.csv")
    rows = []
    for i, task in enumerate(tasks, 1):
        rows.extend(make_rows(task))
        print(f"[{target.name}] {i}/{len(tasks)} {task}", flush=True)
        if i % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(target, index=False)
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f"[{target.name}] wrote {len(rows)} rows -> {target}", flush=True)
    return rows


def merge_shards(out_csv):
    """Concatenate <stem>.shard*.csv into out_csv and return the DataFrame."""
    out_csv = Path(out_csv)
    parts = sorted(out_csv.parent.glob(out_csv.stem + ".shard*.csv"))
    if not parts:
        return pd.read_csv(out_csv) if out_csv.exists() else None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"merged {len(parts)} shards -> {out_csv} ({len(df)} rows)")
    return df
