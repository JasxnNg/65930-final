"""Architecture exploration for decode THROUGHPUT (speculative, gamma=4, alpha=0.8).

Probe finding: on-chip SRAM layout (KV-tier size, buffer split) does NOT move throughput
in the standard decode mapping (KV is streamed from DRAM once per step; the tier only
saves energy). Throughput at scale is governed by DRAM bandwidth up to a compute ceiling.
We therefore (A) compare named architecture configurations on throughput AND energy, and
(B) sweep DRAM bandwidth x batch to map the throughput roofline and how the speculative
break-even batch shifts with bandwidth.

Sharded by task.
"""
import specdec as sd

GAMMA, ALPHA = 4, 0.8
MB = 1024 * 1024 * 8
OUT = "results/arch_throughput.csv"


def _tup(d):
    return tuple(sorted(d.items()))


# (A) named configs evaluated at a few (batch, ctx) operating points
CONFIGS = {
    "unified_global":   ("baseline", ()),
    "dedicated_KV_tier": ("kv_buffer", ()),
    "split_K_V":        ("kv_split", ()),
    "big_global_256MB": ("hier", _tup({"GLOBAL_BUFFER_SIZE": 256 * MB})),
    "fast_KV_16TBps":   ("hier", _tup({"KV_CACHE_READ_BW": 16384e9})),
    "hi_DRAM_3TBps":    ("hier", _tup({"DRAM_BW": 3000e9})),
    "tpu_v8":           ("hier", _tup({"DRAM_BW": 3000e9, "GLOBAL_BUFFER_SIZE": 512 * MB,
                                       "GLOBAL_READ_BW": 8192e9, "KV_CACHE_READ_BW": 16384e9})),
}
OP_POINTS = [(16, 2048), (64, 4096), (256, 4096)]

# (B) DRAM bandwidth x batch roofline
DRAM_BWS = [100e9, 614e9, 1500e9, 3000e9, 8000e9]
BS = [1, 4, 16, 64, 256, 1024]
CTX_ROOFLINE = 4096

TASKS = ([("named", name, B, ctx) for name in CONFIGS for (B, ctx) in OP_POINTS]
         + [("roofline", f"{bw/1e9:.0f}GBps", B, CTX_ROOFLINE) for bw in DRAM_BWS for B in BS])
_BW_OF = {f"{bw/1e9:.0f}GBps": bw for bw in DRAM_BWS}


def make_rows(task):
    kind, name, B, ctx = task
    if kind == "named":
        arch, extra = CONFIGS[name]
    else:
        arch, extra = "hier", _tup({"DRAM_BW": _BW_OF[name]})
    base = sd.baseline_batch(arch, ctx, B, extra=extra)
    spec = sd.spec_batch(arch, ctx, GAMMA, ALPHA, B, extra=extra)
    # energy per token: spec round energy / accepted tokens (per sequence)
    se, _ = sd.spec_round_cost(arch, ctx, GAMMA, extra=extra)
    y = sd.expected_tokens_per_round(GAMMA, ALPHA)
    return [{"kind": kind, "config": name, "arch": arch, "batch": B, "ctx": ctx,
             "base_tput": base["throughput"], "spec_tput": spec["throughput"],
             "base_tpot": base["tpot"], "spec_tpot": spec["tpot"],
             "tput_gain": base["tpot"] / spec["tpot"],
             "spec_energy_per_tok": se / y}]


if __name__ == "__main__":
    sd.run_sweep(make_rows, TASKS, OUT, checkpoint_every=5)
