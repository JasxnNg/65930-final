"""Batch-size study: TPOT / throughput, the speculative break-even batch, and how it
depends on lookahead, draft/target ratio, and the accelerator memory system.

Throughput-gain == TPOT-speedup (both are 1/per-step-latency ratios), so a single sweep
answers both "minimize TPOT" and "maximize tokens/s". Decode is weight-bound at small
batch (speculation is ~free) and compute-bound at large batch (speculation wastes work),
so the spec speedup decays with B and crosses 1.0 at a break-even batch B*. We sweep:
  - main : gamma x alpha at full batch range (find B*(gamma,alpha), throughput knee)
  - ratio: draft->target pairs (does a cheaper draft push B* higher?)
  - bw   : edge / TPU-v4 / TPU-v8 DRAM bandwidth (hardware lever on B*)

Sharded by (config, batch) task.
"""
import specdec as sd

CTX = 2048
BS_MAIN = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
BS_OTHER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
OUT = "results/batch.csv"


def _tup(d):
    return tuple(sorted(d.items()))


CONFIGS = [dict(study="batch_main", setting="6.7b->175b", arch="kv_buffer",
                draft="gpt3_6.7b", target="gpt3_175b", extra=(),
                gammas=[1, 2, 4, 8], alphas=[0.6, 0.7, 0.8, 0.9], Bs=BS_MAIN)]
for d, t in [("gpt3_6.7b", "gpt3_30b"), ("gpt3_1.3b", "gpt3_13b"), ("gpt3_6.7b", "gpt3_175b")]:
    CONFIGS.append(dict(study="batch_ratio", setting=f"{d}->{t}", arch="kv_buffer",
                        draft=d, target=t, extra=(), gammas=[4], alphas=[0.8], Bs=BS_OTHER))
for nm, bw in [("edge", 100e9), ("tpu_v4", 614e9), ("tpu_v8", 3000e9)]:
    CONFIGS.append(dict(study="batch_bw", setting=nm, arch="hier",
                        draft="gpt3_6.7b", target="gpt3_175b", extra=_tup({"DRAM_BW": bw}),
                        gammas=[4], alphas=[0.8], Bs=BS_OTHER))

TASKS = [(ci, B) for ci, c in enumerate(CONFIGS) for B in c["Bs"]]


def make_rows(task):
    ci, B = task
    c = CONFIGS[ci]
    meta = {"study": c["study"], "setting": c["setting"], "arch": c["arch"],
            "draft": c["draft"], "target": c["target"], "ctx": CTX, "batch": B}
    b = sd.baseline_batch(c["arch"], CTX, B, target=c["target"], extra=c["extra"])
    rows = [{**meta, "method": "baseline", "gamma": 0, "alpha": 0.0,
             "tpot": b["tpot"], "throughput": b["throughput"],
             "speedup": 1.0}]
    for g in c["gammas"]:
        for a in c["alphas"]:
            s = sd.spec_batch(c["arch"], CTX, g, a, B, draft=c["draft"],
                              target=c["target"], extra=c["extra"])
            rows.append({**meta, "method": "spec", "gamma": g, "alpha": a,
                         "tpot": s["tpot"], "throughput": s["throughput"],
                         "speedup": b["tpot"] / s["tpot"]})
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, TASKS, OUT, checkpoint_every=4)
