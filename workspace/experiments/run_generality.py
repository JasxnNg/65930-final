"""Priority 2 - generality of the speculative-decoding trends.

Three sub-studies, all emitting the same per-token baseline-vs-spec rows so they can be
compared on common axes:
  (a) precision  : weight/KV/activation bit-width (4 / 8 / 16) on the kv_buffer arch.
  (b) model scale: several draft->target model pairs (width-scaled, E=128 fixed).
  (c) hierarchy  : edge / TPU-v4 / TPU-v8-like SRAM sizes and DRAM/SRAM bandwidths
                   (folds in the "emulate TPU v8" idea) on the parametrized arch_hier.

Sharded by task (a task = one setting at one context, looping gamma) for cache locality.
"""
import specdec as sd

CTXS = [512, 1024, 2048, 4096]
GAMMAS = [1, 2, 4, 6, 8]
ALPHAS = sd.ALPHAS_DEFAULT

OUT = "results/generality.csv"

PRECISIONS = [4, 8, 16]

PAIRS = [
    ("gpt3_1.3b", "gpt3_13b"),
    ("gpt3_6.7b", "gpt3_30b"),
    ("gpt3_6.7b", "llama_70b"),
    ("gpt3_6.7b", "gpt3_175b"),   # paper's pair
    ("gpt3_13b", "gpt3_175b"),
]

HIERARCHIES = {
    "edge": dict(GLOBAL_BUFFER_SIZE=1024*1024*8*8, GLOBAL_READ_BW=512e9,
                 GLOBAL_WRITE_BW=256e9, DRAM_BW=100e9,
                 KV_CACHE_BUFFER_SIZE=1024*1024*8*8, KV_CACHE_READ_BW=512e9,
                 KV_CACHE_WRITE_BW=256e9),
    "tpu_v4": dict(),  # arch_hier defaults
    "tpu_v8": dict(GLOBAL_BUFFER_SIZE=1024*1024*512*8, GLOBAL_READ_BW=8192e9,
                   GLOBAL_WRITE_BW=4096e9, DRAM_BW=3000e9,
                   KV_CACHE_BUFFER_SIZE=1024*1024*512*8, KV_CACHE_READ_BW=16384e9,
                   KV_CACHE_WRITE_BW=4096e9),
}


def _tup(d):
    return tuple(sorted(d.items()))


def build_tasks():
    tasks = []
    for ctx in CTXS:
        for bits in PRECISIONS:
            tasks.append(dict(study="precision", arch="kv_buffer", ctx=ctx, bits=bits,
                              draft=sd.DRAFT_MODEL, target=sd.TARGET_MODEL, extra=(),
                              meta={"study": "precision", "setting": f"{bits}bit"}))
        for draft, target in PAIRS:
            tasks.append(dict(study="scale", arch="kv_buffer", ctx=ctx, bits=8,
                              draft=draft, target=target, extra=(),
                              meta={"study": "scale", "setting": f"{draft}->{target}"}))
        for name, params in HIERARCHIES.items():
            tasks.append(dict(study="hierarchy", arch="hier", ctx=ctx, bits=8,
                              draft=sd.DRAFT_MODEL, target=sd.TARGET_MODEL,
                              extra=_tup(params),
                              meta={"study": "hierarchy", "setting": name}))
    return tasks


def make_rows(task):
    rows = []
    for gamma in GAMMAS:
        rows.extend(sd.core_rows(
            task["arch"], task["ctx"], gamma, alphas=ALPHAS,
            draft=task["draft"], target=task["target"], bits=task["bits"],
            extra=task["extra"], meta=task["meta"]))
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, build_tasks(), OUT, checkpoint_every=2)
