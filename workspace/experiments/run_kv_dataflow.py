"""Priority 3 - KV-cache organizations and alternative verification dataflows.

Sub-studies (all vanilla-spec per-token rows, on the parametrized arch_hier unless an
arch comparison is the point):
  (a) kv_org      : KV in shared global SRAM (baseline arch) vs unified KV buffer
                    (kv_buffer) vs separate K/V buffers (kv_split).
  (b) kv_size     : dedicated KV buffer capacity sweep (8/32/128/512 MB).
  (c) kv_bw       : KV read bandwidth sweep -> approximates paged vs contiguous layouts
                    (paging lowers effective read BW).
  (d) verify_fanout: spatial parallel-verify fanout (1/2/4/8/16) in the PE array.

Sharded by task (one setting at one context, looping gamma).
"""
import specdec as sd

CTXS = [512, 1024, 2048, 4096]
GAMMAS = [1, 2, 4, 6, 8]
ALPHAS = sd.ALPHAS_DEFAULT
OUT = "results/kv_dataflow.csv"

MB = 1024 * 1024 * 8  # bits per MB
KV_SIZES_MB = [8, 32, 128, 512]
KV_READ_BWS = [512e9, 1024e9, 4096e9, 16384e9]   # low BW ~ paged overhead
VERIFY_FANOUTS = [1, 2, 4, 8, 16]


def _tup(d):
    return tuple(sorted(d.items()))


def build_tasks():
    tasks = []
    for ctx in CTXS:
        # (a) KV organization: compare architectures directly
        for arch in ("baseline", "kv_buffer", "kv_split"):
            tasks.append(dict(arch=arch, ctx=ctx, extra=(),
                              meta={"study": "kv_org", "setting": arch}))
        # (b) KV buffer capacity
        for mb in KV_SIZES_MB:
            tasks.append(dict(arch="hier", ctx=ctx,
                              extra=_tup({"KV_CACHE_BUFFER_SIZE": mb * MB}),
                              meta={"study": "kv_size", "setting": f"{mb}MB"}))
        # (c) KV read bandwidth (paged approximation)
        for bw in KV_READ_BWS:
            tasks.append(dict(arch="hier", ctx=ctx,
                              extra=_tup({"KV_CACHE_READ_BW": bw}),
                              meta={"study": "kv_bw", "setting": f"{bw/1e12:.2f}TBps"}))
        # (d) verification fanout
        for vf in VERIFY_FANOUTS:
            tasks.append(dict(arch="hier", ctx=ctx,
                              extra=_tup({"VERIFY_FANOUT": vf}),
                              meta={"study": "verify_fanout", "setting": f"vf{vf}"}))
    return tasks


def make_rows(task):
    rows = []
    for gamma in GAMMAS:
        rows.extend(sd.core_rows(task["arch"], task["ctx"], gamma, alphas=ALPHAS,
                                 extra=task["extra"], meta=task["meta"]))
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, build_tasks(), OUT, checkpoint_every=2)
