"""P3 extension - long-context KV-cache organization study.

At <=4k context, decode is weight-bound and KV organization is second-order. KV traffic
grows linearly with context while weight traffic is fixed, so this sweep pushes context
to 32k to find where the dedicated/split KV buffer starts to matter (latency & energy).
Sharded by context.
"""
import specdec as sd

CTXS = [4096, 8192, 16384, 32768]
GAMMA = 4
ALPHAS = sd.ALPHAS_DEFAULT
OUT = "results/kv_longctx.csv"

MB = 1024 * 1024 * 8
KV_READ_BWS = [512e9, 4096e9, 16384e9]


def _tup(d):
    return tuple(sorted(d.items()))


def make_rows(ctx):
    rows = []
    for arch in ("baseline", "kv_buffer", "kv_split"):
        rows.extend(sd.core_rows(arch, ctx, GAMMA, alphas=ALPHAS, extra=(),
                                 meta={"study": "kv_org", "setting": arch}))
    for bw in KV_READ_BWS:
        rows.extend(sd.core_rows("hier", ctx, GAMMA, alphas=ALPHAS,
                                 extra=_tup({"KV_CACHE_READ_BW": bw}),
                                 meta={"study": "kv_bw", "setting": f"{bw/1e12:.2f}TBps"}))
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, CTXS, OUT, checkpoint_every=1)
