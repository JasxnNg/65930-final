"""Priority 1 - data for the theoretically optimal scheduler.

Sweeps the best architecture (kv_buffer) over (context, gamma) and emits per-token
latency/energy for baseline vs speculative across a fine alpha grid. The scheduler
analysis (oracle gamma*, static gamma=4, closed-form optimum, regret) is done in
analysis/figures from this CSV.

Sharded by context (one Slurm array task per context) for map-cache locality.
"""
import specdec as sd

ARCH = "kv_buffer"
CTXS = sd.CTX_DEFAULT          # 512..4096 step 512
GAMMAS = sd.GAMMA_DEFAULT      # 1..12
ALPHAS = sd.ALPHAS_DEFAULT     # 0.2..0.95 step 0.05

OUT = "results/scheduler.csv"


def make_rows(ctx):
    rows = []
    for gamma in GAMMAS:
        rows.extend(sd.core_rows(ARCH, ctx, gamma, alphas=ALPHAS,
                                 meta={"experiment": "scheduler"}))
    return rows


if __name__ == "__main__":
    sd.run_sweep(make_rows, CTXS, OUT, checkpoint_every=1)
