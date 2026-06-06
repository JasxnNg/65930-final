"""Generate publication figures + key numeric validations from results/*.csv.

Runs inside the AccelForge container (which has numpy/pandas/matplotlib). Each section
is guarded by file existence so it can be run as experiments complete.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "experiments")
import specdec as sd  # noqa: E402  (for merge_shards)

RES = Path("results")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 9})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


# ===========================================================================
# Priority 1 - optimal scheduler
# ===========================================================================
def scheduler():
    sd.merge_shards(RES / "scheduler.csv")
    df = pd.read_csv(RES / "scheduler.csv")
    ctxs = sorted(df.ctx.unique())
    alphas = sorted(df.alpha.unique())

    # oracle gamma* and static gamma=4 per (ctx, alpha), by latency
    def best(metric):
        idx = df.groupby(["ctx", "alpha"])[metric].idxmin()
        return df.loc[idx].set_index(["ctx", "alpha"])

    oracle = best("spec_L")
    g4 = df[df.gamma == 4].set_index(["ctx", "alpha"])

    def yield_(g, a):
        return (g + 1) if a >= 1 else (1 - a ** (g + 1)) / (1 - a)

    # closed-form optimum: round_L(gamma) ~= a_fit + b_fit*gamma (per ctx), then
    # minimize (a_fit + b_fit*gamma)/yield(gamma, alpha) over continuous gamma.
    def continuous_gstar(ctx, alpha, gmax=12):
        s = df[(df.ctx == ctx)].drop_duplicates("gamma").sort_values("gamma")
        b_fit, a_fit = np.polyfit(s.gamma, s.round_L, 1)
        gg = np.linspace(1, gmax, 400)
        cost = (a_fit + b_fit * gg) / np.array([yield_(g, alpha) for g in gg])
        return gg[np.argmin(cost)]

    # ---- Fig A (money plot): per-token latency vs gamma at ctx=2048 ----
    c0 = 2048 if 2048 in ctxs else ctxs[len(ctxs) // 2]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for a in [0.4, 0.6, 0.7, 0.8, 0.9]:
        s = df[(df.ctx == c0) & (df.alpha == a)].sort_values("gamma")
        if s.empty:
            continue
        (ln,) = ax.plot(s.gamma, s.spec_L * 1e3, "o-", label=f"α={a}")
        gs = s.loc[s.spec_L.idxmin()].gamma
        ax.plot(gs, s.spec_L.min() * 1e3, "*", color=ln.get_color(), ms=14)
    ax.axvline(4, color="gray", ls="--", lw=1, label="static γ=4")
    ax.set_xlabel("lookahead γ"); ax.set_ylabel("per-token latency (ms)")
    ax.set_title(f"Latency vs γ is U-shaped; γ* rises with α (ctx={c0})")
    ax.legend(fontsize=8)
    save(fig, "p1_latency_vs_gamma")

    # ---- Fig B: oracle gamma*(alpha) per ctx + closed-form overlay ----
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in ctxs:
        gs = [oracle.loc[(c, a)].gamma for a in alphas]
        ax.plot(alphas, gs, "o-", label=f"ctx={c}", alpha=0.8)
    cf = [continuous_gstar(c0, a) for a in alphas]
    ax.plot(alphas, cf, "k--", lw=2, label=f"closed-form γ* (ctx={c0})")
    ax.axhline(4, color="gray", ls=":", label="static γ=4")
    ax.set_xlabel("acceptance α"); ax.set_ylabel("latency-optimal γ*")
    ax.set_title("Optimal lookahead is acceptance-dependent")
    ax.legend(fontsize=8)
    save(fig, "p1_gamma_star_vs_alpha")

    # ---- Fig C: regret of static γ=4 vs oracle (%) ----
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for c in ctxs:
        reg = [100 * (g4.loc[(c, a)].spec_L / oracle.loc[(c, a)].spec_L - 1) for a in alphas]
        ax.plot(alphas, reg, "o-", label=f"ctx={c}")
    ax.axvspan(0.6, 0.8, color="green", alpha=0.08, label="typical α band")
    ax.set_xlabel("acceptance α"); ax.set_ylabel("latency regret of γ=4 vs oracle (%)")
    ax.set_title("Static γ=4 is near-optimal in the typical α band")
    ax.legend(fontsize=8)
    save(fig, "p1_gamma4_regret")

    # ---- numeric validation ----
    G = np.array([[oracle.loc[(c, a)].gamma for a in alphas] for c in ctxs])
    print("\n[P1] oracle γ* distribution:", pd.Series(G.ravel()).value_counts().to_dict())
    band = [a for a in alphas if 0.6 <= a <= 0.8]
    reg_band = [100 * (g4.loc[(c, a)].spec_L / oracle.loc[(c, a)].spec_L - 1)
                for c in ctxs for a in band]
    reg_all = [100 * (g4.loc[(c, a)].spec_L / oracle.loc[(c, a)].spec_L - 1)
               for c in ctxs for a in alphas]
    print(f"[P1] regret of static γ=4 vs oracle: mean(typical α 0.6-0.8)={np.mean(reg_band):.2f}% "
          f"| worst(all α)={max(reg_all):.2f}%")


def _nearest_alpha(df, a):
    return df.alpha.iloc[(df.alpha - a).abs().argmin()]


def _load(name):
    sd.merge_shards(RES / f"{name}.csv")
    return pd.read_csv(RES / f"{name}.csv")


# ===========================================================================
# Priority 2 - generality (precision, model scale, memory hierarchy)
# ===========================================================================
def generality():
    df = _load("generality")
    a0 = _nearest_alpha(df, 0.8)
    g0 = 4
    studies = {"precision": "Precision (bit-width)",
               "scale": "Model scale (draft→target)",
               "hierarchy": "Memory hierarchy"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (study, title) in zip(axes, studies.items()):
        sub = df[(df.study == study) & (df.gamma == g0) & (df.alpha == a0)]
        for setting, s in sub.groupby("setting"):
            s = s.sort_values("ctx")
            ax.plot(s.ctx, s.latency_speedup, "o-", label=setting)
        ax.set_title(title); ax.set_xlabel("context length")
        ax.set_ylabel("latency speedup vs baseline")
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.legend(fontsize=8)
    fig.suptitle(f"Generality of speculative speedup (γ={g0}, α={a0:.2f})")
    save(fig, "p2_generality_speedup")

    # energy ratio companion
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (study, title) in zip(axes, studies.items()):
        sub = df[(df.study == study) & (df.gamma == g0) & (df.alpha == a0)]
        for setting, s in sub.groupby("setting"):
            s = s.sort_values("ctx")
            ax.plot(s.ctx, s.energy_ratio, "o-", label=setting)
        ax.set_title(title); ax.set_xlabel("context length")
        ax.set_ylabel("spec/baseline energy ratio")
        ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.legend(fontsize=8)
    fig.suptitle(f"Speculative energy ratio (γ={g0}, α={a0:.2f})")
    save(fig, "p2_generality_energy")

    # validation: gamma* per setting at alpha0
    print("\n[P2] latency-optimal γ* per setting (α={:.2f}, ctx=2048):".format(a0))
    sub = df[(df.alpha == a0) & (df.ctx == 2048)]
    for setting, s in sub.groupby("setting"):
        gstar = int(s.loc[s.spec_L.idxmin()].gamma)
        print(f"   {setting:<24} γ*={gstar}  speedup={s.latency_speedup.max():.2f}x")


# ===========================================================================
# Priority 3 - KV organizations and verification dataflows
# ===========================================================================
def kv_dataflow():
    df = _load("kv_dataflow")
    a0 = _nearest_alpha(df, 0.8)
    g0 = 4

    # (a) kv_org, (b) kv_size, (c) kv_bw : latency vs ctx
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (study, title) in zip(
            axes, [("kv_org", "KV organization"), ("kv_size", "KV buffer size"),
                   ("kv_bw", "KV read BW (paged↔contig)")]):
        sub = df[(df.study == study) & (df.gamma == g0) & (df.alpha == a0)]
        for setting, s in sub.groupby("setting"):
            s = s.sort_values("ctx")
            ax.plot(s.ctx, s.spec_L, "o-", label=setting)
        ax.set_title(title); ax.set_xlabel("context length")
        ax.set_ylabel("spec per-token latency (s)"); ax.legend(fontsize=8)
    fig.suptitle(f"KV-cache organizations (γ={g0}, α={a0:.2f})")
    save(fig, "p3_kv_organizations")

    # (d) verify fanout: latency vs fanout at several ctx
    sub = df[(df.study == "verify_fanout") & (df.gamma == g0) & (df.alpha == a0)].copy()
    sub["vf"] = sub.setting.str.replace("vf", "").astype(int)
    fig, (axl, axe) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ctx, s in sub.groupby("ctx"):
        s = s.sort_values("vf")
        axl.plot(s.vf, s.spec_L, "o-", label=f"ctx={ctx}")
        axe.plot(s.vf, s.spec_E, "o-", label=f"ctx={ctx}")
    for ax, lab in ((axl, "latency"), (axe, "energy")):
        ax.set_xlabel("verify fanout (parallel speculative positions)")
        ax.set_ylabel(f"spec per-token {lab}"); ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Verification dataflow: parallel-verify fanout (γ={g0}, α={a0:.2f})")
    save(fig, "p3_verify_fanout")

    print("\n[P3] KV-org spec latency @ctx=2048 (γ=4, α={:.2f}):".format(a0))
    s = df[(df.study == "kv_org") & (df.gamma == g0) & (df.alpha == a0) & (df.ctx == 2048)]
    for _, r in s.iterrows():
        print(f"   {r.setting:<12} L={r.spec_L:.3e}  E={r.spec_E:.3e}")


# ===========================================================================
# EAGLE-3
# ===========================================================================
def eagle():
    df = _load("eagle")
    ctxs = sorted(df.ctx.unique())
    c0 = 2048 if 2048 in ctxs else ctxs[len(ctxs) // 2]
    TAU_CAL = 6.0                                 # calibrated to paper (tau~5.8-6.6)
    PREFERRED_TREE = "tree_d6_n63"                # full-binary layers: 1 2 4 8 16 32
    PAPER_LO, PAPER_HI = 4.1, 5.5                  # paper batch-1 speedup range (Table 1)

    base = df[df.method == "baseline"].set_index("ctx").L_per_tok
    van = df[df.method == "vanilla"]
    eag = df[df.method == "eagle"]
    tree_options = sorted(eag.config.unique())
    TREE = PREFERRED_TREE if PREFERRED_TREE in tree_options else tree_options[len(tree_options) // 2]
    a0 = _nearest_alpha(van, 0.8)

    def best_van(ctx, metric="L_per_tok"):
        s = van[(van.ctx == ctx) & (van.alpha == a0)]
        return s.loc[s[metric].idxmin()]

    def eagle_cal(ctx):
        s = eag[(eag.ctx == ctx) & (eag.config == TREE) & (eag.tau == TAU_CAL)]
        return s.iloc[0]

    # Fig 1: per-token latency & energy vs ctx (baseline / best vanilla / EAGLE tau=6)
    fig, (axl, axe) = plt.subplots(1, 2, figsize=(11, 4.2))
    series = {
        "baseline": ([base[c] for c in ctxs],
                     [df[(df.method == "baseline") & (df.ctx == c)].E_per_tok.iloc[0] for c in ctxs], "k:"),
        f"best vanilla (α={a0:.2f})": ([best_van(c).L_per_tok for c in ctxs],
                                       [best_van(c).E_per_tok for c in ctxs], "o--"),
        f"EAGLE-3 (τ={TAU_CAL:.0f})": ([eagle_cal(c).L_per_tok for c in ctxs],
                                       [eagle_cal(c).E_per_tok for c in ctxs], "s-"),
    }
    for label, (L, E, st) in series.items():
        axl.plot(ctxs, L, st, label=label); axe.plot(ctxs, E, st, label=label)
    axl.set_ylabel("per-token latency (s)"); axe.set_ylabel("per-token energy (J)")
    for ax in (axl, axe):
        ax.set_xlabel("context length"); ax.legend()
    fig.suptitle("EAGLE-3 (calibrated τ) vs vanilla spec vs baseline (depth-aware)")
    save(fig, "p4_eagle_vs_vanilla")

    # Fig 2: speedup vs ACHIEVED acceptance length tau (the common currency)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    v = van[van.ctx == c0].sort_values("tau")
    ax.plot(v.tau, v.latency_speedup, "o", ms=4, alpha=0.5, label="vanilla (γ,α sweep)")
    for tree, s in eag[eag.ctx == c0].groupby("config"):
        s = s.sort_values("tau")
        ax.plot(s.tau, s.latency_speedup, "s-", label=f"EAGLE-3 {tree}")
    ax.axvspan(5.8, 6.6, color="green", alpha=0.10, label="paper τ (Table 1)")
    ax.axhspan(PAPER_LO, PAPER_HI, color="orange", alpha=0.08, label="paper speedup")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("achieved acceptance length τ (tokens/round)")
    ax.set_ylabel("latency speedup vs baseline")
    ax.set_title(f"Speedup vs acceptance length (ctx={c0})")
    ax.legend(fontsize=8)
    save(fig, "p4_eagle_speedup_vs_tau")

    e0 = eagle_cal(c0)
    print(f"\n[EAGLE] ctx={c0}, calibrated τ={TAU_CAL:.0f}, tree={TREE}:")
    print(f"   EAGLE-3 speedup = {e0.latency_speedup:.2f}x  (paper batch-1: {PAPER_LO}-{PAPER_HI}x)")
    print(f"   best vanilla (α={a0:.2f}) speedup = {best_van(c0).latency_speedup:.2f}x "
          f"(τ={best_van(c0).tau:.2f})")
    for tau in sorted(eag.tau.unique()):
        s = eag[(eag.ctx == c0) & (eag.config == TREE) & (eag.tau == tau)]
        if len(s):
            print(f"   τ={tau:.0f}: EAGLE-3 speedup = {s.iloc[0].latency_speedup:.2f}x")


def kv_longctx():
    df = _load("kv_longctx")
    a0 = _nearest_alpha(df, 0.8)
    sub = df[(df.study == "kv_org") & (df.alpha == a0)]
    ctxs = sorted(sub.ctx.unique())
    fig, (axl, axe) = plt.subplots(1, 2, figsize=(11, 4.2))
    for org, s in sub.groupby("setting"):
        s = s.sort_values("ctx")
        axl.plot(s.ctx, s.spec_L * 1e3, "o-", label=org)
        axe.plot(s.ctx, s.spec_E, "o-", label=org)
    axl.set_ylabel("spec per-token latency (ms)"); axe.set_ylabel("spec per-token energy (J)")
    for ax in (axl, axe):
        ax.set_xlabel("context length"); ax.set_xscale("log", base=2); ax.legend()
    fig.suptitle(f"KV organization at long context (γ=4, α={a0:.2f})")
    save(fig, "p3_kv_longctx")

    # energy savings of dedicated KV buffer vs shared-global baseline
    print(f"\n[P3-long] energy savings of kv_buffer vs baseline (γ=4, α={a0:.2f}):")
    for c in ctxs:
        b = sub[(sub.setting == "baseline") & (sub.ctx == c)].spec_E.iloc[0]
        k = sub[(sub.setting == "kv_buffer") & (sub.ctx == c)].spec_E.iloc[0]
        bl = sub[(sub.setting == "baseline") & (sub.ctx == c)].spec_L.iloc[0]
        kl = sub[(sub.setting == "kv_buffer") & (sub.ctx == c)].spec_L.iloc[0]
        print(f"   ctx={c:6d}: energy -{100*(1-k/b):.1f}%  latency -{100*(1-kl/bl):.1f}%")


def _breakeven(bs, speedups):
    """First batch where speedup crosses below 1.0 (linear interp on log2 B)."""
    import numpy as _np
    bs = _np.asarray(bs, float); sp = _np.asarray(speedups, float)
    below = _np.where(sp < 1.0)[0]
    if len(below) == 0:
        return float("inf")
    i = below[0]
    if i == 0:
        return bs[0]
    x0, x1 = _np.log2(bs[i - 1]), _np.log2(bs[i])
    y0, y1 = sp[i - 1], sp[i]
    return float(2 ** (x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)))


# ===========================================================================
# Batch size: TPOT, break-even batch, load-aware lookahead
# ===========================================================================
def batch():
    df = _load("batch")
    main = df[df.study == "batch_main"]
    a0 = 0.8
    Bs = sorted(main.batch.unique())
    base = main[main.method == "baseline"].sort_values("batch")

    # Fig A: TPOT growth + TPOT-speedup decay with batch
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(11, 4.2))
    axt.plot(base.batch, base.tpot * 1e3, "k:o", label="baseline")
    g4 = main[(main.method == "spec") & (main.gamma == 4) & (main.alpha == a0)].sort_values("batch")
    axt.plot(g4.batch, g4.tpot * 1e3, "s-", label="spec γ=4")
    axt.set_xscale("log", base=2); axt.set_yscale("log"); axt.set_xlabel("batch size")
    axt.set_ylabel("TPOT (ms/output token)"); axt.legend()
    axt.set_title(f"TPOT rises with batch (α={a0})")
    for g in [1, 2, 4, 8]:
        s = main[(main.method == "spec") & (main.gamma == g) & (main.alpha == a0)].sort_values("batch")
        axs.plot(s.batch, s.speedup, "o-", label=f"γ={g}")
    axs.axhline(1, color="k", lw=1, ls="--", label="break-even")
    axs.set_xscale("log", base=2); axs.set_xlabel("batch size")
    axs.set_ylabel("TPOT speedup (baseline/spec)"); axs.legend(fontsize=8)
    axs.set_title("Speculation decays with batch → break-even B*")
    save(fig, "p5_batch_breakeven")

    # Fig B: load-aware lookahead — TPOT-optimal γ falls with batch
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for a in [0.7, 0.8, 0.9]:
        gstar = []
        for B in Bs:
            s = main[(main.method == "spec") & (main.alpha == a) & (main.batch == B)]
            gstar.append(s.loc[s.tpot.idxmin()].gamma if len(s) else None)
        ax.plot(Bs, gstar, "o-", label=f"α={a}")
    ax.set_xscale("log", base=2); ax.set_xlabel("batch size")
    ax.set_ylabel("TPOT-optimal lookahead γ*")
    ax.set_title("Load-aware lookahead: γ* shrinks as batch grows")
    ax.legend()
    save(fig, "p5_load_aware_gamma")

    # Fig C: hardware levers on the break-even batch
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
    bw = df[(df.study == "batch_bw") & (df.method == "spec")]
    for setting, s in bw.groupby("setting"):
        s = s.sort_values("batch")
        axb.plot(s.batch, s.speedup, "o-", label=setting)
    axb.axhline(1, color="k", lw=1, ls="--")
    axb.set_xscale("log", base=2); axb.set_xlabel("batch size")
    axb.set_ylabel("TPOT speedup"); axb.set_title("DRAM bandwidth shifts B* (γ=4, α=0.8)")
    axb.legend(fontsize=8)
    ratio = df[(df.study == "batch_ratio") & (df.method == "spec")]
    for setting, s in ratio.groupby("setting"):
        s = s.sort_values("batch")
        axr.plot(s.batch, s.speedup, "o-", label=setting)
    axr.axhline(1, color="k", lw=1, ls="--")
    axr.set_xscale("log", base=2); axr.set_xlabel("batch size")
    axr.set_ylabel("TPOT speedup"); axr.set_title("Draft/target ratio shifts B* (γ=4, α=0.8)")
    axr.legend(fontsize=8)
    save(fig, "p5_breakeven_hardware")

    # numeric validation
    print("\n[BATCH] break-even batch B* (speedup<1), main 6.7b->175b, α=0.8:")
    for g in [1, 2, 4, 8]:
        s = main[(main.method == "spec") & (main.gamma == g) & (main.alpha == a0)].sort_values("batch")
        be = _breakeven(s.batch.tolist(), s.speedup.tolist())
        print(f"   γ={g}: B*={be:.0f}" if be != float("inf") else f"   γ={g}: B*>{Bs[-1]}")
    print("[BATCH] B* vs DRAM bandwidth (γ=4, α=0.8):")
    for setting, s in df[(df.study == "batch_bw") & (df.method == "spec")].groupby("setting"):
        s = s.sort_values("batch")
        be = _breakeven(s.batch.tolist(), s.speedup.tolist())
        print(f"   {setting:<8}: B*={be:.0f}" if be != float("inf") else f"   {setting:<8}: B*>{s.batch.max()}")


# ===========================================================================
# Architecture exploration for throughput
# ===========================================================================
def arch_throughput():
    df = _load("arch_throughput")
    named = df[df.kind == "named"]
    op = (64, 4096)
    s = named[(named.batch == op[0]) & (named.ctx == op[1])].copy().sort_values("spec_tput")

    # Fig A: throughput vs energy across named configs
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(s.config, s.spec_tput, color="steelblue")
    ax.set_xlabel("spec throughput (output tok/s)")
    ax.set_title(f"What helps throughput? (B={op[0]}, ctx={op[1]}, γ=4, α=0.8)")
    ax2 = ax.twiny()
    ax2.plot(s.spec_energy_per_tok, s.config, "D", color="darkorange", label="energy/token")
    ax2.set_xlabel("spec energy per token (J)  ◆")
    save(fig, "p6_arch_named")

    # Fig B: DRAM-bandwidth roofline — throughput vs batch
    roof = df[df.kind == "roofline"]
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(11, 4.2))
    for bw, g in sorted(roof.groupby("config"), key=lambda kv: float(kv[0][:-4])):
        g = g.sort_values("batch")
        axt.plot(g.batch, g.spec_tput, "o-", label=bw)
        axs.plot(g.batch, g.tput_gain, "o-", label=bw)
    axt.set_xscale("log", base=2); axt.set_yscale("log")
    axt.set_xlabel("batch size"); axt.set_ylabel("spec throughput (tok/s)")
    axt.set_title("Throughput roofline vs DRAM bandwidth"); axt.legend(fontsize=8)
    axs.axhline(1, color="k", lw=1, ls="--")
    axs.set_xscale("log", base=2); axs.set_xlabel("batch size")
    axs.set_ylabel("spec speedup"); axs.set_title("Faster DRAM → smaller speculative speedup")
    axs.legend(fontsize=8)
    save(fig, "p6_dram_roofline")

    print(f"\n[ARCH] spec throughput by config (B={op[0]}, ctx={op[1]}):")
    for _, r in s.iterrows():
        print(f"   {r.config:<20} tput={r.spec_tput:8.1f} tok/s  energy/tok={r.spec_energy_per_tok:.4f} J")


SECTIONS = {"scheduler": scheduler, "generality": generality,
            "kv_dataflow": kv_dataflow, "eagle": eagle, "kv_longctx": kv_longctx,
            "batch": batch, "arch_throughput": arch_throughput}

if __name__ == "__main__":
    which = sys.argv[1:] or list(SECTIONS)
    for name in which:
        if (RES / f"{name}.csv").exists() or any(RES.glob(f"{name}.shard*.csv")):
            print(f"=== {name} ===")
            SECTIONS[name]()
        else:
            print(f"skip {name} (no results)")
