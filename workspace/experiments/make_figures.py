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
    a0 = _nearest_alpha(df, 0.8)

    def best(method, ctx, a, metric="L_per_tok"):
        s = df[(df.method == method) & (df.ctx == ctx) & (df.alpha == a)]
        return s.loc[s[metric].idxmin()]

    # Fig: per-token latency vs ctx for baseline / best vanilla / best eagle
    fig, (axl, axe) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, style in [("baseline", "k:"), ("vanilla", "o--"), ("eagle", "s-")]:
        L = [best(method, c, a0, "L_per_tok").L_per_tok for c in ctxs]
        E = [best(method, c, a0, "E_per_tok").E_per_tok for c in ctxs]
        axl.plot(ctxs, L, style, label=method)
        axe.plot(ctxs, E, style, label=method)
    axl.set_ylabel("per-token latency (s)"); axe.set_ylabel("per-token energy (J)")
    for ax in (axl, axe):
        ax.set_xlabel("context length"); ax.legend()
    fig.suptitle(f"EAGLE-3 vs vanilla spec vs baseline (depth-aware, α={a0:.2f})")
    save(fig, "p4_eagle_vs_vanilla")

    # Fig: speedup vs alpha at ctx=2048
    c0 = 2048 if 2048 in ctxs else ctxs[len(ctxs) // 2]
    alphas = sorted(df.alpha.unique())
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for method, style in [("vanilla", "o--"), ("eagle", "s-")]:
        sp = [best(method, c0, a).latency_speedup for a in alphas]
        ax.plot(alphas, sp, style, label=f"best {method}")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("acceptance α"); ax.set_ylabel("latency speedup vs baseline")
    ax.set_title(f"EAGLE-3 vs vanilla: speedup vs α (ctx={c0})")
    ax.legend()
    save(fig, "p4_eagle_speedup_vs_alpha")

    print(f"\n[EAGLE] @ctx={c0}, α={a0:.2f}:")
    for m in ("baseline", "vanilla", "eagle"):
        b = best(m, c0, a0)
        print(f"   {m:<9} L={b.L_per_tok:.3e}  speedup={b.latency_speedup:.2f}x  cfg={b.get('config','-')}")


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


SECTIONS = {"scheduler": scheduler, "generality": generality,
            "kv_dataflow": kv_dataflow, "eagle": eagle, "kv_longctx": kv_longctx}

if __name__ == "__main__":
    which = sys.argv[1:] or list(SECTIONS)
    for name in which:
        if (RES / f"{name}.csv").exists() or any(RES.glob(f"{name}.shard*.csv")):
            print(f"=== {name} ===")
            SECTIONS[name]()
        else:
            print(f"skip {name} (no results)")
