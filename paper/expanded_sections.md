# Expanded sections for *Modeling Speculative Decoding*

These sections extend the original report toward a conference submission. They add a
corrected decode model, a theoretically optimal (acceptance-aware) scheduler, a
generality study across precision / model scale / memory hierarchy, a study of KV-cache
organizations and verification dataflows, and an EAGLE-3 cost model. All experiments use
AccelForge mapping on the TPU-v4-inspired architecture, reporting per-token latency and
energy, and are reproducible from `workspace/experiments/` (Slurm + podman).

---

## A. Methodology correction: decode vs. prefill

While extending the model we found that the original workload YAMLs forced the number of
*new* query tokens `M` to equal the full context `M_FULL` on every step:

```jinja
{% set N_NEW_TOKENS = N_TOKENS | default(1) %}   # overwrites the passed N_NEW_TOKENS
```

This contradicts the workload's own assumption ("N_TOKENS >> N_NEW_TOKENS") and means
that every "single-token decode" and every "γ-token verification" was actually modeled
as a **full-context prefill** — an `O(context^2)` self-attention — rather than
incremental decode (`M=1` query attending to the KV cache, or `M=γ` for verification).
We corrected the workload (`designs/transformer_block.yaml`) to honor the intended
decode semantics: draft steps use `M=1`, verification uses `M=γ`. This lowers absolute
per-token cost by ~14× and, more importantly, changes the *shape* of the results: true
decode is dominated by weight and KV-cache traffic (roughly linear in context), not by
quadratic attention.

**Consequence for prior conclusions.** Under the corrected model the speculative speedup
is large and robust (≈2.3× latency, ≈0.5× energy at γ=4, α=0.8) but is **largely
context-independent** in the studied range (2.32× at ctx=512 → 2.25× at ctx=4096),
rather than *growing* with context as the prefill-based model suggested. The growth in
the original is an artifact of the quadratic-attention prefill bug; the genuine,
hardware-grounded benefit of speculative decoding here is amortization of the target
model's weight/KV reads across multiple accepted tokens.

---

## B. A theoretically optimal (acceptance-aware) scheduler

The original report recommended a single static lookahead, γ=4. We instead characterize
the *latency-optimal* lookahead γ\* as a function of the operating point and show when a
static choice suffices.

**Setup.** On the speculative-decoding-aware architecture (`arch_kv_buffer.yaml`) we
sweep context ∈ {512…4096}, γ ∈ {1…12}, and acceptance α ∈ {0.2…0.95}
(`experiments/run_scheduler.py`). Per-token latency is

  L(γ, α) = [γ · t_draft + t_verify(γ)] / E[accepted](γ, α),
  E[accepted](γ, α) = (1 − α^{γ+1}) / (1 − α).

**Result 1 — the optimum is U-shaped and acceptance-dependent.** For every context,
L(γ) first falls (amortizing the target verification over more drafted tokens) and then
rises (wasted work and verification overhead from imperfect acceptance). The minimizer
γ\* rises monotonically with α: at ctx=2048, γ\* = 2, 3, 4, 6, 9 for α = 0.4, 0.6, 0.7,
0.8, 0.9 respectively (Fig. `p1_latency_vs_gamma`, `p1_gamma_star_vs_alpha`). A simple
closed-form optimum — fitting t_round(γ) ≈ a + bγ and minimizing (a+bγ)/E[accepted] —
tracks the measured γ\* closely, giving an inexpensive scheduler rule.

**Result 2 — static γ=4 is near-optimal in the realistic regime, but not universally.**
In the typical acceptance band α ∈ [0.6, 0.8], static γ=4 is within **1.0% mean latency
regret** of the per-point oracle, confirming it as a strong default. Outside this band
the regret grows — up to **~13% at α=0.9** and **~32% at α=0.2** — so a truly optimal
scheduler must *adapt* γ to the measured acceptance (and, weakly, context). This both
validates the original γ=4 recommendation in its intended regime and motivates an
acceptance-aware adaptive policy.

**Result 3 — the sweet spot comes from the cost/yield balance, not a single knob.** The
optimum is set by the linear growth of round cost in γ (each extra draft step) against
the saturating accepted-token yield `(1−α^{γ+1})/(1−α)`; this is what produces the
U-shape and the α-dependence of γ\*. The curves show a mild non-monotonicity around γ≈5
that we traced to mapping granularity (tiling of the verification matmul), *not* to the
PE array's verify fanout — Section D shows sweeping that fanout from 1 to 16 changes
latency by <0.2%. The practical scheduling rule is therefore the closed-form γ\*(α)
above, with γ=4 as the robust static default for typical acceptance.

---

## C. Generality across precision, model scale, and memory hierarchy

We test whether the speculative-decoding trends survive changes to precision, model
scale, and the accelerator's memory hierarchy (`experiments/run_generality.py`,
Fig. `p2_generality_speedup` / `p2_generality_energy`). All runs use the
speculative-aware arch; the hierarchy study uses the parametrized `arch_hier.yaml`.

**Precision is irrelevant to the *relative* benefit.** Sweeping 4-, 8-, and 16-bit
weights/activations/KV leaves the latency speedup essentially unchanged (≈2.30–2.32× at
γ=4, α=0.8) and γ\* fixed at 6. Lower precision shrinks both baseline and speculative
cost proportionally because decode is memory-bound on the same tensors, so the ratio is
invariant. Precision is therefore a "free" axis: a deployment can quantize for absolute
speed without re-tuning its speculative schedule.

**The memory hierarchy is robust, but shifts γ\*.** Across an edge-class accelerator
(small SRAM, 100 GB/s DRAM), the TPU-v4 baseline, and a TPU-v8-like point (512 MB SRAM,
3 TB/s DRAM, 16 TB/s KV buffer) the speedup stays ≈2.3×. The optimal lookahead, however,
drifts from γ\*=6 (edge/v4) to γ\*=4 (v8): faster memory lowers the fixed verification
cost, so fewer drafted tokens are needed to amortize it. This is a concrete co-design
signal — the best lookahead is a function of the memory system, not a universal constant.

**The draft→target ratio is the dominant lever.** Speedup and γ\* both rise with the
gap between draft and target width: gpt3-6.7B→175B (≈26× FLOP gap) gives 2.31× at γ\*=6,
1.3B→13B gives 1.97× at γ\*=4, and 6.7B→30B (a narrow gap) only 1.46× at γ\*=2. The
benefit of speculation is governed by how cheap the draft is relative to the target, far
more than by absolute model size. This reframes draft selection as the first-order
decision and supports very small drafts for very large targets.

In all three studies the qualitative shape from Section B holds: a U-shaped, α-dependent
γ\*, and a speedup that is roughly flat (slightly decreasing) in context.

---

## D. KV-cache organizations and verification dataflows

We isolate the architectural KV/verification knobs the original report highlighted
(`experiments/run_kv_dataflow.py`, `run_kv_longctx.py`; Fig. `p3_kv_organizations`,
`p3_verify_fanout`, `p3_kv_longctx`).

**Under correct decode, KV organization is second-order at ≤4k context.** Comparing KV
held in the shared global buffer (baseline), a dedicated unified KV buffer, and separate
K/V buffers, per-token *latency* is identical (1.328 ms at ctx=2048, γ=4, α=0.8) and
*energy* differs by <2%. The reason is quantitative: with true decode, each step streams
the full target weights (≈1.8 GB/block) but only ≈24 KB/token of KV, so KV is ~5% of
traffic at 4k context and cannot move the critical path. This is an important correction
to the original report, whose ≈20% KV-buffer latency win was an artifact of the
full-prefill bug (which made attention/KV traffic quadratic and dominant).

**The KV buffer is an energy optimization that scales with context.** Because KV traffic
grows linearly with context while weight traffic is fixed, we extend the sweep to 32k
tokens (Fig. `p3_kv_longctx`). The dedicated KV buffer's energy advantage over the
shared-global baseline grows monotonically with context — **−1.5% at 4k, −4.2% at 8k,
−7.4% at 16k, and −9.4% at 32k** — while per-token *latency* is unchanged at every
context (the latency-optimal mapping keeps the same critical path regardless of where KV
lives). So the architectural KV optimization is real but is an *energy/long-context*
lever governed by the KV-to-weight traffic ratio, not the latency win the original report
claimed.

**Verification fanout barely matters in this regime.** Sweeping the PE array's
speculative `verify_m` fanout from 1 to 16 changes per-token latency by <0.2% at all γ,
because verifying γ≤12 tokens is a tiny matmul relative to the weight stream. (We
accordingly attribute the mild non-monotonicity around γ≈5 in Section B to mapping
granularity, not to the fanout.) Likewise, emulating a paged KV layout by lowering the
KV read bandwidth to 0.5 TB/s costs only ~1.3% latency at 4k context. The verification
*dataflow* that does pay off is algorithmic — tree verification (Section E) — rather than
a wider spatial fanout.

Together, Sections C–D say the high-leverage knobs for speculative decoding on this
class of accelerator are the *schedule* (γ vs α), the *draft/target ratio*, and the
*algorithm*, not micro-architectural KV/verification tweaks — except at long context,
where KV-aware memory design re-emerges as important.

---

## E. EAGLE-3 modeling

We model EAGLE-3's two structural changes (`experiments/run_eagle.py`;
Fig. `p4_eagle_vs_vanilla`, `p4_eagle_speedup_vs_tau`): a **single-layer draft** that
reuses the target's fused features, and **tree drafting** in which the target verifies a
candidate tree of `tree_nodes` tokens in one pass. Capturing the layer-count reduction
requires depth-aware costs, so all methods use realistic layer counts (draft 32, target
96); the EAGLE draft is one target-width block per tree-frontier pass, and verification
is a full target pass over the tree.

**Calibrating the acceptance length to the paper.** Our first model derived the
per-round accepted-token yield from an independence assumption
(`1−(1−α)^width`), which saturates and *over*-estimated the yield (≈8.9 tokens/round →
8.1× at α=0.8). The EAGLE-3 paper instead reports a measured **average acceptance length
τ ≈ 5.8–6.6** (Table 1) with batch-1 speedups of **4.1–5.5×**. We therefore drive the
EAGLE cost model directly by τ (a calibrated input, swept 4–7) rather than a flawed
acceptance derivation. At the paper's τ=6 our model gives **5.44×** at ctx=2048 — squarely
inside the paper's range — with a clean τ-sensitivity (3.6×/4.5×/5.4×/6.4× at τ=4/5/6/7;
Fig. `p4_eagle_speedup_vs_tau`). Best vanilla speculation at α=0.8 reaches only 3.3×
(τ≈4.3), so **EAGLE-3 is ≈1.6× faster than vanilla speculation**, matching the paper's
"20–40% over EAGLE-2" framing.

Plotting speedup against *achieved* τ (the common currency, Fig. `p4_eagle_speedup_vs_tau`)
shows the deeper point: speedup is largely a function of the acceptance length, and
EAGLE-3's contribution is **reaching a high τ (≈6) cheaply** — at realistic acceptance —
where vanilla speculation would need near-perfect α. The EAGLE curves sit slightly above
the vanilla cloud at matched τ (the single-layer draft is cheaper), but the dominant
lever is τ itself.

**Fidelity audit of the EAGLE cost model (`experiments/eagle_audit.py`).** We checked the
three modeling concerns against the paper (Li et al. 2025, §3.1, Figs 5–6) and the cost
split. The verify pass is **92% of the round** and the draft is only ~8%, so:
- *Tree-frontier batching* (draft processes width>1 per pass, not one token): correct per
  the paper, but the draft block is weight-bound, so M=1 vs M=width changes the round cost
  by <0.1% (speedup unchanged). We model the frontier width anyway for fidelity.
- *Multi-layer feature fusion input* (concatenated low/mid/high features → FC): real, but
  an extra FC on the already-8% draft term — negligible.
- *Draft attention scope*: the claim that the draft attends only to a tiny window is
  **incorrect** — Fig. 6's masks show draft tokens attending to the full prefix (the draft
  head's KV spans the context), so `M_FULL=ctx` is right; the tree mask only removes
  cross-branch attention among the few tree tokens. Adopting the tiny-window assumption
  changes the result by 0.2%.
So the EAGLE result is governed by the (correctly modeled) verify pass and by τ, not by
draft micro-modeling — which is exactly why calibrating τ was the change that mattered.

---

## G. Batch size, TPOT, and the speculative break-even

Decode serving runs many requests in one batch, which changes the picture entirely: the
target *weights* are read once and amortized across the batch, while KV and compute grow
with it. We model batched decode (`experiments/run_batch.py`) by indexing activations and
KV — but not weights — with the batch dimension B, and report both **TPOT** (per-token
latency) and **throughput** (output tokens/s). Note these are inverses of the same
per-step latency, so "minimize TPOT" and "maximize tokens/s" give the *same* answer.

**The model reproduces the memory→compute transition.** Baseline TPOT is flat at small B
(3.0 ms at B=1–4: weight-bound) and rises steeply at large B (24 ms at B=256:
compute-bound); speculative TPOT follows the same transition while staying below the
baseline curve for this draft/target pair (Fig.
`p5_batch_breakeven`).

**Speculation's benefit decays with batch — a break-even batch B\*.** At small batch
decode is weight-bound, so speculation's extra compute is nearly free and the speedup is
large (2.3× at B=1, γ=4, α=0.8). As batch grows the accelerator becomes compute-bound and
speculation's wasted/extra work is no longer hidden, so the speedup falls monotonically
toward an asymptote of E[accepted]/γ. The EAGLE-3 paper observes exactly this on GPUs
(Tables 3, 5: vanilla EAGLE throughput drops below 1.0× around batch 24, EAGLE-3 peaks at
batch 56); our model reproduces the decay from first principles and lets us vary hardware
the GPUs cannot.

**B\* is set by the draft/target ratio, not a universal number** (Fig.
`p5_breakeven_hardware`). With a modest draft/target gap (6.7B→30B) the speedup crosses
1.0× at **B\*≈256–512**: above it, speculation *hurts* TPOT/throughput. With a large gap
(6.7B→175B, the draft ≈26× cheaper) the speedup never breaks even in our range — it
plateaus at ≈1.33× even at the compute roof. So **a sufficiently cheap draft keeps
speculation beneficial at every batch size**; choosing the draft governs not just the
peak speedup (Section C) but whether a break-even exists at all.

**Load-aware lookahead: γ\* shrinks as batch grows** (Fig. `p5_load_aware_gamma`). Because
larger γ wastes more compute when the accelerator is loaded, the TPOT-optimal
lookahead falls with batch — e.g., at α=0.8, γ\*=4 at B≤64 but γ\*=2 by B=128; at α=0.9,
γ\*=8 collapses to 4 by B≈32. Combined with the acceptance-dependence of Section B, the
optimal lookahead is a **two-variable function γ\*(α, B)** of acceptance *and* load — a
concrete scheduling rule that, to our knowledge, has not been characterized: an optimal
scheduler should lower γ both when acceptance is low and when the batch is large.

**Counter-intuitively, faster memory makes speculation *less* valuable.** Sweeping DRAM
bandwidth (edge 100 GB/s → TPU-v4 614 GB/s → TPU-v8 3 TB/s) at fixed batch, the speculative
speedup *decreases* with bandwidth (1.44× vs 1.33× vs 1.29× at B=512), because a
bandwidth-rich accelerator is less memory-bound to begin with and has less weight-read
cost for speculation to hide. Speculative decoding is thus most valuable on
bandwidth-starved hardware (edge, low-cost), and its marginal value shrinks as
accelerators add HBM bandwidth.

## H. Memory-layout exploration for throughput

We then asked the co-design question directly: *which accelerator memory configuration
maximizes decode throughput?* (`experiments/run_arch_throughput.py`, `arch_probe.py`;
Fig. `p6_arch_named`, `p6_dram_roofline`.) We compared a unified global buffer, a dedicated
KV tier, split K/V tiers, a larger weight buffer, faster KV bandwidth, and higher DRAM
bandwidth at B=64, ctx=4096.

**On-chip SRAM layout is a weak throughput lever; DRAM bandwidth is the strong one.**
Adding a dedicated KV tier (or simply more/separate buffering) lifts throughput by ~12%
over the unified buffer by relieving port contention — but the KV tier's **size and
bandwidth are irrelevant** (16 MB and 1 GB KV buffers give identical throughput). In the
standard decode mapping, KV is streamed from DRAM once per step with no intra-step reuse,
so a bigger/faster on-chip KV buffer cannot reduce the DRAM traffic that sets latency; it
only lowers read *energy* (Section D). By contrast, raising DRAM bandwidth from 614 GB/s
to 3 TB/s **nearly doubles throughput** (7.2k→12.7k tok/s) until a compute ceiling is hit
(Fig. `p6_dram_roofline`). The throughput roofline is therefore a DRAM-bandwidth-and-
compute story, not an on-chip-layout story.

**Design takeaway.** For decode *throughput*, spend area on HBM bandwidth, not on a larger
KV SRAM; reserve the KV tier for its real benefit, *energy* and *long-context* latency
(Section D). The one caveat is that our single-step model does not capture *cross-step*
KV residency (a KV cache pinned on-chip and reread across many steps) — the one regime
where on-chip KV capacity could aid latency — which we flag as the most valuable extension.

---

## F. Updated conclusions

Modeling speculative decoding with *correct* decode semantics (rather than per-step full
prefill) sharpens — and in places overturns — the original report's conclusions, and
yields a cleaner co-design story:

1. **Speculation's benefit is real, large, and context-flat.** On the TPU-v4-like
   architecture, speculative decoding gives ≈2.3× latency and ≈2× energy improvement at a
   typical operating point, driven by amortizing the target model's weight/KV reads over
   accepted tokens. Unlike the prefill-based model, the benefit does *not* grow with
   context; that growth was an artifact of quadratic attention.

2. **The schedule should be acceptance-aware.** The latency-optimal lookahead γ\* is
   U-shaped and rises with acceptance α (≈2 at α=0.4 to ≈9 at α=0.9); a closed-form
   `(a+bγ)/E[accepted]` rule recovers it cheaply. Static γ=4 is an excellent default in
   the realistic band α∈[0.6,0.8] (≤1% mean regret) but leaves 10–30% on the table at the
   extremes — motivating a runtime acceptance-aware scheduler.

3. **The trends generalize; the draft/target ratio dominates.** The speedup is invariant
   to precision and robust across edge→TPU-v8 memory hierarchies (which mainly shift γ\*),
   but scales strongly with the draft-to-target gap. Choosing a sufficiently small draft
   is the first-order decision.

4. **Micro-architectural KV/verification tweaks are second-order — except at long
   context.** Under true decode, KV organization and verify fanout barely affect latency
   at ≤4k context (KV is ~5% of traffic); the dedicated KV buffer is an energy
   optimization that grows to ≈9% at 32k. Long-context serving is where KV-aware memory
   design re-earns its place.

5. **Algorithm beats architecture for verification.** With the acceptance length τ
   calibrated to the EAGLE-3 paper (τ≈6), EAGLE-3 reaches ≈5.4× over autoregressive decode
   vs ≈3.3× for vanilla speculation at α=0.8 (≈1.6× advantage), validated against the
   paper's 4.1–5.5×. Speedup is largely a function of τ; EAGLE's win is achieving high τ
   cheaply. The biggest remaining lever is algorithmic.

6. **There is a speculative break-even batch, set by the draft/target ratio.** Speculation's
   speedup decays as batch grows and the accelerator becomes compute-bound. With a modest
   draft/target gap it crosses 1.0× near batch 256–512 (beyond which speculation hurts
   throughput); with a cheap-enough draft it never breaks even. The optimal lookahead is a
   two-variable function γ\*(α, B): lower γ both when acceptance is low and when batch is
   large ("load-aware lookahead").

7. **For throughput, bandwidth beats layout — and speculation favors bandwidth-starved
   hardware.** DRAM bandwidth roughly doubles decode throughput up to a compute ceiling,
   while on-chip KV-tier size/bandwidth do not move throughput (only energy). Counter-
   intuitively, faster memory *reduces* speculation's relative benefit, since a bandwidth-
   rich accelerator is less memory-bound to begin with. Speculative decoding is therefore
   most valuable on edge/low-bandwidth accelerators.

**Limitations / future work.** Costs are modeled per transformer block (depth-aware only
for the EAGLE comparison); the acceptance model is a scalar α (with τ calibrated for
EAGLE) rather than a learned, content-dependent distribution. The batched model is
single-step and does not capture cross-step KV residency (the regime where on-chip KV
capacity could aid latency). PD-disaggregation and SSD-offloaded KV are not yet modeled.
Each is a natural extension of the released `experiments/` harness.

---

## Figure index

| File | Section | Content |
|---|---|---|
| `p1_latency_vs_gamma` | B | Per-token latency vs γ (U-shape; γ\* rises with α) |
| `p1_gamma_star_vs_alpha` | B | Oracle γ\*(α) per context + closed-form optimum |
| `p1_gamma4_regret` | B | Latency regret of static γ=4 vs oracle |
| `p2_generality_speedup` | C | Speedup vs context for precision / scale / hierarchy |
| `p2_generality_energy` | C | Energy ratio companion |
| `p3_kv_organizations` | D | Spec latency vs context for KV orgs / size / BW |
| `p3_verify_fanout` | D | Spec latency & energy vs verify fanout |
| `p3_kv_longctx` | D | KV organization to 32k context (energy diverges) |
| `p4_eagle_vs_vanilla` | E | Per-token latency & energy: baseline / vanilla / EAGLE (τ=6) |
| `p4_eagle_speedup_vs_tau` | E | Speedup vs achieved acceptance length τ; paper τ/speedup bands |
| `p5_batch_breakeven` | G | TPOT growth + speedup decay vs batch (break-even) |
| `p5_load_aware_gamma` | G | TPOT-optimal γ\*(α, B): γ\* shrinks with batch |
| `p5_breakeven_hardware` | G | Break-even batch vs DRAM bandwidth and draft/target ratio |
| `p6_arch_named` | H | Throughput (and energy) across memory-layout configurations |
| `p6_dram_roofline` | H | Throughput roofline + speedup vs batch across DRAM bandwidths |

All figures are written as both PNG and PDF to `workspace/figures/`. Raw results are in
`workspace/results/*.csv`; regenerate with `experiments/make_figures.py`.
