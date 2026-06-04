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
Fig. `p4_eagle_vs_vanilla`, `p4_eagle_speedup_vs_alpha`): a **single-layer draft** that
reuses the target's features (instead of a full draft model), and **tree drafting** in
which the target verifies a (depth×width) candidate tree in one pass. Capturing the
layer-count reduction requires depth-aware costs, so this experiment uses realistic
layer counts (draft 32, target 96) for all three methods; the cheap draft is one
target-width block and verification covers the whole tree
(`expected_tokens_tree(depth,width,α)` with per-level acceptance `1−(1−α)^width`).

**EAGLE-3 roughly doubles the speculative speedup.** At ctx=2048, α=0.8 the best vanilla
configuration (γ=8) reaches 3.3× over autoregressive decode, while the best EAGLE-3 tree
(depth 8, width 4) reaches **8.1×** — and it cuts per-token energy by a similar factor.
Two effects compound: the single-layer draft is far cheaper per drafted token than a
full small model, and the tree raises the accepted-token yield per verification, so the
expensive target pass is amortized over many more tokens. The advantage widens with α
(Fig. `p4_eagle_speedup_vs_alpha`), where wide trees almost always extend the accepted
prefix.

This makes the algorithmic verification dataflow the single largest lever we found,
consistent with Section D's conclusion that algorithm beats micro-architecture for
verification. *Caveat:* our tree-acceptance model treats candidates as independent and
omits EAGLE's feature-fusion/training effects, so the absolute 8× is an optimistic upper
bound; the qualitative ≈2× advantage over vanilla speculation is the robust takeaway.

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

5. **Algorithm beats architecture for verification.** EAGLE-3's single-layer draft plus
   tree verification roughly doubles the speedup over vanilla speculation (≈8× vs ≈3.3× at
   α=0.8). The biggest remaining lever is algorithmic, reinforcing that speculative
   decoding is best co-designed across algorithm, schedule, and hardware.

**Limitations / future work.** Costs are modeled per transformer block (depth-aware only
for the EAGLE comparison); the acceptance model is a scalar α (and an independence-based
tree model for EAGLE) rather than a learned, content-dependent distribution; batching,
PD-disaggregation, and SSD-offloaded KV are not yet modeled. Each is a natural extension
of the released `experiments/` harness.

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
| `p4_eagle_vs_vanilla` | E | Per-token latency & energy: baseline / vanilla / EAGLE |
| `p4_eagle_speedup_vs_alpha` | E | Speedup vs α: best vanilla vs best EAGLE tree |

All figures are written as both PNG and PDF to `workspace/figures/`. Raw results are in
`workspace/results/*.csv`; regenerate with `experiments/make_figures.py`.
