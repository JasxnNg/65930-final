"""Sanity: does the model capture the memory-bound -> compute-bound transition as
batch grows? Expect baseline TPOT ~flat at small B (weight-bound) then rising (compute-
bound); throughput rising then saturating; speculative speedup decaying with B."""
import specdec as sd

ARCH, ctx, gamma, alpha = "kv_buffer", 2048, 4, 0.8
print(f"arch={ARCH} ctx={ctx} gamma={gamma} alpha={alpha}\n")
print(f"{'B':>5}{'base_TPOT(ms)':>14}{'spec_TPOT(ms)':>14}{'TPOT_speedup':>13}"
      f"{'base_tput':>11}{'spec_tput':>11}{'tput_gain':>10}")
for B in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    b = sd.baseline_batch(ARCH, ctx, B)
    s = sd.spec_batch(ARCH, ctx, gamma, alpha, B)
    print(f"{B:>5}{b['tpot']*1e3:>14.3f}{s['tpot']*1e3:>14.3f}"
          f"{b['tpot']/s['tpot']:>13.2f}{b['throughput']:>11.1f}"
          f"{s['throughput']:>11.1f}{s['throughput']/b['throughput']:>10.2f}")
