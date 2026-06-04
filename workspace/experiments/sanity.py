"""Sanity checks for the CORRECTED decode model (draft M=1, verify M=gamma).

We validate qualitative behavior rather than matching the old (buggy full-prefill)
numbers:
  1. draft/verify/baseline latency grows with context (KV reads grow).
  2. verify(gamma) costs more than a single decode step.
  3. speculative per-token latency beats baseline at large context for moderate gamma.
"""
import specdec as sd

ARCH = "kv_buffer"


def show(tag, ctx):
    de, dl = sd.draft_step(ARCH, ctx)
    be, bl = sd.baseline_step(ARCH, ctx)
    ve, vl = sd.verify_step(ARCH, ctx, 4)
    print(f"{tag} ctx={ctx:5d} | draft L={dl:.3e} E={de:.3e} | "
          f"baseline L={bl:.3e} E={be:.3e} | verify(4) L={vl:.3e} E={ve:.3e}")
    return dl, bl, vl


print("== per-step costs vs context ==")
d512, b512, v512 = show("", 512)
d4096, b4096, v4096 = show("", 4096)

print("\n== spec vs baseline per-token (gamma=4, alpha=0.8) ==")
for ctx in (512, 1024, 2048, 4096):
    be, bl = sd.baseline_per_token(ARCH, ctx, 4)
    se, sl = sd.spec_per_token(ARCH, ctx, 4, 0.8)
    print(f"ctx={ctx:5d} | baseline L={bl:.3e} | spec L={sl:.3e} | "
          f"speedup={bl/sl:.2f}x | energy_ratio={se/be:.2f}")

checks = {
    "draft grows with ctx": d4096 > d512,
    "verify > draft": v512 > d512,
    "spec beats baseline @ctx4096": (lambda: sd.baseline_per_token(ARCH, 4096, 4)[1]
                                     > sd.spec_per_token(ARCH, 4096, 4, 0.8)[1])(),
}
print("\n== checks ==")
for k, v in checks.items():
    print(f"  [{'OK' if v else 'FAIL'}] {k}")
raise SystemExit(0 if all(checks.values()) else 1)
