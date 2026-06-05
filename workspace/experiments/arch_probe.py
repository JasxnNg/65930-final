"""Probe: at high batch (where KV traffic ~ B*ctx dominates), do KV-buffer SIZE and DRAM
bandwidth actually move speculative THROUGHPUT/TPOT? Decides the arch-experiment design."""
import specdec as sd

MB = 1024 * 1024 * 8
g, a = 4, 0.8


def tput(extra, B, ctx):
    return sd.spec_batch("hier", ctx, g, a, B, extra=tuple(sorted(extra.items())))["throughput"]


for (B, ctx) in [(16, 2048), (64, 4096), (256, 4096)]:
    print(f"\nB={B} ctx={ctx}  (KV ~ {B*ctx*2*96*128/8/1e6:.0f} MB total, 8-bit)")
    print("  KV buffer size sweep (DRAM=614GB/s):")
    for mb in [16, 64, 256, 1024]:
        t = tput({"KV_CACHE_BUFFER_SIZE": mb * MB}, B, ctx)
        print(f"    KV={mb:>4}MB -> throughput={t:8.1f} tok/s")
    print("  DRAM bandwidth sweep (KV buffer=128MB):")
    for bw in [100e9, 614e9, 3000e9, 8000e9]:
        t = tput({"DRAM_BW": bw}, B, ctx)
        print(f"    DRAM={bw/1e9:>5.0f}GB/s -> throughput={t:8.1f} tok/s")
