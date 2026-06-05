"""Audit the EAGLE-3 modeling critiques against the cost breakdown.

For a representative tree (depth=8, width=4) at ctx=2048, alpha=0.8 on kv_buffer, we
split the round cost into the DRAFT term and the VERIFY term, then re-evaluate the draft
term under three modeling choices to see whether the critiques move the result:
  (cur)  draft M=1 per pass,  M_FULL=ctx          (our current model)
  (#1)   draft M=width per pass, M_FULL=ctx        (critique 1: tree frontier width)
  (#3)   draft M=width per pass, M_FULL=tree depth (critique 3: tiny draft KV -- claimed)
The verify term (full target stack over the tree) is identical in all three.
"""
import specdec as sd

ARCH = "kv_buffer"
ctx, depth, width, alpha = 2048, 8, 4, 0.8
LT = sd.MODELS[sd.TARGET_MODEL]["n_layers"]
tree_nodes = depth * width
y = sd.expected_tokens_tree(depth, width, alpha)

# baseline (full target autoregressive decode, per token)
be, bl = sd.baseline_step(ARCH, ctx)
bl *= LT


def draft_term(m, mfull_is_ctx):
    """depth single-layer (target-width) draft passes."""
    e = l = 0.0
    for i in range(depth):
        mfull = (ctx + i) if mfull_is_ctx else depth
        de, dl = sd._map(ARCH, sd.TARGET_MODEL, mfull, m, 1, 8, False, ())
        e += de
        l += dl
    return e, l


ve, vl = sd.verify_step(ARCH, ctx, tree_nodes)
ve, vl = ve * LT, vl * LT  # full target stack verifies the tree

print(f"ctx={ctx} tree(d={depth},w={width}) tree_nodes={tree_nodes} "
      f"accepted/round={y:.2f}  baseline L/tok={bl*1e3:.2f}ms")
print(f"verify term: L={vl*1e3:.2f}ms (full {LT}-layer target over {tree_nodes} tokens)\n")
print(f"{'model':<34}{'draft_L(ms)':>12}{'draft %round':>13}{'per-tok(ms)':>13}{'speedup':>9}")
for name, m, mfull_ctx in [("cur: M=1, M_FULL=ctx", 1, True),
                           ("#1 : M=width, M_FULL=ctx", width, True),
                           ("#3 : M=width, M_FULL=depth", width, False)]:
    de, dl = draft_term(m, mfull_ctx)
    rl = dl + vl
    print(f"{name:<34}{dl*1e3:>12.3f}{100*dl/rl:>12.1f}%{rl/y*1e3:>13.3f}{bl/(rl/y):>8.2f}x")
