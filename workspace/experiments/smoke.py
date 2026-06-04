"""Smoke test: confirm AccelForge runs in-container and time one draft + one verify map.

Mirrors the proven milestone3_runs.ipynb usage so we validate the API and measure
per-map wall-time before launching any large sweep.
"""
import os
import time
from pathlib import Path

import accelforge as af

af.set_n_parallel_jobs(int(os.environ.get("AF_JOBS", os.cpu_count())), print_message=True)

ARCH = Path("designs/arch.yaml")
DRAFT = Path("designs/gpt3_6.7B_kv_cache.yaml")
TARGET = Path("designs/gpt3_175b_kv_cache.yaml")


def draft_result(tokens):
    spec = af.Spec.from_yaml(
        ARCH, DRAFT,
        jinja_parse_data={"BATCH_SIZE": 1, "N_TOKENS": tokens, "N_NEW_TOKENS": 1},
    )
    # keep everything in the first memory level (MainMemory) like milestone3
    for node in spec.arch.nodes:
        if isinstance(node, af.arch.Memory):
            node.tensors.keep = "All"
            break
    spec.mapper.metrics = af.mapper.Metrics.LATENCY
    return spec.map_workload_to_arch(print_progress=False)


def verify_result(tokens, lookahead):
    spec = af.Spec.from_yaml(
        ARCH, TARGET,
        jinja_parse_data={"BATCH_SIZE": 1, "N_TOKENS": tokens + lookahead, "N_NEW_TOKENS": lookahead},
    )
    for node in spec.arch.nodes:
        if isinstance(node, af.arch.Memory) and node.name == "MainMemory":
            node.tensors.keep = "All"
            break
    spec.mapper.metrics = af.mapper.Metrics.LATENCY
    return spec.map_workload_to_arch(print_progress=False)


if __name__ == "__main__":
    t0 = time.time()
    d = draft_result(1024)
    t1 = time.time()
    print(f"[draft]  ctx=1024  energy={float(d.energy()):.6e}  latency={float(d.latency()):.6e}  ({t1-t0:.1f}s)")

    v = verify_result(1024, 4)
    t2 = time.time()
    print(f"[verify] ctx=1024 lookahead=4  energy={float(v.energy()):.6e}  latency={float(v.latency()):.6e}  ({t2-t1:.1f}s)")
    print(f"[total] {t2-t0:.1f}s")
