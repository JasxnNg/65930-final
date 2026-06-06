# Speculative-decoding experiments (conference expansion)

AccelForge mapping experiments extending *Modeling Speculative Decoding*. Everything runs
inside the `timeloopaccelergy/accelforge` container on Slurm (`gen_pop_rhel8`, `--qos=asic`)
via podman; the login node is not used for compute.

## Layout
- `specdec.py` — reusable cost model (corrected decode semantics: draft M=1, verify M=γ),
  model/arch registries, `core_rows`, sharded `run_sweep`, EAGLE tree model.
- `run_scheduler.py` — P1, optimal scheduler (ctx × γ × α on `arch_kv_buffer`).
- `run_generality.py` — P2, precision / model-scale / memory-hierarchy.
- `run_kv_dataflow.py`, `run_kv_longctx.py` — P3, KV organizations + verify dataflow (to 32k).
- `run_eagle.py` — EAGLE-3 (depth-aware: 1-layer draft + explicit full-binary tree
  layers with same-layer dedupe + tree verify; yield from a calibrated acceptance length
  τ, paper-grounded τ≈6).
- `run_batch.py` — batch sweep: TPOT/throughput, speculative break-even batch B*,
  load-aware lookahead γ*(α,B), and B* vs draft/target ratio and DRAM bandwidth.
- `run_arch_throughput.py` — memory-layout configs + DRAM-bandwidth roofline for throughput.
- `make_figures.py` — reads `results/*.csv`, writes `figures/*.{png,pdf}` + validations.
  Sections: scheduler, generality, kv_dataflow, kv_longctx, eagle, batch, arch_throughput.
- `slurm/run.sbatch`, `slurm/podman_run.sh` — Slurm array wrapper + container launcher.
- `smoke.py`, `sanity.py`, `batch_sanity.py`, `arch_probe.py`, `eagle_audit.py` —
  environment / model / cost-split sanity checks.

## Run
```bash
cd /home/mihika/65930-final/workspace
# each array index becomes SHARD_ID; array size becomes SHARD_COUNT (see specdec.get_shard)
sbatch --array=0-7  experiments/slurm/run.sbatch experiments/run_scheduler.py
sbatch --array=0-10 experiments/slurm/run.sbatch experiments/run_generality.py
sbatch --array=0-15 experiments/slurm/run.sbatch experiments/run_kv_dataflow.py
sbatch --array=0-3  experiments/slurm/run.sbatch experiments/run_kv_longctx.py
sbatch --array=0-3  experiments/slurm/run.sbatch experiments/run_eagle.py
sbatch --array=0-11 experiments/slurm/run.sbatch experiments/run_batch.py
sbatch --array=0-9  experiments/slurm/run.sbatch experiments/run_arch_throughput.py
# after completion, merge shards + generate figures:
srun -p gen_pop_rhel8 --qos=asic -c 2 --mem=8G --time=00:15:00 \
  bash experiments/slurm/podman_run.sh experiments/make_figures.py
```
Sharded outputs are `results/<name>.shard<i>.csv`; `make_figures.py` merges them into
`results/<name>.csv`. Generate figures only after a sweep's array has fully finished
(merging mid-run picks up partial checkpoints).

## Notes
- podman storage is node-local; `podman_run.sh` selects `/scratch/podman/$USER` or
  `/tmp/podman-$USER` and pulls the image on first use per node.
- The container interpreter is `python3`; `python` does not exist.
