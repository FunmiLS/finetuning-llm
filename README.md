# TPU 2026

Shared code and experiment records for the GRPO finetuning practical on
`google/gemma-3-1b-it` with GSM8K.

## Team Members
Barbara Koch, Rowan D'Auria

## Main Findings (Part 1.3: Improving on the baseline)

Full write-up in [report/report.pdf](report/report.pdf), §1.3 and Appendix A.3.

**The baseline collapses; group size is why.** The default `NUM_GENERATIONS = 2`
run degenerates to 0.08% exact numeric match on the full 1319-question GSM8K test
split, against 47.38% for the un-finetuned `gemma-3-1b-it`. At K = 2 the normalised
advantage keeps only the *sign* of the reward gap (the group std is proportional to
|r1 - r2|, so the magnitude cancels), leaving one effective sample per group.

**Raising K to 8 fixes it.** K = 8 gives a graded advantage and 7 effective samples
per group, and it trains stably: KL and response length stay flat where K = 2
diverges. A sweep over K ∈ {2, 4, 8, 16} shows K = 2 is the only unstable setting.

**Reward reweighting helps, but not significantly.** Replacing the discontinuous
`check_answer` ladder with a smooth exponential in relative error, plus gating the
near-saturated format terms down to a 1.0 entry fee, adds +2.50 pp (p = 0.064).

| Model | Exact numeric match (n=1319) | Δ vs. base |
|---|---|---|
| Base `gemma-3-1b-it` | 47.38 [44.81, 50.11] | — |
| Baseline GRPO (K=2) | 0.08 [0.00, 0.23] | −47.31 *** |
| K=8 | 56.03 [53.45, 58.76] | +8.64 *** |
| K=8 + reweighted reward | **58.53 [55.88, 61.18]** | +11.14 *** |

Greedy decoding, seed 42, 5864 steps, lr 3e-6, β = 0.08. Brackets are 95% bootstrap
CIs (paired for Δ). *** p < 0.001, McNemar with Holm correction.

**Caveats.** One seed per run, so the CIs cover test-set sampling only, not training
variability. Comparisons are per-step rather than compute-matched, so some of the
K = 8 gain may come from its greater rollout throughput.

**What didn't work.** Training only on the hard+medium GSM8K difficulty buckets
scored *below* the K = 8 baseline — sparser correctness rewards make more groups
degenerate (all-correct or all-incorrect), which contributes no learning signal.
At K = 2, β = 1e-6 avoids the late collapse (51.56% on the 64-question suite) while
β = 0.32 collapses outright. See Appendix A.3 for the full trial tables.

## Repository Map

| Path | Purpose |
|---|---|
| `scripts/` | Training, evaluation, reward, model, data, config, and chat scripts. |
| `runs/` | Per-run metadata, logs, config snapshots, and run notes. |
| `analysis/` | Scripts for exporting metrics, plotting, and uncertainty estimates. |
| `report_assets/` | Final plots, tables, and small artefacts used in the report. |
| `docs/` | Shared planning, experiment tracking, setup notes, and patch history. |
| `GROUP_PROJECT_PLAN.md` | Operational plan for the group project. |
| `tpu-setup.md` | TPU environment and setup instructions. |
| `tunix.ipynb` | Original notebook reference. |

## Core Docs

- [Run log](docs/RUNS.md)
- [Experiment plan](docs/EXPERIMENTS.md)
- [Setup notes](docs/SETUP_NOTES.md)
- [Baseline patches](docs/BASELINE_PATCHES.md)
- [Project plan](GROUP_PROJECT_PLAN.md)
- [TPU setup](tpu-setup.md)
- [Scripts guide](scripts/README.md)

## Access the TPU

```bash
gcloud auth login

export TEAM=dakolo

gcloud alpha compute tpus tpu-vm ssh $TEAM \
  --zone=us-east5-a --project=tpu-2026 --tunnel-through-iap

cd tpu-2026
```
## Author

**Funmi Looi-Somoye**

ol306@cam.ac.uk

University of Cambridge

## License

The MIT license described in **LICENSE** applies to the code in this repository.

Original code sourced from Dr Boris Bolliet