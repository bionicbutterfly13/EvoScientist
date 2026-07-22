# Stage 2 discrimination reference (Qwen3-1.7B Jacobian Lens)

> **NOT YET EXECUTED — NO KNOWN-GOOD RUN EXISTS.** This document describes what a
> passing Stage 2 discrimination run must contain. No such run has been
> performed. Execution is not authorized until Dr. Mani ratifies the open
> questions and the provisional preregistered thresholds below. Every threshold
> in the table is PROVISIONAL and binding only after ratification.

Use this dated reference when preparing, reviewing, or reproducing the Stage 2
observational-discrimination study. Re-resolve every remote resource at
execution time; this file is a design/provenance snapshot, not a floating source
of truth.

## Design notebook snapshot

- Prepared: 2026-07-22
- Notebook name: `jspace_colab_stage2_discrimination.ipynb`
- Canonical source identity: 52,257 bytes; SHA-256
  `c39df1d0087e091e908528f68a70358ca3c83e0e04cfdad4784f10d7656d9809`
- Structure: nbformat 4, 22 cells, 10 code cells
- Stage: 2 (observational discrimination) per `SKILL.md`, "Experiment Stage Gates"
- Prerequisite: Stage 1 (measurement reproduction) partially completed by the
  pinned Qwen3-1.7B smoke test on one Tesla T4
- Boundary: observation only; no fitting, steering, ablation, activation edit,
  Sakshi/Elume mutation, transfer, or external publication
- Optional artifact download is an explicit authorization gate
- The measurement cell refuses to run unless `THRESHOLDS_RATIFIED` is True

## Pinned identities (copied verbatim from the Stage 1 smoke test)

- Jacobian Lens repository: `https://github.com/anthropics/jacobian-lens.git`
- Jacobian Lens commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Model: `Qwen/Qwen3-1.7B`
- Model revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Expected model width: 2048
- Expected model layers: 28
- Lens repository: `neuronpedia/jacobian-lens`
- Lens revision: `a4114d7752d11eb546e6cf372213d7e75526d3a1`
- Lens file: `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`
- Lens size (successful Stage 1 run): 226,501,315 bytes
- Lens SHA-256 (successful Stage 1 run):
  `6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6`
- Lens fit metadata: 466 prompts recorded in the artifact
- Stage 1 anchor prompt SHA-256:
  `daeaa63881dc0f58be689307a81b1fbc347674424f1cae45819f82372804f5a6`

## Fixed measurement loci

- Layers: 6, 13, 20, 26 (the Stage 1 selection over the fitted source layers)
- Position: -2
- Top-k retained: 10; rank statistics computed over top-N = 50 (sparse summaries
  and scalar statistics only; full-vocabulary logits are never persisted)

## Capacity gate (carried forward unchanged)

- CUDA required
- Minimum total GPU VRAM: 14.0 GiB
- Abort before model/lens downloads if the gate fails
- Do not lower the threshold solely to force completion

## What a passing discrimination run must contain

A run is only interpretable if the kill criteria never fired: the capacity gate
passed, every pinned revision/size/SHA-256 resolved to the recorded identity,
and the Stage 1 anchor prompt (stimulus `s00`) reproduced its recorded top-k
token IDs with maximum Jacobian-logit difference 0.0.

Artifacts (all under schema `jspace-observation-discrimination/v1`):

- 50 per-prompt observation artifacts `jspace_observation_<sha256[:16]>.json`,
  each containing:
  - the pinned model/lens/instrumentation/runtime provenance blocks;
  - `input` with the prompt SHA-256, token/byte counts, and
    `raw_prompt_persisted=false`;
  - `stimulus` with `id`, `category`, and `stimulus_manifest_sha256`;
  - `measurement` with the six top-k readouts keyed by the four selected layers:
    `jacobian_lens`, `logit_lens_baseline`, `output_baseline`,
    `prompt_only_baseline`, `random_vector_baseline` (per seed), and
    `non_jspace_baseline` (shuffled-layer and mismatched-probe);
  - `measurement.repeatability_same_runtime` with same-runtime x2 repeat results;
  - a `discrimination` block with per-locus D, top-10 Jaccard, Spearman over
    top-50, output-argmax rank, and scalar logit-difference summaries;
  - `retention` with all three flags false.
- 1 aggregate run artifact `jspace_discrimination_<sha256[:16]>.json` containing
  the cross-prompt statistics, per-category medians, the stimulus-manifest
  digest, the preregistered thresholds as recorded before execution, the paired
  Wilcoxon result, and the pass/ambiguity/fail/kill decision.

Every artifact must pass
`validate_observation.py <file>` (the extended validator with
`jspace-observation-discrimination/v1` support). A passing per-prompt or
aggregate artifact re-hashes to its content-addressed filename prefix.

## PROVISIONAL preregistered thresholds

> Binding only after Dr. Mani ratifies them. No threshold may change after data
> collection begins.

| Criterion | Constant | Provisional value |
|---|---|---|
| Reproduction (kill anchor) | `STAGE1_RERUN_NOISE_MAX_ABS_LOGIT_DIFF` | 0.0 (anchor s00 top-k IDs identical) |
| Specificity: min prompt fraction | `SPECIFICITY_MIN_PROMPT_FRACTION` | 0.80 |
| Specificity: D margin over logit-lens | `SPECIFICITY_D_MARGIN` | 0.10 |
| Added information: max median top-10 Jaccard (Jacobian vs logit-lens) | `ADDED_INFO_MEDIAN_JACCARD_MAX` | 0.70 |
| Added information: paired Wilcoxon significance | `WILCOXON_ALPHA` | 0.01 |
| Same-runtime repeats | `SAME_RUNTIME_REPEATS` | 2 |
| Inference seeds (determinism) | `INFERENCE_SEEDS` | [0, 1] |
| Random-vector control seeds | `RANDOM_VECTOR_SEEDS` | [0, 1, 2] |

### Decision rule

- **Success:** reproduction holds AND specificity holds against both control
  families (random-vector and structure-broken) on at least 80% of prompts by
  the D margin AND added information holds (median top-10 Jaccard vs logit-lens
  ≤ 0.70 with paired Wilcoxon p < 0.01, and the Jacobian readout not within
  rerun noise of the output or prompt-only baselines).
- **Ambiguity:** conditions partially met; reported as a split result with the
  per-category table, no promotion of the readout.
- **Failure:** Jacobian within same-runtime rerun tolerance of the logit lens
  across every category and layer, OR indistinguishable from the random-vector
  control. Failure is a valid finding.
- **Kill:** capacity gate fails; OR any pinned revision/size/SHA-256 mismatches;
  OR the anchor prompt fails to reproduce. Stop and report the exact blocker.

## What this will not establish (even on success)

- cross-runtime or cross-hardware reproducibility (single T4 class only);
- calibrated probabilities or construct-level uncertainty;
- predictive or construct validity beyond these 50 synthetic prompts;
- causal mediation (Stage 3);
- functional global-workspace properties;
- subjective or phenomenal consciousness;
- readiness for Sakshi, Elume, or other observer integration.

## Open verification items before ratified execution

1. Confirm the jlens raw-vector transport primitive names at commit
   `581d398...`; the notebook's capability probe fails fast if the resolved names
   differ from the documented candidates. The random-vector and structure-broken
   baselines depend on this primitive.
2. Confirm the prompt-only baseline unembedding path (`hf_model.model.norm` +
   `hf_model.lm_head`) matches the logit-lens decode used inside `lens.apply`, so
   the six readouts share one vocabulary basis.
3. Verify batched/sequential application across 50 prompts stays within T4
   memory; if not, that is a measured capacity result, handled by sequential
   execution or a larger runtime, not by weakening the gate.
