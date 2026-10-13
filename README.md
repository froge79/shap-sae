# shap-sae

Tests whether SHAP attributions over sparse-autoencoder (SAE) features of GPT-2 Small predict what actually happens when those features are intervened on.

## Problem

Attribution methods (SHAP, Integrated Gradients, gradient×activation) rank SAE features by how much they explain a sentiment probe's output. Whether that ranking reflects the model's *causal* behaviour is rarely checked. This project measures it: if a feature is ranked important, does steering or ablating it move GPT-2's output by a correspondingly large amount, in the predicted direction?

## How it works

1. **Extract.** Run SST-2 through GPT-2 Small and encode the layer-7 residual stream with the `gpt2-small-resid-post-v5-32k` SAE (32,768 features, last-token activations). Data is split three ways: probe training, validation, and a held-out split used for all attribution and causal evaluation.
2. **Probe.** Fit an L1 logistic probe (linear arm) and a small MLP probe (MLP arm) on the SAE features, then select a 75-feature mask.
3. **Attribute.** Compute global and per-example attributions with exact LinearSHAP / DeepSHAP, Integrated Gradients, gradient×activation, and raw probe weights. All methods attribute the same target within an arm (margin for linear, pre-sigmoid logit for MLP).
4. **Intervene.** Steer or ablate individual features inside GPT-2 and measure the change in the sentiment logit difference (` wonderful` vs ` awful`).
5. **Compare.** Rank-correlate attribution scores with measured causal effects, with multiple-comparison control, multi-seed replication, and a known-ground-truth minimal-pairs test of the harness itself. Multi-feature group steering is evaluated separately.

For a linear probe, IG, gradient×activation and interventional SHAP are analytically identical, so the linear arm serves as a sanity check; the MLP arm is where the methods genuinely differ.

## Scope and limitations

- **One model, layer, SAE and task:** GPT-2 Small, layer 7, one pretrained SAE, binary SST-2 sentiment. Results may not transfer.
- **Attribution and intervention differ:** attributions are computed at the last token; interventions edit every position.
- **Mismatched readouts:** the steering strength α is calibrated against a layer-11 residual probe but evaluated on the GPT-2 logit difference, so it is a rough "measurable but unsaturated" magnitude, not a tuned hyperparameter.
- **Small samples:** per-run correlations use ~20 features (116 in the full-pool run), so p-values are descriptive unless FDR-corrected.
- **Sentiment is largely lexical on SST-2**, which limits how much feature structure there is to explain (see `10_simplicity_check`).
- **Not comparable across arms:** effect magnitudes differ between the linear and MLP arms and between per-feature and group steering; compare rankings and curve shapes only.
- **Research code, not a library.** Results in `run2/` come from a single full pipeline run; the pipeline has not been independently reproduced.

## Running it

Requires Python ≥ 3.14, [uv](https://docs.astral.sh/uv/), and a GPU (≈23 GB) for the model-dependent steps; peak RAM is ~18 GB.

```bash
uv sync
export SHAP_SAE_OUTPUTS=run2/outputs SHAP_SAE_CHECKPOINTS=run2/checkpoints
uv run python src/core/1_extract.py
```

The full step-by-step pipeline (about 7 hours end to end), expected outputs and sanity checks are in [`CLAUDE.md`](CLAUDE.md); design conventions are in [`AGENTS.md`](AGENTS.md).

## Layout

```
src/core/           activation extraction, probe, feature mask
src/global_linear/  logistic-probe arm: SHAP, rankings, steering, seed sweeps
src/global_mlp/     MLP-probe arm (mirrors the above)
src/local/          per-example vs global attribution faithfulness
src/global_group/   multi-feature steering vectors and plots
src/groundtruth/    minimal-pair test with known ground truth
run2/outputs/       results of the current pipeline run
```
