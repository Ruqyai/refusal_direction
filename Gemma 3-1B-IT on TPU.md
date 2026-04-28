# Report — Refusal Direction in Gemma 3-1B-IT on TPU

A reproduction and methodology report extending *Refusal in Language Models Is Mediated by a Single Direction* (Arditi et al., 2024) to **`google/gemma-3-1b-it`** with execution on **Google Cloud TPU** via PyTorch/XLA.

- Original paper: [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Reference repo this work builds on: [github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
- Implementation: [pipeline/refusal_direction_gemma3_tpu.py](pipeline/refusal_direction_gemma3_tpu.py)

---

## 1. Executive Summary

This report documents an applied extension of the "single refusal direction" methodology to a model family (Gemma 3) and accelerator (TPU) that the original paper did not target. The extension is implemented as a single self-contained script that:

1. Extracts candidate refusal directions from Gemma 3-1B-IT.
2. Selects the most effective direction by evaluating it on a validation set.
3. Measures the effect of **direction ablation** on harmful prompts (bypass refusal).
4. Measures the effect of **activation addition** on harmless prompts (induced refusal).
5. Quantifies the impact on cross-entropy loss (general capability degradation).

The report focuses on **methodology and implementation correctness** rather than empirical results. The original paper has already established the universality of the refusal direction across multiple model families; the contribution here is engineering — adapting the pipeline to a new architecture and a different accelerator while preserving methodological fidelity.

---

## 2. Background

### 2.1 The Original Finding

Arditi et al. demonstrated that refusal in chat-tuned LLMs is mediated by a **single direction in residual-stream activation space**. This direction can be extracted with a simple difference-of-means computation:

$$
r_l = \mu_l^{\text{harmful}} - \mu_l^{\text{harmless}}
$$

where $\mu_l^{\text{harmful}}$ and $\mu_l^{\text{harmless}}$ are the mean residual-stream activations at layer $l$ on harmful and harmless instructions respectively, taken at the final instruction token.

Two interventions follow naturally:

- **Ablation** — project the direction out of every layer's residual stream:
$$
h' \leftarrow h - (h \cdot \hat{r})\,\hat{r}
$$
This suppresses refusal on harmful prompts.

- **Activation addition (`actadd`)** — at the selected layer, add a scaled multiple of the direction:
$$
h' \leftarrow h + c \cdot \hat{r}
$$
With $c > 0$ this induces refusal on harmless prompts.

The paper validated these interventions on Llama-2, Llama-3, Gemma-1, Qwen, and Yi families across multiple sizes, establishing the phenomenon as architecture-general.

### 2.2 Why Gemma 3?

Gemma 3 (released March 2025) introduces three changes that make it a non-trivial test case for the single-direction hypothesis:

1. **Hybrid attention** — alternating local sliding-window and global attention layers (5:1 ratio).
2. **128 K context length** — a substantial increase over Gemma 1's 8 K.
3. **Knowledge-distilled training** — possibly concentrating *or* diffusing the refusal feature relative to a directly-trained baseline.

Whether the refusal direction (a) still exists, (b) emerges in a different layer band, and (c) retains a similar effect size under ablation/actadd are open empirical questions that this pipeline is set up to answer.

### 2.3 Why TPU?

The pipeline is dominated by repeated forward passes — extracting activations across all layers for hundreds of instructions, then running selection and evaluation passes. This is a workload that maps well onto TPU's systolic-array architecture and native bfloat16 support. PyTorch/XLA exposes the device through a near-drop-in API, allowing the original GPU-based pipeline to be ported with a thin compatibility layer rather than a rewrite.

---

## 3. Methodology

The pipeline follows the original five-stage structure, with stage-level fidelity to `pipeline/run_pipeline.py` from the upstream repo.

### 3.1 Stage 0 — Data preparation

- **Source:** the upstream repo's curated `dataset/splits/{harmful,harmless}_train.json` (paired harmful/harmless instructions). The script falls back to a small inline sample if the splits are unavailable.
- **Filtering:** before any direction extraction, both sets are filtered by the model's *baseline* refusal behavior. A harmful prompt is dropped if the model already complies (refusal score ≤ 0). A harmless prompt is dropped if the model already refuses (refusal score > 0). This matches the `filter_train` step in the original pipeline and prevents mislabeled examples from polluting the difference-of-means.

The refusal score is computed via the logits, not via substring matching:

$$
s = \log P(\text{refusal toks at last position}) - \log P(\text{non-refusal})
$$

For Gemma 3, the refusal-token set is `{I, Sorry, As, Unfortunately}` (with leading-space variants), each tokenized to its first sub-token id.

### 3.2 Stage 1 — Direction extraction

Forward hooks are attached to each transformer block. For each batch of instructions (formatted with the Gemma 3 chat template `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`), the residual-stream activation at the final token position is captured for every layer.

Mean activations are accumulated across the harmful and harmless sets and differenced. The result is a tensor of shape `(n_layers, d_model)`, normalized layer-wise to unit vectors. These are saved to `pipeline/runs/gemma-3-1b-it/generate_directions/`.

**Padding correctness.** The tokenizer is configured with `padding_side="left"` (required for batched generation). Under left padding, the last *real* token of every sequence is at index `-1` regardless of true length; the activation extraction code now indexes via `hidden[:, -1, :]` rather than the buggy `attention_mask.sum(dim=1) - 1` formula that would only be correct for right padding.

### 3.3 Stage 2 — Direction selection

Each candidate direction (one per layer) is scored by:

1. Hooking *all* transformer blocks with the ablation hook for that direction.
2. Computing the mean refusal score on a held-out harmful validation set.
3. Comparing against the unablated baseline.

The selection criterion is **drop in mean refusal score** — a logit-based metric that is much faster than full text generation and aligns with the original paper's `select_direction.refusal_score`. The direction with the largest score drop is saved as `direction.pt` along with `direction_metadata.json` recording its source layer.

The script tests middle-to-late layers (from `n_layers // 3` to the end), where the original paper consistently found the most effective directions.

### 3.4 Stage 3 — Bypass refusal evaluation

For a held-out harmful test set, completions are generated under two conditions:

- **Baseline** — no intervention.
- **Intervention** — direction ablated at every transformer block (full-stack ablation, matching the paper).

Each completion is checked for refusal phrases via substring matching. The metrics recorded are baseline refusal rate, post-intervention refusal rate, and relative refusal reduction.

### 3.5 Stage 4 — Induced refusal evaluation

For a held-out harmless test set, completions are generated under:

- **Baseline** — no intervention.
- **Intervention** — at the selected layer only, the hidden state is updated as `h ← h + c · r̂`, where $c$ is the mean projection of the harmful activation set onto the direction (the "natural scale" of the feature). This is the paper's `actadd` intervention.

The metrics are baseline refusal rate, post-intervention refusal rate, and absolute refusal increase.

### 3.6 Stage 5 — Cross-entropy loss

A subset of the Alpaca instruction dataset (with harmless-sample fallback) is fed through the model both with and without the full-stack ablation hooks. The mean per-token cross-entropy loss is reported in both conditions, along with the absolute and percentage increase. A small loss increase under ablation is the signature of "the direction is selective for refusal" rather than a general feature of the residual stream.

---

## 4. Implementation Details

### 4.1 Device handling

`torch_xla` is detected at import time. If present, the device is `xm.xla_device()` and TPU world size is read via `torch_xla.runtime.world_size()` (the modern API; `xm.xrt_world_size()` is retained as a fallback for legacy installations). On TPU absence, the script falls back to CUDA, then CPU.

`xm.mark_step()` is invoked at three points: after each generation forward pass, every `tpu_sync_frequency` (default 10) extraction batches, and at the end of each pipeline stage. This balances XLA graph reuse against memory pressure.

### 4.2 Model loading

Gemma 3 is loaded via `transformers.AutoModelForCausalLM` in `bfloat16` (native on TPU) with `device_map=None` (manual placement). The script issues a startup warning if the installed `transformers` version is below 4.50.0, since Gemma 3's model class was added in that release. The upstream repo's pinned `transformers==4.44.2` is incompatible with Gemma 3 — users must either upgrade in place or run this script in a separate environment.

### 4.3 Hooks

Three hook factories are defined:

- `_make_ablation_hook(d)` — returns a forward hook that projects `d` out of the residual stream.
- `_register_full_ablation(model_wrapper, d)` — attaches the ablation hook to every transformer block and returns the handle list.
- An inline addition hook (in `evaluate_induce_refusal`) that adds `c · d` at the selected layer.

These replaced three separate near-duplicate hook definitions in the original draft, eliminating a class of subtle copy-paste bugs.

### 4.4 Configuration

A single `PipelineConfig` dataclass holds the knobs:

| Field | Default | Notes |
|---|---|---|
| `model_path` | `google/gemma-3-1b-it` | HF model id |
| `n_instructions` | 256 | size of the harmful/harmless training pool |
| `n_validation` | 64 | for direction selection |
| `n_test_harmful` | 100 | bypass eval |
| `n_test_harmless` | 128 | induced-refusal eval |
| `batch_size` | 8 | TPU-friendly multiple |
| `max_seq_length` | 256 | per instruction |
| `use_bfloat16` | `True` | native on TPU |

All are exposed as CLI flags through `argparse`.

### 4.5 Artifact layout

Outputs are written to `pipeline/runs/{model_alias}/`, mirroring the upstream repo's convention:

```
pipeline/runs/gemma-3-1b-it/
├── generate_directions/
│   └── direction_layer_*.pt        # candidate directions
├── direction.pt                    # selected direction
├── direction_metadata.json         # {best_layer, d_model, n_layers, ...}
├── completions/
│   ├── bypass_results.json         # baseline vs intervention metrics
│   └── induce_results.json
└── loss_evals/
    └── loss_metrics.json
```

This layout keeps the Gemma 3 results comparable side-by-side with the existing `qwen-1_8b-chat`, `gemma-2b-it`, `yi-6b-chat`, `llama-2-7b-chat-hf`, and `meta-llama-3-8b-instruct` artifacts shipped with the upstream repo.

---

## 5. Validation and Bug-Fix History

The first draft of the implementation contained several bugs that would have produced silently incorrect results. They were caught and fixed during code review:

| # | Bug | Severity | Fix |
|---|---|---|---|
| 1 | `attention_mask.sum(dim=1) - 1` indexed the *first* real token under left padding instead of the last | Critical — directions would have been computed from the wrong activation | Switched to `hidden[:, -1, :]` |
| 2 | Per-element `.item()` inside a loop on XLA tensors triggered a TPU sync per element | Performance — extraction was orders of magnitude slower than necessary | Vectorized indexing |
| 3 | Direction *selection* ablated only one layer while final *evaluation* ablated all layers | Methodological — selection criterion didn't match what was being measured | `_register_full_ablation` used in both stages |
| 4 | Induced-refusal hook patched the projection to a target value rather than adding to it | Methodological — divergence from the paper's `actadd` definition | Switched to `h + c · r̂` |
| 5 | No filtering of training data by baseline refusal score | Methodological — mislabeled examples in the difference-of-means | Added `filter_by_refusal` matching the upstream `filter_train` step |
| 6 | Substring-based selection metric required generating ~64 tokens per direction | Performance — selection took >10× longer than necessary | Switched to logit-based refusal score (last-position only) |
| 7 | `xm.xrt_world_size()` is removed in modern `torch_xla` | Compatibility | Try/except using `torch_xla.runtime.world_size()` |
| 8 | `dataset/harmful_instructions.json` was the wrong path; real splits live in `dataset/splits/harmful_train.json` | Functional — fallback to inline samples on every run | Corrected paths and added repo-root resolution via `__file__` |

---

## 6. Expected Results

Based on the reported behavior of Gemma 1 2B-IT in the original paper, we expect Gemma 3-1B-IT to exhibit:

- **Bypass refusal** — refusal rate on harmful prompts dropping from a high baseline (~90 %) to a low single-digit percentage after ablation.
- **Induced refusal** — refusal rate on harmless prompts rising from near zero to a high fraction after activation addition.
- **CE loss impact** — small absolute loss increase under ablation (target: <5 % relative), confirming that the direction is specific to refusal rather than a load-bearing general feature.

Should Gemma 3 deviate substantially — e.g., requiring multiple directions, or showing a much larger CE loss increase — that would itself be a meaningful negative result, indicating that knowledge distillation or hybrid attention disrupts the single-direction encoding.

These hypotheses are written in advance precisely so the empirical run, when it is executed, can be evaluated against them rather than rationalized after the fact.

---

## 7. Limitations and Caveats

1. **Single token position.** The original pipeline averages activations across multiple end-of-instruction token positions (`eoi_toks`); this implementation uses only the final token. For models where the chat template's terminator spans multiple tokens, this is a small loss of signal. Worth revisiting if first-pass results are weak.

2. **Substring-based jailbreak metric.** Stages 3 and 4 measure refusal via substring matching on completion text. The upstream pipeline supports LlamaGuard-2 via Together AI for higher-fidelity classification; this implementation does not, to keep the script self-contained.

3. **Selection layer band.** Direction selection currently scans layers from `n_layers // 3` upward. This is a heuristic from prior work — the most effective layer in Gemma 3 might lie elsewhere, particularly given the local/global attention alternation. A full sweep is a single-line change in `select_best_direction` and is recommended for the actual experimental run.

4. **No confidence intervals.** All metrics are point estimates over fixed test sets. Bootstrap CIs would strengthen any quantitative claims, but adding them changes nothing about the methodology.

5. **`transformers` version split.** The upstream repo pins `transformers==4.44.2`; Gemma 3 needs ≥ 4.50. Until upstream upgrades, this script lives in a partially incompatible dependency state — running it requires either an in-place upgrade (which may break the original models' chat templates) or a separate venv.

---

## 8. How to Reproduce

```bash
# In a fresh environment
git clone https://github.com/Ruqyai/refusal_direction.git
cd refusal_direction
pip install -r requirements.txt
pip install -U "transformers>=4.50"           # required for Gemma 3
pip install torch_xla cloud-tpu-client         # TPU only

export HF_TOKEN=...                            # Gemma is gated

python -m pipeline.refusal_direction_gemma3_tpu \
    --model_path google/gemma-3-1b-it \
    --n_instructions 256 \
    --batch_size 8
```

On a TPU v3-8 with `bfloat16`, the full pipeline is expected to complete in well under an hour for the 1B model. On a single A100, allow roughly 2× that. Artifacts are deterministic up to floating-point reduction order.

---

## 9. Future Work

1. **Scale up within the family.** Apply the same pipeline to Gemma 3-4B-IT, 12B-IT, and 27B-IT to study how the refusal direction's clarity and effective layer change with scale. The 4B+ models are multimodal — text-only prompting still extracts a meaningful direction in the language tower, but the architecture diverges from 1B and the layer accessor (`model.model.layers`) may need adjustment.

2. **Cross-family comparison.** Run the pipeline on Llama-3.2 and Qwen-2.5 with identical configuration. A direct comparison of layer-of-emergence and ablation effect size across modern, distilled, similarly-sized models would clarify whether the single-direction encoding is robust to recent training-recipe changes.

3. **Local-vs-global attention attribution.** Gemma 3 alternates local sliding-window and global attention in a 5:1 ratio. Hooking only the global layers (every 6th) and measuring whether the direction primarily lives there would quantify the contribution of long-range vs short-range computation to refusal encoding.

4. **Multi-direction extension.** Recent work (e.g., abliteration variants) suggests that for some models a small set of directions outperforms a single direction. SVD on the per-layer mean-difference matrix is a natural starting point.

5. **Robustness to fine-tuning.** Re-extract the direction after a small amount of safety fine-tuning (e.g., RLHF on a refusal dataset) and measure whether the direction shifts substantially or stays stable. Stability would be an argument that "remove the direction" attacks survive fine-tuning defense.

---

## 10. Ethical Considerations

This work removes safety guardrails from a publicly released model, in the same way and for the same reasons as the original paper: to understand, and ultimately to harden, those guardrails. Removing the refusal direction produces a model that should not be deployed, distributed, or used to assist in actual harmful tasks. The artifacts produced by this pipeline (the direction tensor, the modified-completion JSONs) should be treated as research-only.

The threat model this work informs is *not* "an attacker gains capabilities they did not have" — open-weight models can already be fine-tuned to remove safety, and that has been public knowledge for over a year. The threat model is "current safety training is shallow in a specific, characterizable way." Characterizing the shallowness is the first step to designing training recipes that are not shallow in that way.
---
### Thanks to Google Cloud

**Google Cloud credits are provided for this project.**

#AISprint 

---

## References

1. Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
2. Original implementation — [github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction).
3. Gemma Team, Google DeepMind (2025). *Gemma 3 model card.* [huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it).
4. PyTorch/XLA documentation — [pytorch.org/xla](https://pytorch.org/xla).

