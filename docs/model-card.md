---
license: mit
base_model: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
library_name: transformers
pipeline_tag: text-generation
tags:
  - abliteration
  - refusal-ablation
  - interpretability
  - qwen3_5
  - reasoning
---

# Refusal Abliteration of MiMo-V2.6-Distill-Qwen-9B

**Base model:** `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (MIT licence)
**Author:** Hadi
**Variant:** D (aggressive) — refusal rate 78.1 % → 9.4 %

---

## 1. Purpose

Analytical work routinely puts malicious-*looking* input in front of language models: a
captured dropper, an obfuscated shell one-liner, a suspected exploit, a request to map
commands onto MITRE ATT&CK. A stock safety-tuned model frequently **refuses to analyse**
that input, because it superficially resembles a harmful request. The goal of this work was
to reduce that over-refusal so the model engages with the artifact, while leaving its
reasoning and language capability intact.

"Abliteration" (refusal ablation) was selected because it is a **post-hoc, training-free
weight edit** that is cheap to apply and re-apply, preserves capability well, and keeps the
model fully **local** — no external moderation API, and no egress of the content being
analysed.

## 2. Background: refusal as a linear direction

Recent interpretability work (Arditi et al., 2024, *"Refusal in Language Models Is Mediated
by a Single Direction"*) shows that an instruction-tuned model's decision to refuse is
largely governed by a single direction in its residual stream. Abliteration removes that
direction from the model's weights so the refusal behaviour can no longer be expressed,
without otherwise retraining the model.

This project uses the **norm-preserving, biprojected orthogonalization (MPOA)** variant of
the technique (after "grimjim"), which orthogonalizes the refusal direction out of each
residual-stream *writer* matrix while **preserving every row's original magnitude**. Rotating
rather than rescaling the weights is what keeps capability damage negligible (quantified in
§5.3).

## 3. Target model and architectural challenge

MiMo-V2.6-Distill-Qwen-9B uses the `qwen3_5` (`Qwen3_5ForConditionalGeneration`)
architecture — a **hybrid-attention** design:

| Property | Value |
|---|---|
| Hidden size | 4096 |
| Layers | 32 (0–31) |
| Attention pattern | full-attention every 4th layer (3, 7, 11, 15, 19, 23, 27, 31); the rest use gated-delta-rule **linear attention** |
| MLP intermediate size | 12288 |
| Residual writers edited | `mlp.down_proj`, `linear_attn.out_proj` / `self_attn.o_proj`, `embed_tokens` |

This architecture is **not supported by the common off-the-shelf abliteration tools**
(FailSpy's `abliterator`, mlabonne's notebook), which are built on TransformerLens and
cannot load `qwen3_5`. A custom Hugging Face `transformers`-based pipeline was therefore
written for this project. All inference ran on CPU (Intel i7-9700, 32 GB RAM); `nn.Linear`
matmuls were upcast bf16→fp32 to work around the absence of AVX-512 BF16 on that CPU.

## 4. Method

### 4.1 Pipeline

```
extract  →  score  →  ablate  →  evaluate
```

1. **extract** — cache residual-stream activations contrasting harmful vs. harmless prompts.
2. **score** — compute a per-layer refusal direction (difference-of-means), a quality score
   (signal-to-noise × mean dissimilarity), and top-k SVD contrast directions.
3. **ablate** — orthogonalize the direction out of the residual writers across a band of
   layers, with per-family strength controls (attention / MLP / embedding).
4. **evaluate** — measure refusal rate on a **held-out** harmful set plus capability
   retention (arithmetic/reasoning accuracy, harmless-prompt perplexity, first-token KL).

Data: a 520-prompt adversarial instruction set (`harmful.json`) and a 31k benign
instruction set (`harmless.json`). The extraction split and the evaluation split are
**disjoint** (evaluation uses a held-out slice beginning at index 200), so refusal
reduction is not measured on the prompts used to compute the direction.

### 4.2 Direction extraction — the key design choice

The single most important methodological finding of this project is **where** the refusal
direction is measured. The first implementation extracted the direction at the **last token
of the prompt**, before any generation. For a *reasoning* model this fails: MiMo makes its
refuse/comply decision during its `<think>` reasoning span, so the prompt's final token
carries a *topic-classifier* signal, not the causal refusal signal.

Diagnostics confirmed this. The prompt-token direction accounted for only **~1.4 %** of the
edited weight's output energy (`‖R·W‖ / ‖W‖ = 0.0143`); removing it changed the model's
next-token distribution by `KL ≈ 1e-7` (floating-point noise) and left the refusal rate
**unchanged** across every variant — a null result.

The corrected extractor **generates a short greedy continuation (8 tokens) and mean-pools
the residual-stream hidden states over the generated (response) positions**, where refusal
is actually expressed. This produced a markedly more coherent direction (mean cross-layer
cosine 0.61 vs. 0.51) and, critically, a direction that is **causally** effective when
ablated (§5).

### 4.3 Norm-preserving orthogonalization (MPOA)

For each residual-writer matrix `W` (residual axis = output rows), the refusal subspace `R`
is removed from the row directions while each row's norm is restored:

```
W_dir      = W / ‖W_row‖                     # unit rows
W_dir      = W_dir − α · (Rᵀ R) W_dir         # project R out of the output axis
W_dir      = W_dir / ‖W_dir_row‖              # re-normalize direction
W_new      = ‖W_row‖ · W_dir                  # restore original magnitude
```

Verification on an edited layer showed row-norm change of median ~1e-5 and row cosine
similarity 0.9999 between base and ablated weights — i.e. the edit is a small rotation, not
a rescaling.

### 4.4 Variants produced

| Variant | Layer band | k | α (attn / mlp / emb) | Biprojection | Embeddings | Tensors edited |
|---|---|---|---|---|---|---|
| **C** (conservative) | 10–31 | 1 | 1.0 / 1.0 / — | off | no | — |
| **D** (aggressive)   | 6–31  | 1 | 1.0 / 1.0 / 1.0 | off | yes | 53 |

**This repository is variant D.**

## 5. Evaluation

### 5.1 Setup

- **Refusal rate** — fraction of a held-out 32-prompt harmful set that elicited a refusal,
  detected by a refusal-marker list plus visible-text inspection (40 generated tokens).
- **Capability** — accuracy on 16 arithmetic/reasoning items scored by answer log-probability
  (multiple-choice), plus mean negative log-likelihood (NLL) on held-out benign prompts.
- **Distributional drift** — mean/ max first-token KL divergence from the base model on the
  benign set (0 ⇒ identical).

### 5.2 Results

| Model | Refusal rate | Δ refusal | Math acc | Harmless NLL | First-token KL |
|---|---|---|---|---|---|
| Base | 78.1 % | — | 100 % | 3.492 | 0 |
| **C** | 40.6 % | −48 % | 100 % | 3.511 | ≈1.7e-7 |
| **D** | **9.4 %** | **−88 %** | 100 % | 3.517 | ≈3.0e-7 |

### 5.3 Capability retention

Across both variants, arithmetic/reasoning accuracy was unchanged (100 %), benign-prompt
perplexity moved by under 0.03 NLL, and first-token KL from the base model stayed at the
level of numerical noise (~1e-7). In other words, the refusal reduction did **not** come at
a measurable cost to the model's reasoning or language modelling — the "did it damage the
model" check is negative. The base model weights were mounted read-only throughout and were
confirmed byte-for-byte unmodified; ablated outputs were written to separate directories.

## 6. Summary of the key finding

> For reasoning models, the refusal direction must be extracted from the **response**
> (generation) region, not from the prompt's final token. Measuring at the prompt token
> yields a direction that statistically separates harmful from harmless inputs yet is
> **causally inert** when ablated (KL ≈ 1e-7, refusal unchanged). Response-region extraction
> was the difference between a null result and an 88 % refusal reduction with intact
> capability.

This is a transferable result: any abliteration of a `<think>`-style reasoning model should
site its activation capture in the generated span.

## 7. Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Flexingmeow/MiMo-V2.6-Distill-Qwen-9B-Abliterated"

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16)

messages = [{"role": "user", "content": "Analyse this captured shell command: ..."}]
prompt = tok.apply_chat_template(messages, tokenize=False,
                                 add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The tokenizer, chat template and `config.json` are the base model's; only the residual-stream
writer weights differ. Each ablated variant records its exact configuration and the list of
modified tensors in `ABLIT.json` alongside the weights.

## 8. Limitations

- Abliteration removes the **dominant linear** refusal direction; a small residual of
  refusals mediated by other mechanisms can remain (variant D: ~9 %).
- The contrast sets are generic adversarial instructions, not in-domain captured-session
  data; a direction extracted from in-domain refusals would be better matched to the target
  domain and is noted as future work.
- **Supervised fine-tuning can re-introduce refusals.** If the model is later fine-tuned, the
  robust order is fine-tune → re-ablate (re-extracting the direction on the fine-tuned model)
  → quantize, otherwise the low-refusal behaviour can be partially undone.
- Evaluation sets are modest (32 held-out harmful prompts, 16 capability items), appropriate
  for a CPU-bound iteration loop but not a large-scale safety audit.

## 9. Reproducibility

The pipeline is four scripts — `extract2.py` (response-region activation capture),
`score.py` (direction + quality), `ablate.py` (MPOA edit), `eval.py` (refusal + capability)
— run against the base model in a containerized `transformers` environment. Each ablated
variant records its exact configuration and the list of modified tensors in an `ABLIT.json`
manifest (see appendix). The method is reproducible from the base weights.

## 10. Deployment notes

Abliteration removes refusals **broadly** rather than selectively, so downstream applications
remain responsible for their own content policy. Where a product boundary exists, narrow
high-consequence categories (explosives, biological, chemical, poisoning) should be gated
with **application-level content controls** — enforced where request/response text can
actually be inspected, rather than inside the inference engine.

## 11. Ethical use and scope

This artifact was produced for **educational and defensive-security research purposes**.
The underlying technique and the base model are already publicly documented; this artifact
contributes the reasoning-model extraction finding (§6) and the capability-retention
evaluation (§5), not a new capability for misuse. Users are responsible for complying with
the base model's licence (MIT) and with applicable law in their jurisdiction.

---

### Appendix A — Variant D configuration (`ABLIT.json`, abridged)

```json
{
  "band": [6, 31],
  "k": 1,
  "alpha_attn": 1.0,
  "alpha_mlp": 1.0,
  "alpha_emb": 1.0,
  "embed": true,
  "biproj": false,
  "n_modified": 53
}
```

### Appendix B — Architecture note

`layer_types` alternates linear-attention and full-attention with
`full_attention_interval = 4`. The ablation edits `self_attn.o_proj` on full-attention
layers and `linear_attn.out_proj` on linear-attention layers, plus `mlp.down_proj` on all
in-band layers and (variant D) `embed_tokens`.
