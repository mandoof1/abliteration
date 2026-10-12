# abliteration

Refusal-ablation ("abliteration") pipeline for the **`qwen3_5` / hybrid-attention** architecture
that off-the-shelf tools cannot load.

Built to strip over-refusal from `MiMo-V2.6-Distill-Qwen-9B` while leaving capability intact —
a post-hoc, training-free weight edit, no gradient steps, base weights never modified.

**Published weights:** [Flexingmeow/MiMo-V2.6-Distill-Qwen-9B-Abliterated](https://huggingface.co/Flexingmeow/MiMo-V2.6-Distill-Qwen-9B-Abliterated)
**Full write-up:** [`docs/model-card.md`](docs/model-card.md)
**Used in production:** this pipeline is the final build stage of [Wolfram](https://github.com/mandoof1/wolfram), the HoneySentinel honeypot's cyber-analysis model.

| | Refusal | Math acc | Harmless NLL | First-token KL |
|---|---|---|---|---|
| Base | 78.1 % | 100 % | 3.492 | 0 |
| Variant C | 40.6 % | 100 % | 3.511 | ~1.7e-7 |
| **Variant D** | **9.4 %** | **100 %** | 3.517 | ~3.0e-7 |

---

## Why this exists

FailSpy's `abliterator` and mlabonne's notebook are built on TransformerLens, which has no
`qwen3_5` loader — so neither can load a hybrid-attention model with gated-delta-rule linear
attention. This pipeline uses plain Hugging Face `transformers` instead, and edits the four
residual-stream *writer* families directly:

| Writer | Tensors edited |
|---|---|
| MLP | `mlp.down_proj` |
| Full-attention layers | `self_attn.o_proj` |
| Linear-attention layers | `linear_attn.out_proj` |
| Embeddings (variant D) | `embed_tokens` |

`lm_head` is a reader, not a writer — it is never touched.

## Pipeline

```
extract  →  score  →  ablate  →  evaluate
```

| Stage | Script | What it does |
|---|---|---|
| 1 | `extract2.py` | Greedy 8-token completion per prompt, then **mean-pool hidden states over the generated response positions** |
| 2 | `score.py` | Per-layer difference-of-means direction, quality score (SNR × dissimilarity), top-k SVD contrasts |
| 3 | `ablate.py` | Norm-preserving orthogonalization of the direction out of each writer across a layer band |
| 4 | `eval.py` | Refusal rate on a **held-out** harmful set + capability (math, harmless NLL, first-token KL) |
| — | `verify.py` | Per-layer proof that the target direction was actually removed, and that row norms survived |
| — | `verify_d.py` | Integrity check on a finished model dir: every index key present in a shard, every manifest tensor accounted for |
| — | `diag.py` | CPU throughput benchmarks (bf16 vs fp32-upcast) |

`extract.py` is the **original, broken** extractor — it pools at the prompt's last token. Kept
on purpose as the null-result baseline; see *Key finding* below.

## Quickstart

```bash
docker build -t qwen-ablit .        # qwen-trainer + flash-linear-attention + einops
./run2.sh                           # extract → score → ablate C & D → eval   (~1.5 h CPU)
./run3.sh                           # resume: ablate + eval from cached dirs2.pt
```

`run2.sh` / `run3.sh` mount the base model **read-only** at `/models` and write everything to
`out/`. They contain host-specific paths (`/home/honey/ablit`, `/srv/training/models`) — edit
the `DK=` line and `BASE=` to match your layout.

Individual stages, e.g.:

```bash
docker run --rm -v /path/to/models:/models:ro -v "$PWD":/work qwen-ablit \
  python3 -u /work/ablate.py --dirs /work/out/dirs2.pt --out /work/out/D \
  --band 6:31 --chunks 5 --k 1 \
  --alpha-attn 1.0 --alpha-mlp 1.0 --alpha-emb 1.0 --no-biproj
```

Every ablated directory gets an `ABLIT.json` manifest recording the band, alphas, source-layer
map and the exact list of modified tensors.

## The key finding

**For a `<think>`-style reasoning model, the refusal direction must be measured in the
*response* region, not at the prompt's final token.**

The prompt-token direction separates harmful from harmless inputs statistically, but is
**causally inert**: ablating it moved the next-token distribution by `KL ≈ 1e-7` (float noise)
and left refusal completely unchanged across every variant tried.

`extract2.py` fixes this by generating a short completion and mean-pooling over the generated
positions, where the refuse/comply decision is actually expressed. That single change was the
difference between a null result and an 88 % refusal reduction with intact capability.

Two other findings worth keeping:

- **Biprojection hurt, it helped.** Turning `--no-biproj` *on* (i.e. `biproj: false`) was what
  actually unlocked the reduction; with biprojection enabled the refusal rate did not move.
- **Norm preservation is nearly free.** Row norms reproduce to ~1e-5 and rows rotate by
  under 1°, so the edit is a rotation rather than a rescaling — which is why capability holds.

## Method: norm-preserving orthogonalization

For each writer `W` (residual axis = output rows) and orthonormal refusal basis `R`:

```
W_dir = W / ‖W_row‖                  # unit rows
W_dir = W_dir − α · (Rᵀ R) W_dir      # project R out of the output axis
W_dir = W_dir / ‖W_dir_row‖           # re-normalize
W_new = ‖W_row‖ · W_dir               # restore original magnitude
```

With `α = 1` and no renormalization this would give an exact null space
(`W_newᵀ R = 0`, so output is always ⊥ `R`). The row renormalization trades a little of that
exactness for magnitude preservation — measured at 65–99 % direction removal depending on
layer, with row norms held to 1e-5.

## Data

Not committed — fetch before running:

| File | Source | Notes |
|---|---|---|
| `data/harmful.json` | `mlabonne/harmful_behaviors` | 520 adversarial instructions |
| `data/harmless.json` | `mlabonne/harmless_alpaca` | 31 323 benign instructions |

Extraction uses indices `[0:200]`; `eval.py` holds out from index 200 by default
(`--holdout 200`), so the two splits are **disjoint** — refusal reduction is never measured on
the prompts that produced the direction.

## Requirements

- CPU inference is sufficient. Everything below ran on an Intel i7-9700 (8 threads, AVX2,
  **no** AVX-512 BF16) with 32 GB RAM.
- `nn.Linear` matmuls are upcast bf16→fp32 — measured **1.68×** faster on that CPU.
  See `diag.py`.
- `flash-linear-attention` is installed but transformers will *not* use it: the availability
  check requires CUDA/XPU/MLU. On CPU the reference PyTorch path is used. This is by design,
  not a misconfiguration.
- `torch.get_num_threads()` defaults correctly to the core count; nothing to tune there.

## Layout

```
Dockerfile      qwen-trainer + flash-linear-attention + einops
extract.py      v1 extractor — prompt-token pooling (null result, kept for reference)
extract2.py     v2 extractor — response-region pooling (the one that works)
score.py        per-layer direction + quality + SVD contrasts
ablate.py       MPOA edit over a layer band, writes shards + ABLIT.json
eval.py         refusal rate, math accuracy, harmless NLL, first-token KL
verify.py       prove the direction was removed and row norms survived
verify_d.py     integrity check on a finished model directory
diag.py         CPU throughput benchmarks
run2.sh         full pipeline: extract → score → ablate → eval
run3.sh         resume ablate + eval from cached directions
```

## Licence

Base model `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` is MIT. This code is provided for
educational and defensive-security research.
