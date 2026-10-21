# Hugging Face support

First-class **directory** load for **decoder-only** LLMs: `config.json` + tokenizer files + `*.safetensors` (single file or Hub shards).

Related: [gguf_support.md](gguf_support.md) · [huggingface_decoder_completeness.md](huggingface_decoder_completeness.md) (how remaining Hub gaps close)

## Laptop / CPU memory dial (kimi-style)

`uaii generate` and `uaii chat` default to **`--preset laptop`**:

| Preset | Resident target | Pin | Expert LRU | KV window | Streaming |
|---|---|---|---|---|---|
| `laptop` (default) | ~8 GiB | auto-shrunk trunk (layers → lm_head → embed) + always-pin norm | leftover after trunk (cap 1 GiB) | 2048 tokens (+ 4 sink) | on when weights exceed budget |
| `desktop` | ~32 GiB | 16 layers | 4 GiB | 4096 | on under pressure |
| `server` | ~128 GiB | dense if it fits | 32 GiB | 8192 | on under pressure |
| `phone` | ~2 GiB, then half of OS-available RAM | norm only, unless embed / lm_head still fit | 256 MiB, then cap/8 | 512 (256 under 1.5 GiB, 128 under 1 GiB) | on, one staging buffer, 4 threads |
| `none` | unlimited | — | — | grow with `--max-context` | off |

Weights stay on disk (`file#tensor` refs). The session **pins a trunk that fits the budget** (final norm always; embeddings, lm_head, and the first N layers while they still fit) and **pages the rest** through a double buffer of **packed** bytes. Under `--preset laptop`, if embed or `lm_head` would blow the 8 GiB floor, they are unpinned and streamed like any other tensor — the same dial kimi-k3-in-c uses. Routed MoE expert tensors are **never fully pinned**; an LRU keeps recently used experts in **leftover** RAM after the trunk, never by shrinking the trunk to make room. KV is a **sliding window** (default 2048 on laptop, plus 4 attention-sink tokens) so long chats cannot OOM. GGUF Mixtral/Qwen banks are split per expert (`#e=N`) and only the **top-k** experts of that token are staged. GGUF block-quant weights (Q4_K, Q5_K, …) stay packed in staging and in the expert LRU — they are **not** widened to f32 on the way in. Peak RSS is reported on generate.

This is the same idea as kimi-k3-in-c: **storage is a scheduler resource**. Supported families (Llama-like, Qwen2/2.5/3 + Qwen2-MoE, Mistral/Mixtral, Gemma 1/2, GPT-2, Kimi/DeepSeek MLA **GGUF**) can run on an 8 GB laptop when the checkpoint is GGUF-quantized or the largest dense tensor is smaller than about half the budget. It will be **slow** (disk + CPU) rather than OOM. Token latency is dominated by paging, not by “the model is small so it should be instant.” Override KV with `--kv-window`. A single **unquantized** FFN/`lm_head` larger than ~half of RAM still fail-closes (`Convert to GGUF Q4_K or use --preset desktop|server`). Hugging Face BF16 folders of 70B-class models need a GGUF conversion for the 8 GB path; we do not silently dequant a 4 GiB tensor into an 8 GiB box.

`phone` is that same dial for Android and iOS. The budget starts at 2 GiB and is clamped to half of what the OS can still give the process, so `--ram-gb 8` on a phone with 6 GiB free becomes 3 GiB. Staging is a single buffer (laptop and above keep the prefetch buffer). See [mobile.md](mobile.md).

Disable with `--preset none` if the whole checkpoint already fits in RAM and you want the fastest load.

## Directory layout

A model directory is accepted when it contains `config.json` (see `HfLoader::accepts`).

| File / pattern | Role |
|---|---|
| `config.json` | **Required.** Architecture + dim fields (fail closed). |
| `*.safetensors` | Weights. Single `model.safetensors` or shards. |
| `model.safetensors.index.json` | Optional Hub shard map. Any `*.safetensors.index.json` is also accepted. |
| `tokenizer.json` / `tokenizer.model` | Optional for tokenize. |
| `tokenizer_config.json` | Optional; `chat_template` when present. |

## Architectures

**Product target** is the six mainstream families in [huggingface_decoder_completeness.md](huggingface_decoder_completeness.md) §1.1: Llama-like, Qwen, Mistral/Mixtral, Kimi (GGUF MLA first), OpenAI open-weight (GPT-2, gpt-oss), Gemma (Gemini-class open weights). Not ChatGPT/Gemini APIs. Not every Hub CausalLM.

Layout is still **detected from weight names** inside a family, not a hard allowlist of class strings:

| Family | Tensor hints |
|---|---|
| Llama / Mistral / Qwen / Gemma / Phi | `model.layers.*.self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` |
| Fused QKV | `qkv_proj` / `W_pack` / `attention.query_key_value` (sliced) |
| GPT-2 | `transformer.h.*.attn.c_attn` (Conv1D) + GELU MLP + LayerNorm (no RoPE) |
| Mixtral / Qwen2-MoE | `block_sparse_moe` or `mlp.experts.*` — per-expert weights, streamed via expert LRU |

**Denied:** vision / audio / encoder-decoder / T5 / BERT / Whisper / Mamba / RWKV / LLaVA-style towers. Kimi/DeepSeek **HF MLA** is denied until completeness P1.8; use GGUF (`uaii generate --model kimi.gguf`). OpenAI **gpt-oss** is detected as `family=gpt_oss` and fail-closed (MXFP4: convert to BF16/GGUF; dense packing is not Llama-wired).

## Required `config.json` fields

Critical dims fail closed. Aliases: `n_embd`, `n_layer`, `n_head`, `n_inner`, `n_positions`, `layer_norm_epsilon`, `rotary_emb_base`. GPT-2-style configs may omit `intermediate_size` (defaults to `4 * hidden`).

**MoE** keys (`num_local_experts`, `num_experts_per_tok`) are **accepted**. **HF DeepSeek MLA** (`q_a_proj` / `kv_a_proj_with_mqa`) still fails closed — use **DeepSeek GGUF** (expand-then-attend).

## What is not decoder HF yet

Weights are read as **F32, F16, or BF16** safetensors (integer I8/U8 promoted to f32). F8 and GPTQ/AWQ/bitsandbytes/HQQ repos are rejected: convert to dense BF16/F16 or to GGUF.

| Gap | Notes |
|---|---|
| F8 safetensors | Convert the checkpoint to BF16 |
| GPTQ / AWQ / bitsandbytes / HQQ / MXFP4 | Convert to dense BF16/F16 or GGUF |
| HF DeepSeek / Kimi MLA | Use GGUF MLA path (`uaii generate --model kimi.gguf`) |
| gpt-oss MoE packing | `uaii inspect` names `family=gpt_oss`; generate is blocked until the family handler |
| ALiBi | Fail-closed (`position_embedding_type=alibi`) |
| Vision / audio / encoder-decoder / SSM / RWKV | Out of scope (plugin scaffolds) |

Family math that **is** wired: Gemma RMSNorm `(1+w)`, Qwen3 QK-norm, GPT-2 `wpe`, Mistral/Gemma sliding window, Gemma2 extra residual norms + logit cap, Phi partial rotary + longrope factors, Qwen2-MoE shared expert, unknown `hidden_act` fail-closed.

`rope_scaling` (`llama3`, `linear`, `dynamic` NTK, `yarn`, `longrope`/`su` with factor arrays) is applied on RoPE. Unknown types fail closed.

`tokenizer.json` is a real BPE pipeline: ByteLevel pre-tokenizer, byte map, `added_tokens`. Unigram JSON is rejected (`use tokenizer.model`).

## CLI

```bash
uaii pull org/model --outdir ./models/hf/org__model
uaii inspect ./models/hf/org__model
uaii generate --model ./models/hf/org__model --dry-run
uaii generate --model ./models/hf/org__model --prompt "Hello" --json
uaii tokenize encode "hello" --hf ./models/hf/org__model
# Kimi / DeepSeek MLA: GGUF, not a Hub folder
uaii generate --model kimi.gguf --prompt "Hello"
# storage-first is the default (laptop). Override:
uaii generate --model ./models/hf/org__model --preset desktop --ram-gb 24 -p "Hello"
```
