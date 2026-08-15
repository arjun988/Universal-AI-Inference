# Hugging Face support

First-class **directory** load for **decoder-only** LLMs: `config.json` + tokenizer files + `*.safetensors` (single file or Hub shards).

Related: [gguf_support.md](gguf_support.md) · [upgrade.md](../upgrade.md)

## Laptop / CPU memory dial (kimi-style)

`uaii generate` and `uaii chat` default to **`--preset laptop`**:

| Preset | Resident target | Pin | Expert LRU | KV window | Streaming |
|---|---|---|---|---|---|
| `laptop` (default) | ~8 GiB | first 2 layers + embed/lm_head/norm | 1 GiB | 2048 tokens (+ 4 sink) | on when weights exceed budget |
| `desktop` | ~32 GiB | 16 layers | 4 GiB | 4096 | on under pressure |
| `server` | ~128 GiB | dense if it fits | 32 GiB | 8192 | on under pressure |
| `none` | unlimited | — | — | grow with `--max-context` | off |

Weights stay on disk (`file#tensor` refs). The session **pins the trunk** (embeddings, final norm, lm_head, first N layers) and **pages the rest** through a double buffer. Routed MoE expert tensors are **never fully pinned**; an LRU keeps recently used experts in RAM. KV is a **sliding window** (default 2048 on laptop, plus 4 attention-sink tokens) so long chats cannot OOM. GGUF Mixtral/Qwen banks are split per expert (`#e=N`) and only the **top-k** experts of that token are staged — the kimi-k3-in-c residency rule. Peak RSS is reported on generate.

This is the same idea as kimi-k3-in-c: **storage is a scheduler resource**. A 30B+ decoder and **500B-class MoE** (page top-k experts, bounded KV) can run on an 8 GB laptop; it will be **slow** (disk + CPU) rather than OOM. Token latency is dominated by paging, not by “the model is small so it should be instant.” Override KV with `--kv-window`. Dense checkpoints still stream layer tensors; a single un-sliced FFN bank that is larger than RAM is the remaining fail case (`--preset none` if everything already fits).

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

Any Hub card whose `architectures[0]` is a **decoder-only** class:

- suffix `ForCausalLM` or `LMHeadModel` (Llama, Mistral, Mixtral, Qwen, Gemma, Phi, GPT-2, GPT-NeoX, Falcon, Olmo, Granite, …)
- **Denied:** vision / audio / encoder-decoder / T5 / BERT / Whisper / Mamba / RWKV / LLaVA-style towers

Unknown tensor **layouts** still fail closed with a clear error (not a silent Identity graph).

Layout is **detected from weight names**, not a hard allowlist of class strings:

| Family | Tensor hints |
|---|---|
| Llama / Mistral / Qwen / Gemma / Phi | `model.layers.*.self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj` |
| Fused QKV | `qkv_proj` / `W_pack` / `attention.query_key_value` (sliced) |
| GPT-2 | `transformer.h.*.attn.c_attn` (Conv1D) + GELU MLP + LayerNorm (no RoPE) |
| Mixtral / Qwen2-MoE | `block_sparse_moe` or `mlp.experts.*` — per-expert weights, streamed via expert LRU |

## Required `config.json` fields

Critical dims fail closed. Aliases: `n_embd`, `n_layer`, `n_head`, `n_inner`, `n_positions`, `layer_norm_epsilon`, `rotary_emb_base`. GPT-2-style configs may omit `intermediate_size` (defaults to `4 * hidden`).

**MoE** keys (`num_local_experts`, `num_experts_per_tok`) are **accepted**. **HF DeepSeek MLA** (`q_a_proj` / `kv_a_proj_with_mqa`) still fails closed — use **DeepSeek GGUF** (expand-then-attend).

## What is not decoder HF yet

| Gap | Notes |
|---|---|
| HF DeepSeek MLA | Use GGUF MLA path |
| Learned GPT-2 `wpe` | Attention runs; absolute position table is not added |
| Vision / audio / encoder-decoder / SSM / RWKV | Out of scope (plugin scaffolds) |
| IQ* via HF | HF path is F16/F32 safetensors; GGUF carries packed quants |

## CLI

```bash
uaii pull org/model --outdir ./models/hf/org__model
uaii generate --model ./models/hf/org__model --prompt "Hello" --json
# storage-first is the default (laptop). Override:
uaii generate --model ./models/hf/org__model --preset desktop --ram-gb 24 -p "Hello"
```
