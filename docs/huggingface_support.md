# Hugging Face support

First-class **directory** load for decoder-only LLMs: `config.json` + tokenizer files + `*.safetensors` (single file or Hub shards).

Related: [gguf_support.md](gguf_support.md) · [upgrade.md](../upgrade.md)

## Directory layout

A model directory is accepted when it contains `config.json` (see `HfLoader::accepts`).

| File / pattern | Role |
|---|---|
| `config.json` | **Required.** Strict architecture + dim fields (fail closed). |
| `*.safetensors` | Weights. Single `model.safetensors` or shards. |
| `model.safetensors.index.json` | Optional Hub shard map (`weight_map`: tensor → shard filename). Any `*.safetensors.index.json` is also accepted. |
| `tokenizer.json` / `tokenizer.model` / related | Optional for tokenize; loader does not require them to build the IR graph. |
| `tokenizer_config.json` | Optional; `chat_template` string is read when present. |
| `generation_config.json` | Ignored by the loader today. |

Without an index file, UAII scans all `*.safetensors` in the directory and builds a tensor → shard map from headers.

Expected tensor naming is **Llama-style**:

- `model.embed_tokens.weight` (or `embed_tokens.weight`)
- `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
- `model.layers.{i}.input_layernorm.weight` / `post_attention_layernorm.weight`
- `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
- `model.norm.weight`
- `lm_head.weight` **or** tied embeddings (reuse embed)

## Allowlisted architectures

From `include/uaii/loaders/hf.hpp` / `libs/uaii-loaders/src/hf.cpp` (`kSupportedArchs` / `hf_architecture_supported`):

| Architecture |
|---|
| `LlamaForCausalLM` |
| `MistralForCausalLM` |
| `Qwen2ForCausalLM` |
| `Qwen3ForCausalLM` |
| `GemmaForCausalLM` |
| `Gemma2ForCausalLM` |
| `Gemma3ForCausalLM` |
| `Phi3ForCausalLM` |
| `PhiForCausalLM` |
| `InternLM2ForCausalLM` |
| `InternLMForCausalLM` |
| `StableLmForCausalLM` |
| `CohereForCausalLM` |
| `OlmoForCausalLM` |
| `Olmo2ForCausalLM` |
| `GraniteForCausalLM` |
| `ExaoneForCausalLM` |

Unknown `architectures[0]` → `NotImplemented` with a clear error (no silent Identity graph).

## Required `config.json` fields + aliases

Critical fields **fail closed** (missing / non-positive → `InvalidArgument`).

| Canonical field | Accepted aliases |
|---|---|
| `architectures[0]` | (required array; first string used) |
| `hidden_size` | `n_embd`, `d_model` |
| `num_hidden_layers` | `n_layer`, `n_layers` |
| `num_attention_heads` | `n_head` |
| `intermediate_size` | `n_inner`, `ffn_dim` |
| `vocab_size` | — |
| `max_position_embeddings` | `n_positions`, `max_sequence_length` |
| `rms_norm_eps` | `rms_norm_epsilon`, `layer_norm_eps`, `layer_norm_epsilon` |

Optional / derived:

| Field | Behavior |
|---|---|
| `model_type` | Stored in graph metadata when present. |
| `num_key_value_heads` | Defaults to `num_attention_heads` (MHA) when omitted. |
| `rope_theta` | Defaults to `10000` when omitted. |
| `head_dim` | Defaults to `hidden_size / num_attention_heads`. |
| `tie_word_embeddings` | Bool; if true or `lm_head` missing, reuse embed weights. |

**Rejected** (fail closed): MoE keys (`num_local_experts`, `num_experts`, `moe_intermediate_size`); MLA keys (`kv_lora_rank`, `q_lora_rank`) — message points to **DeepSeek GGUF + UAII MLA expand path**.

RoPE mode is chosen from architecture name: NeoX-style (`mode=2`) for Qwen / Gemma / Phi / Olmo / StableLM families; interleaved (`mode=0`) otherwise (e.g. Llama / Mistral).

## What works

- **RoPE** on Q/K (theta + head_dim + mode), same class of graph as GGUF
- **GQA** via `num_key_value_heads` → Attention `kv_heads`
- **Tied embeddings** when `lm_head` is absent or `tie_word_embeddings`
- Dense **SwiGLU** MLP (`gate` / `up` / `down`)
- **Sharded** Hub repos via index JSON
- CLI: `uaii generate --model ./hf-dir`, `uaii convert ./hf-dir`, `uaii pull <org/model>`

## What does **not** work (yet)

| Gap | Notes |
|---|---|
| **MoE HF** | Rejected when expert config keys are present; use GGUF MoE `ffn_*_exps` instead |
| **MLA / DeepSeek HF** | `kv_lora_rank` / `q_lora_rank` fail closed — use DeepSeek GGUF MLA expand path |
| **GPT-2 layouts** | `transformer.h.*` not mapped |
| **Vision / multimodal** | Out of scope for this loader |
| **IQ\* via HF** | HF path is F32 safetensors today; IQ\* remains a GGUF gap |
| Fused QKV / `W_pack`, GPT-NeoX alternate names, encoder-decoder | Not auto-mapped |

## CLI examples

```bash
# Download a Hub repo into a local cache/dir
uaii pull hf-internal-testing/tiny-random-LlamaForCausalLM
uaii pull Qwen/Qwen2.5-0.5B-Instruct --outdir ./models/qwen

# Generate from an HF directory (memory dial)
uaii generate --model ./models/qwen --preset laptop -p "Hello" -n 32

# Convert HF directory → UAII IR JSON (+ weight refs)
uaii convert ./models/qwen -o qwen.uaii.json
```

## Fixtures

Tiny Llama-shaped config (weights produced by unit tests):  
[`tests/fixtures/hf/tiny_llama/`](../tests/fixtures/hf/tiny_llama/).

Env: `UAII_MAX_LAYERS` caps imported layers (same idea as GGUF).
