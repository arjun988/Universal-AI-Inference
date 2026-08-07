# GGUF support

Capability-based transformer import (not a hard-coded Llama-only allowlist).

## Architectures

Import is driven by **tensor layout + metadata**, not a short name whitelist.

| Condition | Behavior |
|---|---|
| `token_embd` / emb + `blk.N.attn_q` (+ optional `output` / tied emb) | **Transformer import** for **any** `general.architecture` string |
| Arch-prefixed KV (`{arch}.block_count`, `{arch}.attention.head_count`, …) | Read first; fall back to `llama.*` / `general.*` |
| Missing `general.architecture` but `blk.0.*` present | Still imports (legacy / hand-built) |
| Tied embeddings (no `output.weight`) | Reuse `token_embd.weight` as lm_head |
| MoE (`expert_count` > 0 or `ffn_*_exps` tensors) | **Fail closed** with a clear error (needs expert ops) |
| Weights-only dump (no emb + lm stack) | Any architecture — `Identity` graph |
| Non-`blk.*` layouts (raw GPT-2/Bloom naming, Mamba, RWKV, …) | Not auto-mapped yet — weights graph or convert to `blk.*` GGUF |

Examples that import when they use llama.cpp-style `blk.*` naming:  
`llama`, `llama3`, `mistral`, `qwen2`, `qwen3`, `phi3`, `gemma`, `gemma2`, `internlm2`, `stablelm`, `command-r`, `deepseek2` (dense), and any future arch that follows the same tensor scheme.

## Quant types

| Type | Transformer weights | In-memory quant GEMM |
|---|---|---|
| F32, F16 | yes | dense f32 / dequant |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | yes | yes (`keep_quantized_weights`) |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | yes (CPU tile path) |
| IQ* / unknown enum | **fail closed** (`NotImplemented`) | no |

Use `gguf_type_supported()` before `gguf_type_to_quant()`; unknown types are not silently mapped to F32.

## Graph features (transformer path)

- **RoPE** on Q/K (`theta` from `{arch}.rope.freq_base`, default 10000)
- **GQA metadata**: `{arch}.attention.head_count_kv` → `kv_heads` on Attention
- **RMSNorm** eps from `{arch}.attention.layer_norm_rms_epsilon`
- **FFN**: SwiGLU when `ffn_gate` present; else GELU MLP (`ffn_up` + `ffn_down`)
- **Shapes**: seq=1 decode — tokens `[1,1]`, activations `[1,dim]` (matches `Session::generate`)
- **KV cache**: `use_kv_cache` + `layer_id` on Attention

## Runtime

- Prefill + greedy decode via `Session::generate` / `uaii_session_generate`
- Long sequence bounded by `{arch}.context_length` / `max_context` (fail-closed in generate)
- Env:
  - `UAII_MAX_LAYERS` — optional cap (`0` = unlimited up to 512 absolute max; unset = model `block_count`)
  - `UAII_GEMM`, `UAII_NUM_THREADS`
