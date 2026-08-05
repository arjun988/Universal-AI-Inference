# GGUF support freeze list

Finite Done line for Llama-family GGUF (plan Workstream C).

## Architectures

| `general.architecture` | Transformer import (`blk.N.*`) |
|---|---|
| `llama`, `llama3`, `mistral`, `qwen2`, `phi3` (case-insensitive) | Supported |
| Other non-empty values | **Fail closed** (`NotImplemented`) |
| Missing metadata | Allowed when `blk.0.*` tensors present (legacy / hand-built) |
| Weights-only dump (no emb+lm stack) | Any architecture — `Identity` graph |

## Quant types

| Type | Transformer weights | In-memory quant GEMM |
|---|---|---|
| F32, F16 | yes | dense f32 / dequant |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | yes | yes (`keep_quantized_weights`) |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | yes (CPU tile path) |
| IQ* / unknown enum | **fail closed** (`NotImplemented`) | no |

Use `gguf_type_supported()` before `gguf_type_to_quant()`; unknown types are not silently mapped to F32.

## Graph features (transformer path)

- **RoPE** on Q/K after projections (`theta` from `llama.rope.freq_base` or `general.rope.freq_base`, default 10000)
- **GQA metadata**: `llama.attention.head_count_kv` → `kv_heads` on Attention (kernel uses `num_heads` today)
- **Shapes**: seq=1 decode — tokens `[1,1]`, activations `[1,dim]` (matches `Session::generate` step loop)
- **KV cache**: `use_kv_cache` + `layer_id` on Attention

## Runtime

- Prefill + greedy decode via `Session::generate` / `uaii_session_generate`
- Long sequence bounded by `max_context` / `context_length` metadata (fail-closed in generate)
- Env:
  - `UAII_MAX_LAYERS` — optional cap (`0` = unlimited up to 512 absolute max; unset = model `block_count`)
  - `UAII_GEMM`, `UAII_NUM_THREADS`
