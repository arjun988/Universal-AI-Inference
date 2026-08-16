# Hugging Face decoder completeness

Product goal: **mainstream families run correctly** — Qwen, Llama-like, Mistral/Mixtral, Kimi, OpenAI open-weight, and Gemma (the open Gemini-class weights). A user can `uaii pull` then `uaii generate` on those checkpoints and get **correct tokens**, or a **precise error that names the missing capability**. Silent wrong output is a bug.

This is **not** “every `*ForCausalLM` on the Hub”. Unknown families fail closed. Once the families in §1.1 are green, a new Llama-like or Qwen-like card should work without a new runtime; a new paper’s attention still needs a new family handler.

Related: [huggingface_support.md](huggingface_support.md) · [gguf_support.md](gguf_support.md) · [architecture.md](architecture.md)

This is an implementation contract, not a marketing page. Every item cites the current code path and the change that closes it.

**P0 (this document §P0.1–P0.7) is implemented:** BF16 ingest, Llama-3/YaRN/linear/dynamic RoPE scaling, byte-level `tokenizer.json`, sharded `uaii pull`, `uaii inspect` / `generate --dry-run`, Kimi via GGUF MLA, gpt-oss fail-closed (MXFP4 / packing). P1+ remains open.

---

## 1. What “complete” means

A family is **supported** only when all of the following hold for its dense F16/BF16 Hub folder **and** its GGUF `blk.*` (when llama.cpp publishes one):

1. **Ingest.** `config.json` + `*.safetensors` (single file or shards) load into UAII IR.
2. **Weights.** Every tensor used by the graph dequantizes to f32 (or stays packed on a proven GEMM path).
3. **Math.** Attention, RoPE, norms, MLP / MoE, and the LM head match the Hugging Face reference for that architecture.
4. **Tokenizer.** Encode / decode match the Hub tokenizer for the same string, including special tokens.
5. **Chat.** Instruct/chat cards use the shipped `chat_template` (or a documented family equivalent).
6. **Fail closed.** Anything outside this contract returns `NotImplemented` / `InvalidArgument` with the tensor name, dtype, or config key that blocked it.

Recognition is **`model_type` / `architectures[0]` first**, then tensor-name binding inside that family. Guessing structure from names alone is how Qwen3, Gemma2, Kimi, and gpt-oss silently go wrong.

### 1.1 Mainstream families (the product)

Map brand → what actually exists on disk. API-only models are not Hub checkpoints and will never load here.

| Brand (user language) | What we ship | HF `model_type` / class (examples) | GGUF arch | Today | Must close to call it supported |
|---|---|---|---|---|---|
| **Llama-like** | Llama 2/3/3.1/3.2, CodeLlama, TinyLlama, SmolLM, Olmo, Granite (dense) | `llama`, `olmo`, `olmo2`, `granite` · `LlamaForCausalLM` | `llama`, `llama3` | Layout + **BF16 ingest** + Llama-3 `rope_scaling`. Tokenizer byte-level BPE | P1 Gemma-class math still open; Olmo/Granite golden TBD |
| **Qwen** | Qwen2, Qwen2.5, Qwen3 dense; Qwen2-MoE / Qwen3-MoE | `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe` · `Qwen2ForCausalLM`, `Qwen3ForCausalLM` | `qwen2`, `qwen3`, `qwen2moe` | Dense layout + QKV bias + BF16 + RoPE scaling. **Qwen3 QK-norm missing.** MoE shared expert missing | P1.2, P2.3 |
| **Mistral** | Mistral 7B / Nemo / Small (dense), Mixtral 8x7B / 8x22B | `mistral`, `mixtral` · `MistralForCausalLM`, `MixtralForCausalLM` | `mistral`, `mixtral` | Dense + Mixtral experts + BF16. Sliding window not applied | P1.4 |
| **Kimi** | Moonshot **Kimi K2** (and K2-style) open weights | usually `deepseek_v3` / Kimi class · **MLA + MoE**, not Llama | `deepseek2` / `deepseekv3` when converted | **HF MLA rejected.** GGUF MLA expand path exists | **P0.6** (GGUF golden) then **P1.8** HF MLA bind. Shared expert P2.3 |
| **OpenAI** | **Open-weight only:** GPT-2; **gpt-oss-20B / 120B**. Not ChatGPT / GPT-4 / o-series APIs (no weights) | `gpt2` · `GPT2LMHeadModel`; `gpt_oss` · `GptOssForCausalLM` | `gpt2` (often non-`blk`); gpt-oss when llama.cpp names it | GPT-2 layout yes, **`wpe` not applied**. gpt-oss is a **new family** (MoE, YaRN, sinks, often MXFP4) | P1.3 (GPT-2); **P0.7** gpt-oss handler or documented GGUF path |
| **Gemini-class** | Google **Gemma / Gemma 2 / Gemma 3** (the open models). **Gemini** Pro/Flash APIs have no Hub checkpoint | `gemma`, `gemma2`, `gemma3` · `GemmaForCausalLM`, `Gemma2ForCausalLM` | `gemma`, `gemma2`, `gemma3` | Layout may match Llama-like tensors. **RMSNorm is `(1+w)` — wrong today.** Gemma2 extra norms, sliding window, logit cap missing | P0.1, P0.3, P1.1, P1.2, P1.4 |

Vision/audio siblings stay denied: LLaVA, Pixtral, Qwen2-VL, Gemma multimodal towers, Kimi-VL.

### 1.2 Will these “just work” after a generic decoder guesser?

No. After BF16 + tokenizer + Llama-3 RoPE:

- **Llama-like dense and Mistral dense / Mixtral** — expected to generate (short context).
- **Qwen2 / Qwen2.5 dense** — expected to generate; **Qwen3 still needs QK-norm**.
- **Gemma** — will load and still be **wrong** until P1.1.
- **Kimi** — will **not** load from HF until MLA is wired; use GGUF meanwhile (P0.6).
- **OpenAI gpt-oss** — will **not** load as GPT-2; needs P0.7.
- **Gemini API** — never. Point users at Gemma.

That is why the family table is the done list, not “any decoder”.

Mamba, RWKV, Jamba, vision-language towers, and encoder-decoder models stay on the deny list.

---

## 2. Why Hub models fail today

The loader does **not** allowlist class names. It accepts `*ForCausalLM` / `*LMHeadModel`, then guesses the block from tensor names (`pick_layout` in `libs/uaii-loaders/src/hf.cpp`). That is the right design. The failures are downstream of it: dtypes, missing operators, incomplete tokenizer, and incomplete Hub download.

Four independent gates. A model must pass all four.

| Gate | Code | Passes when | Typical Hub miss |
|---|---|---|---|
| Architecture | `hf_architecture_supported` + `arch_denied` | suffix CausalLM / LMHeadModel, not on deny list, not `is_encoder_decoder` | LLaVA, Whisper, T5, Mamba, BERT |
| Layout | `pick_layout` | known QKV + MLP / MoE tensor names | DeepSeek MLA, novel packing, GPTQ `qweight` |
| Weights | `safetensors_load_tensor_f32` | tensor dtype is **F32 or F16 only** | almost every 2024–2026 checkpoint (BF16) |
| Tokenizer | `load_hf_tokenizer` | `vocab.json`+`merges.txt`, `tokenizer.model` (if SentencePiece built), or a BPE `tokenizer.json` | byte-level GPT-2 BPE without `byte_fallback`, Unigram JSON, added tokens |

The website and `huggingface_support.md` previously described gate 1 as “any decoder folder”. That overclaim is what this document exists to retire.

---

## 3. Issue register

Work these in order. Later items assume earlier ones exist. Each row is closed only when the tests in §5 pass.

### P0 — models that already “look supported” must actually run

These are the issues users hit first. Closing P0 makes Llama 3, Qwen 2.5, Mistral, and Phi-3 F16/BF16 folders generate instead of erroring.

#### P0.1 BF16 (and remaining safetensors dtypes)

| | |
|---|---|
| Symptom | `safetensors dtype not supported for f32 load: BF16` after a successful layout match |
| Cause | `safetensors_load_tensor_f32` (`libs/uaii-loaders/src/safetensors.cpp`) only converts F32 and F16. `safetensors_to_dtype` already maps BF16. `bf16_bits_to_f32` already exists in `libs/uaii-quant/src/codec.cpp`. The session already coerces BF16 graph tensors to f32 buffers. The **read path** is the hole. |
| Resolution | Add a conversion table in one place (`safetensors_load_tensor_f32` and any streaming alias of it). Required: F32, F16, BF16. Optional same PR: I8 / U8 as integer→f32. F8_E4M3 / F8_E5M2: convert or fail with “convert this checkpoint to BF16”. Never memcpy BF16 bits into an f32 buffer. |
| Tests | Unit: round-trip F32, F16, BF16, and a rejected F8. Integration: load a tiny BF16 Llama-style fixture and run one generate step. |
| Done when | Llama-3.x and Qwen2.5 Hub BF16 folders load without a dtype error. |

#### P0.2 Llama-3 / Qwen RoPE scaling

| | |
|---|---|
| Symptom | Model loads, tokens are garbage at context > ~2k, or even at short context vs Hugging Face |
| Cause | HF loader emits RoPE with `theta` + `mode` only (`hf.cpp` ~875). `rope_f32` implements interleaved vs NeoX pairing and optional `rotary_dim`. It does **not** read `rope_scaling` (`llama3`, `yarn`, `linear`, `dynamic`, `longrope`). Llama 3.1/3.2 and many Qwen long-context cards are wrong without this. |
| Resolution | Parse `rope_scaling` in `hf_parse_config_json`. Pass factor, low/high freq factors, original context, and type into the RoPE op (new attributes; keep defaults = identity). Implement Llama-3 frequency interpolation in `rope_f32` to match Hugging Face `Llama3ScalingRotaryEmbedding`. Yarn / linear / dynamic as follow-on if the same attribute set covers them; otherwise fail closed naming the `rope_scaling.type`. |
| Tests | Numeric: first 8 RoPE frequencies vs a Python reference for Llama-3.1 8B config. Golden: 32-token greedy decode vs `transformers` on a 1–2 layer fixture. |
| Done when | Llama 3.1/3.2 short prompts match Hugging Face greedy tokens on a tiny or reduced-layer fixture. |

#### P0.3 Tokenizer byte-level BPE + special tokens

| | |
|---|---|
| Symptom | Model loads; output is wrong from token 1; or encode length ≠ Hugging Face |
| Cause | `load_hf_tokenizer` pulls `model.vocab` + `model.merges` and ignores `pre_tokenizer`, `decoder`, `byte_fallback`, `added_tokens`, and `unk_token`. GPT-2 / Llama-3 `tokenizer.json` is byte-level BPE. Missing the byte map silently mangles UTF-8. |
| Resolution | Parse Hugging Face `tokenizer.json` as a real pipeline: pre-tokenizer type, byte-level map, added/special tokens (bos, eos, pad, unk, eot), and decode. Prefer using the same vocab Hugging Face ships rather than a parallel BPE. Unigram / SentencePiece inside JSON: either call SentencePiece or fail closed (“rebuild with SentencePiece” / “use tokenizer.model”). |
| Tests | Encode/decode round-trip for English, CJK, and emoji against Hugging Face `AutoTokenizer` on Llama-3, Qwen2, Mistral, and GPT-2 fixtures (tokenizer files only, no weights). |
| Done when | `uaii tokenize` on those four `tokenizer.json` files matches Hugging Face ids. |

#### P0.4 Sharded Hub pull

| | |
|---|---|
| Symptom | `uaii pull` with only curl writes `config.json` and maybe one `model.safetensors`; 7B+ repos are incomplete |
| Cause | Curl fallback downloads a fixed file list and never walks `model.safetensors.index.json` (`libs/uaii-cli/src/commands/pull.cpp`). |
| Resolution | If `huggingface-cli` / `hf` exists, keep using it. Curl path: download `config.json` + tokenizer files + `*.safetensors.index.json`, parse `weight_map`, download every unique shard, refuse to print “pulled” unless every shard is present and sized. Prefer `huggingface_hub` snapshot when Python is available. |
| Tests | Mock index with two shards; assert both files exist or the command exits non-zero. |
| Done when | `uaii pull` of a two-shard fixture is complete without huggingface-cli. |

#### P0.5 Capability report before generate

| | |
|---|---|
| Symptom | Users discover gaps at runtime, mid-load |
| Cause | No probe. `uaii doctor` does not inspect a model directory. |
| Resolution | `uaii inspect <hf-dir>` (or `uaii generate --dry-run`) prints: architecture, layout family, weight dtypes, tokenizer kind, RoPE scaling type, MoE, denied features. Unknown dtype / layout / scaling must appear as `blocked:` lines with the exact error generate would throw. |
| Tests | BF16 Llama dir → `weights: BF16` (after P0.1: `ok`). GPTQ dir → `blocked: quantized safetensors`. DeepSeek MLA dir → `blocked: HF MLA`. |
| Done when | Support replies can ask for `uaii inspect` output instead of a full generate log. |

#### P0.6 Kimi via GGUF MLA (ship this first)

| | |
|---|---|
| Symptom | Users expect `kimi` / Kimi K2 to run like Llama. HF dir hits the MLA reject. |
| Cause | Kimi K2 is DeepSeek-V3-class (MLA + MoE), not Llama. HF path refuses `q_a_proj` / `kv_a_proj_with_mqa`. GGUF already has expand-then-attend (`docs/gguf_support.md`). |
| Resolution | Treat **GGUF** as the supported Kimi path in v1: golden `uaii generate` on a tiny DeepSeek-V3/Kimi `blk.*` fixture with MLA tensors. Document `uaii generate --model kimi.gguf`. Do not tell users to `uaii pull` a Kimi HF folder until P1.8. `uaii inspect` on a Kimi HF dir must say `blocked: HF MLA; use GGUF`. |
| Tests | Existing GGUF MLA graph tests plus greedy tokens vs llama.cpp (or a stored blob) on a tiny fixture. |
| Done when | Kimi/DeepSeek GGUF is a documented, tested path. HF still errors clearly. |

#### P0.7 OpenAI gpt-oss family handler

| | |
|---|---|
| Symptom | `GptOssForCausalLM` is CausalLM; layout guesser will mis-wire it as Llama or fail on MXFP4 / MoE packing |
| Cause | gpt-oss is MoE + YaRN + attention sinks, often MXFP4 in the official card. It is not GPT-2 and not Llama. |
| Resolution | Allowlist `gpt_oss`. If weights are BF16/F16 dense (community conversions), implement the published block (router + experts + sinks + YaRN). If weights are MXFP4, **fail closed** with convert-to-BF16 or GGUF — same policy as GPTQ. GPT-2 stays a separate family (P1.3). ChatGPT API models stay out of scope. |
| Tests | Config parse for `gpt_oss`; MXFP4 dir rejected with convert hint; BF16 fixture (if we commit one) greedy vs Hugging Face. |
| Done when | `uaii inspect` names `family=gpt_oss` and either `ok` or `blocked: mxfp4 convert to BF16/GGUF`. |

---

### P1 — family math that loads today but is wrong or incomplete

Closing P1 makes **Gemma (Gemini-class)**, **GPT-2**, **Qwen3**, and **Kimi HF** match reference logits, not merely parse.

#### P1.1 Gemma RMSNorm (`(1 + weight)`)

| | |
|---|---|
| Symptom | Gemma / Gemma2 loads; perplexity is nonsense |
| Cause | `rms_norm_f32` multiplies by `w[i]`. Gemma stores a delta: `x * rsqrt * (1 + w)`. |
| Resolution | RMSNorm attribute `offset=1` (default 0). Set it from `model_type` gemma / gemma2 / gemma3. |
| Tests | One-row RMSNorm vs NumPy for both offset 0 and 1. |

#### P1.2 QK-norm

| | |
|---|---|
| Symptom | Qwen3, Gemma2, Olmo2, some Llama variants: attention is wrong |
| Cause | Layout never binds `q_norm` / `k_norm`. Graph goes Q→RoPE→Attention. |
| Resolution | Detect `self_attn.q_norm.weight` / `k_norm.weight`. Insert RMSNorm on Q and K (per-head last axis) before RoPE. |
| Tests | Graph dump contains `q_norm` / `k_norm` nodes when tensors exist; fixture logits vs Hugging Face. |

#### P1.3 GPT-2 learned `wpe`

| | |
|---|---|
| Symptom | GPT-2 / GPT-J-style absolute position models generate without position signal |
| Cause | Documented. Embed is token table only. `transformer.wpe.weight` is ignored. |
| Resolution | If `wpe` (or `embed_positions`) exists, add `tokens_pos` + Embedding + Add before the first layer. Positions must follow the sliding KV absolute index (same as RoPE). |
| Tests | Two-token prompt: hidden state after embed+wpe matches Hugging Face. |

#### P1.4 Sliding window + extra Gemma2 norms / logit cap

| | |
|---|---|
| Symptom | Gemma2/3 and Mistral sliding-window cards disagree with Hugging Face on long context |
| Cause | Attention is full causal. Gemma2 also has post-attn and post-MLP RMSNorm and logit softcapping. |
| Resolution | Pass `sliding_window` into Attention. Add layout for Gemma2 residual norms. Optional `final_logit_softcapping` / `attn_logit_softcapping` as tanh clip attributes. Fail closed if `use_sliding_window` is true and the kernel cannot apply it. |
| Tests | Attention mask with window=8 vs dense causal on a toy QKV. |

#### P1.5 Partial rotary (`partial_rotary_factor`)

| | |
|---|---|
| Symptom | Phi / Phi-3, some Falcon: RoPE applied to the full head |
| Cause | `rotary_dim` is used for MLA GGUF, not wired from HF `partial_rotary_factor` / `rotary_pct`. |
| Resolution | `rotary_dim = head_dim * partial_rotary_factor` (or explicit `rotary_dim`) on both Q and K RoPE nodes. |
| Tests | Phi-3 mini config fixture: `rotary_dim` metadata equals expected. |

#### P1.6 Parallel residual (GPT-J / GPT-NeoX)

| | |
|---|---|
| Symptom | NeoX-style models: attn and MLP both take the pre-norm hidden state |
| Cause | Graph is always sequential: attn residual then MLP on that residual. |
| Resolution | If config `use_parallel_residual` or tensor pattern shows one norm feeding both branches, emit `out = x + attn(x) + mlp(x)`. |
| Tests | Graph has both branches reading the same pre-norm tensor. |

#### P1.7 Activations beyond SiLU / GELU

| | |
|---|---|
| Symptom | `hidden_act: gelu_pytorch_tanh` / `relu2` / `swish` silently uses SiLU or GELU |
| Cause | `use_gelu` is a substring check; everything else is SiLU. |
| Resolution | Map known acts to kernels. Unknown act → `NotImplemented` with the string. |
| Tests | Config with `hidden_act: "foo"` fails closed. |

#### P1.8 Kimi / DeepSeek MLA on Hugging Face

| | |
|---|---|
| Symptom | Kimi K2 and DeepSeek-V3 HF folders error; GGUF works after P0.6 |
| Cause | HF tensor names (`q_a_proj`, `kv_a_proj_with_mqa`, `q_b_proj`, …) are not bound. GGUF expand-then-attend already exists. |
| Resolution | Reuse the GGUF MLA graph builder with an HF name map. Required for **Kimi** to be a first-class Hub path. Do not start until P0 Llama/Qwen goldens match Hugging Face. Include shared expert (P2.3) in the same PR if Kimi K2 has one. |
| Tests | Tiny HF MLA fixture greedy vs the GGUF path on identical weights. |
| Done when | `uaii generate --model ./Kimi-K2-dir` runs on dense BF16. MXFP4/quantized Kimi still convert-or-GGUF. |

---

### P2 — layouts and formats that must fail clearly until implemented

Do not pretend these work. Implement or keep the deny/error. After P2, Hub traffic **outside §1.1** is either supported or explained.

#### P2.1 Quantized Hub repos (GPTQ, AWQ, bitsandbytes, HQQ, compressed-tensors)

| | |
|---|---|
| Symptom | Layout fail on `qweight` / `qzeros`, or garbage if a tensor name collides |
| Cause | No `quantization_config` handling. HF path is dense F16/BF16/F32. Packed GEMM exists for **GGUF** quants only. |
| Resolution | **Do not unpack GPTQ/AWQ in v1.** On `quantization_config` or `qweight` tensors, error: `HF quantized checkpoints are not loaded. Convert to BF16 safetensors or GGUF (Q4_K / IQ4_NL) and retry.` Same message for official MXFP4 gpt-oss / Kimi cards. Optional later: GPTQ unpack-to-f32 for correctness, not speed. |
| Tests | Tiny GPTQ-shaped index is rejected with that message. |

#### P2.2 Hugging Face DeepSeek MLA

Superseded by **P1.8** (Kimi is a product family). Until P1.8 merges, keep the current fail-closed message and the GGUF redirect (P0.6).

#### P2.3 Qwen2-MoE shared expert / DeepSeek shared expert

| | |
|---|---|
| Symptom | Mixtral-style experts load; shared-expert models drop a residual MLP |
| Cause | MoE branch only wires routed experts (`block_sparse_moe` / `mlp.experts`). `mlp.shared_expert` is ignored. |
| Resolution | Detect `shared_expert` / `shared_experts` weights; add parallel SwiGLU and add into the MoE output. |
| Tests | Graph contains shared-expert MatMuls when tensors exist. |

#### P2.4 ALiBi (Bloom, MPT, some Falcon)

| | |
|---|---|
| Symptom | `rope_mode = -1` and no bias → no position information |
| Cause | Only RoPE and (soon) `wpe` exist. |
| Resolution | Attention attribute `alibi=true` + `num_heads`. Or fail closed if `position_embedding_type == "alibi"` until the kernel exists. |
| Tests | Fail closed first; kernel test when implemented. |

#### P2.5 SSM / RWKV / Jamba / vision-language

| | |
|---|---|
| Symptom | Denied by `arch_denied` |
| Cause | No IR ops. |
| Resolution | Keep deny. Error must name the architecture and point at plugins. Do not accept `*ForCausalLM` for these. |
| Tests | `MambaForCausalLM`, `RwkvForCausalLM`, `LlavaForConditionalGeneration` rejected. |

#### P2.6 `.bin` / `.pt` without safetensors

| | |
|---|---|
| Symptom | Folder with only `pytorch_model.bin` is not an HF decoder load |
| Cause | Loader requires safetensors. |
| Resolution | Keep requiring safetensors. Error: `need *.safetensors (convert with transformers / safetensors.torch.save_file)`. Do not parse pickle. |
| Tests | Dir with only `config.json` + `.bin` fails with that message. |

---

### P3 — operational completeness (so support stays easy)

#### P3.1 Layout catalog instead of one `pick_layout`

Split detection into named families: `llama`, `qwen2`, `qwen3`, `mistral`, `mixtral`, `gemma`, `gemma2`, `gpt2`, `gpt_oss`, `kimi_mla` / `deepseek_v3`. `uaii inspect` prints the family. A new Hub naming scheme adds a family + tests, not another `else if` buried in 150 lines.

Unknown layout error must list: first 20 tensor names, expected examples (`model.layers.0.self_attn.q_proj.weight`), and a link to this document.

#### P3.2 Golden-model matrix in CI

See §5. This is what prevents “we fixed Llama and broke Gemma”.

#### P3.3 Docs and website stay in lockstep with the matrix

`docs/huggingface_support.md` and `website/app/docs/huggingface/page.tsx` list **supported families and dtypes**, not “any decoder”. When a P-item closes, update both in the same PR.

#### P3.4 Convert hint, not a second runtime

Users with GPTQ/AWQ/GGUF-only needs get one documented path:

```text
# Dense BF16 Hub → UAII
uaii pull org/model --outdir ./hf-model
uaii generate --model ./hf-model --prompt "Hello"

# Already GGUF
uaii generate --model model.gguf --prompt "Hello"

# Quantized HF (not loaded)
# Convert with the publisher's tool to BF16 or to GGUF, then use the lines above.
```

Do not add a half-working GPTQ kernel “to say we support it”.

---

## 4. Recommended implementation order

Order is the §1.1 product: Llama + Qwen2 + Mistral first (users think these already work), then Gemma, then Kimi GGUF, then gpt-oss, then Kimi HF.

| Step | Item | Unlocks |
|---|---|---|
| 1 | P0.1 BF16 load | Llama / Qwen / Mistral / Gemma Hub folders ingest |
| 2 | P0.3 tokenizer.json | Tokens are not garbage (Llama-3, Qwen, Mistral, GPT-2) |
| 3 | P0.2 RoPE scaling | Llama 3 and long-context Qwen correctness |
| 4 | P0.5 inspect | Supportable errors (`family=…`, `blocked:…`) |
| 5 | P0.4 sharded pull | `uaii pull` of 7B+ Qwen/Llama repos |
| 6 | P1.1 + P1.4 | **Gemma / Gemini-class** actually correct (not just loaded) |
| 7 | P1.2 | **Qwen3** (and Gemma2 QK-norm) |
| 8 | P0.6 | **Kimi** via GGUF MLA (ship while HF MLA is open) |
| 9 | P0.7 | **OpenAI gpt-oss** (BF16 or convert hint) |
| 10 | P1.3 | GPT-2 `wpe` |
| 11 | P3.1 layout catalog + P3.2 CI matrix | No regressions across the six brands |
| 12 | P1.8 + P2.3 | **Kimi / DeepSeek HF** + Qwen-MoE shared expert |
| 13 | P1.5–P1.7, P2.1 documented reject | Phi partial rotary, unknown acts, GPTQ/MXFP4 |

Do not start P1.8 until step 3 goldens match Hugging Face. Wrong MLA is worse than a clear GGUF redirect.

---

## 5. Tests that make “no further issues” true

Bugs will still exist. What this section prevents is **unknown families silently succeeding** and **known families regressing**.

### 5.1 Unit (no Hub download)

| Test | Asserts |
|---|---|
| `safetensors_load_tensor_f32` F32 / F16 / BF16 | max abs error vs NumPy |
| rejected dtype | error contains the dtype string |
| `hf_parse_config_json` Llama 3 `rope_scaling` | fields populated |
| unknown `hidden_act` | `NotImplemented` |
| `arch_denied` samples | Mamba, LLaVA, BERT, T5 |
| tokenizer encode/decode | fixtures for Llama-3, Qwen2, Mistral, GPT-2, Gemma (SentencePiece or tokenizer.json) |
| RMSNorm offset 0 vs 1 | numeric |
| QK-norm graph rewrite | nodes present iff weights present |

### 5.2 Golden generate (tiny weights, committed)

Commit **tiny random** safetensors (not 7B weights) with real `config.json` + tokenizer files for each **§1.1 family**:

- Llama-style split QKV + SwiGLU (BF16) + `rope_scaling.type=llama3`
- Qwen2 with QKV bias + NeoX RoPE
- Qwen3 with QK-norm (after P1.2)
- Mixtral 2-layer 4-expert
- Gemma with RMSNorm offset 1
- GPT-2 Conv1D + GELU + `wpe`
- DeepSeek/Kimi MLA GGUF (P0.6); HF MLA fixture after P1.8
- gpt-oss config (P0.7): either tiny BF16 MoE or a reject-MXFP4 test

For each: greedy 8 tokens vs a Python `transformers` script in `tests/goldens/` (or precomputed logits blob). Tolerance: exact token match on greedy; logits MAE documented.

### 5.3 Optional nightly (real Hub, not required for merge)

| Repo (examples) | Family |
|---|---|
| `HuggingFaceTB/SmolLM2-135M` | Llama-like BF16 |
| `Qwen/Qwen2.5-0.5B` | Qwen2 |
| `Qwen/Qwen3-0.6B` (after P1.2) | Qwen3 QK-norm |
| `mistralai/Mistral-7B-v0.3` or a tiny Mistral stand-in | Mistral sliding window |
| `gpt2` | OpenAI GPT-2 |
| `openai/gpt-oss-20b` (BF16 conversion if official is MXFP4) | gpt-oss |
| `google/gemma-2-2b-it` (if license CI allows) | Gemini-class |
| A small Kimi/DeepSeek **GGUF** | Kimi via P0.6 |

Nightly may skip gated repos. Merge CI must never depend on Hub or licenses.

Nightly may skip gated repos. Merge CI must never depend on Hub or licenses.

### 5.4 Definition of done for the whole document

§1.1 is green, not “the Hub is done”.

- [ ] Llama-like: P0.1–P0.3; golden greedy matches Hugging Face
- [ ] Qwen2/2.5: same; Qwen3: P1.2
- [ ] Mistral dense + Mixtral: P0.1, P0.3, P1.4
- [ ] Gemma / Gemma2: P1.1, P1.2, P1.4 (this is the Gemini-class Hub path)
- [ ] Kimi: P0.6 GGUF documented and tested; P1.8 if we claim HF folders
- [ ] OpenAI: GPT-2 `wpe` (P1.3); gpt-oss P0.7 (`ok` or convert hint, never silent Llama)
- [ ] `uaii inspect` prints `family=` from the table and `blocked:` outside it
- [ ] Website Hugging Face page lists these six brands, dtypes F16/BF16, and the Gemini/ChatGPT API exclusion
- [ ] GPTQ/AWQ/MXFP4/SSM either converted or rejected with §6 messages
- [ ] No path copies BF16/F16 bits into f32 without conversion
- [ ] No path applies RoPE without honouring `rope_scaling` when present

When those boxes are ticked, a new **Llama-like or Qwen2-like** Hub model is a config problem, not a new runtime. Kimi-class and gpt-oss stay explicit families.

---

## 6. Error catalogue (user-facing)

Every message must include **what failed** and **what to do**. Implement these strings (or equivalent) so support can grep them.

| Condition | Error |
|---|---|
| Encoder-decoder / deny list | `unsupported Hugging Face architecture (decoder-only CausalLM required): {arch}` |
| Unknown layout | `HF decoder layout not recognized under {dir}. First tensors: … Expected e.g. model.layers.0.self_attn.q_proj.weight. See docs/huggingface_decoder_completeness.md` |
| BF16 before P0.1 | `safetensors dtype not supported for f32 load: BF16 (convert to F16/F32 or wait for BF16 ingest)` |
| F8 / unknown dtype | `safetensors dtype not supported: {dtype}. Export BF16/F16 safetensors.` |
| `quantization_config` | `HF quantized checkpoints are not loaded. Convert to BF16 safetensors or GGUF.` |
| MLA HF (before P1.8) | `HF DeepSeek/Kimi MLA is not mapped. Use a GGUF (uaii generate --model model.gguf).` |
| gpt-oss MXFP4 | `gpt-oss MXFP4 is not loaded. Convert to BF16 safetensors or GGUF.` |
| Tokenizer missing | `no HF tokenizer in {dir} (need tokenizer.json, vocab.json+merges.txt, or tokenizer.model)` |
| SentencePiece off | `tokenizer.model found but UAII was built without SentencePiece. Rebuild with -DUAII_WITH_SENTENCEPIECE=ON` |
| Rope scaling unknown | `unsupported rope_scaling.type={type}` |
| Curl pull incomplete | `pull incomplete: missing shard {name} listed in model.safetensors.index.json` |
| Single tensor > RAM | `weight {name} is {bytes} which exceeds the memory preset. Use --preset desktop\|server or a quantized GGUF.` |

---

## 7. Code touch list

| Area | Files |
|---|---|
| Dtype ingest | `libs/uaii-loaders/src/safetensors.cpp`, tests next to `hf_loader_test.cpp` |
| Config + layout + graph | `libs/uaii-loaders/src/hf.cpp`, `include/uaii/loaders/hf.hpp` |
| RoPE / RMSNorm / Attention | `libs/uaii-kernels/src/transformer.cpp`, `norm.cpp`, `dispatch.cpp` |
| Tokenizer | `libs/uaii-tokenizers/src/hf_tokenizer.cpp`, `bpe_tokenizer.cpp` |
| Pull | `libs/uaii-cli/src/commands/pull.cpp` |
| Inspect | new command or `inspect.cpp` |
| Docs | this file, `huggingface_support.md`, `website/app/docs/huggingface/page.tsx` |

Core session, C API, and Python SDK should not change except where new `SessionOptions` are required (they should not be for P0).

---

## 8. What we will still not claim

After this work, UAII still will not:

- Run **Gemini** or **ChatGPT / GPT-4** (no weights; use Gemma and gpt-oss / GPT-2)
- Match llama.cpp tokens/s or GPU coverage
- Run vision, audio, encoder-decoder, SSM, or RWKV (including Kimi-VL, Pixtral, Qwen2-VL)
- Load GPTQ/AWQ/MXFP4 at speed (convert to BF16 or GGUF)
- Guarantee trillion-parameter RSS
- Treat Kimi HF folders as Llama-like

Those stay documented limits, not bugs.

---

## 9. Support playbook

When someone says “this Hub model does not run”:

1. Ask for `uaii inspect <dir>` (or generate’s first error line).
2. Match it to §6.
3. If `blocked:` is a P-item still open, point at this document’s section. Do not patch a one-off in the CLI.
4. If inspect says `ok` and generate is wrong, it is a golden-matrix miss: add a fixture, do not tweak prompts.
5. If the architecture is denied, say so and offer GGUF only when that arch exists in `gguf_support.md`.
