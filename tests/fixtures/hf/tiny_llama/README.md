# tiny_llama fixture

Minimal Hugging Face–style `config.json` for `LlamaForCausalLM` (tiny dims).

**Weights are not checked in.** The unit test `uaii_hf_loader_test` writes a temporary directory with random-ish F32 safetensors (`model.embed_tokens.weight`, `model.layers.{0,1}.*`, `model.norm.weight`) via `safetensors_write_f32`, then loads it with `HfLoader`.

Optional: regenerate a local weight file for manual experiments:

```bash
# from repo build tree, after building tests
./uaii_hf_loader_test
```

Or copy this `config.json` next to any compatible safetensors set that uses Llama tensor names.
