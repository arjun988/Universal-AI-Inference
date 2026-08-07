# Try UAII in 5 minutes

Operator path: build the runtime → run a tiny generate → optional Operator UI.

## 1. Build (≈2 min)

```bash
git clone https://github.com/arjun988/Universal-AI-Inference.git
cd Universal-AI-Inference
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target uaii --parallel
```

Windows App Control often blocks unsigned `uaii.exe`. Prefer WSL:

```bash
bash scripts/build_uaii_wsl.sh
# binary: ~/uaii-wsl-build-dash/libs/uaii-cli/uaii
```

## 2. Generate text (≈1 min)

```bash
# Tiny in-tree demo model (no download)
./build/libs/uaii-cli/uaii generate --demo --prompt "Hello" --max-new-tokens 16

# Sampling (temp / top-p / top-k / rep-penalty)
./build/libs/uaii-cli/uaii generate --demo --prompt "Hello" \
  --temperature 0.8 --top-p 0.9 --top-k 40 --seed 42 --max-new-tokens 32

# Your GGUF (dense or MoE with blk.* + ffn_*_exps)
./build/libs/uaii-cli/uaii generate --model /path/to/model.gguf \
  --prompt "Write a haiku about inference" --max-new-tokens 64
```

## 3. Operator UI (optional, ≈2 min)

Not a double-click consumer app — a local **operator console** over the CLI:

```bash
cd dashboard
npm run install:all
npm run build && npm start
# → http://127.0.0.1:8787
```

Point it at your `uaii` binary (auto-detect, or `UAII_BIN` / `UAII_USE_WSL=1`). Chat uses temperature / top-p / top-k from the UI.

## 4. Reproduce benches (cite CI, don’t guess)

```bash
cmake -S . -B build -DUAII_BUILD_BENCHMARKS=ON -DUAII_WITH_OPENBLAS=ON
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --suite all --providers all --trials 21 --json
```

CI uploads JSON artifacts from the `benchmarks` job. Methodology: [docs/benchmarks.md](docs/benchmarks.md).
