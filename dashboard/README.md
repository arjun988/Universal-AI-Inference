# UAII Operator UI

Local **and** self-host **operator console** for Universal AI Inference.

This is **not** a signed desktop “download and double-click” app — you build `uaii`, then run this Node console against it.  
**Separate from `website/`** (marketing/docs).

```text
Browser  →  dashboard server (:8787)  →  uaii / uaii_bench CLI
```

## Features (end-to-end)

| Area | What you get |
|---|---|
| Chat / Run | **Real GGUF LLM chat** (warm `uaii chat`), streaming, tokenize, IR |
| Models | Upload, list, convert GGUF→IR, delete |
| Runtime | `uaii doctor --load-plugins` |
| Benchmarks | `uaii_bench` JSON + table |
| Logs | Recent CLI jobs |
| Settings | Bind/port, binaries, GEMM, threads, token |
| API | REST `/api/*` + OpenAI-compatible `/v1/*` |
| Auth | Token required when binding off localhost |

## Local (laptop)

```bash
# 1) Build UAII runtime first
cmake --build build --target uaii uaii_bench --parallel
# Windows App Control often blocks unsigned .exe — use WSL instead:
#   bash scripts/test_generate_wsl.sh
# Dashboard auto-detects WSL uaii, or set:
#   UAII_USE_WSL=1
#   UAII_WSL_BIN=/home/$USER/uaii-wsl-build-dash/libs/uaii-cli/uaii

# 2) Dashboard
cd dashboard
npm run install:all

# Dev (hot UI on :5174, API on :8787)
npm run dev

# Or one-port production locally
npm run build && npm start
# → http://127.0.0.1:8787
```

Windows:

```powershell
cd dashboard
npm run install:all
.\scripts\start-local.ps1
# → http://127.0.0.1:8787
```

Self-host on Windows LAN:

```powershell
$env:UAII_DASH_TOKEN = -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) })
.\scripts\start-host.ps1
```

## Self-host (company LAN / server)

```bash
export UAII_DASH_TOKEN=$(openssl rand -hex 16)
export UAII_DASH_BIND=0.0.0.0
cd dashboard
./scripts/start-host.sh
```

Docker:

```bash
export UAII_DASH_TOKEN=$(openssl rand -hex 16)
# Point volumes at your built binaries (Linux paths)
export UAII_HOST_BIN=$PWD/../build/libs/uaii-cli/uaii
export UAII_HOST_BENCH=$PWD/../build/benchmarks/uaii_bench
cd dashboard
docker compose up --build
```

Open `http://SERVER:8787`, unlock with the token.

### OpenAI-compatible clients

```bash
curl -s http://SERVER:8787/v1/models \
  -H "Authorization: Bearer $UAII_DASH_TOKEN"

curl -s http://SERVER:8787/v1/chat/completions \
  -H "Authorization: Bearer $UAII_DASH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"uaii-demo-gguf","messages":[{"role":"user","content":"hi"}]}'
```

## Config

Env vars override `uaii-dash.json` (copy from `uaii-dash.example.json`):

| Variable | Default | Meaning |
|---|---|---|
| `UAII_BIN` | auto-detect build tree | Path to `uaii` |
| `UAII_BENCH_BIN` | auto-detect | Path to `uaii_bench` |
| `UAII_MODEL_DIR` | `dashboard/models` | Model library |
| `UAII_DASH_BIND` | `127.0.0.1` | Use `0.0.0.0` to self-host |
| `UAII_DASH_PORT` | `8787` | HTTP port |
| `UAII_DASH_TOKEN` | empty | **Required** if bind ≠ loopback |
| `UAII_GEMM` / `UAII_NUM_THREADS` | — | Passed to CLI |

Non-loopback without a token **refuses to start**.

## Typical user flows

1. **LLM chat:** Models → Import a `.gguf` → Chat → select it → Run (streams tokens). No model → UI asks you to upload.  
2. **OpenAI clients:** `POST /v1/chat/completions` with `model: "uaii-file-<name.gguf>"`  
3. **Tools:** Chat → Tools for tokenize / IR / CLI smoke demos  
4. **Health:** Runtime → Run doctor  
5. **Perf:** Benchmarks → Run microbench  
6. **Team API:** Self-host + Bearer token  

CLI (also used by the dashboard):

```bash
uaii generate --demo --prompt "hello" --max-new-tokens 8 --json
uaii generate --model path/to/model.gguf --prompt "hi" --max-new-tokens 64 --stream
uaii chat --model path/to/model.gguf --jsonl   # warm session worker
```

## PRD

[docs/prd-dashboard.md](../docs/prd-dashboard.md)
