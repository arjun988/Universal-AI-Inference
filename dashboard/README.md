# UAII Dashboard

Local / self-host operator console for [Universal AI Inference](../README.md).

**This app lives in `dashboard/` and is completely separate from `website/`** (the public marketing/docs site).

```text
dashboard/          ← you are here (ops UI + API)
website/            ← product site (Next.js) — do not mix
```

Thin shell: the dashboard spawns the `uaii` CLI. It does not reimplement inference.

## Quick start

1. Build UAII (so `uaii` / `uaii.exe` exists under `build/`).
2. Install and run the dashboard:

```bash
cd dashboard
npm run install:all
npm run dev
```

- UI (Vite): http://127.0.0.1:5174  
- API: http://127.0.0.1:8787  

Production (serve built UI from the API server):

```bash
cd dashboard
npm run install:all
npm run build
npm start
```

Then open http://127.0.0.1:8787

## Config

Copy `uaii-dash.example.json` → `uaii-dash.json`, or use env:

| Env | Meaning |
|---|---|
| `UAII_BIN` | Path to `uaii` executable |
| `UAII_MODEL_DIR` | Model library directory |
| `UAII_DASH_BIND` | Default `127.0.0.1` |
| `UAII_DASH_PORT` | Default `8787` |
| `UAII_DASH_TOKEN` | Required if bind is not loopback |
| `UAII_GEMM` / `UAII_NUM_THREADS` | Passed through to CLI |

## MVP pages

| Page | Action |
|---|---|
| Chat / Run | `uaii run --demo …` |
| Models | List / upload `.gguf` / `.uaii.json`; convert GGUF |
| Runtime | `uaii doctor --load-plugins` |
| Logs | Recent CLI jobs |
| Settings | Binary path, model dir, backend, GEMM, threads |

## Self-host note

Default bind is localhost. For LAN:

```bash
UAII_DASH_BIND=0.0.0.0 UAII_DASH_TOKEN=your-secret npm start
```

Send `Authorization: Bearer your-secret` from clients.

## PRD

See [docs/prd-dashboard.md](../docs/prd-dashboard.md).
