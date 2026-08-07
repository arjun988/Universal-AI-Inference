# PRD: UAII Dashboard (Local + Self-Host)

**Status:** M1–M3 implemented in `dashboard/` — real GGUF LLM chat via `uaii generate` / warm `uaii chat --jsonl`, streaming, OpenAI `/v1`, local + self-host (separate from `website/`)  
**Product:** Universal AI Inference Runtime — operator / developer console  
**Depends on:** UAII CLI + C API / session runtime (existing)

---

## 1. Problem

UAII today is CLI + library. Companies and local users need a **simple UI** to:

- Install / run models without living in a terminal  
- See health, devices, and basic performance  
- Self-host on a server **or** run fully offline on a laptop  

Without a dashboard, adoption stays limited to engineers who already know the stack.

---

## 2. Goals

| Goal | Success signal |
|---|---|
| Local-first | Zip / installer runs offline after first model download (or with bundled tiny demo) |
| Self-host ready | Single container or binary + config; bind to LAN/VPN |
| Honest ops | Surfaces `uaii doctor` truth (backends, GEMM, fallbacks) |
| Thin shell | Dashboard orchestrates UAII; does **not** reimplement the runtime |

### Non-goals (v1)

- Multi-tenant SaaS billing / cloud control plane  
- Training / fine-tuning  
- Competing with ChatGPT as a consumer chat app  
- Silent “GPU” labels when math is on CPU  

---

## 3. Users

| Persona | Need |
|---|---|
| **Local power user** | Zip → run → chat with a local GGUF |
| **Company IT / platform** | Self-host on internal server; control models & ports |
| **App engineer** | Inspect sessions, graphs, profiles while integrating C/Python API |
| **Eval / ML eng** | Run smoke benches; compare provider notes |

---

## 4. Deployment modes

| Mode | Package | Notes |
|---|---|---|
| **Local desktop** | Zip + `.exe` / `.app` / Linux AppImage | Bundles dashboard UI + `uaii` binary; models in user data dir |
| **Self-host server** | Docker Compose **or** single static binary + config | LAN only by default; optional reverse proxy |
| **Dev sidecar** | `uaii dash` starts local web UI against existing build | For contributors |

Security default: **localhost-only** bind; auth token required if exposed beyond loopback.

---

## 5. Feature list (what to make)

### P0 — MVP (ship first)

1. **Install / launch** — one command or double-click; health check via `uaii doctor`  
2. **Model library** — list local models; import path / drop `.gguf`; show size & quant  
3. **Chat / generate** — prompt → tokens (streaming if API allows); stop button  
4. **Runtime panel** — backend, GEMM provider, threads, device string (from doctor)  
5. **Job log** — last runs, errors, copyable logs  
6. **Settings** — model dir, port, bind address, max context, threads, `UAII_GEMM`  

### P1 — Useful for companies

7. **Multi-model slots** — switch active model without restart (or warm reload)  
8. **API gateway** — optional OpenAI-compatible HTTP for internal apps  
9. **Users / token auth** — single shared token or basic users (self-host)  
10. **Profiler view** — load Chrome-trace / session profile JSON  
11. **Benchmarks UI** — run `uaii_bench` suites; show JSON table (ref / OpenBLAS)  
12. **Update channel** — check GitHub releases (optional, offline-safe skip)  

### P2 — Later

13. Multi-user concurrent sessions with queues  
14. Model download manager (Hugging Face) with checksums  
15. RBAC, audit log, SSO  
16. Cluster / multi-node (out of scope until single-node is solid)  

---

## 6. UX sketch (one composition)

```text
┌─────────────────────────────────────────────┐
│  UAII                          [Local|Host] │
├──────────┬──────────────────────────────────┤
│ Models   │  Chat / Generate                 │
│ Runtime  │  [ prompt .............. ] [Run] │
│ Bench    │  streaming tokens…               │
│ Logs     │                                  │
│ Settings │  Backend: cpu · GEMM: openblas   │
└──────────┴──────────────────────────────────┘
```

Brand + one primary action (Run). No card farm in the hero.

---

## 7. Architecture

```text
Browser / Desktop shell
        │  HTTP (localhost or LAN)
        ▼
Dashboard server (Rust/Go/Node — TBD)
        │  spawn / FFI / subprocess
        ▼
uaii CLI + libuaii_capi  (Session::generate, doctor, bench)
        │
        ▼
Models on disk (user-configured directory)
```

**Rule:** all inference goes through UAII. Dashboard is UI + process manager + thin HTTP.

---

## 8. Packaging checklist

| Artifact | Contents |
|---|---|
| **Windows zip / installer** | `uaii-dash.exe`, `uaii.exe`, C API DLL if needed, static web assets, LICENSE, README |
| **Linux tarball / Docker** | same + `docker-compose.yml` (port, volume for models) |
| **macOS** | `.app` or brew cask later |
| **Config** | `uaii-dash.toml` — bind, port, model_dir, token, threads |

Code-sign Windows/macOS builds where possible (WDAC / Gatekeeper).

---

## 9. Security & compliance (self-host)

- Default bind `127.0.0.1`; warn if `0.0.0.0`  
- Bearer token for any non-loopback bind  
- No telemetry by default  
- Models stay on customer disk  
- Clear data paths for IT (model cache, logs)  

---

## 10. Metrics (how we know it works)

- Cold start to first UI &lt; 5s on mid laptop (no model load)  
- First token from tiny/demo model &lt; 10s after load  
- Doctor panel matches CLI `uaii doctor`  
- Self-host: second machine on LAN can chat with token  

---

## 11. Suggested milestones

| Milestone | Deliverable |
|---|---|
| **M0** | Design + OpenAPI for dash server |
| **M1** | Local-only MVP: doctor + model list + generate + logs ✅ (`dashboard/`) |
| **M2** | Linux Docker + start scripts; auth token; settings ✅ (signed Windows zip later) |
| **M3** | Bench UI + OpenAI-compatible `/v1` ✅ (profile viewer still light) |
| **M4** | Polish, signing, company self-host docs (partial) |

---

## 12. Open decisions

1. **Shell:** Tauri / Electron vs browser-only + system tray  
2. **Dash server language:** Go/Rust (single binary) vs Node (faster UI iterate)  
3. **Chat protocol:** custom WS vs OpenAI-compatible from day one  
4. **Bundle OpenBLAS** in desktop zip or require system lib  

**Recommendation:** M1 = browser UI + small local server (single binary), localhost only; M2 = zip/Docker; defer Electron unless offline file UX demands it.

---

## 13. One-line product promise

> **UAII Dashboard:** run and operate local / self-hosted inference with the same honest runtime you get from the CLI — packaged for people who won’t live in a terminal.
