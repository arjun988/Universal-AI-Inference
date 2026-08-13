import Link from "next/link";
import { GITHUB_REPO_URL } from "@/lib/site";

export default function HomePage() {
  return (
    <main>
      <section className="home-hero" aria-label="Hero">
        <div className="home-hero-rail">
          <div className="hero-copy">
            <p className="hero-kicker">
              <i /> Open source · HF decoder-ready
            </p>
            <p className="hero-brand">UAII</p>
            <h1 className="hero-title">Any model → UAII IR → any hardware.</h1>
            <p className="hero-lede">
              Universal inference runtime — <strong>usable today</strong> for Hugging Face
              decoder-only LLMs and GGUF <code>blk.*</code> transformers. One IR, honest
              backends, CLI + operator dashboard. More architectures land as plugins; we fail
              closed instead of silently wrong.
            </p>
            <div className="hero-cta">
              <Link className="btn btn-primary" href="/docs/getting-started/">
                Get started
              </Link>
              <Link className="btn btn-secondary" href="/docs/huggingface/">
                Hugging Face
              </Link>
              <a
                className="btn btn-secondary"
                href={GITHUB_REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
              >
                View on GitHub
              </a>
            </div>
            <div className="hero-meta">
              <span>C++17 · CMake</span>
              <span>C API 0.3.0</span>
              <span>MIT License</span>
            </div>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="terminal">
              <div className="terminal-bar">
                <span />
                <span />
                <span />
                <label>uaii · doctor</label>
              </div>
              <pre>
                <span className="dim">$ </span>uaii doctor
                {"\n\n"}
                Modules
                {"\n"}
                {"  "}uaii-kernels{"     "}GEMM=ref-tiled
                {"\n"}
                {"  "}uaii-backends{"    "}CPU real · CUDA optional
                {"\n"}
                {"  "}uaii-loaders{"     "}HF dirs · GGUF · Safetensors
                {"\n\n"}
                <span className="dim">$ </span>uaii pull org/model
                {"\n"}
                <span className="dim">$ </span>uaii generate --model ./hf-dir --prompt &quot;hi&quot;
                {"\n"}
                {"  "}
                <span className="ok">ok</span>
                {"  "}decoder-only HF · GGUF blk.*
                {"\n\n"}
                <span className="dim">$ </span>cd dashboard && npm start
                {"\n"}
                {"  "}
                <span className="dim">http://127.0.0.1:8787 · HF + GGUF chat</span>
              </pre>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Readiness</p>
          <h2>Ship decoder-only now; widen later</h2>
          <p className="section-lede">
            Tell users the truth: HF CausalLM allowlist + GGUF decoders work today. Vision,
            SSM, RWKV, and marketplace are development — not silent “universal” claims.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>Ready</b>
              <h3>HF + GGUF decoders</h3>
              <p>CLI pull/generate, dashboard chat, memory presets, IQ4_NL / Q-quants.</p>
            </div>
            <div className="flow-step">
              <b>Partial</b>
              <h3>CUDA / MoE dial</h3>
              <p>Decode Attention on CUDA when enabled; expert stream / LRU still deepening.</p>
            </div>
            <div className="flow-step">
              <b>Building</b>
              <h3>Other arches</h3>
              <p>Plugin scaffolds for SSM / RWKV / vision; user arch picker later.</p>
            </div>
            <div className="flow-step">
              <b>Docs</b>
              <h3>Hugging Face</h3>
              <p>
                <Link href="/docs/huggingface/">HF decoder guide</Link> ·{" "}
                <Link href="/docs/getting-started/">Quick start</Link>
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Benchmarks</p>
          <h2>Absolute numbers, not strawmen</h2>
          <p className="section-lede">
            Local WSL2 · i9-14900HX · OpenBLAS + ref · <code>uaii_bench</code> v3 — absolute
            GFLOP/s, bandwidth, attention.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>OpenBLAS</b>
              <h3>425 GFLOP/s</h3>
              <p>1024³ f32 GEMM · 284 GFLOP/s at 512³ · median of 21.</p>
            </div>
            <div className="flow-step">
              <b>ref-tiled</b>
              <h3>15.1 GFLOP/s</h3>
              <p>Same shape — always-on baseline without vendor BLAS.</p>
            </div>
            <div className="flow-step">
              <b>Bandwidth</b>
              <h3>16.6 GB/s</h3>
              <p>STREAM triad on ~256 MiB working set.</p>
            </div>
            <div className="flow-step">
              <b>Method</b>
              <h3>--suite all</h3>
              <p>
                JSON: <code>benchmarks/results/local_wsl.json</code>
              </p>
            </div>
          </div>
          <div className="hero-cta" style={{ marginTop: "1.5rem" }}>
            <Link className="btn btn-secondary" href="/docs/benchmarks/">
              Full methodology
            </Link>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Pipeline</p>
          <h2>What it does</h2>
          <p className="section-lede">
            One runtime path from model file to tokens — ingest, plan, execute, integrate.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>01</b>
              <h3>Ingest</h3>
              <p>HF directories, GGUF, Safetensors, ONNX/MLX/PyTorch sidecars → UAII IR.</p>
            </div>
            <div className="flow-step">
              <b>02</b>
              <h3>Plan</h3>
              <p>Validate shapes, fuse ops, reuse memory, cache plans to disk.</p>
            </div>
            <div className="flow-step">
              <b>03</b>
              <h3>Execute</h3>
              <p>Quant GEMM, KV-cache generate, streaming weights, honest GPU caps.</p>
            </div>
            <div className="flow-step">
              <b>04</b>
              <h3>Ship</h3>
              <p>CLI, stable C ABI, Python SDK — same session semantics everywhere.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Capabilities</p>
          <h2>Built for real inference stacks</h2>
          <p className="section-lede">
            Not a toy demo loop — formats, quants, backends, and tooling that compose.
          </p>
          <div className="bento">
            <div className="bento-item">
              <h3>Hugging Face decoder-only</h3>
              <p>
                First-class Hub folders: strict <code>config.json</code>, shards, RoPE/GQA,
                tied emb. Allowlisted CausalLM families — not every Hub card.
              </p>
              <div className="chip-row">
                <span className="chip">uaii pull</span>
                <span className="chip">Llama</span>
                <span className="chip">Qwen</span>
                <span className="chip">Gemma</span>
              </div>
            </div>
            <div className="bento-item">
              <h3>GGUF blk.* transformers</h3>
              <p>
                Capability-based import for llama.cpp-style tensors: dense or MoE, DeepSeek
                MLA expand path, common Q/K quants + IQ4_NL.
              </p>
              <div className="chip-row">
                <span className="chip">llama</span>
                <span className="chip">qwen2/3</span>
                <span className="chip">mistral</span>
                <span className="chip">MLA</span>
              </div>
            </div>
            <div className="bento-item">
              <h3>Operator dashboard</h3>
              <p>
                Local and self-host UI: HF Hub pull, GGUF/HF chat with streaming, doctor,
                benches, OpenAI-compatible <code>/v1</code>, request queue.
              </p>
              <div className="chip-row">
                <span className="chip">chat</span>
                <span className="chip">HF pull</span>
                <span className="chip">/v1 API</span>
              </div>
            </div>
            <div className="bento-item">
              <h3>In-memory quant GEMM</h3>
              <p>
                Run Q4/Q5/Q8 and K-quants without unpacking every weight to f32. IQ4_NL
                supported; other IQ* fail closed with a convert hint.
              </p>
              <div className="chip-row">
                <span className="chip">Q4_0</span>
                <span className="chip">Q5_K</span>
                <span className="chip">IQ4_NL</span>
              </div>
            </div>
            <div className="bento-item">
              <h3>Competitive CPU GEMM</h3>
              <p>
                IGemm with oneDNN / OpenBLAS when linked, tiled parallel ref otherwise. Doctor
                prints the active provider.
              </p>
            </div>
            <div className="bento-item">
              <h3>GPU backends, honest caps</h3>
              <p>
                CUDA decode Attention/RoPE when enabled; Metal/Vulkan/WebGPU/ROCm with
                host-fallback never silent — <code>uaii doctor</code> tells the truth.
              </p>
            </div>
            <div className="bento-item wide">
              <div>
                <h3>Developer surface</h3>
                <p>
                  One mental model across CLI demos, C embedders, and Python notebooks —
                  fail-closed defaults, struct_size ABI, chrome-trace profiling.
                </p>
              </div>
              <div className="matrix-wrap">
                <table className="matrix">
                  <thead>
                    <tr>
                      <th>Surface</th>
                      <th>Entry</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <strong>CLI</strong>
                      </td>
                      <td>
                        <code>generate · chat · run · convert</code>
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <strong>Dashboard</strong>
                      </td>
                      <td>
                        <code>cd dashboard && npm start</code>
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <strong>C API</strong>
                      </td>
                      <td>
                        <code>uaii_capi</code> · 0.3.0
                      </td>
                    </tr>
                    <tr>
                      <td>
                        <strong>Python</strong>
                      </td>
                      <td>
                        <code>pip install -e bindings/python</code>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Quick taste</p>
          <h2>From clone to tokens</h2>
          <p className="section-lede">
            Build once, then diagnose, convert, and run — no special cloud required.
          </p>
          <div className="code-grid">
            <div className="code-card">
              <header>
                Build
                <span>cmake</span>
              </header>
              <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel`}</pre>
            </div>
            <div className="code-card">
              <header>
                Run
                <span>cli</span>
              </header>
              <pre>{`uaii doctor
uaii pull org/model --outdir ./hf-model
uaii generate --model ./hf-model --prompt "hi"
uaii generate --model model.gguf --prompt "hi"`}</pre>
            </div>
            <div className="code-card">
              <header>
                Dashboard
                <span>ui</span>
              </header>
              <pre>{`cd dashboard
npm run install:all
npm run build && npm start
# → http://127.0.0.1:8787`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section cta-band">
        <div className="rail">
          <div>
            <h2>Start building on UAII</h2>
            <p>
              Build once, then pull an HF decoder or GGUF, generate, and open the operator UI.
              HF path: <Link href="/docs/huggingface/">Hugging Face</Link>.
            </p>
          </div>
          <div className="hero-cta" style={{ margin: 0 }}>
            <Link className="btn btn-primary" href="/docs/getting-started/">
              Read the quick start
            </Link>
            <Link className="btn btn-secondary" href="/docs/dashboard/">
              Configure dashboard
            </Link>
            <a
              className="btn btn-secondary"
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              Star on GitHub
            </a>
          </div>
        </div>
      </section>
    </main>
  );
}
