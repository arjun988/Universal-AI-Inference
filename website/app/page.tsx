import Link from "next/link";
import { GITHUB_REPO_URL } from "@/lib/site";

export default function HomePage() {
  return (
    <main>
      <section className="home-hero" aria-label="Hero">
        <div className="home-hero-rail">
          <div className="hero-copy">
            <p className="hero-kicker">
              <i /> Open source inference runtime
            </p>
            <p className="hero-brand">UAII</p>
            <h1 className="hero-title">Any model → UAII IR → any hardware.</h1>
            <p className="hero-lede">
              The modular execution layer for AI inference. Load GGUF, ONNX, Safetensors, and
              more into one IR — then run on CPU or GPU through CLI, C, and Python.
            </p>
            <div className="hero-cta">
              <Link className="btn btn-primary" href="/docs/getting-started/">
                Get started
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
                {"  "}uaii-loaders{"     "}GGUF · ONNX · Safetensors
                {"\n\n"}
                <span className="dim">$ </span>uaii run --demo gguf
                {"\n"}
                {"  "}
                <span className="ok">ok</span>
                {"  "}prefill + decode · KV cache live
                {"\n\n"}
                <span className="dim">$ </span>uaii convert model.gguf -o model.uaii.json
                {"\n"}
                {"  "}
                <span className="dim">wrote UAII IR</span>
              </pre>
            </div>
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
              <p>GGUF, Safetensors, ONNX, MLX, PyTorch sidecars → UAII IR.</p>
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
              <h3>Llama-family GGUF</h3>
              <p>
                Full blk.* stacks with RoPE, Attention + KV, SwiGLU, and fail-closed arch /
                quant gates.
              </p>
              <div className="chip-row">
                <span className="chip">llama</span>
                <span className="chip">mistral</span>
                <span className="chip">qwen2</span>
                <span className="chip">phi3</span>
              </div>
            </div>
            <div className="bento-item">
              <h3>In-memory quant GEMM</h3>
              <p>
                Run Q4/Q5/Q8 and K-quants without unpacking every weight to f32. Keep packs
                under session policy.
              </p>
              <div className="chip-row">
                <span className="chip">Q4_0</span>
                <span className="chip">Q5_K</span>
                <span className="chip">Q8_0</span>
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
                CUDA (cuBLASLt), Metal, Vulkan, WebGPU, ROCm — device memory when enabled;
                host-fallback never silent.
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
                        <code>uaii run · convert · tokenize</code>
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
uaii run --demo toy_mlp
uaii run --demo gguf`}</pre>
            </div>
            <div className="code-card">
              <header>
                Convert
                <span>ir</span>
              </header>
              <pre>{`uaii convert model.gguf \\
  -o model.uaii.json
uaii tokenize encode hello`}</pre>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section cta-band">
        <div className="rail">
          <div>
            <h2>Start building on UAII</h2>
            <p>
              Quick start covers build, CLI smoke, Python bundling, and session configuration.
            </p>
          </div>
          <div className="hero-cta" style={{ margin: 0 }}>
            <Link className="btn btn-primary" href="/docs/getting-started/">
              Read the quick start
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
