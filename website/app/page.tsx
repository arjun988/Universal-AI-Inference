import Link from "next/link";
import {
  GITHUB_GOOD_FIRST_ISSUE_URL,
  GITHUB_NEW_ISSUE_URL,
  GITHUB_REPO_URL,
} from "@/lib/site";

export default function HomePage() {
  return (
    <main>
      <section className="home-hero" aria-label="Hero">
        <div className="home-hero-rail">
          <div className="hero-copy">
            <p className="hero-kicker">
              <i /> Open source · MIT licensed
            </p>
            <p className="hero-brand">UAII</p>
            <h1 className="hero-title">Any model. One IR. Any hardware.</h1>
            <p className="hero-lede">
              UAII compiles Hugging Face decoder-only checkpoints and GGUF{" "}
              <code>blk.*</code> transformers into a single intermediate representation, then
              executes it on CPU or a native GPU. Weight streaming, a sliding KV cache, and
              top-k expert paging keep 30B-class models inside a laptop memory budget.
              Unsupported layouts stop with a precise error instead of producing wrong output.
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
                {"  "}uaii-backends{"    "}CPU · auto GPU
                {"\n"}
                {"  "}recommended{"      "}cpu  (or cuda if native)
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
          <p className="section-label">Scope</p>
          <h2>What runs today</h2>
          <p className="section-lede">
            Hugging Face decoder-only folders and GGUF <code>blk.*</code> transformers are
            supported paths. The CLI and dashboard select the device automatically: a native GPU
            when this binary was built against the vendor SDK and a device is present, CPU
            otherwise. Generation runs on a memory budget you set, so large checkpoints stream
            from disk rather than exhausting RAM.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>Supported</b>
              <h3>HF and GGUF decoders</h3>
              <p>
                CLI pull and generate, dashboard chat, in-memory K-quants and IQ4_NL, Mixtral
                and Qwen MoE expert paging.
              </p>
            </div>
            <div className="flow-step">
              <b>Laptop scale</b>
              <h3>30B to 500B MoE</h3>
              <p>
                Weight streaming, top-k expert I/O, and a 2048-token KV window bound resident
                memory. Throughput at that scale is limited by disk, not capacity.
              </p>
            </div>
            <div className="flow-step">
              <b>In progress</b>
              <h3>Further architectures</h3>
              <p>Plugin scaffolds for SSM, RWKV, and vision encoders.</p>
            </div>
            <div className="flow-step">
              <b>Docs</b>
              <h3>Start here</h3>
              <p>
                <Link href="/docs/huggingface/">Hugging Face guide</Link> ·{" "}
                <Link href="/docs/getting-started/">Quick start</Link>
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Benchmarks</p>
          <h2>Measured on commodity hardware</h2>
          <p className="section-lede">
            Absolute GEMM throughput, memory bandwidth, and attention latency from{" "}
            <code>uaii_bench</code> v3 on a WSL2 host with an Intel Core i9-14900HX. Every
            figure below reproduces with a single command.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>OpenBLAS</b>
              <h3>425 GFLOP/s</h3>
              <p>1024³ f32 GEMM, and 284 GFLOP/s at 512³. Median of 21 trials.</p>
            </div>
            <div className="flow-step">
              <b>ref-tiled</b>
              <h3>15.1 GFLOP/s</h3>
              <p>Same shape on the built-in kernel, with no vendor BLAS linked.</p>
            </div>
            <div className="flow-step">
              <b>Bandwidth</b>
              <h3>16.6 GB/s</h3>
              <p>STREAM triad across a 256 MiB working set.</p>
            </div>
            <div className="flow-step">
              <b>Method</b>
              <h3>--suite all</h3>
              <p>
                Raw results in <code>benchmarks/results/local_wsl.json</code>.
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
          <h2>How a checkpoint becomes tokens</h2>
          <p className="section-lede">
            Four stages, one runtime path: ingest, plan, execute, integrate.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>01</b>
              <h3>Ingest</h3>
              <p>
                Hugging Face directories, GGUF, Safetensors, and ONNX, MLX, or PyTorch sidecars
                compile into UAII IR.
              </p>
            </div>
            <div className="flow-step">
              <b>02</b>
              <h3>Plan</h3>
              <p>
                Shape validation, operator fusion, memory reuse, and a disk-backed plan cache.
              </p>
            </div>
            <div className="flow-step">
              <b>03</b>
              <h3>Execute</h3>
              <p>
                Quantized GEMM, sliding KV cache, top-k expert paging, and verified device
                selection.
              </p>
            </div>
            <div className="flow-step">
              <b>04</b>
              <h3>Integrate</h3>
              <p>
                CLI, stable C ABI, and Python SDK share identical session semantics.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Capabilities</p>
          <h2>Built for production workloads</h2>
          <p className="section-lede">
            Formats, quantization, backends, and tooling that compose into one runtime.
          </p>
          <div className="bento">
            <div className="bento-item">
              <h3>Hugging Face decoder-only</h3>
              <p>
                Hub folders load directly: strict <code>config.json</code> parsing, sharded
                weights, RoPE and GQA, tied embeddings. Any <code>*ForCausalLM</code>, GPT-2, or
                Mixtral layout that maps to the IR is supported. Vision, SSM, and HF MLA are
                not.
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
                Capability-based import for llama.cpp-style tensors: dense or MoE with packed
                banks split per expert, the DeepSeek MLA expand path, and common Q and K quants
                plus IQ4_NL.
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
                A local or self-hosted console for Hub pulls, streaming chat over HF and GGUF
                models, health checks, benchmarks, an OpenAI-compatible <code>/v1</code>{" "}
                endpoint, and a request queue.
              </p>
              <div className="chip-row">
                <span className="chip">chat</span>
                <span className="chip">HF pull</span>
                <span className="chip">/v1 API</span>
              </div>
            </div>
            <div className="bento-item">
              <h3>In-memory quantized GEMM</h3>
              <p>
                Q4, Q5, Q8, and K-quants execute without unpacking every weight to f32. IQ4_NL
                is supported; other IQ variants stop with a conversion hint.
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
                The <code>IGemm</code> interface dispatches to oneDNN or OpenBLAS when linked,
                and a tiled parallel kernel otherwise. <code>uaii doctor</code> reports the
                active provider.
              </p>
            </div>
            <div className="bento-item">
              <h3>GPU backends with verified capabilities</h3>
              <p>
                <code>--backend auto</code> resolves to CPU unless native CUDA, Metal, ROCm, or
                Vulkan support is compiled in and a device responds. CUDA runs decode attention
                and RoPE on device when enabled, and every host fallback is reported by{" "}
                <code>uaii doctor</code>.
              </p>
            </div>
            <div className="bento-item wide">
              <div>
                <h3>Developer surface</h3>
                <p>
                  One mental model across CLI demos, C embedders, and Python notebooks, with
                  strict defaults, <code>struct_size</code> ABI compatibility, and Chrome-trace
                  profiling.
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
          <p className="section-label">Quick start</p>
          <h2>From clone to tokens</h2>
          <p className="section-lede">
            Build once, then diagnose, pull, and generate. No cloud account required.
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
uaii generate --model ./hf-model --prompt "hi" --preset laptop
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

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Contribute</p>
          <h2>Build the runtime with us</h2>
          <p className="section-lede">
            UAII is MIT licensed and developed in the open. Loaders, kernels, backends,
            benchmarks, and documentation all take outside contributions, and every change
            starts with a GitHub issue so the design is agreed before code is written.
          </p>
          <div className="flow">
            <div className="flow-step">
              <b>01</b>
              <h3>Open an issue</h3>
              <p>
                File a bug report or feature request from the{" "}
                <a href={GITHUB_NEW_ISSUE_URL} target="_blank" rel="noopener noreferrer">
                  issue templates
                </a>
                . Include your OS, compiler, CMake version, and the exact command that failed.
              </p>
            </div>
            <div className="flow-step">
              <b>02</b>
              <h3>Claim the work</h3>
              <p>
                Comment on the issue to take it, or pick one labelled{" "}
                <a href={GITHUB_GOOD_FIRST_ISSUE_URL} target="_blank" rel="noopener noreferrer">
                  good first issue
                </a>
                . A maintainer confirms scope before you start.
              </p>
            </div>
            <div className="flow-step">
              <b>03</b>
              <h3>Build and verify</h3>
              <p>
                Fork, branch, then build with CMake and run <code>uaii doctor</code>,{" "}
                <code>ctest</code>, and <code>clang-format</code> before you push.
              </p>
            </div>
            <div className="flow-step">
              <b>04</b>
              <h3>Send the pull request</h3>
              <p>
                Keep it focused, link the issue, explain why the change is needed, and update
                the docs that describe the behaviour you changed.
              </p>
            </div>
          </div>
          <div className="hero-cta" style={{ marginTop: "1.5rem" }}>
            <Link className="btn btn-primary" href="/docs/contribute/">
              Contribution guide
            </Link>
            <a
              className="btn btn-secondary"
              href={GITHUB_NEW_ISSUE_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open an issue
            </a>
            <a
              className="btn btn-secondary"
              href={GITHUB_GOOD_FIRST_ISSUE_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              Good first issues
            </a>
          </div>
        </div>
      </section>

      <section className="home-section cta-band">
        <div className="rail">
          <div>
            <h2>Start building on UAII</h2>
            <p>
              Build the runtime, pull a Hugging Face decoder or GGUF checkpoint, generate your
              first tokens, then open the operator console. Begin with the{" "}
              <Link href="/docs/huggingface/">Hugging Face path</Link>.
            </p>
          </div>
          <div className="hero-cta" style={{ margin: 0 }}>
            <Link className="btn btn-primary" href="/docs/getting-started/">
              Read the quick start
            </Link>
            <Link className="btn btn-secondary" href="/docs/dashboard/">
              Configure the dashboard
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
