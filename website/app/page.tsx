import Link from "next/link";
import { GITHUB_REPO_URL } from "@/lib/site";

export default function HomePage() {
  return (
    <main>
      <section className="home-hero" aria-label="Hero">
        <div className="home-hero-rail">
          <div className="hero-copy">
            <p className="hero-kicker">
              <i /> MIT · C++17 · CMake
            </p>
            <h1 className="hero-brand">Universal AI Inference</h1>
            <p className="hero-lede">
              One runtime for Hugging Face decoder-only folders and GGUF{" "}
              <code>blk.*</code> models. The same paging dial runs on a laptop or a phone.
              Unsupported layouts stop with an error.
            </p>
            <div className="hero-cta">
              <Link className="btn btn-primary" href="/docs/getting-started/">
                Quick start
              </Link>
              <Link className="btn btn-secondary" href="/docs/memory/">
                Memory dial
              </Link>
              <a
                className="btn btn-secondary"
                href={GITHUB_REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </div>
            <div className="hero-meta">
              <span>Package 0.3.1</span>
              <span>C API 0.4.0</span>
              <span>Preset laptop by default</span>
            </div>
          </div>

          <div className="hero-visual" aria-hidden="true">
            <div className="terminal">
              <div className="terminal-bar">
                <span />
                <span />
                <span />
                <label>uaii</label>
              </div>
              <pre>
                <span className="dim">$ </span>uaii doctor
                {"\n"}
                {"  "}c_api{"          "}0.4.0
                {"\n"}
                {"  "}phone_budget{"    "}2048 MiB · staging x1
                {"\n"}
                {"  "}recommended{"     "}cpu
                {"\n\n"}
                <span className="dim">$ </span>uaii generate --model model.q4_k.gguf \
                {"\n"}
                {"    "}--preset phone --prompt &quot;hi&quot;
                {"\n\n"}
                <span className="dim">$ </span>uaii generate --model ./hf-dir \
                {"\n"}
                {"    "}--preset laptop --prompt &quot;hi&quot;
              </pre>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Runtime</p>
          <h2>What it runs</h2>
          <div className="flow">
            <div className="flow-step">
              <b>Models</b>
              <h3>HF decoders and GGUF</h3>
              <p>
                Llama-like, Qwen, Mistral/Mixtral, Gemma, and GPT-2. GGUF adds packed MoE
                experts and MLA. Vision, SSM, and HF MLA stay out.
              </p>
            </div>
            <div className="flow-step">
              <b>Laptop</b>
              <h3>~8 GiB, KV 2048</h3>
              <p>
                Default preset. Pins a trunk that fits, prefetches with two staging buffers,
                and pages the rest. Shrinks to free RAM only when 8 GiB does not fit.
              </p>
            </div>
            <div className="flow-step">
              <b>Phone</b>
              <h3>~2 GiB, KV 512</h3>
              <p>
                One staging buffer, 4 threads. The budget is half of the RAM the OS can still
                give the process. Q4_K stays on the CPU quant GEMM.
              </p>
            </div>
            <div className="flow-step">
              <b>Embed</b>
              <h3>C API 0.4.0</h3>
              <p>
                The C API applies the same dial as the CLI. Phone apps resolve a preset, then
                cancel, background, or drop KV on a memory warning.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <div className="rail">
          <p className="section-label">Start</p>
          <h2>Build, then generate</h2>
          <div className="code-grid">
            <div className="code-card">
              <header>
                Desktop
                <span>cmake</span>
              </header>
              <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/libs/uaii-cli/uaii doctor`}</pre>
            </div>
            <div className="code-card">
              <header>
                Generate
                <span>cli</span>
              </header>
              <pre>{`uaii pull org/model --outdir ./hf-dir
uaii generate --model ./hf-dir \\
  --preset laptop --prompt "hi"
uaii generate --model m.q4_k.gguf \\
  --preset phone --prompt "hi"`}</pre>
            </div>
            <div className="code-card">
              <header>
                Phone
                <span>ndk / ios</span>
              </header>
              <pre>{`cmake -S examples/mobile -B build-android \\
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \\
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24
cmake --build build-android --target uaii_phone`}</pre>
            </div>
          </div>
          <div className="hero-cta" style={{ marginTop: "1.25rem" }}>
            <Link className="btn btn-secondary" href="/docs/mobile/">
              Android and iOS
            </Link>
            <Link className="btn btn-secondary" href="/docs/c-api/">
              C API
            </Link>
            <Link className="btn btn-secondary" href="/docs/benchmarks/">
              Benchmarks
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
