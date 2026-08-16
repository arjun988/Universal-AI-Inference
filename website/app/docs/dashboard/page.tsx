import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Operator UI" };

export default function Page() {
  return (
    <DocLayout
      title="Operator UI"
      active="/docs/dashboard/"
      lede="A web console for the uaii CLI, running locally or self-hosted from dashboard/. Chat with Hugging Face decoder folders and GGUF blk.* models, pull from the Hub, and manage device and memory settings."
    >
      <h2>What it provides</h2>
      <dl className="deflist">
        <div>
          <dt>Chat</dt>
          <dd>
            Generation through <code>uaii generate</code> or a warm{" "}
            <code>uaii chat --jsonl</code> worker, with streaming output, system prompt, token
            limit, temperature, top-p, top-k, repetition penalty, cancellation, and a FIFO
            request queue.
          </dd>
        </div>
        <div>
          <dt>Hugging Face</dt>
          <dd>
            Hub pulls and folder imports for decoder-only CausalLM checkpoints, which require a
            current <code>uaii</code> build. See{" "}
            <Link href="/docs/huggingface/">Hugging Face</Link>.
          </dd>
        </div>
        <div>
          <dt>GGUF</dt>
          <dd>
            Dense and MoE <code>blk.*</code> transformers including the MLA expand path, plus a
            built-in demo model for smoke tests.
          </dd>
        </div>
        <div>
          <dt>Model library</dt>
          <dd>Upload GGUF files, list Hugging Face and GGUF models, convert, and delete.</dd>
        </div>
        <div>
          <dt>Runtime health</dt>
          <dd>
            Run <code>uaii doctor --load-plugins</code> from the browser.
          </dd>
        </div>
        <div>
          <dt>Benchmarks</dt>
          <dd>
            Run <code>uaii_bench</code> and read the results as a table or raw JSON.
          </dd>
        </div>
        <div>
          <dt>Settings</dt>
          <dd>
            Bind address, port, binary paths, device selection defaulting to{" "}
            <strong>auto</strong>, GEMM provider, thread count, and auth token.
          </dd>
        </div>
        <div>
          <dt>API</dt>
          <dd>
            A REST surface under <code>/api/*</code> plus OpenAI-compatible{" "}
            <code>/v1/models</code> and <code>/v1/chat/completions</code>.
          </dd>
        </div>
        <div>
          <dt>Themes</dt>
          <dd>Light and dark, following your system preference by default.</dd>
        </div>
      </dl>

      <h2>Running locally</h2>
      <p>Build the runtime first, then start the dashboard on loopback:</p>
      <pre>{`# 1) Build UAII
cmake --build build --target uaii uaii_bench --parallel

# Windows App Control frequently blocks unsigned .exe files. Build under WSL:
#   bash scripts/build_uaii_wsl.sh
#   bash scripts/test_generate_wsl.sh

# 2) Dashboard
cd dashboard
npm run install:all
npm run build && npm start
# → http://127.0.0.1:8787

# Development, with hot reload
npm run dev   # UI :5174 · API :8787`}</pre>
      <p>
        Helper scripts: <code>dashboard/scripts/start-local.ps1</code> on Windows and{" "}
        <code>dashboard/scripts/start-local.sh</code> on Linux and macOS.
      </p>

      <h2>Self-hosting on a LAN or server</h2>
      <div className="callout">
        <span className="callout-label">Required</span>
        <p>
          Binding to any address other than loopback requires a token. The process{" "}
          <strong>refuses to start</strong> without one.
        </p>
      </div>
      <pre>{`export UAII_DASH_TOKEN=$(openssl rand -hex 16)
export UAII_DASH_BIND=0.0.0.0
cd dashboard
./scripts/start-host.sh
# Open http://SERVER:8787 and unlock with the token`}</pre>
      <p>Docker:</p>
      <pre>{`export UAII_DASH_TOKEN=$(openssl rand -hex 16)
export UAII_HOST_BIN=$PWD/../build/libs/uaii-cli/uaii
export UAII_HOST_BENCH=$PWD/../build/benchmarks/uaii_bench
cd dashboard
docker compose up --build`}</pre>

      <h2>Configuration</h2>
      <p>
        Copy <code>dashboard/uaii-dash.example.json</code> to <code>uaii-dash.json</code>, or set
        environment variables, which take precedence over the file. The default device is{" "}
        <code>&quot;backend&quot;: &quot;auto&quot;</code>; an older config pinned to{" "}
        <code>&quot;cpu&quot;</code> keeps that value until you reset it in Settings. Bind address
        and port changes take effect after restarting the process.
      </p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Default</th>
              <th>Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>UAII_BIN</code>
              </td>
              <td>auto-detect build tree</td>
              <td>Path to <code>uaii</code></td>
            </tr>
            <tr>
              <td>
                <code>UAII_BENCH_BIN</code>
              </td>
              <td>auto-detect</td>
              <td>Path to <code>uaii_bench</code></td>
            </tr>
            <tr>
              <td>
                <code>UAII_MODEL_DIR</code>
              </td>
              <td>
                <code>dashboard/models</code>
              </td>
              <td>Model library directory</td>
            </tr>
            <tr>
              <td>
                <code>UAII_DASH_BIND</code>
              </td>
              <td>
                <code>127.0.0.1</code>
              </td>
              <td>
                Set to <code>0.0.0.0</code> to self-host
              </td>
            </tr>
            <tr>
              <td>
                <code>UAII_DASH_PORT</code>
              </td>
              <td>
                <code>8787</code>
              </td>
              <td>HTTP port</td>
            </tr>
            <tr>
              <td>
                <code>UAII_DASH_TOKEN</code>
              </td>
              <td>(empty)</td>
              <td>
                <strong>Required</strong> when the bind address is not loopback
              </td>
            </tr>
            <tr>
              <td>
                <code>UAII_GEMM</code> / <code>UAII_NUM_THREADS</code>
              </td>
              <td>—</td>
              <td>Passed through to CLI jobs</td>
            </tr>
            <tr>
              <td>
                <code>UAII_USE_WSL</code> / <code>UAII_WSL_BIN</code>
              </td>
              <td>auto on Windows</td>
              <td>
                Force the WSL <code>uaii</code> binary when Windows App Control blocks the .exe
              </td>
            </tr>
            <tr>
              <td>
                <code>UAII_BACKEND</code>
              </td>
              <td>
                <code>auto</code>
              </td>
              <td>
                Device for chat and run jobs. <code>auto</code> uses a native GPU when available,
                or pin <code>cpu</code>, <code>cuda</code>, and so on
              </td>
            </tr>
            <tr>
              <td>
                <code>UAII_MAX_CONTEXT</code>
              </td>
              <td>
                <code>0</code>
              </td>
              <td>
                Maximum session context for chat workers. The sliding KV window from the CLI
                preset still applies
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Common workflows</h2>
      <ol className="steps">
        <li>
          <strong>Verify the install</strong>
          Open Chat, select the <code>uaii-tiny-demo</code> model, and run. Tokens stream
          immediately with no download.
        </li>
        <li>
          <strong>Run a Hugging Face model</strong>
          Open Models, choose Pull from Hub or Import HF folder, then switch to Chat, select the
          folder, and run with paging and device selection applied.
        </li>
        <li>
          <strong>Run a GGUF model</strong>
          Open Models, import a <code>.gguf</code> file, then switch to Chat, select it, set a
          token limit, and run.
        </li>
        <li>
          <strong>Check runtime health</strong>
          Open Runtime and run doctor.
        </li>
        <li>
          <strong>Measure performance</strong>
          Open Benchmarks and run the microbenchmark suite.
        </li>
        <li>
          <strong>Serve a team</strong>
          Self-host, then issue a Bearer token for <code>/v1/chat/completions</code>.
        </li>
      </ol>

      <h2>OpenAI-compatible clients</h2>
      <pre>{`curl -s http://SERVER:8787/v1/models \\
  -H "Authorization: Bearer $UAII_DASH_TOKEN"

curl -s http://SERVER:8787/v1/chat/completions \\
  -H "Authorization: Bearer $UAII_DASH_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"uaii-tiny-demo","messages":[{"role":"user","content":"hi"}]}'

# Reference a local file as "uaii-file-my-model.gguf"`}</pre>

      <h2>Related</h2>
      <ul>
        <li>
          <code>dashboard/README.md</code> and <code>docs/prd-dashboard.md</code> in the
          repository
        </li>
        <li>
          <Link href="/docs/cli/">CLI</Link> — the <code>generate</code> and <code>chat</code>{" "}
          commands the console drives
        </li>
        <li>
          <Link href="/docs/features/">Features</Link> — GGUF architecture coverage
        </li>
        <li>
          <Link href="/docs/configuration/">Configuration</Link> — runtime and dashboard
          environment variables
        </li>
      </ul>
    </DocLayout>
  );
}
