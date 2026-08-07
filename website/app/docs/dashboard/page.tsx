import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Operator UI" };

export default function Page() {
  return (
    <DocLayout
      title="Operator UI"
      active="/docs/dashboard/"
      lede="Local and self-host operator console over the uaii CLI — not a signed ‘download and double-click’ consumer app. Chat with GGUF models (sampling included), manage the model library, run doctor and benches, OpenAI-compatible API. Lives in dashboard/."
    >
      <h2>What you get</h2>
      <ul>
        <li>
          <strong>LLM chat</strong> — real generate via <code>uaii generate</code> / warm{" "}
          <code>uaii chat --jsonl</code>, streaming, system prompt, max tokens, temperature /
          top-p / top-k / repetition penalty, Stop
        </li>
        <li>
          <strong>Any GGUF architecture</strong> with llama.cpp-style <code>blk.*</code> tensors
          (or the built-in tiny demo model)
        </li>
        <li>
          <strong>Models</strong> — upload, list, convert GGUF → IR, delete
        </li>
        <li>
          <strong>Runtime</strong> — <code>uaii doctor --load-plugins</code>
        </li>
        <li>
          <strong>Benchmarks</strong> — run <code>uaii_bench</code>, table + JSON
        </li>
        <li>
          <strong>Settings</strong> — bind, port, binaries, GEMM, threads, auth token
        </li>
        <li>
          <strong>API</strong> — REST <code>/api/*</code> + OpenAI-style{" "}
          <code>/v1/models</code> and <code>/v1/chat/completions</code>
        </li>
        <li>
          <strong>Light / dark</strong> UI (black / white / grey)
        </li>
      </ul>

      <h2>Local (laptop)</h2>
      <p>Build the runtime first, then start the dashboard on loopback:</p>
      <pre>{`# 1) Build UAII
cmake --build build --target uaii uaii_bench --parallel

# Windows App Control often blocks unsigned .exe — use WSL:
#   bash scripts/build_uaii_wsl.sh
#   bash scripts/test_generate_wsl.sh

# 2) Dashboard
cd dashboard
npm run install:all
npm run build && npm start
# → http://127.0.0.1:8787

# Dev (hot UI)
npm run dev   # UI :5174 · API :8787`}</pre>
      <p>
        Windows helper: <code>dashboard/scripts/start-local.ps1</code>. Linux/macOS:{" "}
        <code>dashboard/scripts/start-local.sh</code>.
      </p>

      <h2>Self-host (LAN / server)</h2>
      <p>
        Bind off localhost requires a token — the process <strong>refuses to start</strong> without
        one.
      </p>
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

      <h2>How to configure</h2>
      <p>
        Copy <code>dashboard/uaii-dash.example.json</code> → <code>uaii-dash.json</code>, or use
        environment variables (env wins). Change bind/port in Settings UI, then restart the
        process.
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
                Use <code>0.0.0.0</code> to self-host
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
                <strong>Required</strong> if bind ≠ loopback
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
              <td>Force WSL <code>uaii</code> when App Control blocks .exe</td>
            </tr>
            <tr>
              <td>
                <code>UAII_MAX_CONTEXT</code>
              </td>
              <td>
                <code>0</code>
              </td>
              <td>Session max context for chat workers</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Typical flows</h2>
      <ol>
        <li>
          <strong>Smoke chat</strong> — Chat → Tiny demo / <code>uaii-tiny-demo</code> → Run
          (streams tokens)
        </li>
        <li>
          <strong>Real GGUF</strong> — Models → Import <code>.gguf</code> → Chat → select model →
          set max tokens → Run
        </li>
        <li>
          <strong>Health</strong> — Runtime → Run doctor
        </li>
        <li>
          <strong>Perf</strong> — Benchmarks → Run microbench
        </li>
        <li>
          <strong>Team API</strong> — self-host + Bearer token on{" "}
          <code>/v1/chat/completions</code>
        </li>
      </ol>

      <h2>OpenAI-compatible clients</h2>
      <pre>{`curl -s http://SERVER:8787/v1/models \\
  -H "Authorization: Bearer $UAII_DASH_TOKEN"

curl -s http://SERVER:8787/v1/chat/completions \\
  -H "Authorization: Bearer $UAII_DASH_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"uaii-tiny-demo","messages":[{"role":"user","content":"hi"}]}'

# Or a local file: "uaii-file-my-model.gguf"`}</pre>

      <h2>Related</h2>
      <ul>
        <li>
          Repo folder <code>dashboard/README.md</code> and PRD{" "}
          <code>docs/prd-dashboard.md</code>
        </li>
        <li>
          <Link href="/docs/cli/">CLI</Link> — <code>generate</code> / <code>chat</code>
        </li>
        <li>
          <Link href="/docs/features/">Features</Link> — GGUF multi-arch details
        </li>
        <li>
          <Link href="/docs/configuration/">Configuration</Link> — runtime + dashboard env
        </li>
      </ul>
    </DocLayout>
  );
}
