import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Configuration" };

export default function Page() {
  return (
    <DocLayout
      title="Configuration"
      active="/docs/configuration/"
      lede="Control logging, plugins, GEMM, backends, and session policy through CMake, TOML, environment variables, and API options."
    >
      <h2>Default config file</h2>
      <p>
        Load <code>configs/uaii.toml</code> (or pass <code>--config path</code>). Environment
        overlays win over file values.
      </p>

      <h2>Environment variables</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Purpose</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>UAII_LOG_LEVEL</strong>
              </td>
              <td>
                <code>trace</code> · <code>debug</code> · <code>info</code> · <code>warn</code>{" "}
                · <code>error</code>
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_LOG_COLOR</strong>
              </td>
              <td>Enable / disable ANSI color in logs</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_PLUGIN__DIRS</strong>
              </td>
              <td>Comma-separated plugin search directories</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_NUM_THREADS</strong>
              </td>
              <td>CPU kernel parallelism</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_GEMM</strong>
              </td>
              <td>Preferred GEMM provider hint when multiple are built</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_MAX_LAYERS</strong>
              </td>
              <td>
                Cap GGUF transformer layers (<code>0</code> = unlimited up to hard max)
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_BACKEND</strong>
              </td>
              <td>
                Dashboard / env override for device: <code>auto</code> (default),{" "}
                <code>cpu</code>, <code>cuda</code>, …
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_CAPI_PATH</strong>
              </td>
              <td>Path to <code>uaii_capi</code> shared library for Python / ctypes</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>CMake options</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Option</th>
              <th>Default</th>
              <th>Meaning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>UAII_BUILD_TESTS</code>
              </td>
              <td>ON</td>
              <td>Unit / smoke tests</td>
            </tr>
            <tr>
              <td>
                <code>UAII_BUILD_PLUGINS</code>
              </td>
              <td>ON</td>
              <td>Example plugins</td>
            </tr>
            <tr>
              <td>
                <code>UAII_BUILD_PYTHON</code>
              </td>
              <td>OFF</td>
              <td>pybind11 extension</td>
            </tr>
            <tr>
              <td>
                <code>UAII_WITH_ONEDNN</code> / <code>OPENBLAS</code>
              </td>
              <td>OFF</td>
              <td>Competitive CPU GEMM</td>
            </tr>
            <tr>
              <td>
                <code>UAII_WITH_CUDA</code>
              </td>
              <td>OFF</td>
              <td>CUDA + cuBLASLt device path</td>
            </tr>
            <tr>
              <td>
                <code>UAII_WITH_METAL</code> / <code>VULKAN</code> / <code>WEBGPU</code> /{" "}
                <code>ROCM</code>
              </td>
              <td>OFF</td>
              <td>Optional GPU backends</td>
            </tr>
            <tr>
              <td>
                <code>UAII_WITH_SENTENCEPIECE</code>
              </td>
              <td>OFF</td>
              <td>SentencePiece tokenizer</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Session options</h2>
      <p>
        CLI <code>generate</code> / <code>chat</code> apply the laptop preset and{" "}
        <code>--backend auto</code>. The C API 0.3.0 struct is smaller: library default{" "}
        <code>backend = &quot;cpu&quot;</code>, no <code>kv_window</code> field yet.
      </p>
      <ul>
        <li>
          <code>backend_name</code> / C <code>backend</code> — pass <code>auto</code> to probe
          a native GPU; otherwise <code>cpu</code> / <code>cuda</code> / <code>metal</code> / …
        </li>
        <li>
          <code>compute_dtype</code> — F32 or F16 policy
        </li>
        <li>
          <code>keep_quantized_weights</code> — keep GGUF packs for quant GEMM
        </li>
        <li>
          <code>max_context</code> — generate length bound (0 = from graph metadata). With
          sliding KV this is a soft cap, not the RAM allocation
        </li>
        <li>
          C++ / CLI: <code>kv_window</code> / <code>kv_sink</code> — resident KV window
          (laptop 2048 / 4 sinks). Not in C API 0.3.0
        </li>
        <li>
          C++ / CLI: <code>pin_layers</code>, <code>expert_cache_bytes</code> — storage-first
          trunk / MoE LRU
        </li>
        <li>
          <code>enable_fusion</code>, <code>enable_memory_reuse</code>,{" "}
          <code>enable_streaming</code>, <code>enable_profiler</code>
        </li>
        <li>
          <code>weight_init</code> — default <strong>none</strong> (fail closed)
        </li>
        <li>
          <code>weights_dir</code> / <code>weights_sandbox</code> — weight resolution + path
          sandbox
        </li>
      </ul>
      <p>
        C API callers must set <code>struct_size = sizeof(uaii_session_options)</code> after{" "}
        <code>uaii_session_options_init</code>.
      </p>

      <h2>CLI globals</h2>
      <pre>{`uaii --config configs/uaii.toml --log-level info --load-plugins doctor`}</pre>

      <h2>Dashboard configuration</h2>
      <p>
        The operator UI reads <code>dashboard/uaii-dash.json</code> (see{" "}
        <code>uaii-dash.example.json</code>) and these environment variables. Full guide:{" "}
        <Link href="/docs/dashboard/">Dashboard</Link>.
      </p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Purpose</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>UAII_DASH_BIND</strong>
              </td>
              <td>
                <code>127.0.0.1</code> local · <code>0.0.0.0</code> self-host
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_DASH_PORT</strong>
              </td>
              <td>HTTP port (default <code>8787</code>)</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_DASH_TOKEN</strong>
              </td>
              <td>Bearer token — required for non-loopback bind</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_BIN</strong> / <strong>UAII_BENCH_BIN</strong>
              </td>
              <td>Override detected CLI binaries</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_MODEL_DIR</strong>
              </td>
              <td>Model library directory</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_USE_WSL</strong> / <strong>UAII_WSL_BIN</strong>
              </td>
              <td>Use WSL <code>uaii</code> when Windows blocks unsigned .exe</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_BACKEND</strong>
              </td>
              <td>
                Device for chat/run: <code>auto</code> (default), <code>cpu</code>,{" "}
                <code>cuda</code>, …
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <pre>{`# Local
cd dashboard && npm start

# Self-host
UAII_DASH_BIND=0.0.0.0 UAII_DASH_TOKEN=secret npm start`}</pre>
    </DocLayout>
  );
}
