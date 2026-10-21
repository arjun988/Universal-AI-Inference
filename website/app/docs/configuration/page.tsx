import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Configuration" };

export default function Page() {
  return (
    <DocLayout
      title="Configuration"
      active="/docs/configuration/"
      lede="Control logging, plugins, GEMM providers, backends, and session policy through CMake options, a TOML file, environment variables, and API options."
    >
      <h2>Configuration file</h2>
      <p>
        The runtime loads <code>configs/uaii.toml</code>, or the file you pass to{" "}
        <code>--config</code>. Environment variables override file values.
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
                Verbosity: <code>trace</code>, <code>debug</code>, <code>info</code>,{" "}
                <code>warn</code>, or <code>error</code>
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_LOG_COLOR</strong>
              </td>
              <td>Enable or disable ANSI colour in log output</td>
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
              <td>Preferred GEMM provider when more than one is linked</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_MAX_LAYERS</strong>
              </td>
              <td>
                Cap the number of GGUF transformer layers loaded. <code>0</code> means unlimited,
                up to the hard maximum
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_BACKEND</strong>
              </td>
              <td>
                Device override for the dashboard and CLI jobs: <code>auto</code> by default, or{" "}
                <code>cpu</code>, <code>cuda</code>, and so on
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_CAPI_PATH</strong>
              </td>
              <td>
                Path to the <code>uaii_capi</code> shared library, used by the Python ctypes loader
              </td>
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
              <td>Unit and smoke tests</td>
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
              <td>Native pybind11 extension for the Python SDK</td>
            </tr>
            <tr>
              <td>
                <code>UAII_WITH_ONEDNN</code> / <code>OPENBLAS</code>
              </td>
              <td>OFF</td>
              <td>Link a vendor BLAS for faster CPU GEMM</td>
            </tr>
            <tr>
              <td>
                <code>UAII_WITH_CUDA</code>
              </td>
              <td>OFF</td>
              <td>CUDA device path with cuBLASLt</td>
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
              <td>SentencePiece tokenizer support</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Session options</h2>
      <p>
        CLI <code>generate</code> and <code>chat</code> apply the laptop preset and{" "}
        <code>--backend auto</code>. C API 0.4.0 exposes the same dial. Call{" "}
        <code>uaii_resolve_memory_preset</code>, then{" "}
        <code>uaii_session_options_apply_preset</code>. A struct built against 0.3.0 still works:
        new fields are appended.
      </p>
      <dl className="deflist">
        <div>
          <dt>
            <code>backend_name</code>
          </dt>
          <dd>
            Named <code>backend</code> in C. Pass <code>auto</code> to probe for a native GPU, or
            name a device such as <code>cpu</code>, <code>cuda</code>, or <code>metal</code>
          </dd>
        </div>
        <div>
          <dt>
            <code>compute_dtype</code>
          </dt>
          <dd>F32 or F16 compute policy</dd>
        </div>
        <div>
          <dt>
            <code>keep_quantized_weights</code>
          </dt>
          <dd>Retain GGUF blocks so quantized GEMM can run directly</dd>
        </div>
        <div>
          <dt>
            <code>max_context</code>
          </dt>
          <dd>
            Generation length bound, where 0 reads graph metadata. With a sliding KV cache this
            bounds sequence length, not the memory allocation
          </dd>
        </div>
        <div>
          <dt>
            <code>kv_window</code>, <code>kv_sink</code>
          </dt>
          <dd>
            Resident KV window. Laptop default 2048 with 4 sinks. Phone default 512. On the C API
            since 0.4.0
          </dd>
        </div>
        <div>
          <dt>
            <code>pin_layers</code>, <code>expert_cache_bytes</code>
          </dt>
          <dd>
            Pinned trunk and MoE expert LRU. On the C API since 0.4.0, with{" "}
            <code>max_threads</code> and <code>single_staging</code>
          </dd>
        </div>
        <div>
          <dt>
            <code>enable_fusion</code> and related
          </dt>
          <dd>
            Also <code>enable_memory_reuse</code>, <code>enable_streaming</code>, and{" "}
            <code>enable_profiler</code>
          </dd>
        </div>
        <div>
          <dt>
            <code>weight_init</code>
          </dt>
          <dd>
            Defaults to <strong>none</strong>, so missing weights raise an error
          </dd>
        </div>
        <div>
          <dt>
            <code>weights_dir</code>, <code>weights_sandbox</code>
          </dt>
          <dd>Weight resolution and the path sandbox</dd>
        </div>
      </dl>
      <p>
        C API callers must set <code>struct_size = sizeof(uaii_session_options)</code> immediately
        after <code>uaii_session_options_init</code>.
      </p>

      <h2>Global CLI flags</h2>
      <pre>{`uaii --config configs/uaii.toml --log-level info --load-plugins doctor`}</pre>

      <h2>Dashboard configuration</h2>
      <p>
        The operator UI reads <code>dashboard/uaii-dash.json</code>, modelled on{" "}
        <code>uaii-dash.example.json</code>, plus the environment variables below. Full guide:{" "}
        <Link href="/docs/dashboard/">Operator UI</Link>.
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
                <code>127.0.0.1</code> for local use, or <code>0.0.0.0</code> to self-host
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
              <td>Bearer token, required for any non-loopback bind</td>
            </tr>
            <tr>
              <td>
                <strong>UAII_BIN</strong> / <strong>UAII_BENCH_BIN</strong>
              </td>
              <td>Override the auto-detected CLI binaries</td>
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
              <td>
                Use the WSL <code>uaii</code> binary when Windows blocks unsigned executables
              </td>
            </tr>
            <tr>
              <td>
                <strong>UAII_BACKEND</strong>
              </td>
              <td>
                Device for chat and run jobs: <code>auto</code> by default, or <code>cpu</code>,{" "}
                <code>cuda</code>, and so on
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <pre>{`# Local, on loopback
cd dashboard && npm start

# Self-hosted, which requires a token
UAII_DASH_BIND=0.0.0.0 UAII_DASH_TOKEN=secret npm start`}</pre>
    </DocLayout>
  );
}
