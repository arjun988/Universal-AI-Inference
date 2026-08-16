import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "C API" };

export default function Page() {
  return (
    <DocLayout
      title="C API"
      active="/docs/c-api/"
      lede="A stable ABI for language bindings and embedding. The header is include/uaii/c_api/uaii.h, and the current version is 0.3.0, ahead of 1.0."
    >
      <h2>Versioning</h2>
      <dl className="deflist">
        <div>
          <dt>MAJOR</dt>
          <dd>Removed symbols, layout changes, or altered semantics</dd>
        </div>
        <div>
          <dt>MINOR</dt>
          <dd>
            Backward-compatible additions, where fields are appended and{" "}
            <code>struct_size</code> is honoured
          </dd>
        </div>
        <div>
          <dt>PATCH</dt>
          <dd>Fixes that preserve documented behaviour</dd>
        </div>
      </dl>
      <p>
        Gate your bindings on <code>uaii_get_c_api_version()</code>. The plugin ABI,{" "}
        <code>UAII_PLUGIN_ABI_VERSION</code>, is versioned independently.
      </p>

      <h2>Session options</h2>
      <p>
        Always call <code>uaii_session_options_init</code>, then set{" "}
        <code>struct_size = sizeof(uaii_session_options)</code>. The fields that matter most in
        0.3.0:
      </p>
      <ul>
        <li>
          <code>backend</code>, which defaults to <code>&quot;cpu&quot;</code>; pass{" "}
          <code>&quot;auto&quot;</code> to run the same GPU probe as the CLI. Also{" "}
          <code>weights_dir</code> and <code>weight_init</code>, which defaults to NONE
        </li>
        <li>
          <code>enable_fusion</code>, <code>enable_memory_reuse</code>,{" "}
          <code>enable_streaming</code>, <code>enable_profiler</code>
        </li>
        <li>
          <code>compute_dtype</code>, <code>keep_quantized_weights</code>, and{" "}
          <code>max_context</code>, the generation bound, where 0 reads the limit from graph
          metadata
        </li>
      </ul>
      <div className="callout">
        <span className="callout-label">Not on this ABI yet</span>
        <p>
          The sliding KV window (<code>kv_window</code>) and the laptop memory preset live on the
          CLI and C++ <code>SessionOptions</code>. A C session therefore runs without the
          2048-token window until a later <code>struct_size</code> revision adds the field.
        </p>
      </div>

      <h2>Core calls</h2>
      <pre>{`uaii_session_options opts;
uaii_session_options_init(&opts);
opts.struct_size = sizeof(opts);
opts.enable_profiler = 1;
opts.profile_trace_path = "trace.json";

uaii_session* s = NULL;
uaii_session_create("model.uaii.json", &opts, &s);

float x[] = {1, 2, 3, 4};
uaii_session_set_f32(s, "x", x, 4);
uaii_session_run(s);

/* Optional: uaii_session_generate(...) for token generation */
uaii_session_destroy(s);`}</pre>

      <p>
        The shared library builds from the CMake target <code>uaii_capi</code>. The full stability
        policy is in <code>docs/c_api_stability.md</code> in the repository.
      </p>
    </DocLayout>
  );
}
