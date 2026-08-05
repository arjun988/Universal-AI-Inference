import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "C API" };

export default function Page() {
  return (
    <DocLayout
      title="C API"
      active="/docs/c-api/"
      lede="Stable ABI for language bindings and embedding. Header: include/uaii/c_api/uaii.h. Current version: 0.3.0 (pre-1.0)."
    >
      <h2>Versioning</h2>
      <ul>
        <li>
          <strong>MAJOR</strong> — removed symbols, layout breaks, semantic changes
        </li>
        <li>
          <strong>MINOR</strong> — backward-compatible additions (fields appended; honor{" "}
          <code>struct_size</code>)
        </li>
        <li>
          <strong>PATCH</strong> — fixes that preserve documented behavior
        </li>
      </ul>
      <p>
        Gate bindings on <code>uaii_get_c_api_version()</code>. Plugin ABI (
        <code>UAII_PLUGIN_ABI_VERSION</code>) is versioned separately.
      </p>

      <h2>Session options</h2>
      <p>
        Always call <code>uaii_session_options_init</code> and set{" "}
        <code>struct_size = sizeof(uaii_session_options)</code>. Notable fields in 0.3.0:
      </p>
      <ul>
        <li>
          <code>backend</code>, <code>weights_dir</code>, <code>weight_init</code> (default NONE)
        </li>
        <li>
          <code>enable_fusion</code>, <code>enable_memory_reuse</code>,{" "}
          <code>enable_streaming</code>, <code>enable_profiler</code>
        </li>
        <li>
          <code>compute_dtype</code>, <code>keep_quantized_weights</code>,{" "}
          <code>max_context</code>
        </li>
      </ul>

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
        Shared library CMake target: <code>uaii_capi</code>. Full policy lives in{" "}
        <code>docs/c_api_stability.md</code> in the repository.
      </p>
    </DocLayout>
  );
}
