import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "C API" };

export default function Page() {
  return (
    <DocLayout title="C API stability" active="/docs/c-api/">
      <p>
        Stable ABI header: <code>include/uaii/c_api/uaii.h</code>. Version macros live in{" "}
        <code>version.h</code>. Current API: <strong>1.0.0</strong>.
      </p>

      <h2>Semver policy</h2>
      <ul>
        <li>
          <strong>MAJOR</strong> — breaking signature or semantics changes
        </li>
        <li>
          <strong>MINOR</strong> — backward-compatible additions
        </li>
        <li>
          <strong>PATCH</strong> — fixes that preserve behavior
        </li>
      </ul>
      <p>
        Gate bindings on <code>uaii_get_c_api_version()</code>, not only the package version.
        Plugin ABI (<code>UAII_PLUGIN_ABI_VERSION</code>) is versioned separately.
      </p>

      <h2>Core calls</h2>
      <pre>{`uaii_session_options opts;
uaii_session_options_init(&opts);
opts.enable_profiler = 1;
opts.profile_trace_path = "trace.json";

uaii_session* s = NULL;
uaii_session_create("model.uaii.json", &opts, &s);
float x[] = {1,2,3,4};
uaii_session_set_f32(s, "x", x, 4);
uaii_session_run(s);
/* get_f32 / profile_summary / write_trace */
uaii_session_destroy(s);`}</pre>

      <p>
        Full policy: <code>docs/c_api_stability.md</code> in the repository.
      </p>
    </DocLayout>
  );
}
