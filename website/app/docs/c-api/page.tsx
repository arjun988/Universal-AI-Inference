import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "C API" };

export default function Page() {
  return (
    <DocLayout
      title="C API"
      active="/docs/c-api/"
      lede="Stable ABI in include/uaii/c_api/uaii.h. Current version is 0.4.0. Package version is 0.3.1."
    >
      <h2>Rules</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Bump</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Major</td>
              <td>Removed symbols, layout changes, or new semantics</td>
            </tr>
            <tr>
              <td>Minor</td>
              <td>
                New functions or fields appended to a struct. <code>struct_size</code> keeps old
                callers working
              </td>
            </tr>
            <tr>
              <td>Patch</td>
              <td>Fixes that keep documented behaviour</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Gate bindings on <code>uaii_get_c_api_version()</code>. The plugin ABI is separate.
      </p>

      <h2>0.4.0 fields</h2>
      <p>
        Call <code>uaii_session_options_init</code>, then keep{" "}
        <code>struct_size = sizeof(uaii_session_options)</code>. Fields added after{" "}
        <code>max_context</code>:
      </p>
      <ul>
        <li>
          <code>kv_window</code>, <code>kv_sink</code> (default 4)
        </li>
        <li>
          <code>pin_layers</code>, <code>expert_cache_bytes</code>
        </li>
        <li>
          <code>max_threads</code> (0 = hardware concurrency), <code>single_staging</code>
        </li>
      </ul>
      <p>
        <code>uaii_resolve_memory_preset</code> and <code>uaii_session_options_apply_preset</code>{" "}
        fill those fields. Phone caps at half of free RAM. Laptop, desktop, and server keep their
        targets when free RAM covers them. See <Link href="/docs/memory/">Memory</Link>.
      </p>

      <h2>Session</h2>
      <pre>{`uaii_memory_preset_info info;
memset(&info, 0, sizeof info);
info.struct_size = sizeof info;
uaii_resolve_memory_preset("phone", 0, &info);

uaii_session_options opts;
uaii_session_options_init(&opts);
uaii_session_options_apply_preset(&opts, &info);
opts.backend = "auto";

uaii_session* s = NULL;
uaii_session_create("model.q4_k.gguf", &opts, &s);
uaii_session_set_background(s, 1);   /* paused: app backgrounded */
uaii_session_request_cancel(s);      /* cancelled; cleared at generate start */
uaii_session_memory_warning(s);      /* drop KV, keep longrope scale */
uaii_session_destroy(s);`}</pre>
      <p>
        <code>uaii_system_available_bytes</code> and <code>uaii_process_rss_bytes</code> are the
        counters to log. Weight init stays fail-closed (<code>NONE</code>).
      </p>
    </DocLayout>
  );
}
