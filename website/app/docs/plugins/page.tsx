import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Plugins" };

export default function Page() {
  return (
    <DocLayout
      title="Plugins"
      active="/docs/plugins/"
      lede="Extend UAII with dynamic libraries that match the plugin ABI. Discovery is local — there is no marketplace service in-tree today."
    >
      <h2>ABI exports</h2>
      <p>Every plugin dynamic library must export:</p>
      <ul>
        <li>
          <code>uaii_plugin_get_info</code>
        </li>
        <li>
          <code>uaii_plugin_init</code>
        </li>
        <li>
          <code>uaii_plugin_shutdown</code>
        </li>
      </ul>
      <p>
        The host rejects plugins whose <code>abi_version != UAII_PLUGIN_ABI_VERSION</code>.
        See <code>include/uaii/c_api/plugin_abi.h</code>.
      </p>

      <h2>Examples in-tree</h2>
      <ul>
        <li>
          <code>plugins/example_probe</code> — discovery / load validation
        </li>
        <li>
          <code>plugins/example_op</code> — registers a <code>Neg</code> operator into the CPU
          dispatch path
        </li>
      </ul>

      <h2>Doctor</h2>
      <pre>{`uaii doctor --load-plugins`}</pre>

      <h2>Future marketplace</h2>
      <p>
        A multi-registry marketplace for third-party loaders, ops, and backends is intentionally
        out of scope for the current product freeze. Design notes may live under{" "}
        <code>docs/marketplace.md</code>; nothing ships as a hosted service today.
      </p>
    </DocLayout>
  );
}
