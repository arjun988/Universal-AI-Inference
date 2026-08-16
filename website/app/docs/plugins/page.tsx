import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Plugins" };

export default function Page() {
  return (
    <DocLayout
      title="Plugins"
      active="/docs/plugins/"
      lede="Extend the runtime with dynamic libraries that implement the plugin ABI. Discovery is local, and no hosted registry ships with the project today."
    >
      <h2>Required exports</h2>
      <p>Every plugin library must export three symbols:</p>
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
        The host rejects any plugin whose <code>abi_version</code> does not match{" "}
        <code>UAII_PLUGIN_ABI_VERSION</code>. The contract is defined in{" "}
        <code>include/uaii/c_api/plugin_abi.h</code>.
      </p>

      <h2>Reference plugins</h2>
      <dl className="deflist">
        <div>
          <dt>
            <code>example_probe</code>
          </dt>
          <dd>Validates discovery and loading</dd>
        </div>
        <div>
          <dt>
            <code>example_op</code>
          </dt>
          <dd>
            Registers a <code>Neg</code> operator into the CPU dispatch path
          </dd>
        </div>
      </dl>

      <h2>Loading plugins</h2>
      <pre>{`uaii doctor --load-plugins`}</pre>

      <h2>Distribution</h2>
      <p>
        A hosted registry for third-party loaders, operators, and backends is out of scope for the
        current release. Design notes may appear in <code>docs/marketplace.md</code>, but plugins
        are loaded from local directories today. If you want to contribute a plugin, start with the{" "}
        <Link href="/docs/contribute/">contribution guide</Link>.
      </p>
    </DocLayout>
  );
}
