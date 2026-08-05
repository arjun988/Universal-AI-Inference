import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "CLI" };

export default function Page() {
  return (
    <DocLayout
      title="CLI reference"
      active="/docs/cli/"
      lede="The uaii binary is the fastest way to diagnose the install, convert models, and run graphs."
    >
      <h2>Global flags</h2>
      <ul>
        <li>
          <code>--config &lt;toml&gt;</code> — load configuration
        </li>
        <li>
          <code>--log-level &lt;level&gt;</code> — trace / debug / info / warn / error
        </li>
        <li>
          <code>--no-color</code> — disable ANSI color
        </li>
        <li>
          <code>--load-plugins</code> — discover and load plugins (doctor)
        </li>
      </ul>

      <h2>Commands</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Command</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>doctor</code>
              </td>
              <td>Modules, GEMM, backends, plugins</td>
            </tr>
            <tr>
              <td>
                <code>validate</code> / <code>inspect</code> / <code>graph</code>
              </td>
              <td>IR validation and dumps (<code>text|dot|json|plan</code>)</td>
            </tr>
            <tr>
              <td>
                <code>convert</code>
              </td>
              <td>GGUF / Safetensors / ONNX / MLX / PyTorch sidecar → UAII IR</td>
            </tr>
            <tr>
              <td>
                <code>tokenize</code>
              </td>
              <td>Simple / BPE / SentencePiece / GGUF encode & decode</td>
            </tr>
            <tr>
              <td>
                <code>run</code>
              </td>
              <td>Execute IR or <code>--demo …</code></td>
            </tr>
            <tr>
              <td>
                <code>profile</code> / <code>benchmark</code> / <code>cache</code>
              </td>
              <td>Chrome-trace, timings, plan cache</td>
            </tr>
            <tr>
              <td>
                <code>help</code> / <code>version</code>
              </td>
              <td>Usage and version string</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Cheatsheet</h2>
      <pre>{`uaii doctor --load-plugins
uaii validate <ir>
uaii inspect <ir>
uaii graph <ir> --format plan
uaii convert model.gguf -o model.uaii.json
uaii tokenize encode hello world
uaii run --demo toy_mlp
uaii run <ir> --input x=1,2,3,4 --backend cpu
uaii profile --demo --output uaii_profile.json
uaii benchmark --demo
uaii cache status`}</pre>
    </DocLayout>
  );
}
