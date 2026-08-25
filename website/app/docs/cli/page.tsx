import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "CLI" };

export default function Page() {
  return (
    <DocLayout
      title="CLI reference"
      active="/docs/cli/"
      lede="One binary to diagnose the install, pull Hugging Face models, convert weights, and run generation."
    >
      <h2>Global flags</h2>
      <dl className="deflist">
        <div>
          <dt>
            <code>--config &lt;toml&gt;</code>
          </dt>
          <dd>Load a configuration file</dd>
        </div>
        <div>
          <dt>
            <code>--log-level &lt;level&gt;</code>
          </dt>
          <dd>One of trace, debug, info, warn, or error</dd>
        </div>
        <div>
          <dt>
            <code>--no-color</code>
          </dt>
          <dd>Disable ANSI colour output</dd>
        </div>
        <div>
          <dt>
            <code>--load-plugins</code>
          </dt>
          <dd>
            Discover and load plugins, used with <code>doctor</code>
          </dd>
        </div>
      </dl>

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
              <td>
                Reports loaded modules, the active GEMM provider, backend capabilities, and the{" "}
                <code>recommended:</code> device
              </td>
            </tr>
            <tr>
              <td>
                <code>validate</code> / <code>inspect</code> / <code>graph</code>
              </td>
              <td>
                Validate an IR graph and dump it as <code>text</code>, <code>dot</code>,{" "}
                <code>json</code>, or <code>plan</code>
              </td>
            </tr>
            <tr>
              <td>
                <code>pull</code>
              </td>
              <td>
                Download a Hugging Face repository into a local directory for the decoder-only
                path
              </td>
            </tr>
            <tr>
              <td>
                <code>convert</code>
              </td>
              <td>
                Compile a Hugging Face folder, GGUF, Safetensors, ONNX, MLX, or PyTorch sidecar
                into UAII IR
              </td>
            </tr>
            <tr>
              <td>
                <code>tokenize</code>
              </td>
              <td>
                Encode and decode with simple, BPE, SentencePiece, GGUF, or Hugging Face
                tokenizers
              </td>
            </tr>
            <tr>
              <td>
                <code>generate</code>
              </td>
              <td>
                Generate from a Hugging Face CausalLM folder, a GGUF <code>blk.*</code> model, or{" "}
                <code>--demo</code>. Accepts <code>--preset</code>, <code>--backend</code>,{" "}
                <code>--kv-window</code>, <code>--json</code>, and <code>--stream</code>
              </td>
            </tr>
            <tr>
              <td>
                <code>chat</code>
              </td>
              <td>
                Keep a warm session as a JSONL worker (<code>--jsonl</code>), which the operator UI
                drives
              </td>
            </tr>
            <tr>
              <td>
                <code>run</code>
              </td>
              <td>
                Execute an IR graph, or a built-in graph with <code>--demo</code>
              </td>
            </tr>
            <tr>
              <td>
                <code>profile</code> / <code>benchmark</code> / <code>cache</code>
              </td>
              <td>Chrome-trace capture, operator timings, and plan cache management</td>
            </tr>
            <tr>
              <td>
                <code>help</code> / <code>version</code>
              </td>
              <td>Usage text and the version string</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Generate and chat</h2>
      <p>
        The device defaults to <code>--backend auto</code>, which uses a native GPU when this
        binary can drive one and CPU otherwise. Pass <code>--backend cpu</code> to pin the host. A
        physical GPU does not qualify if the binary was configured with{" "}
        <code>UAII_WITH_CUDA=OFF</code>, since only host math is compiled in; auto stays on CPU in
        that case.
      </p>
      <p>
        The memory budget defaults to <code>--preset laptop</code>: roughly 8 GiB, a trunk
        that is auto-shrunk until it fits (layers, then <code>lm_head</code>, then embed), packed
        GGUF staging, an expert LRU filled from leftover RAM, and a 2048-token KV window with 4
        sinks. GGUF Mixtral and Qwen packed banks are split per expert, and only the routed top-k
        are paged in. <code>--preset none</code> disables streaming entirely.
      </p>
      <pre>{`# One-shot generation from any GGUF with blk.* tensors
uaii generate --model path/to/model.gguf \\
  --prompt "Hello" --max-new-tokens 64 --stream

# Built-in tiny model, no download required
uaii generate --demo --prompt "hi" --max-new-tokens 8 --json

# Longer conversations, with the sliding KV window kept at its default
uaii generate --model path/to/model.gguf --kv-window 2048 --prompt "Hello"

# Warm worker for the dashboard or automation
uaii chat --model path/to/model.gguf --jsonl`}</pre>

      <h2>Cheatsheet</h2>
      <pre>{`uaii doctor --load-plugins
uaii validate <ir>
uaii inspect <ir>
uaii graph <ir> --format plan
uaii convert model.gguf -o model.uaii.json
uaii tokenize encode hello world
uaii generate --model model.gguf --prompt "hi" --json
uaii run --demo toy_mlp
uaii run <ir> --input x=1,2,3,4 --backend auto
uaii profile --demo --output uaii_profile.json
uaii benchmark --demo
uaii cache status`}</pre>
    </DocLayout>
  );
}
