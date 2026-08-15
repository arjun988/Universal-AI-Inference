import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Hugging Face" };

export default function Page() {
  return (
    <DocLayout
      title="Hugging Face"
      active="/docs/huggingface/"
      lede="Decoder-only Hub folders — layout auto-detect, laptop memory dial, CPU unless a native GPU is present."
    >
      <h2>Supported now</h2>
      <p>
        A folder with <code>config.json</code> + <code>*.safetensors</code> (± shard index) for
        decoder-only classes (<code>*ForCausalLM</code>, <code>*LMHeadModel</code>). Tensor
        names are auto-detected — Llama-style, GPT-2, fused QKV, Mixtral experts.
      </p>
      <div className="chip-row" style={{ marginBottom: "1rem" }}>
        <span className="chip">Llama / Qwen / Gemma</span>
        <span className="chip">GPT-2 / NeoX</span>
        <span className="chip">Mixtral MoE</span>
        <span className="chip">laptop preset</span>
        <span className="chip">backend auto</span>
      </div>
      <p>
        Generate/chat default to a kimi-style <strong>storage-first</strong> laptop budget:
        pin embed + first layers, stream the rest, LRU + <strong>top-k-only</strong> paging for
        MoE experts, sliding KV (2048 tokens + 4 sinks). Big models stay on disk; they are
        slow, not silent-OOM. Details: repo <code>docs/huggingface_support.md</code>.
      </p>

      <h2>What size fits</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Scale</th>
              <th>Expectation on ~8 GB</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>~0.5B–8B</strong>
              </td>
              <td>Comfortable in RAM (still a cold load on first generate)</td>
            </tr>
            <tr>
              <td>
                <strong>30B+ decoder</strong>
              </td>
              <td>Designed to run by paging weights; disk + CPU bound, not OOM</td>
            </tr>
            <tr>
              <td>
                <strong>~500B-class MoE</strong>
              </td>
              <td>
                Same path: page routed experts, bound KV. Slow NVMe traffic. Experts still
                dequant to f32 one-at-a-time (not kimi MXFP4 packed matmul)
              </td>
            </tr>
            <tr>
              <td>
                <strong>Trillion / 1.5 TB</strong>
              </td>
              <td>Not a measured RSS claim</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        A dense checkpoint whose <em>single</em> FFN tensor exceeds RAM can still fail. MoE is
        the laptop path. Override with <code>--preset desktop|server|none</code>,{" "}
        <code>--ram-gb</code>, <code>--kv-window</code>.
      </p>

      <h2>CLI</h2>
      <pre>{`uaii pull org/model --outdir ./models/hf/org__model
uaii generate --model ./models/hf/org__model --prompt "Hello" --json
uaii generate --model ./models/hf/org__model --preset none --prompt "Hello"
uaii generate --model ./models/hf/org__model --kv-window 4096 --prompt "Hello"
uaii convert ./models/hf/org__model -o model.uaii.json`}</pre>
      <p>
        Device default is <code>--backend auto</code> (native GPU if this binary can use it,
        else CPU). Pin with <code>--backend cpu</code>.
      </p>

      <h2>Dashboard</h2>
      <p>
        Models → <strong>Pull from Hub</strong> or <strong>Import HF folder</strong>, then Chat.
        Settings → Backend defaults to Auto-detect GPU. Point <code>UAII_BIN</code> at a
        rebuilt <code>uaii</code>. If an older <code>uaii-dash.json</code> still has{" "}
        <code>&quot;backend&quot;: &quot;cpu&quot;</code>, switch it back to Auto.
      </p>
      <p>
        See <Link href="/docs/dashboard/">Operator UI</Link>.
      </p>

      <h2>Not every Hub model</h2>
      <ul>
        <li>Vision / audio / encoder-decoder / SSM / RWKV — later (plugin scaffolds)</li>
        <li>HF DeepSeek MLA — use DeepSeek GGUF expand path</li>
        <li>GPT-2 learned <code>wpe</code> table is not applied</li>
      </ul>
    </DocLayout>
  );
}
