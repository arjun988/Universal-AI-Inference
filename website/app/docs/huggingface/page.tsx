import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Hugging Face" };

export default function Page() {
  return (
    <DocLayout
      title="Hugging Face"
      active="/docs/huggingface/"
      lede="First-class decoder-only directory load — the recommended way to try UAII on Hub models today."
    >
      <h2>Supported now</h2>
      <p>
        A folder with <code>config.json</code> + <code>*.safetensors</code> (± shard index) for
        allowlisted <strong>CausalLM</strong> architectures and Llama-style tensor names.
      </p>
      <div className="chip-row" style={{ marginBottom: "1rem" }}>
        <span className="chip">Llama</span>
        <span className="chip">Mistral</span>
        <span className="chip">Qwen2/3</span>
        <span className="chip">Gemma</span>
        <span className="chip">Phi</span>
        <span className="chip">InternLM</span>
        <span className="chip">Olmo / Granite / …</span>
      </div>
      <p>
        Full field table and denials: repo{" "}
        <code>docs/huggingface_support.md</code>. Unknown arches and MoE/MLA HF configs fail
        closed (use GGUF MLA path for DeepSeek).
      </p>

      <h2>CLI</h2>
      <pre>{`# Download (needs huggingface-cli or curl)
uaii pull org/model --outdir ./models/hf/org__model

# Generate
uaii generate --model ./models/hf/org__model --prompt "Hello" --preset laptop --json

# Convert to IR
uaii convert ./models/hf/org__model -o model.uaii.json`}</pre>

      <h2>Dashboard</h2>
      <p>
        Models → <strong>Pull from Hub</strong> or <strong>Import HF folder</strong>, then Chat.
        Point <code>UAII_BIN</code> at a rebuilt <code>uaii</code> that includes the HF loader.
      </p>
      <p>
        See <Link href="/docs/dashboard/">Operator UI</Link>.
      </p>

      <h2>Not every Hub model</h2>
      <ul>
        <li>Vision / audio / encoder-decoder / SSM / RWKV — later (plugin scaffolds exist)</li>
        <li>Non-Llama tensor layouts (GPT-2 <code>transformer.h</code>, fused QKV) — not mapped yet</li>
        <li>HF MoE / DeepSeek MLA configs — use GGUF for MLA today</li>
      </ul>
    </DocLayout>
  );
}
