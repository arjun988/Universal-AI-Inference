import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Hugging Face" };

export default function Page() {
  return (
    <DocLayout
      title="Hugging Face"
      active="/docs/huggingface/"
      lede="Run decoder-only Hub folders when the layout, dtype, and tokenizer are in the supported set. Automatic layout detection, a configurable memory budget, and CPU execution unless a native GPU is present."
    >
      <h2>What loads</h2>
      <p>
        A folder with <code>config.json</code> and dense <code>*.safetensors</code> (F16, BF16, or
        F32, single file or shards) for decoder-only classes such as <code>*ForCausalLM</code> and{" "}
        <code>*LMHeadModel</code>. Tensor names are detected across Llama-style layouts, GPT-2,
        fused QKV, Mixtral experts, Qwen3 QK-norm, and Gemma residual norms. Probe a directory with{" "}
        <code>uaii inspect</code> before generate — <code>status: ok</code> means the family math
        is wired (Gemma <code>(1+w)</code> RMSNorm, GPT-2 <code>wpe</code>, Qwen3 QK-norm, sliding
        window), not merely that tensors were named. Chat uses a documented family equivalent of
        the shipped Jinja <code>chat_template</code>, not a Jinja interpreter.
      </p>
      <p>
        Quantized Hub repos (GPTQ, AWQ, bitsandbytes, HQQ, compressed-tensors, MXFP4) are not
        loaded: convert them to dense BF16/F16 or to GGUF. Kimi / DeepSeek MLA folders are blocked
        on Hugging Face; run the GGUF instead (<code>uaii generate --model kimi.gguf</code>).
        gpt-oss is detected and fail-closed until a family handler exists.
      </p>
      <div className="chip-row" style={{ marginBottom: "1rem" }}>
        <span className="chip">Llama</span>
        <span className="chip">Qwen</span>
        <span className="chip">Mistral</span>
        <span className="chip">Gemma (1+w RMS)</span>
        <span className="chip">Kimi GGUF</span>
      </div>
      <p>
        Generation defaults to the laptop preset: pin a trunk that fits, page packed GGUF weights,
        and keep a 2048-token KV window. Laptop keeps two staging buffers, so an unquantized
        tensor larger than about half the budget fail-closes. Convert Hugging Face BF16 7B and
        larger to GGUF Q4_K. Phone is a separate preset with one staging buffer. See{" "}
        <Link href="/docs/memory/">Memory</Link>.
      </p>

      <h2>Model sizes</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Scale</th>
              <th>What the dial does</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>0.5B to 8B</strong>
              </td>
              <td>Often fits the laptop budget. First generate is a cold load</td>
            </tr>
            <tr>
              <td>
                <strong>Larger dense GGUF</strong>
              </td>
              <td>
                Pages packed weights. Throughput follows disk and CPU. No device tok/s is published
                for this path
              </td>
            </tr>
            <tr>
              <td>
                <strong>MoE GGUF</strong>
              </td>
              <td>
                Pages the routed experts. Convert Mixtral and Qwen-MoE to Q4_K. BF16 expert banks
                that exceed the budget fail closed
              </td>
            </tr>
            <tr>
              <td>
                <strong>Hugging Face BF16, 7B and up</strong>
              </td>
              <td>Fail-closed. Convert to GGUF Q4_K</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        Paging works at tensor granularity, so a dense checkpoint with a single FFN tensor larger
        than available RAM can still fail to load. Mixture-of-experts checkpoints are the
        reliable path on constrained machines. Adjust the budget with{" "}
        <code>--preset laptop|desktop|server|phone|none</code>, <code>--ram-gb</code>, and{" "}
        <code>--kv-window</code>. Phone success is peak RSS, not tokens per second.
      </p>

      <h2>CLI</h2>
      <pre>{`uaii pull org/model --outdir ./models/hf/org__model
uaii inspect ./models/hf/org__model
uaii generate --model ./models/hf/org__model --dry-run
uaii generate --model ./models/hf/org__model --prompt "Hello" --json
uaii generate --model kimi.gguf --prompt "Hello"
uaii generate --model ./models/hf/org__model --preset none --prompt "Hello"
uaii generate --model ./models/hf/org__model --kv-window 4096 --prompt "Hello"
uaii convert ./models/hf/org__model -o model.uaii.json`}</pre>
      <p>
        The device defaults to <code>--backend auto</code>, which uses a native GPU when this
        binary can drive one and CPU otherwise. Pass <code>--backend cpu</code> to pin the host.
      </p>

      <h2>Dashboard</h2>
      <p>
        Open <strong>Models</strong>, choose <strong>Pull from Hub</strong> or{" "}
        <strong>Import HF folder</strong>, then switch to <strong>Chat</strong>. The backend
        setting defaults to automatic GPU detection. Point <code>UAII_BIN</code> at a current{" "}
        <code>uaii</code> build. An older <code>uaii-dash.json</code> pinned to{" "}
        <code>&quot;backend&quot;: &quot;cpu&quot;</code> keeps that value, so reset it to Auto in
        Settings. Full guide: <Link href="/docs/dashboard/">Operator UI</Link>.
      </p>

      <h2>Current limitations</h2>
      <ul>
        <li>
          Weight dtypes: F16, BF16, and F32 safetensors. F8 and GPTQ/AWQ/MXFP4 fail closed with a
          convert hint
        </li>
        <li>
          GPTQ, AWQ, and bitsandbytes repos are not loaded. Convert to dense F16/BF16 or to GGUF
        </li>
        <li>
          Vision, audio, encoder-decoder, SSM, and RWKV models are not supported. Plugin scaffolds
          exist for the architectures on the roadmap
        </li>
        <li>
          Hugging Face DeepSeek / Kimi MLA checkpoints are not mapped. Use the GGUF expand path
        </li>
        <li>
          gpt-oss dense packing is fail-closed (<code>family=gpt_oss</code>). Convert MXFP4 to BF16
          or GGUF; there is no Llama fallback
        </li>
      </ul>
    </DocLayout>
  );
}
