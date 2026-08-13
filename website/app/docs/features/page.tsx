import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Features" };

export default function Page() {
  return (
    <DocLayout
      title="Features"
      active="/docs/features/"
      lede="Modular inference runtime: HF decoder dirs and GGUF blk.* LLMs are the ready product surface; other formats and arches plug in over time."
    >
      <h2>What UAII is</h2>
      <p>
        An <strong>execution platform for AI inference</strong> — not a single-model engine.
        Loaders, operators, backends, schedulers, and storage providers plug into a common
        core. Today we recommend <Link href="/docs/huggingface/">Hugging Face decoder-only</Link>{" "}
        and GGUF transformers; see <Link href="/docs/huggingface/">Hugging Face</Link> for
        allowlist limits before overclaiming Hub coverage.
      </p>

      <h2>Model formats</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Format</th>
              <th>What you get</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>Hugging Face dir</strong>
              </td>
              <td>
                First-class for allowlisted <strong>CausalLM</strong> families with Llama-style
                tensor names. <code>uaii pull</code>, <code>generate --model ./dir</code>,
                dashboard Hub pull. Not every Hub architecture.
              </td>
            </tr>
            <tr>
              <td>
                <strong>GGUF</strong>
              </td>
              <td>
                Capability-based import for llama.cpp-style <code>blk.*</code> decoder tensors
                (dense or MoE; DeepSeek MLA expand path). Reads <code>{"{arch}.*"}</code>{" "}
                metadata. Common Q/K quants + IQ4_NL; exotic IQ* fail closed.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Safetensors</strong>
              </td>
              <td>
                Used inside HF dirs; standalone weight graphs also import when layout maps to IR
              </td>
            </tr>
            <tr>
              <td>
                <strong>ONNX</strong>
              </td>
              <td>Import to IR (sidecar JSON or proto when enabled)</td>
            </tr>
            <tr>
              <td>
                <strong>MLX</strong>
              </td>
              <td>
                <code>config.json</code> + <code>.safetensors</code> (weights + config import)
              </td>
            </tr>
            <tr>
              <td>
                <strong>PyTorch</strong>
              </td>
              <td>
                <code>.pt</code> / <code>.pth</code> via exported <code>.onnx</code> or{" "}
                <code>.uaii.json</code> sidecar
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Quantization & compute</h2>
      <ul>
        <li>
          In-memory GGUF block quants: <code>Q4_0</code>, <code>Q4_1</code>, <code>Q5_0</code>
          , <code>Q5_1</code>, <code>Q8_0</code>, <code>Q2_K</code>–<code>Q6_K</code>
        </li>
        <li>Pack/unpack helpers: F16, BF16, INT8, INT4, NF4, MXFP4</li>
        <li>
          Session policy: <code>compute_dtype</code> (F32 / F16),{" "}
          <code>keep_quantized_weights</code>
        </li>
        <li>
          Unsupported / other IQ* GGUF types fail closed — no silent Ones fills (
          <code>IQ4_NL</code> supported)
        </li>
      </ul>

      <h2>LLM runtime</h2>
      <ul>
        <li>
          Prefill + decode (greedy or temp/top-k/top-p) via <code>Session::generate</code> /{" "}
          <code>uaii_session_generate</code>
        </li>
        <li>
          CLI: <code>uaii pull</code>, <code>uaii generate</code>, <code>uaii chat --jsonl</code>{" "}
          (warm session for the dashboard)
        </li>
        <li>KV cache with context limits from model metadata; memory presets / peak RSS</li>
        <li>
          Tokenizers: HF tokenizer JSON, GGUF <code>tokenizer.ggml.*</code>, BPE /
          SentencePiece (optional), SimpleTokenizer for demos
        </li>
      </ul>

      <h2>Dashboard</h2>
      <p>
        Operator UI in <code>dashboard/</code> — local or self-host. HF Hub pull + GGUF/HF
        chat with streaming, model library, doctor, benches, request queue, Bearer auth on
        LAN, OpenAI-compatible <code>/v1/chat/completions</code>. See{" "}
        <Link href="/docs/dashboard/">Dashboard</Link>.
      </p>

      <h2>Hardware</h2>
      <p>
        CPU is always first-class. GPU backends are optional compile-time features;{" "}
        <code>uaii doctor</code> prints whether native init succeeded and which ops still
        host-fallback. See{" "}
        <Link href="/docs/backends/">Backends</Link>.
      </p>

      <h2>Runtime tooling</h2>
      <ul>
        <li>Graph validator; JSON + binary IR (<code>.uaii.json</code> / <code>.uaii</code>)</li>
        <li>Planner: fusion, memory reuse, storage plan, disk plan cache</li>
        <li>Weight streaming; CUDA async H2D overlap when native</li>
        <li>Chrome-trace profiler, benchmark CLI, plugin operator ABI</li>
      </ul>
    </DocLayout>
  );
}
