import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Features" };

export default function Page() {
  return (
    <DocLayout
      title="Features"
      active="/docs/features/"
      lede="UAII is a modular inference runtime: models become UAII IR, then run on pluggable backends through one session surface."
    >
      <h2>What UAII is</h2>
      <p>
        An <strong>execution platform for AI inference</strong> — not a single-model engine.
        Loaders, operators, backends, schedulers, and storage providers plug into a common
        core. The product loop is: ingest → validate/plan → execute → integrate (CLI / C /
        Python).
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
                <strong>GGUF</strong>
              </td>
              <td>
                Llama-family transformer import (<code>llama</code>, <code>llama3</code>,{" "}
                <code>mistral</code>, <code>qwen2</code>, <code>phi3</code>) with RMSNorm,
                QKV, RoPE, Attention + KV, SwiGLU, lm_head
              </td>
            </tr>
            <tr>
              <td>
                <strong>Safetensors</strong>
              </td>
              <td>Weight graphs / HF-style layouts → UAII IR</td>
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
        <li>Unsupported / IQ* GGUF types fail closed — no silent Ones fills</li>
      </ul>

      <h2>LLM runtime</h2>
      <ul>
        <li>
          Prefill + greedy decode via <code>Session::generate</code> /{" "}
          <code>uaii_session_generate</code>
        </li>
        <li>First-class KV cache with context limits from model metadata</li>
        <li>
          Tokenizers: BPE, SentencePiece (optional), GGUF <code>tokenizer.ggml.*</code>,
          SimpleTokenizer for demos
        </li>
      </ul>

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
