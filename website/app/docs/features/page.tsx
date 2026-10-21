import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Features" };

export default function Page() {
  return (
    <DocLayout
      title="Features"
      active="/docs/features/"
      lede="A modular inference runtime. Hugging Face decoder folders and GGUF blk.* transformers are the supported surface today, and other formats and architectures arrive as plugins."
    >
      <h2>What UAII is</h2>
      <p>
        An <strong>execution platform for AI inference</strong> rather than a single-model
        engine. Loaders, operators, backends, schedulers, and storage providers all plug into a
        common core. The supported paths today are{" "}
        <Link href="/docs/huggingface/">Hugging Face decoder-only</Link> folders and GGUF
        transformers. Layout detection covers Llama-style models, GPT-2, fused QKV, and Mixtral
        experts. Vision, SSM, and HF MLA layouts are not yet mapped.
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
                Direct support for decoder-only <strong>CausalLM</strong> families with
                automatic layout detection, through <code>uaii pull</code>,{" "}
                <code>generate --model ./dir</code>, and dashboard Hub pulls. Vision, SSM, and
                HF MLA layouts are excluded.
              </td>
            </tr>
            <tr>
              <td>
                <strong>GGUF</strong>
              </td>
              <td>
                Capability-based import for llama.cpp-style <code>blk.*</code> decoder tensors,
                dense or MoE with per-expert <code>#e=N</code> slices, including the DeepSeek MLA
                expand path. Reads <code>{"{arch}.*"}</code> metadata. Common Q and K quants plus
                IQ4_NL are supported; other IQ variants stop with an error.
              </td>
            </tr>
            <tr>
              <td>
                <strong>Safetensors</strong>
              </td>
              <td>
                Used inside Hugging Face folders. Standalone weight graphs also import when the
                layout maps to the IR
              </td>
            </tr>
            <tr>
              <td>
                <strong>ONNX</strong>
              </td>
              <td>Imports to the IR from a sidecar JSON file, or from proto when enabled</td>
            </tr>
            <tr>
              <td>
                <strong>MLX</strong>
              </td>
              <td>
                <code>config.json</code> plus <code>.safetensors</code>, importing both weights
                and configuration
              </td>
            </tr>
            <tr>
              <td>
                <strong>PyTorch</strong>
              </td>
              <td>
                <code>.pt</code> and <code>.pth</code> through an exported <code>.onnx</code> or{" "}
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
        <li>Pack and unpack helpers for F16, BF16, INT8, INT4, NF4, and MXFP4</li>
        <li>
          Session policy through <code>compute_dtype</code>, either F32 or F16, and{" "}
          <code>keep_quantized_weights</code>
        </li>
        <li>
          Unrecognised GGUF quantization types stop with an error and a conversion hint rather
          than filling weights with placeholder values
        </li>
      </ul>

      <h2>LLM runtime</h2>
      <ul>
        <li>
          Prefill and decode with greedy or temperature, top-k, and top-p sampling, through{" "}
          <code>Session::generate</code> and <code>uaii_session_generate</code>
        </li>
        <li>
          CLI entry points <code>uaii pull</code>, <code>uaii generate</code>, and{" "}
          <code>uaii chat --jsonl</code>, the warm session the dashboard drives
        </li>
        <li>
          Sliding-window KV cache. Laptop keeps 2048 tokens and 4 sinks. Phone keeps 512, or fewer
          when the cap is tight
        </li>
        <li>
          Presets <code>laptop</code>, <code>desktop</code>, <code>server</code>, and{" "}
          <code>phone</code>. See the <Link href="/docs/memory/">memory dial</Link>
        </li>
        <li>
          Tokenizers for Hugging Face tokenizer JSON, GGUF <code>tokenizer.ggml.*</code> metadata,
          BPE, optional SentencePiece, and a simple vocabulary for demos
        </li>
      </ul>

      <h2>Operator dashboard</h2>
      <p>
        The console in <code>dashboard/</code> runs locally or self-hosted. It provides Hub pulls,
        streaming chat over GGUF and Hugging Face models, a model library, health checks,
        benchmarks, a request queue, Bearer authentication for non-loopback binds, and an
        OpenAI-compatible <code>/v1/chat/completions</code> endpoint. See{" "}
        <Link href="/docs/dashboard/">Operator UI</Link>.
      </p>

      <h2>Hardware</h2>
      <p>
        CPU execution is always fully supported. The CLI and dashboard default to{" "}
        <code>--backend auto</code>, which selects a native GPU only when this binary was built
        against the vendor SDK <em>and</em> a device is present, and CPU in every other case.{" "}
        <code>uaii doctor</code> reports the <code>recommended:</code> device alongside any{" "}
        <code>host_fallback</code> flags. See <Link href="/docs/backends/">Backends</Link>.
      </p>

      <h2>Runtime tooling</h2>
      <ul>
        <li>
          Graph validator with JSON and binary IR formats, <code>.uaii.json</code> and{" "}
          <code>.uaii</code>
        </li>
        <li>Planner covering fusion, memory reuse, storage plans, and a disk plan cache</li>
        <li>
          Weight streaming, top-k expert I/O, and asynchronous CUDA host-to-device overlap when
          native support is compiled in
        </li>
        <li>Chrome-trace profiler, benchmark CLI, and a versioned plugin operator ABI</li>
      </ul>
    </DocLayout>
  );
}
