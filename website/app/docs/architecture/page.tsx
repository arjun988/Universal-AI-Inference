import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Architecture" };

export default function Page() {
  return (
    <DocLayout
      title="Architecture"
      active="/docs/architecture/"
      lede="Every model compiles to UAII IR. The session plans that graph, then executes it on CPU or on a native GPU when device selection finds one."
    >
      <h2>Pipeline</h2>
      <div className="pipeline" aria-label="Architecture pipeline">
        <span>Loader</span>
        <i>→</i>
        <span>UAII IR</span>
        <i>→</i>
        <span>Validator</span>
        <i>→</i>
        <span>Planner</span>
        <i>→</i>
        <span>Scheduler</span>
        <i>→</i>
        <span>Backend</span>
      </div>

      <h2>Modules</h2>
      <dl className="deflist">
        <div>
          <dt>
            <code>uaii-ir</code>
          </dt>
          <dd>
            Graph and tensor types, operator registry, validator, serialization, and plan
            representation
          </dd>
        </div>
        <div>
          <dt>
            <code>uaii-runtime</code>
          </dt>
          <dd>
            Session lifecycle, <code>DeviceScheduler</code>, sliding KV cache, and memory presets
          </dd>
        </div>
        <div>
          <dt>
            <code>uaii-kernels</code>
          </dt>
          <dd>
            CPU operators, the <code>IGemm</code> interface, and quantized GEMM
          </dd>
        </div>
        <div>
          <dt>
            <code>uaii-backends</code>
          </dt>
          <dd>CPU, plus optional CUDA, Metal, Vulkan, WebGPU, and ROCm</dd>
        </div>
        <div>
          <dt>
            <code>uaii-loaders</code>
          </dt>
          <dd>
            Hugging Face directories, GGUF including per-expert MoE slices, Safetensors, ONNX,
            MLX, and PyTorch
          </dd>
        </div>
        <div>
          <dt>
            <code>uaii-tokenizers</code>
          </dt>
          <dd>Simple vocabularies, BPE, SentencePiece, and GGUF metadata</dd>
        </div>
        <div>
          <dt>
            <code>uaii-planner</code>
          </dt>
          <dd>Fusion, memory and storage plans, and the disk plan cache</dd>
        </div>
        <div>
          <dt>
            <code>uaii-storage</code>
          </dt>
          <dd>Memory mapping, weight streaming, and the expert LRU cache</dd>
        </div>
        <div>
          <dt>
            <code>uaii-capi</code>
          </dt>
          <dd>The stable C ABI that SDKs build on</dd>
        </div>
      </dl>

      <h2>Storage-first execution</h2>
      <p>
        CLI and dashboard generation apply the laptop preset unless you pass another one. Weights
        stay on disk. The session pins a trunk that fits and pages the rest. Phone uses one
        staging buffer. Laptop and above use two. An unquantized tensor larger than about half
        the budget fail-closes.
      </p>
      <ul>
        <li>
          Final norm stays resident. Embeddings, <code>lm_head</code>, and early layers stay only
          while pin plus staging still fits. GGUF block-quants stay packed
        </li>
        <li>
          KV is a sliding window (2048 on laptop, 512 on phone) plus 4 attention sinks. RoPE keeps
          absolute positions
        </li>
        <li>
          Packed expert banks split into <code>#e=N</code>. Only the routed top-k are staged
        </li>
        <li>
          <code>auto</code> picks the first native GPU that reports a real device. Attention and
          RoPE stay on CPU. Q4_K stays in the CPU quant GEMM
        </li>
      </ul>
      <p>
        C API 0.4.0 applies the same dial through <code>uaii_resolve_memory_preset</code>. The
        Python <code>Session.from_path</code> helper does not call that yet, so it stays on{" "}
        <code>cpu</code> with <code>kv_window = 0</code> unless you pass those options yourself.
      </p>

      <h2>IR on disk</h2>
      <dl className="deflist">
        <div>
          <dt>
            <code>.uaii.json</code>
          </dt>
          <dd>Readable JSON for hand-authored graphs and debugging</dd>
        </div>
        <div>
          <dt>
            <code>.uaii</code>
          </dt>
          <dd>
            Native binary with the <code>UAIR</code> magic number, aligned to the schema in{" "}
            <code>schemas/uaii_ir.fbs</code>
          </dd>
        </div>
      </dl>
      <p>
        Full detail is in <code>docs/architecture.md</code> in the repository.
      </p>
    </DocLayout>
  );
}
