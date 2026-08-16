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
        The CLI and dashboard apply a memory budget by default, so the process pages weights in
        rather than mapping an entire checkpoint into RAM. This is how 30B-class dense decoders
        and 500B-class MoE checkpoints run on a laptop: throughput becomes bound by NVMe
        bandwidth, and capacity stops being the limit.
      </p>
      <ul>
        <li>
          Embeddings, <code>lm_head</code>, the final norm, and a few trunk layers stay pinned.
          Everything else streams through <code>file#tensor</code> references
        </li>
        <li>
          The KV cache keeps a resident window, 2048 tokens by default, plus 4 attention sinks.
          RoPE preserves absolute positions, so long conversations compact the window instead of
          growing memory
        </li>
        <li>
          GGUF packed expert banks are split into per-expert references (<code>#e=N</code>). After
          routing, only the top-k experts are staged, and the rest remain on disk
        </li>
        <li>
          <code>auto</code> selects the first native GPU that reports a real device. A backend
          built without its vendor SDK does not qualify
        </li>
      </ul>
      <p>
        C API 0.3.0 and the default Python session do <strong>not</strong> apply this preset. They
        run on <code>cpu</code> with <code>kv_window = 0</code> unless you set the options the ABI
        exposes. Because paging works per tensor, a dense tensor larger than available RAM can
        still fail to load.
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
