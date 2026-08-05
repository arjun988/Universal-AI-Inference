import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Architecture" };

export default function Page() {
  return (
    <DocLayout
      title="Architecture"
      active="/docs/architecture/"
      lede="Models become UAII IR. The session plans and executes that graph on a chosen backend."
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
      <ul>
        <li>
          <code>uaii-ir</code> — graph, tensors, operator registry, validator, serialize, plan
        </li>
        <li>
          <code>uaii-runtime</code> — session lifecycle, DeviceScheduler, KV cache, demos
        </li>
        <li>
          <code>uaii-kernels</code> — CPU ops, <code>IGemm</code>, quant GEMM
        </li>
        <li>
          <code>uaii-backends</code> — CPU + optional CUDA / Metal / Vulkan / WebGPU / ROCm
        </li>
        <li>
          <code>uaii-loaders</code> — GGUF, Safetensors, ONNX, MLX, PyTorch
        </li>
        <li>
          <code>uaii-tokenizers</code> — Simple, BPE, SentencePiece, GGUF wiring
        </li>
        <li>
          <code>uaii-planner</code> — fusion, memory/storage plans, disk cache
        </li>
        <li>
          <code>uaii-storage</code> — mmap, streaming weights
        </li>
        <li>
          <code>uaii-capi</code> — stable C ABI for SDKs
        </li>
      </ul>

      <h2>IR on disk</h2>
      <ul>
        <li>
          <code>.uaii.json</code> — hand-authored / debug JSON
        </li>
        <li>
          <code>.uaii</code> — native binary (magic <code>UAIR</code>; schema-aligned with{" "}
          <code>schemas/uaii_ir.fbs</code>)
        </li>
      </ul>
      <p>
        Deep dive: repository file <code>docs/architecture.md</code>.
      </p>
    </DocLayout>
  );
}
