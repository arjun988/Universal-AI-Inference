import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Architecture" };

export default function Page() {
  return (
    <DocLayout
      title="Architecture"
      active="/docs/architecture/"
      lede="Models become UAII IR. The session plans and executes that graph on CPU, or on a native GPU when auto-select finds one."
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
          <code>uaii-runtime</code> — session lifecycle, DeviceScheduler, sliding KV, memory
          presets
        </li>
        <li>
          <code>uaii-kernels</code> — CPU ops, <code>IGemm</code>, quant GEMM
        </li>
        <li>
          <code>uaii-backends</code> — CPU + optional CUDA / Metal / Vulkan / WebGPU / ROCm
        </li>
        <li>
          <code>uaii-loaders</code> — HF directories, GGUF (per-expert MoE slices), Safetensors,
          ONNX, MLX, PyTorch
        </li>
        <li>
          <code>uaii-tokenizers</code> — Simple, BPE, SentencePiece, GGUF wiring
        </li>
        <li>
          <code>uaii-planner</code> — fusion, memory/storage plans, disk cache
        </li>
        <li>
          <code>uaii-storage</code> — mmap, streaming weights, expert LRU
        </li>
        <li>
          <code>uaii-capi</code> — stable C ABI for SDKs
        </li>
      </ul>

      <h2>Storage-first (CLI / dashboard)</h2>
      <p>
        Generate and chat default to a laptop memory dial so the process pages weights instead
        of mapping a whole checkpoint. That is the product path for 30B+ decoders and ~500B-class
        MoE on a laptop — slow NVMe I/O, not a silent OOM, and not a proven trillion-parameter
        RSS claim.
      </p>
      <ul>
        <li>
          Pin embed / lm_head / final norm plus a few trunk layers; stream the rest via{" "}
          <code>file#tensor</code> refs
        </li>
        <li>
          Sliding KV: resident window (laptop default 2048) + attention sinks (default 4).
          RoPE keeps absolute positions; long chats compact the window instead of growing RAM
        </li>
        <li>
          MoE: GGUF packed expert banks split to per-expert refs (<code>#e=N</code>); after
          routing, only top-k experts are staged. Unused experts stay on disk
        </li>
        <li>
          Backends: <code>auto</code> picks the first native GPU with a real device; host-fallback
          stubs do not count
        </li>
      </ul>
      <p>
        C API 0.3.0 and the Python default session do <strong>not</strong> apply that laptop
        preset — they stay on <code>cpu</code> with <code>kv_window = 0</code> unless you pass
        options the ABI exposes. Dense tensors larger than RAM can still fail.
      </p>

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
