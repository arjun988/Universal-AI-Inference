import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Backends" };

export default function Page() {
  return (
    <DocLayout
      title="Backends"
      active="/docs/backends/"
      lede="CPU is always real. CLI and dashboard default to auto: native GPU if this binary can actually drive a device, otherwise CPU. Host-fallback is never silent."
    >
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Backend</th>
              <th>Default build</th>
              <th>With SDK + device</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>CPU</strong>
              </td>
              <td>Full f32 kernels; tiled/ref GEMM</td>
              <td>oneDNN / OpenBLAS when linked</td>
            </tr>
            <tr>
              <td>
                <strong>CUDA</strong>
              </td>
              <td>Host-fallback executable</td>
              <td>
                Device memory, <strong>cuBLASLt</strong>, Add/Mul/RMSNorm/Softmax/Silu;
                decode Attention + RoPE on GPU when native init OK; prefill Attention may still
                CPU
              </td>
            </tr>
            <tr>
              <td>
                <strong>Metal</strong>
              </td>
              <td>Host-fallback</td>
              <td>Shared buffers; MatMul / Add / RMSNorm via runtime MSL</td>
            </tr>
            <tr>
              <td>
                <strong>Vulkan</strong>
              </td>
              <td>Host-fallback</td>
              <td>Device buffers; Add compute when available; remaining ops in caps</td>
            </tr>
            <tr>
              <td>
                <strong>WebGPU</strong>
              </td>
              <td>Host-fallback</td>
              <td>Buffer path when headers present; compute limited</td>
            </tr>
            <tr>
              <td>
                <strong>ROCm</strong>
              </td>
              <td>Host-fallback</td>
              <td>HIP memory; MatMul (rocBLAS); Add / RMSNorm HIP kernels</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Auto-select</h2>
      <p>
        <code>uaii generate</code>, <code>chat</code>, <code>run</code>, and the dashboard
        default to <code>--backend auto</code>. That picks the first native GPU that actually
        has a device (CUDA, then Metal, ROCm, Vulkan). Host-fallback stubs do <strong>not</strong>{" "}
        count — auto stays on CPU. <code>--backend cpu</code> (or Settings → CPU) pins the
        host. The C API / Python <code>Session</code> still default to <code>&quot;cpu&quot;</code>;
        pass <code>&quot;auto&quot;</code> if you want the same probe.
      </p>

      <h2>Scheduling</h2>
      <p>
        <code>DeviceScheduler</code> places ops on the preferred device. When{" "}
        <code>attention_host_fallback</code> is true, Attention and RoPE are scheduled to CPU;
        the session run loop honors those decisions.
      </p>

      <h2>Doctor</h2>
      <pre>{`uaii doctor`}</pre>
      <p>
        Prints the active GEMM provider, per-backend <code>host_fallback</code> /{" "}
        <code>attention_host_fallback</code> flags, and a <code>recommended:</code> line matching
        auto-select. Silent “GPU name, CPU math” without those flags is treated as a bug.
      </p>
    </DocLayout>
  );
}
