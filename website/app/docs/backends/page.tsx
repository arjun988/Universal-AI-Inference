import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Backends" };

export default function Page() {
  return (
    <DocLayout
      title="Backends"
      active="/docs/backends/"
      lede="CPU is always real. Optional GPU backends expose device memory and kernels when built with the matching SDK — and advertise host-fallback honestly."
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
                Attention may host-fallback (flagged)
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

      <h2>Scheduling</h2>
      <p>
        <code>DeviceScheduler</code> places ops on the preferred device. When{" "}
        <code>attention_host_fallback</code> is true, Attention and RoPE are scheduled to CPU;
        the session run loop honors those decisions.
      </p>

      <h2>Doctor</h2>
      <pre>{`uaii doctor`}</pre>
      <p>
        Prints the active GEMM provider and per-backend <code>host_fallback</code> /{" "}
        <code>attention_host_fallback</code> flags. Silent “GPU name, CPU math” without those
        flags is treated as a bug.
      </p>
    </DocLayout>
  );
}
