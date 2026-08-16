import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Backends" };

export default function Page() {
  return (
    <DocLayout
      title="Backends"
      active="/docs/backends/"
      lede="CPU execution is always complete. The CLI and dashboard default to auto, which selects a native GPU only when this binary can drive a real device, and every host fallback is reported."
    >
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Backend</th>
              <th>Default build</th>
              <th>With vendor SDK and a device</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>CPU</strong>
              </td>
              <td>Complete f32 kernel set with tiled reference GEMM</td>
              <td>Dispatches to oneDNN or OpenBLAS when linked</td>
            </tr>
            <tr>
              <td>
                <strong>CUDA</strong>
              </td>
              <td>Runs, with all math on the host</td>
              <td>
                Device memory, <strong>cuBLASLt</strong>, and Add, Mul, RMSNorm, Softmax, and Silu
                on device. Decode attention and RoPE run on GPU once native initialisation
                succeeds; prefill attention may still run on the host
              </td>
            </tr>
            <tr>
              <td>
                <strong>Metal</strong>
              </td>
              <td>Runs, with all math on the host</td>
              <td>Shared buffers, with MatMul, Add, and RMSNorm compiled as runtime MSL</td>
            </tr>
            <tr>
              <td>
                <strong>Vulkan</strong>
              </td>
              <td>Runs, with all math on the host</td>
              <td>
                Device buffers, with Add as a compute shader. Remaining operators are listed in
                the reported capabilities
              </td>
            </tr>
            <tr>
              <td>
                <strong>WebGPU</strong>
              </td>
              <td>Runs, with all math on the host</td>
              <td>Buffer path when headers are present. Compute coverage is limited</td>
            </tr>
            <tr>
              <td>
                <strong>ROCm</strong>
              </td>
              <td>Runs, with all math on the host</td>
              <td>HIP memory, MatMul through rocBLAS, and HIP kernels for Add and RMSNorm</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Device selection</h2>
      <p>
        <code>uaii generate</code>, <code>chat</code>, <code>run</code>, and the dashboard all
        default to <code>--backend auto</code>, which selects the first native GPU that reports a
        real device, probing CUDA, then Metal, ROCm, and Vulkan. A backend compiled without its
        vendor SDK does not qualify, so auto stays on CPU rather than claiming a device it cannot
        use. Pass <code>--backend cpu</code>, or select CPU in Settings, to pin the host. The C
        API and Python <code>Session</code> default to <code>&quot;cpu&quot;</code>; pass{" "}
        <code>&quot;auto&quot;</code> to run the same probe.
      </p>

      <h2>Scheduling</h2>
      <p>
        <code>DeviceScheduler</code> assigns each operator to the preferred device. When{" "}
        <code>attention_host_fallback</code> is set, attention and RoPE are scheduled onto the
        CPU, and the session run loop honours those assignments.
      </p>

      <h2>Verifying what will run</h2>
      <pre>{`uaii doctor`}</pre>
      <p>
        Doctor reports the active GEMM provider, the per-backend <code>host_fallback</code> and{" "}
        <code>attention_host_fallback</code> flags, and a <code>recommended:</code> line matching
        what auto-select will choose. A backend that reports a GPU name while executing on the
        host without setting those flags is a bug worth reporting.
      </p>
    </DocLayout>
  );
}
