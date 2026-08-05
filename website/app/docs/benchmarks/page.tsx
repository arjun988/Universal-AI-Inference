import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Benchmarks" };

export default function Page() {
  return (
    <DocLayout
      title="Benchmarks"
      active="/docs/benchmarks/"
      lede="Reproducible CPU microbenchmarks from the in-tree uaii_bench harness — not marketing fiction."
    >
      <div className="matrix-wrap" style={{ marginBottom: "1.75rem" }}>
        <table className="matrix">
          <thead>
            <tr>
              <th>Workload</th>
              <th>Baseline</th>
              <th>UAII</th>
              <th>Gain</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>f32 GEMM 1024³</strong> vs naive ijk
              </td>
              <td>6187 ms</td>
              <td>
                <strong>452 ms</strong>
              </td>
              <td>
                <strong>13.7×</strong>
              </td>
            </tr>
            <tr>
              <td>
                <strong>f32 GEMM 512³</strong> vs naive ijk
              </td>
              <td>146 ms</td>
              <td>
                <strong>24 ms</strong>
              </td>
              <td>
                <strong>6.2×</strong>
              </td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0 weights</strong> vs f32
              </td>
              <td>16.0 MiB</td>
              <td>
                <strong>2.25 MiB</strong>
              </td>
              <td>
                <strong>7.1× smaller</strong>
              </td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0 MatMul</strong> vs unpack+f32
              </td>
              <td>12.1 ms</td>
              <td>
                <strong>8.0 ms</strong>
              </td>
              <td>
                <strong>1.5×</strong>
              </td>
            </tr>
            <tr>
              <td>
                <strong>Session MLP</strong> (fused)
              </td>
              <td>—</td>
              <td>
                <strong>~0.01 ms</strong>/iter
              </td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>Host:</strong> Windows 11 · MinGW g++ 15 · Release ·{" "}
        <code>GEMM=ref-tiled</code> (no oneDNN / OpenBLAS). Sample JSON:{" "}
        <code>benchmarks/results/sample_windows_mingw.json</code>.
      </p>

      <h2>Reproduce</h2>
      <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAII_BUILD_BENCHMARKS=ON
cmake --build build --target uaii_bench --parallel
./build/benchmarks/uaii_bench --iters 8 --json`}</pre>

      <h2>What we measure</h2>
      <ul>
        <li>
          <strong>Naive GEMM</strong> — classic triple-loop baseline (no tiling / threads)
        </li>
        <li>
          <strong>UAII GEMM</strong> — <code>default_gemm()</code> (tiled ref or vendor BLAS)
        </li>
        <li>
          <strong>Quant path</strong> — packed Q4_0 GEMM vs full unpack then f32
        </li>
        <li>
          <strong>Session</strong> — timed <code>Session::run</code> on a fused MLP
        </li>
      </ul>

      <h2>Going faster</h2>
      <p>
        Build with <code>-DUAII_WITH_ONEDNN=ON</code>, <code>UAII_WITH_OPENBLAS=ON</code>, or{" "}
        <code>UAII_WITH_CUDA=ON</code>, then re-run and quote the provider from{" "}
        <code>uaii doctor</code>. Absolute LLM tokens/s depends on model and hardware — these
        benches validate UAII&apos;s own kernels.
      </p>

      <p>
        Deep dive: repository <code>docs/benchmarks.md</code>. Also see{" "}
        <Link href="/docs/examples/">Examples</Link> for CLI smoke paths.
      </p>
    </DocLayout>
  );
}
