import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Benchmarks" };

export default function Page() {
  return (
    <DocLayout
      title="Benchmarks"
      active="/docs/benchmarks/"
      lede="uaii_bench v3 — multi-provider GEMM (ref + OpenBLAS), memory bandwidth, attention, session, and Q4."
    >
      <div className="matrix-wrap" style={{ marginBottom: "1.75rem" }}>
        <table className="matrix">
          <thead>
            <tr>
              <th>Workload</th>
              <th>Result</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <strong>OpenBLAS</strong> 256 / 512 / 1024
              </td>
              <td>
                <strong>141 / 284 / 425 GFLOP/s</strong>
              </td>
              <td>32 threads · WSL2</td>
            </tr>
            <tr>
              <td>
                <strong>ref-tiled</strong> 256 / 512 / 1024
              </td>
              <td>
                <strong>7.7 / 11.8 / 15.1 GFLOP/s</strong>
              </td>
              <td>Always-on baseline</td>
            </tr>
            <tr>
              <td>
                <strong>Bandwidth</strong> triad
              </td>
              <td>
                <strong>16.6 GB/s</strong>
              </td>
              <td>~256 MiB STREAM-style</td>
            </tr>
            <tr>
              <td>
                <strong>Attention</strong> e2e
              </td>
              <td>
                <strong>59.7 ms</strong>
              </td>
              <td>B1 H8 S512 D64 · ref GEMM</td>
            </tr>
            <tr>
              <td>
                <strong>Session</strong> 8×512
              </td>
              <td>
                <strong>2.77 ms</strong>
              </td>
              <td>
                <code>Session::run</code>
              </td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0</strong> memory
              </td>
              <td>
                <strong>7.11×</strong> · 4.5 MiB
              </td>
              <td>vs 32 MiB f32</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>Host:</strong> local WSL2 · Intel Core i9-14900HX · Release · linked{" "}
        <code>ref</code> + <code>openblas</code>. JSON:{" "}
        <code>benchmarks/results/local_wsl.json</code>.
      </p>

      <h2>Reproduce</h2>
      <pre>{`sudo apt install -y libopenblas-dev libdnnl-dev
TRIALS=21 bash scripts/run_bench_wsl.sh
# or:
./build/benchmarks/uaii_bench --suite all --providers all --trials 21 --json`}</pre>

      <h2>Suites</h2>
      <ul>
        <li>
          <code>gemm</code> — cycles linked providers (<code>ref</code>, <code>onednn</code>,{" "}
          <code>openblas</code>)
        </li>
        <li>
          <code>bandwidth</code> — STREAM copy / scale / add / triad
        </li>
        <li>
          <code>attention</code> — QKᵀ + softmax + AV microbench
        </li>
        <li>
          <code>session</code> / <code>quant</code> — IR stack + Q4_0 packed path
        </li>
      </ul>

      <h2>What we do not claim</h2>
      <ul>
        <li>Tokens/s on public LLMs</li>
        <li>Wins vs llama.cpp / ONNX Runtime / TensorRT</li>
        <li>oneDNN numbers unless <code>onednn</code> is in <code>linked_providers</code></li>
      </ul>

      <p>
        Deep dive: repository <code>docs/benchmarks.md</code>. Also see{" "}
        <Link href="/docs/examples/">Examples</Link>.
      </p>
    </DocLayout>
  );
}
