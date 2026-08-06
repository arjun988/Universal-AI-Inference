import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Benchmarks" };

export default function Page() {
  return (
    <DocLayout
      title="Benchmarks"
      active="/docs/benchmarks/"
      lede="Absolute, reproducible CPU microbenchmarks — GFLOP/s and median times, with full environment disclosure. Not a bake-off against other runtimes."
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
                <strong>f32 GEMM 256³</strong>
              </td>
              <td>
                <strong>7.78 GFLOP/s</strong> · 4.32 ms
              </td>
              <td>
                <code>ref-tiled</code>, 32 threads
              </td>
            </tr>
            <tr>
              <td>
                <strong>f32 GEMM 512³</strong>
              </td>
              <td>
                <strong>11.1 GFLOP/s</strong> · 24.3 ms
              </td>
              <td>Median of 21 trials</td>
            </tr>
            <tr>
              <td>
                <strong>f32 GEMM 1024³</strong>
              </td>
              <td>
                <strong>14.6 GFLOP/s</strong> · 147 ms
              </td>
              <td>Median of 10 trials</td>
            </tr>
            <tr>
              <td>
                <strong>Session</strong> 8×512 stack
              </td>
              <td>
                <strong>2.86 ms</strong> median
              </td>
              <td>
                <code>Session::run</code> only
              </td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0 weights</strong> 2048×4096
              </td>
              <td>
                <strong>4.5 MiB</strong> vs 32 MiB f32
              </td>
              <td>7.11× format-defined compression</td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0 MatMul</strong>
              </td>
              <td>3.45 ms packed · 12.9 ms unpack+f32</td>
              <td>Two UAII paths, synthetic blocks</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>Host:</strong> local WSL2 · Intel Core i9-14900HX · 32 threads · Release ·{" "}
        <code>GEMM=ref-tiled</code>. Published JSON:{" "}
        <code>benchmarks/results/local_wsl.json</code>.
      </p>

      <h2>Reproduce</h2>
      <pre>{`# WSL / Linux (recommended on locked-down Windows hosts)
bash scripts/run_bench_wsl.sh

# Native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAII_BUILD_BENCHMARKS=ON
cmake --build build --target uaii_bench --parallel
export UAII_BENCH_CPU="Your Exact CPU Model"
./build/benchmarks/uaii_bench --trials 21 --warmup 5 --json`}</pre>

      <h2>CI</h2>
      <p>
        GitHub Actions job <code>benchmarks</code> builds the same harness on Ubuntu, Windows, and
        macOS and uploads JSON artifacts for regression visibility. Those runner numbers are{" "}
        <strong>not</strong> the published table above — cite{" "}
        <code>local_wsl.json</code> (or your own machine JSON).
      </p>

      <h2>What we measure</h2>
      <ul>
        <li>
          <strong>Dense GEMM</strong> — absolute GFLOP/s from median trial time (
          <code>2·N³</code> FLOPs)
        </li>
        <li>
          <strong>Session</strong> — synthetic 8×512 MatMul+ReLU stack;{" "}
          <code>Session::run</code> only
        </li>
        <li>
          <strong>Quant</strong> — GGUF Q4_0 packed GEMM + format memory ratio
        </li>
        <li>
          <strong>Optional</strong> — <code>--vs-naive</code> appendix only (not a product claim)
        </li>
      </ul>

      <h2>What we do not claim</h2>
      <ul>
        <li>Tokens/s on public LLMs</li>
        <li>Wins vs llama.cpp / ONNX Runtime / TensorRT</li>
        <li>GPU numbers unless you build that backend and publish its JSON</li>
      </ul>

      <h2>Going faster</h2>
      <p>
        Build with <code>-DUAII_WITH_ONEDNN=ON</code>, <code>UAII_WITH_OPENBLAS=ON</code>, or{" "}
        <code>UAII_WITH_CUDA=ON</code>, then re-run and quote <code>gemm_provider</code> from the
        JSON / <code>uaii doctor</code>.
      </p>

      <p>
        Deep dive: repository <code>docs/benchmarks.md</code>. Also see{" "}
        <Link href="/docs/examples/">Examples</Link> for CLI smoke paths.
      </p>
    </DocLayout>
  );
}
