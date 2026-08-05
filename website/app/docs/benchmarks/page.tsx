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
                <strong>f32 GEMM 512³</strong>
              </td>
              <td>
                <strong>11.4 GFLOP/s</strong> · 23.6 ms
              </td>
              <td>
                <code>ref-tiled</code>, no oneDNN
              </td>
            </tr>
            <tr>
              <td>
                <strong>f32 GEMM 1024³</strong>
              </td>
              <td>
                <strong>4.7 GFLOP/s</strong> · 452 ms
              </td>
              <td>Cache/bandwidth limited on ref path</td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0 weights</strong> 1024×4096
              </td>
              <td>
                <strong>2.25 MiB</strong> vs 16.0 MiB f32
              </td>
              <td>7.11× format-defined compression</td>
            </tr>
            <tr>
              <td>
                <strong>Q4_0 MatMul</strong>
              </td>
              <td>8.0 ms packed · 12.1 ms unpack+f32</td>
              <td>Two UAII paths, synthetic blocks</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>Host:</strong> Windows 11 · MinGW g++ 15 · Release ·{" "}
        <code>GEMM=ref-tiled</code>. Sample JSON:{" "}
        <code>benchmarks/results/sample_windows_mingw.json</code>. Regenerate with median of 21
        trials before citing elsewhere.
      </p>

      <h2>Reproduce</h2>
      <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUAII_BUILD_BENCHMARKS=ON
cmake --build build --target uaii_bench --parallel
export UAII_BENCH_CPU="Your Exact CPU Model"
./build/benchmarks/uaii_bench --trials 21 --warmup 5 --json`}</pre>

      <h2>CI artifacts</h2>
      <p>
        GitHub Actions job <code>benchmarks</code> runs the same harness on Ubuntu, Windows, and
        macOS and uploads <code>uaii-bench-&lt;os&gt;-&lt;sha&gt;</code> JSON artifacts (median of 21
        trials). Prefer those for public citation when local unsigned builds are blocked.
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
