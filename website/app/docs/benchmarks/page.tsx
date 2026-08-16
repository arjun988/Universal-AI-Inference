import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Benchmarks" };

export default function Page() {
  return (
    <DocLayout
      title="Benchmarks"
      active="/docs/benchmarks/"
      lede="Absolute measurements from uaii_bench v3: multi-provider GEMM, memory bandwidth, attention, session throughput, and Q4 memory footprint."
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
              <td>32 threads on WSL2</td>
            </tr>
            <tr>
              <td>
                <strong>ref-tiled</strong> 256 / 512 / 1024
              </td>
              <td>
                <strong>7.7 / 11.8 / 15.1 GFLOP/s</strong>
              </td>
              <td>Built-in kernel, no vendor BLAS</td>
            </tr>
            <tr>
              <td>
                <strong>Bandwidth</strong> triad
              </td>
              <td>
                <strong>16.6 GB/s</strong>
              </td>
              <td>STREAM triad, 256 MiB working set</td>
            </tr>
            <tr>
              <td>
                <strong>Attention</strong> e2e
              </td>
              <td>
                <strong>59.7 ms</strong>
              </td>
              <td>B1 H8 S512 D64 on the ref GEMM</td>
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
              <td>Compression against 32 MiB in f32</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>Host:</strong> WSL2 on an Intel Core i9-14900HX, Release build, with{" "}
        <code>ref</code> and <code>openblas</code> linked. Raw output is committed at{" "}
        <code>benchmarks/results/local_wsl.json</code>.
      </p>

      <h2>Reproduce</h2>
      <pre>{`sudo apt install -y libopenblas-dev libdnnl-dev
TRIALS=21 bash scripts/run_bench_wsl.sh
# or:
./build/benchmarks/uaii_bench --suite all --providers all --trials 21 --json`}</pre>

      <h2>Suites</h2>
      <dl className="deflist">
        <div>
          <dt>
            <code>gemm</code>
          </dt>
          <dd>
            Cycles every linked provider: <code>ref</code>, <code>onednn</code>, and{" "}
            <code>openblas</code>
          </dd>
        </div>
        <div>
          <dt>
            <code>bandwidth</code>
          </dt>
          <dd>STREAM copy, scale, add, and triad</dd>
        </div>
        <div>
          <dt>
            <code>attention</code>
          </dt>
          <dd>QKᵀ, softmax, and AV as one microbenchmark</dd>
        </div>
        <div>
          <dt>
            <code>session</code>, <code>quant</code>
          </dt>
          <dd>The IR execution stack and the Q4_0 packed path</dd>
        </div>
      </dl>

      <h2>Scope of these numbers</h2>
      <p>
        Everything above is a kernel or session measurement on one host. Three figures are
        deliberately absent, because they are not measured yet:
      </p>
      <ul>
        <li>Tokens per second on public LLM checkpoints</li>
        <li>Comparisons against llama.cpp, ONNX Runtime, or TensorRT</li>
        <li>
          oneDNN throughput, which is only reported when <code>onednn</code> appears in{" "}
          <code>linked_providers</code>
        </li>
      </ul>

      <p>
        Full methodology is in <code>docs/benchmarks.md</code> in the repository. See also{" "}
        <Link href="/docs/examples/">Examples</Link>.
      </p>
    </DocLayout>
  );
}
