import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Quick start" };

export default function Page() {
  return (
    <DocLayout
      title="Quick start"
      active="/docs/getting-started/"
      lede="Build the C++ runtime, smoke the CLI, then wire the C API or Python SDK. This site is a static Next.js export — no application backend."
    >
      <h2>Prerequisites</h2>
      <ul>
        <li>C++17 compiler (MSVC 2019+, Clang 10+, or GCC 9+)</li>
        <li>CMake 3.20+</li>
        <li>Ninja (recommended) or your platform generator</li>
        <li>Optional: CUDA Toolkit, Vulkan SDK, ROCm, oneDNN, OpenBLAS, SentencePiece, Python 3.10+</li>
      </ul>

      <h2>Build</h2>
      <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\
  -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel`}</pre>
      <p>
        On Windows multi-config generators, binaries often land under{" "}
        <code>build/Release/</code>. Single-config Ninja builds typically produce{" "}
        <code>build/libs/uaii-cli/uaii</code> (or <code>uaii.exe</code>).
      </p>

      <h2>First commands</h2>
      <pre>{`uaii version
uaii doctor
uaii doctor --load-plugins

uaii validate examples/ir/toy_mlp.uaii.json
uaii run --demo toy_mlp
uaii run --demo gguf
uaii run --demo parity`}</pre>

      <h2>Run a graph</h2>
      <pre>{`uaii run examples/ir/toy_mlp.uaii.json \\
  --weight-init ones \\
  --input x=1,2,3,4 \\
  --output y_prob`}</pre>

      <h2>Python (optional)</h2>
      <pre>{`python bindings/python/scripts/bundle_native.py --build-dir build
pip install -e bindings/python
python examples/python/load_run_profile.py`}</pre>
      <p>
        If the loader cannot find the shared library, set <code>UAII_CAPI_PATH</code> to{" "}
        <code>uaii_capi.dll</code> / <code>.so</code> / <code>.dylib</code>.
      </p>

      <h2>Tests</h2>
      <pre>{`ctest --test-dir build -C Release --output-on-failure`}</pre>

      <h2>Next</h2>
      <ul>
        <li>
          <Link href="/docs/features/">Features</Link> — formats, quants, backends
        </li>
        <li>
          <Link href="/docs/configuration/">Configuration</Link> — TOML, env, CMake flags
        </li>
        <li>
          <Link href="/docs/examples/">Examples</Link> — CLI, IR, generate, profile
        </li>
      </ul>
    </DocLayout>
  );
}
