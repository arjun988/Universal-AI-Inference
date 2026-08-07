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
uaii generate --demo --prompt "hi" --max-new-tokens 4 --json
uaii generate --model path/to/model.gguf --prompt "hi" --max-new-tokens 64`}</pre>
      <p>
        GGUF import is capability-based: any <code>general.architecture</code> with
        llama.cpp-style <code>blk.*</code> tensors (not limited to Llama). See{" "}
        <Link href="/docs/features/">Features</Link>.
      </p>

      <h2>Dashboard (optional UI)</h2>
      <pre>{`cd dashboard
npm run install:all
npm run build && npm start
# → http://127.0.0.1:8787

# Self-host
UAII_DASH_BIND=0.0.0.0 UAII_DASH_TOKEN=secret npm start`}</pre>
      <p>
        Chat, models, doctor, benches, OpenAI-compatible <code>/v1</code>. Configure with env
        or Settings — full guide: <Link href="/docs/dashboard/">Dashboard</Link>.
      </p>

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
          <Link href="/docs/features/">Features</Link> — formats, multi-arch GGUF, quants
        </li>
        <li>
          <Link href="/docs/dashboard/">Dashboard</Link> — local / self-host UI config
        </li>
        <li>
          <Link href="/docs/configuration/">Configuration</Link> — TOML, env, CMake flags
        </li>
        <li>
          <Link href="/docs/cli/">CLI</Link> — <code>generate</code>, <code>chat</code>, run
        </li>
      </ul>
    </DocLayout>
  );
}
