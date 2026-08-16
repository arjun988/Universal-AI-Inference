import { DocLayout } from "@/components/DocLayout";
import Link from "next/link";

export const metadata = { title: "Quick start" };

export default function Page() {
  return (
    <DocLayout
      title="Quick start"
      active="/docs/getting-started/"
      lede="Build the C++ runtime, generate from a Hugging Face or GGUF checkpoint, then integrate through the C API or Python SDK."
    >
      <p>
        Start with a Hugging Face CausalLM folder or a GGUF <code>blk.*</code> model. Generation
        defaults to <code>--preset laptop</code>, which streams weights and slides the KV cache,
        and <code>--backend auto</code>, which uses a native GPU when one is available and CPU
        otherwise. The <Link href="/docs/huggingface/">Hugging Face guide</Link> covers the Hub
        path in detail.
      </p>

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
        Windows multi-config generators place binaries under <code>build/Release/</code>.
        Single-config Ninja builds produce <code>build/libs/uaii-cli/uaii</code>, or{" "}
        <code>uaii.exe</code> on Windows.
      </p>

      <h2>First commands</h2>
      <pre>{`uaii version
uaii doctor

# Hugging Face decoder-only
uaii pull org/model --outdir ./models/hf/org__model
uaii generate --model ./models/hf/org__model --prompt "hi" --max-new-tokens 64 --preset laptop

# Or a GGUF blk.* transformer, same memory preset and device selection
uaii generate --model path/to/model.gguf --prompt "hi" --max-new-tokens 64

# Smoke test the IR stack with no model weights
uaii generate --demo --prompt "hi" --max-new-tokens 4 --json`}</pre>
      <p>
        Hugging Face support requires a current build, so rebuild <code>uaii</code> after
        pulling. Point the dashboard at that binary with <code>UAII_BIN</code>.
      </p>

      <h2>Dashboard (optional UI)</h2>
      <pre>{`cd dashboard
npm run install:all
npm run build && npm start
# → http://127.0.0.1:8787

# Point at the rebuilt CLI, required for Hugging Face models
UAII_BIN=/path/to/uaii npm start

# Self-host on a LAN or server
UAII_DASH_BIND=0.0.0.0 UAII_DASH_TOKEN=secret npm start`}</pre>
      <p>
        The console provides Hub pulls, streaming chat over HF and GGUF models, health checks,
        benchmarks, and an OpenAI-compatible <code>/v1</code> endpoint. The default backend
        setting is automatic GPU detection. See the{" "}
        <Link href="/docs/dashboard/">operator UI guide</Link>.
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
        If the loader cannot locate the shared library, set <code>UAII_CAPI_PATH</code> to the
        full path of <code>uaii_capi.dll</code>, <code>.so</code>, or <code>.dylib</code>.
      </p>

      <h2>Tests</h2>
      <pre>{`ctest --test-dir build -C Release --output-on-failure`}</pre>

      <h2>Next steps</h2>
      <ul>
        <li>
          <Link href="/docs/huggingface/">Hugging Face</Link> — the Hub path, supported model
          sizes, and memory paging
        </li>
        <li>
          <Link href="/docs/features/">Features</Link> — formats, quantization, and runtime
          capabilities
        </li>
        <li>
          <Link href="/docs/dashboard/">Operator UI</Link> — local and self-hosted console
        </li>
        <li>
          <Link href="/docs/cli/">CLI</Link> — <code>pull</code>, <code>generate</code>, and{" "}
          <code>chat</code>
        </li>
        <li>
          <Link href="/docs/contribute/">Contribute</Link> — open an issue and send your first
          pull request
        </li>
      </ul>
    </DocLayout>
  );
}
