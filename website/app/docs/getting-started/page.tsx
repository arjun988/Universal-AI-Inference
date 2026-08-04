import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Getting started" };

export default function Page() {
  return (
    <DocLayout title="Getting started" active="/docs/getting-started/">
      <p>
        Build the C++ runtime, then use the CLI or Python SDK. This documentation site is a
        static Next.js export — no application backend.
      </p>

      <h2>Build</h2>
      <pre>{`cmake -S . -B build -DUAII_BUILD_TESTS=ON -DUAII_BUILD_PLUGINS=ON
cmake --build build --config Release --parallel`}</pre>

      <h2>CLI smoke</h2>
      <pre>{`uaii doctor
uaii run --demo toy_mlp
uaii run --demo optimize
uaii profile --demo`}</pre>

      <h2>Python path</h2>
      <pre>{`# After build (uaii_capi shared library)
pip install -e bindings/python
python examples/python/load_run_profile.py`}</pre>

      <p>
        Optional native extension: configure with <code>-DUAII_BUILD_PYTHON=ON</code> (pulls
        pybind11 via FetchContent).
      </p>

      <h2>Repo docs</h2>
      <ul>
        <li>
          <code>docs/plan.md</code> — roadmap
        </li>
        <li>
          <code>docs/architecture.md</code> — modules
        </li>
        <li>
          <code>docs/c_api_stability.md</code> — C API semver
        </li>
        <li>
          <code>docs/roadmap/PHASE7.md</code> — Phase 7 checklist
        </li>
      </ul>
    </DocLayout>
  );
}
