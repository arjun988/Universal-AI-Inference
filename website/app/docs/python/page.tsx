import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Python SDK" };

export default function Page() {
  return (
    <DocLayout
      title="Python SDK"
      active="/docs/python/"
      lede="Load a model or an IR graph, run inference, and capture profiles without touching the C++ internals."
    >
      <h2>Install</h2>
      <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

python bindings/python/scripts/bundle_native.py --build-dir build
pip install -e bindings/python`}</pre>
      <p>
        For the native extension, configure with <code>-DUAII_BUILD_PYTHON=ON</code>, which fetches
        pybind11. Without it, the package calls <code>uaii_capi</code> through ctypes.
      </p>
      <p>
        If the shared library cannot be found, set <code>UAII_CAPI_PATH</code> to the full path of{" "}
        <code>uaii_capi</code>.
      </p>

      <h2>Load, run, profile</h2>
      <pre>{`import uaii

session = uaii.Session.from_path(
    "examples/ir/toy_mlp.uaii.json",
    weight_init="ones",
    profile=True,
    trace_path="uaii_py_profile.json",
)
session.set_tensor("x", [1.0, 2.0, 3.0, 4.0])
session.run()
print(session.get_tensor("y_prob"))
print(session.profile_summary())`}</pre>

      <h2>Convert models</h2>
      <pre>{`uaii.convert_model("model.gguf", "model.uaii.json")`}</pre>

      <h2>Defaults worth knowing</h2>
      <dl className="deflist">
        <div>
          <dt>
            <code>weight_init=&quot;none&quot;</code>
          </dt>
          <dd>
            A model with missing weights raises an error rather than filling tensors with
            placeholder values
          </dd>
        </div>
        <div>
          <dt>
            <code>backend=&quot;cpu&quot;</code>
          </dt>
          <dd>
            The default for both Python and the C API. Pass{" "}
            <code>backend=&quot;auto&quot;</code> to run the same probe as the CLI, which selects
            a native GPU when one is usable. <code>uaii doctor</code> reports which backends fall
            back to the host
          </dd>
        </div>
        <div>
          <dt>Memory preset</dt>
          <dd>
            <code>Session.from_path</code> does not call <code>uaii_resolve_memory_preset</code>.
            Use the CLI, or the C API, for laptop and phone paging. The ctypes struct includes the
            0.4.0 fields so a later binding can set them
          </dd>
        </div>
      </dl>
      <pre>{`session = uaii.Session.from_path("model.gguf", backend="auto")`}</pre>
    </DocLayout>
  );
}
