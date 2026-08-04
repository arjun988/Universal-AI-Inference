import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Python SDK" };

export default function Page() {
  return (
    <DocLayout title="Python SDK" active="/docs/python/">
      <p>
        The Phase 7 exit criterion: an external developer can <strong>load</strong> a model or
        IR, <strong>run</strong> inference, and <strong>profile</strong> without reading library
        internals.
      </p>

      <h2>Install</h2>
      <pre>{`cmake -S . -B build && cmake --build build --config Release
pip install -e bindings/python
# optional: -DUAII_BUILD_PYTHON=ON for pybind11 module`}</pre>

      <h2>Load → run → profile</h2>
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

      <h2>Backends</h2>
      <ul>
        <li>
          <code>uaii._uaii</code> — pybind11 native module when built
        </li>
        <li>
          ctypes over <code>uaii_capi</code> shared library (default after CMake build)
        </li>
      </ul>
      <p>
        Set <code>UAII_CAPI_PATH</code> if the loader cannot find the shared library.
      </p>
    </DocLayout>
  );
}
