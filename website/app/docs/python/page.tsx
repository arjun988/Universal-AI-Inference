import { DocLayout } from "@/components/DocLayout";

export const metadata = { title: "Python SDK" };

export default function Page() {
  return (
    <DocLayout
      title="Python SDK"
      active="/docs/python/"
      lede="Load a model or UAII IR, run inference, and capture profiles without reading C++ internals."
    >
      <h2>Install</h2>
      <pre>{`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel

python bindings/python/scripts/bundle_native.py --build-dir build
pip install -e bindings/python`}</pre>
      <p>
        Optional native extension: configure with <code>-DUAII_BUILD_PYTHON=ON</code> (pybind11
        via FetchContent). Otherwise the package uses ctypes against <code>uaii_capi</code>.
      </p>
      <p>
        If the shared library is not found, set <code>UAII_CAPI_PATH</code> to the full path of{" "}
        <code>uaii_capi</code>.
      </p>

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

      <h2>Defaults</h2>
      <ul>
        <li>
          <code>weight_init=&quot;none&quot;</code> — fail closed (no silent Ones fill)
        </li>
        <li>
          GPU backend names require a native build + device for real device kernels; otherwise
          host-fallback applies and is reported by doctor / capabilities
        </li>
      </ul>
    </DocLayout>
  );
}
