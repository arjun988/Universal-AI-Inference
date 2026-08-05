#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "uaii/c_api/uaii.h"
#include "uaii/ir/serialize.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/profiler/profiler.hpp"
#include "uaii/runtime/session.hpp"
#include "uaii/version.hpp"

namespace py = pybind11;

namespace {

uaii::runtime::WeightInit parse_weight_init(const std::string& s) {
  if (s == "zeros") return uaii::runtime::WeightInit::Zeros;
  if (s == "ones") return uaii::runtime::WeightInit::Ones;
  if (s == "sequence") return uaii::runtime::WeightInit::Sequence;
  return uaii::runtime::WeightInit::None;
}

class PySession {
 public:
  void create_from_path(const std::string& path,
                        const std::string& backend,
                        const std::string& weights_dir,
                        const std::string& weight_init,
                        bool profile,
                        const std::string& trace_path,
                        bool fusion) {
    uaii::ir::Graph graph;
    uaii::Error err;
    if (path.size() >= 5 &&
        (path.rfind(".uaii") != std::string::npos || path.rfind(".json") != std::string::npos)) {
      err = uaii::ir::load_graph(path, &graph);
    } else {
      err = uaii::loaders::load_model(path, &graph);
    }
    if (!err.ok()) throw std::runtime_error(err.to_string());

    uaii::runtime::SessionOptions opts;
    opts.backend_name = backend;
    opts.weights_dir = weights_dir;
    opts.weight_init = parse_weight_init(weight_init);
    opts.enable_profiler = profile;
    opts.profile_trace_path = trace_path;
    opts.enable_fusion = fusion;
    err = session_.create(std::move(graph), opts);
    if (!err.ok()) throw std::runtime_error(err.to_string());
  }

  void set_f32(const std::string& name, const std::vector<float>& values) {
    uaii::Error err = session_.set_tensor_f32(name, values);
    if (!err.ok()) throw std::runtime_error(err.to_string());
  }

  std::vector<float> get_f32(const std::string& name) {
    std::vector<float> out;
    uaii::Error err = session_.get_tensor_f32(name, &out);
    if (!err.ok()) throw std::runtime_error(err.to_string());
    return out;
  }

  void run() {
    uaii::Error err = session_.run();
    if (!err.ok()) throw std::runtime_error(err.to_string());
  }

  std::string profile_summary() const { return session_.profiler().summary(); }

  void write_trace(const std::string& path) {
    uaii::Error err = uaii::profiler::write_chrome_trace(session_.profiler(), path);
    if (!err.ok()) throw std::runtime_error(err.to_string());
  }

  std::string debug_stats() const { return session_.debug_stats(); }

 private:
  uaii::runtime::Session session_;
};

void convert_model(const std::string& inp, const std::string& out) {
  uaii::Error err = uaii::loaders::convert_model(inp, out);
  if (!err.ok()) throw std::runtime_error(err.to_string());
}

}  // namespace

PYBIND11_MODULE(_uaii, m) {
  m.doc() = "UAII Runtime native extension (Phase 7)";
  m.def("version", []() { return std::string(uaii::version_string()); });
  m.def("c_api_version", []() { return std::string(UAII_C_API_VERSION_STRING); });
  m.def("convert_model", &convert_model, py::arg("input_path"), py::arg("output_path"));

  py::class_<PySession>(m, "NativeSession")
      .def(py::init<>())
      .def("create_from_path", &PySession::create_from_path,
           py::arg("path"),
           py::arg("backend") = "cpu",
           py::arg("weights_dir") = "",
           py::arg("weight_init") = "none",
           py::arg("profile") = false,
           py::arg("trace_path") = "",
           py::arg("fusion") = true)
      .def("set_f32", &PySession::set_f32)
      .def("get_f32", &PySession::get_f32)
      .def("run", &PySession::run)
      .def("profile_summary", &PySession::profile_summary)
      .def("write_trace", &PySession::write_trace)
      .def("debug_stats", &PySession::debug_stats);
}
