uaii_add_module(ir
  SOURCES
    src/version.cpp
    src/dtype.cpp
    src/graph.cpp
    src/registry.cpp
    src/builtins.cpp
    src/validator.cpp
    src/plan.cpp
    src/json_util.cpp
    src/serialize_json.cpp
    src/serialize_binary.cpp
    src/serialize.cpp
  DEPS uaii::core
)
