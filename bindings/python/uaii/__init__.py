"""Universal AI Inference Runtime — Python SDK (Phase 7).

High-level API for load → run → profile without reading C++ internals.

Backends (in order):
  1. Native pybind11 module ``uaii._uaii`` (build with ``-DUAII_BUILD_PYTHON=ON``)
  2. ctypes loader over the shared library ``uaii_capi``
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

from . import _backend

__all__ = [
    "Session",
    "convert_model",
    "version",
    "c_api_version",
    "UaiiError",
]

UaiiError = _backend.UaiiError


def version() -> str:
    return _backend.version()


def c_api_version() -> str:
    return _backend.c_api_version()


def convert_model(input_path: str, output_path: str) -> None:
    """Convert GGUF/Safetensors → UAII IR (.uaii.json / .uaii)."""
    _backend.convert_model(input_path, output_path)


class Session:
    """Inference session: load IR or model, set inputs, run, read outputs, profile."""

    def __init__(self) -> None:
        self._impl = _backend.create_session()

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        backend: str = "cpu",
        weights_dir: str = "",
        weight_init: str = "ones",
        profile: bool = False,
        trace_path: str = "",
        fusion: bool = True,
    ) -> "Session":
        """Open a UAII IR file or model file (GGUF / Safetensors)."""
        s = cls()
        s._impl.create(
            path,
            backend=backend,
            weights_dir=weights_dir,
            weight_init=weight_init,
            profile=profile,
            trace_path=trace_path,
            fusion=fusion,
        )
        return s

    @classmethod
    def from_model(cls, path: str, **kwargs) -> "Session":
        """Alias for :meth:`from_path` (load model or IR)."""
        return cls.from_path(path, **kwargs)

    def set_tensor(self, name: str, values: Sequence[float]) -> None:
        self._impl.set_f32(name, list(float(v) for v in values))

    def get_tensor(self, name: str) -> List[float]:
        return list(self._impl.get_f32(name))

    def run(self) -> None:
        self._impl.run()

    def profile_summary(self) -> str:
        return self._impl.profile_summary()

    def write_trace(self, path: str) -> None:
        self._impl.write_trace(path)

    def debug_stats(self) -> str:
        return self._impl.debug_stats()
