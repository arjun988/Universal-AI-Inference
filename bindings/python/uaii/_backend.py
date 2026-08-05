"""Select native pybind or ctypes C-API backend."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_char_p, c_float, c_int, c_size_t, c_uint64, c_void_p
from pathlib import Path
from typing import List, Optional, Protocol, Sequence


class UaiiError(RuntimeError):
    pass


class _SessionImpl(Protocol):
    def create(
        self,
        path: str,
        *,
        backend: str,
        weights_dir: str,
        weight_init: str,
        profile: bool,
        trace_path: str,
        fusion: bool,
    ) -> None: ...

    def set_f32(self, name: str, values: Sequence[float]) -> None: ...
    def get_f32(self, name: str) -> List[float]: ...
    def run(self) -> None: ...
    def profile_summary(self) -> str: ...
    def write_trace(self, path: str) -> None: ...
    def debug_stats(self) -> str: ...


def _try_native():
    try:
        from . import _uaii  # type: ignore

        return _uaii
    except Exception:
        return None


_NATIVE = _try_native()


class _NativeSession:
    def __init__(self) -> None:
        assert _NATIVE is not None
        self._s = _NATIVE.NativeSession()

    def create(self, path: str, **kw) -> None:
        self._s.create_from_path(
            path,
            backend=kw.get("backend", "cpu"),
            weights_dir=kw.get("weights_dir", ""),
            weight_init=kw.get("weight_init", "none"),
            profile=kw.get("profile", False),
            trace_path=kw.get("trace_path", ""),
            fusion=kw.get("fusion", True),
        )

    def set_f32(self, name: str, values: Sequence[float]) -> None:
        self._s.set_f32(name, list(values))

    def get_f32(self, name: str) -> List[float]:
        return list(self._s.get_f32(name))

    def run(self) -> None:
        self._s.run()

    def profile_summary(self) -> str:
        return self._s.profile_summary()

    def write_trace(self, path: str) -> None:
        self._s.write_trace(path)

    def debug_stats(self) -> str:
        return self._s.debug_stats()


class _CTypesOpts(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("backend", c_char_p),
        ("weights_dir", c_char_p),
        ("weight_init", c_int),
        ("enable_fusion", c_int),
        ("enable_memory_reuse", c_int),
        ("enable_profiler", c_int),
        ("profile_trace_path", c_char_p),
        ("budget_bytes", c_uint64),
        ("enable_streaming", c_int),
        ("allow_missing_weights", c_int),
        ("weights_sandbox", c_char_p),
        ("compute_dtype", c_int),
        ("keep_quantized_weights", c_int),
        ("max_context", ctypes.c_int64),
    ]


def _lib_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["uaii_capi.dll", "libuaii_capi.dll"]
    if sys.platform == "darwin":
        return ["libuaii_capi.dylib"]
    return ["libuaii_capi.so"]


def _find_capi() -> Optional[Path]:
    env = os.environ.get("UAII_CAPI_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # Packaged wheel / sdist: native lib shipped under uaii/_native/
    for name in _lib_names():
        candidates.append(here.parent / "_native" / name)
    # Walk up to repo root and common build dirs
    for parent in [here.parent, *here.parents]:
        for name in _lib_names():
            candidates.extend(
                [
                    parent / name,
                    parent / "build" / name,
                    parent / "build" / "Release" / name,
                    parent / "build" / "Debug" / name,
                    parent / "build" / "libs" / "uaii-capi" / name,
                    parent / "build" / "libs" / "uaii-capi" / "Release" / name,
                    parent / "build" / "libs" / "uaii-capi" / "Debug" / name,
                ]
            )
        if (parent / "CMakeLists.txt").exists() and parent.name != "uaii":
            break
    for c in candidates:
        if c.is_file():
            return c
    return None


class _CTypesLib:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self) -> None:
        L = self.lib
        L.uaii_get_version_string.restype = c_char_p
        L.uaii_get_c_api_version_string.restype = c_char_p
        L.uaii_last_error.restype = c_char_p
        L.uaii_status_name.restype = c_char_p
        L.uaii_status_name.argtypes = [c_int]
        L.uaii_session_options_init.argtypes = [POINTER(_CTypesOpts)]
        L.uaii_session_create.argtypes = [c_char_p, c_void_p, POINTER(c_void_p)]
        L.uaii_session_create.restype = c_int
        L.uaii_session_destroy.argtypes = [c_void_p]
        L.uaii_session_set_f32.argtypes = [c_void_p, c_char_p, POINTER(c_float), c_size_t]
        L.uaii_session_set_f32.restype = c_int
        L.uaii_session_get_f32.argtypes = [
            c_void_p,
            c_char_p,
            POINTER(c_float),
            c_size_t,
            POINTER(c_size_t),
        ]
        L.uaii_session_get_f32.restype = c_int
        L.uaii_session_run.argtypes = [c_void_p]
        L.uaii_session_run.restype = c_int
        L.uaii_session_profile_summary.argtypes = [c_void_p, c_char_p, c_size_t]
        L.uaii_session_profile_summary.restype = c_int
        L.uaii_session_write_trace.argtypes = [c_void_p, c_char_p]
        L.uaii_session_write_trace.restype = c_int
        L.uaii_convert_model.argtypes = [c_char_p, c_char_p]
        L.uaii_convert_model.restype = c_int

    def check(self, status: int) -> None:
        if status != 0:
            err = self.lib.uaii_last_error()
            msg = err.decode("utf-8", "replace") if err else f"status={status}"
            raise UaiiError(msg)


_CTYPES: Optional[_CTypesLib] = None


def _ctypes() -> _CTypesLib:
    global _CTYPES
    if _CTYPES is None:
        path = _find_capi()
        if path is None:
            raise UaiiError(
                "Neither uaii._uaii nor uaii_capi shared library found. "
                "Build the runtime (uaii_capi) or configure -DUAII_BUILD_PYTHON=ON. "
                "Set UAII_CAPI_PATH to the library if needed."
            )
        _CTYPES = _CTypesLib(path)
    return _CTYPES


_WEIGHT = {"none": 0, "zeros": 1, "ones": 2, "sequence": 3}


class _CTypesSession:
    def __init__(self) -> None:
        self._lib = _ctypes()
        self._ptr = c_void_p()
        self._keep: list[bytes] = []

    def create(self, path: str, **kw) -> None:
        opts = _CTypesOpts()
        self._lib.lib.uaii_session_options_init(ctypes.byref(opts))
        b_backend = kw.get("backend", "cpu").encode()
        b_weights = kw.get("weights_dir", "").encode()
        b_trace = kw.get("trace_path", "").encode()
        self._keep = [b_backend, b_weights, b_trace]
        opts.backend = b_backend
        opts.weights_dir = b_weights if b_weights else None
        opts.weight_init = _WEIGHT.get(kw.get("weight_init", "none"), 0)
        opts.struct_size = ctypes.sizeof(_CTypesOpts)
        opts.enable_fusion = 1 if kw.get("fusion", True) else 0
        opts.enable_memory_reuse = 1
        opts.enable_profiler = 1 if kw.get("profile", False) else 0
        opts.profile_trace_path = b_trace if b_trace else None
        out = c_void_p()
        st = self._lib.lib.uaii_session_create(
            path.encode(), ctypes.byref(opts), ctypes.byref(out)
        )
        self._lib.check(st)
        self._ptr = out

    def set_f32(self, name: str, values: Sequence[float]) -> None:
        arr = (c_float * len(values))(*values)
        st = self._lib.lib.uaii_session_set_f32(
            self._ptr, name.encode(), arr, c_size_t(len(values))
        )
        self._lib.check(st)

    def get_f32(self, name: str) -> List[float]:
        n = c_size_t(0)
        st = self._lib.lib.uaii_session_get_f32(
            self._ptr, name.encode(), None, c_size_t(0), ctypes.byref(n)
        )
        self._lib.check(st)
        arr = (c_float * n.value)()
        st = self._lib.lib.uaii_session_get_f32(
            self._ptr, name.encode(), arr, n, ctypes.byref(n)
        )
        self._lib.check(st)
        return [arr[i] for i in range(n.value)]

    def run(self) -> None:
        self._lib.check(self._lib.lib.uaii_session_run(self._ptr))

    def profile_summary(self) -> str:
        buf = ctypes.create_string_buffer(4096)
        self._lib.check(
            self._lib.lib.uaii_session_profile_summary(self._ptr, buf, len(buf))
        )
        return buf.value.decode("utf-8", "replace")

    def write_trace(self, path: str) -> None:
        self._lib.check(self._lib.lib.uaii_session_write_trace(self._ptr, path.encode()))

    def debug_stats(self) -> str:
        return self.profile_summary()

    def __del__(self) -> None:
        if getattr(self, "_ptr", None) and self._ptr:
            try:
                self._lib.lib.uaii_session_destroy(self._ptr)
            except Exception:
                pass


def create_session() -> _SessionImpl:
    if _NATIVE is not None:
        return _NativeSession()
    return _CTypesSession()


def version() -> str:
    if _NATIVE is not None:
        return str(_NATIVE.version())
    return _ctypes().lib.uaii_get_version_string().decode("utf-8")


def c_api_version() -> str:
    if _NATIVE is not None:
        return str(_NATIVE.c_api_version())
    return _ctypes().lib.uaii_get_c_api_version_string().decode("utf-8")


def convert_model(input_path: str, output_path: str) -> None:
    if _NATIVE is not None:
        _NATIVE.convert_model(input_path, output_path)
        return
    lib = _ctypes()
    lib.check(lib.lib.uaii_convert_model(input_path.encode(), output_path.encode()))
