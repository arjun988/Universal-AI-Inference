#!/usr/bin/env python3
"""Emit embedded SPIR-V for UAII Vulkan compute kernels (stdlib only)."""

from __future__ import annotations

import struct
from pathlib import Path

# SPIR-V 1.0 opcodes (subset)
OP_CAPABILITY = 17
OP_EXTENSION = 10
OP_EXT_INST_IMPORT = 11
OP_MEMORY_MODEL = 14
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
OP_NAME = 5
OP_DECORATE = 71
OP_TYPE_VOID = 19
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 23
OP_TYPE_VECTOR = 18
OP_TYPE_POINTER = 22
OP_TYPE_FUNCTION = 33
OP_TYPE_STRUCT = 30
OP_TYPE_RUNTIME_ARRAY = 29
OP_CONSTANT = 43
OP_VARIABLE = 59
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_LABEL = 248
OP_RETURN = 253
OP_LOAD = 61
OP_STORE = 62
OP_ACCESS_CHAIN = 65
OP_FADD = 129
OP_FMUL = 131
OP_EXT_INST = 12
OP_UL_LESS_THAN = 151
OP_BRANCH_CONDITIONAL = 250
OP_BRANCH = 249
OP_COMPOSITE_EXTRACT = 81
OP_CONVERT_U_TO_F = 114
OP_CONVERT_F_TO_U = 115
OP_IADD = 130
OP_FDIV = 152
OP_FNEG = 127

CAP_SHADER = 1
MEM_LOGICAL = 0
MEM_GLSL450 = 1
EXEC_GLCOMPUTE = 5
EXEC_LOCAL_SIZE = 17
STORAGE_INPUT = 1
STORAGE_UNIFORM = 2
STORAGE_FUNCTION = 7
STORAGE_SSBO = 12
DEC_BUILTIN = 11
DEC_BINDING = 33
DEC_DESCRIPTOR_SET = 34
DEC_NON_WRITABLE = 39
DEC_NON_READABLE = 40
BUILTIN_GLOBAL_INVOCATION_ID = 28
EXT_GLSL450_SQRT = 31
EXT_GLSL450_RSQRT = 32


class Spv:
    def __init__(self) -> None:
        self.next_id = 1
        self.body: list[int] = []

    def id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def emit(self, op: int, *ops: int) -> None:
        self.body.append((len(ops) + 1) << 16 | op)
        self.body.extend(ops)

    def str_words(self, s: str) -> list[int]:
        b = s.encode("ascii") + b"\0"
        while len(b) % 4:
            b += b"\0"
        return list(struct.unpack(f"<{len(b)//4}I", b))

    def emit_str(self, op: int, *ops: int, text: str = "") -> None:
        words = list(ops) + self.str_words(text)
        self.body.append((len(words) + 1) << 16 | op)
        self.body.extend(words)

    def build(self) -> bytes:
        hdr = struct.pack("<IIIII", 0x07230203, 0x00010500, 0, self.next_id, 0)
        return hdr + b"".join(struct.pack("<I", w) for w in self.body)


def gen_add() -> bytes:
    s = Spv()
    void_t = s.id()
    float_t = s.id()
    uint_t = s.id()
    v3uint_t = s.id()
    ptr_in_v3 = s.id()
    ptr_f_ro = s.id()
    ptr_f_wo = s.id()
    ptr_u = s.id()
    ssbo_a = s.id()
    ssbo_b = s.id()
    ssbo_c = s.id()
    ubo = s.id()
    glsl = s.id()
    fn_t = s.id()
    main = s.id()
    gid_v = s.id()
    arr_f = s.id()
    struct_ssbo = s.id()
    struct_ubo = s.id()
    c0 = s.id()
    c256 = s.id()
    bb0 = s.id()
    bb1 = s.id()
    bb2 = s.id()
    gid = s.id()
    idx = s.id()
    n = s.id()
    cmp = s.id()
    pa = s.id()
    pb = s.id()
    pc = s.id()
    va = s.id()
    vb = s.id()
    vc = s.id()
    pu = s.id()

    s.emit(OP_CAPABILITY, CAP_SHADER)
    s.emit_str(OP_EXT_INST_IMPORT, glsl, text="GLSL.std.450")
    s.emit(OP_MEMORY_MODEL, MEM_LOGICAL, MEM_GLSL450)
    ep_words = [EXEC_GLCOMPUTE, main] + s.str_words("main") + [gid_v]
    s.body.append((len(ep_words) + 1) << 16 | OP_ENTRY_POINT)
    s.body.extend(ep_words)
    s.emit(OP_EXECUTION_MODE, main, EXEC_LOCAL_SIZE, 256, 1, 1)
    for var, name in ((ssbo_a, "A"), (ssbo_b, "B"), (ssbo_c, "C"), (ubo, "Params")):
        s.emit_str(OP_NAME, var, text=name)
    s.emit_str(OP_NAME, gid_v, text="gl_GlobalInvocationID")
    s.emit(OP_DECORATE, gid_v, DEC_BUILTIN, BUILTIN_GLOBAL_INVOCATION_ID)
    for var, bind in ((ssbo_a, 0), (ssbo_b, 1), (ssbo_c, 2), (ubo, 3)):
        s.emit(OP_DECORATE, var, DEC_DESCRIPTOR_SET, 0)
        s.emit(OP_DECORATE, var, DEC_BINDING, bind)
    s.emit(OP_DECORATE, ssbo_a, DEC_NON_WRITABLE)
    s.emit(OP_DECORATE, ssbo_b, DEC_NON_WRITABLE)
    s.emit(OP_DECORATE, ssbo_c, DEC_NON_READABLE)

    s.emit(OP_TYPE_VOID, void_t)
    s.emit(OP_TYPE_FLOAT, float_t, 32)
    s.emit(OP_TYPE_INT, uint_t, 32, 0)
    s.emit(OP_TYPE_VECTOR, v3uint_t, uint_t, 3)
    s.emit(OP_TYPE_POINTER, ptr_in_v3, STORAGE_INPUT, v3uint_t)
    s.emit(OP_TYPE_RUNTIME_ARRAY, arr_f, float_t)
    s.emit(OP_TYPE_STRUCT, struct_ssbo, arr_f)
    s.emit(OP_TYPE_POINTER, ptr_f_ro, STORAGE_SSBO, float_t)
    s.emit(OP_TYPE_POINTER, ptr_f_wo, STORAGE_SSBO, float_t)
    s.emit(OP_TYPE_STRUCT, struct_ubo, uint_t)
    s.emit(OP_TYPE_POINTER, ptr_u, STORAGE_UNIFORM, struct_ubo)
    s.emit(OP_TYPE_FUNCTION, fn_t, void_t)

    s.emit(OP_CONSTANT, uint_t, c0, 0)
    s.emit(OP_CONSTANT, uint_t, c256, 256)

    s.emit(OP_VARIABLE, ptr_in_v3, gid_v, STORAGE_INPUT)
    s.emit(OP_VARIABLE, ptr_f_ro, ssbo_a, STORAGE_SSBO)
    s.emit(OP_VARIABLE, ptr_f_ro, ssbo_b, STORAGE_SSBO)
    s.emit(OP_VARIABLE, ptr_f_wo, ssbo_c, STORAGE_SSBO)
    s.emit(OP_VARIABLE, ptr_u, ubo, STORAGE_UNIFORM)

    s.emit(OP_FUNCTION, void_t, main, 0, fn_t)
    s.emit(OP_LABEL, bb0)
    s.emit(OP_LOAD, v3uint_t, gid, ptr_in_v3, gid_v)
    s.emit(OP_COMPOSITE_EXTRACT, uint_t, idx, gid, c0)
    s.emit(OP_ACCESS_CHAIN, ptr_u, pu, ubo, c0)
    s.emit(OP_LOAD, uint_t, n, pu)
    s.emit(OP_UL_LESS_THAN, cmp, idx, n)
    s.emit(OP_BRANCH_CONDITIONAL, cmp, bb1, bb2)
    s.emit(OP_LABEL, bb1)
    s.emit(OP_ACCESS_CHAIN, ptr_f_ro, pa, ssbo_a, idx)
    s.emit(OP_ACCESS_CHAIN, ptr_f_ro, pb, ssbo_b, idx)
    s.emit(OP_ACCESS_CHAIN, ptr_f_wo, pc, ssbo_c, idx)
    s.emit(OP_LOAD, float_t, va, pa)
    s.emit(OP_LOAD, float_t, vb, pb)
    s.emit(OP_FADD, float_t, vc, va, vb)
    s.emit(OP_STORE, pc, vc)
    s.emit(OP_BRANCH, bb2)
    s.emit(OP_LABEL, bb2)
    s.emit(OP_RETURN)
    s.emit(OP_FUNCTION_END)
    return s.build()


def gen_matmul() -> bytes:
    """Reuse add module until dedicated matmul SPIR-V is checked in."""
    return gen_add()


def gen_rmsnorm() -> bytes:
    return gen_add()


def words_to_cpp(name: str, data: bytes) -> str:
    u32 = list(struct.unpack(f"<{len(data)//4}I", data))
    lines = [f"inline constexpr std::uint32_t {name}[] = {{"]
    for i in range(0, len(u32), 6):
        chunk = u32[i : i + 6]
        lines.append("  " + ", ".join(f"0x{w:08x}u" for w in chunk) + ",")
    lines.append("};")
    lines.append(f"inline constexpr std::size_t {name}_words = {len(u32)};")
    return "\n".join(lines)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "libs" / "uaii-backends" / "src" / "vulkan_spirv.hpp"
    add = gen_add()
    matmul = gen_matmul()
    rmsnorm = gen_rmsnorm()
    content = """// Generated by tools/gen_vulkan_spirv.py — embedded SPIR-V compute modules.
#pragma once

#include <cstddef>
#include <cstdint>

namespace uaii {
namespace backends {
namespace native {
namespace vulkan_spirv {

"""
    content += words_to_cpp("kAddSpv", add) + "\n\n"
    content += words_to_cpp("kMatMulSpv", matmul) + "\n\n"
    content += words_to_cpp("kRmsNormSpv", rmsnorm) + "\n\n"
    content += "}  // namespace vulkan_spirv\n}  // namespace native\n}  // namespace backends\n}  // namespace uaii\n"
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
