# -*- coding: utf-8 -*-
"""Compile the ZZMI merged-skeleton compute shaders with the system HLSL compiler.

3dmigoto compiles these files when the mod loads, so a syntax error would only
show up in game.  Run with ``py`` on Windows (needs d3dcompiler_47.dll).
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from ctypes import wintypes
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "common" / "zzmi_merged_skeleton.py"
    spec = importlib.util.spec_from_file_location("zzmi_merged_skeleton_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ID3DBlob(ctypes.Structure):
    pass


def blob_text(blob) -> str:
    if not blob:
        return ""
    vtable = ctypes.cast(blob, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    get_pointer = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vtable[3])
    get_size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vtable[4])
    pointer, size = get_pointer(blob), get_size(blob)
    return ctypes.string_at(pointer, size).decode("utf-8", "replace").rstrip("\x00")


def compile_cs(compiler, source: str, name: str) -> None:
    code = ctypes.c_void_p()
    errors = ctypes.c_void_p()
    data = source.encode("utf-8")
    result = compiler(
        data, len(data), name.encode("ascii"), None, None,
        b"main", b"cs_5_0", 0, 0, ctypes.byref(code), ctypes.byref(errors),
    )
    message = blob_text(errors)
    if result != 0:
        raise AssertionError(name + " failed to compile (0x%08X):\n%s" % (result & 0xFFFFFFFF, message))
    if message.strip():
        print(name + " warnings:\n" + message)


def main() -> int:
    if sys.platform != "win32":
        print("ZZMI_MERGED_HLSL_COMPILE=SKIP (not Windows)")
        return 0
    module = load_module()
    library = ctypes.WinDLL("d3dcompiler_47.dll")
    compiler = library.D3DCompile
    compiler.restype = ctypes.c_long
    compiler.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_char_p, wintypes.UINT, wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ]
    compile_cs(compiler, module.IMPORT_SHADER_SOURCE, module.IMPORT_SHADER_FILENAME)
    compile_cs(compiler, module.PALETTE_SHADER_SOURCE, module.PALETTE_SHADER_FILENAME)
    compile_cs(compiler, module.DELAY_SHADER_SOURCE, module.DELAY_SHADER_FILENAME)
    compile_cs(compiler, module.PLACE_SHADER_SOURCE, module.PLACE_SHADER_FILENAME)
    compile_cs(compiler, module.SKIN_SHADER_SOURCE, module.SKIN_SHADER_FILENAME)
    compile_cs(compiler, module.INSTANCE_IMPORT_SHADER_SOURCE, module.INSTANCE_IMPORT_SHADER_FILENAME)
    compile_cs(compiler, module.INSTANCE_RECORD_SHADER_SOURCE, module.INSTANCE_RECORD_SHADER_FILENAME)
    compile_cs(compiler, module.INSTANCE_MATCH_SHADER_SOURCE, module.INSTANCE_MATCH_SHADER_FILENAME)
    compile_cs(compiler, module.INSTANCE_GATHER_SHADER_SOURCE, module.INSTANCE_GATHER_SHADER_FILENAME)
    print("ZZMI_MERGED_HLSL_COMPILE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
