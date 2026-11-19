"""Minimal windowless D3D11 WARP compute test harness; owns every COM resource."""
import ctypes as C
from pathlib import Path

import numpy as np

U, P, H = C.c_uint32, C.c_void_p, C.c_long


def check(hr):
    if hr < 0:
        raise OSError(f"D3D11 HRESULT 0x{hr & 0xffffffff:08x}")


def method(obj, slot, result, *arguments):
    pointer = C.cast(obj, C.POINTER(C.POINTER(P))).contents[slot]
    return C.WINFUNCTYPE(result, P, *arguments)(pointer)


def release(obj):
    if obj and obj.value:
        method(obj, 2, U)(obj)
        obj.value = None


class BufferDesc(C.Structure):
    _fields_ = [(name, U) for name in ("bytes", "usage", "bind", "cpu", "misc", "stride")]


class TextureDesc(C.Structure):
    _fields_ = [(name, U) for name in ("width", "mips", "array", "format", "usage", "bind", "cpu", "misc")]


class InitialData(C.Structure):
    _fields_ = [("data", P), ("pitch", U), ("slice", U)]


class Mapped(C.Structure):
    _fields_ = [("data", P), ("row_pitch", U), ("depth_pitch", U)]


class UAVDesc(C.Structure):
    _fields_ = [(name, U) for name in ("format", "dimension", "first", "count", "flags")]


def compile_shader(path):
    dll = C.WinDLL("d3dcompiler_47.dll")
    function = dll.D3DCompile
    function.argtypes = [P, C.c_size_t, C.c_char_p, P, P, C.c_char_p, C.c_char_p, U, U, C.POINTER(P), C.POINTER(P)]
    function.restype = H
    source = Path(path).read_bytes()
    blob, errors = P(), P()
    result = function(source, len(source), str(path).encode(), None, None, b"main", b"cs_5_0", 1 << 11, 0,
                      C.byref(blob), C.byref(errors))
    def contents(value):
        if not value.value:
            return b""
        return C.string_at(method(value, 3, P)(value), method(value, 4, C.c_size_t)(value))
    try:
        if result < 0:
            raise RuntimeError(contents(errors).decode(errors="replace"))
        return contents(blob)
    finally:
        release(errors)
        release(blob)


class Warp:
    def __init__(self):
        self.device, self.context, feature = P(), P(), U()
        self.resources = []
        dll = C.WinDLL("d3d11.dll")
        create = dll.D3D11CreateDevice
        create.argtypes = [P, U, P, U, P, U, U, C.POINTER(P), C.POINTER(U), C.POINTER(P)]
        create.restype = H
        levels = (U * 1)(0xb000)
        check(create(None, 5, None, 0, levels, 1, 7,
                     C.byref(self.device), C.byref(feature), C.byref(self.context)))
        self.feature_level = feature.value

    def own(self, pointer):
        self.resources.append(pointer)
        return pointer

    def close(self):
        if self.context.value:
            method(self.context, 110, None)(self.context)
        for resource in reversed(self.resources):
            release(resource)
        self.resources.clear()
        release(self.context)
        release(self.device)

    def buffer(self, values, *, bind=0x88, structured=True, staging=False):
        values = np.ascontiguousarray(values)
        desc = BufferDesc(values.nbytes, 3 if staging else 0, 0 if staging else bind,
                          0x20000 if staging else 0, 0x40 if structured else 0, 16 if structured else 0)
        initial = InitialData(values.ctypes.data, 0, 0)
        result = P()
        check(method(self.device, 3, H, P, P, P)(self.device, C.byref(desc), None if staging else C.byref(initial), C.byref(result)))
        return self.own(result)

    def texture(self, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        desc = TextureDesc(len(values), 1, 1, 2, 0, 8, 0, 0)
        initial = InitialData(values.ctypes.data, values.nbytes, 0)
        result = P()
        check(method(self.device, 4, H, P, P, P)(self.device, C.byref(desc), C.byref(initial), C.byref(result)))
        return self.own(result)

    def view(self, resource, *, uav=False, typed_count=None):
        result = P()
        desc = UAVDesc(3, 1, 0, typed_count, 0) if typed_count is not None else None
        check(method(self.device, 8 if uav else 7, H, P, P, P)(
            self.device, resource, C.byref(desc) if desc is not None else None, C.byref(result)))
        return self.own(result)

    def shader(self, path):
        binary = compile_shader(path)
        result = P()
        check(method(self.device, 18, H, P, C.c_size_t, P, P)(self.device, binary, len(binary), None, C.byref(result)))
        return self.own(result)

    def update(self, resource, values):
        values = np.ascontiguousarray(values)
        method(self.context, 48, None, P, U, P, P, U, U)(
            self.context, resource, 0, None, values.ctypes.data, values.nbytes, 0)

    def dispatch(self, shader, srvs, uavs, groups=1, constant_buffer=None):
        context = self.context
        # Unbind first to avoid SRV/UAV aliasing warnings across test dispatches.
        empty = (P * 3)()
        method(context, 68, None, U, U, P, P)(context, 0, 3, empty, None)
        for slot, view in srvs.items():
            values = (P * 1)(view.value)
            method(context, 67, None, U, U, P)(context, slot, 1, values)
        values = (P * len(uavs))(*(value.value if value else None for value in uavs))
        method(context, 68, None, U, U, P, P)(context, 0, len(uavs), values, None)
        method(context, 69, None, P, P, U)(context, shader, None, 0)
        if constant_buffer is not None:
            values = (P * 1)(constant_buffer.value)
            method(context, 71, None, U, U, P)(context, 0, 1, values)
        method(context, 41, None, U, U, U)(context, groups, 1, 1)
        method(context, 68, None, U, U, P, P)(context, 0, 3, empty, None)

    def read(self, resource, shape, dtype=np.float32):
        values = np.empty(shape, dtype=dtype)
        # Staging buffers are not structured resources; CopyResource cares
        # about byte dimensions, while Map exposes the exact original bytes.
        stage = self.buffer(values, bind=0, structured=False, staging=True)
        method(self.context, 47, None, P, P)(self.context, stage, resource)
        mapped = Mapped()
        check(method(self.context, 14, H, P, U, U, U, P)(self.context, stage, 0, 1, 0, C.byref(mapped)))
        try:
            C.memmove(values.ctypes.data, mapped.data, values.nbytes)
        finally:
            method(self.context, 15, None, P, U)(self.context, stage, 0)
        return values
