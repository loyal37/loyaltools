# -*- coding: utf-8 -*-
"""Run the ZZMI merged-skeleton compute shaders on a D3D11 WARP device.

The shaders are executed exactly as the mod ini drives them: one import
dispatch per component (palette bound as a buffer view, read through the
shader's tbuffer), then the palette builder for every component.  The result is
compared with ``reference_palette`` from common/zzmi_merged_skeleton.py.

Windows only; WARP is the software rasterizer that ships with Windows, so no
GPU is needed.  ``py tests/zzmi_merged_gpu_runtime_check.py``
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from ctypes import wintypes
from pathlib import Path

import numpy


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "common" / "zzmi_merged_skeleton.py"
    spec = importlib.util.spec_from_file_location("zzmi_merged_skeleton_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Minimal D3D11 compute through COM vtables
# ---------------------------------------------------------------------------

DXGI_FORMAT_R32G32B32A32_FLOAT = 2
DXGI_FORMAT_R32G32B32A32_UINT = 3
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_UNORDERED_ACCESS = 0x80
D3D11_USAGE_DEFAULT = 0
D3D11_USAGE_STAGING = 3
D3D11_CPU_ACCESS_READ = 0x20000
D3D11_MAP_READ = 1
D3D_DRIVER_TYPE_WARP = 5
D3D11_SDK_VERSION = 7


class BufferDesc(ctypes.Structure):
    _fields_ = [("ByteWidth", wintypes.UINT), ("Usage", wintypes.UINT), ("BindFlags", wintypes.UINT),
                ("CPUAccessFlags", wintypes.UINT), ("MiscFlags", wintypes.UINT),
                ("StructureByteStride", wintypes.UINT)]


class SubresourceData(ctypes.Structure):
    _fields_ = [("pSysMem", ctypes.c_void_p), ("SysMemPitch", wintypes.UINT), ("SysMemSlicePitch", wintypes.UINT)]


class SrvDesc(ctypes.Structure):
    _fields_ = [("Format", wintypes.UINT), ("ViewDimension", wintypes.UINT),
                ("FirstElement", wintypes.UINT), ("NumElements", wintypes.UINT),
                ("_pad", wintypes.UINT * 2)]


class UavDesc(ctypes.Structure):
    _fields_ = [("Format", wintypes.UINT), ("ViewDimension", wintypes.UINT),
                ("FirstElement", wintypes.UINT), ("NumElements", wintypes.UINT), ("Flags", wintypes.UINT)]


class MappedSubresource(ctypes.Structure):
    _fields_ = [("pData", ctypes.c_void_p), ("RowPitch", wintypes.UINT), ("DepthPitch", wintypes.UINT)]


def _method(obj, index, restype, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    function = prototype(vtable[index])
    return lambda *args: function(obj, *args)


def _check(result, what):
    if result != 0:
        raise RuntimeError(what + " failed: 0x%08X" % (result & 0xFFFFFFFF))


class Compute:
    def __init__(self):
        d3d11 = ctypes.WinDLL("d3d11.dll")
        device = ctypes.c_void_p()
        context = ctypes.c_void_p()
        level = wintypes.UINT()
        _check(d3d11.D3D11CreateDevice(
            None, D3D_DRIVER_TYPE_WARP, None, 0, None, 0, D3D11_SDK_VERSION,
            ctypes.byref(device), ctypes.byref(level), ctypes.byref(context),
        ), "D3D11CreateDevice")
        self.device, self.context = device.value, context.value
        compiler = ctypes.WinDLL("d3dcompiler_47.dll").D3DCompile
        compiler.restype = ctypes.c_long
        compiler.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_char_p, wintypes.UINT, wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        self.compiler = compiler

    def shader(self, source: str):
        code, errors = ctypes.c_void_p(), ctypes.c_void_p()
        data = source.encode("utf-8")
        _check(self.compiler(data, len(data), b"shader", None, None, b"main", b"cs_5_0", 0, 0,
                             ctypes.byref(code), ctypes.byref(errors)), "D3DCompile")
        pointer = _method(code.value, 3, ctypes.c_void_p)()
        size = _method(code.value, 4, ctypes.c_size_t)()
        shader = ctypes.c_void_p()
        _check(_method(self.device, 18, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_void_p))(pointer, size, None, ctypes.byref(shader)),
               "CreateComputeShader")
        return shader.value

    def buffer(self, data: numpy.ndarray, uav: bool):
        data = numpy.ascontiguousarray(data)
        desc = BufferDesc(data.nbytes, D3D11_USAGE_DEFAULT,
                          D3D11_BIND_SHADER_RESOURCE | (D3D11_BIND_UNORDERED_ACCESS if uav else 0), 0, 0, 0)
        initial = SubresourceData(data.ctypes.data, 0, 0)
        buffer = ctypes.c_void_p()
        _check(_method(self.device, 3, ctypes.c_long, ctypes.POINTER(BufferDesc), ctypes.POINTER(SubresourceData),
                       ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(desc), ctypes.byref(initial), ctypes.byref(buffer)),
               "CreateBuffer")
        return buffer.value

    def srv(self, buffer, fmt: int, elements: int):
        desc = SrvDesc(fmt, 1, 0, elements)
        view = ctypes.c_void_p()
        _check(_method(self.device, 7, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(SrvDesc),
                       ctypes.POINTER(ctypes.c_void_p))(buffer, ctypes.byref(desc), ctypes.byref(view)),
               "CreateShaderResourceView")
        return view.value

    def uav(self, buffer, fmt: int, elements: int):
        desc = UavDesc(fmt, 1, 0, elements, 0)
        view = ctypes.c_void_p()
        _check(_method(self.device, 8, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(UavDesc),
                       ctypes.POINTER(ctypes.c_void_p))(buffer, ctypes.byref(desc), ctypes.byref(view)),
               "CreateUnorderedAccessView")
        return view.value

    def dispatch(self, shader, srvs: list, uav, groups: int):
        srv_array = (ctypes.c_void_p * len(srvs))(*srvs)
        uav_array = (ctypes.c_void_p * 1)(uav)
        null_srvs = (ctypes.c_void_p * len(srvs))()
        null_uav = (ctypes.c_void_p * 1)()
        set_srvs = _method(self.context, 67, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p)
        set_uavs = _method(self.context, 68, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p)
        _method(self.context, 69, None, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT)(shader, None, 0)
        set_srvs(0, len(srvs), srv_array)
        set_uavs(0, 1, uav_array, None)
        _method(self.context, 41, None, wintypes.UINT, wintypes.UINT, wintypes.UINT)(groups, 1, 1)
        set_uavs(0, 1, null_uav, None)
        set_srvs(0, len(srvs), null_srvs)

    def read(self, buffer, nbytes: int) -> numpy.ndarray:
        desc = BufferDesc(nbytes, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, 0)
        staging = ctypes.c_void_p()
        _check(_method(self.device, 3, ctypes.c_long, ctypes.POINTER(BufferDesc), ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(desc), None, ctypes.byref(staging)),
               "CreateBuffer(staging)")
        _method(self.context, 47, None, ctypes.c_void_p, ctypes.c_void_p)(staging.value, buffer)
        mapped = MappedSubresource()
        _check(_method(self.context, 14, ctypes.c_long, ctypes.c_void_p, wintypes.UINT, wintypes.UINT, wintypes.UINT,
                       ctypes.POINTER(MappedSubresource))(staging.value, 0, D3D11_MAP_READ, 0, ctypes.byref(mapped)),
               "Map")
        data = numpy.frombuffer(ctypes.string_at(mapped.pData, nbytes), dtype=numpy.uint8).copy()
        _method(self.context, 15, None, ctypes.c_void_p, wintypes.UINT)(staging.value, 0)
        return data


def run_runtime(module, profile: dict, palettes: dict) -> dict:
    """Import every merged component's palette, then build every palette on WARP."""
    gpu = Compute()
    import_shader = gpu.shader(module.IMPORT_SHADER_SOURCE)
    palette_shader = gpu.shader(module.PALETTE_SHADER_SOURCE)
    bones_count = int(profile["bones_count"])
    rows = bones_count * 3

    pool = gpu.buffer(numpy.zeros((rows, 4), numpy.float32), uav=True)
    pool_srv = gpu.srv(pool, DXGI_FORMAT_R32G32B32A32_FLOAT, rows)
    pool_uav = gpu.uav(pool, DXGI_FORMAT_R32G32B32A32_FLOAT, rows)
    table = numpy.asarray(module.runtime_component_table(profile), dtype=numpy.uint32).reshape(-1, 4)
    table_buffer = gpu.buffer(table, uav=False)
    table_srv = gpu.srv(table_buffer, DXGI_FORMAT_R32G32B32A32_UINT, len(table))

    merged = module.merged_components(profile)
    params_srv = {}
    for component in merged:
        params = numpy.asarray(module.runtime_params(profile, component["component_id"]), dtype=numpy.uint32).reshape(1, 4)
        params_srv[component["component_id"]] = gpu.srv(gpu.buffer(params, uav=False), DXGI_FORMAT_R32G32B32A32_UINT, 1)

    for component in merged:
        palette = numpy.ascontiguousarray(palettes[component["ib_hash"]], dtype=numpy.float32).reshape(-1, 4)
        palette_srv = gpu.srv(gpu.buffer(palette, uav=False), DXGI_FORMAT_R32G32B32A32_FLOAT, len(palette))
        gpu.dispatch(import_shader, [palette_srv, table_srv, params_srv[component["component_id"]]], pool_uav,
                     (component["vg_count"] + 63) // 64)

    pool_values = gpu.read(pool, rows * 16).view(numpy.float32).reshape(bones_count, 3, 4)
    results = {"pool": pool_values}
    for component in merged:
        output = gpu.buffer(numpy.zeros((rows, 4), numpy.float32), uav=True)
        output_uav = gpu.uav(output, DXGI_FORMAT_R32G32B32A32_FLOAT, rows)
        gpu.dispatch(palette_shader, [pool_srv, table_srv, params_srv[component["component_id"]]], output_uav,
                     (bones_count + 63) // 64)
        results[component["ib_hash"]] = gpu.read(output, rows * 16).view(numpy.float32).reshape(bones_count, 3, 4)
    return results


# ---------------------------------------------------------------------------
# Synthetic character: three components in different root spaces
# ---------------------------------------------------------------------------

def random_rigid(rng) -> numpy.ndarray:
    q, _ = numpy.linalg.qr(rng.normal(size=(3, 3)))
    if numpy.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    matrix = numpy.eye(4)
    matrix[:3, :3] = q
    matrix[:3, 3] = rng.normal(scale=0.5, size=3)
    return matrix


def synthetic_case(module):
    rng = numpy.random.default_rng(7)
    world_bones = [random_rigid(rng) for _ in range(12)]
    roots = {"aaaa0001": random_rigid(rng), "bbbb0002": random_rigid(rng), "cccc0003": random_rigid(rng)}
    bone_sets = {"aaaa0001": [0, 1, 2, 3, 4], "bbbb0002": [4, 5, 6, 7], "cccc0003": [5, 8, 9, 10, 11]}
    palettes = {}
    candidates = []
    for order, (ib_hash, bones) in enumerate(bone_sets.items()):
        palette = numpy.stack([
            (numpy.linalg.inv(roots[ib_hash]) @ world_bones[bone])[:3] for bone in bones
        ]).astype(numpy.float32)
        palettes[ib_hash] = palette
        candidates.append({
            "ib_hash": ib_hash, "unique_strs": [ib_hash + "-3-0"], "blend_vb": ib_hash[::-1],
            "pose_vs": "e8425f64cfb887cd", "pose_order": order + 1, "vertex_count": 100 - order,
            "indices_per_vertex": 4, "palette": palette, "object_matrix": roots[ib_hash],
            "shape_keyed": False, "weighted_counts": numpy.full(len(bones), 10),
        })
    profile, _ = module.plan_merged_skeleton(candidates)
    profile = module.validate_profile(profile)
    return profile, palettes, roots, world_bones, bone_sets


def main() -> int:
    if sys.platform != "win32":
        print("ZZMI_MERGED_GPU_RUNTIME=SKIP (not Windows)")
        return 0
    module = load_module()
    profile, palettes, roots, world_bones, bone_sets = synthetic_case(module)
    results = run_runtime(module, profile, palettes)

    expected_pool = numpy.concatenate([palettes[c["ib_hash"]] for c in module.merged_components(profile)])
    assert numpy.array_equal(results["pool"], expected_pool), "import shader did not copy the palettes exactly"

    worst_reference = worst_truth = 0.0
    for component in module.merged_components(profile):
        ib_hash = component["ib_hash"]
        gpu_palette = results[ib_hash]
        reference = module.reference_palette(profile, results["pool"], component["component_id"])
        worst_reference = max(worst_reference, float(numpy.abs(gpu_palette - reference).max()))
        own = slice(component["vg_offset"], component["vg_offset"] + component["vg_count"])
        assert numpy.array_equal(gpu_palette[own], results["pool"][own]), "own bones must be copied unchanged"
        # Every pool slot must equal the true bone, seen from this component's root.
        for other in module.merged_components(profile):
            for local_id, bone in enumerate(bone_sets[other["ib_hash"]]):
                truth = (numpy.linalg.inv(roots[ib_hash]) @ world_bones[bone])[:3]
                worst_truth = max(worst_truth, float(numpy.abs(gpu_palette[other["vg_offset"] + local_id] - truth).max()))
    assert worst_reference < 1e-5, worst_reference
    assert worst_truth < 1e-5, worst_truth
    print("ZZMI_MERGED_GPU_RUNTIME=PASS (reference %.1e, truth %.1e)" % (worst_reference, worst_truth))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
