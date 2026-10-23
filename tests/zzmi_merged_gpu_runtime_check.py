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
DXGI_FORMAT_R32_TYPELESS = 39
D3D11_BIND_VERTEX_BUFFER = 0x1
D3D11_BIND_CONSTANT_BUFFER = 0x4
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_UNORDERED_ACCESS = 0x80
D3D11_USAGE_DEFAULT = 0
D3D11_USAGE_STAGING = 3
D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS = 0x20
D3D11_SRV_DIMENSION_BUFFEREX = 11
D3D11_RAW_VIEW_FLAG = 1
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

    def raw_buffer(self, data: numpy.ndarray, bind_flags: int, stride: int = 0):
        """A buffer as 3dmigoto creates a ByteAddressBuffer resource (raw views allowed)."""
        data = numpy.ascontiguousarray(data)
        desc = BufferDesc(data.nbytes, D3D11_USAGE_DEFAULT, bind_flags, 0,
                          D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, stride)
        initial = SubresourceData(data.ctypes.data, 0, 0)
        buffer = ctypes.c_void_p()
        _check(_method(self.device, 3, ctypes.c_long, ctypes.POINTER(BufferDesc), ctypes.POINTER(SubresourceData),
                       ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(desc), ctypes.byref(initial), ctypes.byref(buffer)),
               "CreateBuffer(raw)")
        return buffer.value

    def raw_srv(self, buffer, nbytes: int):
        desc = SrvDesc(DXGI_FORMAT_R32_TYPELESS, D3D11_SRV_DIMENSION_BUFFEREX, 0, nbytes // 4)
        desc._pad[0] = D3D11_RAW_VIEW_FLAG
        view = ctypes.c_void_p()
        _check(_method(self.device, 7, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(SrvDesc),
                       ctypes.POINTER(ctypes.c_void_p))(buffer, ctypes.byref(desc), ctypes.byref(view)),
               "CreateShaderResourceView(raw)")
        return view.value

    def raw_uav(self, buffer, nbytes: int):
        desc = UavDesc(DXGI_FORMAT_R32_TYPELESS, 1, 0, nbytes // 4, D3D11_RAW_VIEW_FLAG)
        view = ctypes.c_void_p()
        _check(_method(self.device, 8, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(UavDesc),
                       ctypes.POINTER(ctypes.c_void_p))(buffer, ctypes.byref(desc), ctypes.byref(view)),
               "CreateUnorderedAccessView(raw)")
        return view.value

    def create_buffer_result(self, nbytes: int, bind_flags: int, misc_flags: int, stride: int) -> int:
        """HRESULT of CreateBuffer for this description (the runtime validates it even without the debug layer)."""
        desc = BufferDesc(nbytes, D3D11_USAGE_DEFAULT, bind_flags, 0, misc_flags, stride)
        buffer = ctypes.c_void_p()
        result = _method(self.device, 3, ctypes.c_long, ctypes.POINTER(BufferDesc), ctypes.c_void_p,
                         ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(desc), None, ctypes.byref(buffer))
        if result == 0 and buffer.value:
            _method(buffer.value, 2, wintypes.ULONG)()
        return result

    def constant_buffer(self, data: numpy.ndarray):
        data = numpy.ascontiguousarray(data)
        desc = BufferDesc(data.nbytes, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0)
        initial = SubresourceData(data.ctypes.data, 0, 0)
        buffer = ctypes.c_void_p()
        _check(_method(self.device, 3, ctypes.c_long, ctypes.POINTER(BufferDesc), ctypes.POINTER(SubresourceData),
                       ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(desc), ctypes.byref(initial), ctypes.byref(buffer)),
               "CreateBuffer(constant)")
        return buffer.value

    def dispatch_bound(self, shader, cbs: list, srvs: list, uavs: list, groups: int, ini_params=None):
        """Dispatch with constant buffers and several UAVs bound (and IniParams at t120)."""
        set_cbs = _method(self.context, 71, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p)
        set_srvs = _method(self.context, 67, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p)
        set_uavs = _method(self.context, 68, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p)
        _method(self.context, 69, None, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT)(shader, None, 0)
        set_cbs(0, len(cbs), (ctypes.c_void_p * len(cbs))(*cbs))
        set_srvs(0, len(srvs), (ctypes.c_void_p * len(srvs))(*srvs))
        if ini_params is not None:
            set_srvs(120, 1, (ctypes.c_void_p * 1)(ini_params))
        set_uavs(0, len(uavs), (ctypes.c_void_p * len(uavs))(*uavs), None)
        _method(self.context, 41, None, wintypes.UINT, wintypes.UINT, wintypes.UINT)(groups, 1, 1)
        set_uavs(0, len(uavs), (ctypes.c_void_p * len(uavs))(), None)
        set_srvs(0, len(srvs), (ctypes.c_void_p * len(srvs))())
        if ini_params is not None:
            set_srvs(120, 1, (ctypes.c_void_p * 1)())
        set_cbs(0, len(cbs), (ctypes.c_void_p * len(cbs))())

    def ini_params(self, x0, y0=0.0, z0=0.0, w0=0.0):
        """IniParams as 3dmigoto binds them: a Texture1D of float4 at t120 (only x0..w0 used here)."""
        data = numpy.zeros((8, 4), numpy.float32)
        data[0] = (x0, y0, z0, w0)
        desc = (wintypes.UINT * 8)(len(data), 1, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_USAGE_DEFAULT,
                                   D3D11_BIND_SHADER_RESOURCE, 0, 0)
        initial = SubresourceData(data.ctypes.data, 0, 0)
        texture = ctypes.c_void_p()
        _check(_method(self.device, 4, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(SubresourceData),
                       ctypes.POINTER(ctypes.c_void_p))(desc, ctypes.byref(initial), ctypes.byref(texture)),
               "CreateTexture1D")
        view_desc = SrvDesc(DXGI_FORMAT_R32G32B32A32_FLOAT, 2, 0, 1)  # TEXTURE1D, mip 0, one mip
        view = ctypes.c_void_p()
        _check(_method(self.device, 7, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(SrvDesc),
                       ctypes.POINTER(ctypes.c_void_p))(texture.value, ctypes.byref(view_desc), ctypes.byref(view)),
               "CreateShaderResourceView(Texture1D)")
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


def check_frame_delay(module, profile: dict, palettes: dict) -> float:
    """Two frames of the uniform one-frame delay, as the delayed export drives them.

    Frame 1 imports every palette (merged parts and a face in the extra range),
    [Present] copies the pool; frame 2 imports other poses.  Everything built
    in frame 2 must come from frame 1: the merged palettes, the face's own
    palette and the own palette of a part that uses only its own bones.
    """
    rng = numpy.random.default_rng(11)
    profile = dict(profile)
    face_id = len(profile["components"])
    profile["components"] = list(profile["components"]) + [{
        "component_id": face_id, "ib_hash": "ffff0009", "role": module.ROLE_SHAPE_KEYED,
        "palette_bone_count": 6, "vg_offset": 0, "vg_count": 0,
    }]
    layout = module.delayed_pool_layout(profile, [face_id])
    assert layout == [{"component_id": face_id, "offset": profile["bones_count"], "count": 6,
                       "table_index": len(module.merged_components(profile))}], layout
    total = profile["bones_count"] + 6
    merged = module.merged_components(profile)
    frames = []
    for frame in range(2):
        poses = {
            component["ib_hash"]: (palettes[component["ib_hash"]] if frame == 0 else
                                   rng.normal(size=palettes[component["ib_hash"]].shape)).astype(numpy.float32)
            for component in merged
        }
        poses["ffff0009"] = rng.normal(size=(6, 3, 4)).astype(numpy.float32)
        frames.append(poses)

    gpu = Compute()
    import_shader = gpu.shader(module.IMPORT_SHADER_SOURCE)
    palette_shader = gpu.shader(module.PALETTE_SHADER_SOURCE)
    delay_shader = gpu.shader(module.DELAY_SHADER_SOURCE)

    def rw(rows):
        buffer = gpu.buffer(numpy.zeros((rows, 4), numpy.float32), uav=True)
        return (buffer, gpu.srv(buffer, DXGI_FORMAT_R32G32B32A32_FLOAT, rows),
                gpu.uav(buffer, DXGI_FORMAT_R32G32B32A32_FLOAT, rows))

    def uint4(values):
        array = numpy.asarray(values, dtype=numpy.uint32).reshape(-1, 4)
        return gpu.srv(gpu.buffer(array, uav=False), DXGI_FORMAT_R32G32B32A32_UINT, len(array))

    pool, pool_srv, pool_uav = rw(total * 3)
    prev, prev_srv, prev_uav = rw(total * 3)
    table_srv = uint4(module.runtime_component_table(profile, layout))
    params = {component["component_id"]: uint4(module.runtime_params(profile, component["component_id"], layout))
              for component in merged + [profile["components"][face_id]]}
    counts = {component["component_id"]: component["vg_count"] for component in merged}
    counts[face_id] = 6

    def import_frame(poses):
        for component in merged + [profile["components"][face_id]]:
            palette = numpy.ascontiguousarray(poses[component["ib_hash"]]).reshape(-1, 4)
            source = gpu.srv(gpu.buffer(palette, uav=False), DXGI_FORMAT_R32G32B32A32_FLOAT, len(palette))
            gpu.dispatch(import_shader, [source, table_srv, params[component["component_id"]]], pool_uav,
                         (counts[component["component_id"]] + 63) // 64)

    import_frame(frames[0])
    gpu.dispatch(delay_shader, [pool_srv, uint4([0, total, 0, 0])], prev_uav, (total + 63) // 64)  # [Present]
    import_frame(frames[1])

    previous = numpy.concatenate([frames[0][component["ib_hash"]] for component in merged])
    current_pool = gpu.read(pool, total * 48).view(numpy.float32).reshape(total, 3, 4)
    assert numpy.array_equal(current_pool[profile["bones_count"]:], frames[1]["ffff0009"]), "face import"
    worst = 0.0
    for component in merged:
        palette, _, palette_uav = rw(profile["bones_count"] * 3)
        gpu.dispatch(palette_shader, [prev_srv, table_srv, params[component["component_id"]]], palette_uav,
                     (profile["bones_count"] + 63) // 64)
        built = gpu.read(palette, profile["bones_count"] * 48).view(numpy.float32).reshape(-1, 3, 4)
        reference = module.reference_palette(profile, previous, component["component_id"])
        worst = max(worst, float(numpy.abs(built - reference).max()))
        # A part that uses only its own bones gets its own bones of the previous frame.
        own, _, own_uav = rw(component["vg_count"] * 3)
        gpu.dispatch(delay_shader, [prev_srv, uint4([component["vg_offset"], component["vg_count"], 0, 0])],
                     own_uav, (component["vg_count"] + 63) // 64)
        own_bones = gpu.read(own, component["vg_count"] * 48).view(numpy.float32).reshape(-1, 3, 4)
        assert numpy.array_equal(own_bones, frames[0][component["ib_hash"]]), component["ib_hash"]
    face, _, face_uav = rw(6 * 3)
    gpu.dispatch(delay_shader, [prev_srv, uint4([layout[0]["offset"], 6, 0, 0])], face_uav, 1)
    face_bones = gpu.read(face, 6 * 48).view(numpy.float32).reshape(-1, 3, 4)
    assert numpy.array_equal(face_bones, frames[0]["ffff0009"]), "face must get its previous palette"
    assert worst < 1e-5, worst
    return worst


def object_constants(rng, now: numpy.ndarray, previous: numpy.ndarray) -> numpy.ndarray:
    """29 registers as the game binds them at vs-cb1 (row-vector matrices)."""
    rows = rng.normal(size=(29, 4))
    rows[0:4] = now.T
    rows[4:8] = numpy.linalg.inv(now.T)
    rows[20:24] = previous.T
    return rows


def check_placement(module) -> float:
    """The placement shader against reference_placement, and the property it is for.

    Rows 4-7 and 20-23 of the game's object constants are filled with garbage:
    only rows 0-3 are written for every pass, the shader must not read the rest.
    """
    rng = numpy.random.default_rng(5)
    gpu = Compute()
    shader = gpu.shader(module.PLACE_SHADER_SOURCE)
    self_now, self_previous, placed_previous = random_rigid(rng), random_rigid(rng), random_rigid(rng)
    self_rows = rng.normal(size=(29, 4))
    self_rows[0:4] = self_now.T
    history = numpy.zeros((9, 4))
    history[0:4] = self_previous.T
    history[4:8] = placed_previous.T
    sentinel = numpy.full((9, 4), 7.0)
    worst = 0.0
    DXGI_FORMAT_R32_FLOAT = 41

    def params(mode):
        return gpu.srv(gpu.buffer(numpy.asarray([[mode, 0, 0, 0]], numpy.uint32), uav=False),
                       DXGI_FORMAT_R32G32B32A32_UINT, 1)

    def run(self_data, history_data, place, shift_buffer):
        srvs = [gpu.srv(gpu.buffer(numpy.asarray(history_data, numpy.float32), uav=False),
                        DXGI_FORMAT_R32G32B32A32_FLOAT, 9),
                params(1 if place else 0)]
        output = gpu.buffer(numpy.zeros((29, 4), numpy.float32), uav=True)
        record = gpu.buffer(sentinel.astype(numpy.float32), uav=True)
        gpu.dispatch_bound(shader, [gpu.constant_buffer(numpy.asarray(self_data, numpy.float32))], srvs,
                           [gpu.uav(output, DXGI_FORMAT_R32G32B32A32_FLOAT, 29),
                            gpu.uav(record, DXGI_FORMAT_R32G32B32A32_FLOAT, 9),
                            gpu.uav(shift_buffer, DXGI_FORMAT_R32_FLOAT, 4)], 1)
        return (gpu.read(output, 29 * 16).view(numpy.float32).reshape(29, 4),
                gpu.read(record, 9 * 16).view(numpy.float32).reshape(9, 4))

    def read_shift(shift_buffer):
        values = gpu.read(shift_buffer, 16).view(numpy.float32)
        return values[:3].astype(numpy.float64) if values[3] == 1.0 else None

    def fresh_shift(values=(0, 0, 0, 0)):
        return gpu.buffer(numpy.asarray(values, numpy.float32), uav=True)

    not_drawn_last_frame = history.copy()
    not_drawn_last_frame[0:4] = 0
    first_placed_frame = history.copy()
    first_placed_frame[4:8] = 0
    not_object = self_rows.copy()
    not_object[0:4] = rng.normal(size=(4, 4))
    earlier_shift = rng.normal(size=3)
    cases = [
        ("placed", self_rows, history, True, None),
        ("placed after another part", self_rows, history, True, earlier_shift),
        ("first placed frame", self_rows, first_placed_frame, True, None),
        ("part not drawn last frame", self_rows, not_drawn_last_frame, True, None),
        ("record only", self_rows, history, False, None),
        ("not object constants", not_object, history, True, None),
    ]
    for name, self_data, history_data, place, shift in cases:
        shift_buffer = fresh_shift((0, 0, 0, 0) if shift is None else tuple(shift) + (1,))
        output, record = run(self_data, history_data, place, shift_buffer)
        expected_output, expected_record, expected_shift = module.reference_placement(
            numpy.asarray(self_data, numpy.float32), numpy.asarray(history_data, numpy.float32), place,
            None if shift is None else numpy.asarray(shift, numpy.float32))
        error = float(numpy.abs(output - expected_output).max())
        assert error < 1e-4, (name, error)
        worst = max(worst, error)
        if expected_record is None:
            assert numpy.array_equal(record, sentinel.astype(numpy.float32)), name + ": record must stay"
        else:
            error = float(numpy.abs(record[:8] - expected_record).max())
            assert error < 1e-4, (name, error)
            worst = max(worst, error)
        gpu_shift = read_shift(shift_buffer)
        assert (gpu_shift is None) == (expected_shift is None), name + ": shift set"
        if gpu_shift is not None:
            assert numpy.abs(gpu_shift - expected_shift).max() < 1e-5, name + ": shift"
        if name in ("part not drawn last frame", "record only", "not object constants"):
            assert numpy.array_equal(output, numpy.asarray(self_data, numpy.float32)), name + ": must pass through"
            assert gpu_shift is None, name + ": must not set the shift"

    # The first part drawn in a frame is drawn in its pose of the previous frame at its current position,
    # whatever rows 4-7 and 20-23 of the game's constants hold, and records the current matrix and the one
    # drawn with for the next frame.
    shift_buffer = fresh_shift()
    output, record = run(self_rows, history, True, shift_buffer)
    assert numpy.abs(output[0:3] - self_previous.T[0:3]).max() < 1e-6, "rotation of the previous frame"
    assert numpy.abs(output[3] - self_now.T[3]).max() < 1e-5, "current position"
    assert numpy.abs(output[0:4] @ output[4:8] - numpy.eye(4)).max() < 1e-4, "rows 4-7 must be the inverse"
    assert numpy.abs(output[20:24] - placed_previous.T).max() < 1e-6, "previous placement for motion vectors"
    assert numpy.abs(record[0:4] - self_now.T).max() < 1e-6, "the game's current matrix is recorded"
    assert numpy.abs(record[4:8] - output[0:4]).max() < 1e-6, "the matrix drawn with is recorded"

    # A second part of the same frame (another root) is moved by the first part's distance, not its own,
    # so both keep the offset they had in the previous frame.
    other_now, other_previous = random_rigid(rng), random_rigid(rng)
    other_rows = rng.normal(size=(29, 4))
    other_rows[0:4] = other_now.T
    other_history = numpy.zeros((9, 4))
    other_history[0:4] = other_previous.T
    other_history[4:8] = random_rigid(rng).T
    other_output, _ = run(other_rows, other_history, True, shift_buffer)
    assert numpy.abs(other_output[3, :3] - output[3, :3]
                     - (other_previous.T[3, :3] - self_previous.T[3, :3])).max() < 1e-5, "parts must stay together"

    # [Present] clears the shift for the next frame.
    gpu.dispatch_bound(shader, [], [None, params(2)], [None, None, gpu.uav(shift_buffer, DXGI_FORMAT_R32_FLOAT, 4)], 1)
    assert numpy.array_equal(gpu.read(shift_buffer, 16).view(numpy.float32), numpy.zeros(4, numpy.float32)), "clear"
    return worst


def _blend_rows(rng, count: int, bone_count: int, weight_format: str, indices_first: bool):
    """A weight buffer in one of the formats the skinning shader decodes, and its element list."""
    indices = rng.integers(0, bone_count, size=(count, 4)).astype(numpy.uint32)
    weights = rng.random(size=(count, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    if weight_format == "R32G32B32A32_FLOAT":
        weight_bytes = weights.astype(numpy.float32).view(numpy.uint8).reshape(count, -1)
    elif weight_format == "R16G16B16A16_FLOAT":
        weight_bytes = weights.astype(numpy.float16).view(numpy.uint8).reshape(count, -1)
    elif weight_format == "R16G16B16A16_UNORM":
        weight_bytes = numpy.round(weights * 65535).astype(numpy.uint16).view(numpy.uint8).reshape(count, -1)
    else:
        weight_bytes = numpy.round(weights * 255).astype(numpy.uint8)
    index_bytes = indices.view(numpy.uint8).reshape(count, -1)
    weight_element = ("BLENDWEIGHTS", 0, weight_format, weight_bytes.shape[1])
    index_element = ("BLENDINDICES", 0, "R32G32B32A32_UINT", 16)
    if indices_first:
        return numpy.hstack([index_bytes, weight_bytes]), [index_element, weight_element]
    return numpy.hstack([weight_bytes, index_bytes]), [weight_element, index_element]


def check_same_frame_skin(module) -> float:
    """The same-frame skinning shader against reference_same_frame_skin, on buffers created as 3dmigoto does.

    The output buffer is created with the description 3dmigoto gives it, which D3D11 must accept for a
    buffer that is both written and drawn from.  When the matcher did not find every component of the
    copy, every vertex is written at the origin (nothing is drawn).  The element after the vertices says
    whether they are a valid result and of which game buffer; the copy's previous output (vb3) gets this
    frame's result when it is not that buffer's valid result, and is left alone when it is.  Without a
    previous output (w0 = 0) whatever buffer is left in the slot is never touched.
    """
    rng = numpy.random.default_rng(23)
    gpu = Compute()
    shader = gpu.shader(module.SKIN_SHADER_SOURCE)
    count, bone_count = 150, 9
    vertices = rng.normal(size=(count, 10)).astype(numpy.float32)
    vertices[:, 9] = rng.choice([-1.0, 1.0], size=count)
    position_elements = [("POSITION", 0, "R32G32B32_FLOAT", 12), ("NORMAL", 0, "R32G32B32_FLOAT", 12),
                         ("TANGENT", 0, "R32G32B32A32_FLOAT", 16)]
    position_bytes = vertices.tobytes()
    position_srv = gpu.raw_srv(gpu.raw_buffer(vertices, D3D11_BIND_SHADER_RESOURCE), len(position_bytes))
    output_flags = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS
    nbytes = count * module.SKIN_VERTEX_STRIDE

    cases = [(fmt, False) for fmt in module.SKIN_WEIGHT_FORMATS] + [("R8G8B8A8_UNORM", True), ("one bone", False)]
    worst = 0.0
    for weight_format, indices_first in cases:
        if weight_format == "one bone":
            blend = rng.integers(0, bone_count, size=(count, 1)).astype(numpy.uint32).view(numpy.uint8).reshape(count, -1)
            elements = [("BLENDINDICES", 0, "R32_UINT", 4)]
        else:
            blend, elements = _blend_rows(rng, count, bone_count, weight_format, indices_first)
        layout, reason = module.same_frame_skin_layout(position_elements, elements)
        assert layout is not None, (weight_format, reason)
        blend_bytes = numpy.ascontiguousarray(blend).tobytes()
        blend_srv = gpu.raw_srv(gpu.raw_buffer(numpy.frombuffer(blend_bytes, numpy.uint8), D3D11_BIND_SHADER_RESOURCE),
                                len(blend_bytes))
        # As 3dmigoto creates `type = RWByteAddressBuffer`, `stride = 40`, `array = <vertices> + 1`.
        total = nbytes + module.SKIN_VERTEX_STRIDE
        # The game buffers' identities (as XXMI encodes them: normal floats).
        key0, key3, stranger = 3.75e-12, 2.5e7, 6.125
        # (found, what the previous output holds, previous output bound): its buffer's valid result, an
        # invalid one, another buffer's valid result (the output passed between buffers); unbound: a
        # buffer left in u1 by whoever used the slot before, which says it is not a valid result.
        for found, held, bound in ((True, "invalid", True), (True, "valid", True), (True, "another's", True),
                                   (False, "invalid", True), (True, "invalid", False)):
            skinned = gpu.raw_buffer(numpy.full(total, 7, numpy.uint8), output_flags, module.SKIN_VERTEX_STRIDE)
            initial_previous = numpy.full(total // 4, 1234.5, numpy.float32)
            initial_previous[nbytes // 4 + 1] = stranger if held == "another's" else key3
            initial_previous.view(numpy.uint32)[nbytes // 4] = 0 if held == "invalid" else 1
            previous = gpu.raw_buffer(initial_previous, output_flags, module.SKIN_VERTEX_STRIDE)
            uavs = [gpu.raw_uav(skinned, total), gpu.raw_uav(previous, total)]
            palette = rng.normal(size=(bone_count * 3, 4)).astype(numpy.float32)
            params = module.same_frame_skin_params(count, layout)
            choice = numpy.zeros((2, 4), numpy.uint32)
            choice[0, 0] = 1 if found else 0
            srvs = [gpu.srv(gpu.buffer(palette, uav=False), DXGI_FORMAT_R32G32B32A32_FLOAT, len(palette)),
                    position_srv, blend_srv,
                    gpu.srv(gpu.buffer(numpy.asarray([params], numpy.uint32), uav=False), DXGI_FORMAT_R32G32B32A32_UINT, 1),
                    gpu.srv(gpu.buffer(choice, uav=False), DXGI_FORMAT_R32G32B32A32_UINT, len(choice))]
            gpu.dispatch_bound(shader, [], srvs, uavs, (count + 63) // 64,
                               gpu.ini_params(0, 0, key0, key3 if bound else 0))
            written = gpu.read(skinned, total)
            got = written[:nbytes].view(numpy.float32).reshape(count, 10)
            assert written[nbytes:nbytes + 8].view(numpy.uint32).tolist() == [
                1 if found else 0, int(numpy.float32(key0).view(numpy.uint32))], "validity of the result, whose"
            expected = module.reference_same_frame_skin(palette, position_bytes, blend_bytes, params, found=found)
            scale = max(1.0, float(numpy.abs(expected).max()))
            error = float(numpy.abs(got - expected).max()) / scale
            assert error < 1e-5, (weight_format, found, error)
            got_previous = gpu.read(previous, total)
            if found:
                assert numpy.array_equal(got[:, 9], vertices[:, 9]), "tangent w kept"
            else:
                assert not got.any(), weight_format + ": a copy without all its bones is not drawn"
            if found and held != "valid" and bound:
                assert numpy.array_equal(got_previous[:nbytes], written[:nbytes]), held + " history replaced"
            else:
                assert numpy.array_equal(got_previous, initial_previous.view(numpy.uint8)), \
                    "valid history kept" if bound else "a buffer left in the slot must not be written"
            worst = max(worst, error)

    # The description above was accepted (raw views, vertex buffer + unordered access, stride 40); a
    # structured buffer cannot be a vertex buffer, and the runtime refuses it, so this check means something.
    D3D11_RESOURCE_MISC_BUFFER_STRUCTURED = 0x40
    assert gpu.create_buffer_result(nbytes, output_flags, D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS,
                                    module.SKIN_VERTEX_STRIDE) == 0
    assert gpu.create_buffer_result(nbytes, output_flags, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
                                    module.SKIN_VERTEX_STRIDE) != 0, "structured vertex buffers must be refused"

    # Layouts the shader cannot skin are refused (the part is then skinned the standard way).
    for elements in ([("BLENDWEIGHTS", 0, "R10G10B10A2_UNORM", 4), ("BLENDINDICES", 0, "R32G32B32A32_UINT", 16)],
                     [("BLENDWEIGHTS", 0, "R32G32B32A32_FLOAT", 16), ("BLENDINDICES", 0, "R8G8B8A8_UINT", 4)]):
        assert module.same_frame_skin_layout(position_elements, elements)[0] is None, elements
    assert module.same_frame_skin_layout(position_elements[:2], [("BLENDINDICES", 0, "R32_UINT", 4)])[0] is None
    return worst


def _rotation_z(angle: float, translation) -> numpy.ndarray:
    matrix = numpy.eye(4)
    matrix[:2, :2] = [[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]]
    matrix[:3, 3] = translation
    return matrix


def check_instances(module) -> float:
    """Copies of the character, skinned interleaved into store slots, matched and gathered per copy.

    The second copy stands 1.35 m away, turned, in a slightly different pose (as two copies of an agent
    were seen in the game).  One component's record is of the previous frame (it is drawn after the part
    being skinned).  Every copy must get the palette built from its own bones only.  A copy keeps the
    instances it was matched with (owner entries, by the identities of its drawn part's buffers), so a
    copy that has just appeared next to it, even 10 cm away, never takes them; an instance a record of
    this frame puts elsewhere is never taken.  A slot that passes to another buffer is cleared.
    """
    profile, palettes, roots, world_bones, bone_sets = synthetic_case(module)
    merged = module.merged_components(profile)
    ids = [component["component_id"] for component in merged]
    by_ib = {component["ib_hash"]: component for component in merged}
    host = merged[0]
    shared = {4, 5}

    def scene(moves, seed):
        """Every copy's world bones, roots and palettes: the first as in the synthetic case, the others
        moved by ``moves``, their limbs in another pose."""
        rng = numpy.random.default_rng(seed)
        worlds = [list(world_bones)]
        copy_roots = [dict(roots)]
        for move in moves:
            worlds.append([])
            for index, bone in enumerate(world_bones):
                bend = numpy.eye(4) if index in shared else _rotation_z(rng.normal(scale=0.3), rng.normal(scale=0.02, size=3))
                worlds[-1].append(move @ bone @ bend)
            copy_roots.append({ib: move @ root for ib, root in roots.items()})
        copy_palettes = [{
            ib: numpy.stack([(numpy.linalg.inv(copy_roots[copy][ib]) @ worlds[copy][bone])[:3] for bone in bones])
            for ib, bones in bone_sets.items()} for copy in range(len(worlds))]
        return worlds, copy_roots, copy_palettes

    far = scene([_rotation_z(2.1, (1.3, -0.4, 0.0))], 31)
    near = scene([_rotation_z(0.05, (0.1, 0.0, 0.0))], 37)
    three = scene([_rotation_z(0.0, (1.0, 0.0, 0.0)), _rotation_z(0.0, (3.0, 0.0, 0.0))], 41)
    # The shared bones of the near copies are 6 to 25 cm apart: closer than any distance a record of the
    # previous frame could be trusted with, farther than a record of this frame can be off.
    gaps = [float(numpy.linalg.norm(near[0][0][bone][:3, 3] - near[0][1][bone][:3, 3])) for bone in shared]
    assert all(module.SAME_FRAME_DISTANCE < gap < 0.25 for gap in gaps), gaps

    slots = module.instance_slot_count(len(ids))
    rows = module.instance_slot_rows(profile, ids)
    owner_rows = module.instance_owner_rows(slots, len(ids))
    host_index = module.runtime_component_index(profile, host["component_id"])
    frame = 1234.0
    # Interleaved slots, as the game skins the copies' components in any order.
    slot_of = {("aaaa0001", 0): 7, ("aaaa0001", 1): 2, ("bbbb0002", 0): 30, ("bbbb0002", 1): 11,
               ("cccc0003", 0): 0, ("cccc0003", 1): 41,
               ("aaaa0001", 2): 60, ("bbbb0002", 2): 61, ("cccc0003", 2): 62}

    def keys_of(copy):
        """The identities of the copy's drawn part's two buffers (normal floats, as XXMI encodes them)."""
        return 1.5e-3 * (copy + 1), 7.25e5 * (copy + 1)

    gpu = Compute()
    import_shader = gpu.shader(module.INSTANCE_IMPORT_SHADER_SOURCE)
    record_shader = gpu.shader(module.INSTANCE_RECORD_SHADER_SOURCE)
    match_shader = gpu.shader(module.INSTANCE_MATCH_SHADER_SOURCE)
    gather_shader = gpu.shader(module.INSTANCE_GATHER_SHADER_SOURCE)
    clear_shader = gpu.shader(module.INSTANCE_CLEAR_SHADER_SOURCE)
    palette_shader = gpu.shader(module.PALETTE_SHADER_SOURCE)

    def uint4(values):
        array = numpy.asarray(values, dtype=numpy.uint32).reshape(-1, 4)
        return gpu.srv(gpu.buffer(array, uav=False), DXGI_FORMAT_R32G32B32A32_UINT, len(array))

    def rw(rows_count, dtype=numpy.float32, fmt=DXGI_FORMAT_R32G32B32A32_FLOAT, initial=None):
        data = numpy.zeros((rows_count, 4), dtype) if initial is None else numpy.ascontiguousarray(initial, dtype)
        buffer = gpu.buffer(data, uav=True)
        return buffer, gpu.srv(buffer, fmt, rows_count), gpu.uav(buffer, fmt, rows_count)

    table_srv = uint4(module.runtime_component_table(profile))
    needed_values = module.instance_needed(profile, ids)
    needed_srv = uint4(needed_values)
    params = {component["ib_hash"]: module.instance_params(profile, component["component_id"], slots, rows)
              for component in merged}
    params_srv = {ib: uint4(values) for ib, values in params.items()}

    class State:
        pass

    def build(world, imported, recorded, stale=None, owners=None):
        """The store, records and owner entries of a frame (records of ``stale`` are of the previous one)."""
        _, copy_roots, copy_palettes = world
        state = State()
        state.world = world
        state.store, state.store_srv, state.store_uav = rw(slots * rows)
        state.records, state.records_srv, state.records_uav = rw(slots * module.INSTANCE_RECORD_ROWS)
        state.owners, _, state.owners_uav = rw(owner_rows, initial=owners)

        def import_bones(ib, copy, stamp):
            # One import for every part: which part it is comes in z0.
            source = numpy.ascontiguousarray(copy_palettes[copy][ib], dtype=numpy.float32).reshape(-1, 4)
            gpu.dispatch_bound(import_shader, [], [gpu.srv(gpu.buffer(source, uav=False), DXGI_FORMAT_R32G32B32A32_FLOAT,
                                                            len(source)), table_srv, params_srv[host["ib_hash"]]],
                               [state.store_uav], 1, gpu.ini_params(slot_of[(ib, copy)], stamp,
                                                                    module.runtime_component_index(profile, by_ib[ib]["component_id"])))

        for ib, copy in imported:
            import_bones(ib, copy, frame)
        for ib, copy in recorded:
            root, stamp = copy_roots[copy][ib], frame
            if stale and (ib, copy) in stale:
                # A record of the previous frame was written then, after that frame's import.
                root, stamp = stale[(ib, copy)] @ root, frame - 1
                import_bones(ib, copy, stamp)
            constants = gpu.constant_buffer(numpy.ascontiguousarray(root.T, dtype=numpy.float32))
            # One record for every part: which part it is, the record reads from the slot's import.
            gpu.dispatch_bound(record_shader, [constants], [params_srv[host["ib_hash"]], state.store_srv],
                               [state.records_uav], 1, gpu.ini_params(slot_of[(ib, copy)], slot_of[(ib, copy)], stamp))
            if stamp != frame and (ib, copy) in imported:
                import_bones(ib, copy, frame)
        return state

    def owners_of(state):
        return gpu.read(state.owners, owner_rows * 16).view(numpy.float32).reshape(-1, 4)

    def match(state, copy, keys=None):
        """The match shader for ``copy``'s drawn part, against the reference (choice and owner entries)."""
        _, copy_roots, _ = state.world
        choice, choice_srv, choice_uav = rw(1 + len(ids), numpy.uint32, DXGI_FORMAT_R32G32B32A32_UINT)
        host_rows = numpy.ascontiguousarray(copy_roots[copy][host["ib_hash"]].T, dtype=numpy.float32)
        host_slot = slot_of[(host["ib_hash"], copy)]
        keys = keys_of(copy) if keys is None else keys
        expected_owners = owners_of(state).astype(numpy.float64)
        gpu.dispatch_bound(match_shader, [gpu.constant_buffer(host_rows)],
                           [state.store_srv, state.records_srv, table_srv, params_srv[host["ib_hash"]], needed_srv],
                           [choice_uav, state.owners_uav], 1, gpu.ini_params(host_slot, frame, keys[0], keys[1]))
        got = gpu.read(choice, (1 + len(ids)) * 16).view(numpy.uint32).reshape(-1, 4)
        expected = module.reference_instance_match(
            profile, gpu.read(state.store, slots * rows * 16).view(numpy.float32),
            gpu.read(state.records, slots * module.INSTANCE_RECORD_ROWS * 16).view(numpy.float32),
            params[host["ib_hash"]], needed_values, host_rows, host_slot, frame,
            [float(numpy.float32(key)) for key in keys], expected_owners)
        assert numpy.array_equal(got, expected), (got, expected)
        assert numpy.array_equal(owners_of(state), expected_owners.astype(numpy.float32)), "owner entries"
        return got, choice_srv

    def matched(choice, copy):
        return choice[0, 0] == 1 and all(choice[1 + k, 0] == slot_of[(c["ib_hash"], copy)] for k, c in enumerate(merged))

    def kept(copy, stamp, keys=None):
        """Owner entries as if ``copy`` had been matched in frame ``stamp`` (its drawn part's buffers)."""
        owners = numpy.zeros((owner_rows, 4), numpy.float32)
        x, y = keys_of(copy) if keys is None else keys
        for component in merged[1:]:
            owners[slot_of[(component["ib_hash"], copy)] * len(ids) + host_index] = (x, y, stamp, 1)
        return owners

    everything = [(ib, copy) for copy in (1, 0) for ib in bone_sets]
    state = build(far, everything, everything, stale={("cccc0003", 0): _rotation_z(0.02, (0.03, -0.04, 0.0))})
    worst = 0.0
    for copy in (0, 1):
        _, copy_roots, copy_palettes = far
        choice, choice_srv = match(state, copy)
        assert matched(choice, copy), (copy, choice)
        pool, pool_srv, pool_uav = rw(profile["bones_count"] * 3)
        gpu.dispatch_bound(gather_shader, [], [state.store_srv, table_srv, params_srv[host["ib_hash"]], choice_srv],
                           [pool_uav], (profile["bones_count"] + 63) // 64)
        gathered = gpu.read(pool, profile["bones_count"] * 48).view(numpy.float32).reshape(-1, 3, 4)
        expected_pool = numpy.concatenate([copy_palettes[copy][component["ib_hash"]] for component in merged])
        assert numpy.abs(gathered - expected_pool).max() < 1e-6, copy
        palette, _, palette_uav = rw(profile["bones_count"] * 3)
        gpu.dispatch(palette_shader, [pool_srv, table_srv, uint4(module.runtime_params(profile, host["component_id"]))],
                     palette_uav, (profile["bones_count"] + 63) // 64)
        built = gpu.read(palette, profile["bones_count"] * 48).view(numpy.float32).reshape(-1, 3, 4)
        worst = max(worst, float(numpy.abs(built - module.reference_palette(profile, expected_pool, host["component_id"])).max()))
        # Every bone, converted to the copy's host space, is the copy's own bone.
        for other in merged:
            for local_id, bone in enumerate(bone_sets[other["ib_hash"]]):
                truth = (numpy.linalg.inv(copy_roots[copy][host["ib_hash"]]) @ far[0][copy][bone])[:3]
                worst = max(worst, float(numpy.abs(built[other["vg_offset"] + local_id] - truth).max()))
    # The copies keep what they were matched with.
    assert numpy.array_equal(owners_of(state), kept(0, frame) + kept(1, frame)), owners_of(state)

    # A slot that passes to another buffer is cleared: its header, its records, its owner entries; no other.
    before = [gpu.read(buffer, size * 16).view(numpy.float32).reshape(-1, 4).copy()
              for buffer, size in ((state.store, slots * rows), (state.records, slots * module.INSTANCE_RECORD_ROWS),
                                   (state.owners, owner_rows))]
    cleared = slot_of[("bbbb0002", 0)]
    for slot in (cleared, slots + 3):  # a slot out of range clears nothing
        gpu.dispatch_bound(clear_shader, [], [params_srv[host["ib_hash"]]],
                           [state.store_uav, state.records_uav, state.owners_uav], 1, gpu.ini_params(slot))
    after = [gpu.read(buffer, size * 16).view(numpy.float32).reshape(-1, 4)
             for buffer, size in ((state.store, slots * rows), (state.records, slots * module.INSTANCE_RECORD_ROWS),
                                  (state.owners, owner_rows))]
    for name, old, new, first, width in (("header", before[0], after[0], cleared * rows, 1),
                                         ("records", before[1], after[1], cleared * 5, 5),
                                         ("owner entries", before[2], after[2], cleared * len(ids), len(ids))):
        assert old[first:first + width].any(), name + " of the slot were there"
        assert not new[first:first + width].any(), name + " of the slot cleared"
        rest = numpy.ones(len(old), bool)
        rest[first:first + width] = False
        assert numpy.array_equal(old[rest], new[rest]), name + " of other slots kept"

    # Two candidates and no records: the copies cannot be told apart, the copy is not drawn yet.
    state = build(far, everything, [(ib, copy) for ib, copy in everything if ib != "bbbb0002"])
    assert match(state, 0)[0][0, 0] == 0
    # A copy that has just appeared (no records yet) next to one that has records of this frame: they put
    # the other copy's instances elsewhere, so the new copy takes the only ones left, its own.
    state = build(far, everything, [(ib, 0) for ib in bone_sets])
    assert matched(match(state, 1)[0], 1), "the new copy takes its own instances"
    assert matched(match(state, 0)[0], 0)
    # One copy's component not skinned this frame: the other copy's instance is the only candidate and a
    # metre away, so that copy is not matched rather than given another copy's bones.
    culled = [(ib, copy) for ib, copy in everything if (ib, copy) != ("cccc0003", 0)]
    state = build(far, culled, culled)
    assert match(state, 0)[0][0, 0] == 0, "a culled component must not be borrowed"
    assert matched(match(state, 1)[0], 1)
    # The same with three copies, the others' instances 1 m and 3 m away (records of this frame): both are
    # elsewhere, however clearly one is nearer than the other.
    all_three = [(ib, copy) for copy in (2, 1, 0) for ib in bone_sets]
    culled = [(ib, copy) for ib, copy in all_three if (ib, copy) != ("cccc0003", 0)]
    state = build(three, culled, culled)
    assert match(state, 0)[0][0, 0] == 0, "an instance a record of this frame puts elsewhere must not be taken"
    for copy in (1, 2):
        assert matched(match(state, copy)[0], copy), copy
    # At speed with a low frame rate the records (of the previous frame) are 35 cm off: every
    # candidate has a record, and the copy's own instance is still clearly the nearest.
    fast = {(ib, copy): _rotation_z(0.03, (0.3, 0.18, 0.0)) for ib, copy in everything if ib != host["ib_hash"]}
    state = build(far, everything, everything, stale=fast)
    for copy in (0, 1):
        assert matched(match(state, copy)[0], copy), copy
    # A component skinned only once is taken without a record (a single copy that has just appeared).
    single = [(ib, 0) for ib in bone_sets]
    state = build(far, single, [])
    assert matched(match(state, 0)[0], 0)
    # Only the first copy's drawn part and another copy's instance of a component were skinned this frame
    # (as far as drawn parts go a single copy, and a single candidate): it is still not taken when a record
    # of this frame puts it elsewhere, or when the other copy keeps it.
    lone = [(ib, 0) for ib in bone_sets if ib != "cccc0003"] + [("cccc0003", 1)]
    state = build(far, lone, lone)
    assert match(state, 0)[0][0, 0] == 0, "a single candidate a record of this frame puts elsewhere"
    state = build(far, lone, [(ib, copy) for ib, copy in lone if copy == 0], owners=kept(1, frame - 1))
    assert match(state, 0)[0][0, 0] == 0, "a single candidate another copy keeps"

    # A copy appears 10 cm from one that keeps nothing yet and whose records are of the previous frame (it
    # is drawn after the part being skinned): nothing tells the copies' instances apart, so the new copy is
    # not matched rather than borrowing.  With records of this frame, 10 cm is elsewhere: it takes its own.
    behind = {(ib, 0): numpy.eye(4) for ib in bone_sets if ib != host["ib_hash"]}
    state = build(near, everything, [(ib, 0) for ib in bone_sets], stale=behind)
    assert match(state, 1)[0][0, 0] == 0, "a new copy 10 cm away must not borrow bones"
    assert not owners_of(state).any(), "a copy that is not matched keeps nothing"
    state = build(near, everything, [(ib, 0) for ib in bone_sets])
    assert matched(match(state, 1)[0], 1), "records of this frame put the other copy's instances elsewhere"
    assert matched(match(state, 0)[0], 0), "records of this frame agree exactly"
    # The same, the first copy having been matched before (owner entries of two frames ago, written when
    # its other buffer was the drawn one, as the game alternates them): the new copy takes the only
    # instances nobody keeps, its own; the first copy keeps its own although its records are of the
    # previous frame and 15 cm off.
    moving = {(ib, 0): _rotation_z(0.0, (0.15, 0.0, 0.0)) for ib in bone_sets if ib != host["ib_hash"]}
    for order in ((1, 0), (0, 1)):
        state = build(near, everything, [(ib, 0) for ib in bone_sets], stale=moving,
                      owners=kept(0, frame - 2, keys_of(0)[::-1]))
        for copy in order:
            assert matched(match(state, copy)[0], copy), (order, copy)
    # Owner entries older than OWNER_FRAMES are no longer kept: as if nobody kept anything.
    state = build(near, everything, [(ib, 0) for ib in bone_sets], stale=moving, owners=kept(0, frame - 10))
    assert match(state, 1)[0][0, 0] == 0, "stale owner entries must not exclude anything"
    # Owner entries that records of this frame contradict (each copy keeping the other's instances) are not
    # followed: neither copy is matched until the entries lapse.
    state = build(far, everything, everything,
                  owners=kept(0, frame - 1, keys_of(1)) + kept(1, frame - 1, keys_of(0)))
    for copy in (0, 1):
        assert match(state, copy)[0][0, 0] == 0, copy
    assert worst < 1e-4, worst
    return worst


def main() -> int:
    if sys.platform != "win32":
        print("ZZMI_MERGED_GPU_RUNTIME=SKIP (not Windows)")
        return 0
    instance_error = check_instances(load_module())
    skin_error = check_same_frame_skin(load_module())
    place_error = check_placement(load_module())
    module = load_module()
    profile, palettes, roots, world_bones, bone_sets = synthetic_case(module)
    results = run_runtime(module, profile, palettes)
    delay_error = check_frame_delay(module, profile, palettes)

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
    print("ZZMI_MERGED_GPU_RUNTIME=PASS (reference %.1e, truth %.1e, delayed %.1e, placement %.1e, same-frame skin %.1e,"
          " instances %.1e)" % (worst_reference, worst_truth, delay_error, place_error, skin_error, instance_error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
