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
D3D11_BIND_CONSTANT_BUFFER = 0x4
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

    def constant_buffer(self, data: numpy.ndarray):
        data = numpy.ascontiguousarray(data)
        desc = BufferDesc(data.nbytes, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0)
        initial = SubresourceData(data.ctypes.data, 0, 0)
        buffer = ctypes.c_void_p()
        _check(_method(self.device, 3, ctypes.c_long, ctypes.POINTER(BufferDesc), ctypes.POINTER(SubresourceData),
                       ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(desc), ctypes.byref(initial), ctypes.byref(buffer)),
               "CreateBuffer(constant)")
        return buffer.value

    def dispatch_bound(self, shader, cbs: list, srvs: list, uavs: list, groups: int):
        """Dispatch with constant buffers and several UAVs bound."""
        set_cbs = _method(self.context, 71, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p)
        set_srvs = _method(self.context, 67, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p)
        set_uavs = _method(self.context, 68, None, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p)
        _method(self.context, 69, None, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT)(shader, None, 0)
        set_cbs(0, len(cbs), (ctypes.c_void_p * len(cbs))(*cbs))
        set_srvs(0, len(srvs), (ctypes.c_void_p * len(srvs))(*srvs))
        set_uavs(0, len(uavs), (ctypes.c_void_p * len(uavs))(*uavs), None)
        _method(self.context, 41, None, wintypes.UINT, wintypes.UINT, wintypes.UINT)(groups, 1, 1)
        set_uavs(0, len(uavs), (ctypes.c_void_p * len(uavs))(), None)
        set_srvs(0, len(srvs), (ctypes.c_void_p * len(srvs))())
        set_cbs(0, len(cbs), (ctypes.c_void_p * len(cbs))())

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


def main() -> int:
    if sys.platform != "win32":
        print("ZZMI_MERGED_GPU_RUNTIME=SKIP (not Windows)")
        return 0
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
    print("ZZMI_MERGED_GPU_RUNTIME=PASS (reference %.1e, truth %.1e, delayed %.1e, placement %.1e)"
          % (worst_reference, worst_truth, delay_error, place_error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
