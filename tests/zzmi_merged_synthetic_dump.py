# -*- coding: utf-8 -*-
"""Builds a Zenless Zone Zero style frame analysis folder for the merged skeleton.

Every component is skinned in the space of its own root bone, as the game does:
the palette bound at ``vs-t0`` of the stream-output draw is
``inverse(root) * world_bone`` and the indexed draws carry ``root`` in the first
four rows of ``vs-cb1``.  Components that share a bone agree on it in world
space only.

Components (in skinning order):

* ``body``      4 weights, bones 0..7, root = hips.
* ``hair``      4 weights, bones 6, 8..11 (6 = head, shared with the body).
* ``face``      shape keyed: its position buffer is copied and rewritten by a
                compute pass right before its skinning draw.  Uses the head.
* ``accessory`` one index per vertex (the rigid skinning shader), bone 11,
                shared with the hair.  Skinned last.
* ``bangboo``   one index per vertex, bones 20..21 that no other component
                uses: another object in the frame.

Vertex buffers get distinct resource addresses in log.txt, so the face's
GPU-written buffer can be told apart from the others.
"""

from __future__ import annotations

import numpy

import zzmi_synthetic_dump as base


VS_RIGID = "6666666666666666"

COMPONENTS = [
    {"name": "body", "ib": "1a1a1a01", "position": "1b1b1b01", "texcoord": "1c1c1c01", "blend": "1d1d1d01",
     "draw": "1e1e1e01", "pose_id": 1, "bones": [0, 1, 2, 3, 4, 5, 6, 7], "root": 0, "rigid": False, "origin": 0.0},
    {"name": "hair", "ib": "1a1a1a02", "position": "1b1b1b02", "texcoord": "1c1c1c02", "blend": "1d1d1d02",
     "draw": "1e1e1e02", "pose_id": 2, "bones": [6, 8, 9, 10, 11], "root": 6, "rigid": False, "origin": 3.0},
    {"name": "face", "ib": "1a1a1a03", "position": "1b1b1b03", "texcoord": "1c1c1c03", "blend": "1d1d1d03",
     "draw": "1e1e1e03", "pose_id": 4, "bones": [6, 12, 13], "root": 6, "rigid": False, "origin": 6.0,
     "shape_keyed": True},
    {"name": "accessory", "ib": "1a1a1a04", "position": "1b1b1b04", "texcoord": "1c1c1c04", "blend": "1d1d1d04",
     "draw": "1e1e1e04", "pose_id": 5, "bones": [11, 14], "root": 11, "rigid": True, "origin": 9.0},
    {"name": "bangboo", "ib": "1a1a1a05", "position": "1b1b1b05", "texcoord": "1c1c1c05", "blend": "1d1d1d05",
     "draw": "1e1e1e05", "pose_id": 6, "bones": [20, 21], "root": 20, "rigid": True, "origin": 12.0},
]
SHAPE_KEY_SOURCE = "1f1f1f03"
BONE_COUNT = 22


def _pointer(resource_hash: str) -> str:
    return "0x0000021%09X" % int(resource_hash, 16)


def _rigid(rng) -> numpy.ndarray:
    q, _ = numpy.linalg.qr(rng.normal(size=(3, 3)))
    if numpy.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    matrix = numpy.eye(4)
    matrix[:3, :3] = q
    matrix[:3, 3] = rng.normal(scale=0.4, size=3) + (600.0, 1.0, 0.0)
    return matrix


class _Writer(base._DumpWriter):
    def call(self, draw_id, text, slots=()):
        self.log.append("%06d %s" % (draw_id, text))
        for slot, resource_hash in slots:
            self.log.append("       %s: resource=%s hash=%s" % (slot, _pointer(resource_hash), resource_hash))


def _texcoord_and_blend(component, count):
    texcoord = numpy.zeros(count, dtype=base.BODY_TEXCOORD_DTYPE)
    if component["rigid"]:
        blend = numpy.zeros(count, dtype=base.FACE_BLEND_DTYPE)
    else:
        blend = numpy.zeros(count, dtype=base.BODY_BLEND_DTYPE)
    bone_count = len(component["bones"])
    for vertex in range(count):
        texcoord["COLOR"][vertex] = (128, 128, 0, 0)
        texcoord["TEXCOORD1"][vertex] = (0.0, 0.0)
        if component["rigid"]:
            blend["BLENDINDICES"][vertex] = vertex % bone_count
        else:
            blend["BLENDWEIGHTS"][vertex] = (0.625, 0.375, 0.0, 0.0)
            blend["BLENDINDICES"][vertex] = (vertex % bone_count, (vertex + 1) % bone_count, 0, 0)
    return texcoord, blend


def _skin(palette, position, blend, rigid):
    if rigid:
        matrices = palette[blend["BLENDINDICES"].reshape(-1)]
    else:
        matrices = numpy.zeros((len(position), 3, 4))
        for k in range(4):
            matrices += blend["BLENDWEIGHTS"][:, k, None, None] * palette[blend["BLENDINDICES"][:, k]]
    points = numpy.c_[position["POSITION"], numpy.ones(len(position))]
    posed = position.copy()
    posed["POSITION"] = numpy.einsum("vij,vj->vi", matrices, points)
    posed["NORMAL"] = numpy.einsum("vij,vj->vi", matrices[:, :, :3], position["NORMAL"])
    posed["TANGENT"][:, :3] = numpy.einsum("vij,vj->vi", matrices[:, :, :3], position["TANGENT"][:, :3])
    return posed


def build_dump(folder):
    """Write the dump; return {name: {...}} with the data the regressions need."""
    rng = numpy.random.default_rng(20261021)
    world_bones = [_rigid(rng) for _ in range(BONE_COUNT)]
    writer = _Writer(folder)
    diffuse = base._dds(256, 256, 99, 65536)
    result = {"world_bones": world_bones, "components": {}}

    for component in COMPONENTS:
        position, uvs, grid_indices, quad_indices = base._object_geometry(component["origin"])
        indices = numpy.concatenate([grid_indices, quad_indices])
        texcoord, blend = _texcoord_and_blend(component, len(position))
        texcoord["TEXCOORD"] = uvs
        root = world_bones[component["root"]]
        palette = numpy.stack([
            (numpy.linalg.inv(root) @ world_bones[bone])[:3] for bone in component["bones"]
        ]).astype(numpy.float32)
        posed = _skin(palette.astype(numpy.float64), position, blend, component["rigid"])

        pose_id = component["pose_id"]
        pose_vs = VS_RIGID if component["rigid"] else base.VS_POSE
        shaders = "-vs=" + pose_vs
        blend_elements = base.FACE_BLEND_ELEMENTS if component["rigid"] else base.BODY_BLEND_ELEMENTS
        layout = base.POSITION_ELEMENTS + base.BODY_TEXCOORD_ELEMENTS + blend_elements
        if component.get("shape_keyed"):
            # The game copies the base mesh and a compute pass adds the expression.
            writer.call(pose_id, "CopyResource(pDstResource:%s, pSrcResource:%s)" % (
                _pointer(component["position"]), _pointer(SHAPE_KEY_SOURCE)),
                [("Src", SHAPE_KEY_SOURCE), ("Dst", component["position"])])
            writer.call(pose_id, "CSSetUnorderedAccessViews(StartSlot:0, NumUAVs:1, "
                        "ppUnorderedAccessViews:0x000000E7901CD9C0, pUAVInitialCounts:0x000000E7901CD9B8)",
                        [(0, component["position"])])
            writer.call(pose_id, "Dispatch(ThreadGroupCountX:1, ThreadGroupCountY:1, ThreadGroupCountZ:1)")
        writer.vertex_buffers(pose_id, [component["position"], component["texcoord"], component["blend"]])
        writer.call(pose_id, "SOSetTargets(NumBuffers:1, ppSOTargets:0x000000D507A2DB00, "
                    "pOffsets:0x000000D507A2DAF0)", [(0, component["draw"])])
        writer.call(pose_id, "IASetPrimitiveTopology(Topology:1)")
        writer.shader(pose_id, "VS", pose_vs)
        writer.call(pose_id, "Draw(VertexCount:%d, StartVertexLocation:0)" % len(position))
        for slot, (buffer_hash, data) in enumerate([
            (component["position"], position), (component["texcoord"], texcoord), (component["blend"], blend),
        ]):
            stem = "%06d-vb%d=%s%s" % (pose_id, slot, buffer_hash, shaders)
            writer.write(stem + ".buf", data.tobytes())
            writer.write(stem + ".txt", base._vb_text(data.dtype.itemsize, len(data), "pointlist", layout))
        writer.write("%06d-vs-t0=%08x%s.buf" % (pose_id, 0x2a000000 + pose_id, shaders), palette.tobytes())

        draw_id = 20 + pose_id
        draw_shaders = "-vs=" + base.VS_COLOR + "-ps=" + base.PS_COLOR
        writer.vertex_buffers(draw_id, [component["draw"], component["texcoord"], component["blend"], component["draw"]])
        writer.call(draw_id, "IASetPrimitiveTopology(Topology:4)")
        writer.shader(draw_id, "VS", base.VS_COLOR)
        writer.shader(draw_id, "PS", base.PS_COLOR)
        writer.call(draw_id, "IASetIndexBuffer(pIndexBuffer:0x000002153EF21860, Format:57, Offset:0) hash="
                    + component["ib"])
        writer.call(draw_id, "DrawIndexed(IndexCount:%d, StartIndexLocation:0, BaseVertexLocation:0)" % len(indices))
        stem = "%06d-ib=%s%s" % (draw_id, component["ib"], draw_shaders)
        writer.write(stem + ".buf", indices.astype(numpy.uint16).tobytes())
        writer.write(stem + ".txt", base._ib_text(0, len(indices)))
        for slot, buffer_hash, data in ((0, component["draw"], posed), (1, component["texcoord"], texcoord),
                                        (3, component["draw"], posed)):
            stem = "%06d-vb%d=%s%s" % (draw_id, slot, buffer_hash, draw_shaders)
            writer.write(stem + ".buf", data.tobytes())
            writer.write(stem + ".txt", base._vb_text(data.dtype.itemsize, len(data), "trianglelist",
                                                     base.POSITION_ELEMENTS + base.BODY_TEXCOORD_ELEMENTS))
        constants = numpy.zeros((29, 4), dtype=numpy.float32)
        constants[0:4] = root.T          # rows 0..3: object to world, row-vector convention
        writer.write("%06d-vs-cb1=%08x%s.buf" % (draw_id, 0x2b000000 + draw_id, draw_shaders), constants.tobytes())
        writer.texture(draw_id, 3, "%08x" % (0x71100000 + pose_id), draw_shaders, "dds", diffuse, "BC7_UNORM_SRGB")
        if component.get("shape_keyed"):
            normal, normal_format = base._dds(256, 256, 10, 65536), "R16G16B16A16_FLOAT"
        else:
            normal, normal_format = base._dds(256, 256, 28, 65536), "R8G8B8A8_UNORM"
        writer.texture(draw_id, 4, "%08x" % (0x72100000 + pose_id), draw_shaders, "dds", normal, normal_format)

        result["components"][component["name"]] = {
            "spec": component, "position": position, "texcoord": texcoord, "blend": blend,
            "indices": indices, "palette": palette, "posed": posed, "root": root,
            "unique_str": component["ib"] + "-" + str(len(indices)) + "-0",
        }
    writer.finish()
    return result
