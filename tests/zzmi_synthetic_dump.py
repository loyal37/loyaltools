# -*- coding: utf-8 -*-
"""Builds a small 3dmigoto frame analysis folder shaped like a Zenless Zone Zero dump.

The layout follows a real ZZMI capture: characters are pre-skinned by a
stream-output ``Draw`` (pointlist, vb0=position vb1=texcoord vb2=blend) and
then drawn with ``DrawIndexed`` using the stream-output buffer as vb0. The
blend buffer stays bound at vb2 for the indexed draws but is not dumped there
because the input layout does not reference it.

Three objects are written:

* ``body``: two parts, four-influence blend buffer, 8-bit vertex colours.
* ``face``: one part, single-index blend buffer (no weights), float colours.
  It has the same vertex count as ``body``, so both stream-output buffers get
  the same hash and only the shared texcoord/blend hashes tell them apart.
* ``prop``: a static mesh inside a shared interleaved buffer, drawn with a
  non-zero BaseVertexLocation.

The body's normal map is dumped as a jpg, as ZZMI does without the ``dds``
analysis option. The jpg written here is only a header; a regression that
decodes it replaces the file with a real image first.

The tangent handedness matches what the real game data shows: the tangent
follows dP/dU and ``w = sign(dot(cross(N, T), dP/dV))`` with D3D texture
coordinates (v pointing down).
"""

from __future__ import annotations

import os
import struct

import numpy


VS_POSE = "1111111111111111"
VS_COLOR = "2222222222222222"
VS_OUTLINE = "3333333333333333"
PS_COLOR = "4444444444444444"
PS_OUTLINE = "5555555555555555"

BODY = {
    "ib": "0a0a0a01", "position": "0b0b0b01", "texcoord": "0c0c0c01", "blend": "0d0d0d01",
    "draw": "0e0e0e00", "pose_id": 1,
}
FACE = {
    "ib": "0a0a0a02", "position": "0b0b0b02", "texcoord": "0c0c0c02", "blend": "0d0d0d02",
    "draw": "0e0e0e00", "pose_id": 2,
}
PROP = {"ib": "0a0a0a03", "vb": "0b0b0b03"}

TEX_DIFFUSE = "71000001"
TEX_NORMAL = "72000001"
TEX_LIGHT = "73000001"
TEX_MATERIAL = "74000001"
TEX_FACE_DIFFUSE = "71000002"
TEX_FACE_NORMAL = "72000002"
RENDER_TARGET = "7f000001"
DEPTH_TARGET = "7f000002"

POSITION_DTYPE = numpy.dtype([("POSITION", "<f4", 3), ("NORMAL", "<f4", 3), ("TANGENT", "<f4", 4)])
BODY_TEXCOORD_DTYPE = numpy.dtype([
    ("COLOR", "u1", 4), ("TEXCOORD", "<f2", 2), ("TEXCOORD1", "<f4", 2),
    ("TEXCOORD2", "<f2", 2), ("TEXCOORD3", "<f2", 2),
])
BODY_BLEND_DTYPE = numpy.dtype([("BLENDWEIGHTS", "<f4", 4), ("BLENDINDICES", "<u4", 4)])
FACE_TEXCOORD_DTYPE = numpy.dtype([
    ("COLOR", "<f4", 4), ("TEXCOORD", "<f4", 2), ("TEXCOORD1", "<f4", 2),
    ("TEXCOORD2", "<f4", 2), ("TEXCOORD3", "<f4", 2),
])
FACE_BLEND_DTYPE = numpy.dtype([("BLENDINDICES", "<u4", 1)])
PROP_DTYPE = numpy.dtype([
    ("POSITION", "<f4", 3), ("NORMAL", "<f4", 3), ("TANGENT", "<f4", 4), ("TEXCOORD", "<f4", 2),
])

_BOGUS = ("R8G8B8A8_UNORM", 0, 0)


def _grid(columns, rows, origin, cell, mirror_u):
    """A flat grid in the XZ plane facing -Y, plus its D3D texture coordinates."""
    positions, uvs = [], []
    for j in range(rows + 1):
        for i in range(columns + 1):
            positions.append((origin[0] + i * cell, origin[1], origin[2] + j * cell))
            u = i / columns
            uvs.append((1.0 - u if mirror_u else u, 1.0 - j / rows))
    indices = []
    for j in range(rows):
        for i in range(columns):
            a = j * (columns + 1) + i
            b, d = a + 1, a + columns + 1
            c = d + 1
            indices += [a, b, c, a, c, d]
    tangent = (-1.0, 0.0, 0.0, 1.0) if mirror_u else (1.0, 0.0, 0.0, -1.0)
    return (
        numpy.array(positions, dtype=numpy.float32),
        numpy.array(uvs, dtype=numpy.float32),
        numpy.array(indices, dtype=numpy.int64),
        tangent,
    )


def _object_geometry(origin_x):
    """25 + 4 vertices: a 4x4 grid followed by a mirrored-UV quad."""
    grid_p, grid_uv, grid_i, grid_t = _grid(4, 4, (origin_x, 0.0, 1.0), 0.125, mirror_u=False)
    quad_p, quad_uv, quad_i, quad_t = _grid(1, 1, (origin_x, 0.0, 2.0), 0.25, mirror_u=True)
    position = numpy.zeros(len(grid_p) + len(quad_p), dtype=POSITION_DTYPE)
    position["POSITION"] = numpy.concatenate([grid_p, quad_p])
    position["NORMAL"] = (0.0, -1.0, 0.0)
    position["TANGENT"][:len(grid_p)] = grid_t
    position["TANGENT"][len(grid_p):] = quad_t
    uvs = numpy.concatenate([grid_uv, quad_uv])
    return position, uvs, grid_i, quad_i + len(grid_p)


def _element(name, index, fmt, slot, offset):
    return (name, index, fmt, slot, offset)


def _vb_text(stride, vertex_count, topology, elements):
    lines = [
        "stride: %d" % stride, "first vertex: 0", "vertex count: %d" % vertex_count,
        "topology: " + topology,
    ]
    for number, (name, index, fmt, slot, offset) in enumerate(elements):
        lines += [
            "element[%d]:" % number, "  SemanticName: " + name, "  SemanticIndex: %d" % index,
            "  Format: " + fmt, "  InputSlot: %d" % slot, "  AlignedByteOffset: %d" % offset,
            "  InputSlotClass: per-vertex", "  InstanceDataStepRate: 0",
        ]
    return "\n".join(lines + ["", "vertex-data:", "", ""])


def _ib_text(first_index, index_count):
    return "\n".join([
        "byte offset: 0", "first index: %d" % first_index, "index count: %d" % index_count,
        "topology: trianglelist", "format: DXGI_FORMAT_R16_UINT", "", "0 1 2", "",
    ])


def _dds(width, height, dxgi_format, payload_size):
    header = bytearray(148)
    header[0:4] = b"DDS "
    struct.pack_into("<IIIII", header, 4, 124, 0x1007, height, width, 0)
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    header[84:88] = b"DX10"
    struct.pack_into("<IIIII", header, 128, dxgi_format, 3, 0, 1, 0)
    return bytes(header) + bytes(payload_size)


def _jpg(width, height):
    return (
        b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        + b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3)
        + b"\x01\x11\x00\x02\x11\x00\x03\x11\x00" + b"\xff\xd9"
    )


class _DumpWriter:
    def __init__(self, folder):
        self.folder = folder
        self.log = ["analyse_options: 0000063d"]
        os.makedirs(os.path.join(folder, "deduped"), exist_ok=True)

    def write(self, name, data):
        mode = "w" if isinstance(data, str) else "wb"
        with open(os.path.join(self.folder, name), mode) as handle:
            handle.write(data)

    def call(self, draw_id, text, slots=()):
        self.log.append("%06d %s" % (draw_id, text))
        for slot, resource_hash in slots:
            self.log.append("       %s: resource=0x00000215B3BA37E0 hash=%s" % (slot, resource_hash))

    def vertex_buffers(self, draw_id, hashes):
        self.call(
            draw_id,
            "IASetVertexBuffers(StartSlot:0, NumBuffers:%d, ppVertexBuffers:0x000000D507A2DA60, "
            "pStrides:0x000000D507A2DA40, pOffsets:0x000000D507A2DA20)" % len(hashes),
            list(enumerate(hashes)),
        )

    def shader(self, draw_id, kind, shader_hash):
        text = "%sSetShader(p%sShader:0x0000021500000000, ppClassInstances:0x0000000000000000, NumClassInstances:0)" % (
            kind, {"VS": "Vertex", "PS": "Pixel"}[kind])
        self.call(draw_id, text + (" hash=" + shader_hash if shader_hash else ""))

    def texture(self, draw_id, slot, tex_hash, shaders, extension, data, format_name, contaminated=False):
        name = "%06d-ps-t%d=%s%s%s.%s" % (
            draw_id, slot, "!S!=" if contaminated else "", tex_hash, shaders, extension)
        self.write(name, data)
        self.log.append("%06d 3DMigoto Dumping Texture2D %s -> %s" % (
            draw_id, os.path.join(self.folder, name),
            os.path.join(self.folder, "deduped", tex_hash + "-" + format_name + "." + extension)))

    def finish(self):
        self.write("log.txt", "\n".join(self.log) + "\n")


def _body_buffers():
    position, uvs, grid_indices, quad_indices = _object_geometry(0.0)
    count = len(position)
    texcoord = numpy.zeros(count, dtype=BODY_TEXCOORD_DTYPE)
    blend = numpy.zeros(count, dtype=BODY_BLEND_DTYPE)
    for vertex in range(count):
        # 160 / 161 / 255 are the values an sRGB byte colour layer cannot keep apart.
        texcoord["COLOR"][vertex] = (160 + vertex % 2, 255 - vertex, 32 + vertex, vertex % 3)
        texcoord["TEXCOORD1"][vertex] = ((vertex % 5) * 0.25 - 0.5, (vertex % 3) * 0.5 - 0.5)
        texcoord["TEXCOORD2"][vertex] = ((vertex % 4) * 0.25, (vertex % 2) * 0.5)
        texcoord["TEXCOORD3"][vertex] = ((vertex % 8) * 0.125, 0.25)
        if vertex % 4 == 0:
            blend["BLENDWEIGHTS"][vertex] = (1.0, 0.0, 0.0, 0.0)
            blend["BLENDINDICES"][vertex] = (vertex % 7, 0, 0, 0)
        else:
            blend["BLENDWEIGHTS"][vertex] = (0.5, 0.25, 0.1875, 0.0625)
            blend["BLENDINDICES"][vertex] = [(vertex + offset) % 7 for offset in range(4)]
    texcoord["TEXCOORD"] = uvs
    return position, texcoord, blend, grid_indices, quad_indices


def _face_buffers():
    position, uvs, grid_indices, quad_indices = _object_geometry(4.0)
    count = len(position)
    texcoord = numpy.zeros(count, dtype=FACE_TEXCOORD_DTYPE)
    blend = numpy.zeros(count, dtype=FACE_BLEND_DTYPE)
    for vertex in range(count):
        texcoord["COLOR"][vertex] = (vertex / 1024.0, 1.0, 0.0, 0.5)
        texcoord["TEXCOORD1"][vertex] = (0.25, -0.25)
        texcoord["TEXCOORD2"][vertex] = (0.5, 0.5)
        texcoord["TEXCOORD3"][vertex] = (0.0, 0.75)
        blend["BLENDINDICES"][vertex] = vertex % 3
    texcoord["TEXCOORD"] = uvs
    return position, texcoord, blend, numpy.concatenate([grid_indices, quad_indices])


POSITION_ELEMENTS = [
    _element("POSITION", 0, "R32G32B32_FLOAT", 0, 0),
    _element("NORMAL", 0, "R32G32B32_FLOAT", 0, 12),
    _element("TANGENT", 0, "R32G32B32A32_FLOAT", 0, 24),
]
BODY_TEXCOORD_ELEMENTS = [
    _element("COLOR", 0, "R8G8B8A8_UNORM", 1, 0),
    _element("TEXCOORD", 0, "R16G16_FLOAT", 1, 4),
    _element("TEXCOORD", 1, "R32G32_FLOAT", 1, 8),
    _element("TEXCOORD", 2, "R16G16_FLOAT", 1, 16),
    _element("TEXCOORD", 3, "R16G16_FLOAT", 1, 20),
]
FACE_TEXCOORD_ELEMENTS = [
    _element("COLOR", 0, "R32G32B32A32_FLOAT", 1, 0),
    _element("TEXCOORD", 0, "R32G32_FLOAT", 1, 16),
    _element("TEXCOORD", 1, "R32G32_FLOAT", 1, 24),
    _element("TEXCOORD", 2, "R32G32_FLOAT", 1, 32),
    _element("TEXCOORD", 3, "R32G32_FLOAT", 1, 40),
]
# Shader inputs the mesh does not provide show up as slot 0 / offset 0 / R8G8B8A8_UNORM.
BOGUS_ELEMENTS = [_element("TEXCOORD", index, *_BOGUS) for index in (4, 5, 6, 7)]
BODY_BLEND_ELEMENTS = [
    _element("BLENDWEIGHTS", 0, "R32G32B32A32_FLOAT", 2, 0),
    _element("BLENDINDICES", 0, "R32G32B32A32_UINT", 2, 16),
]
FACE_BLEND_ELEMENTS = [_element("BLENDINDICES", 0, "R32_UINT", 2, 0)]


def _write_pose_draw(writer, spec, position, texcoord, blend, texcoord_elements, blend_elements):
    draw_id = spec["pose_id"]
    shaders = "-vs=" + VS_POSE
    layout = POSITION_ELEMENTS + texcoord_elements + BOGUS_ELEMENTS + blend_elements
    writer.vertex_buffers(draw_id, [spec["position"], spec["texcoord"], spec["blend"]])
    writer.call(
        draw_id,
        "SOSetTargets(NumBuffers:1, ppSOTargets:0x000000D507A2DB00, pOffsets:0x000000D507A2DAF0)",
        [(0, spec["draw"])],
    )
    writer.call(draw_id, "IASetPrimitiveTopology(Topology:1)")
    writer.shader(draw_id, "VS", VS_POSE)
    writer.shader(draw_id, "PS", "")
    writer.call(draw_id, "Draw(VertexCount:%d, StartVertexLocation:0)" % len(position))
    # The stream-output target is unbound right after the draw; it is logged under the next id.
    writer.call(
        draw_id + 1,
        "SOSetTargets(NumBuffers:1, ppSOTargets:0x000000D507A2D970, pOffsets:0x000000D507A2DAF8)",
    )
    for slot, (buffer_hash, data) in enumerate([
        (spec["position"], position), (spec["texcoord"], texcoord), (spec["blend"], blend),
    ]):
        stem = "%06d-vb%d=%s%s" % (draw_id, slot, buffer_hash, shaders)
        writer.write(stem + ".buf", data.tobytes())
        writer.write(stem + ".txt", _vb_text(data.dtype.itemsize, len(data), "pointlist", layout))
    # An index buffer left bound from the previous frame is dumped even for a non-indexed draw.
    stale = "%06d-ib=0f0f0f0f%s" % (draw_id, shaders)
    writer.write(stale + ".buf", numpy.arange(6, dtype=numpy.uint16).tobytes())
    writer.write(stale + ".txt", "byte offset: 0\ntopology: pointlist\nformat: DXGI_FORMAT_R16_UINT\n\n0\n1\n")


def _write_indexed_draw(writer, draw_id, spec, vs_hash, ps_hash, first_index, index_count,
                        index_data, posed, texcoord, texcoord_elements, textures):
    shaders = "-vs=" + vs_hash + "-ps=" + ps_hash
    writer.vertex_buffers(draw_id, [spec["draw"], spec["texcoord"], spec["blend"], spec["draw"]])
    writer.call(
        draw_id,
        "OMSetRenderTargets(NumViews:1, ppRenderTargetViews:0x000000D507A2B5D0, pDepthStencilView:0x000002144B2A59A0)",
        [(0, RENDER_TARGET), ("D", DEPTH_TARGET)],
    )
    writer.call(draw_id, "IASetPrimitiveTopology(Topology:4)")
    writer.shader(draw_id, "VS", vs_hash)
    writer.shader(draw_id, "PS", ps_hash)
    writer.call(
        draw_id,
        "IASetIndexBuffer(pIndexBuffer:0x000002153EF21860, Format:57, Offset:0) hash=" + spec["ib"],
    )
    writer.call(
        draw_id,
        "DrawIndexed(IndexCount:%d, StartIndexLocation:%d, BaseVertexLocation:0)" % (index_count, first_index),
    )
    stem = "%06d-ib=%s%s" % (draw_id, spec["ib"], shaders)
    writer.write(stem + ".buf", index_data.astype(numpy.uint16).tobytes())
    writer.write(stem + ".txt", _ib_text(first_index, index_count))

    # vb0 / vb3 hold the posed copy. In this pass the tangent input is unused, so the header
    # only describes 24 of the 40 bytes; the texcoord header repeats an offset.
    draw_layout = (
        POSITION_ELEMENTS[:2] + [_element("TANGENT", 0, *_BOGUS)] + texcoord_elements[:-1]
        + [_element("TEXCOORD", 3, texcoord_elements[-2][2], 1, texcoord_elements[-2][4])]
        + [_element("TEXCOORD", 4, "R32G32B32_FLOAT", 3, 0)]
    )
    for slot, buffer_hash, data in ((0, spec["draw"], posed), (1, spec["texcoord"], texcoord), (3, spec["draw"], posed)):
        stem = "%06d-vb%d=%s%s" % (draw_id, slot, buffer_hash, shaders)
        writer.write(stem + ".buf", data.tobytes())
        writer.write(stem + ".txt", _vb_text(data.dtype.itemsize, len(data), "trianglelist", draw_layout))

    writer.write("%06d-o0=%s%s.jpg" % (draw_id, RENDER_TARGET, shaders), _jpg(64, 64))
    for slot, tex_hash, extension, data, format_name, contaminated in textures:
        writer.texture(draw_id, slot, tex_hash, shaders, extension, data, format_name, contaminated)


def build_dump(folder):
    """Write the dump and return the source buffers the regressions compare against."""
    writer = _DumpWriter(folder)

    body_position, body_texcoord, body_blend, body_grid, body_quad = _body_buffers()
    face_position, face_texcoord, face_blend, face_indices = _face_buffers()
    body_indices = numpy.concatenate([body_grid, body_quad])

    _write_pose_draw(writer, BODY, body_position, body_texcoord, body_blend,
                     BODY_TEXCOORD_ELEMENTS, BODY_BLEND_ELEMENTS)
    _write_pose_draw(writer, FACE, face_position, face_texcoord, face_blend,
                     FACE_TEXCOORD_ELEMENTS, FACE_BLEND_ELEMENTS)

    # The posed copies are deliberately displaced: extracting them would move the mesh.
    body_posed = body_position.copy()
    body_posed["POSITION"] += (10.0, 10.0, 10.0)
    face_posed = face_position.copy()
    face_posed["POSITION"] += (10.0, 10.0, 10.0)

    diffuse = _dds(256, 256, 99, 65536)
    light = _dds(256, 256, 95, 65536)
    color_textures = [
        (0, RENDER_TARGET, "jpg", _jpg(64, 64), "R8G8B8A8_UNORM_SRGB", False),
        (1, DEPTH_TARGET, "jpg", _jpg(64, 64), "R16_UNORM", False),
        (3, TEX_DIFFUSE, "dds", diffuse, "BC7_UNORM_SRGB", False),
        (4, TEX_NORMAL, "jpg", _jpg(256, 256), "R8G8B8A8_UNORM", True),
        (5, TEX_LIGHT, "dds", light, "BC6H_UF16", False),
        (6, TEX_MATERIAL, "dds", light, "BC6H_UF16", False),
    ]
    outline_textures = [
        (2, TEX_DIFFUSE, "dds", diffuse, "BC7_UNORM_SRGB", False),
        (3, TEX_LIGHT, "dds", light, "BC6H_UF16", False),
    ]
    _write_indexed_draw(writer, 10, BODY, VS_COLOR, PS_COLOR, 0, len(body_grid), body_indices,
                        body_posed, body_texcoord, BODY_TEXCOORD_ELEMENTS, color_textures)
    _write_indexed_draw(writer, 11, BODY, VS_COLOR, PS_COLOR, len(body_grid), len(body_quad), body_indices,
                        body_posed, body_texcoord, BODY_TEXCOORD_ELEMENTS, color_textures)
    _write_indexed_draw(writer, 12, BODY, VS_OUTLINE, PS_OUTLINE, 0, len(body_grid), body_indices,
                        body_posed, body_texcoord, BODY_TEXCOORD_ELEMENTS, outline_textures)
    _write_indexed_draw(writer, 13, FACE, VS_COLOR, PS_COLOR, 0, len(face_indices), face_indices,
                        face_posed, face_texcoord, FACE_TEXCOORD_ELEMENTS, [
                            (3, TEX_FACE_DIFFUSE, "dds", diffuse, "BC7_UNORM_SRGB", False),
                            (4, TEX_FACE_NORMAL, "jpg", _jpg(256, 256), "R16G16B16A16_FLOAT", False),
                        ])

    # Static prop: vertices 5..8 of a shared interleaved buffer, indices 6..11 of a shared IB.
    prop_position, prop_uv, prop_local_indices, prop_tangent = _grid(1, 1, (8.0, 0.0, 1.0), 0.5, mirror_u=False)
    prop_buffer = numpy.zeros(12, dtype=PROP_DTYPE)
    prop_buffer["POSITION"][5:9] = prop_position
    prop_buffer["NORMAL"][5:9] = (0.0, -1.0, 0.0)
    prop_buffer["TANGENT"][5:9] = prop_tangent
    prop_buffer["TEXCOORD"][5:9] = prop_uv
    prop_indices = numpy.concatenate([numpy.zeros(6, dtype=numpy.int64), prop_local_indices])
    prop_shaders = "-vs=" + VS_COLOR + "-ps=" + PS_COLOR
    writer.vertex_buffers(14, [PROP["vb"]])
    writer.call(14, "IASetIndexBuffer(pIndexBuffer:0x000002153EF21861, Format:57, Offset:0) hash=" + PROP["ib"])
    writer.call(14, "DrawIndexed(IndexCount:6, StartIndexLocation:6, BaseVertexLocation:5)")
    writer.write("000014-ib=%s%s.buf" % (PROP["ib"], prop_shaders), prop_indices.astype(numpy.uint16).tobytes())
    writer.write("000014-ib=%s%s.txt" % (PROP["ib"], prop_shaders), _ib_text(6, 6))
    writer.write("000014-vb0=%s%s.buf" % (PROP["vb"], prop_shaders), prop_buffer.tobytes())
    writer.write("000014-vb0=%s%s.txt" % (PROP["vb"], prop_shaders), _vb_text(48, 12, "trianglelist", [
        _element("POSITION", 0, "R32G32B32_FLOAT", 0, 0),
        _element("NORMAL", 0, "R32G32B32_FLOAT", 0, 12),
        _element("TANGENT", 0, "R32G32B32A32_FLOAT", 0, 24),
        _element("TEXCOORD", 0, "R32G32_FLOAT", 0, 40),
    ]))
    writer.finish()

    return {
        "body": {
            "position": body_position, "texcoord": body_texcoord, "blend": body_blend,
            "grid_indices": body_grid, "quad_indices": body_quad,
        },
        "face": {
            "position": face_position, "texcoord": face_texcoord, "blend": face_blend,
            "indices": face_indices,
        },
        "prop": {"buffer": prop_buffer[5:9], "indices": prop_local_indices},
    }
