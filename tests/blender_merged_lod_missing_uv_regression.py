"""Regression for missing secondary float2 UVs in merged-skeleton LODs."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy

from LoyalTools.common.d3d11_gametype import D3D11GameType
from LoyalTools.ui.universal.efmi import ExportEFMI


def _submesh(*, secondary_uv=False, primary_uv=True):
    fields = [
        ("TEXCOORD" if primary_uv else "UNKNOWN", 0, "R32G32_FLOAT", 8),
        ("TEXCOORD" if secondary_uv else "UNKNOWN", 4, "R32G32_FLOAT", 8),
        ("COLOR", 0, "R8G8B8A8_SNORM", 4),
    ]
    game_type = D3D11GameType.from_submesh_json_dict(
        {"GPU-PreSkinning": True, "WorkGameType": "MissingUVRegression",
         "CategoryDrawCategoryMap": {}},
        override_d3d11_element_list=[
            {"SemanticName": name, "SemanticIndex": index, "Format": fmt,
             "ByteWidth": width, "ExtractSlot": "vb1",
             "ExtractTechnique": "trianglelist", "Category": "Texcoord"}
            for name, index, fmt, width in fields
        ],
    )
    # Deliberately nonzero padding/color must not become the missing UV layer.
    raw = struct.pack("<4f4b", 0.25, 0.5, 0.125, 0.75, 127, -64, 32, -1)
    raw += struct.pack("<4f4b", 0.75, 1.0, 0.5, 0.25, -32, 64, -127, 1)
    return SimpleNamespace(
        unique_str="deadbeef-6-0", d3d11_game_type=game_type,
        category_buffer_dict={
            "Texcoord": numpy.frombuffer(raw, dtype=numpy.uint8).copy(),
        },
    )


def _semantic(name="TEXCOORD", index=4, fmt="R32G32_FLOAT", width=8):
    return {"name": name, "index": index, "format": fmt, "stride": width}


def main():
    exporter = ExportEFMI.__new__(ExportEFMI)
    base_uv = numpy.asarray([[0.25, 0.5], [0.75, 1.0]], dtype=numpy.float32)
    extra_uv = numpy.asarray([[0.125, 0.75], [0.5, 0.25]], dtype=numpy.float32)
    for present in (False, True):
        submesh = _submesh(secondary_uv=present)
        before = submesh.category_buffer_dict["Texcoord"].tobytes()
        for fmt, dtype, width in (
            ("R32G32_FLOAT", numpy.float32, 8),
            ("R16G16_FLOAT", numpy.float16, 4),
            ("R32G32_FLOAT", numpy.float32, 12),
        ):
            layout = {"semantics": [
                _semantic(index=0), _semantic(fmt=fmt, width=width),
                _semantic("COLOR", 0, "R8G8B8A8_SNORM", 4),
            ]}
            category, stride, data = exporter._build_lod_slot_buffer(
                submesh, "VB1", layout
            )
            assert category == "Texcoord" and stride == 8 + width + 4
            rows = data.reshape(2, stride)
            assert rows[:, :8].tobytes() == base_uv.tobytes()
            expected = numpy.zeros(
                (2, width // numpy.dtype(dtype).itemsize), dtype=dtype
            )
            if present:
                expected[:, :2] = extra_uv
            assert rows[:, 8:8 + width].tobytes() == expected.tobytes()
            source_rows = submesh.category_buffer_dict["Texcoord"].reshape(2, 20)
            assert rows[:, -4:].tobytes() == source_rows[:, -4:].tobytes()
            assert submesh.category_buffer_dict["Texcoord"].tobytes() == before

    # Do not hide missing required data or unsupported UV formats.
    for semantic in (
        _semantic(index=0),
        _semantic("POSITION", 0, "R32G32B32_FLOAT", 12),
        _semantic("BLENDINDICES", 0, "R16G16B16A16_UINT", 8),
        _semantic("COLOR", 1, "R8G8B8A8_SNORM", 4),
        _semantic(fmt="R32G32_UINT"),
        _semantic(fmt="R32G32B32_FLOAT", width=12),
    ):
        try:
            exporter._build_lod_slot_buffer(
                _submesh(primary_uv=False), "VB1", {"semantics": [semantic]}
            )
        except ValueError as exc:
            assert "缺少 LOD 语义" in str(exc), str(exc)
        else:
            raise AssertionError(
                "Missing required semantic did not fail: " + str(semantic)
            )

    print("MERGED_LOD_MISSING_UV_REGRESSION=PASS")


if __name__ == "__main__":
    main()
