# -*- coding: utf-8 -*-
"""Focused Blender regression for EFMI merged-skeleton LoD buffers."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy

from LoyalTools.common.d3d11_gametype import D3D11GameType
from LoyalTools.ui.universal.efmi import ExportEFMI


def _game_type():
    return D3D11GameType.from_submesh_json_dict(
        {
            "GPU-PreSkinning": True,
            "WorkGameType": "LODRegression",
            "CategoryDrawCategoryMap": {},
        },
        override_d3d11_element_list=[
            {
                "SemanticName": "POSITION",
                "SemanticIndex": 0,
                "Format": "R32G32B32_FLOAT",
                "ByteWidth": 12,
                "ExtractSlot": "vb0",
                "ExtractTechnique": "trianglelist",
                "Category": "Position",
            },
            {
                "SemanticName": "TEXCOORD",
                "SemanticIndex": 0,
                "Format": "R32G32_FLOAT",
                "ByteWidth": 8,
                "ExtractSlot": "vb1",
                "ExtractTechnique": "trianglelist",
                "Category": "Texcoord",
            },
            {
                "SemanticName": "COLOR",
                "SemanticIndex": 0,
                "Format": "R8G8B8A8_UNORM",
                "ByteWidth": 4,
                "ExtractSlot": "vb1",
                "ExtractTechnique": "trianglelist",
                "Category": "Texcoord",
            },
            {
                "SemanticName": "BLENDWEIGHT",
                "SemanticIndex": 0,
                "Format": "R32G32B32A32_FLOAT",
                "ByteWidth": 16,
                "ExtractSlot": "vb2",
                "ExtractTechnique": "trianglelist",
                "Category": "Blend",
            },
            {
                "SemanticName": "BLENDINDICES",
                "SemanticIndex": 0,
                "Format": "R16G16B16A16_UINT",
                "ByteWidth": 8,
                "ExtractSlot": "vb2",
                "ExtractTechnique": "trianglelist",
                "Category": "Blend",
            },
        ],
    )


def main():
    unique_str = "deadbeef-6-0"
    texcoord_bytes = b"".join(
        (
            struct.pack("<2f4B", 0.25, 0.50, 1, 2, 3, 4),
            struct.pack("<2f4B", 0.75, 1.00, 5, 6, 7, 8),
        )
    )
    submesh = SimpleNamespace(
        unique_str=unique_str,
        d3d11_game_type=_game_type(),
        category_buffer_dict={
            "Position": numpy.zeros(24, dtype=numpy.uint8),
            "Texcoord": numpy.frombuffer(texcoord_bytes, dtype=numpy.uint8).copy(),
            "Blend": numpy.zeros(48, dtype=numpy.uint8),
        },
    )
    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.submesh_model_list = [submesh]
    exporter.merged_lod_variant_buffers = {}
    exporter.merged_lod_blend_remaps = {}
    exporter.merged_skeleton_profile = {
        "components": [
            {
                "component_id": 0,
                "unique_str": unique_str,
                "cpu_posed": False,
                "vg_count": 3,
                "lods": [
                    {
                        "vg_map": {"0": 2, "1": 0, "2": 1},
                        "vb_formats": {
                            "VB1": {
                                "semantics": [
                                    {
                                        "name": "TEXCOORD",
                                        "index": 0,
                                        "format": "R32G32_FLOAT",
                                        "stride": 8,
                                    }
                                ]
                            }
                        },
                    }
                ],
            }
        ]
    }

    exporter._build_merged_lod_export_buffers()
    variant = exporter.merged_lod_variant_buffers[(unique_str, 1, "VB1")]
    assert variant["stride"] == 8
    assert variant["filename"] == unique_str + "-Texcoord-LOD1.buf"
    assert variant["data"].tobytes() == (
        struct.pack("<2f", 0.25, 0.50) + struct.pack("<2f", 0.75, 1.00)
    )
    remap = exporter.merged_lod_blend_remaps[(0, 1)]
    assert remap["data"].tolist() == [2, 0, 1]

    lines = []
    exporter._append_merged_buffer_bindings(lines, submesh)
    assert "if $lod_level == 0" in lines
    assert "elif $lod_level == 1" in lines
    assert any(
        line == "    vb1 = " + variant["resource_name"] for line in lines
    )
    print("MERGED_LOD_BUFFER_REGRESSION=PASS")


if __name__ == "__main__":
    main()
