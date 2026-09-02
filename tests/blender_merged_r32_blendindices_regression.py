# -*- coding: utf-8 -*-
"""Regression for R32 source BLENDINDICES in EFMI Merged Skeleton export."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy

# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LoyalTools.common.d3d11_gametype import D3D11GameType
from LoyalTools.common.obj_buffer_helper import ObjBufferHelper
from LoyalTools.common.submesh_model import SubMeshModel
from LoyalTools.ui.universal.efmi import ExportEFMI
from LoyalTools.utils.ssmt_error_utils import Fatal


def _r32_source_game_type() -> D3D11GameType:
    return D3D11GameType.from_submesh_json_dict(
        {
            "GPU-PreSkinning": True,
            "WorkGameType": "R32BlendIndicesRegression",
            "CategoryDrawCategoryMap": {"Blend": "Blend"},
        },
        override_d3d11_element_list=[
            {
                "SemanticName": "BLENDWEIGHTS",
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
                "Format": "R32G32B32A32_UINT",
                "ByteWidth": 16,
                "ExtractSlot": "vb2",
                "ExtractTechnique": "trianglelist",
                "Category": "Blend",
            },
        ],
    )


def _lod_source_game_type() -> D3D11GameType:
    source = D3D11GameType.from_submesh_json_dict(
        {
            "GPU-PreSkinning": True,
            "WorkGameType": "R32BlendIndicesLODRegression",
            "CategoryDrawCategoryMap": {
                "Position": "Position",
                "Blend": "Blend",
            },
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
                "SemanticName": "NORMAL",
                "SemanticIndex": 0,
                "Format": "R32G32B32_FLOAT",
                "ByteWidth": 12,
                "ExtractSlot": "vb0",
                "ExtractTechnique": "trianglelist",
                "Category": "Position",
            },
            {
                "SemanticName": "BLENDWEIGHTS",
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
                "Format": "R32G32B32A32_UINT",
                "ByteWidth": 16,
                "ExtractSlot": "vb2",
                "ExtractTechnique": "trianglelist",
                "Category": "Blend",
            },
        ],
    )
    return SubMeshModel._build_efmi_merged_skeleton_game_type(source)


def _check_lod_layout_conversion() -> None:
    game_type = _lod_source_game_type()
    position_rows = numpy.zeros(
        2,
        dtype=numpy.dtype([("position", "<f4", (3,)), ("normal", "<f4", (3,))]),
    )
    position_rows["position"] = numpy.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=numpy.float32
    )
    blend_rows = numpy.zeros(
        2,
        dtype=numpy.dtype([("weights", "<f4", (4,)), ("indices", "<u2", (4,))]),
    )
    blend_rows["weights"] = numpy.asarray(
        [[0.5, 0.25, 0.25, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=numpy.float32
    )
    blend_rows["indices"] = numpy.asarray(
        [[290, 405, 2, 0], [405, 290, 0, 0]], dtype=numpy.uint16
    )
    packed_tbn = numpy.asarray([[0x41234567], [0x7ABCDEF0]], dtype=numpy.uint32)
    submesh = SimpleNamespace(
        unique_str="a4bb34f9-94791-0",
        d3d11_game_type=game_type,
        category_buffer_dict={
            "Position": position_rows.view(numpy.uint8).reshape(-1).copy(),
            "Blend": blend_rows.view(numpy.uint8).reshape(-1).copy(),
        },
        efmi_packed_tbn=packed_tbn,
    )
    exporter = ExportEFMI.__new__(ExportEFMI)

    category, stride, data = exporter._build_lod_slot_buffer(
        submesh,
        "VB0",
        {
            "semantics": [
                {
                    "name": "POSITION",
                    "index": 0,
                    "format": "R32G32B32_FLOAT",
                    "stride": 12,
                },
                {
                    "name": "ENCODEDDATA",
                    "index": 0,
                    "format": "R32_UINT",
                    "stride": 4,
                },
            ]
        },
    )
    assert category == "Position"
    assert stride == 16
    position_lod_rows = data.reshape(2, stride)
    assert position_lod_rows[:, :12].view(numpy.float32).reshape(2, 3).tolist() == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    assert position_lod_rows[:, 12:16].copy().view(numpy.uint32).reshape(-1).tolist() == [
        0x41234567,
        0x7ABCDEF0,
    ]

    category, stride, data = exporter._build_lod_slot_buffer(
        submesh,
        "VB2",
        {
            "semantics": [
                {
                    "name": "BLENDWEIGHTS",
                    "index": 0,
                    "format": "R16G16B16A16_UNORM",
                    "stride": 8,
                },
                {
                    "name": "BLENDINDICES",
                    "index": 0,
                    "format": "R8G8B8A8_UINT",
                    "stride": 4,
                },
            ]
        },
    )
    assert category == "Blend"
    assert stride == 16
    blend_lod_dtype = numpy.dtype(
        [("weights", "<u2", (4,)), ("indices", "<u2", (4,))]
    )
    blend_lod_rows = data.view(blend_lod_dtype).reshape(-1)
    assert blend_lod_rows["indices"].tolist() == [
        [290, 405, 2, 0],
        [405, 290, 0, 0],
    ]


def main() -> None:
    source = _r32_source_game_type()
    normalized = SubMeshModel._build_efmi_merged_skeleton_game_type(source)

    source_indices = source.ElementNameD3D11ElementDict["BLENDINDICES"]
    output_indices = normalized.ElementNameD3D11ElementDict["BLENDINDICES"]
    assert source_indices.Format == "R32G32B32A32_UINT"
    assert source_indices.ByteWidth == 16
    assert source.CategoryStrideDict["Blend"] == 32
    assert output_indices.Format == "R16G16B16A16_UINT"
    assert output_indices.ByteWidth == 8
    assert normalized.CategoryStrideDict["Blend"] == 24

    values = numpy.asarray([[290, 405, 2, 0]], dtype=numpy.uint32)
    parsed = ObjBufferHelper._parse_blendindices(
        {0: values},
        SimpleNamespace(SemanticIndex=0, Format="R16G16B16A16_UINT"),
    )
    assert parsed.dtype == numpy.uint16
    assert parsed.tolist() == values.tolist()

    overflow = numpy.asarray([[65536, 0, 0, 0]], dtype=numpy.uint32)
    try:
        ObjBufferHelper._parse_blendindices(
            {0: overflow},
            SimpleNamespace(SemanticIndex=0, Format="R16G16B16A16_UINT"),
        )
    except Fatal as exc:
        assert "0-65535" in str(exc)
    else:
        raise AssertionError("R16 BLENDINDICES overflow was not rejected")

    _check_lod_layout_conversion()

    print("MERGED_R32_BLENDINDICES_REGRESSION=PASS")


if __name__ == "__main__":
    main()
