# -*- coding: utf-8 -*-
"""Regression for bones that a component's LoD skeleton does not list.

Component A's LoD mesh no longer uses bone 2, so A's LoD remap has to read a
stand-in bone for it; component B's LoD still has bone 2.  A mod mesh drawn
by A and weighted to bone 2 must read B's slot, which holds the same bone up
close and the right bone at distance.  The distances come from the rest-pose
meshes in the workspace and in the LoD preview workspace.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy

# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LoyalTools.common.d3d11_gametype import D3D11GameType
from LoyalTools.common.efmi_merged_skeleton import (
    LOD_BONE_STAND_IN_DISTANCE,
    build_lod_missing_bone_substitution,
    compute_lod_bone_errors,
    get_component_by_id,
    read_workspace_skin,
)
from LoyalTools.ui.universal.efmi import ExportEFMI


DRAW = "aaaaaaaa-30-0"
LOD_OBJECT = "Character 1"

# Rest-pose position of every bone of the character.
BONES = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.0, 0.0),
    2: (0.0, 0.0, 1.0),
    3: (0.0, 1.0, 0.0),
    4: (1.0, 1.0, 0.0),
}


def _write_mesh(folder, unique_str, bone_ids, weight_format, index_format):
    """Two vertices on each listed bone (local ids in list order)."""
    mesh_folder = os.path.join(folder, unique_str, "TYPE_GPU-EFMI")
    os.makedirs(mesh_folder, exist_ok=True)
    positions, indices = [], []
    for local_id, bone in enumerate(bone_ids):
        for offset in (-0.01, 0.01):
            x, y, z = BONES[bone]
            positions.append((x + offset, y, z))
            indices.append((local_id, 0, 0, 0))
    count = len(positions)

    position_rows = numpy.zeros(count, dtype=[("p", "<f4", (3,)), ("n", "<u4")])
    position_rows["p"] = positions
    position_rows.tofile(os.path.join(mesh_folder, unique_str + "-Position.buf"))

    weight_dtype, weight_one = (
        ("<u2", 65535) if weight_format == "R16G16B16A16_UNORM" else ("u1", 255)
    )
    index_dtype = "<u2" if index_format == "R16G16B16A16_UINT" else "u1"
    blend_rows = numpy.zeros(
        count, dtype=[("w", weight_dtype, (4,)), ("i", index_dtype, (4,))]
    )
    blend_rows["w"][:, 0] = weight_one
    blend_rows["i"] = indices
    blend_rows.tofile(os.path.join(mesh_folder, unique_str + "-Blend.buf"))

    def element(name, fmt, width):
        return {"SemanticName": name, "SemanticIndex": 0, "Format": fmt, "ByteWidth": width}

    metadata = {
        "CategoryBufferList": [
            {
                "FileName": unique_str + "-Position.buf",
                "D3D11ElementList": [
                    element("POSITION", "R32G32B32_FLOAT", 12),
                    element("NORMAL", "R32_UINT", 4),
                ],
            },
            {
                "FileName": unique_str + "-Blend.buf",
                "D3D11ElementList": [
                    element("BLENDWEIGHTS", weight_format, blend_rows.dtype["w"].itemsize),
                    element("BLENDINDICES", index_format, blend_rows.dtype["i"].itemsize),
                ],
            },
        ]
    }
    with open(os.path.join(mesh_folder, unique_str + ".json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file)


def _lod(unique_str, vg_map, fallback=False):
    return {
        "lod_object_name": LOD_OBJECT,
        "unique_str": unique_str,
        "is_fallback": fallback,
        "vg_map": {str(key): value for key, value in vg_map.items()},
        "vb_formats": {},
    }


def _component(component_id, unique_str, vg_offset, vg_map, lod):
    return {
        "component_id": component_id,
        "unique_str": unique_str,
        "cpu_posed": False,
        "vg_offset": vg_offset,
        "vg_count": len(vg_map),
        "vg_map": {str(local): value for local, value in enumerate(vg_map)},
        "lods": [lod],
    }


def _profile():
    return {
        "max_lod_count": 1,
        "lod_preview_workspaces": {LOD_OBJECT: os.path.join("LODPreview", LOD_OBJECT)},
        "components": [
            # Bones 0, 1, 2; its LoD mesh uses only 0 and 1, so bone 2 reads
            # LoD bone 1 (the nearest one left).
            _component(0, DRAW, 0, [0, 1, 2],
                       _lod("a1a1a1a1-12-0", {0: 0, 1: 1, 2: 1})),
            # Bones 3, 4 and the shared bone 2; its LoD has all three.
            _component(1, "bbbbbbbb-30-0", 3, [3, 4, 2],
                       _lod("b1b1b1b1-18-0", {0: 1, 1: 2, 2: 0})),
            # Same mesh at LoD: identity, nothing to check.
            _component(2, "cccccccc-30-0", 6, [0, 4], _lod("cccccccc-30-0", {})),
            # Not drawn at LoD: left to the stale pool substitution.
            _component(3, "dddddddd-30-0", 8, [2], _lod("dddddddd-30-0", {}, True)),
        ],
    }


def _write_workspace(folder):
    _write_mesh(folder, DRAW, [0, 1, 2], "R16G16B16A16_UNORM", "R8G8B8A8_UINT")
    _write_mesh(folder, "bbbbbbbb-30-0", [3, 4, 2], "R16G16B16A16_UNORM", "R8G8B8A8_UINT")
    preview = os.path.join(folder, "LODPreview", LOD_OBJECT)
    _write_mesh(preview, "a1a1a1a1-12-0", [0, 1], "R8G8B8A8_UNORM", "R16G16B16A16_UINT")
    _write_mesh(preview, "b1b1b1b1-18-0", [2, 3, 4], "R8G8B8A8_UNORM", "R16G16B16A16_UINT")


def check_errors(folder, profile):
    skin = read_workspace_skin(folder, DRAW)
    assert skin is not None and skin[0].shape == (6, 3), skin
    assert read_workspace_skin(folder, "missing-1-0") is None

    errors = compute_lod_bone_errors(folder, profile)
    assert sorted(errors) == [0, 1, 2], errors
    a, b = errors[0][1], errors[1][1]
    assert a[0] < 1e-6 and a[1] < 1e-6, a
    assert abs(a[2] - 2 ** 0.5) < 1e-3, a
    assert max(b) < 1e-6, b
    assert errors[2][1] == [0.0, 0.0]

    # Without the preview meshes the remapped components cannot be checked.
    no_preview = dict(profile, lod_preview_workspaces={LOD_OBJECT: "nowhere"})
    unchecked = compute_lod_bone_errors(folder, no_preview)
    assert unchecked[0][1] == [None, None, None], unchecked
    return errors


def check_substitution(profile, errors):
    by_id = get_component_by_id(profile)
    # Bone 2 is read from component 1's slot 5; bones its own LoD has stay.
    assert build_lod_missing_bone_substitution(
        profile, [0, 1, 2], by_id[0], errors
    ) == ({2: 5}, {})

    # No other LoD has the bone: reported, not moved.
    worse = {0: errors[0], 1: {1: [0.0, 0.0, 2.0]}, 2: errors[2]}
    substitution, unresolved = build_lod_missing_bone_substitution(
        profile, [2], by_id[0], worse
    )
    assert substitution == {} and list(unresolved) == [2], unresolved

    # Unknown distances move nothing.
    unknown = {0: {1: [None, None, None]}, 1: errors[1], 2: errors[2]}
    assert build_lod_missing_bone_substitution(
        profile, [2], by_id[0], unknown
    ) == ({}, {})

    # A draw that is never visible at distance keeps its ids.
    assert build_lod_missing_bone_substitution(
        profile, [2], by_id[3], errors
    ) == ({}, {})

    # Close enough counts as the bone itself.
    near = {0: {1: [0.0, 0.0, LOD_BONE_STAND_IN_DISTANCE * 0.9]}, 1: errors[1]}
    assert build_lod_missing_bone_substitution(
        profile, [2], by_id[0], near
    ) == ({}, {})


BLEND_DTYPE = numpy.dtype([("weights", "<u2", (4,)), ("indices", "<u2", (4,))])


def _game_type():
    def element(name, fmt):
        return {
            "SemanticName": name,
            "SemanticIndex": 0,
            "Format": fmt,
            "ByteWidth": 8,
            "ExtractSlot": "vb2",
            "ExtractTechnique": "trianglelist",
            "Category": "Blend",
        }

    return D3D11GameType.from_submesh_json_dict(
        {
            "GPU-PreSkinning": True,
            "WorkGameType": "MergedLodStandInRegression",
            "CategoryDrawCategoryMap": {"Blend": "Blend"},
        },
        override_d3d11_element_list=[
            element("BLENDWEIGHTS", "R16G16B16A16_UNORM"),
            element("BLENDINDICES", "R16G16B16A16_UINT"),
        ],
    )


def check_export(profile, errors):
    rows = numpy.zeros(3, dtype=BLEND_DTYPE)
    rows["indices"] = [[0, 2, 0, 0], [2, 1, 0, 0], [1, 0, 2, 0]]
    rows["weights"] = [[40000, 25535, 0, 0], [65535, 0, 0, 0], [65535, 0, 0, 0]]
    submesh = SimpleNamespace(
        unique_str=DRAW,
        d3d11_game_type=_game_type(),
        category_buffer_dict={"Blend": rows.view(numpy.uint8).reshape(-1).copy()},
        object_export_context_list=[{
            "efmi_component_id": 0,
            "efmi_mesh_name": DRAW,
            "source_object_name": DRAW,
            "export_indices": numpy.arange(3, dtype=numpy.int32),
        }],
    )
    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.merged_skeleton_profile = profile
    exporter.submesh_model_list = [submesh]
    exporter.merged_private_vg_rewrite_stats = {}
    exporter.merged_lod_bone_errors = errors
    exporter._prepare_merged_component_blend_buffers()

    result = submesh.category_buffer_dict["Blend"].view(BLEND_DTYPE).reshape(-1)
    # Weighted bone 2 reads slot 5; the zero-weight channel keeps its id.
    assert result["indices"].tolist() == [[0, 5, 0, 0], [5, 1, 0, 0], [1, 0, 2, 0]], (
        result["indices"].tolist()
    )
    stats = exporter.merged_private_vg_rewrite_stats[DRAW]
    assert stats["lod_stand_in_substituted_channel_count"] == 2, stats
    assert stats["stale_pool_substituted_channel_count"] == 0, stats


def main():
    with tempfile.TemporaryDirectory() as folder:
        profile = _profile()
        _write_workspace(folder)
        errors = check_errors(folder, profile)
    check_substitution(profile, errors)
    check_export(profile, errors)
    print("MERGED_LOD_STAND_IN_REGRESSION=PASS")


if __name__ == "__main__":
    main()
