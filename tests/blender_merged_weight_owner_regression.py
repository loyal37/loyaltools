# -*- coding: utf-8 -*-
"""Regression for merged-skeleton weight ownership, stale pools and LoD remaps.

The profile mirrors the Ardelia case: component 2 deduplicates all of its
bones onto component 1's pool, and component 1 has no independent LoD.  Its
geometry must read component 2's private pool at distance whether it is
renamed into another draw (owner marker) or joined into another object (stale
pool substitution), and component 2 must keep its LoD remap even when the
blueprint has no mesh for it.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy

# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LoyalTools.common.d3d11_gametype import D3D11GameType
from LoyalTools.common.efmi_merged_skeleton import (
    audit_exported_global_vgs,
    build_stale_pool_substitution,
    component_draw_levels,
    find_component_owning_global_vg,
    format_id_ranges,
    resolve_weight_owner_component,
)
from LoyalTools.ui.universal.efmi import ExportEFMI


DRAW = "aaaaaaaa-30-0"
JOINED = "cccccccc-30-0"


def _component(component_id, unique_str, vg_offset, vg_map, lod, cpu=False):
    return {
        "component_id": component_id,
        "unique_str": unique_str,
        "cpu_posed": cpu,
        "vg_offset": vg_offset,
        "vg_count": len(vg_map),
        "vg_map": {str(local): value for local, value in enumerate(vg_map)},
        "lods": [lod],
    }


def _profile():
    independent = {"is_fallback": False, "vg_map": {}, "vb_formats": {}}
    return {
        "max_lod_count": 1,
        "components": [
            # Drawing component, supplies its own pool 0-3.
            _component(0, DRAW, 0, [0, 1, 2, 3], dict(independent)),
            # No independent LoD: pool 4-6 goes stale at distance.
            _component(1, "bbbbbbbb-30-0", 4, [4, 5, 6], {"is_fallback": True}),
            # Same bones as component 1, refreshed in its own pool 7-9.
            _component(2, JOINED, 7, [4, 5, 6], {
                "is_fallback": False,
                "vg_map": {"0": 1, "1": 0, "2": 2},
                "vb_formats": {},
            }),
            # Supplies only one of those bones.
            _component(3, "dddddddd-30-0", 10, [4, 11], dict(independent)),
            # CPU posed components are never attached to the merged skeleton.
            _component(4, "eeeeeeee-30-0", 12, [12, 13], dict(independent), cpu=True),
        ],
    }


def _game_type():
    return D3D11GameType.from_submesh_json_dict(
        {
            "GPU-PreSkinning": True,
            "WorkGameType": "MergedWeightOwnerRegression",
            "CategoryDrawCategoryMap": {"Blend": "Blend"},
        },
        override_d3d11_element_list=[
            {
                "SemanticName": "BLENDWEIGHTS",
                "SemanticIndex": 0,
                "Format": "R16G16B16A16_UNORM",
                "ByteWidth": 8,
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


BLEND_DTYPE = numpy.dtype([("weights", "<u2", (4,)), ("indices", "<u2", (4,))])


def _submesh(indices, weights, contexts):
    rows = numpy.zeros(len(indices), dtype=BLEND_DTYPE)
    rows["indices"] = numpy.asarray(indices, dtype=numpy.uint16)
    rows["weights"] = numpy.asarray(weights, dtype=numpy.uint16)
    return SimpleNamespace(
        unique_str=DRAW,
        d3d11_game_type=_game_type(),
        category_buffer_dict={"Blend": rows.view(numpy.uint8).reshape(-1).copy()},
        object_export_context_list=contexts,
    )


def _context(component_id, name, rows):
    return {
        "efmi_component_id": component_id,
        "efmi_mesh_name": name + ".001",
        "source_object_name": name,
        "export_indices": numpy.asarray(rows, dtype=numpy.int32),
    }


def _export_blend(profile, submesh):
    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.merged_skeleton_profile = profile
    exporter.submesh_model_list = [submesh]
    exporter.merged_private_vg_rewrite_stats = {}
    exporter._prepare_merged_component_blend_buffers()
    rows = submesh.category_buffer_dict["Blend"].view(BLEND_DTYPE).reshape(-1)
    return exporter, rows["indices"].tolist()


def check_profile_helpers():
    profile = _profile()
    by_id = {c["component_id"]: c for c in profile["components"]}

    component, source = resolve_weight_owner_component(
        profile, component_id=2, unique_str_candidates=(DRAW,)
    )
    assert (component["component_id"], source) == (2, "marker")
    component, source = resolve_weight_owner_component(
        profile, component_id="x", unique_str_candidates=("", JOINED)
    )
    assert (component["component_id"], source) == (2, "unique_str")
    component, _ = resolve_weight_owner_component(
        profile, component_id=99, unique_str_candidates=(DRAW,)
    )
    assert component["component_id"] == 0
    assert resolve_weight_owner_component(profile) == (None, None)

    owners = [find_component_owning_global_vg(profile, vg) for vg in (3, 4, 9, 14)]
    assert [c and c["component_id"] for c in owners] == [0, 1, 2, None]
    assert format_id_ranges([9, 5, 4, 6, 1]) == "1, 4-6, 9"

    assert component_draw_levels(profile, by_id[0]) == {1}
    assert component_draw_levels(profile, by_id[1]) == set()
    assert component_draw_levels(profile, by_id[4]) == set()
    assert component_draw_levels(profile, None) == {1}

    # Stale ids move as a group to the component supplying most of them.
    assert build_stale_pool_substitution(
        profile, [0, 4, 5, 6], drawing_component=by_id[0]
    ) == {4: 7, 5: 8, 6: 9}
    # A draw that is never visible at distance keeps its ids.
    assert build_stale_pool_substitution(
        profile, [4, 5, 6], drawing_component=by_id[1]
    ) == {}
    # Coverage ties prefer the drawing component.
    assert build_stale_pool_substitution(
        profile, [4], drawing_component=by_id[3]
    ) == {4: 10}
    # CPU pools have no refreshed equivalent here and stay for the audit.
    assert build_stale_pool_substitution(
        profile, [12], drawing_component=by_id[0]
    ) == {}
    partial = copy.deepcopy(profile)
    partial["components"][2]["lods"][0]["is_fallback"] = True
    assert build_stale_pool_substitution(
        partial, [4, 5, 6], drawing_component=partial["components"][0]
    ) == {4: 10}

    warnings = audit_exported_global_vgs(
        profile, [0, 4, 12, 14], drawing_component_id=0, label=DRAW
    )
    assert len(warnings) == 3, warnings
    assert "14" in warnings[0] and "Component 1" in warnings[1]
    assert "Component 4" in warnings[2]
    assert audit_exported_global_vgs(profile, [0, 7, 8, 9], drawing_component_id=0) == []
    assert audit_exported_global_vgs(profile, [0, 4, 5], drawing_component_id=1) == []


def check_joined_component():
    # The joined object keeps only the drawing component's marker, so the
    # joined part must be moved by stale pool substitution. Zero-weight
    # padding is left alone.
    submesh = _submesh(
        indices=[[0, 1, 0, 0], [4, 5, 6, 0], [0, 4, 0, 0]],
        weights=[[65535, 0, 0, 0], [30000, 20000, 15535, 0], [65535, 0, 0, 0]],
        contexts=[_context(0, DRAW, [0, 1, 2])],
    )
    exporter, indices = _export_blend(_profile(), submesh)
    assert indices == [[0, 1, 0, 0], [7, 8, 9, 0], [0, 4, 0, 0]], indices
    stats = exporter.merged_private_vg_rewrite_stats[DRAW]
    assert stats["stale_pool_substituted_channel_count"] == 3
    assert stats["changed_channel_count"] == 3


def check_renamed_component():
    # A renamed object keeps its own marker and is rewritten by its owner.
    submesh = _submesh(
        indices=[[0, 1, 0, 0], [4, 5, 6, 0]],
        weights=[[65535, 0, 0, 0], [30000, 20000, 15535, 0]],
        contexts=[
            _context(0, DRAW, [0]),
            _context(2, DRAW + "." + JOINED, [1]),
        ],
    )
    exporter, indices = _export_blend(_profile(), submesh)
    assert indices == [[0, 1, 0, 0], [7, 8, 9, 0]], indices
    stats = exporter.merged_private_vg_rewrite_stats[DRAW]
    assert stats["segment_count"] == 2
    assert sorted(stats["rewrite_map_by_component"]) == [0, 2]
    assert stats["stale_pool_substituted_channel_count"] == 0


def check_bone_only_component_keeps_remap():
    # Component 2 has no mesh in the blueprint but still captures bones at
    # LoD 1, so EFMI's bone importer needs its remap.
    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.merged_skeleton_profile = _profile()
    exporter.submesh_model_list = [
        _submesh([[0, 0, 0, 0]], [[65535, 0, 0, 0]], [])
    ]
    exporter.merged_lod_variant_buffers = {}
    exporter.merged_lod_blend_remaps = {}
    exporter._build_merged_lod_export_buffers()
    assert sorted(exporter.merged_lod_blend_remaps) == [(2, 1)]
    assert exporter.merged_lod_blend_remaps[(2, 1)]["data"].tolist() == [1, 0, 2]
    assert exporter.merged_lod_variant_buffers == {}


def main():
    check_profile_helpers()
    check_joined_component()
    check_renamed_component()
    check_bone_only_component_keeps_remap()
    print("MERGED_WEIGHT_OWNER_REGRESSION=PASS")


if __name__ == "__main__":
    main()
