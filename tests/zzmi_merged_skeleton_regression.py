# -*- coding: utf-8 -*-
"""Regression: Zenless Zone Zero merged skeleton extraction and runtime data.

Runs with plain ``py`` (no Blender):

* merged extraction from a synthetic dump: character selection, roles,
  pool layout, anchors, canonical ids and the SubmeshJson metadata;
* the palette reference reproduces every bone in every component's space;
* the export helpers (private remap, dependencies, runtime tables);
* profile validation rejects broken contracts.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import zzmi_merged_synthetic_dump as synthetic  # noqa: E402


def load(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = TESTS_DIR.parent
zms = load(ROOT / "common" / "zzmi_merged_skeleton.py", "zzmi_merged_skeleton_under_test")
extractor_module = load(ROOT / "extract" / "zzmi_dump_extractor.py", "zzmi_dump_extractor_under_test")


def check_extraction(temp_folder: str):
    dump_folder = os.path.join(temp_folder, "FrameAnalysis-merged")
    workspace = os.path.join(temp_folder, "workspace")
    os.makedirs(dump_folder)
    source = synthetic.build_dump(dump_folder)
    parts = source["components"]

    result = extractor_module.ZZMIDumpExtractor(dump_folder).extract_merged_skeleton(workspace, copy_textures=False)
    profile = zms.load_profile(workspace)
    by_ib = zms.get_component_by_ib(profile)
    ib = {name: data["spec"]["ib"] for name, data in parts.items()}

    # Character: body (root, most vertices first in pool), hair, accessory merged; face kept for
    # textures only; the bangboo shares no bone and is not extracted at all.
    assert [component["ib_hash"] for component in profile["components"]] == [
        ib["body"], ib["hair"], ib["accessory"], ib["face"],
    ], [component["ib_hash"] for component in profile["components"]]
    assert [component["role"] for component in profile["components"]] == ["merged", "merged", "merged", "shape_keyed"]
    assert ib["bangboo"] not in by_ib
    assert any(ib["bangboo"] in warning for warning in result.warnings)
    assert any(ib["face"] in warning and "形态键" in warning for warning in result.warnings)
    assert sorted(result.unique_strs) == sorted(parts[name]["unique_str"] for name in ("body", "hair", "accessory", "face"))

    body, hair, accessory, face = (by_ib[ib[name]] for name in ("body", "hair", "accessory", "face"))
    assert (body["vg_offset"], body["vg_count"]) == (0, 8)
    assert (hair["vg_offset"], hair["vg_count"]) == (8, 5)
    assert (accessory["vg_offset"], accessory["vg_count"]) == (13, 2)
    assert profile["bones_count"] == 15
    assert profile["root_component_id"] == body["component_id"]
    assert accessory["indices_per_vertex"] == 1 and accessory["pose_vs"] == synthetic.VS_RIGID
    assert face["vg_count"] == 0 and face["vg_map"] == {}

    # Anchors: hair hangs off the body through the head (body bone 6 = hair bone 0),
    # the accessory off the hair through hair bone 4 (world bone 11).
    assert hair["parent_component_id"] == body["component_id"]
    assert (hair["anchor_parent_slot"], hair["anchor_self_slot"]) == (6, 8)
    assert accessory["parent_component_id"] == hair["component_id"]
    assert (accessory["anchor_parent_slot"], accessory["anchor_self_slot"]) == (12, 13)

    # Canonical ids: shared bones collapse onto the copy uploaded first.
    assert hair["vg_map"]["0"] == 6, hair["vg_map"]
    assert [hair["vg_map"][str(local)] for local in range(1, 5)] == [9, 10, 11, 12]
    assert accessory["vg_map"] == {"0": 12, "1": 14}
    assert body["vg_map"] == {str(local): local for local in range(8)}

    # SubmeshJson metadata of every part.
    for name in ("body", "hair", "accessory", "face"):
        unique_str = parts[name]["unique_str"]
        json_path = os.path.join(workspace, unique_str, "TYPE_GPU-ZZMI", unique_str + ".json")
        with open(json_path, encoding="utf-8") as handle:
            metadata = zms.parse_submesh_metadata(json.load(handle))
        component = by_ib[ib[name]]
        assert metadata["component_id"] == component["component_id"] and metadata["role"] == component["role"]

    # The palette of every merged component holds every pool bone in its own root space.
    pool = numpy.concatenate([parts[name]["palette"] for name in ("body", "hair", "accessory")]).astype(numpy.float64)
    world = source["world_bones"]
    for name in ("body", "hair", "accessory"):
        component = by_ib[ib[name]]
        palette = zms.reference_palette(profile, pool, component["component_id"])
        root_inverse = numpy.linalg.inv(parts[name]["root"])
        for other in ("body", "hair", "accessory"):
            other_component = by_ib[ib[other]]
            for local_id, bone in enumerate(parts[other]["spec"]["bones"]):
                truth = (root_inverse @ world[bone])[:3]
                slot = other_component["vg_offset"] + local_id
                error = float(numpy.abs(palette[slot] - truth).max())
                assert error < 1e-4, (name, other, local_id, error)
    return profile, by_ib, ib


def check_export_helpers(profile, by_ib, ib):
    body, hair, accessory = by_ib[ib["body"]], by_ib[ib["hair"]], by_ib[ib["accessory"]]
    # The hair reads the head from its own copy, the accessory its shared bone from its own copy.
    assert zms.build_private_remap(hair)[6] == 8
    assert zms.build_private_remap(accessory)[12] == 13
    assert zms.build_private_remap(body)[6] == 6

    assert zms.anchor_chain_ids(profile, accessory["component_id"]) == [
        accessory["component_id"], hair["component_id"], body["component_id"],
    ]
    # The body drawing a hair bone needs the hair palette and nothing else.
    assert zms.palette_dependencies(profile, body["component_id"], {10}) == {hair["component_id"]}
    # The accessory drawing a body bone needs its chain (hair, body).
    assert zms.palette_dependencies(profile, accessory["component_id"], {2}) == {
        hair["component_id"], body["component_id"],
    }
    try:
        zms.slot_owner_ids(profile, {15})
    except zms.ZZMIMergedSkeletonError:
        pass
    else:
        raise AssertionError("slots outside the pool must be rejected")

    # The skinning layout is recorded per component.
    assert body["layout_signature"].startswith("Position=POSITION:R32G32B32_FLOAT:12,"), body["layout_signature"]
    assert "Blend=BLENDINDICES:R32_UINT:4" in accessory["layout_signature"], accessory["layout_signature"]

    table = zms.runtime_component_table(profile)
    assert table == [0, 8, 0, 1, 0, 0, 0, 0, 8, 5, 0, 1, 6, 8, 0, 0, 13, 2, 1, 1, 12, 13, 0, 0], table
    assert zms.runtime_params(profile, accessory["component_id"]) == [2, 3, 15, 0]

    # Uniform one-frame delay: the face gets a pool range after the merged bones.
    face = next(component for component in profile["components"] if component["role"] == zms.ROLE_SHAPE_KEYED)
    face_bones = face["palette_bone_count"]
    layout = zms.delayed_pool_layout(profile, [face["component_id"]])
    assert layout == [{"component_id": face["component_id"], "offset": 15, "count": face_bones, "table_index": 3}], layout
    assert zms.runtime_component_table(profile, layout) == table + [15, face_bones, 3, 1, 0, 0, 0, 0]
    assert zms.runtime_params(profile, face["component_id"], layout) == [3, 3, 15, 0]
    assert zms.runtime_params(profile, accessory["component_id"], layout) == [2, 3, 15, 0]
    try:
        zms.delayed_pool_layout(profile, [body["component_id"]])
    except zms.ZZMIMergedSkeletonError:
        pass
    else:
        raise AssertionError("merged components already have a pool range")


def check_validation(profile):
    def rejects(mutate, label):
        broken = copy.deepcopy(profile)
        mutate(broken)
        try:
            zms.validate_profile(broken)
        except zms.ZZMIMergedSkeletonError:
            return
        raise AssertionError("validation accepted: " + label)

    rejects(lambda p: p["components"][1].__setitem__("vg_offset", 9), "pool gap")
    rejects(lambda p: p["components"][0].__setitem__("parent_component_id", 1), "no root / cycle")
    rejects(lambda p: p["components"][1].__setitem__("anchor_self_slot", 2), "anchor outside own range")
    rejects(lambda p: p["components"][1]["vg_map"].pop("4"), "incomplete vg_map")
    rejects(lambda p: p["components"][3].__setitem__("role", "cpu"), "unknown role")
    rejects(lambda p: p["components"][2]["unique_strs"].append(p["components"][1]["unique_strs"][0]), "foreign part")


def check_planner_edge_cases():
    """Coincident bones never become anchors; a component reachable only through them stays standalone."""
    rng = numpy.random.default_rng(3)
    world = [synthetic._rigid(rng) for _ in range(6)]
    roots = [synthetic._rigid(rng) for _ in range(3)]

    def palette(root, bones):
        return numpy.stack([(numpy.linalg.inv(root) @ world[bone])[:3] for bone in bones]).astype(numpy.float32)

    # c2 uses bone 4 twice in its palette: both copies match the body's bone 4, so no pair is unique.
    candidates = [
        {"ib_hash": "c0000001", "bones": [0, 1, 2, 3, 4], "root": roots[0], "count": 300},
        {"ib_hash": "c0000002", "bones": [3, 5], "root": roots[1], "count": 200},
        {"ib_hash": "c0000003", "bones": [4, 4], "root": roots[2], "count": 100},
    ]
    planned = []
    for order, item in enumerate(candidates):
        planned.append({
            "ib_hash": item["ib_hash"], "unique_strs": [item["ib_hash"] + "-3-0"], "blend_vb": "", "pose_vs": "a" * 16,
            "pose_order": order + 1, "vertex_count": item["count"], "indices_per_vertex": 4,
            "palette": palette(item["root"], item["bones"]), "object_matrix": item["root"], "shape_keyed": False,
            "weighted_counts": numpy.ones(len(item["bones"]), dtype=numpy.int64),
        })
    profile, warnings = zms.plan_merged_skeleton(planned)
    roles = {component["ib_hash"]: component["role"] for component in profile["components"]}
    assert roles == {"c0000001": "merged", "c0000002": "merged", "c0000003": "standalone"}, roles
    assert any("c0000003" in warning for warning in warnings)
    zms.validate_profile(profile)


def main():
    with tempfile.TemporaryDirectory(prefix="loyal_zzmi_merged_") as temp_folder:
        profile, by_ib, ib = check_extraction(temp_folder)
        check_export_helpers(profile, by_ib, ib)
        check_validation(profile)
    check_planner_edge_cases()
    print("ZZMI_MERGED_SKELETON_REGRESSION=PASS")


if __name__ == "__main__":
    main()
