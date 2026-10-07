# -*- coding: utf-8 -*-
"""Pure-Python regression for applying a LoD mapping from JSON.

A LoD mapping only depends on the full-detail and LoD meshes, so it can be
applied again without either capture: to a profile that was extracted again in
the same workspace (from the copy kept before the extraction), or to another
workspace of the same character.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "_loyaltools_lod_json_restore"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
profile_module = importlib.import_module(PACKAGE + ".common.efmi_merged_skeleton")
extractor_module = importlib.import_module(PACKAGE + ".extract.dump_workspace_extractor")
mapper_module = importlib.import_module(PACKAGE + ".extract.merged_skeleton_lod_mapper")

LOD_OBJECT = "Character 400"
UV_LAYOUT = {"VB1": {"semantics": [
    {"name": "TEXCOORD", "index": 0, "format": "R16G16_FLOAT", "stride": 4},
]}}


def _draw(unique_str):
    ib_hash, index_count, first_index = unique_str.split("-")
    return ib_hash, int(index_count), int(first_index)


def _component(component_id, unique_str, vg_offset, vg_count, lods=(), **extra):
    ib_hash, index_count, first_index = _draw(unique_str)
    component = {
        "component_id": component_id,
        "unique_str": unique_str,
        "ib_hash": ib_hash,
        "index_count": index_count,
        "first_index": first_index,
        "vertex_count": 100 + component_id,
        "cpu_posed": False,
        "vg_offset": vg_offset,
        "vg_count": vg_count,
        "vg_map": {str(local): vg_offset + local for local in range(vg_count)},
        "lods": list(lods),
    }
    component.update(extra)
    return component


def _lod(unique_str, lod_object_name=LOD_OBJECT, **extra):
    ib_hash, index_count, first_index = _draw(unique_str)
    lod = {
        "lod_object_name": lod_object_name,
        "ib_hash": ib_hash,
        "vb0_hash": "aabbccdd",
        "vertex_offset": 5,
        "vertex_count": 50,
        "index_offset": 3,
        "index_count": index_count,
        "first_index": first_index,
        "unique_str": unique_str,
        "is_fallback": False,
        "vg_map": {},
        "vb_formats": {},
    }
    lod.update(extra)
    return lod


def _profile(components, **top):
    profile = {
        "format_version": 1,
        "mode": profile_module.PROFILE_MODE,
        "object_name": "Character 1000",
        "components": components,
    }
    profile.update(top)
    return profile


def _mapped_source():
    """The profile as it was after the capture based LoD mapping."""
    return _profile(
        [
            _component(0, "aaaaaaaa-300-0", 0, 3, [_lod(
                "11111111-120-0", vg_map={"0": 1, "1": 0, "2": 2},
                vb_formats=UV_LAYOUT, first_vertex=7,
            )]),
            # No LoD mesh of its own: the entry mirrors the main draw.
            _component(1, "bbbbbbbb-90-0", 3, 2, [_lod(
                "bbbbbbbb-90-0", is_fallback=True, first_vertex=0,
            )], first_vertex=0),
            _component(2, "cccccccc-60-0", 5, 4, [_lod("33333333-30-0")]),
            # The LoD reuses the full-detail draw itself.
            _component(3, "dddddddd-45-12", 9, 2, [_lod(
                "dddddddd-45-12", vg_map={"0": 0, "1": 1},
            )]),
            _component(4, "eeeeeeee-21-0", 11, 1, [_lod("55555555-9-0")]),
        ],
        lod_sources={LOD_OBJECT: "D:\\FrameAnalysis-LOD"},
        last_lod_object_name=LOD_OBJECT,
    )


def _reextracted_target():
    """The same character extracted again: new ids and offsets, no LoD."""
    return _profile([
        _component(0, "bbbbbbbb-90-0", 0, 2, first_vertex=4),
        _component(1, "aaaaaaaa-300-0", 2, 3),
        _component(2, "cccccccc-60-0", 5, 5),
        _component(3, "dddddddd-45-12", 10, 2),
        _component(4, "ffffffff-33-0", 12, 1),
    ])


def _normalized(profile):
    return profile_module.validate_profile(copy.deepcopy(profile))


def _entry_signature(source):
    return (
        source["ib_hash"], source["index_count"], source["first_index"],
        source.get("first_vertex", 0),
    )


def check_apply_pairs_components_by_draw():
    source = _normalized(_mapped_source())
    target = _normalized(_reextracted_target())
    reports = profile_module.apply_lod_mapping(target, source, _mapped_source())
    assert len(reports) == 1
    report = reports[0]
    assert report["applied"] and not report["legacy"]
    assert (report["matched"], report["lower_poly"]) == (2, 1)
    assert [item["status"] for item in report["statuses"]] == [
        profile_module.LOD_ENTRY_FALLBACK,
        profile_module.LOD_ENTRY_MATCHED,
        profile_module.LOD_ENTRY_INCOMPATIBLE,
        profile_module.LOD_ENTRY_MATCHED,
        profile_module.LOD_ENTRY_MISSING,
    ]
    assert "本地顶点组数量不同" in report["statuses"][2]["problem"]

    by_unique = {c["unique_str"]: c for c in target["components"]}
    source_by_unique = {c["unique_str"]: c for c in source["components"]}
    # Matched entries are the stored ones, untouched.
    for unique_str in ("aaaaaaaa-300-0", "dddddddd-45-12"):
        assert by_unique[unique_str]["lods"] == source_by_unique[unique_str]["lods"]
    assert by_unique["aaaaaaaa-300-0"]["lods"][0]["first_vertex"] == 7

    # Every other component mirrors its own full-detail draw, so export
    # creates no separate LoD entry point for it.
    for unique_str in ("bbbbbbbb-90-0", "cccccccc-60-0", "ffffffff-33-0"):
        component = by_unique[unique_str]
        lod = component["lods"][0]
        assert lod["is_fallback"] and lod["unique_str"] == unique_str
        assert lod["vg_map"] == {} and lod["vb_formats"] == {}
        assert _entry_signature(lod) == _entry_signature(component)
    assert by_unique["bbbbbbbb-90-0"]["lods"][0]["first_vertex"] == 4
    assert by_unique["bbbbbbbb-90-0"]["lods"][0]["vb0_hash"] == "aabbccdd"
    assert "first_vertex" not in by_unique["cccccccc-60-0"]["lods"][0]
    assert by_unique["cccccccc-60-0"]["lods"][0]["vb0_hash"] == ""

    applied = profile_module.validate_profile(target)
    assert applied["max_lod_count"] == 1
    assert applied["lod_object_names"] == [LOD_OBJECT]

    # Applying the same mapping again changes nothing.
    again = _normalized(applied)
    profile_module.apply_lod_mapping(again, source, _mapped_source())
    assert profile_module.validate_profile(again) == applied


def check_skinning_type_and_vertex_group_range():
    source = _normalized(_mapped_source())
    target = _reextracted_target()
    target["components"][1].update(cpu_posed=True, vg_count=0, vg_map={})
    target = _normalized(target)
    report = profile_module.apply_lod_mapping(target, source)[0]
    assert report["statuses"][1]["status"] == profile_module.LOD_ENTRY_INCOMPATIBLE

    source = _mapped_source()
    source["components"][0]["lods"][0]["vg_map"]["9"] = 0
    target = _normalized(_reextracted_target())
    report = profile_module.apply_lod_mapping(target, _normalized(source))[0]
    assert report["statuses"][1]["status"] == profile_module.LOD_ENTRY_INCOMPATIBLE
    assert "超出" in report["statuses"][1]["problem"]


def check_unrelated_mapping_changes_nothing():
    source = _normalized(_profile([
        _component(0, "99999999-12-0", 0, 1, [_lod("88888888-6-0")]),
    ]))
    target = _normalized(_reextracted_target())
    before = copy.deepcopy(target)
    reports = profile_module.apply_lod_mapping(target, source)
    assert [report["applied"] for report in reports] == [False]
    assert target == before


def check_other_lod_objects_are_kept_and_levels_added():
    source = _mapped_source()
    for component in source["components"]:
        component["lods"].append(_lod(
            component["unique_str"].replace(component["ib_hash"], "77777777"),
            lod_object_name="Character 200", vertex_count=20,
        ))
    target = _reextracted_target()
    target["components"][1]["lods"] = [
        _lod("11111111-120-0", vg_map={"0": 0}),
        _lod("66666666-12-0", lod_object_name="Character 900", vertex_count=10),
    ]
    for component in target["components"]:
        if not component["lods"]:
            component["lods"] = [
                _lod("66666666-12-0", lod_object_name="Character 900", vertex_count=10),
            ]
    target = _normalized(target)
    reports = profile_module.apply_lod_mapping(target, _normalized(source))
    assert [report["lod_object_name"] for report in reports] == [
        LOD_OBJECT, "Character 200",
    ]
    assert all(report["applied"] for report in reports)
    applied = profile_module.validate_profile(target)
    assert applied["max_lod_count"] == 3
    main = next(
        c for c in applied["components"] if c["unique_str"] == "aaaaaaaa-300-0"
    )
    assert [lod["lod_object_name"] for lod in main["lods"]] == [
        LOD_OBJECT, "Character 200", "Character 900",
    ]
    # The stale entry of the same LoD object was replaced, not duplicated.
    assert main["lods"][0]["vg_map"] == {"0": 1, "1": 0, "2": 2}


def check_entries_from_before_first_index_was_stored():
    raw = _mapped_source()
    for component in raw["components"]:
        for lod in component["lods"]:
            for key in ("first_index", "unique_str", "is_fallback"):
                del lod[key]
            lod.pop("first_vertex", None)
        component.pop("first_vertex", None)
    target = _normalized(_reextracted_target())
    report = profile_module.apply_lod_mapping(
        target, profile_module.validate_profile(copy.deepcopy(raw)), raw
    )[0]
    assert report["legacy"]
    by_unique = {c["unique_str"]: c for c in target["components"]}
    # Same index range as the component: the first index is the component's.
    reused = by_unique["dddddddd-45-12"]["lods"][0]
    assert (reused["first_index"], reused["unique_str"]) == (12, "dddddddd-45-12")
    # An independent LoD index buffer keeps the legacy default.
    independent = by_unique["aaaaaaaa-300-0"]["lods"][0]
    assert (independent["first_index"], independent["unique_str"]) == (
        0, "11111111-120-0",
    )
    assert "first_vertex" not in independent and "first_vertex" not in reused


def check_same_draw_entry_takes_the_component_base_vertex():
    def restored_lod(source_first_vertex):
        source = _mapped_source()
        if source_first_vertex is not None:
            source["components"][3]["lods"][0]["first_vertex"] = source_first_vertex
        target = _reextracted_target()
        target["components"][3]["first_vertex"] = 9
        target = _normalized(target)
        profile_module.apply_lod_mapping(target, _normalized(source), source)
        component = target["components"][3]
        return component, component["lods"][0]

    # Recorded before the base vertex existed: it is the component's own draw,
    # so export must not see a second, different entry point for it.
    component, lod = restored_lod(None)
    assert lod["first_vertex"] == 9 and not lod["is_fallback"]
    assert _entry_signature(lod) == _entry_signature(component)
    # A recorded value is kept as it is.
    assert restored_lod(3)[1]["first_vertex"] == 3


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def _read_bytes(path):
    with open(path, "rb") as file:
        return file.read()


def _make_preview(workspace, components=("11111111-120-0", "33333333-30-0")):
    preview = os.path.join(workspace, "LODPreview", LOD_OBJECT)
    _write_json(
        os.path.join(preview, "Import.json"),
        {unique_str: "GPU-EFMI" for unique_str in components},
    )
    return preview


def check_backup_keeps_the_last_mapping():
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-backup-") as workspace:
        assert profile_module.backup_lod_mapping(workspace) == []
        profile_path = profile_module.get_profile_path(workspace)
        backup_path = profile_module.get_lod_backup_path(workspace)

        profile_module.write_profile(workspace, _mapped_source())
        assert profile_module.backup_lod_mapping(workspace) == [LOD_OBJECT]
        mapped = _read_bytes(profile_path)
        assert _read_bytes(backup_path) == mapped

        # A profile without a mapping, or an unreadable one, never replaces it.
        profile_module.write_profile(workspace, _reextracted_target())
        assert profile_module.backup_lod_mapping(workspace) == []
        assert _read_bytes(backup_path) == mapped
        with open(profile_path, "w", encoding="utf-8") as file:
            file.write("{ not json")
        assert profile_module.backup_lod_mapping(workspace) == []
        assert _read_bytes(backup_path) == mapped


def _fixture_component(unique_str):
    """A capture component with one draw, like the vendored MigotoComponent."""
    ib_hash, index_count, first_index = _draw(unique_str)
    draw = extractor_module.DrawIndexedInstanced(
        bindings=[], raw_command=None, index_count=index_count,
        first_index=first_index, first_vertex=4, instance_count=1, first_instance=0,
    )
    ib = extractor_module.IndexBuffer(hash=ib_hash, pointer="0x1234")
    resources = SimpleNamespace(
        get_by_slot=lambda slot, ib=ib: ib if slot == "ib" else None
    )
    result = SimpleNamespace()
    result.raw_data = SimpleNamespace(shader_calls=[
        SimpleNamespace(draw_call=draw, model_resources=resources),
    ])
    result.mesh = SimpleNamespace(
        format=SimpleNamespace(vertex_count=3, first_vertex=0), cpu_posed=False,
    )
    result.metadata = SimpleNamespace(
        ib_hash=ib_hash, vb0_hash="deadbeef", vertex_offset=0, vertex_count=3,
        index_offset=first_index, index_count=index_count,
    )
    return result


def _extract_again(workspace):
    """Run the real merged extraction over fixture components."""
    target = _reextracted_target()["components"]
    extractor = object.__new__(extractor_module.DumpWorkspaceExtractor)
    extractor.dump_folder = "main-capture"
    extractor.get_merged_skeleton_candidates = Mock(return_value=[SimpleNamespace(
        id="Character 1000",
        components=[_fixture_component(c["unique_str"]) for c in target],
    )])
    extractor._build_merged_skeleton_vg_metadata = Mock(return_value=[
        {"vg_offset": c["vg_offset"], "vg_count": c["vg_count"], "vg_map": c["vg_map"]}
        for c in target
    ])
    extractor._build_submesh = Mock(return_value=SimpleNamespace(vertex_count=3))
    extractor._write_submesh = Mock(return_value="unused-submesh.json")
    extractor._update_workspace_root_files = Mock()
    return extractor.extract_merged_skeleton(workspace, copy_textures=False)


def check_extraction_then_restore_in_the_same_workspace():
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-restore-") as workspace:
        source = _mapped_source()
        source["lod_preview_workspaces"] = {
            LOD_OBJECT: os.path.join("LODPreview", LOD_OBJECT),
        }
        profile_module.write_profile(workspace, source)
        mapped = _read_bytes(profile_module.get_profile_path(workspace))
        preview = _make_preview(workspace)
        backup_path = profile_module.get_lod_backup_path(workspace)

        # Extraction still resets the profile, but the mapping is kept aside.
        result = _extract_again(workspace)
        assert not result.warnings
        extracted = profile_module.load_profile(workspace)
        assert not profile_module.profile_has_lod_mapping(extracted)
        assert extracted["max_lod_count"] == 0
        assert _read_bytes(backup_path) == mapped
        assert mapper_module.find_lod_mapping_source(workspace) == backup_path

        # A second extraction before restoring must not lose the kept copy.
        _extract_again(workspace)
        assert _read_bytes(backup_path) == mapped

        restored = mapper_module.restore_merged_skeleton_lod(workspace)
        assert restored.source_path == backup_path and restored.changed
        assert restored.lod_object_names == [LOD_OBJECT]
        assert restored.matched_component_counts == [2]
        assert restored.lower_poly_component_counts == [1]
        assert (restored.component_count, restored.max_lod_count) == (5, 1)
        assert len(restored.warnings) == 3 and restored.notes == []
        assert "没有独立对应的 LOD" in restored.warnings[0]
        assert "不要在其他网格上使用该组件负责的全局顶点组 0-1" in restored.warnings[0]
        assert "不一致" in restored.warnings[1] and "没有 LOD 记录" in restored.warnings[2]

        profile = profile_module.load_profile(workspace)
        assert profile["lod_object_names"] == [LOD_OBJECT]
        assert all(len(c["lods"]) == 1 for c in profile["components"])
        assert profile["lod_sources"] == {LOD_OBJECT: "D:\\FrameAnalysis-LOD"}
        # The preview meshes of this workspace are still found by relative path.
        assert profile["lod_preview_workspaces"] == {
            LOD_OBJECT: os.path.join("LODPreview", LOD_OBJECT),
        }
        assert os.path.isfile(os.path.join(
            workspace, profile["lod_preview_workspaces"][LOD_OBJECT], "Import.json",
        )) and os.path.isdir(preview)
        assert profile["last_lod_object_name"] == LOD_OBJECT
        # The main model stays the newly extracted one.
        assert profile["source_frame_dump"] == "main-capture"
        assert [c["unique_str"] for c in profile["components"]] == [
            c["unique_str"] for c in _reextracted_target()["components"]
        ]

        # Matching again finds the mapping in the profile and leaves it alone.
        profile_path = profile_module.get_profile_path(workspace)
        current = _read_bytes(profile_path)
        again = mapper_module.restore_merged_skeleton_lod(workspace)
        assert again.source_path == profile_path and not again.changed
        assert (again.matched_component_counts, again.max_lod_count) == ([2], 1)
        assert _read_bytes(profile_path) == current


def check_older_format_is_noted_in_the_log_only():
    raw = _mapped_source()
    for component in raw["components"]:
        for lod in component["lods"]:
            for key in ("first_index", "unique_str", "is_fallback"):
                del lod[key]
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-legacy-") as workspace:
        _write_json(profile_module.get_lod_backup_path(workspace), raw)
        profile_module.write_profile(workspace, _reextracted_target())
        restored = mapper_module.restore_merged_skeleton_lod(workspace)
        assert len(restored.notes) == 1 and "旧版格式" in restored.notes[0]
        assert not any("旧版格式" in warning for warning in restored.warnings)
        profile = profile_module.load_profile(workspace)
        reused = next(
            c for c in profile["components"] if c["unique_str"] == "dddddddd-45-12"
        )["lods"][0]
        assert reused["unique_str"] == "dddddddd-45-12" and reused["first_index"] == 12


def check_current_legacy_profile_is_migrated_once():
    lod = _lod("dddddddd-45-12", vg_map={"0": 0, "1": 1})
    del lod["first_index"]
    del lod["unique_str"]
    raw = _profile(
        [_component(0, "dddddddd-45-12", 0, 2, [lod], first_vertex=9)],
        lod_sources={LOD_OBJECT: "D:\\capture-no-longer-present"},
        lod_preview_workspaces={LOD_OBJECT: "LODPreview/kept-location"},
        last_lod_object_name=LOD_OBJECT,
    )
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-current-legacy-") as workspace:
        profile_path = profile_module.get_profile_path(workspace)
        # Write the original JSON, not write_profile's normalized representation.
        _write_json(profile_path, raw)
        before = _read_bytes(profile_path)
        restored = mapper_module.restore_merged_skeleton_lod(workspace)
        assert restored.source_path == profile_path and restored.changed
        assert len(restored.notes) == 1 and "旧版格式" in restored.notes[0]
        profile = profile_module.load_profile(workspace)
        component = profile["components"][0]
        reused = component["lods"][0]
        assert _entry_signature(reused) == _entry_signature(component)
        assert reused["unique_str"] == "dddddddd-45-12"
        assert reused["vg_map"] == {"0": 0, "1": 1}
        for key in ("lod_sources", "lod_preview_workspaces", "last_lod_object_name"):
            assert profile[key] == raw[key]
        migrated = _read_bytes(profile_path)
        assert migrated != before
        assert not mapper_module.restore_merged_skeleton_lod(workspace).changed
        assert _read_bytes(profile_path) == migrated


def check_current_invalid_entries_are_saved_as_fallbacks():
    raw = _mapped_source()
    # One stored fallback no longer mirrors its main draw; another entry refers
    # to a main local group outside this component's palette.
    raw["components"][1]["lods"][0].update(
        ib_hash="99999999", index_count=12, first_index=3, first_vertex=8,
        unique_str="99999999-12-3", vg_map={"0": 1}, vb_formats=UV_LAYOUT,
    )
    raw["components"][2]["lods"][0]["vg_map"] = {"9": 0}
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-current-fallback-") as workspace:
        profile_module.write_profile(workspace, raw)
        profile_path = profile_module.get_profile_path(workspace)
        restored = mapper_module.restore_merged_skeleton_lod(workspace)
        assert restored.changed and restored.matched_component_counts == [3]
        assert len(restored.warnings) == 2
        assert "超出" in restored.warnings[1]
        profile = profile_module.load_profile(workspace)
        for component_id in (1, 2):
            component = profile["components"][component_id]
            fallback = component["lods"][0]
            assert fallback["is_fallback"]
            assert _entry_signature(fallback) == _entry_signature(component)
            assert fallback["unique_str"] == component["unique_str"]
            assert fallback["vg_map"] == {} and fallback["vb_formats"] == {}
        repaired = _read_bytes(profile_path)
        again = mapper_module.restore_merged_skeleton_lod(workspace)
        assert not again.changed and _read_bytes(profile_path) == repaired


def check_current_equal_size_lod_order_does_not_trigger_a_write():
    other_lod = "Character 200"
    lods = [
        _lod("11111111-120-0", lod_object_name=LOD_OBJECT, vg_map={"0": 0}),
        _lod("22222222-120-0", lod_object_name=other_lod, vg_map={"0": 0}),
    ]
    raw = _profile(
        [
            _component(0, "aaaaaaaa-300-0", 0, 1, copy.deepcopy(lods)),
            _component(1, "bbbbbbbb-300-0", 1, 1, copy.deepcopy(lods[::-1])),
        ],
        # Mapping traversal order differs from the first component's tied
        # vertex/index-count order, but the stored entries are already valid.
        lod_sources={other_lod: "D:\\LOD2", LOD_OBJECT: "D:\\LOD1"},
        last_lod_object_name=LOD_OBJECT,
    )
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-current-order-") as workspace:
        profile_module.write_profile(workspace, raw)
        profile_path = profile_module.get_profile_path(workspace)
        before = _read_bytes(profile_path)
        restored = mapper_module.restore_merged_skeleton_lod(workspace)
        assert not restored.changed
        assert restored.matched_component_counts == [2, 2]
        assert _read_bytes(profile_path) == before


def check_current_normalization_defaults_do_not_trigger_a_write():
    lod = _lod("11111111-120-0", vg_map={"0": 0})
    # These absent fields normalize to the correct zero first index already;
    # formatting/default expansion alone is not a semantic compatibility fix.
    del lod["first_index"]
    del lod["unique_str"]
    raw = _profile([_component(0, "aaaaaaaa-300-0", 0, 1, [lod])])
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-current-defaults-") as workspace:
        profile_path = profile_module.get_profile_path(workspace)
        _write_json(profile_path, raw)
        before = _read_bytes(profile_path)
        restored = mapper_module.restore_merged_skeleton_lod(workspace)
        assert not restored.changed and _read_bytes(profile_path) == before


def check_mapping_from_another_workspace():
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-other-") as root:
        mapped_workspace = os.path.join(root, "mapped", "workplace")
        new_workspace = os.path.join(root, "new", "workplace")
        profile_module.write_profile(mapped_workspace, _mapped_source())
        preview = _make_preview(mapped_workspace)
        profile_module.write_profile(new_workspace, _reextracted_target())
        new_profile_path = profile_module.get_profile_path(new_workspace)
        unmapped = _read_bytes(new_profile_path)

        # Nothing of its own to use, and an unrelated folder is rejected.
        for source_path in ("", os.path.join(root, "new"), root):
            try:
                mapper_module.restore_merged_skeleton_lod(new_workspace, source_path)
            except extractor_module.ExtractError:
                pass
            else:
                raise AssertionError("restore without a mapping must fail")
            assert _read_bytes(new_profile_path) == unmapped

        mapped_profile_path = profile_module.get_profile_path(mapped_workspace)
        # The folder holding ``workplace``, the data folder and the JSON all work.
        for source_path in (
            os.path.join(root, "mapped"), mapped_workspace, mapped_profile_path,
        ):
            assert mapper_module.find_lod_mapping_source(source_path) == mapped_profile_path

        restored = mapper_module.restore_merged_skeleton_lod(
            new_workspace, os.path.join(root, "mapped")
        )
        assert restored.changed and restored.source_path == mapped_profile_path
        profile = profile_module.load_profile(new_workspace)
        assert profile["lod_object_names"] == [LOD_OBJECT]
        # The preview meshes stay where they are and are referenced in place.
        recorded = profile["lod_preview_workspaces"][LOD_OBJECT]
        assert os.path.isabs(recorded) and os.path.samefile(recorded, preview)
        # The source workspace is only read.
        assert not os.path.exists(profile_module.get_lod_backup_path(mapped_workspace))

        # A mapping for a different character leaves the profile as it was.
        profile_module.write_profile(new_workspace, _profile([
            _component(0, "12121212-9-0", 0, 1),
        ]))
        other = _read_bytes(new_profile_path)
        try:
            mapper_module.restore_merged_skeleton_lod(new_workspace, mapped_workspace)
        except extractor_module.ExtractError as exc:
            assert "没有可对应的组件" in str(exc)
        else:
            raise AssertionError("an unrelated mapping must not be applied")
        assert _read_bytes(new_profile_path) == other


def main():
    check_apply_pairs_components_by_draw()
    check_skinning_type_and_vertex_group_range()
    check_unrelated_mapping_changes_nothing()
    check_other_lod_objects_are_kept_and_levels_added()
    check_entries_from_before_first_index_was_stored()
    check_same_draw_entry_takes_the_component_base_vertex()
    check_backup_keeps_the_last_mapping()
    check_extraction_then_restore_in_the_same_workspace()
    check_older_format_is_noted_in_the_log_only()
    check_current_legacy_profile_is_migrated_once()
    check_current_invalid_entries_are_saved_as_fallbacks()
    check_current_equal_size_lod_order_does_not_trigger_a_write()
    check_current_normalization_defaults_do_not_trigger_a_write()
    check_mapping_from_another_workspace()
    print("MERGED_LOD_JSON_RESTORE_REGRESSION=PASS")


if __name__ == "__main__":
    main()
