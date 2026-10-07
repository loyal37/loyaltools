# -*- coding: utf-8 -*-
"""Blender regression for the two LoD mapping buttons.

"Frame Analysis match" still matches from the two captures.  "JSON match"
applies the mapping already stored in the workspace's JSON without reading a
capture: its profile, or the copy kept from before the full-detail model was
extracted again.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bpy

# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LOD_OBJECT = "Character 400"


def _component(component_id, unique_str, vg_offset, vg_count, lods=()):
    ib_hash, index_count, first_index = unique_str.split("-")
    return {
        "component_id": component_id,
        "unique_str": unique_str,
        "ib_hash": ib_hash,
        "index_count": int(index_count),
        "first_index": int(first_index),
        "vertex_count": 100,
        "cpu_posed": False,
        "vg_offset": vg_offset,
        "vg_count": vg_count,
        "vg_map": {str(local): vg_offset + local for local in range(vg_count)},
        "lods": list(lods),
    }


def _lod(unique_str, **extra):
    ib_hash, index_count, first_index = unique_str.split("-")
    lod = {
        "lod_object_name": LOD_OBJECT,
        "ib_hash": ib_hash,
        "vb0_hash": "aabbccdd",
        "vertex_offset": 0,
        "vertex_count": 50,
        "index_offset": 0,
        "index_count": int(index_count),
        "first_index": int(first_index),
        "unique_str": unique_str,
        "is_fallback": False,
        "vg_map": {},
        "vb_formats": {},
    }
    lod.update(extra)
    return lod


def _profile(mapped):
    return {
        "format_version": 1,
        "mode": "EFMI_MERGED_SKELETON",
        "object_name": "Character 1000",
        "components": [
            _component(0, "aaaaaaaa-300-0", 0, 2, [
                _lod("11111111-120-0", vg_map={"0": 1, "1": 0}),
            ] if mapped else ()),
            _component(1, "bbbbbbbb-90-0", 2, 1, [
                _lod("bbbbbbbb-90-0", is_fallback=True),
            ] if mapped else ()),
        ],
    }


def _capture(root, name):
    folder = os.path.join(root, name)
    os.makedirs(folder)
    with open(os.path.join(folder, "log.txt"), "w", encoding="utf-8") as file:
        file.write("")
    return folder


class _RecordingLayout:
    """Stands in for UILayout so the panel's draw code can run headless."""

    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return _RecordingLayout(self.calls)
        return call


def _draw_panel(panel_class, context):
    """Run the panel's draw code and check what it passes to Blender."""
    calls = []
    panel_class.draw(SimpleNamespace(layout=_RecordingLayout(calls)), context)
    functions = bpy.types.UILayout.bl_rna.functions
    icons = set(functions["operator"].parameters["icon"].enum_items.keys())
    for name, _args, kwargs in calls:
        if name in functions:
            assert set(kwargs) <= set(functions[name].parameters.keys()), (name, kwargs)
        assert kwargs.get("icon", 'NONE') in icons, (name, kwargs)
    return calls


def main():
    if not hasattr(bpy.context.scene, "loyal_extract_props"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    from LoyalTools.common import efmi_merged_skeleton as profile_module
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.extract import merged_skeleton_lod_mapper as mapper_module
    from LoyalTools.ui.ui_panel_extract import LOYAL_PT_ExtractPanel

    global_props = bpy.context.scene.global_properties
    global_props.workspace_source_mode = "CUSTOM"
    global_props.force_standalone_preset = True
    global_props.standalone_game_preset = LogicName.EFMI
    props = bpy.context.scene.loyal_extract_props
    props.workflow_mode = 'MERGED_SKELETON'

    # The LoD box offers one button per source and no explanatory text.
    calls = _draw_panel(LOYAL_PT_ExtractPanel, bpy.context)
    buttons = {
        args[0]: kwargs.get("text") for name, args, kwargs in calls if name == "operator"
    }
    assert buttons["loyal.map_merged_skeleton_lod"] == "帧分析匹配"
    assert buttons["loyal.map_merged_skeleton_lod_json"] == "JSON 匹配"
    labels = [kwargs.get("text") for name, _args, kwargs in calls if name == "label"]
    assert labels == ["LOD 映射"], labels
    drawn_properties = [args[1] for name, args, _kwargs in calls if name == "prop"]
    assert drawn_properties == [
        "workflow_mode", "frame_dump_folder", "lod_frame_dump_folder", "show_lod_import",
    ], drawn_properties

    def use_workspace(root):
        workspace = os.path.join(root, "workplace")
        os.makedirs(workspace, exist_ok=True)
        global_props.custom_workspace_folder_path = root
        GlobalConfig.read_from_main_json_ssmt4()
        assert os.path.samefile(GlobalConfig.path_workspace_folder(), workspace)
        return workspace

    def set_captures(lod_folder="", full_folder=""):
        props.lod_frame_dump_folder = lod_folder
        props.frame_dump_folder = full_folder
        props.last_report = ""

    def error_of(operator, **kwargs):
        try:
            operator(**kwargs)
        except RuntimeError as exc:
            return str(exc)
        raise AssertionError("expected the operator to fail")

    def json_match(**kwargs):
        props.last_report = ""
        assert bpy.ops.loyal.map_merged_skeleton_lod_json(**kwargs) == {'FINISHED'}
        return props.last_report.split("\n")

    def read(path):
        with open(path, "rb") as file:
            return file.read()

    capture_match = bpy.ops.loyal.map_merged_skeleton_lod
    with tempfile.TemporaryDirectory(prefix="loyal-lod-json-operator-") as root:
        lod_capture = _capture(root, "FrameAnalysis-LOD")
        full_capture = _capture(root, "FrameAnalysis-Full")
        missing_capture = os.path.join(root, "FrameAnalysis-Deleted")

        workspace = use_workspace(os.path.join(root, "character"))
        profile_path = profile_module.get_profile_path(workspace)
        profile_module.write_profile(workspace, _profile(mapped=False))
        unmapped = read(profile_path)

        # Nothing stored: the JSON button says so and changes nothing.
        assert "还没有 LOD 映射" in error_of(bpy.ops.loyal.map_merged_skeleton_lod_json)
        assert read(profile_path) == unmapped

        # The capture button is unchanged: same prompts, and it never falls
        # back to JSON on its own.
        profile_module.write_profile(workspace, _profile(mapped=True))
        mapped = read(profile_path)
        for expected, folders in (
            ("请先选择 LOD 帧分析目录。", {"full_folder": full_capture}),
            ("LOD 帧分析目录不存在: " + missing_capture, {
                "full_folder": full_capture, "lod_folder": missing_capture,
            }),
            ("找不到完整模型帧。请在“帧分析Dump目录”选择最初用于骨骼合并提取的近景帧。", {
                "lod_folder": lod_capture,
            }),
        ):
            set_captures(**folders)
            assert error_of(capture_match).strip() == "Error: " + expected
            assert read(profile_path) == mapped

        calls = []

        def fake_capture_match(**kwargs):
            calls.append(kwargs)
            return mapper_module.LODMapResult(
                lod_object_name=LOD_OBJECT, matched_component_count=1,
                lower_poly_component_count=1, component_count=2, max_lod_count=1,
            )

        original_match = mapper_module.map_merged_skeleton_lod
        mapper_module.map_merged_skeleton_lod = fake_capture_match
        try:
            set_captures(lod_folder=lod_capture, full_folder=full_capture)
            assert capture_match() == {'FINISHED'}
        finally:
            mapper_module.map_merged_skeleton_lod = original_match
        assert props.last_report.split("\n")[0] == "LOD 映射完成: " + LOD_OBJECT
        assert len(calls) == 1 and os.path.samefile(calls[0]["workspace_folder"], workspace)
        assert os.path.samefile(calls[0]["lod_dump_folder"], lod_capture)
        assert os.path.samefile(calls[0]["full_dump_folder"], full_capture)

        # JSON button, profile already mapped: nothing is rewritten, and the
        # capture fields play no part.
        set_captures(lod_folder=missing_capture)
        report = json_match()
        assert report[:3] == [
            "LOD 映射完成 (JSON): " + LOD_OBJECT,
            "匹配 1/2 个组件，1 个使用独立 LOD IB",
            "当前工作空间共 1 级 LOD",
        ], report
        assert len(report) == 4 and "没有独立对应的 LOD" in report[3]
        assert read(profile_path) == mapped

        # JSON button after the model was extracted again: the kept copy.
        assert profile_module.backup_lod_mapping(workspace) == [LOD_OBJECT]
        profile_module.write_profile(workspace, _profile(mapped=False))
        assert json_match()[:3] == report[:3]
        restored = profile_module.load_profile(workspace)
        assert restored["lod_object_names"] == [LOD_OBJECT]
        assert restored["components"][0]["lods"][0]["vg_map"] == {"0": 1, "1": 0}
        assert restored["components"][1]["lods"][0]["is_fallback"]

        # Scripts may name the JSON of another workspace of the character.
        other = use_workspace(os.path.join(root, "character again"))
        profile_module.write_profile(other, _profile(mapped=False))
        assert "还没有 LOD 映射" in error_of(bpy.ops.loyal.map_merged_skeleton_lod_json)
        assert "没有带 LOD 映射的" in error_of(
            bpy.ops.loyal.map_merged_skeleton_lod_json, json_path=missing_capture
        )
        assert json_match(json_path=profile_path)[:3] == report[:3]
        assert profile_module.load_profile(other)["lod_object_names"] == [LOD_OBJECT]

        # A legacy mapping selected from the current profile needs the same
        # persisted migration as one selected from a backup or another folder.
        legacy_workspace = use_workspace(os.path.join(root, "legacy character"))
        legacy_path = profile_module.get_profile_path(legacy_workspace)
        legacy_lod = _lod("dddddddd-45-12", vg_map={"0": 0, "1": 1})
        del legacy_lod["first_index"]
        del legacy_lod["unique_str"]
        legacy = _profile(mapped=False)
        legacy["components"] = [
            _component(0, "dddddddd-45-12", 0, 2, [legacy_lod]),
        ]
        with open(legacy_path, "w", encoding="utf-8") as file:
            json.dump(legacy, file, ensure_ascii=False, indent=4)
        before = read(legacy_path)
        set_captures(lod_folder=missing_capture)
        legacy_report = json_match()
        assert legacy_report[:3] == [
            "LOD 映射完成 (JSON): " + LOD_OBJECT,
            "匹配 1/1 个组件，0 个使用独立 LOD IB",
            "当前工作空间共 1 级 LOD",
        ], legacy_report
        reused = profile_module.load_profile(legacy_workspace)["components"][0]["lods"][0]
        assert (reused["first_index"], reused["unique_str"]) == (12, "dddddddd-45-12")
        migrated = read(legacy_path)
        assert migrated != before
        assert json_match()[:3] == legacy_report[:3]
        assert read(legacy_path) == migrated

    print("BLENDER_MERGED_LOD_JSON_OPERATOR_REGRESSION=PASS")


if __name__ == "__main__":
    main()
