# -*- coding: utf-8 -*-
"""End-to-end Blender export regression for merged-skeleton LoD metadata."""

from __future__ import annotations

import os
import re
import sys

import bpy
import numpy


def _arguments():
    try:
        return sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        return []


def main():
    args = _arguments()
    if len(args) != 3:
        raise RuntimeError("expected WORKSPACE_ROOT OUTPUT_DIR TREE_NAME")
    workspace_root, output_dir, tree_name = args
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    from LoyalTools.blueprint.export_helper import BlueprintExportHelper
    from LoyalTools.blueprint.model import BluePrintModel
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.ui.universal.efmi import ExportEFMI

    props = bpy.context.scene.global_properties
    props.workspace_source_mode = "CUSTOM"
    props.custom_workspace_folder_path = workspace_root
    props.force_standalone_preset = True
    props.standalone_game_preset = LogicName.EFMI
    props.use_specific_generate_mod_folder_path = True
    props.generate_mod_folder_path = output_dir
    GlobalConfig.read_from_main_json_ssmt4()
    BlueprintExportHelper.set_current_buffer_folder_name("Buffer")

    tree = bpy.data.node_groups.get(tree_name)
    if tree is None:
        raise AssertionError("blueprint tree not found: " + tree_name)
    exporter = ExportEFMI(BluePrintModel(tree=tree, context=bpy.context))
    assert exporter.merged_skeleton_profile["max_lod_count"] > 0
    exporter.export()

    ini_files = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith(".ini")
    ]
    assert len(ini_files) == 1, ini_files
    with open(ini_files[0], "r", encoding="utf-8") as ini_file:
        ini_text = ini_file.read()

    submesh_by_unique = {
        submesh.unique_str: submesh for submesh in exporter.submesh_model_list
    }
    expected_lod_entrypoints = 0
    for component in exporter.merged_skeleton_profile["components"]:
        component_id = int(component["component_id"])
        for lod_level, lod in enumerate(component.get("lods", []), start=1):
            if lod["ib_hash"] != component["ib_hash"]:
                expected_lod_entrypoints += 1
                header = (
                    "[TextureOverride_EntryPoint_Component"
                    + str(component_id) + "_LOD" + str(lod_level) + "]"
                )
                assert header in ini_text
                section = ini_text.split(header, 1)[1].split("\n[", 1)[0]
                assert "hash = " + lod["ib_hash"] in section
                assert "$lod_level = " + str(lod_level) in section

            remap = exporter.merged_lod_blend_remaps.get(
                (component_id, lod_level)
            )
            if remap:
                path = os.path.join(output_dir, "Buffer", remap["filename"])
                assert os.path.isfile(path), path
                values = numpy.fromfile(path, dtype=numpy.uint16)
                assert len(values) == int(component["vg_count"])
                expected = [
                    int(lod.get("vg_map", {}).get(str(vg_id), vg_id))
                    for vg_id in range(int(component["vg_count"]))
                ]
                assert values.tolist() == expected
                assert (
                    "Pool_MergedSkeleton_Component_LodRemaps"
                    "[$component_id*$lod_level_count+" + str(lod_level) + "] = ref "
                    + remap["resource_name"]
                ) in ini_text

        submesh = submesh_by_unique.get(component["unique_str"])
        if submesh is None:
            continue
        for lod_level, lod in enumerate(component.get("lods", []), start=1):
            for slot in lod.get("vb_formats", {}):
                variant = exporter.merged_lod_variant_buffers[
                    (component["unique_str"], lod_level, slot)
                ]
                path = os.path.join(output_dir, "Buffer", variant["filename"])
                source_stride = int(
                    submesh.d3d11_game_type.CategoryStrideDict[
                        variant["category"]
                    ]
                )
                exported_vertex_count = (
                    len(submesh.category_buffer_dict[variant["category"]])
                    // source_stride
                )
                assert os.path.getsize(path) == (
                    exported_vertex_count * int(variant["stride"])
                )
                assert "[" + variant["resource_name"] + "]" in ini_text

    actual_lod_entrypoints = len(re.findall(
        r"^\[TextureOverride_EntryPoint_Component\d+_LOD\d+\]\r?$",
        ini_text,
        flags=re.MULTILINE,
    ))
    assert actual_lod_entrypoints == expected_lod_entrypoints
    print("MERGED_LOD_EXPORT_REGRESSION=PASS")
    print("LOD_ENTRYPOINTS=" + str(actual_lod_entrypoints))
    print("LOD_VARIANTS=" + str(len(exporter.merged_lod_variant_buffers)))
    print("LOD_REMAPS=" + str(len(exporter.merged_lod_blend_remaps)))
    print("LOD_INI=" + ini_files[0])


if __name__ == "__main__":
    main()
