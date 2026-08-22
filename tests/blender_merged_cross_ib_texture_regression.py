"""Blender regression for merged-skeleton Cross-IB target textures.

Usage (after opening a .blend on Blender's command line)::

    blender --background project.blend --python tests/blender_merged_cross_ib_texture_regression.py -- \
        WORKSPACE_ROOT OUTPUT_DIR TREE_NAME SOURCE_UNIQUE_STR TARGET_UNIQUE_STR

The generated incoming Cross-IB draw must bind the source mesh buffers together
with the target component textures.  No workspace or .blend file is modified.
"""

from __future__ import annotations

import os
import re
import sys

import bpy


def _arguments() -> list[str]:
    try:
        return sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        return []


def _main() -> None:
    args = _arguments()
    if len(args) != 5:
        raise RuntimeError(
            "expected WORKSPACE_ROOT OUTPUT_DIR TREE_NAME "
            "SOURCE_UNIQUE_STR TARGET_UNIQUE_STR"
        )

    workspace_root, output_dir, tree_name, source_unique, target_unique = args
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

    model = BluePrintModel(tree=tree, context=bpy.context)
    source_key = "indexcount_" + source_unique.split("-")[1]
    target_key = "indexcount_" + target_unique.split("-")[1]
    # The user's saved .blend may not contain the currently edited Cross-IB
    # node.  Inject exactly one mapping in memory so this regression remains
    # deterministic without modifying the project file.
    model.has_cross_ib = True
    model.cross_ib_info_dict = {source_key: [target_key]}
    model.cross_ib_method_dict = {"regression": "MERGED_SKELETON"}
    model.cross_ib_mapping_objects = {
        (source_key, target_key): {source_unique}
    }
    model.cross_ib_source_to_target_dict = {source_key: [target_key]}
    model.cross_ib_target_info = {target_key: [source_key]}
    model.cross_ib_match_mode = "INDEX_COUNT"
    model.cross_ib_object_names = {source_unique}

    exporter = ExportEFMI(model)
    exporter.export()

    target_component = next(
        (
            component
            for component in exporter.merged_skeleton_profile["components"]
            if component["unique_str"] == target_unique
        ),
        None,
    )
    if target_component is None:
        raise AssertionError("target component not found: " + target_unique)

    ini_files = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith(".ini")
    ]
    if len(ini_files) != 1:
        raise AssertionError("expected one generated INI: " + repr(ini_files))
    with open(ini_files[0], "r", encoding="utf-8") as ini_file:
        ini_text = ini_file.read()

    component_header = (
        "[CommandList_Draw_Component"
        + str(target_component["component_id"])
        + "]"
    )
    section_match = re.search(
        re.escape(component_header) + r"(.*?)(?=\n\[|\Z)",
        ini_text,
        flags=re.DOTALL,
    )
    if section_match is None:
        raise AssertionError("target command list not found: " + component_header)
    target_section = section_match.group(0)

    marker = (
        "; 骨骼合并跨 IB: "
        + source_unique
        + " -> indexcount_"
        + str(target_component["index_count"])
    )
    marker_index = target_section.find(marker)
    if marker_index < 0:
        raise AssertionError("Cross-IB draw marker not found: " + marker)
    incoming_section = target_section[marker_index:]
    draw_match = re.search(r"^drawindexed\s*=.*$", incoming_section, re.MULTILINE)
    if draw_match is None:
        raise AssertionError("incoming Cross-IB draw command not found")
    incoming_draw = incoming_section[:draw_match.end()]

    source_buffer_prefix = "Resource_" + source_unique.replace("-", "_")
    target_texture_prefix = "Resource-" + target_unique + "-"
    assert source_buffer_prefix in incoming_draw, incoming_draw
    assert target_texture_prefix in target_section[:marker_index], target_section
    assert "ps-t" not in incoming_draw, incoming_draw
    assert "Resource\\RabbitFx" not in incoming_draw, incoming_draw
    assert "Resource-" not in incoming_draw, incoming_draw

    print("MERGED_CROSS_IB_INHERITED_TARGET_TEXTURES=PASS")
    print("MERGED_CROSS_IB_SOURCE=" + source_unique)
    print("MERGED_CROSS_IB_TARGET=" + target_unique)
    print("MERGED_CROSS_IB_INI=" + ini_files[0])


if __name__ == "__main__":
    _main()
