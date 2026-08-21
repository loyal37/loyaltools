"""Blender regression for merged-skeleton missing-GPU / CPU original-mesh policy.

Usage (after opening a .blend on Blender's command line)::

    blender --background project.blend --python tests/blender_merged_component_policy_regression.py -- \
        WORKSPACE_ROOT OUTPUT_DIR TREE_NAME [CPU_UNIQUE_STR]

When CPU_UNIQUE_STR is provided, the loaded profile is patched in memory to mark
that component CPU posed.  No workspace or .blend file is modified.
"""

from __future__ import annotations

import copy
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
    if len(args) < 3:
        raise RuntimeError(
            "expected WORKSPACE_ROOT OUTPUT_DIR TREE_NAME [CPU_UNIQUE_STR]"
        )

    workspace_root, output_dir, tree_name = args[:3]
    cpu_unique_str = args[3] if len(args) > 3 else ""
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    from LoyalTools.blueprint.export_helper import BlueprintExportHelper
    from LoyalTools.blueprint.model import BluePrintModel
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.ui.universal import efmi as efmi_module
    from LoyalTools.ui.universal.export_helper import ExportHelper

    props = bpy.context.scene.global_properties
    props.workspace_source_mode = "CUSTOM"
    props.custom_workspace_folder_path = workspace_root
    props.force_standalone_preset = True
    props.standalone_game_preset = LogicName.EFMI
    props.use_specific_generate_mod_folder_path = True
    props.generate_mod_folder_path = output_dir
    GlobalConfig.read_from_main_json_ssmt4()
    BlueprintExportHelper.set_current_buffer_folder_name("Buffer")

    if cpu_unique_str:
        original_load_profile = efmi_module.load_profile

        def load_profile_with_cpu_component(workspace_folder, required=True):
            profile = copy.deepcopy(
                original_load_profile(workspace_folder, required=required)
            )
            matched = False
            for component in profile["components"]:
                if component["unique_str"] == cpu_unique_str:
                    component["cpu_posed"] = True
                    matched = True
                    break
            if not matched:
                raise AssertionError("CPU test component not found: " + cpu_unique_str)
            return profile

        efmi_module.load_profile = load_profile_with_cpu_component

    tree = bpy.data.node_groups.get(tree_name)
    if tree is None:
        raise AssertionError("blueprint tree not found: " + tree_name)

    model = BluePrintModel(tree=tree, context=bpy.context)
    exporter = efmi_module.ExportEFMI(model)
    exporter.export()

    profile = exporter.merged_skeleton_profile
    gpu_components = [
        component for component in profile["components"]
        if not component["cpu_posed"]
    ]
    cpu_components = [
        component for component in profile["components"]
        if component["cpu_posed"]
    ]
    blueprint_unique_strs = {
        draw_call_model.get_unique_str()
        for draw_call_model in model.ordered_draw_obj_data_model_list
    }
    connected_cpu_components = [
        component for component in cpu_components
        if component["unique_str"] in blueprint_unique_strs
    ]
    disconnected_cpu_components = [
        component for component in cpu_components
        if component["unique_str"] not in blueprint_unique_strs
    ]
    exported_unique_strs = {
        submesh.unique_str for submesh in exporter.submesh_model_list
    }
    missing_gpu_components = [
        component for component in gpu_components
        if component["unique_str"] not in exported_unique_strs
    ]

    ini_path = os.path.join(output_dir, os.path.basename(workspace_root) + ".ini")
    with open(ini_path, "r", encoding="utf-8") as ini_file:
        ini_text = ini_file.read()

    entrypoint_count = len(re.findall(
        r"^\[TextureOverride_EntryPoint_Component\d+\]$",
        ini_text,
        flags=re.MULTILINE,
    ))
    callback_count = len(re.findall(
        r"^\[CommandList_Draw_Component\d+\]$",
        ini_text,
        flags=re.MULTILINE,
    ))
    expected_active_component_count = (
        len(gpu_components) + len(connected_cpu_components)
    )
    assert entrypoint_count == expected_active_component_count, (
        entrypoint_count,
        expected_active_component_count,
    )
    assert callback_count == expected_active_component_count, (
        callback_count,
        expected_active_component_count,
    )
    if not connected_cpu_components:
        assert "$\\EFMIv1\\gpu_posed = 0" not in ini_text

    for component in missing_gpu_components:
        component_id = component["component_id"]
        assert (
            "[TextureOverride_EntryPoint_Component" + str(component_id) + "]"
            in ini_text
        )
        assert (
            "[CommandList_Draw_Component" + str(component_id) + "]"
            in ini_text
        )

    for cpu_component in cpu_components:
        current_cpu_unique_str = cpu_component["unique_str"]
        cpu_id = cpu_component["component_id"]
        assert current_cpu_unique_str not in exported_unique_strs
        assert not any(
            name.startswith(current_cpu_unique_str + "-")
            for name in os.listdir(os.path.join(output_dir, "Buffer"))
        )
        if cpu_component in disconnected_cpu_components:
            assert (
                "[TextureOverride_EntryPoint_Component" + str(cpu_id) + "]"
                not in ini_text
            )
            assert (
                "[CommandList_Draw_Component" + str(cpu_id) + "]"
                not in ini_text
            )
            textures_folder = os.path.join(output_dir, "Textures")
            if os.path.isdir(textures_folder):
                assert not any(
                    name.startswith(current_cpu_unique_str + "-")
                    for name in os.listdir(textures_folder)
                )
            continue
        assert (
            "[TextureOverride_EntryPoint_Component" + str(cpu_id) + "]"
            in ini_text
        )
        assert (
            "[CommandList_Draw_Component" + str(cpu_id) + "]"
            in ini_text
        )
        cpu_entrypoint = re.search(
            r"\[TextureOverride_EntryPoint_Component" + str(cpu_id)
            + r"\](.*?)(?=\n\[|\Z)",
            ini_text,
            flags=re.DOTALL,
        ).group(0)
        cpu_callback = re.search(
            r"\[CommandList_Draw_Component" + str(cpu_id)
            + r"\](.*?)(?=\n\[|\Z)",
            ini_text,
            flags=re.DOTALL,
        ).group(0)
        assert "$\\EFMIv1\\gpu_posed = 0" in cpu_entrypoint
        assert "drawindexed = INDEX_COUNT, FIRST_INDEX, 0" in cpu_callback
        assert "ib = Resource_" not in cpu_callback
        assert "vb0 = Resource_" not in cpu_callback
    if cpu_unique_str and cpu_unique_str in blueprint_unique_strs:
        # The allow-list is scoped to merged skeleton parsing.  The ordinary
        # pipeline must still see the same connected object.
        ordinary_submeshes = ExportHelper.parse_submesh_model_list_from_blueprint_model(
            model,
            efmi_merged_skeleton=False,
            efmi_merged_skeleton_unique_strs=set(),
        )
        assert cpu_unique_str in {
            submesh.unique_str for submesh in ordinary_submeshes
        }

    print("LOYAL_POLICY_GPU_COMPONENTS=" + str(len(gpu_components)))
    print("LOYAL_POLICY_EXPORTED_SUBMESHES=" + str(len(exported_unique_strs)))
    print("LOYAL_POLICY_MISSING_GPU=" + str(len(missing_gpu_components)))
    print("LOYAL_POLICY_CPU_CONNECTED=" + str(len(connected_cpu_components)))
    print("LOYAL_POLICY_CPU_DISCONNECTED=" + str(len(disconnected_cpu_components)))
    print("LOYAL_POLICY_INI=" + ini_path)


if __name__ == "__main__":
    _main()
