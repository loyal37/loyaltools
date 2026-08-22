# -*- coding: utf-8 -*-
"""Blender regression for isolated, repeatable LoD preview import."""

from __future__ import annotations

import sys

import bpy


def _arguments():
    try:
        return sys.argv[sys.argv.index("--") + 1:]
    except ValueError:
        return []


def main():
    args = _arguments()
    if len(args) != 3:
        raise RuntimeError("expected WORKSPACE_ROOT LOD_DUMP EXPECTED_COMPONENTS")
    workspace_root, lod_dump, expected_components = args
    expected_components = int(expected_components)

    if not hasattr(bpy.context.scene, "loyal_extract_props"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName

    props = bpy.context.scene.global_properties
    props.workspace_source_mode = "CUSTOM"
    props.custom_workspace_folder_path = workspace_root
    props.force_standalone_preset = True
    props.standalone_game_preset = LogicName.EFMI
    GlobalConfig.read_from_main_json_ssmt4()

    extract_props = bpy.context.scene.loyal_extract_props
    extract_props.workflow_mode = 'MERGED_SKELETON'
    extract_props.lod_frame_dump_folder = lod_dump
    assert extract_props.show_lod_import is False

    blueprint_count = len([
        tree for tree in bpy.data.node_groups
        if getattr(tree, "bl_idname", "") == 'SSMTBlueprintTreeType'
    ])
    result = bpy.ops.loyal.import_merged_skeleton_lod()
    assert result == {'FINISHED'}, result

    lod_collections = [
        collection for collection in bpy.data.collections
        if collection.get("LoyalTools:EFMILODPreview")
        and collection.get("LoyalTools:EFMILODObject") == "Character 35051"
    ]
    assert len(lod_collections) == 1, [collection.name for collection in lod_collections]
    collection = lod_collections[0]
    assert collection.color_tag == 'COLOR_03'
    assert len(collection.objects) == expected_components, len(collection.objects)
    assert all(obj.get("LoyalTools:EFMILODPreview") for obj in collection.objects)
    assert all(
        obj.get("LoyalTools:EFMILODObject") == "Character 35051"
        for obj in collection.objects
    )
    assert len([
        tree for tree in bpy.data.node_groups
        if getattr(tree, "bl_idname", "") == 'SSMTBlueprintTreeType'
    ]) == blueprint_count

    # Repeating the command must create a separate numbered collection.
    result = bpy.ops.loyal.import_merged_skeleton_lod()
    assert result == {'FINISHED'}, result
    lod_collections = [
        collection for collection in bpy.data.collections
        if collection.get("LoyalTools:EFMILODPreview")
        and collection.get("LoyalTools:EFMILODObject") == "Character 35051"
    ]
    assert len(lod_collections) == 2, [collection.name for collection in lod_collections]
    print("BLENDER_MERGED_LOD_IMPORT_REGRESSION=PASS")
    print("LOD_OBJECT=Character 35051")
    print("LOD_COMPONENTS=" + str(expected_components))
    print("LOD_COLLECTIONS=2")
    print("BLUEPRINTS_ADDED=0")


if __name__ == "__main__":
    main()
