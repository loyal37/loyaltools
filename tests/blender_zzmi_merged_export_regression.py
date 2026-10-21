# -*- coding: utf-8 -*-
"""Blender regression: Zenless Zone Zero merged skeleton, from extraction to the mod.

Run with:
    blender --background --factory-startup --python-exit-code 1 \
        --python tests/blender_zzmi_merged_export_regression.py

1. "Merged skeleton" extraction of a synthetic dump: merged parts share one set
   of vertex groups, the face keeps its own and the bangboo is left out.
2. An unchanged export needs no runtime: every part skins with its own bones
   and the face keeps the game mesh (textures only).
3. A copy of the hair joined into the body: the body is skinned with a merged
   palette built from the bone pool, the hair imports its bones into the pool
   when it is skinned, and the exported vertices, run through the palette,
   land where the game puts them.
4. The hair and the accessory taken out of the blueprint: as in EFMI's merged
   skeleton they are no longer drawn; the hair still provides its bones.

Every skinning branch is limited to the component's own skinning draw
(``VERTEX_COUNT``): full-screen passes keep a character's vb2 bound and must not
trigger it.  No stream-output target is ever re-bound.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import bpy
import numpy

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parents[1]))
sys.path.insert(0, str(TESTS_DIR))

import zzmi_merged_synthetic_dump as synthetic  # noqa: E402


def _section(ini_text, header):
    assert header in ini_text, header + " missing"
    lines = ini_text.split(header, 1)[1].split("\n[", 1)[0].splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith(";")]


def _export(tree_name, output_dir):
    props = bpy.context.scene.global_properties
    props.generate_mod_folder_path = output_dir
    os.makedirs(output_dir, exist_ok=True)
    bpy.ops.ssmt.generate_mod_blueprint(blueprint_name=tree_name)
    ini_files = [name for name in os.listdir(output_dir) if name.lower().endswith(".ini")]
    assert len(ini_files) == 1, ini_files
    with open(os.path.join(output_dir, ini_files[0]), encoding="utf-8") as handle:
        return handle.read()


def _read_body_buffers(mod_folder, ib_hash):
    meshes = os.path.join(mod_folder, "Meshes")
    position = numpy.fromfile(os.path.join(meshes, ib_hash + "-Position.buf"), dtype=numpy.float32).reshape(-1, 10)
    blend = numpy.fromfile(os.path.join(meshes, ib_hash + "-Blend.buf"), dtype=numpy.uint8).reshape(-1, 32)
    weights = blend[:, :16].copy().view(numpy.float32)
    indices = blend[:, 16:].copy().view(numpy.uint32).astype(numpy.int64)
    return position, weights, indices


def _draw_count(mod_folder, ib_hash):
    return os.path.getsize(os.path.join(mod_folder, "Meshes", ib_hash + "-Position.buf")) // 40


def _check_runtime(zms, profile, source, position, weights, indices, drawing, provider):
    """Skin the exported vertices with the palette the runtime builds and compare with the game."""
    parts = source["components"]
    merged = zms.merged_components(profile)
    pool = numpy.concatenate([parts[name]["palette"] for name in ("body", "hair", "accessory")]).astype(numpy.float64)
    assert [component["ib_hash"] for component in merged] == [parts[name]["spec"]["ib"] for name in ("body", "hair", "accessory")]
    drawing_component = zms.get_component_by_ib(profile)[parts[drawing]["spec"]["ib"]]
    palette = zms.reference_palette(profile, pool, drawing_component["component_id"])
    matrices = numpy.zeros((len(position), 3, 4))
    for k in range(4):
        matrices += weights[:, k, None, None] * palette[indices[:, k]]
    skinned = numpy.einsum("vij,vj->vi", matrices, numpy.c_[position[:, :3], numpy.ones(len(position))])

    to_drawing = numpy.linalg.inv(parts[drawing]["root"]) @ parts[provider]["root"]
    expected = {}
    for name, transform in ((drawing, numpy.eye(4)), (provider, to_drawing)):
        rest = parts[name]["position"]["POSITION"].astype(numpy.float32)
        posed = parts[name]["posed"]["POSITION"].astype(numpy.float64)
        for vertex in range(len(rest)):
            moved = (transform @ numpy.r_[posed[vertex], 1.0])[:3]
            expected.setdefault(rest[vertex].tobytes(), []).append((name, moved))
    worst = {drawing: 0.0, provider: 0.0}
    counts = {drawing: 0, provider: 0}
    for vertex in range(len(position)):
        candidates = expected[position[vertex, :3].tobytes()]
        errors = [(float(numpy.abs(skinned[vertex] - moved).max()), name) for name, moved in candidates]
        error, name = min(errors)
        worst[name] = max(worst[name], error)
        counts[name] += 1
    assert counts[drawing] and counts[provider], counts
    assert worst[drawing] < 1e-5 and worst[provider] < 1e-4, worst
    return worst, counts


def main():
    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    import LoyalTools
    assert LoyalTools.__file__.startswith(str(TESTS_DIR.parents[1])), LoyalTools.__file__
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.common import zzmi_merged_skeleton as zms
    from LoyalTools.common.global_key_count_helper import GlobalKeyCountHelper

    with tempfile.TemporaryDirectory(prefix="loyal_zzmi_merged_export_") as temp_folder:
        dump_folder = os.path.join(temp_folder, "FrameAnalysis-merged")
        os.makedirs(dump_folder)
        source = synthetic.build_dump(dump_folder)
        parts = source["components"]
        ib = {name: data["spec"]["ib"] for name, data in parts.items()}

        props = bpy.context.scene.global_properties
        props.workspace_source_mode = "CUSTOM"
        props.custom_workspace_folder_path = os.path.join(temp_folder, "zzmi_merged")
        props.force_standalone_preset = True
        props.standalone_game_preset = LogicName.ZZMI
        props.use_specific_generate_mod_folder_path = True
        props.open_mod_folder_after_generate_mod = False
        props.zzz_outline_optimization = False
        GlobalConfig.read_from_main_json_ssmt4()
        workspace = GlobalConfig.path_workspace_folder()

        # 1. extraction
        scene = bpy.context.scene
        scene.loyal_extract_props.workflow_mode = 'MERGED_SKELETON'
        scene.loyal_extract_props.frame_dump_folder = dump_folder
        assert bpy.ops.loyal.extract_from_dump() == {'FINISHED'}
        assert scene.loyal_extract_props.last_report.startswith("骨骼合并提取完成"), scene.loyal_extract_props.last_report
        profile = zms.load_profile(workspace)
        tree_name = props.selected_blueprint_name
        tree = bpy.data.node_groups[tree_name]
        assert tree.get(zms.TREE_MARKER_MERGED_SKELETON) and not tree.get("LoyalTools:EFMIMergedSkeleton")

        objects = {name: bpy.data.objects.get(parts[name]["unique_str"]) for name in parts}
        assert objects["bangboo"] is None
        for name in ("body", "hair", "accessory"):
            assert len(objects[name].vertex_groups) == profile["bones_count"] == 15, name
            assert objects[name][zms.OBJECT_MARKER_ROLE] == zms.ROLE_MERGED
        assert len(objects["face"].vertex_groups) <= 3
        assert objects["face"][zms.OBJECT_MARKER_ROLE] == zms.ROLE_SHAPE_KEYED
        # The hair's head weights land on the body's head group (canonical id 6).
        head_group = objects["hair"].vertex_groups["6"].index
        assert any(group.group == head_group for vertex in objects["hair"].data.vertices for group in vertex.groups)

        # The import button rebuilds the same scene from the profile.
        assert bpy.ops.loyal.import_workspace() == {'FINISHED'}
        copies = [obj for obj in bpy.data.objects if obj.name.startswith(parts["hair"]["unique_str"] + ".")]
        assert copies and len(copies[0].vertex_groups) == 15
        for obj in [obj for obj in bpy.data.objects if "." in obj.name and obj.name.split(".")[0] in
                    {data["unique_str"] for data in parts.values()}]:
            bpy.data.objects.remove(obj, do_unlink=True)
        props.selected_blueprint_name = tree_name

        # 2. unchanged export
        mod_a = os.path.join(temp_folder, "mod_unchanged")
        ini_a = _export(tree_name, mod_a)
        # $active0..N are numbered per ini: body, hair, accessory and the face
        assert GlobalKeyCountHelper.generated_mod_number == 4, GlobalKeyCountHelper.generated_mod_number
        assert "ZZMS" not in ini_a, "an unchanged export needs no runtime and hides nothing"
        assert not os.path.isdir(os.path.join(mod_a, "Shaders"))
        body_blend_lines = _section(ini_a, "[TextureOverride_VB_" + ib["body"] + "_" + ib["body"] + "_Blend]")
        assert "draw = " + str(_draw_count(mod_a, ib["body"])) + ", 0" in body_blend_lines, body_blend_lines
        assert "vb0 = Resource" + ib["body"] + "Position" in body_blend_lines, body_blend_lines
        assert "[TextureOverride_IB_" + ib["face"] + "]" not in ini_a
        assert "hash = " + parts["face"]["spec"]["blend"] not in ini_a
        assert not os.path.exists(os.path.join(mod_a, "Meshes", ib["face"] + "-Position.buf"))
        face_lines = _section(ini_a, "[TextureOverride_" + parts["face"]["unique_str"].replace("-", "_") + "]")
        assert face_lines[:3] == ["hash = " + ib["face"], "match_first_index = 0",
                                  "match_index_count = " + str(len(parts["face"]["indices"]))], face_lines
        assert face_lines[-1] == r"run = CommandList\ZZMI\SetTextures", face_lines
        assert any(line.startswith(r"Resource\ZZMI\Diffuse = ref ") for line in face_lines), face_lines
        assert not any(line.startswith(("ib =", "handling", "drawindexed")) for line in face_lines), face_lines
        _, _, body_indices = _read_body_buffers(mod_a, ib["body"])
        assert body_indices.max() < 8, "unchanged body must keep its own (local) bone ids"
        _, hair_weights, hair_indices = _read_body_buffers(mod_a, ib["hair"])
        assert hair_indices[hair_weights > 0].max() < 5, "unchanged hair must keep its own (local) bone ids"

        # 3. a copy of the hair joined into the body
        hair_obj, body_obj = objects["hair"], objects["body"]
        hair_copy = hair_obj.copy()
        hair_copy.data = hair_obj.data.copy()
        scene.collection.objects.link(hair_copy)
        bpy.ops.object.select_all(action='DESELECT')
        hair_copy.select_set(True)
        body_obj.select_set(True)
        bpy.context.view_layer.objects.active = body_obj
        bpy.ops.object.join()
        mod_b = os.path.join(temp_folder, "mod_hair_in_body")
        ini_b = _export(tree_name, mod_b)
        assert GlobalKeyCountHelper.generated_mod_number == 4, "the counter must restart for every export"
        body, hair = ib["body"], ib["hair"]
        body_vertices = len(parts["body"]["position"])
        hair_vertices = len(parts["hair"]["position"])
        body_blend_b = _section(ini_b, "[TextureOverride_VB_" + body + "_" + body + "_Blend]")
        assert body_blend_b[-3:] == [
            "elif DRAW_TYPE == 1 && VERTEX_COUNT == " + str(body_vertices),
            "run = CommandListZZMS_Skin_" + body, "endif"], body_blend_b
        assert _section(ini_b, "[CommandListZZMS_Skin_" + body + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + body,
            "run = CustomShaderZZMS_Palette_" + body,
            "vs-t0 = ref ResourceZZMS_Palette_" + body,
            "vb0 = Resource" + body + "Position",
            "vb2 = Resource" + body + "Blend",
            "draw = " + str(_draw_count(mod_b, body)) + ", 0",
        ]
        assert _section(ini_b, "[CommandListZZMS_Skin_" + hair + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + hair,
            "vb0 = Resource" + hair + "Position",
            "vb2 = Resource" + hair + "Blend",
            "draw = " + str(_draw_count(mod_b, hair)) + ", 0",
        ]
        hair_blend_b = _section(ini_b, "[TextureOverride_VB_" + hair + "_" + hair + "_Blend]")
        assert "elif DRAW_TYPE == 1 && VERTEX_COUNT == " + str(hair_vertices) in hair_blend_b, hair_blend_b
        for forbidden in ("so0", "FRAME_NUMBER", "Reskin", "$zzms_seen"):
            assert forbidden not in ini_b, forbidden
        assert "dispatch = 1, 1, 1" in _section(ini_b, "[CustomShaderZZMS_Palette_" + body + "]")
        assert _section(ini_b, "[ResourceZZMS_Components]")[-1] == "data = " + " ".join(
            str(value) for value in zms.runtime_component_table(profile))
        assert _section(ini_b, "[ResourceZZMS_Pool]") == [
            "type = RWBuffer", "format = R32G32B32A32_FLOAT", "array = 45",
            "bind_flags = shader_resource unordered_access"]
        assert "[CommandListZZMS_Skin_" + ib["accessory"] + "]" not in ini_b
        for filename in (zms.IMPORT_SHADER_FILENAME, zms.PALETTE_SHADER_FILENAME):
            assert os.path.isfile(os.path.join(mod_b, "Shaders", filename)), filename
        position, weights, indices = _read_body_buffers(mod_b, body)
        active = weights > 0
        assert set(numpy.unique(indices[active]).tolist()) <= set(range(13))
        assert (indices[active] >= 9).any(), "hair bones must come from the hair's range"
        worst, counts = _check_runtime(zms, profile, source, position, weights, indices, "body", "hair")
        print("hair in body: vertices " + repr(counts) + ", worst error " + repr(worst))

        # 4. the hair and the accessory taken out of the blueprint are hidden; the hair still provides bones
        removed = {parts["hair"]["unique_str"], parts["accessory"]["unique_str"]}
        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") == 'SSMTNode_Object_Info' and node.object_name in removed:
                tree.nodes.remove(node)
        mod_c = os.path.join(temp_folder, "mod_hair_provider")
        ini_c = _export(tree_name, mod_c)
        assert GlobalKeyCountHelper.generated_mod_number == 2, GlobalKeyCountHelper.generated_mod_number
        accessory = ib["accessory"]
        assert "[TextureOverride_VB_" + hair + "_" + hair + "_Blend]" not in ini_c
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_" + hair + "]") == [
            "hash = " + parts["hair"]["spec"]["blend"],
            "if DRAW_TYPE == 1 && VERTEX_COUNT == " + str(hair_vertices),
            "run = CommandListZZMS_Skin_" + hair,
            "elif DRAW_TYPE == 2 || DRAW_TYPE == 4",
            "checktextureoverride = ib",
            "endif"]
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_IB_" + hair + "]") == ["hash = " + hair, "handling = skip"]
        assert _section(ini_c, "[CommandListZZMS_Skin_" + hair + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + hair,
        ]
        # The accessory provides no bone to the body: hidden only, its skinning draw is left alone.
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_" + accessory + "]") == [
            "hash = " + parts["accessory"]["spec"]["blend"],
            "if DRAW_TYPE == 2 || DRAW_TYPE == 4",
            "checktextureoverride = ib",
            "endif"]
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_IB_" + accessory + "]") == [
            "hash = " + accessory, "handling = skip"]
        assert "[CommandListZZMS_Skin_" + accessory + "]" not in ini_c
        # The face is not merged: leaving it out of the blueprint never hides it.
        assert "Hidden_" + ib["face"] not in ini_c
        assert "so0" not in ini_c
        assert not os.path.exists(os.path.join(mod_c, "Meshes", hair + "-Position.buf"))

    print("BLENDER_ZZMI_MERGED_EXPORT_REGRESSION=PASS")


main()
