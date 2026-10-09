# -*- coding: utf-8 -*-
"""Blender regression: extract a synthetic Zenless Zone Zero dump, import it, export a mod.

Run with:
    blender --background --factory-startup --python-exit-code 1 \
        --python tests/blender_zzmi_export_regression.py

The mesh is exported untouched, so every buffer must come back equal to the
frame analysis data, the ini must follow the XXMI-Tools Zenless Zone Zero
layout, and marked textures must be written the SlotFix way.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import bpy
import numpy

TESTS_DIR = Path(__file__).resolve().parent
# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(TESTS_DIR.parents[1]))
sys.path.insert(0, str(TESTS_DIR))

import zzmi_synthetic_dump as synthetic  # noqa: E402


def _section(ini_text, header):
    """Lines of one ini section, without blank lines and comments."""
    assert header in ini_text, header + " missing"
    lines = ini_text.split(header, 1)[1].split("\n[", 1)[0].splitlines()
    return [line for line in lines if line.strip() and not line.startswith(";")]


def _assert_section(ini_text, header, expected):
    actual = _section(ini_text, header)
    assert actual == expected, header + " -> " + repr(actual)


def _rows_by_position(position_rows, *other_rows):
    """Exported vertices are re-ordered; key every row by its position bytes."""
    table = {}
    for index in range(len(position_rows)):
        key = position_rows["POSITION"][index].tobytes()
        assert key not in table, "duplicate exported vertex"
        table[key] = (position_rows[index],) + tuple(rows[index] for rows in other_rows)
    return table


def _export(tree_name, output_dir):
    props = bpy.context.scene.global_properties
    props.generate_mod_folder_path = output_dir
    os.makedirs(output_dir, exist_ok=True)
    bpy.ops.ssmt.generate_mod_blueprint(blueprint_name=tree_name)
    ini_files = [name for name in os.listdir(output_dir) if name.lower().endswith(".ini")]
    assert len(ini_files) == 1, ini_files
    with open(os.path.join(output_dir, ini_files[0]), encoding="utf-8") as handle:
        return handle.read()


def main():
    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.extract.zzmi_dump_extractor import FRAME_EXTRACT_KEY

    with tempfile.TemporaryDirectory(prefix="loyal_zzmi_export_") as temp_folder:
        dump_folder = os.path.join(temp_folder, "FrameAnalysis-synthetic")
        workspace_root = os.path.join(temp_folder, "zzmi_regression")
        os.makedirs(dump_folder)
        source = synthetic.build_dump(dump_folder)

        props = bpy.context.scene.global_properties
        props.workspace_source_mode = "CUSTOM"
        props.custom_workspace_folder_path = workspace_root
        props.force_standalone_preset = True
        props.standalone_game_preset = LogicName.ZZMI
        props.use_specific_generate_mod_folder_path = True
        props.open_mod_folder_after_generate_mod = False
        props.zzz_outline_optimization = False
        GlobalConfig.read_from_main_json_ssmt4()
        assert GlobalConfig.logic_name == LogicName.ZZMI
        workspace_folder = GlobalConfig.path_workspace_folder()

        body_ib, face_ib = synthetic.BODY["ib"], synthetic.FACE["ib"]
        body_grid, body_quad, face = body_ib + "-96-0", body_ib + "-6-96", face_ib + "-102-0"

        # "Extract all": no DrawIB rows; every pre-skinned mesh of the frame is extracted,
        # imported and wired into a blueprint, the static prop is left out. A workflow mode
        # left over from an Endfield session must not matter.
        scene = bpy.context.scene
        scene.loyal_extract_props.workflow_mode = 'MERGED_SKELETON'
        scene.loyal_extract_props.frame_dump_folder = dump_folder
        assert len(scene.loyal_extract_ib_items) == 0
        assert bpy.ops.loyal.extract_zzmi_whole_frame() == {'FINISHED'}
        tree_name = props.selected_blueprint_name
        assert tree_name in bpy.data.node_groups
        imported = {obj.name: obj for obj in bpy.data.objects if obj.name[:8] in (body_ib, face_ib, synthetic.PROP["ib"])}
        assert sorted(imported) == sorted([body_grid, body_quad, face]), sorted(imported)
        # Vertex colours are shader parameters; a byte layer would shift 160 to 161.
        for obj in imported.values():
            assert [attribute.data_type for attribute in obj.data.color_attributes] == ["FLOAT_COLOR"]
        # The dump's normal map is a lossy jpg: reported once, not marked.
        assert "NormalMap" in scene.loyal_extract_props.last_report
        # The Endfield-only button refuses to run for other presets.
        props.standalone_game_preset = LogicName.EFMI
        try:
            bpy.ops.loyal.extract_zzmi_whole_frame()
        except RuntimeError as error:
            assert "ZZMI" in str(error)
        else:
            raise AssertionError("extract-all must be rejected outside the ZZMI preset")
        props.standalone_game_preset = LogicName.ZZMI
        GlobalConfig.read_from_main_json_ssmt4()

        # ------------------------------------------------------------------ export, outline off
        output_dir = os.path.join(temp_folder, "mod_plain")
        ini_text = _export(tree_name, output_dir)

        _assert_section(ini_text, "[TextureOverride_%s_%s_VertexLimitRaise]" % (body_ib, body_ib), [
            "hash = " + synthetic.BODY["draw"],
            "override_byte_stride = 40",
            "override_vertex_count = 29",
        ])
        _assert_section(ini_text, "[TextureOverride_VB_%s_%s_Blend]" % (body_ib, body_ib), [
            "hash = " + synthetic.BODY["blend"],
            "handling = skip",
            "if DRAW_TYPE == 2 || DRAW_TYPE == 4",
            "\tvb1 = Resource" + body_ib + "Texcoord",
            "\tvb2 = Resource" + body_ib + "Blend",
            "\tchecktextureoverride = ib",
            "elif DRAW_TYPE == 1",
            "\tvb0 = Resource" + body_ib + "Position",
            "\tvb2 = Resource" + body_ib + "Blend",
            "\tdraw = 29, 0",
            "endif",
        ])
        _assert_section(ini_text, "[TextureOverride_VB_%s_%s_Texcoord]" % (body_ib, body_ib), [
            "hash = " + synthetic.BODY["texcoord"],
            "vb1 = Resource" + body_ib + "Texcoord",
        ])
        # The position buffer is only swapped in from the blend override.
        assert "_Position]" not in ini_text
        assert "uav_byte_stride" not in ini_text

        grid_section = _section(ini_text, "[TextureOverride_%s]" % body_grid.replace("-", "_"))
        assert grid_section[:4] == [
            "hash = " + body_ib,
            "match_first_index = 0",
            "match_index_count = 96",
            "ib = Resource_%s_Index" % body_grid.replace("-", "_"),
        ]
        assert grid_section[4:8] == [
            "Resource\\ZZMI\\Diffuse = ref Resource-" + body_grid + "-DiffuseMap",
            "Resource\\ZZMI\\LightMap = ref Resource-" + body_grid + "-LightMap",
            "Resource\\ZZMI\\MaterialMap = ref Resource-" + body_grid + "-MaterialMap",
            "run = CommandList\\ZZMI\\SetTextures",
        ]
        assert grid_section[-1] == "drawindexed = 96,0,0"
        quad_section = _section(ini_text, "[TextureOverride_%s]" % body_quad.replace("-", "_"))
        assert "match_first_index = 96" in quad_section and "match_index_count = 6" in quad_section
        assert quad_section[-1] == "drawindexed = 6,0,0"
        assert not re.search(r"^ps-t\d+ = ", ini_text, flags=re.MULTILINE)
        assert "CommandListSkinTexture" not in ini_text
        for name in ("DiffuseMap", "LightMap", "MaterialMap"):
            assert os.path.isfile(os.path.join(output_dir, "Textures", body_grid + "-" + name + ".dds"))

        meshes = os.path.join(output_dir, "Meshes")

        def read(draw_ib, category, dtype):
            return numpy.fromfile(os.path.join(meshes, draw_ib + "-" + category + ".buf"), dtype=dtype)

        # --- body: every exported vertex equals its frame analysis source ---------------------
        exported = _rows_by_position(
            read(body_ib, "Position", synthetic.POSITION_DTYPE),
            read(body_ib, "Texcoord", synthetic.BODY_TEXCOORD_DTYPE),
            read(body_ib, "Blend", synthetic.BODY_BLEND_DTYPE),
        )
        body = source["body"]
        assert len(exported) == len(body["position"]) == 29
        for index in range(29):
            position, texcoord, blend = exported[body["position"]["POSITION"][index].tobytes()]
            assert numpy.allclose(position["NORMAL"], body["position"]["NORMAL"][index], atol=1e-4)
            # Tangents are recomputed by Blender; direction and handedness must match the game's.
            assert numpy.allclose(position["TANGENT"][:3], body["position"]["TANGENT"][index][:3], atol=1e-4)
            assert position["TANGENT"][3] == body["position"]["TANGENT"][index][3]
            assert texcoord["COLOR"].tolist() == body["texcoord"]["COLOR"][index].tolist()
            for name in ("TEXCOORD", "TEXCOORD1", "TEXCOORD2", "TEXCOORD3"):
                assert numpy.allclose(
                    texcoord[name].astype(numpy.float32), body["texcoord"][name][index].astype(numpy.float32), atol=1e-6,
                ), name
            # Unused influences are (bone 0, weight 0); compare the weight each bone ends up with.
            source_weights, exported_weights = {}, {}
            for bone, weight in zip(body["blend"]["BLENDINDICES"][index].tolist(), body["blend"]["BLENDWEIGHTS"][index].tolist()):
                source_weights[bone] = source_weights.get(bone, 0.0) + weight
            for bone, weight in zip(blend["BLENDINDICES"].tolist(), blend["BLENDWEIGHTS"].tolist()):
                exported_weights[bone] = exported_weights.get(bone, 0.0) + weight
            for bone in set(source_weights) | set(exported_weights):
                assert abs(source_weights.get(bone, 0.0) - exported_weights.get(bone, 0.0)) < 1e-6

        # Index buffers address the combined vertex buffer and reproduce the same triangles.
        exported_position = read(body_ib, "Position", synthetic.POSITION_DTYPE)["POSITION"]
        for unique_str, source_indices in ((body_grid, body["grid_indices"]), (body_quad, body["quad_indices"])):
            index_buffer = numpy.fromfile(os.path.join(meshes, unique_str + "-Index.buf"), dtype=numpy.uint32)
            assert len(index_buffer) == len(source_indices)
            exported_triangles = sorted(
                tuple(sorted(row.tobytes() for row in triangle))
                for triangle in exported_position[index_buffer].reshape(-1, 3, 3)
            )
            source_triangles = sorted(
                tuple(sorted(row.tobytes() for row in triangle))
                for triangle in body["position"]["POSITION"][source_indices].reshape(-1, 3, 3)
            )
            assert exported_triangles == source_triangles

        # --- face: single-index blend buffer and float colours survive -------------------------
        assert _section(ini_text, "[TextureOverride_VB_%s_%s_Blend]" % (face_ib, face_ib))[-2] == "\tdraw = 29, 0"
        exported = _rows_by_position(
            read(face_ib, "Position", synthetic.POSITION_DTYPE),
            read(face_ib, "Texcoord", synthetic.FACE_TEXCOORD_DTYPE),
            read(face_ib, "Blend", synthetic.FACE_BLEND_DTYPE),
        )
        assert len(exported) == 29
        for index in range(29):
            _, texcoord, blend = exported[source["face"]["position"]["POSITION"][index].tobytes()]
            assert numpy.allclose(texcoord["COLOR"], source["face"]["texcoord"]["COLOR"][index], atol=1e-7)
            assert blend["BLENDINDICES"].tolist() == source["face"]["blend"]["BLENDINDICES"][index].tolist()
        face_section = _section(ini_text, "[TextureOverride_%s]" % face.replace("-", "_"))
        assert "Resource\\ZZMI\\Diffuse = ref Resource-" + face + "-DiffuseMap" in face_section

        # ------------------------------------------------------------------ export, outline on
        # A flat mesh has no silhouette crease: the recomputed outline data is zero everywhere,
        # while everything outside TEXCOORD1 stays as it was.
        props.zzz_outline_optimization = True
        _export(tree_name, os.path.join(temp_folder, "mod_outline"))
        meshes = os.path.join(temp_folder, "mod_outline", "Meshes")
        outlined = read(body_ib, "Texcoord", synthetic.BODY_TEXCOORD_DTYPE)
        assert not outlined["TEXCOORD1"].any()
        plain = numpy.fromfile(
            os.path.join(output_dir, "Meshes", body_ib + "-Texcoord.buf"), dtype=synthetic.BODY_TEXCOORD_DTYPE,
        )
        assert plain["TEXCOORD1"].any()
        for name in ("COLOR", "TEXCOORD", "TEXCOORD2", "TEXCOORD3"):
            assert outlined[name].tobytes() == plain[name].tobytes()
        props.zzz_outline_optimization = False

        # ------------------------------------------------------------------ SSMT4 workspaces
        # Without the extractor's metadata the original TheHerta4 layout is kept.
        for unique_str in (body_grid, body_quad, face):
            json_path = os.path.join(workspace_folder, unique_str, "TYPE_GPU-ZZMI", unique_str + ".json")
            with open(json_path, encoding="utf-8") as handle:
                submesh_json = json.load(handle)
            del submesh_json[FRAME_EXTRACT_KEY]
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(submesh_json, handle)
        legacy_ini = _export(tree_name, os.path.join(temp_folder, "mod_legacy"))
        _assert_section(legacy_ini, "[TextureOverride_VB_%s_%s_Position]" % (body_ib, body_ib), [
            "hash = " + synthetic.BODY["position"],
            "vb0 = Resource" + body_ib + "Position",
        ])
        _assert_section(legacy_ini, "[TextureOverride_VB_%s_%s_Blend]" % (body_ib, body_ib), [
            "hash = " + synthetic.BODY["blend"],
            "vb2 = Resource" + body_ib + "Blend",
            "handling = skip",
            "draw = 29, 0",
        ])
        assert "uav_byte_stride = 4" in legacy_ini
        assert "DRAW_TYPE" not in legacy_ini
        assert "match_index_count" not in legacy_ini

    print("BLENDER_ZZMI_EXPORT_REGRESSION=PASS")


if __name__ == "__main__":
    main()
