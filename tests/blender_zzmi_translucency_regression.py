"""Run with Blender --background --factory-startup --python-exit-code 1 --python this_file.

Check TTLib's mask contract, per-draw state isolation, Cross IB offsets/materials,
and real extraction -> node -> export in ordinary and merged skinning modes.
"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bpy

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parents[1]))
sys.path.insert(0, str(TESTS))

from blender_translucency_node_regression import _drawcall, _translucency, _expect_error
import zzmi_merged_synthetic_dump as synthetic


def _export(tree_name, output_dir, frame_delay=False, same_frame=False):
    props = bpy.context.scene.global_properties
    props.generate_mod_folder_path = output_dir
    props.zzz_merged_frame_delay = frame_delay
    props.zzz_merged_same_frame = same_frame
    os.makedirs(output_dir, exist_ok=True)
    assert bpy.ops.ssmt.generate_mod_blueprint(blueprint_name=tree_name) == {'FINISHED'}
    files = list(Path(output_dir).glob("*.ini"))
    assert len(files) == 1, files
    return files[0].read_text(encoding="utf-8")


def _same_meshes(first, second):
    a, b = Path(first) / "Meshes", Path(second) / "Meshes"
    assert sorted(path.name for path in a.iterdir()) == sorted(path.name for path in b.iterdir())
    for path in a.iterdir():
        assert path.read_bytes() == (b / path.name).read_bytes(), path.name


def _section(text, header):
    return [line.strip() for line in text.split(header, 1)[1].split("\n[", 1)[0].splitlines()
            if line.strip() and not line.strip().startswith(";")]


def _check_api(root):
    from LoyalTools.ui.universal.zzmi_translucency import ZZMITranslucency
    from LoyalTools.common.m_ini_helper import M_IniHelper
    from LoyalTools.common.m_ini_builder import M_IniBuilder
    from LoyalTools.blueprint.node_postprocess_material import SSMTNode_PostProcess_Material

    helper = ZZMITranslucency()
    plain = _drawcall("aaaaaaaa-30-0", 5)
    assert helper.draw_lines([plain]) == M_IniHelper.get_drawindexed_str_list([plain])
    mask = root / "mask.dds"
    mask.write_bytes(b"mask fixture")
    masked = _drawcall("aaaaaaaa-60-0", 7, _translucency("TEXTURE", channel=2, texture=str(mask)))
    flat = _drawcall("aaaaaaaa-90-0", 67, _translucency(alpha=0), "$swapkey0")
    lines = helper.draw_lines([masked, plain, flat], {masked.obj_name: 117})
    assert r"$\TTL\_2 = 117" in lines
    assert r"$\TTL\mask_channel = 2" in lines
    assert r"$\TTL\mask_invert = 1" in lines and r"$\TTL\alpha = 1" in lines
    assert "  " + r"$\TTL\alpha = 0" in lines
    assert "if $swapkey0 == 1" in lines and "endif" in lines
    assert plain.get_drawindexed_str() in lines
    assert "  " + flat.get_drawindexed_str() not in lines
    for index, line in enumerate(lines):
        if line.strip() == r"run = CommandList\TTL\Draw":
            assert [item.strip() for item in lines[index+1:index+5]] == [
                r"Resource\TTL\TransparencyTex = null", r"$\TTL\_1 = 0",
                r"$\TTL\_2 = 0", r"$\TTL\_3 = 0",
            ]
    for line in lines:
        if "\\TTL\\" in line:
            assert not SSMTNode_PostProcess_Material._is_generated_material_line(line)
    # Shader equation with our invert=1, alpha=1, threshold=1: coverage = source.
    for sample in (0.0, 0.0001, 0.25, 0.5, 1.0):
        inverted = 1.0 - sample
        coverage = 0.0 if inverted >= 1 else (1.0 - inverted)
        assert abs(coverage - sample) < 1e-7
    helper.draw_lines([masked])
    builder = M_IniBuilder()
    helper.append_resources(builder)
    assert sum(line.startswith("[Resource-TTLMask-") for section in builder.ini_section_list
               for line in section.SectionLineList) == 1
    other = root / "other"
    other.mkdir()
    (other / mask.name).write_bytes(b"different")
    masked.translucency["texture"] = str(other / mask.name)
    assert "同名" in _expect_error(helper.draw_lines, [masked])
    masked.translucency["texture"] = str(root / "missing.dds")
    assert "找不到" in _expect_error(helper.draw_lines, [masked])
    masked.translucency["texture"] = ""
    assert "没有填" in _expect_error(helper.draw_lines, [masked])
    masked.translucency = _translucency("DIFFUSE_A", channel=3)
    assert "DiffuseMap" in _expect_error(helper.draw_lines, [masked])


def _check_cross_ib():
    from LoyalTools.ui.universal.zzmi import ExportZZMI
    from LoyalTools.ui.universal.zzmi_translucency import ZZMITranslucency
    from LoyalTools.common.m_ini_builder import M_IniBuilder

    source_call = _drawcall("aaaaaaaa-30-0", 7, _translucency("DIFFUSE_A", channel=3), "$swapkey1")
    target_call = _drawcall("bbbbbbbb-60-0", 0)

    def model(ib, call):
        submesh = SimpleNamespace(unique_str=call.obj_name, match_first_index=0,
                                  match_index_count=call.index_count, drawcall_model_list=[call])
        return SimpleNamespace(
            draw_ib=ib, submesh_model_list=[submesh], submesh_ib_dict={call.obj_name: [0, 1, 2]},
            obj_name_draw_offset={call.obj_name: 123},
            get_submesh_texture_override_suffix=lambda sm: sm.unique_str,
            get_submesh_ib_resource_name=lambda sm: "Resource" + ib + "Index",
            get_submesh_texture_markup_info_list=lambda sm: [SimpleNamespace(
                mark_type="Slot", mark_name="DiffuseMap", get_resource_name=lambda: "Resource" + ib + "Diffuse")],
        )

    exporter = ExportZZMI.__new__(ExportZZMI)
    exporter.translucency = ZZMITranslucency()
    exporter.cross_ib_info_dict = {"aaaaaaaa_0": ["bbbbbbbb_0"]}
    exporter.cross_ib_object_names = {source_call.obj_name}
    exporter.has_cross_ib = True
    exporter.drawib_model_list = [model("aaaaaaaa", source_call), model("bbbbbbbb", target_call)]
    exporter.merged_skeleton = SimpleNamespace(draw_begin_lines=lambda ib: [], draw_end_lines=lambda ib: [])
    builder = M_IniBuilder()
    for drawib in exporter.drawib_model_list:
        exporter.add_unity_vs_texture_override_ib_sections(builder, drawib)
    source, target = [[line.strip() for line in section.SectionLineList] for section in builder.ini_section_list]
    assert r"run = CommandList\TTL\Draw" not in source
    assert r"Resource\TTL\TransparencyTex = ref ResourcebbbbbbbbDiffuse" in target
    assert r"Resource\TTL\TransparencyTex = ref ResourceaaaaaaaaDiffuse" not in target
    assert r"$\TTL\_2 = 123" in target
    assert "if $swapkey1 == 1" in target


def _insert_node(tree, obj_name):
    source = next(node for node in tree.nodes if node.bl_idname == "SSMTNode_Object_Info" and node.object_name == obj_name)
    destinations = [link.to_socket for link in source.outputs[0].links]
    assert destinations
    for link in list(source.outputs[0].links):
        tree.links.remove(link)
    node = tree.nodes.new("SSMTNode_Translucency")
    tree.links.new(source.outputs[0], node.inputs[0])
    for destination in destinations:
        tree.links.new(node.outputs[0], destination)
    return node, source, destinations


def _remove_node(tree, wiring):
    node, source, destinations = wiring
    tree.nodes.remove(node)
    for destination in destinations:
        tree.links.new(source.outputs[0], destination)


def _check_exports(root, merged):
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.blueprint.model import BluePrintModel
    from LoyalTools.ui.universal.zzmi import ExportZZMI

    root.mkdir()
    dump = root / "dump"
    dump.mkdir()
    parts = synthetic.build_dump(str(dump))["components"]
    props = bpy.context.scene.global_properties
    props.workspace_source_mode = "CUSTOM"
    props.custom_workspace_folder_path = str(root / "workspace")
    props.force_standalone_preset = True
    props.standalone_game_preset = LogicName.ZZMI
    props.use_specific_generate_mod_folder_path = True
    props.open_mod_folder_after_generate_mod = False
    props.zzz_outline_optimization = False
    GlobalConfig.read_from_main_json_ssmt4()
    scene = bpy.context.scene
    scene.loyal_extract_props.frame_dump_folder = str(dump)
    scene.loyal_extract_props.workflow_mode = 'MERGED_SKELETON' if merged else 'STANDARD'
    if merged:
        assert bpy.ops.loyal.extract_from_dump() == {'FINISHED'}
    else:
        assert bpy.ops.loyal.extract_zzmi_whole_frame() == {'FINISHED'}
    tree = bpy.data.node_groups[props.selected_blueprint_name]
    body_name = parts["body"]["unique_str"]
    if merged:
        # Force foreign bones into the body so same-frame skinning really runs.
        hair = bpy.data.objects[parts["hair"]["unique_str"]]
        body = bpy.data.objects[body_name]
        copied = hair.copy()
        copied.data = hair.data.copy()
        scene.collection.objects.link(copied)
        bpy.ops.object.select_all(action='DESELECT')
        copied.select_set(True)
        body.select_set(True)
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.join()
    modes = [(False, False), (True, False), (False, True)] if merged else [(False, False)]
    for delay, same in modes:
        baseline = root / f"baseline_{delay}_{same}"
        baseline_ini = _export(tree.name, str(baseline), delay, same)
        wiring = _insert_node(tree, body_name)
        node = wiring[0]
        node.use_diffuse_alpha = True
        output = root / f"ttl_{delay}_{same}"
        ini = _export(tree.name, str(output), delay, same)
        assert r"run = CommandList\TTL\Draw" in ini and "EFMITTL" not in ini
        assert r"$\TTL\mask_channel = 3" in ini and r"$\TTL\mask_invert = 1" in ini
        assert r"Resource\TTL\TransparencyTex = ref Resource-" + body_name + "-DiffuseMap" in ini
        _same_meshes(str(baseline), str(output))
        body_section = _section(ini, "[TextureOverride_" + body_name.replace("-", "_") + "]")
        ttl_index = body_section.index(r"run = CommandList\TTL\Draw")
        if delay or same:
            before = [line for line in body_section[:ttl_index] if "ZZMS" in line]
            after = [line for line in body_section[ttl_index+1:] if "ZZMS" in line]
            assert before and after, body_section
        if not delay and not same:
            mask = root / "opacity.dds"
            mask.write_bytes(b"mask fixture")
            node.use_mask_r = True
            node.mask_texture_path = str(mask)
            textured = _export(tree.name, str(output))
            assert r"$\TTL\mask_channel = 0" in textured
            assert (output / "Textures" / mask.name).read_bytes() == mask.read_bytes()
            (output / "Textures" / mask.name).write_bytes(b"user edit")
            _export(tree.name, str(output))
            assert (output / "Textures" / mask.name).read_bytes() == b"user edit"
            node.use_mask_r = False
            node.alpha = 0.25
            flat = _export(tree.name, str(output))
            assert r"$\TTL\alpha = 0.25" in flat
        _remove_node(tree, wiring)
        assert _export(tree.name, str(root / f"restored_{delay}_{same}"), delay, same) == baseline_ini
    if merged:
        wiring = _insert_node(tree, parts["face"]["unique_str"])
        model = BluePrintModel(tree=tree, context=bpy.context)
        assert "形态键部件" in _expect_error(ExportZZMI, model)
        _remove_node(tree, wiring)
    # Avoid object-name collisions when the next fixture imports the same hashes.
    bpy.data.node_groups.remove(tree, do_unlink=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def main():
    bpy.ops.preferences.addon_enable(module="LoyalTools")
    with tempfile.TemporaryDirectory(prefix="loyal_zzmi_ttl_") as folder:
        root = Path(folder)
        _check_api(root)
        _check_cross_ib()
        _check_exports(root / "standard", False)
        _check_exports(root / "merged", True)
    print("BLENDER_ZZMI_TRANSLUCENCY_REGRESSION=PASS")


if __name__ == "__main__":
    main()
