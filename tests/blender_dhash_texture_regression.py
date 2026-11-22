"""Original texture fingerprints survive extraction/marking and scope INI bindings.

Run with Blender --background --factory-startup --python-exit-code 1 --python FILE.
No user project, workspace or mod is modified.
"""

import json
import contextlib
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LoyalTools.common import texture_mark_helper
from LoyalTools.common.efmi_merged_texture import resolve_merged_auto_textures, copy_merged_auto_textures
from LoyalTools.common.m_ini_builder import M_IniBuilder
from LoyalTools.common.texture_metadata_helper import TextureMetadataResolver
from LoyalTools.efmi_extract.migoto_io.migoto_model.frame_model.api_calls.resource_calls import (
    ResourceSlotBinding, ConstantBufferSlotBinding, PSSetShaderResources,
)
from LoyalTools.efmi_extract.migoto_io.migoto_model.types import ResourceSlot, ShaderType, SlotType
from LoyalTools.extract.dump_workspace_extractor import DumpWorkspaceExtractor
from LoyalTools.ui.universal.efmi import ExportEFMI


def check_parser():
    for suffix, expected in (("", None), (" dhash=None", None),
                             (" dhash=0000000000000000", "0000000000000000"),
                             (" dhash=ac439e680ab342c5", "ac439e680ab342c5")):
        binding = ResourceSlotBinding.from_dumped_binding(SimpleNamespace(
            line="17: view=0x01 resource=0x02 hash=12345678" + suffix,
        ))
        resources = []
        PSSetShaderResources(bindings=[binding], raw_command=None, start_slot=17, num_views=1).execute(
            SimpleNamespace(set_resource=lambda slot, resource: resources.append(resource)), None,
        )
        assert resources[0].dhash == expected
        assert resources[0].hash == "12345678"
    constant = ConstantBufferSlotBinding.from_dumped_binding(SimpleNamespace(
        line="0: resource=0x03 hash=abcdef01 first_constant=10 num_constants=20",
    ))
    assert constant.first_constant == 10 and constant.dhash is None
    return resources[0]


def check_metadata(folder):
    source = folder / "original.dds"
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<II", header, 12, 512, 512)
    source.write_bytes(header + bytes(512 * 512))
    (folder / "log.txt").write_text("")
    extractor = DumpWorkspaceExtractor(str(folder))
    resources = {}
    formats = ["BC7_UNORM_SRGB", "BC7_UNORM", "BC5_UNORM"]
    for index, format_name in enumerate(formats):
        resource = check_parser()
        resource.hash = f"1234567{index}"
        resource.dhash = f"ac439e680ab342c{index}"
        resource.bin_path_deduped = source
        resource.data_descriptor = SimpleNamespace(data_format=format_name)
        resource.usage_descriptor = SimpleNamespace(call_id=12)
        resources[ResourceSlot(ShaderType.Pixel, SlotType.Texture, 16 + index)] = resource
    record = SimpleNamespace(shader_call=SimpleNamespace(
        id=42, model_resources=SimpleNamespace(textures=resources),
    ))
    normal_entries = extractor._collect_call_textures(record)
    merged_entries, merged_calls = extractor._collect_merged_component_textures(
        SimpleNamespace(textures={slot: [resource] for slot, resource in resources.items()}),
        "aaaaaaaa-6-0", [],
    )
    assert len(normal_entries) == len(merged_entries) == 3
    assert [entry.dhash for entry in normal_entries] == [entry.dhash for entry in merged_entries]
    bindings = None
    for merged, entries, calls in ((False, normal_entries, normal_entries),
                                    (True, merged_entries, merged_calls)):
        target = folder / ("merged" if merged else "normal")
        target.mkdir()
        build = SimpleNamespace(unique_str="aaaaaaaa-6-0", texture_entries=entries,
                                call_texture_entries=calls)
        extractor._write_textures(build, str(target), [], merged_component_scoped=merged)
        slots = json.loads((target / "TextureSlots.json").read_text())
        assert slots["slots"]["ps-t16"][0]["dhash"] == "ac439e680ab342c0"
        candidates = texture_mark_helper.list_candidate_textures(str(target))
        by_call = texture_mark_helper.list_candidate_textures_by_call(str(target))
        assert candidates[0]["dhash"] == next(iter(by_call.values()))[0]["dhash"]
        mark_file = target / "mesh.json"
        mark_file.write_text('{"TextureMarkUpInfoList": []}')
        marks = []
        for candidate, name in zip(candidates, ("DiffuseMap", "LightMap", "NormalMap")):
            filename = texture_mark_helper.make_named_mark_copy(
                str(target), candidate["filename"], build.unique_str, name,
            )
            mark = dict(MarkName=name, MarkType="Slot", MarkSlot=candidate["slot"],
                        MarkHash=candidate["hash"], MarkFileName=filename,
                        SourceDHash=candidate["dhash"], SourceFormat=candidate["format"])
            texture_mark_helper.write_mark(str(mark_file), mark)
            marks.append(mark)
        manual = TextureMetadataResolver.normalize_texture_markup_info_list(
            texture_mark_helper.read_marks(str(mark_file)),
        )
        assert [mark.source_dhash for mark in manual] == [entry.dhash for entry in entries]
        if merged:
            # Editing the named DDS must not change the original fingerprint or be overwritten.
            edited = target / "aaaaaaaa-6-0-DiffuseMap.dds"
            edited.write_bytes(b"user customized diffuse")
            bindings = resolve_merged_auto_textures(str(target), build.unique_str)
            assert [item.source_dhash for item in bindings] == [mark.source_dhash for mark in manual]
            assert bindings[0].source_path == str(edited)
            copy_merged_auto_textures(bindings, str(target))
            assert edited.read_bytes() == b"user customized diffuse"
    return bindings


def check_exports(props, textures, folder):
    props.use_dhash_texture = True
    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.dhash_texture_filters = {}
    lines = []
    for component in (0, 1):
        exporter._append_texture_binding_lines(lines, textures, component_id=component)
    assert lines[:3] == [
        "ps-t[Filter_Component0_Diffuse] = ref Resource-aaaaaaaa-6-0-DiffuseMap",
        "ps-t[Filter_Component0_Light] = ref Resource-aaaaaaaa-6-0-LightMap",
        "ps-t[Filter_Component0_Normal] = ref Resource-aaaaaaaa-6-0-NormalMap",
    ]
    assert "Filter_Component1_Diffuse" in lines[3]
    assert not any("LOD" in line for line in lines)
    exporter._append_texture_binding_lines([], textures, component_id=0)
    assert len(exporter.dhash_texture_filters) == 6
    builder = M_IniBuilder()
    exporter._append_dhash_texture_filters(builder)
    ini = folder / "filters.ini"
    builder.save_to_file(str(ini))
    output = ini.read_text()
    assert output.count("match_dhash = ac439e680ab342c0") == 2
    assert "match_format = BC7_UNORM_SRGB" in output
    old = TextureMetadataResolver.normalize_texture_markup_info_list([
        dict(MarkName="DiffuseMap", MarkType="Slot", MarkSlot="ps-t16", MarkFileName="old.dds"),
    ])
    try:
        exporter._append_texture_binding_lines([], old, component_id=7)
    except ValueError as exc:
        assert "Component7" in str(exc) and "old.dds" in str(exc)
    else:
        raise AssertionError("Missing original metadata must not produce a malformed Filter")
    props.use_dhash_texture = False
    lines = []
    exporter._append_texture_binding_lines(lines, old)
    assert lines == ["ps-t16 = Resource-old"]


def check_option(props):
    from blender_efmittl_slot_regression import _RecordingLayout
    from LoyalTools.blueprint.node_obj import SSMTNode_Result_Output
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName

    assert not props.use_dhash_texture
    options = ("use_rabbitfx_slot", "use_efmittl_slot", "use_dhash_texture")
    for selected in options * 2:
        setattr(props, selected, True)
        assert [getattr(props, key) for key in options] == [key == selected for key in options]
    props.use_dhash_texture = False
    GlobalConfig.logic_name = LogicName.EFMI
    tree = bpy.data.node_groups.new("dhash regression", "SSMTBlueprintTreeType")
    node = tree.nodes.new("SSMTNode_Result_Output")
    calls = []
    SSMTNode_Result_Output.draw_buttons(node, bpy.context, _RecordingLayout(calls))
    properties = [args[1] for name, args, _ in calls if name == "prop"]
    assert properties.index("use_dhash_texture") == properties.index("use_efmittl_slot") + 1
    bpy.data.node_groups.remove(tree)


def check_ini_scope(props, textures, folder):
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.m_ini_helper import M_IniHelper
    from LoyalTools.common.m_ini_helper_gui import M_IniHelperGUI

    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.merged_skeleton_mode = False
    exporter.has_cross_ib = False
    exporter.cross_ib_info_dict = {}
    exporter.cross_ib_target_info = {}
    exporter.cross_ib_match_mode = 'INDEX_COUNT'
    exporter.blueprint_model = SimpleNamespace(keyname_mkey_dict={})
    exporter._integrate_object_swap_ini_hook = lambda builder: None
    submesh = SimpleNamespace(unique_str='aaaaaaaa-6-0', match_draw_ib='aaaaaaaa',
                              match_first_index='0', match_index_count='6',
                              category_buffer_dict={}, drawcall_model_list=[])
    drawib = SimpleNamespace(draw_ib='aaaaaaaa', submesh_model_list=[submesh],
                             get_submesh_texture_markup_info_list=lambda model: textures)
    exporter.submesh_model_list = [submesh]
    exporter.drawib_model_list = [drawib]
    props.use_dhash_texture = True
    with contextlib.ExitStack() as stack:
        for owner, method, value in (
            (GlobalConfig, 'get_workspace_name', 'scope'),
            (GlobalConfig, 'path_generate_mod_folder', str(folder)),
            (M_IniHelper, 'generate_hash_style_texture_ini', None),
            (M_IniHelper, 'move_slot_style_textures', None),
            (M_IniHelper, 'get_drawindexed_instanced_str_list', ['drawindexed = 6, 0, 0']),
            (M_IniHelper, 'add_branch_key_sections', None),
            (M_IniHelperGUI, 'add_branch_mod_gui_section', None),
        ):
            stack.enter_context(patch.object(owner, method, return_value=value))
        exporter.generate_ini_file()
        text = (folder / 'scope.ini').read_text()
        assert text.index('SnapshotInitialTextures') < text.index('OverrideTextures')
        assert text.index('ps-t[Filter_Component0_Diffuse]') < text.index('drawindexed =')
        assert text.index('drawindexed =') < text.index('RestoreInitialTextures')
        assert text.count('[Filter_Component0_Diffuse]') == 2  # one use, one declaration
        props.use_dhash_texture = False
        exporter.generate_ini_file()
        text = (folder / 'scope.ini').read_text()
        assert 'Filter_Component' not in text and 'InitialTextures' not in text

    # Texture binding follows the entire LOD buffer selection, without its own branch.
    props.use_dhash_texture = True
    exporter.dhash_texture_filters = {}
    exporter._get_merged_profile_component = lambda unique: {'component_id': 3, 'lods': [{}]}
    exporter._append_merged_level_buffer_bindings = lambda section, submesh, level, indent='': section.append(
        indent + 'vb0 = Resource_LOD' + str(level))
    exporter._get_merged_texture_binding_list = lambda *args: textures
    lines = []
    exporter._append_merged_buffer_bindings(lines, submesh)
    exporter._append_merged_slot_texture_bindings(lines, submesh, drawib)
    assert lines.index('endif') < lines.index(
        'ps-t[Filter_Component3_Diffuse] = ref Resource-aaaaaaaa-6-0-DiffuseMap')
    assert sum('Filter_Component3_Diffuse' in line for line in lines) == 1
    props.use_dhash_texture = False


def main():
    bpy.ops.preferences.addon_enable(module="LoyalTools")
    props = bpy.context.scene.global_properties
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    check_parser()
    check_option(props)
    with tempfile.TemporaryDirectory(prefix="loyal-dhash-") as temp:
        folder = Path(temp)
        textures = check_metadata(folder)
        check_exports(props, textures, folder)
        check_ini_scope(props, textures, folder)
    # Existing modes keep their exact output and their material-node behavior.
    from blender_efmittl_slot_regression import _check_binding_lines, _check_material_node
    _check_binding_lines(props)
    _check_material_node()
    bpy.ops.preferences.addon_disable(module="LoyalTools")
    print("BLENDER_DHASH_TEXTURE_REGRESSION=PASS")


if __name__ == "__main__":
    main()
