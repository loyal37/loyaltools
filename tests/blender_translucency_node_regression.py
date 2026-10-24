# -*- coding: utf-8 -*-
"""Blender regression for the translucency (EFMI TTL) blueprint node.

The node's opacity or mask channel reaches every object behind it, the nearest
node wins, and the merged skeleton export replaces those objects' drawindexed
with EFMI TTL calls.  "A" uses the diffuse texture the component already
binds; "R", "G" and "B" use a mask texture from any folder, which the export
declares and copies to Textures without overwriting a file already there.  A
channel turns the opacity off.  The standard EFMI export refuses the node.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bpy

# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DIFFUSE = "Resource-aaaaaaaa-300-0-DiffuseMap"


def _object(name):
    obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(data)
    return path


def _read(path):
    with open(path, "rb") as file:
        return file.read()


def _expect_error(function, *args):
    try:
        function(*args)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected a ValueError")


def _translucency(mask="NONE", alpha=None, channel=None, texture="", node="n"):
    return {"alpha": alpha, "channel": channel, "texture": texture, "node": node, "mask": mask}


def _check_chains(root):
    from LoyalTools.blueprint.model import BluePrintModel

    names = ["aaaaaaaa-300-0", "bbbbbbbb-90-0", "cccccccc-60-0", "dddddddd-30-0"]
    for name in names:
        _object(name)

    tree = bpy.data.node_groups.new("ttl regression", "SSMTBlueprintTreeType")
    infos = []
    for name in names:
        info = tree.nodes.new("SSMTNode_Object_Info")
        info.object_name = name
        infos.append(info)
    near = tree.nodes.new("SSMTNode_Translucency")
    near.alpha = 0.3
    near.use_diffuse_alpha = True
    far = tree.nodes.new("SSMTNode_Translucency")
    far.alpha = 0.8
    textured = tree.nodes.new("SSMTNode_Translucency")
    textured.use_mask_g = True
    textured.mask_texture_path = os.path.join(root, "masks", "..", "masks", "skirt.dds")
    output = tree.nodes.new("SSMTNode_Result_Output")

    # a -> near -> far -> output, b -> far -> output, c -> textured -> output, d -> output.
    assert [socket.name for socket in far.inputs] == ["Input 1"]
    tree.links.new(infos[0].outputs[0], near.inputs[0])
    tree.links.new(near.outputs[0], far.inputs[0])
    far.update()
    tree.links.new(infos[1].outputs[0], far.inputs[1])
    far.update()
    assert [socket.name for socket in far.inputs] == ["Input 1", "Input 2", "Input 3"]
    tree.links.new(infos[2].outputs[0], textured.inputs[0])
    for source in (far, textured, infos[3]):
        tree.links.new(source.outputs[0], output.inputs[-1])
        output.update()

    model = BluePrintModel(tree=tree, context=bpy.context)
    found = {
        draw_call_model.obj_name: draw_call_model.translucency
        for draw_call_model in model.ordered_draw_obj_data_model_list
    }
    assert found == {
        names[0]: _translucency("DIFFUSE_A", channel=3, node=near.name),
        names[1]: _translucency(alpha=0.8, node=far.name),
        names[2]: _translucency(
            "TEXTURE", channel=1, node=textured.name,
            texture=os.path.join(root, "masks", "skirt.dds"),
        ),
        names[3]: None,
    }, found

    # A muted node passes the objects through untouched.
    for node in (near, far, textured):
        node.mute = True
    model = BluePrintModel(tree=tree, context=bpy.context)
    assert all(
        draw_call_model.translucency is None
        for draw_call_model in model.ordered_draw_obj_data_model_list
    )


def _drawcall(name, offset, translucency=None, condition=None):
    from LoyalTools.common.draw_call_model import DrawCallModel
    from LoyalTools.common.m_key import M_Key

    drawcall = DrawCallModel(obj_name=name)
    drawcall.index_count = int(name.split("-")[1])
    drawcall.vertex_count = 10
    drawcall.index_offset = offset
    drawcall.translucency = translucency
    if condition is not None:
        key = M_Key()
        key.key_name = condition
        key.tmp_value = 1
        drawcall.work_key_list = [key]
    return drawcall


def _new_exporter():
    from LoyalTools.ui.universal.efmi import ExportEFMI

    exporter = object.__new__(ExportEFMI)
    exporter.merged_translucent_draw_names = []
    exporter.merged_translucency_mask_textures = {}
    return exporter


def _check_ini_lines():
    from LoyalTools.common.m_ini_helper import M_IniHelper

    exporter = _new_exporter()

    def lines(drawcalls, diffuse=DIFFUSE):
        section = []
        exporter._append_merged_draw_lines(section, drawcalls, diffuse)
        return section

    # Without the node the output is exactly what it was before.
    plain = [_drawcall("aaaaaaaa-300-0", 0), _drawcall("aaaaaaaa-90-0", 300, condition="$swapkey0")]
    assert lines(plain) == M_IniHelper.get_drawindexed_str_list(plain)
    assert exporter.merged_translucent_draw_names == []

    masked = _translucency("DIFFUSE_A", channel=3)
    flat = _translucency(alpha=0.25)
    drawcalls = [
        _drawcall("aaaaaaaa-300-0", 0, masked),
        _drawcall("aaaaaaaa-90-0", 300, flat, condition="$swapkey0"),
        _drawcall("aaaaaaaa-60-0", 390),
        _drawcall("aaaaaaaa-30-0", 450, _translucency(alpha=0.0)),
    ]
    assert lines(drawcalls) == [
        "; [mesh:aaaaaaaa-300-0] [vertex_count:10]",
        "Resource\\EFMITTL\\Mask = ref " + DIFFUSE,
        "$\\EFMITTL\\mask_channel = 3",
        "$\\EFMITTL\\count = 300",
        "$\\EFMITTL\\first = 0",
        "run = CommandList\\EFMITTL\\Draw",
        "; EFMI TTL: drawindexed = 300,0,0",
        "; [mesh:aaaaaaaa-60-0] [vertex_count:10]",
        "drawindexed = 60,390,0",
        "; [mesh:aaaaaaaa-30-0] [vertex_count:10]",
        "$\\EFMITTL\\alpha = 0",
        "$\\EFMITTL\\count = 30",
        "$\\EFMITTL\\first = 450",
        "run = CommandList\\EFMITTL\\Draw",
        "; EFMI TTL: drawindexed = 30,450,0",
        "",
        "if $swapkey0 == 1",
        "  ; [mesh:aaaaaaaa-90-0] [vertex_count:10]",
        "  $\\EFMITTL\\alpha = 0.25",
        "  $\\EFMITTL\\count = 90",
        "  $\\EFMITTL\\first = 300",
        "  run = CommandList\\EFMITTL\\Draw",
        "  ; EFMI TTL: drawindexed = 90,300,0",
        "endif",
        "",
    ], lines(drawcalls)
    assert exporter.merged_translucent_draw_names == [
        "aaaaaaaa-300-0", "aaaaaaaa-30-0", "aaaaaaaa-90-0",
    ]

    # The A channel needs a diffuse texture bound by the component.
    message = _expect_error(lines, [_drawcall("aaaaaaaa-300-0", 0, masked)], "")
    assert "A 通道" in message and "aaaaaaaa-300-0" in message, message
    # Opacity alone does not.
    assert "run = CommandList\\EFMITTL\\Draw" in lines([_drawcall("aaaaaaaa-90-0", 0, flat)], "")


def _check_mask_textures(root):
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.m_ini_builder import M_IniBuilder

    exporter = _new_exporter()

    def lines(drawcalls):
        section = []
        exporter._append_merged_draw_lines(section, drawcalls, DIFFUSE)
        return section

    skirt = _write(os.path.join(root, "anywhere", "skirt mask.dds"), b"skirt")
    veil = _write(os.path.join(root, "elsewhere", "veil.dds"), b"veil")
    green = _translucency("TEXTURE", channel=1, texture=skirt)
    blue = _translucency("TEXTURE", channel=2, texture=skirt)
    red = _translucency("TEXTURE", channel=0, texture=veil)

    # One resource per texture, whichever channel each part reads.
    assert lines([
        _drawcall("aaaaaaaa-300-0", 0, green),
        _drawcall("aaaaaaaa-90-0", 300, blue),
        _drawcall("aaaaaaaa-60-0", 390, red),
    ]) == [
        "; [mesh:aaaaaaaa-300-0] [vertex_count:10]",
        "Resource\\EFMITTL\\Mask = ref Resource-TTLMask-skirt_mask",
        "$\\EFMITTL\\mask_channel = 1",
        "$\\EFMITTL\\count = 300",
        "$\\EFMITTL\\first = 0",
        "run = CommandList\\EFMITTL\\Draw",
        "; EFMI TTL: drawindexed = 300,0,0",
        "; [mesh:aaaaaaaa-90-0] [vertex_count:10]",
        "Resource\\EFMITTL\\Mask = ref Resource-TTLMask-skirt_mask",
        "$\\EFMITTL\\mask_channel = 2",
        "$\\EFMITTL\\count = 90",
        "$\\EFMITTL\\first = 300",
        "run = CommandList\\EFMITTL\\Draw",
        "; EFMI TTL: drawindexed = 90,300,0",
        "; [mesh:aaaaaaaa-60-0] [vertex_count:10]",
        "Resource\\EFMITTL\\Mask = ref Resource-TTLMask-veil",
        "$\\EFMITTL\\mask_channel = 0",
        "$\\EFMITTL\\count = 60",
        "$\\EFMITTL\\first = 390",
        "run = CommandList\\EFMITTL\\Draw",
        "; EFMI TTL: drawindexed = 60,390,0",
        "",
    ]

    builder = M_IniBuilder()
    exporter._append_merged_translucency_mask_resources(builder)
    assert len(builder.ini_section_list) == 1
    assert builder.ini_section_list[0].SectionLineList == [
        "[Resource-TTLMask-skirt_mask]", "filename = Textures/skirt mask.dds", "",
        "[Resource-TTLMask-veil]", "filename = Textures/veil.dds", "",
    ], builder.ini_section_list[0].SectionLineList

    # Copied to Textures; a file already there is kept as it is.
    textures = os.path.join(root, "mod", "Textures")
    _write(os.path.join(textures, "veil.dds"), b"edited in the mod")
    original = GlobalConfig.path_generatemod_texture_folder
    GlobalConfig.path_generatemod_texture_folder = classmethod(lambda cls, draw_ib: textures)
    try:
        exporter._copy_merged_translucency_mask_textures()
    finally:
        GlobalConfig.path_generatemod_texture_folder = original
    assert _read(os.path.join(textures, "skirt mask.dds")) == b"skirt"
    assert _read(os.path.join(textures, "veil.dds")) == b"edited in the mod"
    assert _read(veil) == b"veil"

    # Nothing to declare or copy without a texture channel.
    empty = _new_exporter()
    builder = M_IniBuilder()
    empty._append_merged_translucency_mask_resources(builder)
    assert builder.ini_section_list == []
    empty._copy_merged_translucency_mask_textures()

    # A channel without a texture, a missing file, and two files of one name.
    message = _expect_error(lines, [_drawcall("aaaaaaaa-30-0", 0, _translucency("TEXTURE", channel=2))])
    assert "B 通道" in message and "aaaaaaaa-30-0" in message, message
    missing = os.path.join(root, "nowhere", "gone.dds")
    message = _expect_error(lines, [_drawcall("aaaaaaaa-30-0", 0, _translucency("TEXTURE", channel=0, texture=missing))])
    assert missing in message, message
    other_veil = _write(os.path.join(root, "other", "VEIL.dds"), b"another veil")
    message = _expect_error(lines, [_drawcall("aaaaaaaa-30-0", 0, _translucency("TEXTURE", channel=0, texture=other_veil))])
    assert "同名" in message and other_veil in message, message


class _RecordingLayout:
    """Stands in for UILayout so draw code can run headless."""

    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs, self))
            return _RecordingLayout(self.calls)
        return call


def _recorded_calls(draw, *args):
    calls = []
    draw(_RecordingLayout(calls), *args)
    functions = bpy.types.UILayout.bl_rna.functions
    icons = set(functions["operator"].parameters["icon"].enum_items.keys())
    for name, _args, kwargs, _layout in calls:
        if name in functions:
            assert set(kwargs) <= set(functions[name].parameters.keys()), (name, kwargs)
        assert kwargs.get("icon", 'NONE') in icons, (name, kwargs)
    return calls


def _check_ui():
    from LoyalTools.blueprint.node_menu import SSMT_MT_NodeMenu_Object
    from LoyalTools.blueprint.node_translucency import SSMTNode_Translucency

    calls = _recorded_calls(
        lambda layout: SSMT_MT_NodeMenu_Object.draw(SimpleNamespace(layout=layout), bpy.context)
    )
    entries = [kwargs for name, _args, kwargs, _layout in calls if name == "operator"]
    assert {"text": "半透明", "icon": 'MOD_OPACITY'} in entries, entries
    assert SSMTNode_Translucency.bl_icon == 'MOD_OPACITY'

    node = bpy.data.node_groups["ttl regression"].nodes.new("SSMTNode_Translucency")
    channels = ("use_mask_r", "use_mask_g", "use_mask_b", "use_diffuse_alpha")
    assert (node.alpha, node.mask_texture_path) == (0.5, "")
    assert [node.bl_rna.properties[prop].name for prop in channels] == ["R", "G", "B", "A"]
    assert node.bl_rna.properties["mask_texture_path"].subtype == 'FILE_PATH'

    def selected():
        return [prop for prop in channels if getattr(node, prop)]

    def drawn():
        calls = _recorded_calls(lambda layout: node.draw_buttons(bpy.context, layout))
        props = [
            (args[1:], kwargs)
            for name, args, kwargs, _layout in calls if name in ("prop", "label")
        ]
        alpha_layout = next(layout for name, args, _kwargs, layout in calls if args[1:] == ("alpha",))
        return props, getattr(alpha_layout, "enabled", True)

    # Opacity, the four channel toggles, and the texture path only for R/G/B.
    toggles = [((prop,), {"toggle": True}) for prop in channels]
    assert drawn() == ([(("alpha",), {})] + toggles, True)

    # One channel at a time; clicking the selected one again turns it off.
    node.use_mask_r = True
    assert selected() == ["use_mask_r"]
    path_field = [(("mask_texture_path",), {"text": ""})]
    assert drawn() == ([(("alpha",), {})] + toggles + path_field, False)
    node.use_mask_b = True
    assert selected() == ["use_mask_b"]
    assert drawn() == ([(("alpha",), {})] + toggles + path_field, False)
    node.use_diffuse_alpha = True
    assert selected() == ["use_diffuse_alpha"]
    assert drawn() == ([(("alpha",), {})] + toggles, False)
    node.use_diffuse_alpha = False
    assert selected() == []
    assert drawn() == ([(("alpha",), {})] + toggles, True)


def _check_standard_export_refuses():
    from LoyalTools.ui.universal.efmi import ExportEFMI

    model = SimpleNamespace(
        cross_ib_info_dict={}, cross_ib_method_dict={}, has_cross_ib=False,
        cross_ib_mapping_objects={}, cross_ib_vb_condition_mapping={},
        cross_ib_source_to_target_dict={}, cross_ib_object_vb_condition={},
        cross_ib_target_info={}, cross_ib_match_mode="INDEX_COUNT",
        cross_ib_object_names=set(), _tree=None,
        ordered_draw_obj_data_model_list=[
            _drawcall("dddddddd-30-0", 0, _translucency(alpha=0.5)),
        ],
    )
    message = _expect_error(ExportEFMI, model)
    assert message == "半透明节点只支持 EFMI 骨骼合并导出。", message


def main():
    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    with tempfile.TemporaryDirectory(prefix="loyal-ttl-") as root:
        root = os.path.realpath(root)
        _check_chains(root)
        _check_ui()
        _check_ini_lines()
        _check_mask_textures(root)
        _check_standard_export_refuses()
    print("BLENDER_TRANSLUCENCY_NODE_REGRESSION=PASS")


if __name__ == "__main__":
    main()
