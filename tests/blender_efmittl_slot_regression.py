# -*- coding: utf-8 -*-
"""Blender regression for the "EFMITTL动态槽位" option of the Generate Mod node.

The option sits under "使用RabbitFX槽位风格", and only one of the two can be on.
With it, the diffuse, light map and normal map go to the EFMI TTL library
(Resource\\EFMITTL\\... and run = CommandList\\EFMITTL\\SetTextures), and other
textures keep their fixed slots, as with RabbitFX.  The slot and RabbitFX output
stays what it was.  The material post-process node treats the EFMITTL lines like
RabbitFX's and leaves the translucency node's lines alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import bpy

# Prefer this repository over any older LoyalTools copy installed in Blender.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _texture(name, slot, mark_type="Slot"):
    return SimpleNamespace(
        mark_type=mark_type,
        mark_name=name,
        mark_slot=slot,
        get_resource_name=lambda: "Resource-aaaaaaaa-300-0-" + name,
    )


TEXTURES = [
    _texture("DiffuseMap", "ps-t16"),
    _texture("LightMap", "ps-t17"),
    _texture("NormalMap", "ps-t18"),
    _texture("MaterialMap", "t20"),
    _texture("HashOnly", "", mark_type="Hash"),
]


class _RecordingLayout:
    """Stands in for UILayout so draw code can run headless."""

    def __init__(self, calls):
        self.calls = calls

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return _RecordingLayout(self.calls)
        return call


def _check_option(props):
    from LoyalTools.blueprint.node_obj import SSMTNode_Result_Output
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName

    assert props.bl_rna.properties["use_efmittl_slot"].name == "EFMITTL动态槽位"
    assert (props.use_rabbitfx_slot, props.use_efmittl_slot) == (False, False)

    GlobalConfig.logic_name = LogicName.EFMI
    tree = bpy.data.node_groups.new("efmittl slot regression", "SSMTBlueprintTreeType")
    node = tree.nodes.new("SSMTNode_Result_Output")
    calls = []
    SSMTNode_Result_Output.draw_buttons(node, bpy.context, _RecordingLayout(calls))
    drawn = [args[1] for name, args, _kwargs in calls if name == "prop"]
    index = drawn.index("use_rabbitfx_slot")
    assert drawn[index + 1] == "use_efmittl_slot", drawn

    # One of the two at a time; both can be off.
    props.use_rabbitfx_slot = True
    props.use_efmittl_slot = True
    assert (props.use_rabbitfx_slot, props.use_efmittl_slot) == (False, True)
    props.use_rabbitfx_slot = True
    assert (props.use_rabbitfx_slot, props.use_efmittl_slot) == (True, False)
    props.use_rabbitfx_slot = False
    assert (props.use_rabbitfx_slot, props.use_efmittl_slot) == (False, False)


def _check_binding_lines(props):
    from LoyalTools.ui.universal.efmi import ExportEFMI

    exporter = object.__new__(ExportEFMI)

    def lines(textures, indent=""):
        section = []
        exporter._append_texture_binding_lines(section, textures, indent=indent)
        return section

    props.use_rabbitfx_slot = False
    props.use_efmittl_slot = False
    assert lines(TEXTURES) == [
        "ps-t16 = Resource-aaaaaaaa-300-0-DiffuseMap",
        "ps-t17 = Resource-aaaaaaaa-300-0-LightMap",
        "ps-t18 = Resource-aaaaaaaa-300-0-NormalMap",
        "t20 = Resource-aaaaaaaa-300-0-MaterialMap",
    ], lines(TEXTURES)

    props.use_rabbitfx_slot = True
    assert lines(TEXTURES) == [
        "Resource\\RabbitFx\\Diffuse = ref Resource-aaaaaaaa-300-0-DiffuseMap",
        "Resource\\RabbitFx\\LightMap = ref Resource-aaaaaaaa-300-0-LightMap",
        "Resource\\RabbitFx\\NormalMap = ref Resource-aaaaaaaa-300-0-NormalMap",
        "run = CommandList\\RabbitFx\\SetTextures",
        "ps-t20 = Resource-aaaaaaaa-300-0-MaterialMap",
    ], lines(TEXTURES)
    # RabbitFX keeps calling SetTextures even without its three textures.
    assert lines(TEXTURES[3:]) == [
        "run = CommandList\\RabbitFx\\SetTextures",
        "ps-t20 = Resource-aaaaaaaa-300-0-MaterialMap",
    ]

    props.use_efmittl_slot = True
    assert not props.use_rabbitfx_slot
    assert lines(TEXTURES, indent="    ") == [
        "    Resource\\EFMITTL\\Diffuse = ref Resource-aaaaaaaa-300-0-DiffuseMap",
        "    Resource\\EFMITTL\\LightMap = ref Resource-aaaaaaaa-300-0-LightMap",
        "    Resource\\EFMITTL\\NormalMap = ref Resource-aaaaaaaa-300-0-NormalMap",
        "    run = CommandList\\EFMITTL\\SetTextures",
        "    ps-t20 = Resource-aaaaaaaa-300-0-MaterialMap",
    ], lines(TEXTURES, indent="    ")
    # A diffuse alone is passed alone; nothing to pass, no call.
    assert lines([TEXTURES[0]]) == [
        "Resource\\EFMITTL\\Diffuse = ref Resource-aaaaaaaa-300-0-DiffuseMap",
        "run = CommandList\\EFMITTL\\SetTextures",
    ]
    assert lines(TEXTURES[3:]) == ["ps-t20 = Resource-aaaaaaaa-300-0-MaterialMap"]
    assert lines([]) == []
    props.use_efmittl_slot = False


def _check_material_node():
    from LoyalTools.blueprint.node_postprocess_material import SSMTNode_PostProcess_Material

    generated = SSMTNode_PostProcess_Material._is_generated_material_line
    for line in (
        "Resource\\EFMITTL\\Diffuse = ref Resource-x-DiffuseMap",
        "Resource\\EFMITTL\\LightMap = ref Resource-x-LightMap",
        "Resource\\EFMITTL\\NormalMap = ref Resource-x-NormalMap",
        "run = CommandList\\EFMITTL\\SetTextures",
        "Resource\\RabbitFx\\Diffuse = ref Resource-x-DiffuseMap",
        "run = CommandList\\RabbitFx\\SetTextures",
    ):
        assert generated(line), line
    # The translucency node's lines are not material lines.
    for line in (
        "Resource\\EFMITTL\\Mask = ref Resource-x-DiffuseMap",
        "$\\EFMITTL\\mask_channel = 3",
        "$\\EFMITTL\\alpha = 0.5",
        "run = CommandList\\EFMITTL\\Draw",
    ):
        assert not generated(line), line


def main():
    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")
    props = bpy.context.scene.global_properties

    _check_option(props)
    _check_binding_lines(props)
    _check_material_node()
    print("BLENDER_EFMITTL_SLOT_REGRESSION=PASS")


if __name__ == "__main__":
    main()
