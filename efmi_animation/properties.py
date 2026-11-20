"""Persistent authoring records for EFMI animation bones."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)


def _enabled_update(settings, context):
    from . import rig
    if not settings.enabled and rig.has_linked_physics_chains(settings.id_data):
        settings['enabled'] = True
        settings.disable_blocked = True
        return
    settings.disable_blocked = False
    rig.enabled_changed(settings)


class EFMIAnimationBone(bpy.types.PropertyGroup):
    uid: StringProperty()
    parent_uid: StringProperty()
    bone_id: IntProperty(max=0)
    group_name: StringProperty()


class EFMIAnimationRoot(bpy.types.PropertyGroup):
    uid: StringProperty()
    name: StringProperty(name="Animation Branch Name", options=set())
    follow_group_id: IntProperty(min=0, max=65535)
    frame_start: IntProperty(name="Start Frame", default=1, options=set())
    frame_end: IntProperty(name="End Frame", default=60, options=set())
    playback_speed: FloatProperty(name="Playback Speed", default=1.0, min=0.001,
        soft_min=0.1, soft_max=4.0, options=set(),
        description="In-game loop speed multiplier. A value of 1 uses the Blender scene frame rate.")
    bones: CollectionProperty(type=EFMIAnimationBone)
    active_bone_index: IntProperty(min=0)


class EFMIAnimationObject(bpy.types.PropertyGroup):
    uid: StringProperty()
    object: PointerProperty(type=bpy.types.Object)
    rig: PointerProperty(type=bpy.types.Object)
    preview_modifier_uid: IntProperty()
    roots: CollectionProperty(type=EFMIAnimationRoot)
    active_root_index: IntProperty(min=0)


class EFMIAnimationSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="Enable EFMI Bone Animation", default=False, options=set(),
        update=_enabled_update,
        description="Enable animation authoring, Blender previews, and export to the game")
    disable_blocked: BoolProperty(default=False, options={'SKIP_SAVE'})
    objects: CollectionProperty(type=EFMIAnimationObject)
    active_object_index: IntProperty(min=0)
    next_bone_id: IntProperty(default=-1, max=-1)


CLASSES = (EFMIAnimationBone, EFMIAnimationRoot, EFMIAnimationObject, EFMIAnimationSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.loyaltools_efmi_animation = PointerProperty(type=EFMIAnimationSettings)


def unregister():
    del bpy.types.Scene.loyaltools_efmi_animation
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
