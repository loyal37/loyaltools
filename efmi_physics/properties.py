"""Blend-file persistent physics authoring records; identities never use names."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)


def _enabled_update(self, context):
    from . import rig
    rig.enabled_changed(self.id_data)


class EFMIPhysicsSegment(bpy.types.PropertyGroup):
    uid: StringProperty()
    bone_id: IntProperty(min=0, max=2147483647)
    bone_name: StringProperty()
    group_name: StringProperty()


class EFMIPhysicsChain(bpy.types.PropertyGroup):
    uid: StringProperty()
    name: StringProperty(name="物理链", default="物理链")
    follow_group_id: IntProperty(name="原骨编号", min=0, max=65535,
        description="跟随的原合并顶点组名称，必须小于原骨数量 B，且物体上已存在")
    segments: CollectionProperty(type=EFMIPhysicsSegment)
    active_segment_index: IntProperty(min=0)
    damping: FloatProperty(name="阻尼", default=0.08, min=0.0, max=1.0)
    stiffness: FloatProperty(name="回正强度", default=0.18, min=0.0, max=1.0)
    gravity: FloatVectorProperty(name="模型空间重力", default=(0.0, -9.8, 0.0), size=3,
        description="游戏蒙皮后的模型空间加速度，默认 -Y；这不是世界空间重力")
    cone_angle: FloatProperty(name="最大摆角 (°)", default=45.0, min=0.0, max=180.0)
    wind_strength: FloatProperty(name="微风强度", default=0.0, min=0.0, max=10.0)


class EFMIPhysicsObject(bpy.types.PropertyGroup):
    uid: StringProperty()
    object: PointerProperty(type=bpy.types.Object)
    rig: PointerProperty(type=bpy.types.Object)
    chains: CollectionProperty(type=EFMIPhysicsChain)
    active_chain_index: IntProperty(min=0)
    schema_version: IntProperty(default=0)
    created_base_bone_count: IntProperty(default=0, min=0)
    created_workspace_key: StringProperty()
    created_profile_key: StringProperty()
    native_group_names: StringProperty(default="[]")
    owned_group_ids: StringProperty(default="[]")


class EFMIPhysicsSettings(bpy.types.PropertyGroup):
    enabled: BoolProperty(name="启用 EFMI 物理模块", default=False, update=_enabled_update,
        description="物理制作、每秒自动同步和导出的总开关；关闭保留骨架与权重，所有物理操作停止")
    objects: CollectionProperty(type=EFMIPhysicsObject)
    active_object_index: IntProperty(min=0)
    next_bone_id: IntProperty(default=65536, min=0, max=2147483647)
    base_bone_count: IntProperty(default=0, min=0)
    workspace_key: StringProperty()
    status: StringProperty(default="")


CLASSES = (EFMIPhysicsSegment, EFMIPhysicsChain, EFMIPhysicsObject, EFMIPhysicsSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.loyaltools_efmi_physics = PointerProperty(type=EFMIPhysicsSettings)


def unregister():
    del bpy.types.Scene.loyaltools_efmi_physics
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
