"""Undoable EFMI-only UI commands. Changes occur only on explicit commands."""

import bpy
from bpy.props import EnumProperty

from . import rig

_FOLLOW_ITEMS = []  # Retain strings for Blender's dynamic enum lifetime.


def _follow_items(self, context):
    global _FOLLOW_ITEMS
    entry = rig.active_entry(context.scene) if context and context.scene else None
    base = context.scene.loyaltools_efmi_physics.base_bone_count if entry else 0
    values = rig.original_groups(entry.object, base) if entry else []
    _FOLLOW_ITEMS = [(str(value), "原骨 " + str(value), "跟随原合并顶点组 " + str(value))
                     for value in values]
    return _FOLLOW_ITEMS


class PhysicsCommand:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and context.scene.loyaltools_efmi_physics.enabled
                    and rig.is_merged_workspace())

    def execute(self, context):
        try:
            self.run(context)
            if self.bl_idname not in {"efmi.physics_sync", "efmi.physics_validate"}:
                context.scene.loyaltools_efmi_physics.status = ""
            return {'FINISHED'}
        except (ValueError, RuntimeError, KeyError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class EFMI_OT_PhysicsAddObject(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_add_object"
    bl_label = "添加选中物体"
    bl_description = "添加前请先刷好原合并骨骼权重；创建独立可编辑物理骨架"

    @classmethod
    def poll(cls, context):
        return bool(super().poll(context) and context.mode == 'OBJECT'
                    and context.active_object and context.active_object.type == 'MESH'
                    and context.active_object.get(rig.OBJECT_MARKER_MERGED_SKELETON))

    def run(self, context):
        if context.mode != 'OBJECT':
            raise ValueError("请先退出编辑模式再添加网格物体。")
        rig.add_object(context, context.active_object, rig.current_profile())


class EFMI_OT_PhysicsRemoveObject(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_remove_object"
    bl_label = "移除物体"
    bl_description = "移除该物体的物理骨架及本模块拥有的顶点组和权重；保留网格与原组"

    def run(self, context):
        rig.remove_object(context, context.scene.loyaltools_efmi_physics.active_object_index)


class EFMI_OT_PhysicsAddChain(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_add_chain"
    bl_label = "添加物理链"
    bl_description = "创建一个父骨及其 EFMI_PHYS_ 专属顶点组；原数字组保持不变"

    def run(self, context):
        entry = rig.active_entry(context.scene)
        if not entry:
            raise ValueError("请先添加网格物体。")
        rig.add_chain(context, entry, rig.current_profile())


class EFMI_OT_PhysicsRemoveChain(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_remove_chain"
    bl_label = "删除物理链"
    bl_description = "删除整条链的骨骼、所属物理专属组及权重；原组与其他链保持不变"

    def run(self, context):
        entry = rig.active_entry(context.scene)
        if not entry:
            raise ValueError("请先选择登记物体。")
        rig.remove_chain(context, entry, entry.active_chain_index)


class EFMI_OT_PhysicsAddSegment(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_add_segment"
    bl_label = "追加子骨"
    bl_description = "在链末尾追加真实子骨并分配新的 EFMI_PHYS_ 专属顶点组"

    def run(self, context):
        entry = rig.active_entry(context.scene)
        if not entry:
            raise ValueError("请先选择登记物体。")
        rig.add_segment(context, entry, rig.active_chain(entry), rig.current_profile())


class EFMI_OT_PhysicsRemoveSegment(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_remove_segment"
    bl_label = "删除选中及后续子骨"
    bl_description = "删除选中子骨及后续段，清理其顶点组和权重；父骨请通过删除整链移除"

    def run(self, context):
        entry = rig.active_entry(context.scene)
        chain = rig.active_chain(entry)
        if not chain:
            raise ValueError("请先选择物理链。")
        rig.remove_segment(context, entry, chain)


class EFMI_OT_PhysicsFollowGroup(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_follow_group"
    bl_label = "选择原骨顶点组"
    bl_description = "只列出此物体上编号在原骨范围 B 内的合并顶点组"
    bl_property = "group_id"

    group_id: EnumProperty(name="原合并顶点组", items=_follow_items)

    def invoke(self, context, event):
        entry = rig.active_entry(context.scene)
        if not rig.active_chain(entry):
            self.report({'ERROR'}, "请先选择物理链。")
            return {'CANCELLED'}
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def run(self, context):
        entry = rig.active_entry(context.scene)
        chain = rig.active_chain(entry)
        if chain is None or not self.group_id:
            raise ValueError("请选择有效原骨。")
        profile = rig.current_profile()
        base = rig._check_workspace(context.scene.loyaltools_efmi_physics, profile)
        group_id = int(self.group_id)
        if group_id not in rig.original_groups(entry.object, base):
            raise ValueError("跟随顶点组必须在原骨范围 B 内。")
        chain.follow_group_id = group_id


class EFMI_OT_PhysicsEditRig(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_edit_rig"
    bl_label = "编辑骨骼位置 / 长度 / Roll"
    bl_description = "进入真实骨架编辑模式；父骨头部决定物理挂点，子骨保持线性连接"

    def run(self, context):
        entry = rig.active_entry(context.scene)
        if not entry:
            raise ValueError("请先选择登记物体。")
        rig._check_entry(entry)
        chain = rig.active_chain(entry)
        if not chain or not chain.segments:
            raise ValueError("请先添加物理链。")
        segment = chain.segments[min(chain.active_segment_index, len(chain.segments) - 1)]
        bone = rig.find_bone(entry.rig, segment)
        if bone is None:
            raise ValueError("骨骼已删除，请先同步检查。")
        name = bone.name
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        entry.rig.hide_viewport = False
        entry.rig.hide_set(False)
        entry.rig.select_set(True)
        context.view_layer.objects.active = entry.rig
        bpy.ops.object.mode_set(mode='EDIT')
        for edit_bone in entry.rig.data.edit_bones:
            edit_bone.select = edit_bone.select_head = edit_bone.select_tail = False
        edit_bone = entry.rig.data.edit_bones[name]
        entry.rig.data.edit_bones.active = edit_bone
        edit_bone.select = edit_bone.select_head = edit_bone.select_tail = True


class EFMI_OT_PhysicsSelectMesh(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_select_mesh"
    bl_label = "选择网格与物理顶点组"
    bl_description = "选择网格并激活当前物理骨的 EFMI_PHYS_ 专属顶点组，使用 Blender 原生权重绘制"

    def run(self, context):
        entry = rig.active_entry(context.scene)
        if not entry or not entry.object:
            raise ValueError("网格已删除，请先同步检查。")
        if entry.object.name not in context.view_layer.objects:
            raise ValueError("网格不在当前视图层。")
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        entry.object.hide_viewport = False
        entry.object.hide_set(False)
        entry.object.select_set(True)
        context.view_layer.objects.active = entry.object
        chain = rig.active_chain(entry)
        if chain and chain.segments:
            segment = chain.segments[min(chain.active_segment_index, len(chain.segments) - 1)]
            group = entry.object.vertex_groups.get(rig._segment_group(segment))
            if group:
                entry.object.vertex_groups.active_index = group.index


class EFMI_OT_PhysicsSync(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_sync"
    bl_label = "同步检查"
    bl_description = "清理已删除物体/骨架/骨段的登记与所属权重，保留其余骨段和权重"

    def run(self, context):
        rig.synchronize(context)
        self.report({'INFO'}, context.scene.loyaltools_efmi_physics.status)


class EFMI_OT_PhysicsValidate(PhysicsCommand, bpy.types.Operator):
    bl_idname = "efmi.physics_validate"
    bl_label = "检查导出配置"
    bl_description = "检查全局编号、所属顶点组、原骨挂点和线性父子结构"

    def run(self, context):
        settings = context.scene.loyaltools_efmi_physics
        if not settings.enabled:
            raise ValueError("请先开启“启用 EFMI 物理模块”。")
        records = rig.collect_export_records(context.scene, None, rig.current_profile())
        settings.status = "检查通过：" + str(len(records['chains'])) + " 条链 / " + str(len(records['bones'])) + " 根物理骨。"
        self.report({'INFO'}, settings.status)


CLASSES = (EFMI_OT_PhysicsAddObject, EFMI_OT_PhysicsRemoveObject,
           EFMI_OT_PhysicsAddChain, EFMI_OT_PhysicsRemoveChain,
           EFMI_OT_PhysicsAddSegment, EFMI_OT_PhysicsRemoveSegment,
           EFMI_OT_PhysicsFollowGroup, EFMI_OT_PhysicsEditRig,
           EFMI_OT_PhysicsSelectMesh, EFMI_OT_PhysicsSync, EFMI_OT_PhysicsValidate)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
