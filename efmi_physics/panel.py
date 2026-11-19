"""An independent EFMI-only N-panel; original exporter/toolkit panels are untouched."""

import bpy

from . import rig


class EFMI_UL_PhysicsObjects(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.label(text=item.object.name if item.object else "[网格已删除]", icon='MESH_DATA')
        if item.rig is None:
            layout.label(text="骨架已删除", icon='ERROR')


class EFMI_UL_PhysicsChains(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.prop(item, "name", text="", emboss=False, icon='BONE_DATA')
        layout.label(text="原骨 " + str(item.follow_group_id) + " · " + str(len(item.segments)) + " 段")


class EFMI_UL_PhysicsSegments(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.label(text=("父骨" if index == 0 else "子骨") + " · " + rig._segment_group(item), icon='BONE_DATA')


class EFMI_PT_Physics(bpy.types.Panel):
    bl_label = "EFMI 物理骨"
    bl_idname = "EFMI_PT_Physics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LoyalTools"
    bl_order = 55
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and
                    not getattr(context.scene, 'herta_show_toolkit', False))

    def draw(self, context):
        layout = self.layout
        settings = context.scene.loyaltools_efmi_physics
        warning = layout.box()
        warning.alert = True
        warning.label(text="WARNING：请先刷好", icon='ERROR')
        warning.label(text="原合并骨骼权重！")
        warning.label(text="再将物体添加到物理模块。")
        warning.label(text="否则可能出现挂点或导出异常。")
        layout.prop(settings, "enabled")
        available = rig.is_merged_workspace()
        if not available:
            layout.label(text="仅 EFMI 合并骨骼工作空间可使用。", icon='INFO')
        layout = layout.column()
        layout.enabled = settings.enabled and available
        row = layout.row(align=True)
        row.operator("efmi.physics_add_object", text="添加选中物体", icon='ADD')
        row.operator("efmi.physics_sync", text="同步检查", icon='FILE_REFRESH')
        layout.template_list("EFMI_UL_PhysicsObjects", "", settings, "objects", settings,
                             "active_object_index", rows=2)
        entry = rig.active_entry(context.scene)
        if entry:
            layout.operator("efmi.physics_remove_object", icon='REMOVE')
            if entry.object is None or entry.rig is None:
                layout.label(text="有数据已被删除，请先同步检查。", icon='ERROR')
                return
            layout.label(text="原骨范围：0.." + str(settings.base_bone_count - 1))
            row = layout.row(align=True)
            row.operator("efmi.physics_add_chain", icon='ADD')
            row.operator("efmi.physics_remove_chain", text="删除链", icon='REMOVE')
            layout.template_list("EFMI_UL_PhysicsChains", "", entry, "chains", entry,
                                 "active_chain_index", rows=2)
            chain = rig.active_chain(entry)
            if chain:
                box = layout.box()
                box.prop(chain, "name")
                box.operator("efmi.physics_follow_group", text="跟随原骨：" + str(chain.follow_group_id), icon='VIEWZOOM')
                box.label(text="父骨头部 = 挂点；骨头 / 尾部 / Roll 可真实编辑。", icon='INFO')
                box.template_list("EFMI_UL_PhysicsSegments", "", chain, "segments", chain,
                                  "active_segment_index", rows=3)
                box.operator("efmi.physics_add_segment", icon='ADD')
                remove = box.row()
                remove.enabled = chain.active_segment_index > 0
                remove.operator("efmi.physics_remove_segment", icon='REMOVE')
                box.operator("efmi.physics_edit_rig", icon='EDITMODE_HLT')
                box.label(text="增删骨后退出编辑模式，每秒自动同步所属组。")
                box.operator("efmi.physics_select_mesh", icon='GROUP_VERTEX')
                box.label(text="给 EFMI_PHYS_ 专属组刷权重；原组不改名。")
                params = box.column(align=True)
                params.prop(chain, "damping")
                params.prop(chain, "stiffness")
                params.prop(chain, "gravity")
                params.prop(chain, "cone_angle")
                params.prop(chain, "wind_strength")
                box.label(text="运行时：模型空间重力 / 回正 / 摆角 / 微风。", icon='INFO')
        layout.operator("efmi.physics_validate", icon='CHECKMARK')
        if settings.status:
            layout.label(text=settings.status, icon='INFO')


CLASSES = (EFMI_UL_PhysicsObjects, EFMI_UL_PhysicsChains, EFMI_UL_PhysicsSegments, EFMI_PT_Physics)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
