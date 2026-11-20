"""LoyalTools sidebar for EFMI animation branches."""

import bpy

from . import rig


class EFMI_UL_AnimationObjects(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.label(text=item.object.name if item.object else "Mesh Deleted", icon='MESH_DATA')
        if item.rig is None:
            layout.label(text="Armature Deleted", icon='ERROR')


class EFMI_UL_AnimationRoots(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        layout.prop(item, 'name', text="", emboss=False, icon='BONE_DATA')


class EFMI_UL_AnimationBones(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        parents = {record.uid: record.parent_uid for record in data.bones}
        depth = 0
        parent = item.parent_uid
        while parent and depth < len(parents):
            depth += 1
            parent = parents.get(parent, '')
        layout.label(text="  " * min(depth, 5) + item.group_name, icon='BONE_DATA')


class EFMI_PT_Animation(bpy.types.Panel):
    bl_label = "EFMI Bone Animation"
    bl_idname = 'EFMI_PT_Animation'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "LoyalTools"
    bl_order = 56
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and not getattr(context.scene, 'herta_show_toolkit', False))

    def draw(self, context):
        layout = self.layout
        settings = context.scene.loyaltools_efmi_animation
        available = rig.is_merged_workspace()
        layout.prop(settings, 'enabled')
        if settings.disable_blocked and rig.has_linked_physics_chains(context.scene):
            warning = layout.box()
            warning.alert = True
            warning.label(text="Physics chains use animation bones.", icon='ERROR')
            warning.label(text="Choose native attachments in")
            warning.label(text="EFMI Physics before turning this off.")
        if not available:
            layout.label(text="Open an EFMI merged-skeleton workspace.", icon='INFO')
        controls = layout.column()
        controls.enabled = available and settings.enabled
        controls.operator('efmi.animation_add_object', icon='ADD')
        controls.template_list('EFMI_UL_AnimationObjects', '', settings, 'objects', settings,
                               'active_object_index', rows=2)
        entry = rig.active_entry(context.scene)
        if entry is None:
            return
        controls.operator('efmi.animation_remove_object', icon='REMOVE')
        if entry.object is None or entry.rig is None:
            return
        controls.operator('efmi.animation_add_root', icon='ADD')
        controls.template_list('EFMI_UL_AnimationRoots', '', entry, 'roots', entry,
                               'active_root_index', rows=2)
        root = rig.active_root(entry)
        if root is None:
            return
        box = controls.box()
        box.operator('efmi.animation_follow_group', text="Native Bone " + str(root.follow_group_id), icon='VIEWZOOM')
        range_row = box.row(align=True)
        range_row.alert = root.frame_start > root.frame_end
        range_row.prop(root, 'frame_start')
        range_row.prop(root, 'frame_end')
        box.prop(root, 'playback_speed')
        box.template_list('EFMI_UL_AnimationBones', '', root, 'bones', root, 'active_bone_index', rows=3)
        buttons = box.row(align=True)
        buttons.operator('efmi.animation_add_child', icon='ADD')
        buttons.operator('efmi.animation_remove_branch', text="Remove Branch", icon='REMOVE')
        buttons = box.row(align=True)
        buttons.operator('efmi.animation_edit_rig', icon='EDITMODE_HLT')
        buttons.operator('efmi.animation_select_pose_bone', icon='POSE_HLT')
        box.operator('efmi.animation_select_mesh', icon='GROUP_VERTEX')
        if context.object == entry.rig and context.mode == 'POSE':
            box.label(text="Press K and choose Location & Rotation.")


CLASSES = (EFMI_UL_AnimationObjects, EFMI_UL_AnimationRoots, EFMI_UL_AnimationBones, EFMI_PT_Animation)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
