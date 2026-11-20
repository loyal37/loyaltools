"""Undoable tools for animation branches and Blender's native pose workflow."""

import bpy
from bpy.props import EnumProperty

from . import rig


_FOLLOW_ITEMS = []


def _follow_items(operator, context):
    entry = rig.active_entry(context.scene) if context and context.scene else None
    if entry is None or entry.object is None:
        return []
    try:
        profile = rig.current_profile()
        groups = rig.available_follow_groups(entry.object, profile)
    except (OSError, ValueError, KeyError, TypeError):
        return []
    # Blender requires the callback's strings to stay alive.
    _FOLLOW_ITEMS[:] = [(str(number), "Native Bone " + str(number), "", number) for number in groups]
    return _FOLLOW_ITEMS


class AnimationCommand:
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.scene and context.scene.loyaltools_efmi_animation.enabled
                    and rig.is_merged_workspace())

    def execute(self, context):
        try:
            self.run(context)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


def _entry(context):
    entry = rig.active_entry(context.scene)
    if entry is None:
        raise ValueError("Add a mesh to EFMI Bone Animation first.")
    return entry


def _selected_bone(context):
    entry = _entry(context)
    root = rig.active_root(entry)
    record = rig.active_bone(root)
    if record is None:
        raise ValueError("Select an animation bone first.")
    return entry, root, record


class EFMI_OT_AnimationAddObject(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_add_object'
    bl_label = "Add Selected Mesh"

    @classmethod
    def poll(cls, context):
        return bool(super().poll(context) and context.mode == 'OBJECT'
                    and context.object and context.object.type == 'MESH')

    def run(self, context):
        rig.add_object(context, context.object, rig.current_profile())


class EFMI_OT_AnimationRemoveObject(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_remove_object'
    bl_label = "Remove Animation Setup"
    bl_description = "Remove the animation armature and its vertex groups"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def run(self, context):
        rig.remove_object(context, _entry(context))


class EFMI_OT_AnimationAddRoot(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_add_root'
    bl_label = "Add Animation Root"
    bl_description = "Create an animation branch with its own playback settings and root vertex group"

    def run(self, context):
        rig.add_root(context, _entry(context), rig.current_profile())


class EFMI_OT_AnimationAddChild(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_add_child'
    bl_label = "Add Child Bone"
    bl_description = "Add a child to the selected animation bone with independently keyable location and rotation"

    def run(self, context):
        entry, root, parent = _selected_bone(context)
        rig.add_child(context, entry, root, parent, rig.current_profile())


class EFMI_OT_AnimationRemoveBranch(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_remove_branch'
    bl_label = "Remove Selected Bone and Descendants"
    bl_description = "Remove the selected bone and its descendants along with their animation vertex groups"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def run(self, context):
        rig.remove_branch(context, *_selected_bone(context))


class EFMI_OT_AnimationFollowGroup(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_follow_group'
    bl_label = "Choose Native Bone"
    bl_description = "Choose the native merged bone the current animation branch follows in-game"
    bl_property = 'group_id'

    group_id: EnumProperty(name="Native Merged Vertex Group", items=_follow_items)

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def run(self, context):
        entry, root, _ = _selected_bone(context)
        rig._verify_entry(entry)
        profile = rig.current_profile()
        if not self.group_id or int(self.group_id) not in rig.available_follow_groups(entry.object, profile):
            raise ValueError("Choose a native merged bone supported by the current workspace and mesh.")
        root.follow_group_id = int(self.group_id)


class EFMI_OT_AnimationEditRig(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_edit_rig'
    bl_label = "Edit Bone Rest Pose"
    bl_description = "Adjust the animation bone's head, tail, and roll in Edit Mode"

    def run(self, context):
        entry, _, record = _selected_bone(context)
        rig.select_bone(context, entry, record, 'EDIT')


class EFMI_OT_AnimationSelectPoseBone(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_select_pose_bone'
    bl_label = "Select Pose Bone"
    bl_description = "Select this bone in Pose Mode. Press K in the 3D Viewport to insert keyframes."

    def run(self, context):
        entry, _, record = _selected_bone(context)
        rig.select_bone(context, entry, record, 'POSE')


class EFMI_OT_AnimationSelectMesh(AnimationCommand, bpy.types.Operator):
    bl_idname = 'efmi.animation_select_mesh'
    bl_label = "Select Vertex Group"
    bl_description = "Select the mesh and activate this animation bone's vertex group"

    def run(self, context):
        entry, _, record = _selected_bone(context)
        rig.select_mesh(context, entry, record)


CLASSES = (EFMI_OT_AnimationAddObject, EFMI_OT_AnimationRemoveObject,
           EFMI_OT_AnimationAddRoot, EFMI_OT_AnimationAddChild, EFMI_OT_AnimationRemoveBranch,
           EFMI_OT_AnimationFollowGroup, EFMI_OT_AnimationEditRig,
           EFMI_OT_AnimationSelectPoseBone, EFMI_OT_AnimationSelectMesh)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
