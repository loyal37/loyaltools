"""Own animation armatures, vertex groups and preview modifiers independently."""

import json

import bpy
from mathutils import Vector

from ..common.efmi_merged_skeleton import (OBJECT_MARKER_MERGED_SKELETON, load_profile)
from ..common.global_config import GlobalConfig
from ..efmi_physics.rig import (_follow_head, _single_user_mesh, edit_rig,
                                is_merged_workspace, original_groups, uid)

OBJECT_UID = "LoyalTools:EFMIAnimationObject"
RIG_OWNER = "LoyalTools:EFMIAnimationOwner"
RIG_MARKER = "LoyalTools:EFMIAnimationRig"
BONE_UID = "LoyalTools:EFMIAnimationBone"
OWNED_GROUPS = "LoyalTools:EFMIAnimationGroups"
GROUP_PREFIX = "EFMI_ANIM_"


def current_profile():
    return load_profile(GlobalConfig.path_workspace_folder())


def active_entry(scene):
    settings = scene.loyaltools_efmi_animation
    if settings.objects:
        return settings.objects[min(settings.active_object_index, len(settings.objects) - 1)]
    return None


def active_root(entry):
    if entry and entry.roots:
        return entry.roots[min(entry.active_root_index, len(entry.roots) - 1)]
    return None


def active_bone(root):
    if root and root.bones:
        return root.bones[min(root.active_bone_index, len(root.bones) - 1)]
    return None


def _owns_source(entry):
    obj = entry.object
    return bool(entry.uid and obj and obj.type == 'MESH' and obj.get(OBJECT_UID) == entry.uid)


def _owns_rig(entry):
    rig = entry.rig
    return bool(entry.uid and rig and rig.type == 'ARMATURE' and rig.get(RIG_MARKER)
                and rig.get(RIG_OWNER) == entry.uid)


def _recorded_groups(entry):
    return [bone.group_name for root in entry.roots for bone in root.bones]


def _verify_source(entry):
    if not _owns_source(entry):
        raise ValueError("The mesh does not match its animation registration.")
    names = _recorded_groups(entry)
    if (len(names) != len(set(names)) or
            any(not name.startswith(GROUP_PREFIX) for name in names) or
            entry.object.get(OWNED_GROUPS) != json.dumps(sorted(names))):
        raise ValueError(f"{entry.object.name} has inconsistent animation vertex group records.")


def _verify_entry(entry):
    _verify_source(entry)
    if not _owns_rig(entry):
        raise ValueError("The animation armature is missing or does not match its registration.")
    if entry.rig.data.users > 1:
        raise ValueError("Make the animation armature data single-user.")


def _gpu_follow_owners(profile, number):
    return [component for component in profile['components'] if not component['cpu_posed']
            and int(component['vg_offset']) <= number < int(component['vg_offset']) + int(component['vg_count'])]


def available_follow_groups(obj, profile):
    if not obj or not obj.get(OBJECT_MARKER_MERGED_SKELETON):
        return []
    return [number for number in original_groups(obj, int(profile['bones_count']))
            if len(_gpu_follow_owners(profile, number)) == 1]


def require_follow_component(obj, root, profile):
    """Resolve the saved original group using the current workspace."""
    if not obj.get(OBJECT_MARKER_MERGED_SKELETON):
        raise ValueError(f"{obj.name} must be an EFMI merged-skeleton mesh.")
    number = int(root.follow_group_id)
    label = f"Native attachment {number} on {obj.name}/{root.name}"
    if not 0 <= number < int(profile['bones_count']) or obj.vertex_groups.get(str(number)) is None:
        raise ValueError(label + " is invalid for the current workspace or mesh. Check the workspace or choose another attachment.")
    owners = _gpu_follow_owners(profile, number)
    if len(owners) != 1:
        raise ValueError(label + " must belong to exactly one native GPU bone component. Check the workspace or choose another attachment.")
    return owners[0]


def is_managed_export_copy(entry, target):
    if (not _owns_source(entry) or not _owns_rig(entry) or target == entry.object or
            target.get(OBJECT_UID) != entry.uid):
        return False
    from ..blueprint.preprocess import PreProcessHelper
    from ..blueprint.preprocess_cache import PreProcessCache
    if target.name not in PreProcessHelper.created_copies:
        return False
    for requested, copy_name in PreProcessHelper.original_to_copy_map.items():
        if copy_name == target.name:
            source, source_name = PreProcessCache.resolve_source_object(requested)
            if source == entry.object and PreProcessCache.runtime_copy_matches(target, requested, source_name):
                return True
    return False


def managed_animation_entry(scene, obj, rig_object=None):
    """Resolve the dedicated rig on a registration or a proven preprocessing copy."""
    settings = getattr(scene, 'loyaltools_efmi_animation', None)
    if settings is None or obj is None:
        return None
    for entry in settings.objects:
        if (not _owns_source(entry) or not _owns_rig(entry) or
                (rig_object is not None and entry.rig != rig_object)):
            continue
        if obj == entry.object or is_managed_export_copy(entry, obj):
            return entry
    return None


def is_preview_modifier(entry, modifier):
    if modifier.type != 'ARMATURE':
        return False
    if modifier.object is None:
        return bool(entry.preview_modifier_uid and modifier.persistent_uid == entry.preview_modifier_uid)
    return bool(_owns_rig(entry) and modifier.object == entry.rig)


def _physics_anchor_groups(scene):
    settings = getattr(scene, 'loyaltools_efmi_physics', None)
    if settings is None:
        return set()
    return {chain.animation_anchor_group for entry in settings.objects for chain in entry.chains
            if chain.animation_anchor_group}


def has_linked_physics_chains(scene):
    return bool(_physics_anchor_groups(scene))


def enabled_changed(settings):
    for entry in settings.objects:
        if _owns_source(entry) and _owns_rig(entry):
            for modifier in entry.object.modifiers:
                if is_preview_modifier(entry, modifier):
                    modifier.show_viewport = settings.enabled
                    modifier.show_render = settings.enabled


def _check_mesh_mode(obj):
    if obj and obj.mode != 'OBJECT':
        raise ValueError("Exit mesh Edit Mode or Weight Paint Mode first.")


def _write_owned_groups(entry):
    entry.object[OWNED_GROUPS] = json.dumps(sorted(_recorded_groups(entry)))


def _preview(entry, enabled):
    obj = entry.object
    modifier = obj.modifiers.new(name="EFMI Bone Animation Preview", type='ARMATURE')
    modifier.object = entry.rig
    modifier.use_bone_envelopes = False
    modifier.show_viewport = enabled
    modifier.show_render = enabled
    entry.preview_modifier_uid = modifier.persistent_uid
    return modifier


def add_object(context, obj, profile):
    if not obj or obj.type != 'MESH' or not obj.get(OBJECT_MARKER_MERGED_SKELETON):
        raise ValueError("Select an EFMI mesh with native merged vertex groups.")
    _check_mesh_mode(obj)
    if not available_follow_groups(obj, profile):
        raise ValueError("The current workspace has no native GPU bone this mesh can follow.")
    if obj.get(OBJECT_UID):
        raise ValueError("This mesh already has animation registration data. Select the registered mesh in the panel or use an unregistered mesh.")
    settings = context.scene.loyaltools_efmi_animation
    entry = settings.objects.add()
    entry.uid = uid()
    entry.object = obj
    armature = bpy.data.armatures.new(obj.name + "_EFMIAnimationRig")
    preview = None
    try:
        rig = bpy.data.objects.new(armature.name, armature)
        entry.rig = rig
        context.scene.collection.objects.link(rig)
        context.view_layer.update()
        rig.matrix_world = obj.matrix_world.copy()
        rig.show_in_front = True
        rig[RIG_OWNER] = entry.uid
        rig[RIG_MARKER] = True
        rig.lock_location = rig.lock_rotation = rig.lock_scale = (True, True, True)
        obj[OBJECT_UID] = entry.uid
        _write_owned_groups(entry)
        preview = _preview(entry, settings.enabled)
        _single_user_mesh(obj)
    except Exception:
        if preview:
            obj.modifiers.remove(preview)
        if entry.rig:
            bpy.data.objects.remove(entry.rig, do_unlink=True)
        bpy.data.armatures.remove(armature)
        for key in (OBJECT_UID, OWNED_GROUPS):
            if key in obj:
                del obj[key]
        settings.objects.remove(len(settings.objects) - 1)
        raise
    settings.active_object_index = len(settings.objects) - 1
    return entry


def find_bone(rig, record):
    if rig.mode == 'EDIT':
        rig.update_from_editmode()
    matches = [bone for bone in rig.data.bones if bone.get(BONE_UID) == record.uid]
    if len(matches) > 1:
        raise ValueError(f"Duplicate animation bone identity for {record.group_name}. Delete the duplicate in Edit Mode.")
    return matches[0] if matches else None


def animation_anchor_options(scene, obj):
    """Return the same mesh's live registered animation bones for the picker."""
    entry = managed_animation_entry(scene, obj)
    if entry is None:
        return []
    _verify_entry(entry)
    return [(root, record) for root in entry.roots for record in root.bones
            if entry.object.vertex_groups.get(record.group_name)
            and find_bone(entry.rig, record) is not None]


def resolve_animation_anchor(scene, obj, group_name):
    """Resolve a physics attachment independently of the animation enable switch."""
    if obj is None or obj.type != 'MESH' or not group_name:
        raise ValueError("An animation attachment requires a valid mesh and animation bone vertex group.")
    entry = managed_animation_entry(scene, obj)
    if entry is None:
        raise ValueError(f"{obj.name} needs a valid animation registration to follow {group_name}.")
    _verify_entry(entry)
    for root in entry.roots:
        for record in root.bones:
            if record.group_name != group_name:
                continue
            if not entry.object.vertex_groups.get(group_name) or find_bone(entry.rig, record) is None:
                raise ValueError(f"Animation attachment {group_name} on {obj.name} was deleted.")
            return entry, root, record
    raise ValueError(f"Animation attachment {group_name} is unregistered on {obj.name}.")


def _next_bone_id(scene, obj):
    settings = scene.loyaltools_efmi_animation
    number = min(other.loyaltools_efmi_animation.next_bone_id for other in bpy.data.scenes)
    while number > -2147483648:
        name = GROUP_PREFIX + format(-number, '06d')
        if obj.vertex_groups.get(name) is None:
            settings.next_bone_id = number - 1
            return number, name
        number -= 1
    raise ValueError("Animation bone authoring IDs are exhausted.")


def _add_bone(context, entry, root, parent=None):
    obj = entry.object
    _check_mesh_mode(obj)
    number, name = _next_bone_id(context.scene, obj)
    bone_uid = uid()
    parent_name = None
    if parent:
        parent_bone = find_bone(entry.rig, parent)
        if parent_bone is None:
            raise ValueError("The selected animation bone was deleted.")
        parent_name = parent_bone.name
    with edit_rig(context, entry.rig) as bones:
        bone = bones.new(name)
        bone_name = bone.name
        if bone_name != name:
            bones.remove(bone)
            raise ValueError(f"An animation bone named {name} already exists.")
        bone[BONE_UID] = bone_uid
        if parent_name:
            parent_bone = bones[parent_name]
            bone.parent = parent_bone
            bone.use_connect = False
            bone.head = parent_bone.tail
            bone.tail = parent_bone.tail + (parent_bone.tail - parent_bone.head)
            bone.roll = parent_bone.roll
        else:
            bone.head = _follow_head(obj, root.follow_group_id)
            bone.tail = bone.head + Vector((0.0, 0.0, max(max(obj.dimensions) * 0.05, 0.05)))
    try:
        obj.vertex_groups.new(name=name)
        pose_bone = entry.rig.pose.bones[name]
        pose_bone.rotation_mode = 'XYZ'
        pose_bone.lock_scale = (True, True, True)
    except Exception:
        with edit_rig(context, entry.rig) as bones:
            bones.remove(bones[name])
        group = obj.vertex_groups.get(name)
        if group:
            obj.vertex_groups.remove(group)
        raise
    record = root.bones.add()
    record.uid = bone_uid
    record.parent_uid = parent.uid if parent else ''
    record.bone_id = number
    record.group_name = name
    root.active_bone_index = len(root.bones) - 1
    _write_owned_groups(entry)
    return record


def add_root(context, entry, profile):
    _verify_entry(entry)
    groups = available_follow_groups(entry.object, profile)
    if not groups:
        raise ValueError("The current workspace has no native GPU bone this mesh can follow.")
    root = entry.roots.add()
    root.uid = uid()
    root.name = "Animation Branch " + str(len(entry.roots))
    root.follow_group_id = groups[0]
    root.frame_start = context.scene.frame_start
    root.frame_end = context.scene.frame_end
    try:
        _add_bone(context, entry, root)
    except Exception:
        entry.roots.remove(len(entry.roots) - 1)
        raise
    entry.active_root_index = len(entry.roots) - 1
    return root


def add_child(context, entry, root, parent, profile):
    _verify_entry(entry)
    require_follow_component(entry.object, root, profile)
    return _add_bone(context, entry, root, parent)


def _remove_groups(obj, names):
    _check_mesh_mode(obj)
    if any(obj.vertex_groups.get(name) for name in names):
        _single_user_mesh(obj)
    for name in names:
        group = obj.vertex_groups.get(name)
        if group:
            obj.vertex_groups.remove(group)


def remove_branch(context, entry, root, record):
    _verify_entry(entry)
    _check_mesh_mode(entry.object)
    pending = {record.uid}
    indices = []
    names = []
    for index, item in enumerate(root.bones):
        if item.uid in pending or item.parent_uid in pending:
            pending.add(item.uid)
            indices.append(index)
            bone = find_bone(entry.rig, item)
            if bone:
                names.append(bone.name)
    groups = [root.bones[index].group_name for index in indices]
    if not _physics_anchor_groups(context.scene).isdisjoint(groups):
        raise ValueError("A physics chain follows this branch. Choose another attachment in the physics panel first.")
    if names:
        with edit_rig(context, entry.rig) as bones:
            for name in reversed(names):
                bones.remove(bones[name])
    _remove_groups(entry.object, groups)
    if not record.parent_uid:
        root_index = next(i for i, item in enumerate(entry.roots) if item.uid == root.uid)
        entry.roots.remove(root_index)
        entry.active_root_index = min(entry.active_root_index, max(0, len(entry.roots) - 1))
    else:
        for index in reversed(indices):
            root.bones.remove(index)
        root.active_bone_index = min(root.active_bone_index, max(0, len(root.bones) - 1))
    _write_owned_groups(entry)


def remove_object(context, entry):
    if not _physics_anchor_groups(context.scene).isdisjoint(_recorded_groups(entry)):
        raise ValueError("Physics chains follow animation bones on this mesh. Choose other attachments in the physics panel first.")
    if entry.object:
        _verify_source(entry)
        _check_mesh_mode(entry.object)
    if entry.rig and not _owns_rig(entry):
        raise ValueError("The animation armature does not match its registration.")
    if entry.rig and entry.rig.mode != 'OBJECT':
        if context.object != entry.rig:
            raise ValueError("Exit Edit Mode or Pose Mode on the animation armature first.")
        bpy.ops.object.mode_set(mode='OBJECT')
    if entry.object:
        _remove_groups(entry.object, _recorded_groups(entry))
        for modifier in list(entry.object.modifiers):
            if is_preview_modifier(entry, modifier):
                entry.object.modifiers.remove(modifier)
        for key in (OBJECT_UID, OWNED_GROUPS):
            del entry.object[key]
    if entry.rig:
        armature = entry.rig.data
        bpy.data.objects.remove(entry.rig, do_unlink=True)
        if not armature.users:
            bpy.data.armatures.remove(armature)
    settings = context.scene.loyaltools_efmi_animation
    index = next(i for i, item in enumerate(settings.objects) if item.uid == entry.uid)
    settings.objects.remove(index)
    settings.active_object_index = min(settings.active_object_index, max(0, len(settings.objects) - 1))


def _select_object(context, obj):
    if obj.name not in context.view_layer.objects:
        raise ValueError("The object is outside the current view layer.")
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for selected in context.selected_objects:
        selected.select_set(False)
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def select_mesh(context, entry, record):
    _verify_source(entry)
    group = entry.object.vertex_groups.get(record.group_name)
    if group is None:
        raise ValueError(f"Vertex group {record.group_name} was deleted or renamed.")
    _select_object(context, entry.object)
    entry.object.vertex_groups.active_index = group.index


def select_bone(context, entry, record, mode):
    _verify_entry(entry)
    bone = find_bone(entry.rig, record)
    if bone is None:
        raise ValueError("The selected animation bone was deleted.")
    name = bone.name
    _select_object(context, entry.rig)
    bpy.ops.object.mode_set(mode=mode)
    if mode == 'EDIT':
        bones = entry.rig.data.edit_bones
        selected = bones[name]
        for item in bones:
            item.select = item.select_head = item.select_tail = (item == selected)
    else:
        bones = entry.rig.data.bones
        selected = bones[name]
        # In Blender 5.0 selection moved from Bone to PoseBone.
        if hasattr(selected, 'select'):
            for item in bones:
                item.select = item == selected
        else:
            for item in entry.rig.pose.bones:
                item.select = item.bone == selected
    bones.active = selected
