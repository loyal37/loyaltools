"""Explicit, undoable authoring operations and strict export data collection.

Prefixed physical groups and their authoring IDs are independent of the current
native skeleton size. Blender's mutable group indices are resolved explicitly
at export. A capped timer synchronizes only registered authoring rigs; existing
mesh modifiers and the game's original numeric vertex groups stay untouched.
"""

import json
import hashlib
import os
import time
import uuid
from contextlib import contextmanager
from functools import wraps

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

from ..common.efmi_merged_skeleton import (OBJECT_MARKER_MERGED_SKELETON,
                                         get_profile_path, load_profile)
from ..common.global_config import GlobalConfig
from ..common.logic_name import LogicName

RIG_OWNER = "LoyalTools:EFMIPhysicsOwner"
RIG_MARKER = "LoyalTools:EFMIPhysicsRig"
SEGMENT_UID = "LoyalTools:EFMIPhysicsSegment"
CHAIN_UID = "LoyalTools:EFMIPhysicsChain"
OWNED_GROUPS = "LoyalTools:EFMIPhysicsOwnedGroups"
OWNED_GROUP_NAMES = "LoyalTools:EFMIPhysicsOwnedGroupNames"
OBJECT_UID = "LoyalTools:EFMIPhysicsObject"
GROUP_PREFIX = 'EFMI_PHYS_'
AUTHORING_ID_START = 65536

_pending_cleanup = {}
_cleanup_running = False
_undo_in_progress = False
_authoring_depth = 0
_profile_available_cache = {}
_last_cleanup_error = None
_timer_running = False
_last_sync_time = float('-inf')
_TIMER_KEY = 'LoyalTools:EFMIPhysicsSyncTimer'
_BOOTSTRAP_KEY = 'LoyalTools:EFMIPhysicsBootstrapTimer'


def _authoring_change(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        global _authoring_depth
        scene = getattr(args[0], 'scene', None) if args else None
        if scene is not None and not scene.loyaltools_efmi_physics.enabled:
            return None
        _authoring_depth += 1
        try:
            return function(*args, **kwargs)
        finally:
            _authoring_depth -= 1
    return guarded


def uid():
    return uuid.uuid4().hex


def is_efmi():
    return GlobalConfig.logic_name == LogicName.EFMI


def is_merged_workspace():
    """Cheap UI poll with stat-invalidated validation of the actual profile."""
    if not is_efmi():
        return False
    try:
        path = get_profile_path(GlobalConfig.path_workspace_folder())
        stat = os.stat(path)
        key = (os.path.normcase(path), stat.st_mtime_ns, stat.st_size)
        if key not in _profile_available_cache:
            _profile_available_cache.clear()
            try:
                _profile_available_cache[key] = bool(load_profile(GlobalConfig.path_workspace_folder()))
            except (OSError, ValueError, TypeError):
                _profile_available_cache[key] = False
        return _profile_available_cache[key]
    except (OSError, ValueError, TypeError):
        return False


def workspace_key():
    return os.path.normcase(os.path.abspath(GlobalConfig.path_workspace_folder()))


def current_profile():
    if not is_efmi():
        raise ValueError("物理骨仅支持 EFMI 骨骼合并工作空间。")
    return load_profile(GlobalConfig.path_workspace_folder())


def active_entry(scene):
    settings = scene.loyaltools_efmi_physics
    return (settings.objects[settings.active_object_index]
            if 0 <= settings.active_object_index < len(settings.objects) else None)


def active_chain(entry):
    return (entry.chains[entry.active_chain_index]
            if entry and 0 <= entry.active_chain_index < len(entry.chains) else None)


def _numeric_group(name):
    # Reject aliases such as 01: two differently named groups cannot share an ID.
    return int(name) if name.isascii() and name.isdecimal() and str(int(name)) == name else None


def original_groups(obj, base):
    if obj is None or obj.type != 'MESH':
        return []
    return sorted(number for group in obj.vertex_groups
                  if (number := _numeric_group(group.name)) is not None and number < base)


def _check_workspace(settings, profile):
    base = int(profile['bones_count'])
    if base <= 0 or base > 65535:
        raise ValueError("原合并骨数量 B 必须在 1..65535 之间。")
    if settings.enabled:
        settings.workspace_key = workspace_key()
        settings.base_bone_count = base
        settings.next_bone_id = max(settings.next_bone_id, base)
    return base


def _profile_key(profile):
    content = {'bones_count': profile['bones_count'], 'object_guid': profile.get('object_guid'),
               'components': profile.get('components', [])}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode('utf-8')).hexdigest()


def refresh_current_workspace(scene, profile=None):
    if not scene.loyaltools_efmi_physics.enabled:
        return None
    profile = current_profile() if profile is None else profile
    _check_workspace(scene.loyaltools_efmi_physics, profile)
    return profile


def _owned_ids(obj):
    try:
        values = json.loads(obj.get(OWNED_GROUPS, "[]"))
        return {int(value) for value in values}
    except (ValueError, TypeError):
        raise ValueError("物体的物理顶点组所有权标记损坏，请先修复或恢复文件。")


def _write_owned(entry):
    numbers = sorted(segment.bone_id for chain in entry.chains for segment in chain.segments)
    entry.owned_group_ids = json.dumps(numbers)
    if entry.object:
        entry.object[OWNED_GROUPS] = json.dumps(numbers)
        entry.object[OWNED_GROUP_NAMES] = json.dumps(sorted(_segment_group(segment)
            for chain in entry.chains for segment in chain.segments))


def _group_name(number):
    return GROUP_PREFIX + format(number - AUTHORING_ID_START + 1, '06d')


def _segment_group(segment):
    return segment.group_name or str(segment.bone_id)


def _verified_owned(entry, scene=None, migrate=True):
    """Ownership requires registry and both ID owners, never numeric names alone."""
    obj = entry.object
    if obj is None:
        return set()
    if obj.get(OBJECT_UID) != entry.uid:
        raise ValueError(obj.name + " 的物理物体身份不匹配，拒绝修改顶点组。")
    if entry.rig and entry.rig.get(RIG_OWNER) != entry.uid:
        raise ValueError(obj.name + " 的物理骨架所有权不匹配。")
    numbers = [segment.bone_id for chain in entry.chains for segment in chain.segments]
    if len(numbers) != len(set(numbers)) or set(numbers) != _owned_ids(obj):
        raise ValueError(obj.name + " 的物理所属组标记与骨段记录不一致，请恢复管理元数据后重试。")
    owned = set(numbers)
    if entry.schema_version >= 2:
        if set(json.loads(entry.owned_group_ids)) != owned:
            raise ValueError(obj.name + " 的物理所属组记录不一致。")
        protected = set(json.loads(entry.native_group_names))
        conflicts = sorted(segment.bone_id for chain in entry.chains for segment in chain.segments
                           if _segment_group(segment) in protected)
        if conflicts:
            raise ValueError(obj.name + " 的物理编号与受保护原组冲突：" + str(conflicts))
    elif migrate:
        _require_legacy_bone_proof(entry)
        old = scene.loyaltools_efmi_physics if scene else None
        entry.created_base_bone_count = old.base_bone_count if old else 0
        entry.created_workspace_key = old.workspace_key if old else ''
        owned_names = {_segment_group(segment) for chain in entry.chains for segment in chain.segments}
        entry.native_group_names = json.dumps([group.name for group in obj.vertex_groups
                                              if group.name not in owned_names])
        entry.owned_group_ids = json.dumps(sorted(owned))
        entry.schema_version = 2
    return owned


def _require_legacy_bone_proof(entry):
    """Legacy numeric groups need live bone evidence before any destructive use."""
    _check_entry(entry)
    for chain in entry.chains:
        for segment in chain.segments:
            bone = find_bone(entry.rig, segment)
            if bone is None or bone.get(CHAIN_UID) != chain.uid:
                raise ValueError(entry.object.name + " 的旧物理组 " + _segment_group(segment)
                                 + " 缺少对应的真实骨骼身份；该组与权重已保留，请恢复骨骼或显式修复旧登记后再迁移。")


def _migrate_entry(scene, entry):
    """Upgrade proven legacy owned numeric groups by renaming, retaining weights."""
    _verified_owned(entry, scene, migrate=False)
    if entry.schema_version >= 3:
        return
    _check_entry(entry)
    _check_mesh_mode(entry.object)
    if entry.rig.mode == 'EDIT':
        raise ValueError('请退出骨骼编辑模式后迁移旧物理组。')
    _require_legacy_bone_proof(entry)
    _verified_owned(entry, scene)
    plan = []
    for chain in entry.chains:
        for segment in chain.segments:
            old_name = _segment_group(segment)
            number = allocate_bone_id(scene)
            name = _group_name(number)
            plan.append((segment, old_name, number, name))
    for segment, old_name, number, name in plan:
        group = entry.object.vertex_groups.get(old_name)
        if group is not None:
            group.name = name
        else:
            entry.object.vertex_groups.new(name=name)
        segment.bone_id = number
        segment.group_name = name
    entry.schema_version = 3
    _write_owned(entry)


def managed_physics_entry(scene, obj, rig_object=None):
    """Read-only preview filter ABI; disabled modules do not inspect object data."""
    settings = scene.loyaltools_efmi_physics
    if not settings.enabled:
        return None
    for entry in settings.objects:
        if not entry.object or (rig_object is not None and entry.rig != rig_object):
            continue
        if obj != entry.object and not is_managed_export_copy(entry, obj):
            continue
        try:
            _verified_owned(entry, migrate=False)
            if entry.rig and entry.rig.get(RIG_MARKER) and entry.rig.get(RIG_OWNER) == entry.uid:
                return entry
        except (ValueError, TypeError):
            pass
    return None


def is_managed_export_copy(entry, target):
    """Prove a copy belongs to the current preprocess run, including aliases.

    Hashes identify cache contents, not copy ownership: the serial uncached path
    intentionally has an empty hash. Copied custom properties alone grant no
    permission; both live preprocess registries must identify the target.
    """
    source = entry.object
    if source is None or target == source or target.get(OBJECT_UID) != entry.uid:
        return False
    if source.get(OBJECT_UID) != entry.uid or not entry.rig or entry.rig.get(RIG_OWNER) != entry.uid:
        return False
    from ..blueprint.preprocess import PreProcessHelper
    from ..blueprint.preprocess_cache import PreProcessCache
    if target.name not in PreProcessHelper.created_copies:
        return False
    for requested, copy_name in PreProcessHelper.original_to_copy_map.items():
        if copy_name != target.name:
            continue
        resolved, source_name = PreProcessCache.resolve_source_object(requested)
        if resolved == source and PreProcessCache.runtime_copy_matches(target, requested, source_name):
            return True
    return False


def _check_entry_profile(entry, profile):
    owned = _verified_owned(entry)
    conflicts = sorted(number for number in owned if number < AUTHORING_ID_START)
    if conflicts:
        raise ValueError(entry.object.name + " 的物理编号 " + str(conflicts)
                         + " 尚未迁移为独立物理编号，请退出编辑模式并同步；未修改原骨组。")


def _single_user_mesh(obj):
    # Blender stores deform weights on Mesh, but vertex-group names on Object.
    # Removing a group on a linked duplicate must never rewrite the source mesh.
    if obj.data.users > 1:
        obj.data = obj.data.copy()


def _check_mesh_mode(obj):
    if obj and obj.type == 'MESH' and obj.mode != 'OBJECT':
        raise ValueError("请先退出网格编辑 / 权重绘制模式，再修改物理骨段或顶点组。")


def _remove_groups(obj, names):
    if obj is None or obj.type != 'MESH':
        return
    _check_mesh_mode(obj)
    names = list(names)
    if any(obj.vertex_groups.get(name) for name in names):
        _single_user_mesh(obj)
    for name in names:
        group = obj.vertex_groups.get(name)
        if group:
            # remove() also removes this group's weights; no zero-weight residue.
            obj.vertex_groups.remove(group)


def _remove_owned_groups(entry, ids, scene=None):
    if entry.schema_version < 3:
        _require_legacy_bone_proof(entry)
    owned = _verified_owned(entry, scene)
    if not set(ids).issubset(owned):
        raise ValueError("拒绝删除没有明确物理所有权的顶点组。")
    groups = {_segment_group(segment) for chain in entry.chains for segment in chain.segments
              if segment.bone_id in set(ids)}
    _remove_groups(entry.object, groups)


def allocate_bone_id(scene):
    settings = scene.loyaltools_efmi_physics
    number = max(AUTHORING_ID_START, settings.next_bone_id)
    used = set()
    for other_scene in bpy.data.scenes:
        other = other_scene.loyaltools_efmi_physics
        number = max(number, other.next_bone_id)
        used.update(segment.bone_id for entry in other.objects
                    for chain in entry.chains for segment in chain.segments)
    names = set()
    # Visit registered meshes only, including their protected native groups.
    for other_scene in bpy.data.scenes:
        for entry in other_scene.loyaltools_efmi_physics.objects:
            obj = entry.object
            if obj is None or obj.type != 'MESH':
                continue
            names.update(group.name for group in obj.vertex_groups)
    while number in used or _group_name(number) in names:
        number += 1
    if number >= 2147483647:
        raise ValueError("物理制作编号已经耗尽；最终 GPU 编号仍由导出器按当前 B 紧凑分配。")
    settings.next_bone_id = number + 1
    return number


@contextmanager
def edit_rig(context, rig):
    """Flush edits and restore the user's selection/mode even after an error."""
    if rig is None or rig.type != 'ARMATURE' or rig.name not in context.view_layer.objects:
        raise ValueError("物理骨架已删除或不在当前视图层，请执行同步检查。")
    previous = context.view_layer.objects.active
    previous_mode = previous.mode if previous else 'OBJECT'
    selected = list(context.selected_objects)
    hidden = rig.hide_get()
    hidden_viewport = rig.hide_viewport
    try:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        rig.hide_viewport = False
        rig.hide_set(False)
        rig.select_set(True)
        context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        yield rig.data.edit_bones
    finally:
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        rig.hide_set(hidden)
        rig.hide_viewport = hidden_viewport
        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in selected:
            if obj.name in context.view_layer.objects:
                obj.select_set(True)
        if previous and previous.name in context.view_layer.objects:
            context.view_layer.objects.active = previous
            if previous_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=previous_mode)
        else:
            context.view_layer.objects.active = None


def find_bone(rig, segment):
    if rig is None or rig.type != 'ARMATURE':
        return None
    if rig.mode == 'EDIT':
        rig.update_from_editmode()
    matches = [bone for bone in rig.data.bones if bone.get(SEGMENT_UID) == segment.uid]
    if len(matches) > 1:
        raise ValueError("发现复制的物理骨身份，请删除重复骨后同步检查。")
    return matches[0] if matches else None


def _check_entry(entry):
    if entry.object is None or entry.object.type != 'MESH':
        raise ValueError("登记网格已删除，请执行同步检查。")
    if entry.rig is None or entry.rig.type != 'ARMATURE':
        raise ValueError("物理骨架已删除，请执行同步检查。")
    if entry.rig.get(RIG_OWNER) != entry.uid:
        raise ValueError("物理骨架所有权不匹配，拒绝修改其他骨架。")
    if entry.rig.data.users > 1:
        raise ValueError("物理骨架数据被链接复制，请先使骨架数据单用户化。")


@_authoring_change
def add_object(context, obj, profile):
    settings = context.scene.loyaltools_efmi_physics
    base = _check_workspace(settings, profile)
    if obj is None or obj.type != 'MESH':
        raise ValueError("请先选择一个网格物体。")
    _check_mesh_mode(obj)
    if not obj.get(OBJECT_MARKER_MERGED_SKELETON):
        raise ValueError("此物体不是 EFMI 骨骼合并网格；请先完成合并骨骼导入和权重。")
    if not original_groups(obj, base):
        raise ValueError("物体没有原合并数字顶点组，请先刷好原合并骨骼权重。")
    if any(entry.object == obj for scene in bpy.data.scenes
           for entry in scene.loyaltools_efmi_physics.objects):
        raise ValueError("这个物体已经在场景中登记过；请复制物体后独立添加。")
    # Object duplication copies custom properties and groups. Re-enrolling it
    # intentionally creates an independent rig and fresh global IDs.
    copied_ids = _owned_ids(obj)
    if copied_ids:
        source = next((entry for scene in bpy.data.scenes
                       for entry in scene.loyaltools_efmi_physics.objects
                       if entry.uid == obj.get(OBJECT_UID) and entry.object), None)
        if source is None or _verified_owned(source, migrate=False) != copied_ids:
            raise ValueError("副本的物理所属组缺少可验证来源，拒绝清理可能属于原骨的组。")
        if source.schema_version < 3:
            _require_legacy_bone_proof(source)
        _remove_groups(obj, [_segment_group(segment) for chain in source.chains
                            for segment in chain.segments])
    _single_user_mesh(obj)
    entry = settings.objects.add()
    entry.uid = uid()
    entry.object = obj
    entry.schema_version = 3
    entry.created_base_bone_count = base
    entry.created_workspace_key = workspace_key()
    entry.created_profile_key = _profile_key(profile)
    entry.native_group_names = json.dumps([group.name for group in obj.vertex_groups])
    armature = bpy.data.armatures.new(obj.name + "_EFMI物理骨")
    rig = bpy.data.objects.new(obj.name + "_EFMI物理骨", armature)
    context.scene.collection.objects.link(rig)
    context.view_layer.update()
    rig.matrix_world = obj.matrix_world.copy()
    rig.show_in_front = True
    rig.display_type = 'WIRE'
    rig[RIG_OWNER] = entry.uid
    rig[RIG_MARKER] = True
    armature[RIG_MARKER] = True
    entry.rig = rig
    obj[OBJECT_UID] = entry.uid
    _write_owned(entry)
    settings.active_object_index = len(settings.objects) - 1
    return entry


def _follow_head(obj, group_id):
    index = obj.vertex_groups[str(group_id)].index
    head = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            if group.group == index and group.weight > 0:
                head += vertex.co * group.weight
                total += group.weight
    return head / total if total else head


def _create_segment(context, entry, chain, head=None):
    _check_mesh_mode(entry.object)
    number = allocate_bone_id(context.scene)
    segment = chain.segments.add()
    segment.uid = uid()
    segment.bone_id = number
    segment.group_name = _group_name(number)
    segment.bone_name = str(number)
    previous = (find_bone(entry.rig, chain.segments[-2])
                if len(chain.segments) > 1 else None)
    previous_name = previous.name if previous else None
    try:
        with edit_rig(context, entry.rig) as bones:
            bone = bones.new(str(number))
            bone[SEGMENT_UID] = segment.uid
            bone[CHAIN_UID] = chain.uid
            if previous_name:
                parent = bones[previous_name]
                bone.parent = parent
                bone.use_connect = True
                bone.head = parent.tail
                bone.tail = parent.tail + (parent.tail - parent.head)
                bone.roll = parent.roll
            else:
                bone.head = head if head is not None else (0.0, 0.0, 0.0)
                length = max(max(entry.object.dimensions) * 0.05, 0.05)
                bone.tail = bone.head + Vector((0.0, 0.0, length))
        entry.object.vertex_groups.new(name=segment.group_name)
    except Exception:
        chain.segments.remove(len(chain.segments) - 1)
        raise
    chain.active_segment_index = len(chain.segments) - 1
    _write_owned(entry)
    return segment


@_authoring_change
def add_chain(context, entry, profile):
    _check_workspace(context.scene.loyaltools_efmi_physics, profile)
    _check_entry(entry)
    _migrate_entry(context.scene, entry)
    _check_entry_profile(entry, profile)
    groups = original_groups(entry.object, int(profile['bones_count']))
    if not groups:
        raise ValueError("没有可跟随的原合并顶点组。")
    chain = entry.chains.add()
    chain.uid = uid()
    chain.name = "物理链 " + str(len(entry.chains))
    chain.follow_group_id = groups[0]
    try:
        _create_segment(context, entry, chain, _follow_head(entry.object, groups[0]))
    except Exception:
        entry.chains.remove(len(entry.chains) - 1)
        raise
    entry.active_chain_index = len(entry.chains) - 1
    return chain


@_authoring_change
def add_segment(context, entry, chain, profile):
    _check_workspace(context.scene.loyaltools_efmi_physics, profile)
    _check_entry(entry)
    _migrate_entry(context.scene, entry)
    _check_entry_profile(entry, profile)
    if not chain or not chain.segments:
        raise ValueError("请先添加包含父骨的物理链。")
    for segment in chain.segments:
        if find_bone(entry.rig, segment) is None:
            raise ValueError("链中有手动删除的骨骼，请先同步检查。")
    return _create_segment(context, entry, chain)


@_authoring_change
def _remove_segments(context, entry, chain, indices, base):
    _check_mesh_mode(entry.object)
    _migrate_entry(context.scene, entry)
    ids = [chain.segments[index].bone_id for index in indices]
    names = []
    if entry.rig and entry.rig.type == 'ARMATURE':
        _check_entry(entry)
        for index in indices:
            bone = find_bone(entry.rig, chain.segments[index])
            if bone:
                names.append(bone.name)
        if names:
            with edit_rig(context, entry.rig) as bones:
                for name in reversed(names):
                    if bones.get(name):
                        bones.remove(bones[name])
    _remove_owned_groups(entry, ids, context.scene)
    for index in sorted(indices, reverse=True):
        chain.segments.remove(index)
    chain.active_segment_index = max(0, len(chain.segments) - 1)
    _write_owned(entry)


@_authoring_change
def remove_chain(context, entry, index):
    if not 0 <= index < len(entry.chains):
        raise ValueError("请先选择物理链。")
    chain = entry.chains[index]
    _remove_segments(context, entry, chain, list(range(len(chain.segments))),
                     context.scene.loyaltools_efmi_physics.base_bone_count)
    entry.chains.remove(index)
    entry.active_chain_index = max(0, min(index, len(entry.chains) - 1))
    _write_owned(entry)


@_authoring_change
def remove_segment(context, entry, chain):
    index = chain.active_segment_index
    if not 0 <= index < len(chain.segments):
        raise ValueError("请先选择骨段。")
    if index == 0:
        raise ValueError("父骨必须保留；需要删除整链时请使用删除链。")
    # Removing a middle segment removes its descendants, keeping a linear chain
    # and preserving every unrelated numeric ID and every original group.
    _remove_segments(context, entry, chain, list(range(index, len(chain.segments))),
                     context.scene.loyaltools_efmi_physics.base_bone_count)


def _rig_referenced_elsewhere(scene, entry):
    return any(other.object == entry.object or other.rig == entry.rig
               for other_scene in bpy.data.scenes
               for other in other_scene.loyaltools_efmi_physics.objects
               if other.as_pointer() != entry.as_pointer())


@_authoring_change
def remove_object(context, index):
    settings = context.scene.loyaltools_efmi_physics
    if not 0 <= index < len(settings.objects):
        raise ValueError("请先选择已登记物体。")
    entry = settings.objects[index]
    if not _rig_referenced_elsewhere(context.scene, entry):
        if entry.schema_version < 3:
            _migrate_entry(context.scene, entry)
        ids = [segment.bone_id for chain in entry.chains for segment in chain.segments]
        _remove_owned_groups(entry, ids, context.scene)
        if entry.object:
            for key in (OWNED_GROUPS, OWNED_GROUP_NAMES, OBJECT_UID):
                if key in entry.object:
                    del entry.object[key]
        if entry.rig and entry.rig.get(RIG_OWNER) == entry.uid:
            rig = entry.rig
            armature = rig.data
            if rig.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            bpy.data.objects.remove(rig, do_unlink=True)
            if armature.users == 0:
                bpy.data.armatures.remove(armature)
    removed_uid = entry.uid
    settings.objects.remove(index)
    _pending_cleanup.get(context.scene.as_pointer(), set()).discard(removed_uid)
    settings.active_object_index = max(0, min(index, len(settings.objects) - 1))


def _prune_deleted_bones(scene, entry):
    """Remove only records/groups whose actual bone UUID disappeared.

    No operators or edit-bone mutations are used here, making this safe to call
    after an armature dependency-graph update. Surviving bones and their weights
    stay intact, including when a parent/middle bone was deleted natively.
    """
    _check_entry(entry)
    _check_mesh_mode(entry.object)
    if entry.rig.mode == 'EDIT':
        return 0
    settings = scene.loyaltools_efmi_physics
    if not settings.enabled:
        return 0
    _migrate_entry(scene, entry)
    missing_by_chain = []
    for index, chain in enumerate(entry.chains):
        missing = [i for i, segment in enumerate(chain.segments)
                   if find_bone(entry.rig, segment) is None]
        if missing or not chain.segments:
            missing_by_chain.append((index, missing))
    if not missing_by_chain:
        return 0
    ids = [entry.chains[index].segments[i].bone_id
           for index, missing in missing_by_chain for i in missing]
    _remove_owned_groups(entry, ids, scene)
    for index, missing in reversed(missing_by_chain):
        chain = entry.chains[index]
        for segment_index in reversed(missing):
            chain.segments.remove(segment_index)
        chain.active_segment_index = min(chain.active_segment_index, max(0, len(chain.segments) - 1))
        if not chain.segments:
            entry.chains.remove(index)
    entry.active_chain_index = min(entry.active_chain_index, max(0, len(entry.chains) - 1))
    _write_owned(entry)
    return len(ids)


@_authoring_change
def synchronize(context):
    """Explicitly reconcile deletions. Never guess at renamed user groups."""
    settings = context.scene.loyaltools_efmi_physics
    profile = refresh_current_workspace(context.scene)
    changes = 0
    for index in reversed(range(len(settings.objects))):
        entry = settings.objects[index]
        if entry.object is None or entry.rig is None:
            remove_object(context, index)
            changes += 1
            continue
        if entry.rig.mode == 'EDIT':
            raise ValueError("请先退出骨架编辑模式，再同步已删除骨骼的顶点组。")
        changes += _sync_entry(context.scene, entry, profile)
        for chain in entry.chains:
            for segment in chain.segments:
                bone = find_bone(entry.rig, segment)
                if bone:
                    segment.bone_name = bone.name
        _write_owned(entry)
    settings.status = "同步完成：清理 " + str(changes) + " 处已删除的数据。"
    return changes


@_authoring_change
def _sync_entry(scene, entry, profile):
    if not scene.loyaltools_efmi_physics.enabled:
        return 0
    _check_entry(entry)
    _check_mesh_mode(entry.object)
    if entry.rig.mode == 'EDIT':
        return 0
    _migrate_entry(scene, entry)
    _check_entry_profile(entry, profile)
    # Extrude/duplicate can copy Blender bone custom properties. Preserve the
    # recorded source by its last known name and allocate fresh identities for
    # the new bones. Ambiguous copies are rejected rather than stealing weights.
    for chain in entry.chains:
        for segment in chain.segments:
            matches = [bone for bone in entry.rig.data.bones if bone.get(SEGMENT_UID) == segment.uid]
            if len(matches) > 1:
                source = next((bone for bone in matches if bone.name == segment.bone_name), None)
                if source is None:
                    raise ValueError("复制骨身份不明确，请恢复原骨名称后同步：" + segment.bone_name)
                for bone in matches:
                    if bone != source:
                        bone[SEGMENT_UID] = uid()
                        bone[CHAIN_UID] = ''
    changes = _prune_deleted_bones(scene, entry)
    known = {}
    for chain in entry.chains:
        for segment in chain.segments:
            bone = find_bone(entry.rig, segment)
            if bone:
                segment.bone_name = bone.name
                known[bone.as_pointer()] = chain
                if entry.object.vertex_groups.get(_segment_group(segment)) is None:
                    entry.object.vertex_groups.new(name=_segment_group(segment))
                    changes += 1
    pending = [bone for bone in entry.rig.data.bones if bone.as_pointer() not in known]
    while pending:
        progress = False
        for bone in list(pending):
            parent = bone.parent
            if parent and parent.as_pointer() not in known:
                continue
            if parent:
                chain = known[parent.as_pointer()]
                tail = find_bone(entry.rig, chain.segments[-1])
                if tail != parent:
                    raise ValueError("新增骨 " + bone.name + " 形成分支；每条物理链只能线性延伸，请调整父子关系。")
            else:
                groups = original_groups(entry.object, int(profile['bones_count']))
                if not groups:
                    raise ValueError("新增根骨没有可跟随的原合并组。")
                chain = entry.chains.add()
                chain.uid = uid()
                chain.name = bone.name
                chain.follow_group_id = groups[0]
            number = allocate_bone_id(scene)
            segment = chain.segments.add()
            segment.uid = uid()
            segment.bone_id = number
            segment.group_name = _group_name(number)
            segment.bone_name = bone.name
            bone[SEGMENT_UID] = segment.uid
            bone[CHAIN_UID] = chain.uid
            entry.object.vertex_groups.new(name=segment.group_name)
            known[bone.as_pointer()] = chain
            _write_owned(entry)
            pending.remove(bone)
            progress = True
            changes += 1
        if not progress:
            raise ValueError("新增物理骨的父子关系无法解析。")
    _write_owned(entry)
    return changes


def _process_managed_rig_update(scene, depsgraph):
    """Visit managed records only when an armature changed or cleanup is pending.

    Blender's EditBone data is provisional. Queue it until Object mode, and
    never mutate data during undo/redo reconstruction. This handler only queues
    managed updates; the capped service handles writes outside Edit Mode.
    """
    global _cleanup_running
    if _cleanup_running or _undo_in_progress or _authoring_depth:
        return
    settings = getattr(scene, 'loyaltools_efmi_physics', None)
    if settings is None or not settings.enabled or not settings.objects:
        return
    object_ids, armature_ids = set(), set()
    for update in depsgraph.updates:
        data = update.id.original
        if isinstance(data, bpy.types.Object) and data.type == 'ARMATURE' and data.get(RIG_MARKER):
            object_ids.add(data.as_pointer())
        elif isinstance(data, bpy.types.Armature):
            # Older saved authoring rigs may lack the data marker. The match
            # against a marked registered rig below is still authoritative.
            armature_ids.add(data.as_pointer())
    scene_id = scene.as_pointer()
    pending = _pending_cleanup.get(scene_id, set())
    if not object_ids and not armature_ids and not pending:
        return
    candidates = [entry.uid for entry in settings.objects if entry.rig and entry.object
                  and entry.rig.get(RIG_MARKER) and (entry.uid in pending
                  or entry.rig.as_pointer() in object_ids
                  or entry.rig.data.as_pointer() in armature_ids)]
    if not candidates:
        return
    pending.update(candidates)
    _pending_cleanup[scene_id] = pending


def _run_auto_sync(now=None):
    """One capped service pass; visits registered entries, never all meshes."""
    global _cleanup_running, _last_sync_time
    now = time.monotonic() if now is None else now
    if _cleanup_running or _undo_in_progress or _authoring_depth or now - _last_sync_time < 1.0:
        return
    _last_sync_time = now
    if not is_merged_workspace():
        return
    _cleanup_running = True
    try:
        profile = current_profile()
        for scene in bpy.data.scenes:
            settings = scene.loyaltools_efmi_physics
            if not settings.enabled:
                continue
            refresh_current_workspace(scene, profile)
            for entry in settings.objects:
                if not entry.object or not entry.rig or entry.object.mode != 'OBJECT' or entry.rig.mode == 'EDIT':
                    continue
                try:
                    changes = _sync_entry(scene, entry, profile)
                    if changes:
                        settings.status = "自动同步完成：更新 " + str(changes) + " 处骨段 / 所属组。"
                except Exception as exc:
                    settings.status = "自动同步未完成：" + str(exc)
            _pending_cleanup.pop(scene.as_pointer(), None)
    finally:
        _cleanup_running = False


def _sync_timer():
    if not _timer_running:
        return None
    scenes = getattr(bpy.data, 'scenes', None)
    if scenes is None:
        return 1.0
    if not any(scene.loyaltools_efmi_physics.enabled for scene in scenes):
        return None
    try:
        _run_auto_sync()
    except Exception as exc:
        print("LoyalTools EFMI physics sync: " + str(exc))
    return 1.0


def _deferred_bootstrap():
    """Resume saved master state after addon_utils leaves RestrictBlend context."""
    if not _timer_running:
        return None
    if getattr(bpy.data, 'scenes', None) is None:
        return 0.1
    if bpy.app.driver_namespace.get(_BOOTSTRAP_KEY) is _deferred_bootstrap:
        bpy.app.driver_namespace.pop(_BOOTSTRAP_KEY, None)
    _ensure_timer()
    return None


def _ensure_bootstrap():
    previous = bpy.app.driver_namespace.get(_BOOTSTRAP_KEY)
    if previous is not None and previous is not _deferred_bootstrap and bpy.app.timers.is_registered(previous):
        bpy.app.timers.unregister(previous)
    if not bpy.app.timers.is_registered(_deferred_bootstrap):
        bpy.app.timers.register(_deferred_bootstrap, first_interval=0.0)
    bpy.app.driver_namespace[_BOOTSTRAP_KEY] = _deferred_bootstrap


def _ensure_timer():
    if not _timer_running:
        return
    scenes = getattr(bpy.data, 'scenes', None)
    if scenes is None:
        # Blender deliberately hides bpy.data.scenes while enabling addons.
        # Register only a cancellable one-shot callback in that context.
        _ensure_bootstrap()
        return
    enabled = any(scene.loyaltools_efmi_physics.enabled for scene in scenes)
    previous = bpy.app.driver_namespace.get(_TIMER_KEY)
    if previous is not None and previous is not _sync_timer and bpy.app.timers.is_registered(previous):
        bpy.app.timers.unregister(previous)
    if enabled and not bpy.app.timers.is_registered(_sync_timer):
        bpy.app.timers.register(_sync_timer, first_interval=1.0, persistent=True)
    elif not enabled and bpy.app.timers.is_registered(_sync_timer):
        bpy.app.timers.unregister(_sync_timer)
    bpy.app.driver_namespace[_TIMER_KEY] = _sync_timer


def enabled_changed(scene):
    if scene is None or not hasattr(scene, 'loyaltools_efmi_physics'):
        return
    settings = scene.loyaltools_efmi_physics
    _pending_cleanup.pop(scene.as_pointer(), None)
    if settings.enabled and is_merged_workspace():
        errors = []
        try:
            refresh_current_workspace(scene)
        except Exception as exc:
            errors.append(str(exc))
        for entry in settings.objects:
            try:
                _migrate_entry(scene, entry)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            settings.status = "物理模块已启用，配置待修复：" + errors[0]
    _ensure_timer()


@persistent
def _managed_rig_update(scene, depsgraph):
    global _last_cleanup_error
    try:
        _process_managed_rig_update(scene, depsgraph)
    except Exception as exc:
        # Freed RNA or another addon's context changes must not escape into
        # Blender's event loop. Deduplicate diagnostics until the file changes.
        message = str(exc)
        if message != _last_cleanup_error:
            print("LoyalTools EFMI physics cleanup: " + message)
            _last_cleanup_error = message


@persistent
def _before_undo(_scene):
    global _undo_in_progress
    _undo_in_progress = True
    _pending_cleanup.clear()


@persistent
def _after_undo(_scene):
    global _undo_in_progress
    _pending_cleanup.clear()
    _undo_in_progress = False
    _ensure_timer()


@persistent
def _after_load(_unused):
    global _cleanup_running, _undo_in_progress, _last_cleanup_error, _last_sync_time
    _pending_cleanup.clear()
    _profile_available_cache.clear()
    _cleanup_running = _undo_in_progress = False
    _last_cleanup_error = None
    _last_sync_time = float('-inf')
    _ensure_timer()


_HANDLERS = ((bpy.app.handlers.depsgraph_update_post, _managed_rig_update, 'depsgraph'),
             (bpy.app.handlers.undo_pre, _before_undo, 'before_undo'),
             (bpy.app.handlers.undo_post, _after_undo, 'after_undo'),
             (bpy.app.handlers.redo_pre, _before_undo, 'before_undo'),
             (bpy.app.handlers.redo_post, _after_undo, 'after_undo'),
             (bpy.app.handlers.load_post, _after_load, 'load_post'))


def unregister_handlers():
    global _timer_running
    _timer_running = False
    previous = bpy.app.driver_namespace.pop(_TIMER_KEY, None)
    bootstrap = bpy.app.driver_namespace.pop(_BOOTSTRAP_KEY, None)
    for callback in {previous, _sync_timer, bootstrap, _deferred_bootstrap} - {None}:
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    for handlers, function, key in _HANDLERS:
        for existing in list(handlers):
            if getattr(existing, '_loyaltools_physics_handler', None) == key:
                handlers.remove(existing)
    _after_load(None)


def register_handlers():
    global _timer_running
    unregister_handlers()
    for handlers, function, key in _HANDLERS:
        function._loyaltools_physics_handler = key
        handlers.append(function)
    _timer_running = True
    _ensure_timer()


def chain_params(chain):
    return {"damping": float(chain.damping), "stiffness": float(chain.stiffness),
            "gravity": list(chain.gravity), "cone_angle": float(chain.cone_angle),
            "wind_strength": float(chain.wind_strength)}


def _matrix(matrix):
    return [list(row) for row in matrix]


def collect_export_records(scene, objects, profile):
    """Return validated, name-independent records for the active profile.

    ``objects`` is the set of original Blender source objects actually exported.
    Passing None explicitly collects all registered meshes. Enabled exports fail
    loudly on stale authoring records so deleted/renamed data is never guessed.
    Enabled collection refreshes profile metadata and upgrades proven legacy
    groups. Disabled collection returns immediately without reading object or
    profile data. Rest transforms use armature-local +Y bone convention.
    """
    settings = scene.loyaltools_efmi_physics
    result = {"enabled": bool(settings.enabled), "base_bone_count": settings.base_bone_count,
              "bone_count": 0, "bones": [], "chains": [], "meshes": []}
    if not settings.enabled:
        return result
    if not is_efmi():
        raise ValueError("物理骨仅支持 EFMI。")
    refresh_current_workspace(scene, profile)
    base = int(profile['bones_count'])
    result['base_bone_count'] = base
    export_objects = None if objects is None else list({obj.as_pointer(): obj for obj in objects}.values())
    if export_objects is not None:
        for target in export_objects:
            if any(entry.object == target for entry in settings.objects):
                continue
            source = next((entry for entry in settings.objects
                           if entry.uid == target.get(OBJECT_UID)), None)
            if source and is_managed_export_copy(source, target):
                continue
            if target.get(OBJECT_UID) and _owned_ids(target):
                raise ValueError(target.name + " 是未登记的物理网格副本，请先添加物体以分配独立骨编号。")
    seen_ids = set()
    seen_objects = set()
    animation_settings = getattr(scene, 'loyaltools_efmi_animation', None)
    animation_enabled = bool(animation_settings and animation_settings.enabled)
    for entry in settings.objects:
        obj = entry.object
        targets = [obj] if export_objects is None else [
            target for target in export_objects
            if obj is not None and (target == obj or is_managed_export_copy(entry, target))]
        if not targets:
            continue
        _check_entry(entry)
        _migrate_entry(scene, entry)
        if obj.as_pointer() in seen_objects:
            raise ValueError("物理登记中包含重复网格。")
        seen_objects.add(obj.as_pointer())
        target = targets[0]
        _check_entry_profile(entry, profile)
        for actual_target in targets:
            missing = [_segment_group(segment) for chain in entry.chains for segment in chain.segments
                       if actual_target.vertex_groups.get(_segment_group(segment)) is None]
            if missing:
                raise ValueError(actual_target.name + " 的实际导出网格缺少物理专属组：" + ', '.join(missing)
                                 + "；请检查蓝图前处理是否删除了这些组。")
        if obj.parent or obj.constraints or entry.rig.constraints or entry.rig.animation_data:
            raise ValueError(obj.name + " 的物理网格不支持父级/约束，骨架不支持约束/动画；请先应用静态变换。")
        for pose_bone in entry.rig.pose.bones:
            if pose_bone.constraints or any(
                    abs(pose_bone.matrix_basis[row][column] - (1.0 if row == column else 0.0)) > 1e-6
                    for row in range(4) for column in range(4)):
                raise ValueError(entry.rig.name + " 存在姿态模式变换或骨骼约束；请清除姿态并在编辑模式调整骨骼。")
        if not obj.get(OBJECT_MARKER_MERGED_SKELETON):
            raise ValueError(obj.name + " 不是 EFMI 合并骨骼网格。")
        for actual_target in targets:
            mesh_record = {"object_name": actual_target.name, "source_object_name": obj.name,
                           "rig_name": entry.rig.name, "vg_mapping": {}, "vg_index_mapping": {},
                           "object_matrix_world": _matrix(obj.matrix_world)}
            active_indices = {weight.group for vertex in actual_target.data.vertices for weight in vertex.groups
                              if weight.weight > 0}
            mesh_record['active_group_names'] = [group.name for group in actual_target.vertex_groups
                                                 if group.index in active_indices]
            for group in actual_target.vertex_groups:
                number = _numeric_group(group.name)
                if number is not None:
                    mesh_record["vg_mapping"][group.name] = number
                    mesh_record["vg_index_mapping"][group.index] = number
            for chain in entry.chains:
                for segment in chain.segments:
                    group = actual_target.vertex_groups.get(_segment_group(segment))
                    mesh_record['vg_mapping'][group.name] = segment.bone_id
                    mesh_record['vg_index_mapping'][group.index] = segment.bone_id
            result["meshes"].append(mesh_record)
        for chain in entry.chains:
            follow = int(chain.follow_group_id)
            animation_anchor_group = getattr(chain, 'animation_anchor_group', '')
            animation_source_anchor = None
            anchor_label = obj.name + '/' + chain.name
            if animation_anchor_group:
                if not animation_enabled:
                    raise ValueError(anchor_label + " uses an animation bone attachment. "
                                     "Enable EFMI Bone Animation or select a native bone attachment.")
                if follow != -1:
                    raise ValueError(anchor_label + " has both native and animation bone attachments. Select one attachment.")
                from ..efmi_animation.rig import require_follow_component, resolve_animation_anchor

                animation, animation_root, _ = resolve_animation_anchor(scene, obj, animation_anchor_group)
                require_follow_component(animation.object, animation_root, profile)
                animation_source_anchor = int(animation_root.follow_group_id)
            elif not 0 <= follow < base or obj.vertex_groups.get(str(follow)) is None:
                raise ValueError(obj.name + "/" + chain.name + " 跟随组必须是已存在的原合并骨编号 (0..B-1)。")
            if not chain.segments:
                raise ValueError(chain.name + " 没有父骨，请同步检查。")
            previous = None
            bone_ids = []
            root_head = None
            for segment in chain.segments:
                number = segment.bone_id
                if number < AUTHORING_ID_START or number > 2147483647 or number in seen_ids:
                    raise ValueError("物理骨全局编号无效或重复: " + str(number))
                seen_ids.add(number)
                group = obj.vertex_groups.get(_segment_group(segment))
                if group is None:
                    raise ValueError("物理顶点组 " + _segment_group(segment) + " 已删除或改名，请同步重建。")
                bone = find_bone(entry.rig, segment)
                if bone is None:
                    raise ValueError(chain.name + " 有骨骼被手动删除，请同步检查。")
                if bone.parent != previous:
                    raise ValueError(chain.name + " 必须是父骨加线性子骨；请在编辑模式修正父子关系。")
                if bone.length <= 1e-6:
                    raise ValueError("物理骨长度必须大于零: " + bone.name)
                if bone.get(CHAIN_UID) != chain.uid:
                    raise ValueError("物理骨链身份不匹配: " + bone.name)
                if previous and (bone.head_local - previous.tail_local).length > 1e-5:
                    raise ValueError("子骨头部必须接在上一段尾部: " + bone.name)
                if root_head is None:
                    root_head = list(bone.head_local)
                result["bones"].append({
                    "global_id": number, "chain_id": chain.uid,
                    "group_name": _segment_group(segment),
                    "parent_global_id": bone_ids[-1] if bone_ids else follow,
                    "anchor_global_id": follow, "rig_name": entry.rig.name,
                    "object_name": target.name, "source_object_name": obj.name, "bone_name": bone.name,
                    "rest_matrix": _matrix(bone.matrix_local),
                    "rig_matrix_world": _matrix(entry.rig.matrix_world),
                    "head": list(bone.head_local), "tail": list(bone.tail_local),
                    "root_head": root_head, "length": float(bone.length),
                    "params": chain_params(chain),
                })
                bone_ids.append(number)
                previous = bone
            chain_record = {"chain_id": chain.uid, "name": chain.name,
                            "object_name": target.name, "source_object_name": obj.name, "rig_name": entry.rig.name,
                            "follow_group_id": follow, "bone_ids": bone_ids,
                            "root_head": root_head, "params": chain_params(chain)}
            if animation_anchor_group:
                chain_record.update(animation_anchor_group=animation_anchor_group,
                                    animation_source_anchor=animation_source_anchor)
            result["chains"].append(chain_record)
    result["bone_count"] = len(seen_ids)
    return result


def collect_physics_export(context, profile, exported_objects=None):
    return collect_export_records(context.scene, exported_objects, profile)
