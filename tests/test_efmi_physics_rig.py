"""Run: blender --background --factory-startup --python this_file.py.

Creates only a disposable test scene/file; it neither installs the addon nor
touches a user's Blender preferences. The addon package is imported as a test
namespace so root registration and optional updater dependencies are excluded.
"""

import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

try:
    import bpy
except ImportError:
    raise unittest.SkipTest('requires Blender 4.5')
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "loyaltools_physics_test"
namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = namespace
physics = importlib.import_module(PACKAGE + ".efmi_physics")
rig = importlib.import_module(PACKAGE + ".efmi_physics.rig")


def mesh(name):
    data = bpy.data.meshes.new(name)
    data.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj[rig.OBJECT_MARKER_MERGED_SKELETON] = True
    for number in range(4):
        obj.vertex_groups.new(name=str(number))
    obj.vertex_groups['0'].add([0, 1, 2], 1.0, 'REPLACE')
    return obj


def expect_error(callback, contains):
    try:
        callback()
    except ValueError as exc:
        assert contains in str(exc), (contains, str(exc))
    else:
        raise AssertionError("Expected ValueError: " + contains)


def auto_tick():
    rig._run_auto_sync(now=max(time.monotonic(), rig._last_sync_time + 1.1))


def main():
    physics.register()
    rig.GlobalConfig.logic_name = rig.LogicName.EFMI
    workspace = tempfile.TemporaryDirectory(prefix='loyaltools_physics_workspace_')
    rig.GlobalConfig.path_workspace_folder = classmethod(lambda cls: workspace.name)
    assert not rig.is_merged_workspace()
    assert physics.panel.EFMI_PT_Physics.poll(bpy.context)  # Master switch remains visible.
    path = Path(rig.get_profile_path(workspace.name))
    path.write_text(json.dumps({'mode': 'NORMAL'}), encoding='utf-8')
    assert not rig.is_merged_workspace()
    path.write_text(json.dumps({'format_version': 1, 'mode': 'EFMI_MERGED_SKELETON',
        'components': [{'component_id': 0, 'unique_str': 'test', 'vg_count': 4,
                        'vg_map': {'0': 0, '1': 1, '2': 2, '3': 3}}]}), encoding='utf-8')
    profile = rig.current_profile()
    assert rig.is_merged_workspace()
    assert physics.panel.EFMI_PT_Physics.poll(bpy.context)
    assert not physics.operators.EFMI_OT_PhysicsAddObject.poll(bpy.context)
    rig.register_handlers()
    rig.register_handlers()
    for handlers, function, key in rig._HANDLERS:
        assert sum(getattr(item, '_loyaltools_physics_handler', None) == key for item in handlers) == 1
    scene = bpy.context.scene
    settings = scene.loyaltools_efmi_physics
    assert not settings.enabled
    assert rig.collect_export_records(scene, None, profile)['bones'] == []
    settings.enabled = True
    obj = mesh("Original")
    obj.matrix_world = Matrix.Translation((3, 4, 5)) @ Matrix.Rotation(0.5, 4, 'Z') @ Matrix.Diagonal((2, 1, 1, 1))
    entry = rig.add_object(bpy.context, obj, profile)
    assert entry.rig.get(rig.RIG_MARKER)
    assert not list(obj.modifiers)
    assert (entry.rig.matrix_world.translation - obj.matrix_world.translation).length < 1e-6
    chain = rig.add_chain(bpy.context, entry, profile)
    root_id = chain.segments[0].bone_id
    assert root_id == rig.AUTHORING_ID_START
    rig.add_segment(bpy.context, entry, chain, profile)
    rig.add_segment(bpy.context, entry, chain, profile)
    assert [seg.bone_id for seg in chain.segments] == list(range(rig.AUTHORING_ID_START, rig.AUTHORING_ID_START + 3))
    for segment in chain.segments:
        obj.vertex_groups[rig._segment_group(segment)].add([0], 0.25, 'REPLACE')
    other_chain = rig.add_chain(bpy.context, entry, profile)
    other_id = other_chain.segments[0].bone_id
    assert other_id == rig.AUTHORING_ID_START + 3
    settings.enabled = True
    records = rig.collect_export_records(scene, [obj], profile)
    assert len(records['bones']) == 4
    assert records['bones'][0]['parent_global_id'] == 0
    assert records['bones'][1]['parent_global_id'] == root_id
    assert records['meshes'][0]['vg_index_mapping'][obj.vertex_groups[rig._group_name(other_id)].index] == other_id
    assert records['bones'][0]['rig_matrix_world'][0][3] == 3
    # Real edit-bone changes, including roll, survive collection and renaming.
    root = rig.find_bone(entry.rig, chain.segments[0])
    root_name = root.name
    with rig.edit_rig(bpy.context, entry.rig) as bones:
        bones[root_name].roll = math.radians(33)
    original_matrix = rig.find_bone(entry.rig, chain.segments[0]).matrix_local.copy()
    rig.find_bone(entry.rig, chain.segments[0]).name = "Renamed root"
    obj.name = "Renamed mesh"
    records = rig.collect_export_records(scene, [obj], profile)
    assert records['bones'][0]['bone_name'] == "Renamed root"
    assert Matrix(records['bones'][0]['rest_matrix']) == original_matrix
    # Delete a middle segment and its descendants, preserving all other IDs.
    chain.active_segment_index = 1
    rig.remove_segment(bpy.context, entry, chain)
    assert len(chain.segments) == 1
    assert obj.vertex_groups.get(rig._group_name(root_id + 1)) is None and obj.vertex_groups.get(rig._group_name(root_id + 2)) is None
    assert obj.vertex_groups.get(rig._group_name(other_id)) is not None
    assert all(obj.vertex_groups.get(str(i)) for i in range(4))
    records = rig.collect_export_records(scene, [obj], profile)
    assert records['meshes'][0]['vg_index_mapping'][obj.vertex_groups[rig._group_name(other_id)].index] == other_id
    assert obj.vertex_groups[rig._group_name(other_id)].index != other_id
    # Deleted IDs are never silently recycled.
    new_segment = rig.add_segment(bpy.context, entry, chain, profile)
    assert new_segment.bone_id == rig.AUTHORING_ID_START + 4
    # Wrong original anchor is rejected, and other workflows remain hidden.
    chain.follow_group_id = 8
    expect_error(lambda: rig.collect_export_records(scene, [obj], profile), "原合并骨")
    chain.follow_group_id = 0
    rig.GlobalConfig.logic_name = rig.LogicName.ZZMI
    assert physics.panel.EFMI_PT_Physics.poll(bpy.context)
    assert not rig.is_merged_workspace()
    assert not physics.operators.EFMI_OT_PhysicsAddChain.poll(bpy.context)
    expect_error(lambda: rig.collect_export_records(scene, [obj], profile), "仅支持 EFMI")
    rig.GlobalConfig.logic_name = rig.LogicName.EFMI
    # Only tagged preprocessing copies may resolve back to a source registry.
    copy = obj.copy()
    copy.data = obj.data.copy()
    scene.collection.objects.link(copy)
    expect_error(lambda: rig.collect_export_records(scene, [copy], profile), "未登记")
    copy['_ssmt_preprocess_source'] = obj.name
    copy['_ssmt_preprocess_requested'] = obj.name
    copy['_ssmt_preprocess_hash'] = "testhash"
    preprocessing = importlib.import_module(PACKAGE + '.blueprint.preprocess')
    preprocessing.PreProcessHelper.register_copy_result(obj.name, copy.name)
    records = rig.collect_export_records(scene, [copy], profile)
    assert records['meshes'][0]['object_name'] == copy.name
    assert records['meshes'][0]['source_object_name'] == obj.name
    bpy.data.objects.remove(copy, do_unlink=True)
    # Linked duplicates are made single-user before inherited physical groups
    # are cleared; source vertex weights remain untouched.
    duplicate = obj.copy()
    scene.collection.objects.link(duplicate)
    original_data = obj.data
    duplicate_entry = rig.add_object(bpy.context, duplicate, profile)
    assert duplicate.data != original_data
    assert duplicate.vertex_groups.get(rig._group_name(root_id)) is None
    assert obj.vertex_groups[rig._group_name(root_id)].weight(0) == 0.25
    duplicate_chain = rig.add_chain(bpy.context, duplicate_entry, profile)
    assert duplicate_chain.segments[0].bone_id > rig.AUTHORING_ID_START + 4
    assert duplicate_entry.rig != entry.rig
    # Save/reopen retains PointerProperties and bone UUIDs without handlers.
    obj_name, rig_name = obj.name, entry.rig.name
    with tempfile.TemporaryDirectory(prefix="loyaltools_physics_") as directory:
        path = os.path.join(directory, 'physics_test.blend')
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.open_mainfile(filepath=path)
        scene = bpy.context.scene
        settings = scene.loyaltools_efmi_physics
        entry = settings.objects[0]
        chain = entry.chains[0]
        assert entry.object == bpy.data.objects[obj_name]
        assert entry.rig == bpy.data.objects[rig_name]
        assert rig.find_bone(entry.rig, chain.segments[0]).name == "Renamed root"
        # Manual child deletion is detected, then explicitly synchronized.
        name = rig.find_bone(entry.rig, chain.segments[1]).name
        with rig.edit_rig(bpy.context, entry.rig) as bones:
            bones.remove(bones[name])
        bpy.context.view_layer.update()
        auto_tick()
        assert len(chain.segments) == 1
        rig.synchronize(bpy.context)
        assert len(chain.segments) == 1
        assert entry.object.vertex_groups.get(rig._group_name(rig.AUTHORING_ID_START + 4)) is None
        # A removed rig triggers safe cleanup; original groups stay intact.
        duplicate_entry = settings.objects[1]
        duplicate_obj = duplicate_entry.object
        bpy.data.objects.remove(duplicate_entry.rig, do_unlink=True)
        rig.synchronize(bpy.context)
        assert len(settings.objects) == 1
        assert all(duplicate_obj.vertex_groups.get(str(i)) for i in range(4))
        rig.remove_object(bpy.context, 0)
        assert not settings.objects
        assert all(bpy.data.objects[obj_name].vertex_groups.get(str(i)) for i in range(4))
    # Exercise the public operators and a real Blender undo snapshot. Background
    # execution needs explicit undo_push, unlike the interactive event loop.
    operator_obj = mesh('OperatorMesh')
    operator_name = operator_obj.name
    bpy.context.view_layer.objects.active = operator_obj
    operator_obj.select_set(True)
    rig.current_profile = lambda: profile
    assert bpy.ops.efmi.physics_add_object() == {'FINISHED'}
    assert bpy.ops.efmi.physics_add_chain() == {'FINISHED'}
    assert bpy.ops.efmi.physics_add_segment() == {'FINISHED'}
    settings = bpy.context.scene.loyaltools_efmi_physics
    assert len(settings.objects[0].chains[0].segments) == 2
    assert bpy.ops.efmi.physics_follow_group(group_id='1') == {'FINISHED'}
    assert settings.objects[0].chains[0].follow_group_id == 1
    assert bpy.ops.efmi.physics_edit_rig() == {'FINISHED'}
    assert bpy.context.mode == 'EDIT_ARMATURE'
    assert rig.collect_export_records(bpy.context.scene, None, profile)['bones']
    assert bpy.ops.efmi.physics_select_mesh() == {'FINISHED'}
    assert bpy.context.mode == 'OBJECT'
    assert bpy.context.active_object.name == operator_name
    active_id = settings.objects[0].chains[0].segments[1].bone_id
    assert bpy.context.active_object.vertex_groups.active.name == rig._group_name(active_id)
    assert bpy.ops.efmi.physics_validate() == {'FINISHED'}
    bpy.ops.ed.undo_push(message='Physics before deletion')
    assert bpy.ops.efmi.physics_remove_segment() == {'FINISHED'}
    assert len(settings.objects[0].chains[0].segments) == 1
    bpy.ops.ed.undo_push(message='Physics after deletion')
    assert bpy.ops.ed.undo() == {'FINISHED'}
    settings = bpy.context.scene.loyaltools_efmi_physics
    assert len(settings.objects[0].chains[0].segments) == 2
    restored_id = settings.objects[0].chains[0].segments[1].bone_id
    assert bpy.data.objects[operator_name].vertex_groups.get(rig._group_name(restored_id))
    assert rig.find_bone(settings.objects[0].rig, settings.objects[0].chains[0].segments[1])
    # A real native Edit Mode Delete is deferred until Object mode, and undo
    # restores the bone, registry entry and weight group together.
    entry = settings.objects[0]
    chain = entry.chains[0]
    root_id = chain.segments[0].bone_id
    child_id = chain.segments[1].bone_id
    entry.object.vertex_groups[rig._group_name(root_id)].add([0], 0.6, 'REPLACE')
    entry.object.vertex_groups[rig._group_name(child_id)].add([0], 0.4, 'REPLACE')
    bpy.ops.ed.undo_push(message='Before native child delete')
    chain.active_segment_index = 1
    assert bpy.ops.efmi.physics_edit_rig() == {'FINISHED'}
    assert bpy.ops.armature.delete() == {'FINISHED'}
    bpy.context.view_layer.update()
    assert entry.object.vertex_groups.get(rig._group_name(child_id))  # Deferred in Edit Mode.
    assert bpy.ops.object.mode_set(mode='OBJECT') == {'FINISHED'}
    bpy.context.view_layer.update()
    auto_tick()
    assert len(chain.segments) == 1
    assert entry.object.vertex_groups.get(rig._group_name(child_id)) is None
    assert abs(entry.object.vertex_groups[rig._group_name(root_id)].weight(0) - 0.6) < 1e-6
    assert entry.object.vertex_groups['0'].weight(0) == 1.0
    bpy.ops.ed.undo_push(message='After native child delete')
    assert bpy.ops.ed.undo() == {'FINISHED'}
    settings = bpy.context.scene.loyaltools_efmi_physics
    entry = settings.objects[0]
    chain = entry.chains[0]
    assert len(chain.segments) == 2
    assert rig.find_bone(entry.rig, chain.segments[1])
    assert abs(entry.object.vertex_groups[rig._group_name(child_id)].weight(0) - 0.4) < 1e-6
    # Native parent deletion preserves the surviving child ID and weights, then
    # undo brings the parent and its exact original ownership back.
    chain.active_segment_index = 0
    bpy.ops.ed.undo_push(message='Before native parent delete')
    assert bpy.ops.efmi.physics_edit_rig() == {'FINISHED'}
    assert bpy.ops.armature.delete() == {'FINISHED'}
    assert bpy.ops.object.mode_set(mode='OBJECT') == {'FINISHED'}
    bpy.context.view_layer.update()
    auto_tick()
    assert len(chain.segments) == 1
    assert chain.segments[0].bone_id == child_id
    assert entry.object.vertex_groups.get(rig._group_name(root_id)) is None
    assert abs(entry.object.vertex_groups[rig._group_name(child_id)].weight(0) - 0.4) < 1e-6
    assert entry.object.vertex_groups['0'].weight(0) == 1.0
    bpy.ops.ed.undo_push(message='After native parent delete')
    assert bpy.ops.ed.undo() == {'FINISHED'}
    settings = bpy.context.scene.loyaltools_efmi_physics
    chain = settings.objects[0].chains[0]
    assert [segment.bone_id for segment in chain.segments] == [root_id, child_id]
    assert rig.find_bone(settings.objects[0].rig, chain.segments[0])
    assert abs(settings.objects[0].object.vertex_groups[rig._group_name(root_id)].weight(0) - 0.6) < 1e-6
    assert bpy.ops.ed.redo() == {'FINISHED'}
    settings = bpy.context.scene.loyaltools_efmi_physics
    assert [segment.bone_id for segment in settings.objects[0].chains[0].segments] == [child_id]
    assert settings.objects[0].object.vertex_groups.get(rig._group_name(root_id)) is None
    assert bpy.ops.ed.undo() == {'FINISHED'}
    settings = bpy.context.scene.loyaltools_efmi_physics
    chain = settings.objects[0].chains[0]
    assert [segment.bone_id for segment in chain.segments] == [root_id, child_id]
    # Undo while still inside Edit Mode never removes/recreates weight groups.
    chain.active_segment_index = 1
    assert bpy.ops.efmi.physics_edit_rig() == {'FINISHED'}
    bpy.ops.ed.undo_push(message='Before Edit Mode deletion')
    assert bpy.ops.armature.delete() == {'FINISHED'}
    bpy.ops.ed.undo_push(message='After Edit Mode deletion')
    assert bpy.ops.ed.undo() == {'FINISHED'}
    assert bpy.context.mode == 'EDIT_ARMATURE'
    assert bpy.ops.object.mode_set(mode='OBJECT') == {'FINISHED'}
    bpy.context.view_layer.update()
    settings = bpy.context.scene.loyaltools_efmi_physics
    chain = settings.objects[0].chains[0]
    assert [segment.bone_id for segment in chain.segments] == [root_id, child_id]
    assert rig.find_bone(settings.objects[0].rig, chain.segments[1])
    assert abs(settings.objects[0].object.vertex_groups[rig._group_name(child_id)].weight(0) - 0.4) < 1e-6
    physics.unregister()
    for handlers, function, key in rig._HANDLERS:
        assert not any(getattr(item, '_loyaltools_physics_handler', None) == key for item in handlers)
    workspace.cleanup()
    print("EFMI_PHYSICS_RIG_TESTS_OK")


if __name__ == '__main__':
    main()
