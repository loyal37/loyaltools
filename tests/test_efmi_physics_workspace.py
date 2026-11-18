"""Blender integration: workspace changes, legacy migration and capped service."""

import json
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import bpy
except ImportError:
    raise unittest.SkipTest('requires Blender 4.5')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_efmi_physics_rig import physics, rig, mesh, expect_error, auto_tick


def write_profile(folder, base):
    Path(folder).mkdir(parents=True, exist_ok=True)
    Path(rig.get_profile_path(folder)).write_text(json.dumps({
        'format_version': 1, 'mode': 'EFMI_MERGED_SKELETON',
        'components': [{'component_id': 0, 'unique_str': 'same_imported_asset',
                        'vg_count': base, 'vg_map': {str(i): i for i in range(base)}}]
    }), encoding='utf-8')


def add_native_bone(entry, name, parent_name=None):
    with rig.edit_rig(bpy.context, entry.rig) as bones:
        bone = bones.new(name)
        if parent_name:
            parent = bones[parent_name]
            bone.parent = parent
            bone.use_connect = True
            bone.head = parent.tail
            bone.tail = parent.tail + (parent.tail - parent.head)
        else:
            bone.head = (0, 0, 1)
            bone.tail = (0, 0, 2)


def main():
    globals_module = importlib.import_module(rig.__package__.rsplit('.', 1)[0] + '.common.global_properties')
    export_data = importlib.import_module(rig.__package__.rsplit('.', 1)[0] + '.common.efmi_physics_export')
    globals_module.register()
    physics.register()
    rig.GlobalConfig.logic_name = rig.LogicName.EFMI
    with tempfile.TemporaryDirectory(prefix='physics_workspace_') as directory:
        a, b, c = (str(Path(directory) / name) for name in ('A', 'B', 'C'))
        write_profile(a, 4)
        write_profile(b, 550)
        write_profile(c, 3)
        current = {'path': b}
        rig.GlobalConfig.path_workspace_folder = classmethod(lambda cls: current['path'])
        scene = bpy.context.scene
        settings = scene.loyaltools_efmi_physics
        obj = mesh('ManagedMesh')
        native_names = [group.name for group in obj.vertex_groups]
        assert not settings.enabled
        settings.base_bone_count = 99
        settings.workspace_key = 'stale'
        assert rig.add_object(bpy.context, obj, rig.current_profile()) is None
        assert not settings.objects
        assert rig.refresh_current_workspace(scene) is None
        assert settings.base_bone_count == 99
        assert not bpy.app.timers.is_registered(rig._sync_timer)
        assert rig.collect_export_records(scene, None, None)['enabled'] is False
        assert settings.base_bone_count == 99  # Off does not even read the profile.
        # A mistaken workspace before enrollment does not latch its B forever.
        settings.enabled = True
        assert settings.base_bone_count == 550
        current['path'] = a
        rig.refresh_current_workspace(scene)
        assert settings.base_bone_count == 4
        profile = rig.current_profile()
        entry = rig.add_object(bpy.context, obj, profile)
        chain = rig.add_chain(bpy.context, entry, profile)
        segment = chain.segments[0]
        stable_id, stable_name = segment.bone_id, segment.group_name
        obj.vertex_groups[stable_name].add([0], 0.3, 'REPLACE')
        obj.vertex_groups['0'].add([0], 0.7, 'REPLACE')
        assert stable_id >= 65536 and stable_name.startswith('EFMI_PHYS_')
        assert entry.created_base_bone_count == 4
        # Current B changes in both directions, without renaming or touching any
        # native group or changing authoring IDs/weights.
        for folder, base in ((b, 550), (c, 3), (a, 4)):
            current['path'] = folder
            auto_tick()
            records = rig.collect_export_records(scene, [obj], rig.current_profile())
            assert settings.base_bone_count == base
            assert records['base_bone_count'] == base
            assert records['bones'][0]['global_id'] == stable_id
            assert records['bones'][0]['group_name'] == stable_name
            assert records['meshes'][0]['vg_mapping'][stable_name] == stable_id
            assert '3' not in records['meshes'][0]['active_group_names']
            plan = export_data.prepare_physics_plan(records, rig.current_profile())
            export_data.validate_object_weights([obj], plan)
            if base == 3:
                assert obj.vertex_groups.get('3') and '3' not in plan['meshes'][0]['palette_mapping']
            assert abs(obj.vertex_groups[stable_name].weight(0) - 0.3) < 1e-6
            assert all(obj.vertex_groups.get(name) for name in native_names)
        # Exercise the actual serial uncached entry point. It deliberately tags
        # empty hashes. A posed preview would deform geometry if baked, so exact
        # unchanged copy coordinates also prove the real strip path ran.
        preprocessing = importlib.import_module(rig.__package__.rsplit('.', 1)[0] + '.blueprint.preprocess')
        helper = preprocessing.PreProcessHelper
        scene.global_properties.apply_all_modifiers = True
        scene.global_properties.enable_non_mirror_workflow = False
        root_bone = rig.find_bone(entry.rig, segment)
        root_bone.name = stable_name
        segment.bone_name = root_bone.name
        preview = obj.modifiers.new('Managed physics preview', 'ARMATURE')
        preview.object = entry.rig
        entry.rig.pose.bones[root_bone.name].location.x = 0.5
        bpy.context.view_layer.update()
        original_coords = [tuple(vertex.co) for vertex in obj.data.vertices]
        for requested in (obj.name, 'abcdef12-3-0.' + obj.name):
            helper.reset_runtime_state()
            helper._execute_preprocess_without_cache([requested])
            target = bpy.data.objects[helper.original_to_copy_map[requested]]
            assert target.get('_ssmt_preprocess_hash') == ''
            assert target.get('_ssmt_preprocess_requested') == requested
            assert target.get('_ssmt_preprocess_source') == obj.name
            assert rig.is_managed_export_copy(entry, target)
            assert rig.managed_physics_entry(scene, target, entry.rig) == entry
            assert len(target.modifiers) == 0 and len(obj.modifiers) == 1
            assert [tuple(vertex.co) for vertex in target.data.vertices] == original_coords
            # Collection validates rest authoring, so clear the temporary pose
            # after testing preview stripping and before collecting the copy.
            entry.rig.pose.bones[root_bone.name].location.x = 0
            records = rig.collect_export_records(scene, [target], rig.current_profile())
            assert records['meshes'][0]['object_name'] == target.name
            target.vertex_groups.remove(target.vertex_groups[stable_name])
            expect_error(lambda: rig.collect_export_records(scene, [target], rig.current_profile()), '实际导出网格缺少')
            impostor = target.copy()
            impostor.data = target.data.copy()
            scene.collection.objects.link(impostor)
            assert not rig.is_managed_export_copy(entry, impostor)
            expect_error(lambda: rig.collect_export_records(scene, [impostor], rig.current_profile()), '未登记')
            bpy.data.objects.remove(impostor, do_unlink=True)
            bpy.data.objects.remove(target, do_unlink=True)
            entry.rig.pose.bones[root_bone.name].location.x = 0.5
        entry.rig.pose.bones[root_bone.name].location.x = 0
        obj.modifiers.remove(preview)
        helper.reset_runtime_state()
        # Two real prefix aliases can reuse one editable mesh. They share one
        # physical palette but each export target retains its own group map.
        aliases = [obj.name, 'abcdef12-3-0.' + obj.name]
        helper._execute_preprocess_without_cache(aliases)
        targets = [bpy.data.objects[helper.original_to_copy_map[name]] for name in aliases]
        records = rig.collect_export_records(scene, targets + targets[:1], rig.current_profile())
        assert len(records['meshes']) == 2 and len(records['bones']) == 1 and len(records['chains']) == 1
        plan = export_data.prepare_physics_plan(records, rig.current_profile())
        export_data.validate_object_weights(targets, plan)
        assert len({mesh['palette_mapping'][stable_name] for mesh in plan['meshes']}) == 1
        for target in targets:
            bpy.data.objects.remove(target, do_unlink=True)
        helper.reset_runtime_state()
        # Unselected broken registrations must not block a healthy target.
        broken_obj = mesh('Other workspace broken object')
        broken = rig.add_object(bpy.context, broken_obj, rig.current_profile())
        rig.add_chain(bpy.context, broken, rig.current_profile())
        bpy.data.objects.remove(broken.rig, do_unlink=True)
        assert rig.collect_export_records(scene, [obj], rig.current_profile())['bones']
        expect_error(lambda: rig.collect_export_records(scene, None, rig.current_profile()), '骨架已删除')
        # A bad migration also cannot prevent immediate workspace-B refresh or
        # stop the timer from processing valid registered objects.
        settings.enabled = False
        current['path'] = b
        settings.enabled = True
        assert settings.base_bone_count == 550
        auto_tick()
        assert rig.collect_export_records(scene, [obj], rig.current_profile())['base_bone_count'] == 550
        rig.remove_object(bpy.context, 1)
        current['path'] = a
        rig.refresh_current_workspace(scene)
        # The timer is unique and capped at one registered-object pass/second.
        rig.register_handlers()
        rig.register_handlers()
        assert bpy.app.timers.is_registered(rig._sync_timer)
        assert bpy.app.driver_namespace[rig._TIMER_KEY] is rig._sync_timer
        unmanaged = mesh('UnmanagedMesh')
        unmanaged.vertex_groups.new(name='EFMI_PHYS_999999').add([0], 0.9, 'REPLACE')
        untouched = [(group.name, group.index) for group in unmanaged.vertex_groups]
        calls = []
        original_sync = rig._sync_entry
        def traced(scene, candidate, profile):
            calls.append(candidate.object.name)
            return original_sync(scene, candidate, profile)
        rig._sync_entry = traced
        rig._last_sync_time = float('-inf')
        rig._run_auto_sync(now=100)
        rig._run_auto_sync(now=100.5)
        rig._run_auto_sync(now=101)
        assert calls == [obj.name, obj.name]
        assert [(group.name, group.index) for group in unmanaged.vertex_groups] == untouched
        rig._sync_entry = original_sync
        # Native bone creation gets an independent prefixed group on the pass.
        root_bone = rig.find_bone(entry.rig, segment)
        add_native_bone(entry, 'Native child', root_bone.name)
        assert len(chain.segments) == 1
        auto_tick()
        assert len(chain.segments) == 2
        child = chain.segments[1]
        assert child.bone_name == 'Native child'
        assert obj.vertex_groups.get(child.group_name)
        # Existing owned groups accidentally removed by a user are recreated.
        obj.vertex_groups.remove(obj.vertex_groups[child.group_name])
        auto_tick()
        assert obj.vertex_groups.get(child.group_name)
        # Blender's native extrusion may copy the source bone custom UID. The
        # source keeps its ownership; the new linear child gets a fresh group.
        bpy.ops.ed.undo_push(message='Before native extrusion')
        with rig.edit_rig(bpy.context, entry.rig) as bones:
            for bone in bones:
                bone.select = bone.select_head = bone.select_tail = False
            active = bones[child.bone_name]
            active.select_tail = True
            bones.active = active
            assert bpy.ops.armature.extrude_move(TRANSFORM_OT_translate={'value': (0, 0, 0.2)}) == {'FINISHED'}
        auto_tick()
        assert len(chain.segments) == 3
        extruded_group = chain.segments[2].group_name
        assert obj.vertex_groups.get(extruded_group)
        bpy.ops.ed.undo_push(message='After native extrusion')
        assert bpy.ops.ed.undo() == {'FINISHED'}
        scene = bpy.context.scene
        settings = scene.loyaltools_efmi_physics
        entry = settings.objects[0]
        obj = entry.object
        chain = entry.chains[0]
        child = chain.segments[1]
        assert len(chain.segments) == 2 and obj.vertex_groups.get(extruded_group) is None
        assert bpy.ops.ed.redo() == {'FINISHED'}
        scene = bpy.context.scene
        settings = scene.loyaltools_efmi_physics
        entry = settings.objects[0]
        obj = entry.object
        chain = entry.chains[0]
        child = chain.segments[1]
        assert len(chain.segments) == 3 and obj.vertex_groups.get(extruded_group)
        # Off retains data and blocks every automatic and explicit path. Native
        # deletes/new bones while off are reconciled only after re-enabling.
        child_name = child.group_name
        child_bone = child.bone_name
        settings.enabled = False
        assert not bpy.app.timers.is_registered(rig._sync_timer)
        before = [(group.name, group.index) for group in obj.vertex_groups]
        with rig.edit_rig(bpy.context, entry.rig) as bones:
            bones.remove(bones[child_bone])
        add_native_bone(entry, 'Added while off')
        current['path'] = b
        auto_tick()
        assert settings.base_bone_count == 4
        assert rig.synchronize(bpy.context) is None
        assert rig.add_chain(bpy.context, entry, rig.current_profile()) is None
        assert [(group.name, group.index) for group in obj.vertex_groups] == before
        assert len(chain.segments) == 3
        assert rig.managed_physics_entry(scene, object(), object()) is None
        assert not physics.operators.EFMI_OT_PhysicsAddChain.poll(bpy.context)
        settings.enabled = True
        assert settings.base_bone_count == 550
        assert obj.vertex_groups.get(child_name)  # On itself doesn't delete.
        auto_tick()
        assert obj.vertex_groups.get(child_name) is None
        assert any(s.bone_name == 'Added while off' for ch in entry.chains for s in ch.segments)
        assert all(obj.vertex_groups.get(name) for name in native_names)
        # Reproduce an old v1 file. Only the four matching ownership records
        # authorize renaming its numeric physical group, regardless of new B.
        current['path'] = a
        rig.refresh_current_workspace(scene)
        settings.enabled = False
        legacy = mesh('LegacyMesh')
        settings.enabled = True
        old_entry = rig.add_object(bpy.context, legacy, rig.current_profile())
        old_chain = rig.add_chain(bpy.context, old_entry, rig.current_profile())
        old_segment = old_chain.segments[0]
        legacy.vertex_groups[old_segment.group_name].add([0], 0.45, 'REPLACE')
        uv = legacy.data.uv_layers.new(name='UVMap')
        uv.data[0].uv = (0.2, 0.8)
        uv_before = [tuple(loop.uv) for loop in uv.data]
        settings.enabled = False
        legacy.vertex_groups[old_segment.group_name].name = '4'
        old_segment.bone_id = 4
        old_segment.group_name = ''
        old_entry.schema_version = 0
        old_entry.native_group_names = '[]'
        old_entry.owned_group_ids = '[]'
        legacy[rig.OWNED_GROUPS] = '[4]'
        del legacy[rig.OWNED_GROUP_NAMES]
        current['path'] = b
        settings.enabled = True
        migrated_name = old_segment.group_name
        assert old_entry.schema_version == 3
        assert old_segment.bone_id >= 65536
        assert migrated_name.startswith('EFMI_PHYS_')
        assert legacy.vertex_groups.get('4') is None
        assert abs(legacy.vertex_groups[migrated_name].weight(0) - 0.45) < 1e-6
        assert legacy.vertex_groups['0'].weight(0) == 1.0
        assert [tuple(loop.uv) for loop in legacy.data.uv_layers['UVMap'].data] == uv_before
        assert all(legacy.vertex_groups.get(name) for name in native_names)
        rig.collect_export_records(scene, [legacy], rig.current_profile())
        # Metadata alone cannot authorize a legacy numeric-group migration if
        # the actual segment bone was deleted. Preserve that ambiguous group.
        ambiguous = mesh('Legacy missing actual bone')
        missing_entry = rig.add_object(bpy.context, ambiguous, rig.current_profile())
        missing_chain = rig.add_chain(bpy.context, missing_entry, rig.current_profile())
        missing_segment = missing_chain.segments[0]
        saved_bone_name = missing_segment.bone_name
        saved_bone_uid = missing_segment.uid
        settings.enabled = False
        ambiguous.vertex_groups[missing_segment.group_name].name = '4'
        ambiguous.vertex_groups['4'].add([0], 0.6, 'REPLACE')
        missing_segment.bone_id = 4
        missing_segment.group_name = ''
        missing_entry.schema_version = 0
        missing_entry.owned_group_ids = '[]'
        missing_entry.native_group_names = '[]'
        ambiguous[rig.OWNED_GROUPS] = '[4]'
        del ambiguous[rig.OWNED_GROUP_NAMES]
        with rig.edit_rig(bpy.context, missing_entry.rig) as bones:
            bones.remove(bones[saved_bone_name])
        settings.enabled = True
        auto_tick()
        assert missing_entry.schema_version == 0
        assert ambiguous.vertex_groups.get('4') and abs(ambiguous.vertex_groups['4'].weight(0) - 0.6) < 1e-6
        expect_error(lambda: rig.collect_export_records(scene, [ambiguous], rig.current_profile()), '真实骨骼身份')
        expect_error(lambda: rig.remove_object(bpy.context, 2), '真实骨骼身份')
        expect_error(lambda: rig.remove_chain(bpy.context, missing_entry, 0), '真实骨骼身份')
        duplicate = ambiguous.copy()
        duplicate.data = ambiguous.data.copy()
        scene.collection.objects.link(duplicate)
        expect_error(lambda: rig.add_object(bpy.context, duplicate, rig.current_profile()), '真实骨骼身份')
        assert missing_entry.schema_version == 0 and duplicate.vertex_groups.get('4')
        assert abs(duplicate.vertex_groups['4'].weight(0) - 0.6) < 1e-6
        bpy.data.objects.remove(duplicate, do_unlink=True)
        assert rig.collect_export_records(scene, [legacy], rig.current_profile())['bones']
        # Restoring the real bone identity is an explicit, verifiable repair.
        with rig.edit_rig(bpy.context, missing_entry.rig) as bones:
            restored = bones.new(saved_bone_name)
            restored.head, restored.tail = (0, 0, 0), (0, 0, 1)
            restored[rig.SEGMENT_UID] = saved_bone_uid
            restored[rig.CHAIN_UID] = missing_chain.uid
        rig.synchronize(bpy.context)
        assert missing_entry.schema_version == 3
        rig.remove_object(bpy.context, 2)
        # A forged/mismatched legacy record cannot claim a native numeric VG.
        settings.enabled = False
        old_entry.schema_version = 0
        legacy[rig.OWNED_GROUPS] = '[0]'
        settings.enabled = True
        assert legacy.vertex_groups['0'].weight(0) == 1.0
        expect_error(lambda: rig.synchronize(bpy.context), '不一致')
        legacy[rig.OWNED_GROUPS] = json.dumps([old_segment.bone_id])
        old_entry.schema_version = 3
        # Save/load doesn't invoke BoolProperty.update: persistent load handler
        # must restore a true service and leave a false service stopped.
        managed_name = obj.name
        for state in (False, True):
            settings.enabled = state
            path = str(Path(directory) / ('state_' + str(state) + '.blend'))
            bpy.ops.wm.save_as_mainfile(filepath=path)
            bpy.ops.wm.open_mainfile(filepath=path)
            scene = bpy.context.scene
            settings = scene.loyaltools_efmi_physics
            assert settings.enabled is state
            assert bpy.app.timers.is_registered(rig._sync_timer) is state
        # Removal only cleans verified prefixed groups. It revokes every object
        # marker and registration, so later ticks never touch that object.
        rig.remove_object(bpy.context, 0)
        obj = bpy.data.objects[managed_name]
        assert rig.OBJECT_UID not in obj and rig.OWNED_GROUPS not in obj
        assert rig.OWNED_GROUP_NAMES not in obj
        assert all(obj.vertex_groups.get(name) for name in native_names)
        obj.vertex_groups.new(name='EFMI_PHYS_user_owned').add([0], 0.8, 'REPLACE')
        calls.clear()
        rig._sync_entry = traced
        auto_tick()
        rig._sync_entry = original_sync
        assert managed_name not in calls
        assert abs(obj.vertex_groups['EFMI_PHYS_user_owned'].weight(0) - 0.8) < 1e-6
        physics.unregister()
        globals_module.unregister()
        assert not bpy.app.timers.is_registered(rig._sync_timer)
        assert rig._TIMER_KEY not in bpy.app.driver_namespace
    print('EFMI_PHYSICS_WORKSPACE_TESTS_OK')


if __name__ == '__main__':
    main()
