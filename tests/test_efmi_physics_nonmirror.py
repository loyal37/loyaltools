"""Blender integration: real EFMI import, mirror workflow, preprocess and export.

Run: blender --background --factory-startup --python this_file.py
All meshes, raw captures and exports are disposable fixtures; no game access.
"""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import bpy
except ImportError:
    raise unittest.SkipTest("requires Blender 4.5")
from mathutils import Matrix, Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "loyaltools_nonmirror_test"


def main():
    spec = importlib.util.spec_from_file_location(PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    addon = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = addon
    spec.loader.exec_module(addon)
    addon.register()
    from loyaltools_nonmirror_test.common.global_config import GlobalConfig
    from loyaltools_nonmirror_test.common.efmi_merged_skeleton import write_profile, load_profile, make_submesh_metadata
    from loyaltools_nonmirror_test.common.ssmt_import_helper import SSMTImportHelper
    from loyaltools_nonmirror_test.common.non_mirror_workflow import NonMirrorWorkflowHelper
    from loyaltools_nonmirror_test.common.draw_call_model import DrawCallModel
    from loyaltools_nonmirror_test.blueprint.preprocess import PreProcessHelper
    from loyaltools_nonmirror_test.blueprint.model import BluePrintModel
    from loyaltools_nonmirror_test.ui.universal.efmi import ExportEFMI
    from loyaltools_nonmirror_test.efmi_physics import rig
    exporter_module = importlib.import_module(PACKAGE + ".ui.universal.efmi")

    raw_positions = np.array([[-3., .1, .2], [-2.2, .2, .3], [-3.1, 1.3, .4],
                              [1.4, -.2, .7], [2.7, .3, .8], [1.8, 1.9, 1.1]], dtype=np.float32)

    def weights(obj):
        names = {group.index: group.name for group in obj.vertex_groups}
        return (tuple(sorted(names.values())), tuple(tuple(sorted((names[group.group], float(group.weight))
                     for group in vertex.groups)) for vertex in obj.data.vertices))

    def fixture(folder, mirror, transformed, index):
        scene = bpy.data.scenes.new("Nonmirror fixture " + str(index))
        bpy.context.window.scene = scene
        settings = scene.loyaltools_efmi_physics
        assert not settings.enabled
        scene.global_properties.enable_non_mirror_workflow = mirror
        workspace = Path(folder)
        output = workspace / "export"
        (output / "Meshes").mkdir(parents=True)
        GlobalConfig.path_workspace_folder = classmethod(lambda cls: str(workspace))
        GlobalConfig.path_generate_mod_folder = classmethod(lambda cls: str(output))
        GlobalConfig.path_generatemod_buffer_folder = classmethod(lambda cls: str(output / "Meshes"))
        GlobalConfig.get_workspace_name = classmethod(lambda cls: "NonmirrorFixture")
        GlobalConfig.logic_name = rig.LogicName.EFMI
        unique = ("%08x" % (0x12345670 + index)) + "-6-0"
        raw = workspace / unique / "TYPE_GPU-EFMI"
        raw.mkdir(parents=True)
        component = {"component_id": 1, "unique_str": unique, "ib_hash": unique[:8], "index_count": 6,
                     "first_index": 0, "first_vertex": 0, "vertex_count": 6, "cpu_posed": False,
                     "vg_offset": 2, "vg_count": 4,
                     # Left/right canonical bones are shared with component 0;
                     # this mesh exports them through its private IDs 2 and 3.
                     "vg_map": {"0": 0, "1": 1, "2": 4, "3": 5}, "lods": []}
        provider = {"component_id": 0, "unique_str": "abcdef12-3-0", "ib_hash": "abcdef12", "index_count": 3,
                    "first_index": 0, "first_vertex": 0, "vertex_count": 3, "cpu_posed": False,
                    "vg_offset": 0, "vg_count": 2, "vg_map": {"0": 0, "1": 1}, "lods": []}
        write_profile(str(workspace), {"mode": "EFMI_MERGED_SKELETON", "format_version": 1,
                     "components": [provider, component], "max_instance_count": 2, "object_guid": 12345})
        profile = load_profile(str(workspace))
        (workspace / "Import.json").write_text(json.dumps({unique: "GPU-EFMI"}), encoding="utf-8")

        def element(name, fmt, width, category, slot):
            return {"SemanticName": name, "SemanticIndex": 0, "Format": fmt, "ByteWidth": width,
                    "Category": category, "ExtractSlot": slot, "ExtractTechnique": "trianglelist"}

        categories = [
            {"Type": "Normal", "FileName": "position.buf", "D3D11ElementList": [element("POSITION", "R32G32B32_FLOAT", 12, "Position", "vb0")]},
            {"Type": "Normal", "FileName": "uv.buf", "D3D11ElementList": [element("TEXCOORD", "R32G32_FLOAT", 8, "Texcoord", "vb1")]},
            {"Type": "Normal", "FileName": "blend.buf", "D3D11ElementList": [element("BLENDWEIGHTS", "R16G16B16A16_UNORM", 8, "Blend", "vb2"),
                element("BLENDINDICES", "R8G8B8A8_UINT", 4, "Blend", "vb2")]},
        ]
        data = {"GamePreset": "EFMI", "WorkGameType": "GPU-EFMI", "GPU-PreSkinning": True,
                "CategoryHash": {name: unique[:8] for name in ("Position", "Texcoord", "Blend")},
                "CategoryDrawCategoryMap": {name: name for name in ("Position", "Texcoord", "Blend")},
                "IndexBufferList": [{"DXGI_FORMAT": "R16_UINT", "FileName": "index.buf"}],
                "CategoryBufferList": categories, "TextureMarkUpInfoList": [],
                "EFMIMergedSkeleton": make_submesh_metadata(component)}
        metadata = raw / (unique + ".json")
        metadata.write_text(json.dumps(data), encoding="utf-8")
        raw_positions.tofile(raw / "position.buf")
        np.zeros((6, 2), dtype=np.float32).tofile(raw / "uv.buf")
        blend = np.zeros(6, dtype=[("weights", "<u2", (4,)), ("indices", "u1", (4,))])
        blend["weights"][:, 0] = 65535
        blend["indices"][3:, 0] = 1
        blend.tofile(raw / "blend.buf")
        np.arange(6, dtype=np.uint16).tofile(raw / "index.buf")

        obj = SSMTImportHelper.create_mesh_from_json(str(metadata), merged_skeleton=True)
        before_mirror = weights(obj)
        if mirror:
            NonMirrorWorkflowHelper.process_imported_objects([obj])
        assert weights(obj) == before_mirror, "Mirror import must not swap numeric IDs or weights"
        imported_positions = np.array([vertex.co[:] for vertex in obj.data.vertices])
        expected_import = raw_positions.copy()
        if mirror:
            expected_import[:, 0] *= -1
        np.testing.assert_allclose(imported_positions, expected_import, atol=1e-6)
        if transformed:
            obj.matrix_world = Matrix.Translation((.7, 2., -.4)) @ Matrix.Rotation(.3, 4, "Z") @ Matrix.Diagonal((1.3, .8, 1.1, 1.))
        bpy.context.view_layer.update()
        source_world = obj.matrix_world.copy()
        obj.vertex_groups.new(name="99")  # Old workspace's empty native group.
        settings.enabled = True
        entry = rig.add_object(bpy.context, obj, profile)
        chain_ids = []
        physical_groups = []
        for vertex_index, follow in ((0, 0), (3, 1)):
            chain = rig.add_chain(bpy.context, entry, profile)
            chain.follow_group_id = follow
            bone = rig.find_bone(entry.rig, chain.segments[0])
            with rig.edit_rig(bpy.context, entry.rig) as bones:
                bones[bone.name].head = obj.data.vertices[vertex_index].co
                bones[bone.name].tail = obj.data.vertices[vertex_index].co + Vector((0., 0., .5))
                bones[bone.name].roll = .37
            group_id = chain.segments[0].bone_id
            chain_ids.append(group_id)
            physical_groups.append((chain.segments[0].group_name or str(group_id), vertex_index))
        source_weights = weights(obj)

        scene.global_properties.enable_preprocess_cache = False
        copy_name = PreProcessHelper.execute_preprocess([obj.name])[obj.name]
        prepared = bpy.data.objects[copy_name]
        assert prepared.get('_ssmt_preprocess_hash') == '', "Exercise the real serial uncached export path"
        assert weights(prepared) == source_weights
        prepared_positions = np.array([vertex.co[:] for vertex in prepared.data.vertices])
        expected_export = np.array([source_world @ vertex.co for vertex in obj.data.vertices])
        if mirror:
            expected_export[:, 0] *= -1
        np.testing.assert_allclose(prepared_positions, expected_export, atol=2e-6)

        def blueprint(target=prepared, merged=True):
            return types.SimpleNamespace(ordered_draw_obj_data_model_list=[DrawCallModel(target.name)],
                cross_ib_info_dict={}, cross_ib_method_dict={}, has_cross_ib=False, cross_ib_mapping_objects={},
                cross_ib_vb_condition_mapping={}, cross_ib_source_to_target_dict={}, cross_ib_object_vb_condition={},
                cross_ib_target_info={}, cross_ib_match_mode="DRAW_IB", cross_ib_object_names=[], multi_file_export_nodes=[],
                keyname_mkey_dict={}, _tree={"LoyalTools:EFMIMergedSkeleton": merged})

        # Total OFF: even malformed physics authoring metadata cannot invoke
        # collection, plan validation, shaders, fallback construction or writes.
        settings.enabled = False
        with patch.object(rig, "collect_export_records", side_effect=AssertionError("OFF collected physics")), \
             patch.object(exporter_module, "prepare_physics_plan", side_effect=AssertionError("OFF built a physics plan")), \
             patch.object(exporter_module, "write_runtime_files", side_effect=AssertionError("OFF wrote physics")):
            baseline = ExportEFMI(blueprint())
            baseline.export()
        ini_path = output / "NonmirrorFixture.ini"
        baseline_ini = ini_path.read_bytes()
        baseline_buffers = {path.name: path.read_bytes() for path in (output / "Meshes").glob("*.buf")}
        # The existing exporter normalizes the disposable copy, independently
        # of physics. Its authoring source and numeric group names stay intact.
        prepared_group_names = weights(prepared)[0]
        assert not (output / "res" / "EFMIPhysics").exists()
        assert b"EFMIPhysics" not in baseline_ini
        for group_name, vertex_index in physical_groups:
            for target in (obj, prepared):
                target.vertex_groups[group_name].add([vertex_index], .2 if vertex_index == 0 else .4, "REPLACE")
        source_weights = weights(obj)
        settings.enabled = True
        tree = bpy.data.node_groups.new("Nonmirror VG pipeline " + str(index), 'SSMTBlueprintTreeType')
        tree['LoyalTools:EFMIMergedSkeleton'] = True
        source_node = tree.nodes.new('SSMTNode_Object_Info')
        source_node.object_name = prepared.name
        vg_node = tree.nodes.new('SSMTNode_VertexGroupProcess')
        output_node = tree.nodes.new('SSMTNode_Result_Output')
        tree.links.new(source_node.outputs[0], vg_node.inputs[0])
        tree.links.new(vg_node.outputs[0], output_node.inputs[0])
        owned_before = {name: [prepared.vertex_groups[name].weight(vertex_index)] for name, vertex_index in physical_groups}
        # Both batch and sequential production node paths must preserve owned
        # names and weights, including a conflicting explicit rename mapping.
        ordinary = prepared.vertex_groups.new(name='EFMI_PHYS_user_owned')
        ordinary.add([2], .1, 'REPLACE')
        vg_node._rename_vertex_groups(prepared, {physical_groups[0][0]: '0', '0': physical_groups[0][0]})
        BluePrintModel.clear_object_name_mapping()
        model = BluePrintModel(tree=tree)
        vg_node.process_object(prepared)
        assert prepared.vertex_groups.get('EFMI_PHYS_user_owned') is None
        assert prepared.vertex_groups.get('99') is not None
        for name, vertex_index in physical_groups:
            assert [prepared.vertex_groups[name].weight(vertex_index)] == owned_before[name]
        exporter = ExportEFMI(model)
        exporter.export()
        plan = exporter.efmi_physics_plan
        assert [chain["anchor"] for chain in plan["chains"]] == [0, 1]
        for bi, vertex_index in enumerate((0, 3)):
            np.testing.assert_allclose(plan["bones"][bi]["head"], prepared_positions[vertex_index], atol=2e-6)
            rest = np.array(plan["bones"][bi]["rest_matrix"])
            assert np.linalg.det(rest[:3, :3]) > .9999
            np.testing.assert_allclose(rest @ plan["bones"][bi]["inverse_bind"], np.eye(4), atol=2e-6)
        submesh = exporter.submesh_model_list[0]
        positions = submesh.category_buffer_dict["Position"].view(np.float32).reshape(-1, 3)
        rows = submesh.category_buffer_dict["Blend"].reshape(-1, 16)
        ids = np.ascontiguousarray(rows[:, 8:]).view(np.uint16).reshape(-1, 4)
        exported_weights = np.ascontiguousarray(rows[:, :8]).view(np.uint16).reshape(-1, 4) / 65535.
        for bi, vertex_index in enumerate((0, 3)):
            matches = np.flatnonzero(np.linalg.norm(positions - prepared_positions[vertex_index], axis=1) < 1e-5)
            assert len(matches) == 1
            row = matches[0]
            active = ids[row][exported_weights[row] > 0]
            # Component-private native IDs are 2/3. Custom IDs
            # remain in B+K, and neither mirror path swaps the left/right pair.
            assert set(active) == {2 + bi, plan["global_to_palette"][str(chain_ids[bi])]}, active
        assert weights(obj) == source_weights
        assert b"EFMIPhysics" in ini_path.read_bytes()
        # Stale scene records do not activate physics for an unrelated ordinary
        # EFMI object, including a user-chosen coincidentally prefixed VG name.
        clean_mesh = bpy.data.meshes.new(unique + ' clean')
        clean_mesh.from_pydata([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)], [], [(0, 1, 2)])
        clean_mesh.uv_layers.new(name='TEXCOORD.xy')
        clean = bpy.data.objects.new(unique + '.Ordinary', clean_mesh)
        scene.collection.objects.link(clean)
        clean.vertex_groups.new(name='0').add([0, 1, 2], 1., 'REPLACE')
        clean.vertex_groups.new(name='EFMI_PHYS_user_owned').add([0], .25, 'REPLACE')
        ordinary_export = ExportEFMI(blueprint(clean, False))
        assert not ordinary_export.merged_skeleton_mode and ordinary_export.efmi_physics_plan is None
        # Relevant registered targets must enter strict preflight even when
        # ownership was damaged; they cannot fall back to the ordinary path.
        entry.rig[rig.RIG_OWNER] = 'damaged owner'
        try:
            try:
                ExportEFMI(blueprint())
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged registered ownership silently used the native path')
        finally:
            entry.rig[rig.RIG_OWNER] = entry.uid
        physics_files = {path.relative_to(output): path.read_bytes()
                         for path in (output / "res" / "EFMIPhysics").rglob("*") if path.is_file()}
        settings.enabled = False
        # OFF retains the original VG-node semantics. Its cleanup may remove
        # groups from the disposable copy, but cannot hide active source weights.
        vg_node.process_object(prepared)
        assert all(prepared.vertex_groups.get(name) is None for name, _ in physical_groups)
        all_outputs = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
        # OFF cannot send reserved weights through the original native-B path.
        # Reject at preflight, before buffers or the previous INI are replaced.
        with patch.object(rig, "collect_export_records", side_effect=AssertionError("OFF collected old metadata")):
            try:
                ExportEFMI(blueprint())
            except ValueError as exc:
                assert "仍有物理顶点组权重" in str(exc)
            else:
                raise AssertionError("OFF silently accepted weighted physics groups")
        assert {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()} == all_outputs
        assert weights(obj) == source_weights
        # Restoring native weights is an explicit user action. Empty reserved
        # groups may remain while OFF and do not activate any physics module.
        for group_name, vertex_index in physical_groups:
            for target in (obj, prepared):
                group = target.vertex_groups.get(group_name)
                if group is not None:
                    group.remove([vertex_index])
                else:
                    target.vertex_groups.new(name=group_name)
                target.vertex_groups[str(0 if vertex_index == 0 else 1)].add([vertex_index], 1., "REPLACE")
        restored_weights = weights(obj)
        with patch.object(rig, "collect_export_records", side_effect=AssertionError("OFF collected old metadata")):
            disabled_again = ExportEFMI(blueprint())
            disabled_again.export()
        assert ini_path.read_bytes() == baseline_ini, "OFF must replace the old active runtime INI"
        assert not list(output.rglob("*.ini.bak")), "No inactive backup INI should be introduced"
        assert [path for path in output.rglob("*.ini")] == [ini_path]
        for relative, content in physics_files.items():
            assert (output / relative).read_bytes() == content, "OFF should not write stale physics assets"
        for name, content in baseline_buffers.items():
            assert (output / "Meshes" / name).read_bytes() == content
        assert weights(obj) == restored_weights
        assert set(prepared_group_names).issubset(set(weights(prepared)[0]))
        return {"mirror": mirror, "transformed": transformed, "anchors": [chain["anchor"] for chain in plan["chains"]],
                "heads": [bone["head"] for bone in plan["bones"]], "off_ini_sha256": hashlib.sha256(baseline_ini).hexdigest()}

    results = []
    try:
        for index, (mirror, transformed) in enumerate(((False, False), (True, False), (False, True), (True, True))):
            with tempfile.TemporaryDirectory(prefix="loyaltools_nonmirror_") as directory:
                results.append(fixture(directory, mirror, transformed, index))
    finally:
        addon.unregister()
    print("EFMI_PHYSICS_NONMIRROR_OK", json.dumps(results, ensure_ascii=True))


if __name__ == "__main__":
    main()
