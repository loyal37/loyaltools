"""Headless export regressions. Run: python -m unittest discover -s tests."""
import ast
import copy
import importlib.util
import json
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


physics = load_module("efmi_physics_export", ROOT / "common/efmi_physics_export.py")
merged = load_module("efmi_merged_skeleton", ROOT / "common/efmi_merged_skeleton.py")


def exporter_methods(*names):
    # Exercise the production exporter methods without importing Blender's UI.
    source = ast.parse((ROOT / "ui/universal/efmi.py").read_text(encoding="utf-8"))
    owner = next(item for item in source.body if isinstance(item, ast.ClassDef) and item.name == "ExportEFMI")
    methods = [item for item in owner.body if isinstance(item, ast.FunctionDef) and item.name in names]
    tree = ast.Module(body=[ast.ClassDef(name="Export", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    namespace = {"numpy": np, "copy": copy, "build_anchor_fallback_buffers": physics.build_anchor_fallback_buffers, "FormatUtils": NS(
        get_nptype_from_format=lambda fmt: np.uint16 if fmt == "R16G16B16A16_UINT" else np.uint8,
        format_size=lambda fmt: 8 if fmt == "R16G16B16A16_UINT" else 4,
        unorm16_pattern=re.compile(r".*_UNORM$"), unorm8_pattern=re.compile(r".*_UNORM$"),
        snorm16_pattern=re.compile(r".*_SNORM$"), snorm8_pattern=re.compile(r".*_SNORM$"),
    )}
    exec(compile(ast.fix_missing_locations(tree), "efmi.py", "exec"), namespace)
    return namespace["Export"]


def profile():
    return {"bones_count": 6, "max_instance_count": 8, "max_lod_count": 1,
            "components": [
                {"component_id": 0, "cpu_posed": False, "vg_offset": 0, "vg_count": 3,
                 "unique_str": "meshA", "vg_map": {"0": 0, "1": 1, "2": 2},
                 "lods": [{"vg_map": {"0": 1, "1": 1, "2": 2}}]},
                {"component_id": 1, "cpu_posed": False, "vg_offset": 3, "vg_count": 3,
                 "unique_str": "meshB", "vg_map": {"0": 0, "1": 1, "2": 2},
                 "lods": [{"vg_map": {"0": 2, "1": 1, "2": 2}}]},
            ]}


def records():
    result = {"enabled": True, "base_bone_count": 6, "bones": [], "chains": [], "meshes": []}
    for chain_id, object_name, anchor, ids in [("a", "meshA", 0, [6, 8]), ("b", "meshB", 4, [11, 14])]:
        result["chains"].append({"chain_id": chain_id, "object_name": object_name,
                                 "follow_group_id": anchor, "bone_ids": ids,
                                 "params": {"gravity": [1, 2, -9.8]}})
        previous = anchor
        for index, number in enumerate(ids):
            rest = np.eye(4)
            rest[1, 3] = index
            result["bones"].append({"global_id": number, "chain_id": chain_id,
                                    "parent_global_id": previous, "bone_name": str(number),
                                    "rig_name": object_name + "Rig", "rest_matrix": rest.tolist(),
                                    "rig_matrix_world": np.eye(4).tolist(),
                                    "head": [0, index, 0], "tail": [0, index + 1, 0]})
            previous = number
        groups = {str(number): number for number in list(range(6)) + ids}
        result["meshes"].append({"object_name": object_name, "vg_mapping": groups,
                                 "object_matrix_world": np.eye(4).tolist()})
    return result


def blend_type():
    indices = NS(SemanticName="BLENDINDICES", SemanticIndex=0, ElementName="BLENDINDICES",
                 Format="R16G16B16A16_UINT", ByteWidth=8, Category="Blend", ExtractSlot="vb2")
    return NS(D3D11ElementList=[indices], CategoryStrideDict={"Blend": 8},
              ElementNameD3D11ElementDict={"BLENDINDICES": indices})


def prefixed_records():
    result = records()
    mapping = {6: 65536, 8: 65539, 11: 1000001, 14: 2147483647}
    names = {old: "EFMI_PHYS_" + str(index + 1).zfill(6) for index, old in enumerate(mapping)}
    for bone in result["bones"]:
        original = bone["global_id"]
        bone["global_id"] = mapping[original]
        bone["group_name"] = names[original]
        bone["parent_global_id"] = mapping.get(bone["parent_global_id"], bone["parent_global_id"])
    for chain in result["chains"]:
        chain["bone_ids"] = [mapping[original] for original in chain["bone_ids"]]
    for mesh in result["meshes"]:
        mesh["vg_mapping"] = {names.get(number, name): mapping.get(number, number)
                              for name, number in mesh["vg_mapping"].items()}
    return result


class ExportRegressionTests(unittest.TestCase):
    def test_anchor_resolution_updates_runtime_and_fallback_not_authoring(self):
        p = profile()
        p['components'][0]['lods'][0]['is_fallback'] = True
        rec = records()
        original = copy.deepcopy(rec)
        plan = physics.prepare_physics_plan(rec, p)
        physics.resolve_plan_anchors(plan, p, {'meshA': [p['components'][1]], 'meshB': [p['components'][1]]},
                                     {1: {1: [0.0, 0.0, 0.0]}}, merged.resolve_physics_anchor)
        self.assertEqual(plan['chains'][0]['anchor'], 3)
        self.assertEqual(plan['chains'][0]['component_id'], 1)
        self.assertEqual(plan['chains'][0]['follow_group_id'], 0)
        self.assertEqual(plan['bones'][0]['parent_global_id'], 3)
        self.assertEqual(plan['chains'][1]['anchor'], 4)
        self.assertEqual(rec, original)
        raw = np.array([[6, 7, 8, 0]], dtype=np.uint16).view(np.uint8).ravel()
        fallback = physics.build_anchor_fallback_buffers({'Blend': raw}, blend_type(), plan)['Blend']
        np.testing.assert_array_equal(fallback.view(np.uint16), [3, 3, 4, 0])
        np.testing.assert_array_equal(raw.view(np.uint16), [6, 7, 8, 0])

    def test_anchor_missing_bone_uses_same_canonical_bone(self):
        p = profile()
        errors = {0: {1: [0.28, 0.0, 0.0]}, 1: {1: [0.008, 0.0, 0.0]}}
        self.assertEqual(merged.resolve_physics_anchor(p, 0, [p['components'][0]], errors), (3, 1, []))
        self.assertEqual(merged.resolve_physics_anchor(p, 1, [p['components'][0]], errors), (1, 0, []))
        errors[1][1][0] = 0.2
        with self.assertRaisesRegex(ValueError, '可靠'):
            merged.resolve_physics_anchor(p, 0, [p['components'][0]], errors)

    def test_anchor_checks_union_of_consumers_and_rejects_stale_without_equivalent(self):
        p = profile()
        p['components'][0]['lods'][0]['is_fallback'] = True
        errors = {1: {1: [0.0] * 3}}
        self.assertEqual(merged.resolve_physics_anchor(p, 0, [p['components'][0]], errors), (0, 0, []))
        self.assertEqual(merged.resolve_physics_anchor(p, 0, p['components'], errors), (3, 1, []))
        p['components'][1]['vg_map']['0'] = 99
        with self.assertRaisesRegex(ValueError, '等价骨骼池'):
            merged.resolve_physics_anchor(p, 0, p['components'], errors)

    def test_anchor_missing_evidence_is_reported_not_guessed(self):
        p = profile()
        self.assertEqual(merged.resolve_physics_anchor(p, 0, p['components'], {}), (0, 0, [1]))

    def test_animation_anchor_includes_linked_physics_draw_lods(self):
        p = profile()
        near, distant = p['components']
        near['lods'][0]['is_fallback'] = True
        animation = {
            'chains': [{'chain_id': 'animation', 'anchor': 0, 'component_id': 0, 'segments': [-1, -2]}],
            'bones': [{'group_name': 'EFMI_ANIM_000002', 'palette_index': 8, 'chain_index': 0}],
            'meshes': [{'object_name': 'animated', 'vg_mapping': {'EFMI_ANIM_000001': -1}}],
        }
        linked = {
            'chains': [{'chain_id': 'physics', 'anchor': 0, 'component_id': 0,
                        'animation_anchor_group': 'EFMI_ANIM_000002', 'segments': [65536]}],
            'meshes': [{'object_name': 'routed', 'vg_mapping': {'EFMI_PHYS_000001': 65536}}],
        }
        physics.resolve_animation_anchors(
            animation, linked, p, {'animated': [near], 'routed': [distant]},
            {1: {1: [0.0] * 3}}, merged.resolve_physics_anchor,
        )
        root, attached = animation['chains'][0], linked['chains'][0]
        self.assertEqual((root['authoring_anchor'], root['anchor'], root['component_id']), (0, 3, 1))
        self.assertEqual((attached['anchor'], attached['component_id']), (3, 1))
        self.assertEqual((attached['animation_anchor_palette'], attached['animation_anchor_component']), (8, 1))

        plan = {'chains': [attached, root], 'global_to_palette': {'65536': 6, '-1': 7, '-2': 8}}
        raw = np.array([[6, 7, 8, 1]], dtype=np.uint16).view(np.uint8).ravel()
        fallback = physics.build_anchor_fallback_buffers({'Blend': raw}, blend_type(), plan)['Blend']
        np.testing.assert_array_equal(fallback.view(np.uint16), [3, 3, 3, 1])
        np.testing.assert_array_equal(raw.view(np.uint16), [6, 7, 8, 1])

    def test_draw_components_follow_only_the_mesh_routed_by_cross_ib(self):
        Export = exporter_methods('_extra_bone_draw_components', '_get_submesh_ib_key')
        exporter = Export()
        exporter.has_cross_ib = True
        exporter.cross_ib_match_mode = 'INDEX_COUNT'
        exporter.cross_ib_mapping_objects = {('indexcount_3', 'indexcount_6'): {'redirected'}}
        near = {'unique_str': 'near', 'cpu_posed': False, 'index_count': 3}
        distant = {'unique_str': 'distant', 'cpu_posed': False, 'index_count': 6}
        redirected = NS(obj_name='redirected', match_index_count=3,
                        get_unique_str=lambda: 'near', get_blender_obj_name=lambda: 'redirected_copy')
        ordinary = NS(obj_name='ordinary', match_index_count=3,
                      get_unique_str=lambda: 'near', get_blender_obj_name=lambda: 'ordinary_copy')
        exporter.blueprint_model = NS(ordered_draw_obj_data_model_list=[redirected, ordinary])
        draw_components = exporter._extra_bone_draw_components({'components': [near, distant]})
        self.assertEqual(draw_components, {'redirected_copy': [distant], 'ordinary_copy': [near]})

    def test_disabled_and_empty_are_exact_noops(self):
        values = np.array([[7, 3, 0, 0]], dtype=np.uint16)
        buffers = {"Blend": values.view(np.uint8).ravel()}
        before = buffers["Blend"]
        for rec in [None, {"enabled": False}, {"enabled": True, "bones": []}]:
            self.assertIsNone(physics.prepare_physics_plan(rec, None))
        physics.remap_joined_vertex_groups(buffers, blend_type(), [], None)
        self.assertIs(buffers["Blend"], before)
        mask = np.array([[True, False, False, False]])
        self.assertIs(physics.base_bone_mask(values, mask, None), mask)

    def test_multiple_meshes_chains_and_sparse_authoring_ids(self):
        rec = records()
        original = copy.deepcopy(rec)
        plan = physics.prepare_physics_plan(rec, profile())
        self.assertEqual(plan["global_to_palette"], {"6": 6, "8": 7, "11": 8, "14": 9})
        self.assertEqual(plan["bone_count"], 4)
        self.assertEqual(plan["extended_bone_count"], 10)
        self.assertEqual([bone["parent_index"] for bone in plan["bones"]], [-1, 0, -1, 2])
        self.assertEqual([chain["component_id"] for chain in plan["chains"]], [0, 1])
        self.assertEqual(rec, original)
        # An additional mesh may share already allocated global bones.
        rec["meshes"].append(dict(rec["meshes"][0], object_name="sharedMesh"))
        shared_plan = physics.prepare_physics_plan(rec, profile())
        self.assertEqual(shared_plan["global_to_palette"], plan["global_to_palette"])

    def test_join_uses_names_after_deleted_groups_shift_indices(self):
        plan = physics.prepare_physics_plan(records(), profile())
        # Blender joins groups by name. These indices deliberately differ from
        # the stable numeric names, as after deleting the middle chain's groups.
        groups = [NS(index=0, name="0"), NS(index=1, name="8"), NS(index=2, name="14"), NS(index=3, name="4")]
        values = np.array([[1, 2, 3, 0], [2, 1, 0, 0]], dtype=np.uint16)
        buffers = {"Blend": values.view(np.uint8).ravel()}
        physics.remap_joined_vertex_groups(buffers, blend_type(), groups, plan)
        np.testing.assert_array_equal(buffers["Blend"].view(np.uint16).reshape(-1, 4), [[7, 9, 4, 0], [9, 7, 0, 0]])

    def test_prefixed_groups_keep_exact_authoring_ids_and_dense_runtime_palette(self):
        rec = prefixed_records()
        plan = physics.prepare_physics_plan(rec, profile())
        self.assertEqual(plan["global_to_palette"], {"65536": 6, "65539": 7, "1000001": 8, "2147483647": 9})
        self.assertEqual(plan["group_to_palette"]["EFMI_PHYS_000004"], 9)
        serialized = physics.serialize_authoring_records(plan)
        self.assertIn("2147483647", serialized)
        groups = [NS(index=0, name="0"), NS(index=1, name="EFMI_PHYS_000002"),
                  NS(index=2, name="EFMI_PHYS_000004"), NS(index=3, name="4")]
        values = np.array([[1, 2, 3, 0]], dtype=np.uint16)
        buffers = {"Blend": values.view(np.uint8).ravel()}
        physics.remap_joined_vertex_groups(buffers, blend_type(), groups, plan)
        np.testing.assert_array_equal(buffers["Blend"].view(np.uint16).reshape(-1, 4), [[7, 9, 4, 0]])
        # A large authoring ID is never accepted as an arbitrary numeric group;
        # only the explicitly registered prefixed name can address that bone.
        with self.assertRaisesRegex(ValueError, "未知额外"):
            physics.remap_joined_vertex_groups({"Blend": values.view(np.uint8).ravel()}, blend_type(),
                                               [NS(index=0, name="65536")], plan)

    def test_prefixed_weights_require_explicit_mesh_binding(self):
        plan = physics.prepare_physics_plan(prefixed_records(), profile())
        obj = NS(name="meshA", vertex_groups=[NS(index=0, name="EFMI_PHYS_000001")],
                 data=NS(vertices=[NS(groups=[NS(group=0, weight=.5)])]))
        physics.validate_object_weights([obj], plan)
        obj.vertex_groups[0].name = "EFMI_PHYS_999999"
        with self.assertRaisesRegex(ValueError, "未绑定"):
            physics.validate_object_weights([obj], plan)

    def test_disabled_rejects_only_verified_physics_weights(self):
        obj = NS(name="OFF mesh", vertex_groups=[NS(index=0, name="0"), NS(index=1, name="EFMI_PHYS_000001")],
                 data=NS(vertices=[NS(groups=[NS(group=0, weight=1.), NS(group=1, weight=0.)])]),
                 get=lambda key, default: default)
        physics.reject_disabled_physics_weights([obj])
        obj.data.vertices[0].groups[1].weight = .25
        # An ordinary user's prefixed name is not ownership proof.
        physics.reject_disabled_physics_weights([obj])
        markers = {"LoyalTools:EFMIPhysicsObject": "owner", "LoyalTools:EFMIPhysicsOwnedGroups": "[65536]",
                   "LoyalTools:EFMIPhysicsOwnedGroupNames": '["EFMI_PHYS_000001"]'}
        obj.get = lambda key, default=None: markers.get(key, default)
        rig = NS(get=lambda key, default=None: {"LoyalTools:EFMIPhysicsRig": True, "LoyalTools:EFMIPhysicsOwner": "owner"}.get(key, default))
        entry = NS(uid="owner", object=obj, rig=rig, schema_version=3, owned_group_ids="[65536]", native_group_names='["0"]',
                   chains=[NS(segments=[NS(bone_id=65536, group_name="EFMI_PHYS_000001")])])
        settings = NS(objects=[entry])
        names = physics.verified_physics_group_names(obj, settings)
        self.assertEqual(names, {"EFMI_PHYS_000001"})
        with self.assertRaisesRegex(ValueError, "本次导出已取消"):
            physics.reject_disabled_physics_weights([obj], owned_group_names={obj.name: names})
        self.assertEqual(obj.data.vertices[0].groups[1].weight, .25)
        # OFF node cleanup can strip the export copy. The proven source still
        # has physics weights, so that cleanup cannot bypass the safety check.
        target = NS(name="OFF copy", vertex_groups=[NS(index=0, name="0")], data=NS(vertices=[]), get=obj.get)
        self.assertEqual(physics.verified_physics_group_names(target, settings), set())
        with self.assertRaisesRegex(ValueError, "仍有物理"):
            physics.reject_disabled_physics_weights([target], owned_group_names={target.name: names}, owned_sources={target.name: obj})
        markers["LoyalTools:EFMIPhysicsOwnedGroups"] = "[999]"
        self.assertEqual(physics.verified_physics_group_names(obj, settings), set())
        # With no ownership/owned names, ordinary geometry is not scanned.
        physics.reject_disabled_physics_weights([NS(name="ordinary")])

    def test_empty_native_groups_outside_smaller_workspace_are_preserved(self):
        rec = prefixed_records()
        rec["meshes"][0]["vg_mapping"]["99"] = 99
        rec["meshes"][0]["active_group_names"] = ["0", "EFMI_PHYS_000001"]
        plan = physics.prepare_physics_plan(rec, profile())
        self.assertEqual(plan["meshes"][0]["vg_mapping"]["99"], 99)
        self.assertNotIn("99", plan["group_to_palette"])
        groups = [NS(index=0, name="0"), NS(index=1, name="99")]
        values = np.array([[0, 1, 0, 0]], dtype=np.uint16)
        buffers = {"Blend": values.view(np.uint8).ravel()}
        physics.remap_joined_vertex_groups(buffers, blend_type(), groups, plan, active_group_indices={0})
        np.testing.assert_array_equal(buffers["Blend"].view(np.uint16), [0, 0, 0, 0])
        self.assertEqual(groups[1].name, "99")
        with self.assertRaisesRegex(ValueError, "未知额外"):
            physics.remap_joined_vertex_groups(buffers, blend_type(), groups, plan, active_group_indices={1})
        rec["meshes"][0]["active_group_names"].append("99")
        with self.assertRaisesRegex(ValueError, "未绑定有效"):
            physics.prepare_physics_plan(rec, profile())

    def test_custom_weights_do_not_enter_original_pool_or_lod_audit(self):
        plan = physics.prepare_physics_plan(records(), profile())
        values = np.array([[0, 4, 6, 9]], dtype=np.uint16)
        mask = physics.base_bone_mask(values, np.ones_like(values, dtype=bool), plan)
        np.testing.assert_array_equal(mask, [[True, True, False, False]])
        used = values[mask].tolist()
        self.assertEqual(used, [0, 4])
        owner_rewrite = merged.build_component_private_vg_remap(profile()["components"][1])
        self.assertFalse(set(owner_rewrite) & {6, 9})
        self.assertTrue(all(key < 6 for key in owner_rewrite))
        self.assertEqual(profile()["components"][1]["lods"][0]["vg_map"], {"0": 2, "1": 1, "2": 2})

    def test_lod1_layout_preserves_custom_r16_ids(self):
        Export = exporter_methods("_build_lod_slot_buffer", "_source_category_element_offset",
                                  "_decode_dxgi_values", "_encode_dxgi_values", "_resize_semantic_components")
        values = np.array([[6, 7, 8, 9], [0, 9, 8, 7]], dtype=np.uint16)
        submesh = NS(unique_str="meshA", d3d11_game_type=blend_type(),
                     category_buffer_dict={"Blend": values.view(np.uint8).ravel()})
        category, stride, output = Export()._build_lod_slot_buffer(submesh, "vb2", {
            "semantics": [{"name": "BLENDINDICES", "index": 0, "format": "R8G8B8A8_UINT", "stride": 4}]})
        self.assertEqual((category, stride), ("Blend", 8))
        np.testing.assert_array_equal(output.view(np.uint16).reshape(-1, 4), values)

    def test_mesh_transform_mirror_and_bind_inverse(self):
        rec = records()
        world = np.array([[2, 0, 0, 3], [0, 4, 0, 5], [0, 0, 2, 7], [0, 0, 0, 1]], dtype=float)
        for bone in rec["bones"]:
            bone["rig_matrix_world"] = world.tolist()
        for mesh in rec["meshes"]:
            mesh["object_matrix_world"] = world.tolist()
        for mirror in (False, True):
            plan = physics.prepare_physics_plan(rec, profile(), mirror_x=mirror)
            root = plan["bones"][0]
            np.testing.assert_allclose(root["head"], [-3 if mirror else 3, 5, 7])
            self.assertEqual(root["length"], 4)
            rest = np.array(root["rest_matrix"])
            np.testing.assert_allclose(rest @ root["inverse_bind"], np.eye(4), atol=1e-7)
            np.testing.assert_allclose(rest[:3, :3].T @ rest[:3, :3], np.eye(3), atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(rest[:3, :3]), 1)
            self.assertEqual(plan["chains"][0]["params"]["gravity"], [1, 2, -9.8])

    def test_unknown_unbound_and_native_collision_rejected(self):
        for change in ("native", "unbound", "unknown"):
            rec = records()
            if change == "native":
                rec["bones"][0]["global_id"] = 4
            elif change == "unbound":
                rec["chains"][0]["bone_ids"] = [6]
            else:
                rec["meshes"][0]["vg_mapping"]["42"] = 42
            with self.subTest(change=change), self.assertRaises(ValueError):
                physics.prepare_physics_plan(rec, profile())
        plan = physics.prepare_physics_plan(records(), profile())
        unknown = NS(name="unregisteredCopy", vertex_groups=[NS(index=0, name="14")],
                     data=NS(vertices=[NS(groups=[NS(group=0, weight=1)])]))
        with self.assertRaisesRegex(ValueError, "未绑定"):
            physics.validate_object_weights([unknown], plan)

    def test_atomic_ini_failure_keeps_previous_active_ini(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mod.ini"
            path.write_text("previous")
            with self.assertRaises(RuntimeError):
                with physics.atomic_ini_target(path) as temporary:
                    Path(temporary).write_text("incomplete")
                    raise RuntimeError("shader generation failed")
            self.assertEqual(path.read_text(), "previous")
            self.assertEqual([item.name for item in Path(folder).iterdir()], ["mod.ini"])
            with physics.atomic_ini_target(path) as temporary:
                Path(temporary).write_text("complete")
            self.assertEqual(path.read_text(), "complete")

    def test_runtime_singleton_sections_merge_into_host(self):
        constants, present, remaining = physics.split_runtime_sections(
            "; generated\n[Constants]\nglobal $physics = 1\n[Present]\nrun = CommandListTick\n[CommandListTick]\n$physics = 2")
        self.assertEqual(constants, ["global $physics = 1"])
        self.assertEqual(present, ["run = CommandListTick"])
        self.assertEqual(remaining, ["; generated", "[CommandListTick]", "$physics = 2"])

    def test_explicit_cb_or_unknown_source_rejected_before_writes(self):
        for source in (0, 2, "CB", "unknown", None):
            current = profile()
            current["bones_data_source"] = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "CB 或未知"):
                physics.runtime_profile(current)
        current = profile()
        current["components"][0]["bones_data_source"] = "CB"
        with self.assertRaises(ValueError):
            physics.runtime_profile(current)
        self.assertEqual(physics.runtime_profile(profile())["adapter"]["original_bones"], 6)

    def test_fallback_replaces_only_custom_indices_preserving_weights(self):
        plan = physics.prepare_physics_plan(records(), profile())
        game_type = blend_type()
        weights = NS(SemanticName="BLENDWEIGHTS", SemanticIndex=0, ByteWidth=4, Category="Blend")
        game_type.D3D11ElementList.insert(0, weights)
        game_type.CategoryStrideDict["Blend"] = 12
        values = np.array([[0, 6, 7, 9], [4, 8, 6, 0]], dtype=np.uint16)
        rows = np.zeros((2, 12), dtype=np.uint8)
        rows[:, :4] = [[10, 20, 30, 195], [50, 100, 105, 0]]
        rows[:, 4:] = values.view(np.uint8).reshape(2, 8)
        original = rows.copy()
        fallback = physics.build_anchor_fallback_buffers({"Blend": rows.ravel()}, game_type, plan)["Blend"].reshape(2, 12)
        np.testing.assert_array_equal(rows, original)
        np.testing.assert_array_equal(fallback[:, :4], original[:, :4])
        np.testing.assert_array_equal(np.ascontiguousarray(fallback[:, 4:]).view(np.uint16).reshape(2, 4),
                                      [[0, 0, 0, 4], [4, 4, 0, 0]])

    def test_fallback_resource_selected_by_runtime_bound_for_lod1(self):
        Export = exporter_methods("_append_merged_level_buffer_bindings", "_merged_resource_prefix")
        exporter = Export()
        exporter.merged_lod_variant_buffers = {("meshA", 1, "VB2"): {"resource_name": "Resource_meshA_Blend_LOD1"}}
        exporter.efmi_physics_fallback_buffers = {"Resource_meshA_Blend_LOD1": {"resource_name": "Resource_meshA_Blend_LOD1_PhysicsFallback"}}
        exporter.efmi_physics_runtime = NS(metadata={"bound_variable": "$EFMIPhysics_bound"})
        game_type = blend_type()
        game_type.CategoryExtractSlotDict = {"Blend": "vb2"}
        submesh = NS(unique_str="meshA", category_buffer_dict={"Blend": np.zeros(8, dtype=np.uint8)}, d3d11_game_type=game_type)
        lines = []
        exporter._append_merged_level_buffer_bindings(lines, submesh, 1)
        self.assertEqual(lines, ["if $EFMIPhysics_bound", "    vb2 = Resource_meshA_Blend_LOD1", "else",
                                 "    vb2 = Resource_meshA_Blend_LOD1_PhysicsFallback", "endif"])

    def test_fallback_waits_only_for_chains_actually_weighted_by_submesh(self):
        Export = exporter_methods("_build_physics_fallback_buffers", "_merged_resource_prefix", "_source_category_element_offset",
                                  "_build_active_blend_weight_mask", "_append_merged_level_buffer_bindings")
        exporter = Export()
        exporter.efmi_physics_plan = physics.prepare_physics_plan(records(), profile())
        exporter.efmi_physics_runtime = NS(metadata={"bound_variable": "$bound", "chain_ready_variables": ["$ready0", "$ready1"]})
        exporter.efmi_physics_fallback_buffers = {}
        exporter.merged_lod_variant_buffers = {}
        exporter._get_merged_profile_component = lambda unique: {"lods": []}
        game_type = blend_type()
        weight = NS(SemanticName="BLENDWEIGHTS", SemanticIndex=0, ElementName="BLENDWEIGHTS", Format="R8G8B8A8_UNORM", ByteWidth=4, Category="Blend")
        game_type.D3D11ElementList.insert(0, weight)
        game_type.CategoryStrideDict["Blend"] = 12
        game_type.CategoryExtractSlotDict = {"Blend": "vb2"}
        rows = np.zeros((1, 12), dtype=np.uint8)
        rows[0, :4] = [255, 0, 0, 0]
        rows[0, 4:] = np.array([6, 9, 0, 0], dtype=np.uint16).view(np.uint8)
        # Bone 6 belongs to chain 0; chain 1's bone 9 has zero weight.
        submesh = NS(unique_str="meshA", category_buffer_dict={"Blend": rows.ravel()}, d3d11_game_type=game_type)
        exporter.submesh_model_list = [submesh]
        exporter._build_physics_fallback_buffers()
        fallback = exporter.efmi_physics_fallback_buffers["Resource_meshA_Blend"]
        self.assertEqual(fallback["ready_condition"], "$bound && $ready0")
        lines = []
        exporter._append_merged_level_buffer_bindings(lines, submesh, 0)
        self.assertEqual(lines[0], "if $bound && $ready0")
        # These are the exact conditions emitted for the production binding.
        condition = lines[0][3:]
        for bound, ready0, unrelated, expected in [(True, False, True, False), (True, True, False, True), (False, True, True, False)]:
            expression = condition.replace("$bound", str(bound)).replace("$ready0", str(ready0)).replace("&&", "and")
            self.assertEqual(eval(expression, {"__builtins__": {}}, {}), expected)


if __name__ == "__main__":
    unittest.main()
