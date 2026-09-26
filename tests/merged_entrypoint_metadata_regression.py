# -*- coding: utf-8 -*-
"""Pure-Python regressions for original-draw entry-point metadata."""

from __future__ import annotations

import copy
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "_loyaltools_entrypoint_regression"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
profile_module = importlib.import_module(PACKAGE + ".common.efmi_merged_skeleton")
extractor_module = importlib.import_module(PACKAGE + ".extract.dump_workspace_extractor")
mapper_module = importlib.import_module(PACKAGE + ".extract.merged_skeleton_lod_mapper")
Extractor = extractor_module.DumpWorkspaceExtractor


class Component:
    """Identity-hashable fixture like the vendored MigotoComponent."""


def component(ib_hash="12345678", first_index=6, first_vertices=(17,), index_counts=None):
    result = Component()
    calls = []
    ib = extractor_module.IndexBuffer(hash=ib_hash, pointer="0x1234")
    for index, first_vertex in enumerate(first_vertices):
        count = index_counts[index] if index_counts is not None else 540
        draw = extractor_module.DrawIndexedInstanced(
            bindings=[], raw_command=None, index_count=count, first_index=first_index,
            first_vertex=first_vertex, instance_count=1, first_instance=0,
        )
        resources = SimpleNamespace(get_by_slot=lambda slot, ib=ib: ib if slot == "ib" else None)
        calls.append(SimpleNamespace(draw_call=draw, model_resources=resources))
    result.raw_data = SimpleNamespace(shader_calls=calls)
    # Deliberately different from the original draw: neither is BaseVertexLocation.
    result.mesh = SimpleNamespace(
        format=SimpleNamespace(vertex_count=3, first_vertex=888), cpu_posed=False,
    )
    result.metadata = SimpleNamespace(
        ib_hash=ib_hash, vb0_hash="deadbeef", vertex_offset=777, vertex_count=3,
        index_offset=first_index, index_count=540,
    )
    return result


def profile_component(component_id=0, ib_hash="12345678", first_index=6):
    return {
        "component_id": component_id, "source_component_id": component_id,
        "unique_str": f"{ib_hash}-540-{first_index}", "ib_hash": ib_hash,
        "index_count": 540, "first_index": first_index, "vertex_count": 3,
        "cpu_posed": False, "vg_offset": 0, "vg_count": 0, "vg_map": {},
        "lods": [],
    }


def profile():
    return {
        "format_version": 1, "mode": profile_module.PROFILE_MODE,
        "object_name": "Character Main", "components": [profile_component()],
    }


def lod_metadata():
    return {
        "lod_object_name": "Character LOD", "ib_hash": "87654321",
        "index_count": 540, "first_index": 24, "vertex_count": 3,
    }


class EntryPointMetadataTests(unittest.TestCase):
    def test_signed_int32_validation_and_preservation(self):
        for value in (0, 17, -12, -(1 << 31), (1 << 31) - 1):
            with self.subTest(value=value):
                data = profile()
                data["components"][0]["first_vertex"] = value
                data["components"][0]["lods"] = [dict(lod_metadata(), first_vertex=value)]
                normalized = profile_module.validate_profile(data)
                self.assertEqual(normalized["components"][0]["first_vertex"], value)
                self.assertEqual(normalized["components"][0]["lods"][0]["first_vertex"], value)
        for invalid in (True, False, 1.0, 1.25, "0", None, 1 << 31, -(1 << 31) - 1):
            for location in ("main", "lod"):
                with self.subTest(invalid=invalid, location=location):
                    data = profile()
                    if location == "main":
                        data["components"][0]["first_vertex"] = invalid
                    else:
                        data["components"][0]["lods"] = [dict(lod_metadata(), first_vertex=invalid)]
                    with self.assertRaises(profile_module.MergedSkeletonProfileError):
                        profile_module.validate_profile(data)

    def test_legacy_absence_survives_normalize_and_disk_roundtrip(self):
        data = profile()
        data["components"][0]["lods"] = [lod_metadata()]
        with tempfile.TemporaryDirectory(prefix="loyal-entrypoint-legacy-") as folder:
            profile_module.write_profile(folder, data)
            loaded = profile_module.load_profile(folder)
        self.assertNotIn("first_vertex", loaded["components"][0])
        self.assertNotIn("first_vertex", loaded["components"][0]["lods"][0])
        self.assertEqual(loaded["format_version"], 1)

    def test_draw_source_not_geometry_slice_and_all_passes_checked(self):
        same_base = component(first_vertices=(-12, -12), index_counts=(540, 270))
        self.assertEqual(Extractor.get_component_first_vertex(same_base), -12)
        key, matching, total = Extractor.get_component_primary_draw(same_base)
        self.assertEqual(key, ("12345678", 540, 6))
        self.assertEqual((len(matching), total), (1, 2))
        for counts in ((540, 540), (540, 270)):
            with self.subTest(index_counts=counts):
                inconsistent = component(first_vertices=(0, 2842), index_counts=counts)
                with self.assertRaisesRegex(extractor_module.ExtractError, "不一致"):
                    Extractor.get_component_first_vertex(inconsistent)
        for invalid in (True, 0.0, "17", None, 1 << 31):
            with self.subTest(invalid=invalid):
                with self.assertRaises(extractor_module.ExtractError):
                    Extractor.get_component_first_vertex(component(first_vertices=(invalid,)))

    def test_main_extraction_persists_original_draw_base(self):
        source = component(first_vertices=(17, 17))
        obj = SimpleNamespace(id="Character Main", components=[source])
        extractor = object.__new__(Extractor)
        extractor.dump_folder = "main-capture"
        extractor.get_merged_skeleton_candidates = Mock(return_value=[obj])
        extractor._build_merged_skeleton_vg_metadata = Mock(return_value=[{
            "vg_offset": 0, "vg_count": 0, "vg_map": {},
        }])
        extractor._build_submesh = Mock(return_value=SimpleNamespace(vertex_count=3))
        extractor._write_submesh = Mock(return_value="unused-submesh.json")
        extractor._update_workspace_root_files = Mock()
        with tempfile.TemporaryDirectory(prefix="loyal-entrypoint-main-") as folder:
            extractor.extract_merged_skeleton(folder, copy_textures=False)
            loaded = profile_module.load_profile(folder)
        self.assertEqual(loaded["components"][0]["first_vertex"], 17)
        self.assertEqual(extractor._write_submesh.call_args.kwargs["merged_component"]["first_vertex"], 17)

    def _map_fixture(self, folder, lod_bases=(42, 42), full_bases=(17, 17)):
        full = component(first_vertices=full_bases)
        fallback = component(ib_hash="abcdef01", first_index=12, first_vertices=(-12, -12))
        lod = component(ib_hash="87654321", first_index=24, first_vertices=lod_bases)
        full_obj = SimpleNamespace(id="Character Main", components=[full, fallback])
        lod_obj = SimpleNamespace(id="Character LOD", components=[lod])

        class FixtureExtractor(Extractor):
            def __init__(self, capture):
                self.obj = full_obj if capture == "main-capture" else lod_obj

            def get_merged_skeleton_candidates(self, **_kwargs):
                return [self.obj]

            def extract_merged_skeleton_preview(self, **_kwargs):
                return SimpleNamespace(unique_strs=["87654321-540-24"], warnings=[])

        matcher = Mock()
        matcher.find_matching_lods.return_value = (lod_obj, {full: (lod, {})})
        with patch.object(mapper_module, "DumpWorkspaceExtractor", FixtureExtractor), \
                patch.object(mapper_module, "LODMatcher", return_value=matcher), \
                patch.object(mapper_module, "_serialize_lod_vb_formats", return_value={}):
            mapper_module.map_merged_skeleton_lod(folder, "main-capture", "lod-capture")

    def test_lod_uses_own_capture_main_refresh_and_fallback_uses_main(self):
        data = profile()
        data["components"].append(profile_component(1, "abcdef01", 12))
        with tempfile.TemporaryDirectory(prefix="loyal-entrypoint-map-") as folder:
            profile_module.write_profile(folder, data)
            self._map_fixture(folder)
            loaded = profile_module.load_profile(folder)
        main, fallback = loaded["components"]
        self.assertEqual(main["first_vertex"], 17)
        self.assertEqual(main["lods"][0]["first_vertex"], 42)
        self.assertEqual(main["lods"][0]["first_index"], 24)
        self.assertEqual(main["lods"][0]["vertex_offset"], 777)
        self.assertEqual(fallback["first_vertex"], -12)
        self.assertEqual(fallback["lods"][0]["first_vertex"], -12)
        self.assertTrue(fallback["lods"][0]["is_fallback"])

    def test_inconsistent_lod_or_full_passes_do_not_save_profile(self):
        data = profile()
        data["components"].append(profile_component(1, "abcdef01", 12))
        for overrides in ({"lod_bases": (0, 1062)}, {"full_bases": (0, 2842)}):
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory(prefix="loyal-entrypoint-conflict-") as folder:
                    profile_module.write_profile(folder, copy.deepcopy(data))
                    before = Path(profile_module.get_profile_path(folder)).read_bytes()
                    with self.assertRaisesRegex(extractor_module.ExtractError, "不一致"):
                        self._map_fixture(folder, **overrides)
                    self.assertEqual(Path(profile_module.get_profile_path(folder)).read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
