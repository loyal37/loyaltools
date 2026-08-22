# -*- coding: utf-8 -*-
"""Pure-Python contract regression for the LoD mapping table."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _load_profile_module():
    root = Path(__file__).resolve().parent.parent
    package_name = "_loyaltools_lod_preview_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    return importlib.import_module(package_name + ".common.efmi_merged_skeleton")


def _component(component_id, unique_str, lod_unique_str, fallback=False):
    return {
        "component_id": component_id,
        "unique_str": unique_str,
        "ib_hash": unique_str.split("-")[0],
        "index_count": int(unique_str.split("-")[1]),
        "first_index": int(unique_str.split("-")[2]),
        "vertex_count": 3,
        "cpu_posed": False,
        "vg_offset": component_id,
        "vg_count": 1,
        "vg_map": {"0": component_id},
        "lods": [{
            "lod_object_name": "Character 35051",
            "ib_hash": lod_unique_str.split("-")[0],
            "vb0_hash": "deadbeef",
            "vertex_offset": 0,
            "vertex_count": 3,
            "index_offset": 0,
            "index_count": int(lod_unique_str.split("-")[1]),
            "first_index": int(lod_unique_str.split("-")[2]),
            "unique_str": lod_unique_str,
            "is_fallback": fallback,
            "vg_map": {},
            "vb_formats": {},
        }],
    }


def main():
    module = _load_profile_module()
    profile = {
        "format_version": 1,
        "mode": "EFMI_MERGED_SKELETON",
        "required_efmi_version": "1.4.1",
        "object_guid": 9,
        "max_instance_count": 8,
        "lod_sources": {"Character 35051": r"D:\FrameAnalysis-LOD"},
        "components": [
            _component(0, "650a6c6b-6-0", "f09ecf2c-3-12"),
            _component(1, "844e90f4-3-24", "844e90f4-3-24", fallback=True),
        ],
    }
    groups = module.build_lod_mapping_groups(profile)
    assert len(groups) == 1
    assert groups[0]["lod_object_name"] == "Character 35051"
    assert groups[0]["rows"] == [
        {
            "component_id": 0,
            "lod_unique_str": "f09ecf2c-3-12",
            "main_unique_str": "650a6c6b-6-0",
            "is_fallback": False,
        },
        {
            "component_id": 1,
            "lod_unique_str": "844e90f4-3-24",
            "main_unique_str": "844e90f4-3-24",
            "is_fallback": True,
        },
    ]
    print("MERGED_LOD_PREVIEW_CONTRACT_REGRESSION=PASS")
    print("ROW=f09ecf2c-3-12 -> 650a6c6b-6-0")
    print("FALLBACK=844e90f4-3-24 -> 844e90f4-3-24")


if __name__ == "__main__":
    main()
