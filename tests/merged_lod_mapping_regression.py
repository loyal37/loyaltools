# -*- coding: utf-8 -*-
"""Run LoyalTools LoD mapping against real or fixture FrameAnalysis folders."""

import importlib
import json
import sys
import types
from pathlib import Path


def _load_mapper():
    root = Path(__file__).resolve().parent.parent
    package_name = "_loyaltools_lod_regression"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    return importlib.import_module(package_name + ".extract.merged_skeleton_lod_mapper")


def main():
    if len(sys.argv) != 4:
        raise RuntimeError("Expected: <workspace> <full dump> <LoD dump>")
    workspace, full_dump, lod_dump = sys.argv[1:]
    mapper = _load_mapper()
    result = mapper.map_merged_skeleton_lod(workspace, full_dump, lod_dump)

    profile = json.loads(
        (Path(workspace) / "EFMI_MergedSkeleton.json").read_text(encoding="utf-8")
    )
    assert result.component_count == len(profile["components"])
    assert result.max_lod_count >= 1
    assert all(component.get("lods") for component in profile["components"])
    print("MERGED_LOD_MAPPING_REGRESSION=PASS")
    print("LOD_OBJECT=" + result.lod_object_name)
    print("MATCHED=" + str(result.matched_component_count))
    print("LOWER_POLY=" + str(result.lower_poly_component_count))
    print("MAX_LOD_COUNT=" + str(result.max_lod_count))


if __name__ == "__main__":
    main()
