# -*- coding: utf-8 -*-
"""LoD one-to-one assignment must reject weak edges before solving."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_modules():
    root = Path(__file__).resolve().parent.parent
    package_name = "_loyaltools_lod_threshold_regression"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    matcher_module = importlib.import_module(
        package_name + ".efmi_extract.migoto_io.object_extractor.lod_matcher"
    )
    object_module = importlib.import_module(
        package_name
        + ".efmi_extract.migoto_io.object_extractor.migoto_object.migoto_object"
    )
    return matcher_module, object_module


def _component(component_id: int, ib_hash: str):
    return _COMPONENT_TYPE(
        mesh=SimpleNamespace(),
        textures={},
        metadata=SimpleNamespace(
            mesh_name="Component " + str(component_id),
            ib_hash=ib_hash,
            vertex_count=1,
            vg_map={},
        ),
    )


def main():
    matcher_module, object_module = _load_modules()
    global _COMPONENT_TYPE
    _COMPONENT_TYPE = object_module.MigotoComponent

    full_1 = _component(1, "650a6c6b")
    full_11 = _component(11, "844e90f4")
    lod_correct = _component(1, "f09ecf2c")
    lod_extra = _component(9, "119b1b29")

    graph = matcher_module.SimilarityGraph({
        lod_correct: {full_1: 93.2079, full_11: 53.0326},
        lod_extra: {full_1: 47.5344},
    })

    # Without a threshold, maximizing the total intentionally reproduces the
    # Ardelia failure: 53.03 + 47.53 steals the correct 93.21 pair.
    unfiltered = graph.find_optimal_matching(min_similarity=0.0)
    assert next(iter(unfiltered.data[lod_correct])) is full_11
    assert next(iter(unfiltered.data[lod_extra])) is full_1

    matcher = matcher_module.LODMatcher.__new__(matcher_module.LODMatcher)
    matcher.component_similarity_threshold = 55.0
    matcher.object_similarity_threshold = 55.0
    matcher.skip_components_below_similarity_threshold = False
    matched = matcher.get_best_matching_components(graph)

    assert matched == {lod_correct: full_1}, matched
    assert lod_extra not in matched
    print("MERGED_LOD_THRESHOLD_ASSIGNMENT_REGRESSION=PASS")
    print("CORRECT_FULL=1:650a6c6b")
    print("CORRECT_LOD=1:f09ecf2c")
    print("UNMATCHED_LOD=9:119b1b29")


if __name__ == "__main__":
    main()
