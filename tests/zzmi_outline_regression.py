# -*- coding: utf-8 -*-
"""Regression for the Zenless Zone Zero outline (TEXCOORD1) calculation (no Blender needed)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "common" / "zzmi_outline.py"
    spec = importlib.util.spec_from_file_location("zzmi_outline_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    outline = _load_module()

    # Two unit squares sharing the edge x = 0, folded by 90 degrees: one lies in the XZ plane
    # facing -Y, the other in the YZ plane facing -X. Every square has its own four vertices
    # (hard edge), so the two vertices on each end of the crease share a position.
    positions = numpy.array([
        (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),      # faces -Y
        (0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1),      # faces -X
    ], dtype=numpy.float32)
    normals = numpy.array([(0, -1, 0)] * 4 + [(-1, 0, 0)] * 4, dtype=numpy.float32)
    # Tangent along +X on the first square and along -Y on the second, bitangent sign -1.
    tangents = numpy.array([(1, 0, 0, -1)] * 4 + [(0, -1, 0, -1)] * 4, dtype=numpy.float32)
    indices = numpy.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7])

    texcoord = outline.compute_outline_texcoord(positions, normals, tangents, indices)
    assert texcoord.shape == (8, 2) and texcoord.dtype == numpy.float32

    # Away from the crease the outline normal equals the vertex normal: nothing is written.
    for vertex in (1, 2, 4, 7):
        assert texcoord[vertex].tolist() == [0.0, 0.0], vertex

    # On the crease both faces contribute a right angle, so the outline normal is the
    # bisector (-1, -1, 0) / sqrt(2), 45 degrees off either vertex normal. Projected on the
    # tangent it gives -sqrt(1/2) on the first square and +sqrt(1/2) on the second; the
    # bitangent cross(N, T) * w is along Z on both, so the second component is zero.
    half = math.sqrt(0.5)
    for vertex in (0, 3):
        assert numpy.allclose(texcoord[vertex], (-half, 0.0), atol=1e-6), texcoord[vertex]
    for vertex in (5, 6):
        assert numpy.allclose(texcoord[vertex], (half, 0.0), atol=1e-6), texcoord[vertex]

    # The cut-off is a hard threshold on the divergence from the vertex normal.
    assert not outline.compute_outline_texcoord(positions, normals, tangents, indices, gate_divergence=46.0).any()
    assert outline.compute_outline_texcoord(positions, normals, tangents, indices, gate_divergence=44.0).any()

    # Positions that only differ beyond the rounding precision are welded together.
    nudged = positions.copy()
    nudged[5] += (0.0, 0.0, 4e-4)
    welded = outline.compute_outline_texcoord(nudged, normals, tangents, indices, rounding_precision=3)
    assert numpy.allclose(welded[0], (-half, 0.0), atol=1e-3)
    split = outline.compute_outline_texcoord(nudged, normals, tangents, indices, rounding_precision=8)
    assert split[0].tolist() == [0.0, 0.0]

    # Vertices no triangle refers to stay zero; empty input is handled.
    padded = outline.compute_outline_texcoord(
        numpy.concatenate([positions, [(9, 9, 9)]]).astype(numpy.float32),
        numpy.concatenate([normals, [(0, 0, 1)]]).astype(numpy.float32),
        numpy.concatenate([tangents, [(1, 0, 0, 1)]]).astype(numpy.float32),
        indices,
    )
    assert padded[8].tolist() == [0.0, 0.0]
    assert outline.compute_outline_texcoord(positions, normals, tangents, []).shape == (8, 2)

    print("ZZMI_OUTLINE_REGRESSION=PASS")


if __name__ == "__main__":
    main()
