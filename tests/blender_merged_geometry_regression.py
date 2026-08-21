# -*- coding: utf-8 -*-
'''Blender 实际导入骨骼合并子网格的几何回归检查。'''

import math
import sys

import bpy


def _script_args():
    if "--" not in sys.argv:
        raise RuntimeError("请在 -- 后提供子网格 JSON 路径")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 1:
        raise RuntimeError("参数应为: <submesh.json>")
    return args[0]


def main():
    bpy.ops.preferences.addon_enable(module="LoyalTools")
    from LoyalTools.common.ssmt_import_helper import SSMTImportHelper

    obj = SSMTImportHelper.create_mesh_from_json(
        json_file_path=_script_args(),
        merged_skeleton=True,
    )
    assert obj is not None
    assert len(obj.data.vertices) == 318, len(obj.data.vertices)
    assert len(obj.data.polygons) == 242, len(obj.data.polygons)

    longest_edge = 0.0
    for polygon in obj.data.polygons:
        ids = tuple(polygon.vertices)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            edge_length = (obj.data.vertices[ids[start]].co - obj.data.vertices[ids[end]].co).length
            longest_edge = max(longest_edge, edge_length)

    assert math.isclose(longest_edge, 0.0061200424, rel_tol=1e-5, abs_tol=1e-7), longest_edge
    assert len(obj.vertex_groups) > 256, len(obj.vertex_groups)
    print("BLENDER_MERGED_GEOMETRY_REGRESSION=PASS")
    print("VERTICES=318")
    print("POLYGONS=242")
    print("LONGEST_EDGE=" + format(longest_edge, ".10f"))
    print("VERTEX_GROUPS=" + str(len(obj.vertex_groups)))


if __name__ == "__main__":
    main()
