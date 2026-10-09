# -*- coding: utf-8 -*-
'''
绝区零描边数据 (TEXCOORD1) 重算。

绝区零把描边方向存在 TEXCOORD1 里: 顶点的"平滑描边法线"投影到该顶点切线空间
(TANGENT / BITANGENT) 后的两个分量。改过网格 (移动顶点、合并别的模型) 之后
原来的 TEXCOORD1 不再对应新几何，描边会出现断裂或毛刺，需要按新几何重算。

算法移植自 XXMI-Tools (leotorrez 等，v1.8.2) migoto/exporter.py 的
ModExporter.optimize_outlines 中 Zenless Zone Zero 分支，保持同样的数值约定
(XXMI-Tools 仓库未声明开源许可，来源说明见 README):
- 描边法线 = 同一位置上所有三角形面法线按顶角加权求和后归一化 (用几何面法线，
  不用顶点法线，这样自定义法线不会影响描边)
- 位置按小数位数取整后分组，合并 UV 接缝处位置相同的重复顶点
- 描边法线与顶点法线的夹角小于阈值时写 0 (游戏在这里是硬截断，没有过渡带)

本模块只依赖 numpy，输入输出都是游戏空间的缓冲数据。
'''

import numpy


DEFAULT_ROUNDING_PRECISION = 8
DEFAULT_GATE_DIVERGENCE = 10.0


def _unit_vector(vector: numpy.ndarray) -> numpy.ndarray:
    norm = numpy.linalg.norm(vector, axis=1, keepdims=True)
    norm = numpy.where(norm == 0, 1, norm)
    return vector / norm


def compute_outline_texcoord(
    positions: numpy.ndarray,
    normals: numpy.ndarray,
    tangents: numpy.ndarray,
    indices: numpy.ndarray,
    rounding_precision: int = DEFAULT_ROUNDING_PRECISION,
    gate_divergence: float = DEFAULT_GATE_DIVERGENCE,
) -> numpy.ndarray:
    '''
    positions (N,3) / normals (N,3) / tangents (N,4, w 为副切线符号) / indices (三角形列表)
    返回 (N,2) float32 的 TEXCOORD1。没有被任何三角形引用的顶点得到 0。
    '''
    positions = numpy.asarray(positions, dtype=numpy.float32)
    normals = numpy.asarray(normals, dtype=numpy.float32)
    tangents = numpy.asarray(tangents, dtype=numpy.float32)
    indices = numpy.asarray(indices, dtype=numpy.int64).reshape(-1)

    result = numpy.zeros((len(positions), 2), dtype=numpy.float32)
    if len(indices) == 0 or len(positions) == 0:
        return result

    loops_coord = positions[indices]
    triangles = loops_coord.reshape(-1, 3, 3)
    edge0 = triangles[:, 1] - triangles[:, 2]
    edge1 = triangles[:, 2] - triangles[:, 0]
    edge2 = triangles[:, 0] - triangles[:, 1]

    cross01 = numpy.cross(edge0, edge1)
    cross12 = numpy.cross(edge1, edge2)
    cross20 = numpy.cross(edge2, edge0)

    # 每个角的内角: atan2(|a x b|, -a.b)
    loops_angle = numpy.zeros((len(triangles), 3), dtype=numpy.float32)
    loops_angle[:, 0] = numpy.arctan2(numpy.linalg.norm(cross12, axis=1), -numpy.einsum("ij,ij->i", edge1, edge2))
    loops_angle[:, 1] = numpy.arctan2(numpy.linalg.norm(cross20, axis=1), -numpy.einsum("ij,ij->i", edge2, edge0))
    loops_angle[:, 2] = numpy.arctan2(numpy.linalg.norm(cross01, axis=1), -numpy.einsum("ij,ij->i", edge0, edge1))

    loops_face_normal = _unit_vector(cross01).repeat(3, axis=0)
    loops_weighted_normal = loops_face_normal * loops_angle.reshape(-1)[:, None]

    _, group_of_loop = numpy.unique(
        numpy.round(loops_coord, rounding_precision), axis=0, return_inverse=True,
    )
    group_of_loop = group_of_loop.reshape(-1)
    accumulated_normals = numpy.zeros((int(group_of_loop.max()) + 1, 3), dtype=numpy.float32)
    numpy.add.at(accumulated_normals, group_of_loop, loops_weighted_normal)

    outline_vector = numpy.zeros((len(positions), 3), dtype=numpy.float32)
    outline_vector[indices] = _unit_vector(accumulated_normals[group_of_loop])

    tangent = _unit_vector(tangents[:, 0:3])
    bitangent = numpy.cross(normals[:, 0:3], tangent) * tangents[:, 3:4]
    result[:, 0] = numpy.einsum("ij,ij->i", tangent, outline_vector)
    result[:, 1] = numpy.einsum("ij,ij->i", bitangent, outline_vector)

    aligned = numpy.einsum("ij,ij->i", outline_vector, normals[:, 0:3])
    divergence = numpy.degrees(numpy.arccos(numpy.clip(aligned, -1, 1)))
    result *= (divergence >= gate_divergence).astype(numpy.float32)[:, None]
    return result
