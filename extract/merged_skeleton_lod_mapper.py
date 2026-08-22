# -*- coding: utf-8 -*-
"""EFMI Merged Skeleton LoD mapping for LoyalTools workspaces.

The mapper follows EFMI-Tools v0.6.2: rebuild the full-detail and LoD objects
from their FrameAnalysis captures, match components by hash/geometry, match
the full-detail component's vertex groups to the LoD skeleton, and store only
runtime metadata in ``EFMI_MergedSkeleton.json``.  Blender keeps one editable
mesh per component; alternate LoD vertex layouts are generated during export.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.efmi_merged_skeleton import load_profile, write_profile
from ..efmi_extract.migoto_io.migoto_model.migoto_mesh import (
    GeometryMatcherConfig,
    GeometryMatcherMethod,
)
from ..efmi_extract.migoto_io.object_extractor.lod_matcher import (
    ComponentLowSimilarityError,
    LODMatcher,
    ObjectLowSimilarityError,
)
from .dump_workspace_extractor import DumpWorkspaceExtractor, ExtractError


@dataclass
class LODMapResult:
    lod_object_name: str
    matched_component_count: int
    lower_poly_component_count: int
    component_count: int
    max_lod_count: int
    warnings: list[str] = field(default_factory=list)


def _serialize_lod_vb_formats(full_component, lod_component) -> dict:
    full_layout = full_component.mesh.vertex_buffer.layout
    lod_layout = lod_component.mesh.vertex_buffer.layout
    result = {}
    for input_slot in sorted(full_layout.get_input_slots() | lod_layout.get_input_slots()):
        full_slot_layout = full_layout.get_input_slot_layout(input_slot)
        lod_slot_layout = lod_layout.get_input_slot_layout(input_slot)
        if full_slot_layout.to_string() == lod_slot_layout.to_string():
            continue
        result["VB" + str(input_slot)] = {
            "semantics": [
                {
                    "name": semantic.abstract.enum.value,
                    "index": int(semantic.abstract.index),
                    "format": semantic.format.format,
                    "stride": int(semantic.stride),
                }
                for semantic in lod_slot_layout.semantics
            ]
        }
    return result


def _find_full_components(profile: dict, full_object) -> dict[int, object]:
    """Resolve profile component IDs to the rebuilt full-detail components."""
    resolved = {}
    used = set()
    for profile_component in profile["components"]:
        component_id = int(profile_component["component_id"])
        source_id = int(profile_component.get("source_component_id", component_id))
        expected_hash = profile_component["ib_hash"]

        candidate = None
        if 0 <= source_id < len(full_object.components):
            indexed = full_object.components[source_id]
            if indexed.metadata.ib_hash == expected_hash:
                candidate = indexed
        if candidate is None:
            matches = [
                component for component in full_object.components
                if component.metadata.ib_hash == expected_hash and id(component) not in used
            ]
            if len(matches) == 1:
                candidate = matches[0]
        if candidate is None:
            raise ExtractError(
                "完整模型帧与当前骨骼合并工作空间不一致，找不到组件 "
                + str(component_id) + " 的 IB " + expected_hash + "。"
            )
        resolved[component_id] = candidate
        used.add(id(candidate))
    return resolved


def map_merged_skeleton_lod(
    workspace_folder: str,
    full_dump_folder: str,
    lod_dump_folder: str,
    allow_overwrite: bool = True,
) -> LODMapResult:
    profile = load_profile(workspace_folder, required=True)

    full_extractor = DumpWorkspaceExtractor(full_dump_folder)
    full_candidates = full_extractor.get_merged_skeleton_candidates()
    full_object = full_extractor.select_merged_skeleton_object(
        full_candidates,
        object_name=profile.get("object_name", ""),
    )
    if profile.get("object_name") and str(full_object.id) != profile["object_name"]:
        raise ExtractError(
            "完整模型帧中未找到工作空间记录的对象 " + profile["object_name"] + "。"
        )
    full_components = _find_full_components(profile, full_object)

    lod_extractor = DumpWorkspaceExtractor(lod_dump_folder)
    lod_candidates = lod_extractor.get_merged_skeleton_candidates(
        ignore_incomplete_draw_calls=True,
    )
    if not lod_candidates:
        raise ExtractError("LOD 帧中没有识别到可匹配的显式权重角色。")

    matcher = LODMatcher(
        component_min_vertex_count=0,
        component_hash_blacklist="",
        object_similarity_threshold=55.0,
        component_similarity_threshold=55.0,
        skip_components_below_similarity_threshold=False,
        geo_matcher_main_config=GeometryMatcherConfig(
            method=GeometryMatcherMethod.Voxel,
            sensitivity=0.5,
            voxel_size=0.01,
            samples_count=1000,
        ),
        geo_matcher_prefilter_config=GeometryMatcherConfig(
            method=GeometryMatcherMethod.Voxel,
            sensitivity=0.5,
            voxel_size=0.05,
            samples_count=250,
        ),
        geo_matcher_prefilter_candidates_count=5,
        vg_matcher_candidates_count=3,
    )
    try:
        lod_object, matched_components = matcher.find_matching_lods(
            full_object,
            lod_candidates,
        )
    except (ObjectLowSimilarityError, ComponentLowSimilarityError) as exc:
        raise ExtractError("LOD 几何匹配失败: " + str(exc))
    except (KeyError, ValueError) as exc:
        raise ExtractError("LOD 候选匹配失败: " + repr(exc))

    warnings = []
    lower_poly_count = 0
    for profile_component in profile["components"]:
        component_id = int(profile_component["component_id"])
        full_component = full_components[component_id]
        lod_component, vg_map = matched_components.get(full_component, (None, None))
        if lod_component is None:
            # EFMI-Tools writes the full component as a fallback so every
            # component keeps the same number/order of LoD levels.
            lod_component = full_component
            vg_map = None
            vg_offset = int(profile_component.get("vg_offset", 0))
            vg_count = int(profile_component.get("vg_count", 0))
            warning = (
                "主模型 Component " + str(component_id) + "（"
                + profile_component["unique_str"]
                + "）没有独立对应的 LOD。"
            )
            if not profile_component.get("cpu_posed", False) and vg_count > 0:
                warning += (
                    "不要在其他网格上使用该组件负责的全局顶点组 "
                    + str(vg_offset) + "-" + str(vg_offset + vg_count - 1)
                    + " 的权重。"
                )
            else:
                warning += "已记录为主模型回退项。"
            warnings.append(warning)

        lod_metadata = {
            "lod_object_name": str(lod_object.id),
            "ib_hash": str(lod_component.metadata.ib_hash).lower(),
            "vb0_hash": str(lod_component.metadata.vb0_hash).lower(),
            "vertex_offset": int(lod_component.metadata.vertex_offset),
            "vertex_count": int(lod_component.metadata.vertex_count),
            "index_offset": int(lod_component.metadata.index_offset),
            "index_count": int(lod_component.metadata.index_count),
            "vg_map": {
                str(int(full_vg)): int(lod_vg)
                for full_vg, lod_vg in (vg_map or {}).items()
            },
            "vb_formats": _serialize_lod_vb_formats(full_component, lod_component),
        }
        if lod_metadata["ib_hash"] != profile_component["ib_hash"]:
            lower_poly_count += 1

        previous_lods = list(profile_component.get("lods", []))
        duplicate = [
            lod for lod in previous_lods
            if lod.get("lod_object_name") == lod_metadata["lod_object_name"]
        ]
        if duplicate and not allow_overwrite:
            raise ExtractError(
                "LOD 对象 " + lod_metadata["lod_object_name"]
                + " 已存在；请允许覆盖后重试。"
            )
        profile_component["lods"] = [
            lod for lod in previous_lods
            if lod.get("lod_object_name") != lod_metadata["lod_object_name"]
        ] + [lod_metadata]

    profile["source_frame_dump"] = str(full_dump_folder)
    lod_sources = dict(profile.get("lod_sources", {}))
    lod_sources[str(lod_object.id)] = str(lod_dump_folder)
    profile["lod_sources"] = lod_sources
    write_profile(workspace_folder, profile)
    normalized = load_profile(workspace_folder, required=True)
    return LODMapResult(
        lod_object_name=str(lod_object.id),
        matched_component_count=len(matched_components),
        lower_poly_component_count=lower_poly_count,
        component_count=len(profile["components"]),
        max_lod_count=int(normalized.get("max_lod_count", 0)),
        warnings=warnings,
    )
