# -*- coding: utf-8 -*-
"""EFMI Merged Skeleton LoD mapping for LoyalTools workspaces.

The mapper follows EFMI-Tools v0.6.2: rebuild the full-detail and LoD objects
from their FrameAnalysis captures, match components by hash/geometry, match
the full-detail component's vertex groups to the LoD skeleton, and store only
runtime metadata in ``EFMI_MergedSkeleton.json``.  Blender keeps one editable
mesh per component; alternate LoD vertex layouts are generated during export.
"""

from __future__ import annotations

import copy
import json
import os

from dataclasses import dataclass, field

from ..common.efmi_merged_skeleton import (
    LOD_BACKUP_FILENAME,
    LOD_ENTRY_FALLBACK,
    LOD_ENTRY_INCOMPATIBLE,
    LOD_ENTRY_MATCHED,
    PROFILE_FILENAME,
    apply_lod_mapping,
    get_profile_path,
    load_profile,
    profile_has_lod_mapping,
    validate_profile,
    write_profile,
)
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


# Folder name of a custom workspace's data directory (see GlobalConfig).
WORKSPACE_DATA_FOLDER = "workplace"


@dataclass
class LODMapResult:
    lod_object_name: str
    matched_component_count: int
    lower_poly_component_count: int
    component_count: int
    max_lod_count: int
    warnings: list[str] = field(default_factory=list)
    preview_workspace_folder: str = ""
    preview_component_count: int = 0


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
        # Refresh old profiles from the full-detail capture only.  LoD geometry
        # slicing offsets (or the LoD draw itself) are not the main draw's base.
        try:
            full_first_vertex = full_extractor.get_component_first_vertex(full_component)
        except ExtractError as exc:
            raise ExtractError(
                "主模型 Component " + str(component_id) + " 绘制参数无效: " + str(exc)
            ) from exc
        profile_component["first_vertex"] = full_first_vertex
        lod_component, vg_map = matched_components.get(full_component, (None, None))
        is_fallback = lod_component is None
        if is_fallback:
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

        if is_fallback:
            lod_first_index = int(profile_component.get("first_index", 0))
            lod_first_vertex = full_first_vertex
            lod_unique_str = profile_component["unique_str"]
        else:
            lod_primary_key, _, _ = lod_extractor.get_component_primary_draw(lod_component)
            try:
                lod_first_vertex = lod_extractor.get_component_first_vertex(lod_component)
            except ExtractError as exc:
                raise ExtractError(
                    "LOD Component " + str(component_id) + " 绘制参数无效: " + str(exc)
                ) from exc
            lod_first_index = int(lod_primary_key[2])
            lod_unique_str = (
                str(lod_primary_key[0]).lower() + "-" + str(lod_primary_key[1])
                + "-" + str(lod_primary_key[2])
            )

        lod_metadata = {
            "lod_object_name": str(lod_object.id),
            "ib_hash": str(lod_component.metadata.ib_hash).lower(),
            "vb0_hash": str(lod_component.metadata.vb0_hash).lower(),
            "vertex_offset": int(lod_component.metadata.vertex_offset),
            "vertex_count": int(lod_component.metadata.vertex_count),
            "index_offset": int(lod_component.metadata.index_offset),
            "index_count": int(lod_component.metadata.index_count),
            "first_index": lod_first_index,
            "first_vertex": lod_first_vertex,
            "unique_str": lod_unique_str,
            "is_fallback": is_fallback,
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
    profile["last_lod_object_name"] = str(lod_object.id)

    preview_workspace_folder = ""
    preview_component_count = 0
    try:
        preview_workspace_folder = DumpWorkspaceExtractor.get_lod_preview_workspace_folder(
            workspace_folder,
            str(lod_object.id),
        )
        preview_result = lod_extractor.extract_merged_skeleton_preview(
            workspace_folder=preview_workspace_folder,
            selected_object=lod_object,
            object_name=str(lod_object.id),
        )
        preview_component_count = len(preview_result.unique_strs)
        preview_workspaces = dict(profile.get("lod_preview_workspaces", {}))
        preview_workspaces[str(lod_object.id)] = os.path.relpath(
            preview_workspace_folder,
            os.path.abspath(str(workspace_folder)),
        )
        profile["lod_preview_workspaces"] = preview_workspaces
        warnings.extend(
            "LOD 预览: " + warning for warning in preview_result.warnings
        )
    except Exception as exc:
        warnings.append(
            "LOD 映射已保存，但预览网格准备失败；点击“导入 LOD”时会重试: "
            + repr(exc)
        )
    write_profile(workspace_folder, profile)
    normalized = load_profile(workspace_folder, required=True)
    return LODMapResult(
        lod_object_name=str(lod_object.id),
        matched_component_count=len(matched_components),
        lower_poly_component_count=lower_poly_count,
        component_count=len(profile["components"]),
        max_lod_count=int(normalized.get("max_lod_count", 0)),
        warnings=warnings,
        preview_workspace_folder=preview_workspace_folder,
        preview_component_count=preview_component_count,
    )


@dataclass
class LODRestoreResult:
    """Outcome of applying a LoD mapping that was already stored as JSON."""
    lod_object_names: list[str]
    # One entry per LoD object, in the same order.
    matched_component_counts: list[int]
    lower_poly_component_counts: list[int]
    component_count: int
    max_lod_count: int
    source_path: str
    # False when the workspace profile already held this mapping.
    changed: bool
    warnings: list[str] = field(default_factory=list)
    # Background for the log only; nothing the user has to act on.
    notes: list[str] = field(default_factory=list)


def _read_lod_mapping_source(path: str) -> tuple[dict, dict] | None:
    """Load a profile JSON as ``(normalized, raw)``; None without a LoD mapping."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
        normalized = validate_profile(raw)
    except (OSError, ValueError):
        return None
    if not profile_has_lod_mapping(normalized):
        return None
    return normalized, raw


def find_lod_mapping_source(path: str) -> str:
    """Return the JSON at or under ``path`` that carries a LoD mapping, or "".

    ``path`` may be a workspace's data folder, the folder that contains it, or
    a profile JSON itself.  A workspace's current profile is preferred over the
    copy kept from before its full-detail model was extracted again.
    """
    path = os.path.abspath(str(path))
    if os.path.isfile(path):
        candidates = [path]
    else:
        candidates = [
            os.path.join(folder, filename)
            for folder in (path, os.path.join(path, WORKSPACE_DATA_FOLDER))
            for filename in (PROFILE_FILENAME, LOD_BACKUP_FILENAME)
        ]
    for candidate in candidates:
        if os.path.isfile(candidate) and _read_lod_mapping_source(candidate):
            return candidate
    return ""


def _find_preview_workspace(source: dict, source_folder: str, object_name: str) -> str:
    """Locate the LoD preview workspace that belongs to a stored mapping."""
    recorded = source.get("lod_preview_workspaces", {}).get(object_name, "")
    if recorded:
        preview = recorded if os.path.isabs(recorded) else os.path.join(
            source_folder, recorded
        )
    else:
        preview = DumpWorkspaceExtractor.get_lod_preview_workspace_folder(
            source_folder, object_name
        )
    if os.path.isfile(os.path.join(preview, "Import.json")):
        return os.path.abspath(preview)
    return ""


def restore_merged_skeleton_lod(
    workspace_folder: str,
    source_path: str = "",
) -> LODRestoreResult:
    """Apply a LoD mapping that already exists as JSON; no capture is read.

    Without ``source_path`` the workspace's own JSON is used: its current
    profile, or the copy kept when the full-detail model was extracted again.
    ``source_path`` may instead name the profile, or the workspace, of the
    same character that already holds the mapping.
    """
    workspace_folder = os.path.abspath(str(workspace_folder))
    profile = load_profile(workspace_folder, required=True)
    original_profile = copy.deepcopy(profile)
    source_file = find_lod_mapping_source(source_path or workspace_folder)
    if not source_file:
        if source_path:
            raise ExtractError(
                "所选位置没有带 LOD 映射的 " + PROFILE_FILENAME + ": "
                + str(source_path)
            )
        raise ExtractError("当前工作空间的 JSON 中还没有 LOD 映射，请用帧分析匹配。")
    source, source_raw = _read_lod_mapping_source(source_file)
    source_folder = os.path.dirname(source_file)
    already_current = (
        os.path.normcase(os.path.abspath(source_file))
        == os.path.normcase(get_profile_path(workspace_folder))
    )

    reports = apply_lod_mapping(profile, source, source_raw)
    applied = [report for report in reports if report["applied"]]
    if not applied:
        raise ExtractError(
            "JSON 中的 LOD 映射与当前主模型没有可对应的组件，无法套用: " + source_file
        )
    applied_names = [report["lod_object_name"] for report in applied]

    components = {
        component["unique_str"]: component for component in profile["components"]
    }
    warnings = []
    notes = []
    for report in applied:
        for item in report["statuses"]:
            if item["status"] == LOD_ENTRY_MATCHED:
                continue
            component = components[item["unique_str"]]
            warning = (
                "主模型 Component " + str(item["component_id"]) + "（"
                + item["unique_str"] + "）"
            )
            if item["status"] == LOD_ENTRY_FALLBACK:
                warning += "没有独立对应的 LOD。"
            elif item["status"] == LOD_ENTRY_INCOMPATIBLE:
                warning += (
                    "与 JSON 中的 LOD 记录不一致（" + item["problem"]
                    + "），已按没有独立 LOD 处理；需要用帧分析重新匹配。"
                )
            else:
                warning += (
                    "在 JSON 中没有 LOD 记录，已按没有独立 LOD 处理；"
                    "需要用帧分析重新匹配。"
                )
            vg_offset = int(component.get("vg_offset", 0))
            vg_count = int(component.get("vg_count", 0))
            if not component.get("cpu_posed", False) and vg_count > 0:
                warning += (
                    "不要在其他网格上使用该组件负责的全局顶点组 "
                    + str(vg_offset) + "-" + str(vg_offset + vg_count - 1)
                    + " 的权重。"
                )
            warnings.append(warning)
        if report["legacy"]:
            notes.append(
                "JSON 中 " + report["lod_object_name"]
                + " 的映射来自旧版格式，没有记录 LOD 绘制的 first_index/first_vertex；"
                "独立 LOD IB 的这两项按 0 处理。"
            )
    warnings.extend(
        "JSON 中的 LOD 对象 " + report["lod_object_name"]
        + " 与当前主模型没有可对应的组件，已跳过。"
        for report in reports if not report["applied"]
    )

    if not already_current:
        lod_sources = dict(profile.get("lod_sources", {}))
        preview_workspaces = dict(profile.get("lod_preview_workspaces", {}))
        for object_name in applied_names:
            source_dump = source.get("lod_sources", {}).get(object_name, "")
            if source_dump:
                lod_sources[object_name] = source_dump
            preview = _find_preview_workspace(source, source_folder, object_name)
            if not preview:
                continue
            # Stay relative inside this workspace so the folder can be moved;
            # a preview that lives in another workspace is referenced in place.
            relative = os.path.relpath(preview, workspace_folder) if (
                os.path.splitdrive(preview)[0].lower()
                == os.path.splitdrive(workspace_folder)[0].lower()
            ) else ""
            preview_workspaces[object_name] = (
                relative if relative and not relative.startswith("..") else preview
            )
        if lod_sources:
            profile["lod_sources"] = lod_sources
        if preview_workspaces:
            profile["lod_preview_workspaces"] = preview_workspaces
        last_name = str(source.get("last_lod_object_name", "")).strip()
        profile["last_lod_object_name"] = (
            last_name if last_name in applied_names else applied_names[-1]
        )
    else:
        # Applying entries visits lod_sources order, which need not be the
        # stored level order when two LoDs have equal vertex/index counts.
        # Keep existing levels stable while saving any actual repairs.
        for component, original in zip(profile["components"], original_profile["components"]):
            lod_order = {
                lod["lod_object_name"]: index
                for index, lod in enumerate(original["lods"])
            }
            component["lods"].sort(
                key=lambda lod: lod_order.get(lod["lod_object_name"], len(lod_order))
            )

    normalized = validate_profile(profile)
    changed = not already_current or normalized != original_profile
    if changed:
        # Even the current JSON may need legacy fields or fallback entries
        # repaired. Only a semantically unchanged current profile is a no-op.
        write_profile(workspace_folder, normalized)

    normalized = load_profile(workspace_folder, required=True)
    return LODRestoreResult(
        lod_object_names=applied_names,
        matched_component_counts=[report["matched"] for report in applied],
        lower_poly_component_counts=[report["lower_poly"] for report in applied],
        component_count=len(profile["components"]),
        max_lod_count=int(normalized.get("max_lod_count", 0)),
        source_path=source_file,
        changed=changed,
        warnings=warnings,
        notes=notes,
    )


def ensure_lod_preview_workspace(
    workspace_folder: str,
    lod_object_name: str = "",
) -> tuple[str, object]:
    """Create or refresh the isolated preview workspace for a mapped LoD object."""
    profile = load_profile(workspace_folder, required=True)
    object_name = str(
        lod_object_name or profile.get("last_lod_object_name", "")
    ).strip()
    if not object_name:
        object_names = list(profile.get("lod_object_names", []))
        if not object_names:
            raise ExtractError("当前工作空间还没有 LOD 映射。")
        object_name = object_names[-1]

    lod_dump_folder = profile.get("lod_sources", {}).get(object_name, "")
    if not lod_dump_folder or not os.path.isdir(lod_dump_folder):
        raise ExtractError(
            "找不到 LOD " + object_name + " 的帧分析来源，请重新执行 LOD 映射。"
        )

    extractor = DumpWorkspaceExtractor(lod_dump_folder)
    candidates = extractor.get_merged_skeleton_candidates(
        ignore_incomplete_draw_calls=True,
    )
    lod_object = extractor.select_merged_skeleton_object(
        candidates,
        object_name=object_name,
    )
    if str(lod_object.id) != object_name:
        raise ExtractError("LOD 帧中找不到对象 " + object_name + "。")

    preview_workspace = DumpWorkspaceExtractor.get_lod_preview_workspace_folder(
        workspace_folder,
        object_name,
    )
    result = extractor.extract_merged_skeleton_preview(
        workspace_folder=preview_workspace,
        selected_object=lod_object,
        object_name=object_name,
    )
    preview_workspaces = dict(profile.get("lod_preview_workspaces", {}))
    preview_workspaces[object_name] = os.path.relpath(
        preview_workspace,
        os.path.abspath(str(workspace_folder)),
    )
    profile["lod_preview_workspaces"] = preview_workspaces
    profile["last_lod_object_name"] = object_name
    write_profile(workspace_folder, profile)
    return preview_workspace, result
