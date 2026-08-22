# -*- coding: utf-8 -*-
"""EFMI 1.4.1 Merged Skeleton workspace contract.

This module deliberately has no ``bpy`` dependency.  Extraction, Blender import,
and EFMI export all communicate through the same workspace-level profile while
the normal LoyalTools workflow continues to use Import.json/SubmeshJson only.
"""

from __future__ import annotations

import json
import os


PROFILE_FILENAME = "EFMI_MergedSkeleton.json"
PROFILE_MODE = "EFMI_MERGED_SKELETON"
PROFILE_FORMAT_VERSION = 1
REQUIRED_EFMI_VERSION = "1.4.1"
MAX_VERTEX_GROUP_ID = 65535


class MergedSkeletonProfileError(ValueError):
    pass


def get_profile_path(workspace_folder: str) -> str:
    return os.path.join(os.path.abspath(str(workspace_folder)), PROFILE_FILENAME)


def _as_non_negative_int(value, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise MergedSkeletonProfileError(field_name + " 必须是整数。")
    if result < 0:
        raise MergedSkeletonProfileError(field_name + " 不能为负数。")
    return result


def normalize_vg_map(vg_map, field_name: str = "vg_map") -> dict[str, int]:
    if vg_map is None:
        return {}
    if not isinstance(vg_map, dict):
        raise MergedSkeletonProfileError(field_name + " 必须是对象映射。")

    normalized: dict[str, int] = {}
    for local_id, global_id in vg_map.items():
        local_int = _as_non_negative_int(local_id, field_name + " 的本地顶点组")
        global_int = _as_non_negative_int(global_id, field_name + " 的全局顶点组")
        if global_int > MAX_VERTEX_GROUP_ID:
            raise MergedSkeletonProfileError(
                field_name + " 中的全局顶点组 " + str(global_int)
                + " 超过 R16_UINT 上限 " + str(MAX_VERTEX_GROUP_ID) + "。"
            )
        normalized[str(local_int)] = global_int
    return dict(sorted(normalized.items(), key=lambda item: int(item[0])))


def _normalize_lod_vb_formats(vb_formats, field_name: str) -> dict[str, dict]:
    if vb_formats is None:
        return {}
    if not isinstance(vb_formats, dict):
        raise MergedSkeletonProfileError(field_name + " 必须是对象。")

    normalized = {}
    for raw_slot, raw_buffer in vb_formats.items():
        slot = str(raw_slot).upper()
        if not slot.startswith("VB") or not slot[2:].isdigit():
            raise MergedSkeletonProfileError(field_name + " 的槽位无效: " + slot)
        if not isinstance(raw_buffer, dict):
            raise MergedSkeletonProfileError(field_name + "." + slot + " 必须是对象。")
        raw_semantics = raw_buffer.get("semantics", [])
        if not isinstance(raw_semantics, list) or not raw_semantics:
            raise MergedSkeletonProfileError(
                field_name + "." + slot + ".semantics 不能为空。"
            )

        semantics = []
        for semantic_id, raw_semantic in enumerate(raw_semantics):
            semantic_field = (
                field_name + "." + slot + ".semantics[" + str(semantic_id) + "]"
            )
            if not isinstance(raw_semantic, dict):
                raise MergedSkeletonProfileError(semantic_field + " 必须是对象。")
            name = str(raw_semantic.get("name", "")).upper().strip()
            fmt = str(raw_semantic.get("format", "")).upper().strip()
            if not name or not fmt:
                raise MergedSkeletonProfileError(semantic_field + " 缺少 name/format。")
            semantics.append({
                "name": name,
                "index": _as_non_negative_int(
                    raw_semantic.get("index", 0), semantic_field + ".index"
                ),
                "format": fmt,
                "stride": _as_non_negative_int(
                    raw_semantic.get("stride", 0), semantic_field + ".stride"
                ),
            })
        normalized[slot] = {"semantics": semantics}
    return dict(sorted(normalized.items(), key=lambda item: int(item[0][2:])))


def normalize_lod(lod, field_name: str = "lod") -> dict:
    if not isinstance(lod, dict):
        raise MergedSkeletonProfileError(field_name + " 必须是对象。")
    result = dict(lod)
    result["lod_object_name"] = str(result.get("lod_object_name", "")).strip()
    if not result["lod_object_name"]:
        raise MergedSkeletonProfileError(field_name + " 缺少 lod_object_name。")
    result["ib_hash"] = str(result.get("ib_hash", "")).lower().strip()
    result["vb0_hash"] = str(result.get("vb0_hash", "")).lower().strip()
    if not result["ib_hash"]:
        raise MergedSkeletonProfileError(field_name + " 缺少 ib_hash。")
    for key in ("vertex_offset", "vertex_count", "index_offset", "index_count"):
        result[key] = _as_non_negative_int(result.get(key, 0), field_name + "." + key)
    result["vg_map"] = normalize_vg_map(
        result.get("vg_map", {}), field_name + ".vg_map"
    )
    result["vb_formats"] = _normalize_lod_vb_formats(
        result.get("vb_formats", {}), field_name + ".vb_formats"
    )
    return result


def validate_profile(profile: dict) -> dict:
    if not isinstance(profile, dict):
        raise MergedSkeletonProfileError("骨骼合并配置根节点必须是 JSON 对象。")
    if profile.get("mode") != PROFILE_MODE:
        raise MergedSkeletonProfileError(
            "骨骼合并配置 mode 无效，期望 " + PROFILE_MODE + "。"
        )

    format_version = _as_non_negative_int(
        profile.get("format_version", 0), "format_version"
    )
    if format_version != PROFILE_FORMAT_VERSION:
        raise MergedSkeletonProfileError(
            "不支持的骨骼合并配置版本: " + str(format_version)
        )

    components = profile.get("components")
    if not isinstance(components, list) or not components:
        raise MergedSkeletonProfileError("骨骼合并配置没有 components。")

    normalized_components = []
    seen_component_ids = set()
    seen_unique_strs = set()
    max_global_vg = -1
    for list_index, raw_component in enumerate(components):
        if not isinstance(raw_component, dict):
            raise MergedSkeletonProfileError(
                "components[" + str(list_index) + "] 必须是对象。"
            )

        component = dict(raw_component)
        component_id = _as_non_negative_int(
            component.get("component_id", list_index),
            "components[" + str(list_index) + "].component_id",
        )
        unique_str = str(component.get("unique_str", "")).strip()
        if not unique_str:
            raise MergedSkeletonProfileError(
                "components[" + str(list_index) + "] 缺少 unique_str。"
            )
        if component_id in seen_component_ids:
            raise MergedSkeletonProfileError("component_id 重复: " + str(component_id))
        if unique_str in seen_unique_strs:
            raise MergedSkeletonProfileError("unique_str 重复: " + unique_str)
        seen_component_ids.add(component_id)
        seen_unique_strs.add(unique_str)

        component["component_id"] = component_id
        component["unique_str"] = unique_str
        component["ib_hash"] = str(component.get("ib_hash", "")).lower()
        component["index_count"] = _as_non_negative_int(
            component.get("index_count", 0), unique_str + ".index_count"
        )
        component["first_index"] = _as_non_negative_int(
            component.get("first_index", 0), unique_str + ".first_index"
        )
        component["vertex_count"] = _as_non_negative_int(
            component.get("vertex_count", 0), unique_str + ".vertex_count"
        )
        component["cpu_posed"] = bool(component.get("cpu_posed", False))
        component["vg_offset"] = _as_non_negative_int(
            component.get("vg_offset", 0), unique_str + ".vg_offset"
        )
        component["vg_count"] = _as_non_negative_int(
            component.get("vg_count", 0), unique_str + ".vg_count"
        )
        component["vg_map"] = normalize_vg_map(
            component.get("vg_map", {}), unique_str + ".vg_map"
        )
        raw_lods = component.get("lods", [])
        if not isinstance(raw_lods, list):
            raise MergedSkeletonProfileError(unique_str + ".lods 必须是数组。")
        component["lods"] = [
            normalize_lod(lod, unique_str + ".lods[" + str(lod_id) + "]")
            for lod_id, lod in enumerate(raw_lods)
        ]
        component["lods"].sort(
            key=lambda lod: (lod["vertex_count"], lod["index_count"]),
            reverse=True,
        )
        lod_object_names = [lod["lod_object_name"] for lod in component["lods"]]
        if len(lod_object_names) != len(set(lod_object_names)):
            raise MergedSkeletonProfileError(
                unique_str + ".lods 中 lod_object_name 重复。"
            )

        if not component["cpu_posed"] and component["vg_count"] > 0:
            missing = [
                local_id for local_id in range(component["vg_count"])
                if str(local_id) not in component["vg_map"]
            ]
            if missing:
                raise MergedSkeletonProfileError(
                    unique_str + " 的 vg_map 不完整，缺少本地顶点组: "
                    + ", ".join(map(str, missing[:8]))
                )
        if component["vg_map"]:
            max_global_vg = max(max_global_vg, max(component["vg_map"].values()))

        normalized_components.append(component)

    normalized_components.sort(key=lambda item: item["component_id"])
    expected_ids = list(range(len(normalized_components)))
    actual_ids = [component["component_id"] for component in normalized_components]
    if actual_ids != expected_ids:
        raise MergedSkeletonProfileError(
            "component_id 必须从 0 连续编号，当前为: " + str(actual_ids)
        )

    result = dict(profile)
    result["format_version"] = PROFILE_FORMAT_VERSION
    result["mode"] = PROFILE_MODE
    result["required_efmi_version"] = REQUIRED_EFMI_VERSION
    result["components"] = normalized_components
    result["component_count"] = len(normalized_components)
    result["bones_count"] = sum(
        component["vg_count"]
        for component in normalized_components
        if not component["cpu_posed"]
    )
    result["max_global_vertex_group"] = max_global_vg
    result["max_instance_count"] = max(
        1, _as_non_negative_int(result.get("max_instance_count", 8), "max_instance_count")
    )
    result["object_guid"] = _as_non_negative_int(
        result.get(
            "object_guid",
            sum(component["index_count"] for component in normalized_components),
        ),
        "object_guid",
    )
    result["source_frame_dump"] = str(result.get("source_frame_dump", "")).strip()
    result["max_lod_count"] = max(
        (len(component["lods"]) for component in normalized_components),
        default=0,
    )
    result["lod_object_names"] = sorted({
        lod["lod_object_name"]
        for component in normalized_components
        for lod in component["lods"]
    })
    return result


def load_profile(workspace_folder: str, required: bool = True) -> dict | None:
    path = get_profile_path(workspace_folder)
    if not os.path.isfile(path):
        if required:
            raise MergedSkeletonProfileError(
                "当前工作空间没有 " + PROFILE_FILENAME + "，请先用“骨骼合并”模式提取。"
            )
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            profile = json.load(file)
    except Exception as exc:
        raise MergedSkeletonProfileError("读取骨骼合并配置失败: " + repr(exc))
    return validate_profile(profile)


def write_profile(workspace_folder: str, profile: dict) -> str:
    normalized = validate_profile(profile)
    path = get_profile_path(workspace_folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=4)
    return path


def get_component_by_unique_str(profile: dict) -> dict[str, dict]:
    normalized = validate_profile(profile)
    return {
        component["unique_str"]: component
        for component in normalized["components"]
    }


def make_submesh_metadata(component: dict) -> dict:
    """Return the small metadata block embedded in LoyalTools SubmeshJson."""
    return {
        "Profile": PROFILE_FILENAME,
        "ComponentId": int(component["component_id"]),
        "CpuPosed": bool(component.get("cpu_posed", False)),
        "VGOffset": int(component.get("vg_offset", 0)),
        "VGCount": int(component.get("vg_count", 0)),
        "VGMap": normalize_vg_map(component.get("vg_map", {})),
    }


def parse_submesh_metadata(json_dict: dict) -> dict | None:
    raw = json_dict.get("EFMIMergedSkeleton")
    if not isinstance(raw, dict):
        return None
    return {
        "profile": str(raw.get("Profile", PROFILE_FILENAME)),
        "component_id": _as_non_negative_int(raw.get("ComponentId", 0), "ComponentId"),
        "cpu_posed": bool(raw.get("CpuPosed", False)),
        "vg_offset": _as_non_negative_int(raw.get("VGOffset", 0), "VGOffset"),
        "vg_count": _as_non_negative_int(raw.get("VGCount", 0), "VGCount"),
        "vg_map": normalize_vg_map(raw.get("VGMap", {}), "VGMap"),
    }
