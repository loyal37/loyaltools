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
    for key in (
        "vertex_offset", "vertex_count", "index_offset", "index_count", "first_index"
    ):
        result[key] = _as_non_negative_int(result.get(key, 0), field_name + "." + key)
    result["unique_str"] = str(result.get("unique_str", "")).strip()
    if not result["unique_str"]:
        result["unique_str"] = (
            result["ib_hash"] + "-" + str(result["index_count"])
            + "-" + str(result["first_index"])
        )
    result["is_fallback"] = bool(result.get("is_fallback", False))
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


def build_component_private_vg_remap(component: dict) -> dict[int, int]:
    """Map a component's canonical global groups back to its private pool range.

    Extraction intentionally deduplicates identical bones across components, so
    ``vg_map`` may point at another component's canonical global group.  EFMI's
    LoD importer, however, always writes the current component's matrices to
    ``vg_offset + local_id``.  Exported Blend buffers must therefore read the
    component-private destination for every bone that the component itself can
    provide.  Groups that are not present in this component's ``vg_map`` are
    intentional cross-component weights and are deliberately left untouched.

    When multiple local bones were deduplicated to the same canonical group, we
    prefer the already-canonical private slot (when available), then the lowest
    local id.  Those source matrices are identical by the deduplication contract.
    """
    vg_offset = _as_non_negative_int(
        component.get("vg_offset", 0), "component.vg_offset"
    )
    vg_count = _as_non_negative_int(
        component.get("vg_count", 0), "component.vg_count"
    )
    vg_map = normalize_vg_map(component.get("vg_map", {}), "component.vg_map")

    local_ids_by_global: dict[int, list[int]] = {}
    for local_id_text, global_id in vg_map.items():
        local_id = int(local_id_text)
        if local_id >= vg_count:
            raise MergedSkeletonProfileError(
                "component.vg_map 的本地顶点组 " + str(local_id)
                + " 超过 vg_count=" + str(vg_count) + "。"
            )
        local_ids_by_global.setdefault(int(global_id), []).append(local_id)

    remap: dict[int, int] = {}
    for global_id, local_ids in local_ids_by_global.items():
        canonical_local_id = global_id - vg_offset
        if canonical_local_id in local_ids:
            target_local_id = canonical_local_id
        else:
            target_local_id = min(local_ids)
        remap[global_id] = vg_offset + target_local_id
    return remap


def build_lod_mapping_groups(profile: dict) -> list[dict]:
    """Build UI-friendly LoD unique_str -> full-detail unique_str groups."""
    normalized = validate_profile(profile)
    object_order = []
    for object_name in normalized.get("lod_sources", {}):
        if object_name in normalized.get("lod_object_names", []):
            object_order.append(object_name)
    for object_name in normalized.get("lod_object_names", []):
        if object_name not in object_order:
            object_order.append(object_name)

    groups = []
    for object_name in object_order:
        rows = []
        for component in normalized["components"]:
            lod = next(
                (
                    item for item in component.get("lods", [])
                    if item.get("lod_object_name") == object_name
                ),
                None,
            )
            if lod is None:
                continue
            rows.append({
                "component_id": int(component["component_id"]),
                "lod_unique_str": lod["unique_str"],
                "main_unique_str": component["unique_str"],
                "is_fallback": bool(lod.get("is_fallback", False)),
            })
        groups.append({"lod_object_name": object_name, "rows": rows})
    return groups


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


# ---------------------------------------------------------------------------
# Weight ownership resolution (cross-IB safe)
#
# A merged-skeleton object carries two independent identities:
#   * which draw/IB the geometry is rendered by  -> derived from the object name
#   * which component supplies the bone matrices -> fixed at import time
# Renaming an object to render it through another IB deliberately changes only
# the first one.  The exporter must therefore resolve the weight owner from the
# persistent import marker instead of the object name, otherwise the private
# pool rewrite is applied with the wrong component and the exported
# BLENDINDICES keep canonical ids that no LoD callback refreshes.
# ---------------------------------------------------------------------------

OBJECT_MARKER_MERGED_SKELETON = "LoyalTools:EFMIMergedSkeleton"
OBJECT_MARKER_COMPONENT_ID = "LoyalTools:EFMIComponentId"
OBJECT_MARKER_VG_OFFSET = "LoyalTools:EFMIVGOffset"
OBJECT_MARKER_VG_COUNT = "LoyalTools:EFMIVGCount"

OWNER_SOURCE_MARKER = "marker"
OWNER_SOURCE_UNIQUE_STR = "unique_str"


def get_component_by_id(profile: dict) -> dict[int, dict]:
    """Index the profile components by their ``component_id``."""
    return {
        int(component["component_id"]): component
        for component in profile.get("components", [])
    }


def resolve_weight_owner_component(
    profile: dict,
    component_id=None,
    unique_str_candidates=(),
) -> tuple[dict | None, str | None]:
    """Resolve which component owns an object's vertex weights.

    ``component_id`` is the persistent ``LoyalTools:EFMIComponentId`` marker
    written at import time; it survives renaming and is authoritative.
    ``unique_str_candidates`` is an ordered fallback list (mesh data name first,
    then object name) for projects imported before the marker existed.

    Returns ``(component, source)``; ``source`` is ``None`` when unresolved.
    """
    if component_id is not None:
        try:
            wanted_id = int(component_id)
        except (TypeError, ValueError):
            wanted_id = None
        if wanted_id is not None:
            component = get_component_by_id(profile).get(wanted_id)
            if component is not None:
                return component, OWNER_SOURCE_MARKER

    by_unique = {
        component["unique_str"]: component
        for component in profile.get("components", [])
    }
    for candidate in unique_str_candidates:
        if not candidate:
            continue
        component = by_unique.get(candidate)
        if component is not None:
            return component, OWNER_SOURCE_UNIQUE_STR

    return None, None


def find_component_owning_global_vg(profile: dict, global_id: int) -> dict | None:
    """Return the component whose private bone pool covers ``global_id``.

    EFMI writes a component's matrices to ``vg_offset + local_id``, so the pool
    that actually supplies a global vertex group is the one whose
    ``[vg_offset, vg_offset + vg_count)`` window contains it.
    """
    target = int(global_id)
    for component in profile.get("components", []):
        offset = int(component.get("vg_offset", 0))
        count = int(component.get("vg_count", 0))
        if offset <= target < offset + count:
            return component
    return None


def _component_lod_is_fallback(component: dict, lod_level: int) -> bool:
    """True when the component supplies no independent data at ``lod_level``."""
    lods = component.get("lods", [])
    if len(lods) < lod_level:
        return True
    return bool(lods[lod_level - 1].get("is_fallback", False))


def component_draw_levels(profile: dict, component: dict | None) -> set[int]:
    """LoD levels (``1..max_lod_count``) at which ``component`` is drawn.

    A component is drawn, and captures its own bones, at every level whose LoD
    entry is not a fallback.  ``None`` stands for an unknown draw and is treated
    as visible at every level.
    """
    max_lod_count = int(profile.get("max_lod_count", 0) or 0)
    levels = range(1, max_lod_count + 1)
    if component is None:
        return set(levels)
    if component.get("cpu_posed", False):
        return set()
    return {
        level for level in levels
        if not _component_lod_is_fallback(component, level)
    }


def _pool_refreshed_at(component: dict, draw_levels) -> bool:
    """True when ``component`` rewrites its private pool at every draw level.

    Every GPU component captures its bones at LoD 0.  CPU posed components are
    never attached to the merged skeleton, so their pool is never written.
    """
    if component.get("cpu_posed", False):
        return False
    return not any(
        _component_lod_is_fallback(component, level) for level in draw_levels
    )


def build_stale_pool_substitution(
    profile: dict,
    used_global_ids,
    drawing_component: dict | None = None,
) -> dict[int, int]:
    """Redirect bone ids whose pool goes stale while the draw is still visible.

    Extraction deduplicates identical bones, so one bone can be read from the
    private slot of every component whose ``vg_map`` lists it.  A slot is only
    refreshed while its component is drawn, though: a fallback LoD entry means
    the component is not drawn at that level, so its pool keeps stale matrices.
    Geometry that reaches such a pool without an owner rewrite looks correct up
    close and breaks at distance.  Joining a component into another object is
    the common case, because the joined object keeps only the active object's
    import marker.

    Stale ids are moved to equivalent slots of components that are refreshed at
    every level the draw is visible.  Components are chosen greedily by how
    many of the stale ids they supply, so a joined part keeps reading all of
    its bones from one component, which is normally the one it came from.
    Ties prefer the drawing component, then the lowest component id.  Ids with
    no refreshed equivalent are left unchanged for the audit to report.
    """
    draw_levels = component_draw_levels(profile, drawing_component)
    if not draw_levels:
        return {}
    drawing_id = (
        int(drawing_component["component_id"])
        if drawing_component is not None else None
    )

    stale_bones: dict[int, int] = {}
    for slot in sorted({int(value) for value in used_global_ids}):
        owner = find_component_owning_global_vg(profile, slot)
        if owner is None or _pool_refreshed_at(owner, draw_levels):
            continue
        owner_map = normalize_vg_map(owner.get("vg_map", {}), "component.vg_map")
        local_id = slot - int(owner.get("vg_offset", 0))
        stale_bones[slot] = int(owner_map.get(str(local_id), slot))
    if not stale_bones:
        return {}

    slots_by_component: dict[int, dict[int, int]] = {}
    for component in profile.get("components", []):
        if not _pool_refreshed_at(component, draw_levels):
            continue
        offset = int(component.get("vg_offset", 0))
        vg_map = normalize_vg_map(component.get("vg_map", {}), "component.vg_map")
        bone_slots = slots_by_component.setdefault(int(component["component_id"]), {})
        for local_id, bone in vg_map.items():
            slot = offset + int(local_id)
            bone_slots[int(bone)] = min(slot, bone_slots.get(int(bone), slot))

    substitution: dict[int, int] = {}
    remaining = dict(stale_bones)
    while remaining:
        best = None
        for component_id, bone_slots in slots_by_component.items():
            covered = [slot for slot, bone in remaining.items() if bone in bone_slots]
            if not covered:
                continue
            rank = (-len(covered), component_id != drawing_id, component_id)
            if best is None or rank < best[0]:
                best = (rank, component_id, covered)
        if best is None:
            break
        _, component_id, covered = best
        for slot in covered:
            substitution[slot] = slots_by_component[component_id][remaining.pop(slot)]
    return substitution


def audit_exported_global_vgs(
    profile: dict,
    used_global_ids,
    drawing_component_id=None,
    label: str = "",
) -> list[str]:
    """Report exported bone ids that no component reliably supplies.

    Two conditions are reported, both of which produce a mesh that looks correct
    up close and breaks at distance:

    * the id falls outside every component's private pool, so nothing ever
      writes a matrix there;
    * the id belongs to another component whose LoD entry is a fallback at a
      level where the drawing component is still visible.  ``限制-005``
      forbids borrowing such a range precisely because the far level has no
      component callback to supply the matrices.

    Returns human-readable warning lines; an empty list means nothing to report.
    """
    prefix = (label + ": ") if label else ""

    try:
        owner_id = int(drawing_component_id)
    except (TypeError, ValueError):
        owner_id = None
    draw_levels = component_draw_levels(
        profile,
        get_component_by_id(profile).get(owner_id) if owner_id is not None else None,
    )

    unowned: list[int] = []
    borrowed_fallback: dict[int, list[int]] = {}
    borrowed_cpu: dict[int, list[int]] = {}

    for global_id in sorted({int(value) for value in used_global_ids}):
        component = find_component_owning_global_vg(profile, global_id)
        if component is None:
            unowned.append(global_id)
            continue

        component_id = int(component["component_id"])
        if owner_id is not None and component_id == owner_id:
            continue

        if bool(component.get("cpu_posed", False)):
            borrowed_cpu.setdefault(component_id, []).append(global_id)
            continue

        if any(
            _component_lod_is_fallback(component, level)
            for level in draw_levels
        ):
            borrowed_fallback.setdefault(component_id, []).append(global_id)

    warnings: list[str] = []
    if unowned:
        warnings.append(
            prefix + "权重引用了不属于任何组件骨骼池的全局顶点组 "
            + format_id_ranges(unowned)
            + "，运行时没有任何组件回调会写入这些矩阵。"
        )
    for component_id, ids in sorted(borrowed_fallback.items()):
        component = get_component_by_id(profile)[component_id]
        warnings.append(
            prefix + "借用了 Component " + str(component_id) + " ("
            + str(component.get("unique_str", "?")) + ") 的骨骼池 "
            + format_id_ranges(ids)
            + "，但该组件没有独立 LOD（回退项），也没有其他组件提供等价骨骼。"
            "近景正常，远景这些权重不会被刷新（见 限制-005）。"
        )
    for component_id, ids in sorted(borrowed_cpu.items()):
        component = get_component_by_id(profile)[component_id]
        warnings.append(
            prefix + "借用了 CPU posed Component " + str(component_id) + " ("
            + str(component.get("unique_str", "?")) + ") 的骨骼池 "
            + format_id_ranges(ids)
            + "，CPU 组件不参与合并骨骼初始化，这些矩阵不会被写入。"
        )
    return warnings


def format_id_ranges(ids) -> str:
    """Collapse a sorted id list into a compact ``a-b, c`` representation."""
    values = sorted({int(value) for value in ids})
    if not values:
        return ""
    groups = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append((start, previous))
        start = previous = value
    groups.append((start, previous))
    return ", ".join(
        str(low) if low == high else str(low) + "-" + str(high)
        for low, high in groups
    )
