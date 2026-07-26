from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Iterable

import bpy

from ...blueprint.model import BluePrintModel
from ...common.draw_call_model import DrawCallModel
from ...common.object_prefix_helper import ObjectPrefixHelper


PROP_KIND = "modimp_kind"
PROP_PROFILE_ID = "modimp_profile_id"
PROP_SOURCE_IB_HASH = "modimp_source_ib_hash"
PROP_REGION_HASH = "modimp_region_hash"
PROP_REGION_INDEX_COUNT = "modimp_region_index_count"
PROP_REGION_FIRST_INDEX = "modimp_region_first_index"
PROP_PART_INDEX = "modimp_part_index"
PROP_BMC_IB_HASH = "modimp_bmc_ib_hash"
PROP_BMC_MATCH_INDEX_COUNT = "modimp_bmc_match_index_count"
PROP_BMC_CHUNK_INDEX = "modimp_bmc_chunk_index"
PROP_MATCH_VS_TEXCOORD_HASH = "modimp_match_vs_texcoord_hash"
PROP_MATCH_VS_POSITION_HASH = "modimp_match_vs_position_hash"
PROP_MATCH_VS_OUTLINE_HASH = "modimp_match_vs_outline_hash"
PROP_TEXTURE_SLOTS = "modimp_texture_slots"
PROP_DRAW_TOGGLE = "modimp_draw_toggle"
PROP_DRAW_TOGGLE_KEY = "modimp_draw_toggle_key"
PROP_COLLECTOR_GROUP_SLOT = "modimp_collector_group_slot"
PROP_COLLECTOR_T0_HASH = "modimp_collector_t0_hash"
PROP_COLLECTOR_U0_HASH = "modimp_collector_u0_hash"
PROP_COLLECTOR_U1_HASH = "modimp_collector_u1_hash"
PROP_COLLECTOR_COLLECT_KEY = "modimp_collector_collect_key"
PROP_COLLECTOR_FINISH_CONDITION = "modimp_collector_finish_condition"

PROFILE_ID = "yihuan"

RUNTIME_REGION_PROPS = (
    PROP_PROFILE_ID,
    PROP_MATCH_VS_TEXCOORD_HASH,
    PROP_MATCH_VS_POSITION_HASH,
    PROP_MATCH_VS_OUTLINE_HASH,
    PROP_TEXTURE_SLOTS,
)

RUNTIME_REGION_CONTRACT_PROPS = (
    PROP_MATCH_VS_TEXCOORD_HASH,
    PROP_MATCH_VS_POSITION_HASH,
    PROP_MATCH_VS_OUTLINE_HASH,
    PROP_TEXTURE_SLOTS,
)

COLLECTOR_PROPS = (
    PROP_COLLECTOR_GROUP_SLOT,
    PROP_COLLECTOR_T0_HASH,
    PROP_COLLECTOR_U0_HASH,
    PROP_COLLECTOR_U1_HASH,
    PROP_COLLECTOR_COLLECT_KEY,
    PROP_COLLECTOR_FINISH_CONDITION,
)

HASH8_RE = re.compile(r"^[0-9A-Fa-f]{8}$")


@dataclass
class RegionBuildRecord:
    draw_ib: str
    region_hash: str
    index_count: int
    first_index: int
    collection_name: str
    object_names: list[str] = field(default_factory=list)
    object_conditions: dict[str, str] = field(default_factory=dict)
    has_runtime_contract: bool = False
    missing_contract_fields: list[str] = field(default_factory=list)


@dataclass
class SourceBuildRecord:
    draw_ib: str
    collection_name: str
    region_records: list[RegionBuildRecord] = field(default_factory=list)
    has_collector_contract: bool = False
    missing_collector_fields: list[str] = field(default_factory=list)


@dataclass
class ExportTreeBuildResult:
    root_collections: list[bpy.types.Collection]
    source_records: list[SourceBuildRecord]
    warnings: list[str] = field(default_factory=list)
    created_collection_names: list[str] = field(default_factory=list)

    def has_full_ini_contract(self) -> bool:
        if not self.source_records:
            return False
        for source_record in self.source_records:
            if not source_record.has_collector_contract:
                return False
            for region_record in source_record.region_records:
                if not region_record.has_runtime_contract:
                    return False
        return True


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_str(owner, key: str) -> str:
    try:
        return str(owner.get(key, "") or "").strip()
    except Exception:
        return ""


def _set_optional_prop(owner, key: str, value):
    if value is None:
        return
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return
    owner[key] = value


def _has_any_prop(owner, keys: Iterable[str]) -> bool:
    return any(_optional_str(owner, key) for key in keys)


def _copy_props(source, target, keys: Iterable[str]):
    for key in keys:
        value = _optional_str(source, key)
        if value:
            target[key] = value


def _unique_collection_name(base_name: str) -> str:
    candidate = base_name[:60] if len(base_name) > 60 else base_name
    if bpy.data.collections.get(candidate) is None:
        return candidate

    suffix = 1
    while True:
        numbered = f"{candidate[:55]}_{suffix:03d}"
        if bpy.data.collections.get(numbered) is None:
            return numbered
        suffix += 1


def _ensure_scene_linked(collection: bpy.types.Collection):
    scene_children = bpy.context.scene.collection.children
    if collection.name not in scene_children.keys():
        scene_children.link(collection)


def _collection_parents(target: bpy.types.Collection) -> list[bpy.types.Collection]:
    parents = []
    for collection in bpy.data.collections:
        if target.name in collection.children.keys():
            parents.append(collection)
    scene = getattr(bpy.context, "scene", None)
    if scene and target.name in scene.collection.children.keys():
        parents.append(scene.collection)
    return parents


def unlink_and_remove_collection_tree(collection: bpy.types.Collection):
    for child in list(collection.children):
        unlink_and_remove_collection_tree(child)

    for obj in list(collection.objects):
        collection.objects.unlink(obj)

    for parent in _collection_parents(collection):
        try:
            parent.children.unlink(collection)
        except Exception:
            pass

    if bpy.data.collections.get(collection.name) is not None:
        bpy.data.collections.remove(collection)


def cleanup_collections(collection_names: Iterable[str]):
    for collection_name in list(collection_names):
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            unlink_and_remove_collection_tree(collection)


def _draw_key(draw_call_model: DrawCallModel) -> tuple[str, int, int] | None:
    draw_ib = str(draw_call_model.match_draw_ib or "").strip().lower()
    index_count = _safe_int(draw_call_model.match_index_count)
    first_index = _safe_int(draw_call_model.match_first_index)

    if not HASH8_RE.fullmatch(draw_ib):
        return None
    if index_count is None or first_index is None:
        return None
    return draw_ib, index_count, first_index


def _find_chain_for_draw(blueprint_model: BluePrintModel, draw_call_model: DrawCallModel):
    draw_obj = draw_call_model.get_blender_obj_name()
    draw_export = draw_call_model.obj_name
    for chain in getattr(blueprint_model, "processing_chains", []) or []:
        if not getattr(chain, "is_valid", False) or not getattr(chain, "reached_output", False):
            continue
        candidates = {
            getattr(chain, "object_name", "") or "",
            getattr(chain, "original_object_name", "") or "",
            getattr(chain, "virtual_object_name", "") or "",
            getattr(chain, "export_object_name_override", "") or "",
        }
        get_export_object_name = getattr(chain, "get_export_object_name", None)
        if callable(get_export_object_name):
            try:
                candidates.add(get_export_object_name() or "")
            except Exception:
                pass
        if draw_obj in candidates or draw_export in candidates:
            return chain
    return None


def _condition_from_work_keys(work_key_list) -> str:
    conditions = []
    for work_key in work_key_list or []:
        if not bool(getattr(work_key, "is_swapkey", False)):
            continue
        key_name = str(getattr(work_key, "key_name", "") or "").strip()
        if not key_name:
            continue
        value = getattr(work_key, "tmp_value", 0)
        condition = f"{key_name} == {value}"
        if not conditions:
            conditions.append(condition)
        else:
            operator = str(getattr(work_key, "condition_operator", "&&") or "&&").strip()
            if operator not in {"&&", "||"}:
                operator = "&&"
            conditions.append(f"{operator} {condition}")
    return " ".join(conditions)


def condition_from_swap_work_keys(work_key_list) -> str:
    return _condition_from_work_keys(work_key_list)


def _chain_condition(chain) -> str:
    conditions = []
    swap_condition = _condition_from_work_keys(getattr(chain, "shapekey_params", []) or [])
    if swap_condition:
        conditions.append(f"({swap_condition})" if "&&" in swap_condition or "||" in swap_condition else swap_condition)
    explicit_condition = str(getattr(chain, "ntmi_multifile_condition", "") or "").strip()
    if explicit_condition:
        conditions.append(
            f"({explicit_condition})"
            if "&&" in explicit_condition or "||" in explicit_condition
            else explicit_condition
        )
    return " && ".join(conditions)


def _source_owner_candidates(obj: bpy.types.Object):
    yield obj

    base_name = obj.name
    if base_name.endswith("_copy"):
        original = bpy.data.objects.get(base_name[:-5])
        if original is not None:
            yield original
            for collection in getattr(original, "users_collection", []) or []:
                yield collection

    parsed_prefix = ObjectPrefixHelper.extract_prefix_info(obj.name)
    if parsed_prefix:
        prefix, _separator = parsed_prefix
        for collection_name in (
            prefix,
            f"{prefix}_Export",
            f"{ObjectPrefixHelper.parse_prefix_parts(prefix).get('draw_ib', '')}_Export",
        ):
            collection = bpy.data.collections.get(collection_name)
            if collection is not None:
                yield collection

    for collection in getattr(obj, "users_collection", []) or []:
        yield collection


def _source_root_candidates(draw_ib: str):
    normalized = str(draw_ib or "").strip().lower()
    if not normalized:
        return

    for collection_name in (normalized, f"{normalized}_Export"):
        collection = bpy.data.collections.get(collection_name)
        if collection is not None:
            yield collection

    for collection in bpy.data.collections:
        source_hash = _optional_str(collection, PROP_SOURCE_IB_HASH).lower()
        if source_hash == normalized and _optional_str(collection, PROP_KIND) in {"source_ib", "export_root"}:
            yield collection


def _source_region_candidates(draw_ib: str, index_count: int, first_index: int):
    normalized = str(draw_ib or "").strip().lower()
    if not normalized:
        return

    candidate_names = (
        f"{normalized}-{int(index_count)}-{int(first_index)}",
        f"{normalized}_{int(index_count)}_{int(first_index)}",
        normalized,
    )

    seen = set()
    for root_collection in _source_root_candidates(normalized):
        for child in root_collection.children:
            if child.name in seen:
                continue
            seen.add(child.name)
            region_hash = _optional_str(child, PROP_REGION_HASH).lower()
            child_index_count = _safe_int(child.get(PROP_REGION_INDEX_COUNT))
            child_first_index = _safe_int(child.get(PROP_REGION_FIRST_INDEX))
            if (
                region_hash == normalized
                and child_index_count == int(index_count)
                and child_first_index == int(first_index)
            ):
                yield child

    for collection_name in candidate_names:
        collection = bpy.data.collections.get(collection_name)
        if collection is None or collection.name in seen:
            continue
        seen.add(collection.name)
        yield collection


def _first_existing_owner(obj: bpy.types.Object):
    for owner in _source_owner_candidates(obj):
        if _has_any_prop(owner, RUNTIME_REGION_PROPS + COLLECTOR_PROPS):
            return owner
    return obj


def _texture_slots_from_materials(obj: bpy.types.Object) -> str:
    slots = {}
    if obj.type != "MESH":
        return ""

    def image_path_from_input(material, input_name: str) -> str:
        if not material or not getattr(material, "use_nodes", False) or material.node_tree is None:
            return ""
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:
            return ""
        input_socket = bsdf.inputs.get(input_name)
        if input_socket is None:
            return ""
        visited = set()

        def find_image(socket):
            for link in getattr(socket, "links", []) or []:
                node = link.from_node
                if id(node) in visited:
                    continue
                visited.add(id(node))
                if getattr(node, "bl_idname", "") == "ShaderNodeTexImage":
                    return getattr(node, "image", None)
                for nested_input in getattr(node, "inputs", []) or []:
                    image = find_image(nested_input)
                    if image is not None:
                        return image
            return None

        image = find_image(input_socket)
        if image is None:
            return ""
        path = str(getattr(image, "filepath", "") or "").strip()
        if not path:
            return ""
        try:
            path = bpy.path.abspath(path)
        except Exception:
            pass
        return path

    for material_slot in getattr(obj, "material_slots", []) or []:
        material = getattr(material_slot, "material", None)
        base_path = image_path_from_input(material, "Base Color")
        normal_path = image_path_from_input(material, "Normal")
        if base_path:
            slots.setdefault("ps-t7", {"source_path": base_path, "extension": base_path.rsplit(".", 1)[-1]})
        if normal_path:
            slots.setdefault("ps-t5", {"source_path": normal_path, "extension": normal_path.rsplit(".", 1)[-1]})

    return json.dumps(slots, ensure_ascii=False) if slots else ""


def _copy_runtime_contract(
    obj: bpy.types.Object,
    region_collection: bpy.types.Collection,
    *,
    draw_ib: str,
    index_count: int,
    first_index: int,
):
    for source_owner in _source_owner_candidates(obj):
        _copy_props(source_owner, region_collection, RUNTIME_REGION_PROPS)
        if _region_contract_status(region_collection)[0]:
            break

    if not _region_contract_status(region_collection)[0]:
        for source_region in _source_region_candidates(draw_ib, index_count, first_index):
            _copy_props(source_region, region_collection, RUNTIME_REGION_PROPS)
            if _region_contract_status(region_collection)[0]:
                break

    if PROP_TEXTURE_SLOTS not in region_collection:
        texture_slots = _texture_slots_from_materials(obj)
        if texture_slots:
            region_collection[PROP_TEXTURE_SLOTS] = texture_slots


def _copy_collector_props(objects: list[bpy.types.Object], source_collection: bpy.types.Collection):
    owners = []
    for obj in objects:
        owners.extend(_source_owner_candidates(obj))

    for owner in owners:
        _copy_props(owner, source_collection, COLLECTOR_PROPS)
        if _collector_contract_status(source_collection)[0]:
            break


def _link_object(collection: bpy.types.Collection, obj: bpy.types.Object):
    if obj.name not in collection.objects.keys():
        collection.objects.link(obj)


def _mark_source_collection(collection: bpy.types.Collection, draw_ib: str):
    collection[PROP_KIND] = "source_ib"
    collection[PROP_PROFILE_ID] = PROFILE_ID
    collection[PROP_SOURCE_IB_HASH] = draw_ib.lower()


def _mark_region_collection(
    collection: bpy.types.Collection,
    *,
    draw_ib: str,
    index_count: int,
    first_index: int,
):
    collection[PROP_KIND] = "region"
    collection[PROP_PROFILE_ID] = PROFILE_ID
    collection[PROP_SOURCE_IB_HASH] = draw_ib.lower()
    collection[PROP_REGION_HASH] = draw_ib.lower()
    collection[PROP_REGION_INDEX_COUNT] = int(index_count)
    collection[PROP_REGION_FIRST_INDEX] = int(first_index)


def _region_contract_status(collection: bpy.types.Collection) -> tuple[bool, list[str]]:
    missing = []
    for key in (PROP_MATCH_VS_TEXCOORD_HASH, PROP_MATCH_VS_POSITION_HASH):
        if not _optional_str(collection, key):
            missing.append(key)
    return not missing, missing


def _collector_contract_status(collection: bpy.types.Collection) -> tuple[bool, list[str]]:
    missing = []
    for key in (
        PROP_COLLECTOR_GROUP_SLOT,
        PROP_COLLECTOR_U0_HASH,
        PROP_COLLECTOR_U1_HASH,
        PROP_COLLECTOR_COLLECT_KEY,
        PROP_COLLECTOR_FINISH_CONDITION,
    ):
        if not _optional_str(collection, key):
            missing.append(key)
    return not missing, missing


def build_export_tree(blueprint_model: BluePrintModel, tree_prefix: str = "LoyalTools_NTMI_ModImp") -> ExportTreeBuildResult:
    warnings = []
    created_names = []
    grouped: dict[str, dict[tuple[int, int], list[tuple[DrawCallModel, bpy.types.Object, object]]]] = {}

    for draw_call_model in blueprint_model.ordered_draw_obj_data_model_list:
        key = _draw_key(draw_call_model)
        if key is None:
            warnings.append(f"Skip invalid draw identity: {draw_call_model.obj_name}")
            continue

        obj = bpy.data.objects.get(draw_call_model.get_blender_obj_name())
        if obj is None:
            warnings.append(f"Skip missing object: {draw_call_model.get_blender_obj_name()}")
            continue
        if obj.type != "MESH":
            warnings.append(f"Skip non-mesh object: {obj.name}")
            continue

        draw_ib, index_count, first_index = key
        chain = _find_chain_for_draw(blueprint_model, draw_call_model)
        grouped.setdefault(draw_ib, {}).setdefault((index_count, first_index), []).append(
            (draw_call_model, obj, chain)
        )

    root_collections = []
    source_records = []

    for draw_ib, region_map in sorted(grouped.items()):
        root_name = _unique_collection_name(f"{tree_prefix}_{draw_ib}")
        root_collection = bpy.data.collections.new(root_name)
        _ensure_scene_linked(root_collection)
        _mark_source_collection(root_collection, draw_ib)
        created_names.append(root_collection.name)
        root_collections.append(root_collection)

        all_source_objects = [item[1] for region_items in region_map.values() for item in region_items]
        _copy_collector_props(all_source_objects, root_collection)
        has_collector_contract, missing_collector_fields = _collector_contract_status(root_collection)

        source_record = SourceBuildRecord(
            draw_ib=draw_ib,
            collection_name=root_collection.name,
            has_collector_contract=has_collector_contract,
            missing_collector_fields=missing_collector_fields,
        )
        source_records.append(source_record)

        for (index_count, first_index), items in sorted(region_map.items(), key=lambda item: (item[0][1], item[0][0])):
            region_name = _unique_collection_name(f"{draw_ib}-{index_count}-{first_index}")
            region_collection = bpy.data.collections.new(region_name)
            root_collection.children.link(region_collection)
            _mark_region_collection(
                region_collection,
                draw_ib=draw_ib,
                index_count=index_count,
                first_index=first_index,
            )
            created_names.append(region_collection.name)

            for draw_call_model, obj, chain in items:
                _copy_runtime_contract(
                    obj,
                    region_collection,
                    draw_ib=draw_ib,
                    index_count=index_count,
                    first_index=first_index,
                )
                condition = _condition_from_work_keys(draw_call_model.work_key_list)
                if not condition and chain is not None:
                    condition = _chain_condition(chain)
                _link_object(region_collection, obj)

            has_contract, missing = _region_contract_status(region_collection)
            object_conditions = {}
            for draw_call_model, obj, chain in items:
                condition = _condition_from_work_keys(draw_call_model.work_key_list)
                if not condition and chain is not None:
                    condition = _chain_condition(chain)
                if condition:
                    object_conditions[obj.name] = condition
            source_record.region_records.append(
                RegionBuildRecord(
                    draw_ib=draw_ib,
                    region_hash=draw_ib,
                    index_count=index_count,
                    first_index=first_index,
                    collection_name=region_collection.name,
                    object_names=[item[1].name for item in items],
                    object_conditions=object_conditions,
                    has_runtime_contract=has_contract,
                    missing_contract_fields=missing,
                )
            )

    return ExportTreeBuildResult(
        root_collections=root_collections,
        source_records=source_records,
        warnings=warnings,
        created_collection_names=created_names,
    )
