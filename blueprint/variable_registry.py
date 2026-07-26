import re
from typing import Iterable, Optional

import bpy


OBJECT_SWAP_PREFIX = "swapkey"
SHAPEKEY_PREFIX = "Freq_"

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _get_scene_global_properties(context=None):
    scene = getattr(context, "scene", None) if context is not None else getattr(bpy.context, "scene", None)
    if scene is None:
        return None
    return getattr(scene, "global_properties", None)


def _sanitize_name(text: str, fallback: str = "var") -> str:
    safe_text = re.sub(r"\s+", "_", str(text or "").strip())
    safe_text = _SAFE_NAME_RE.sub("", safe_text)
    if safe_text and safe_text[0].isdigit():
        safe_text = "_" + safe_text
    return safe_text or fallback


def _split_csv(text: str) -> list[str]:
    return [item for item in str(text or "").split(",") if item]


def _join_csv(values: Iterable[str]) -> str:
    return ",".join(values)


def get_used_variable_names(context=None) -> set[str]:
    props = _get_scene_global_properties(context)
    if props is None:
        return set()
    return set(_split_csv(getattr(props, "allocated_variable_names_csv", "")))


def mark_variable_name_used(var_name: str, context=None):
    normalized = normalize_variable_name(var_name)
    if not normalized:
        return
    props = _get_scene_global_properties(context)
    if props is None:
        return
    used_names = get_used_variable_names(context)
    if normalized in used_names:
        return
    used_names.add(normalized)
    props.allocated_variable_names_csv = _join_csv(sorted(used_names))


def normalize_variable_name(var_name: str) -> str:
    cleaned = str(var_name or "").strip()
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    return _sanitize_name(cleaned, fallback="")


def ensure_object_swap_variable_name(node, context=None) -> str:
    current = normalize_variable_name(getattr(node, "assigned_variable_name", ""))
    if current:
        mark_variable_name_used(current, context=context)
        return current

    props = _get_scene_global_properties(context)
    if props is None:
        current = f"{OBJECT_SWAP_PREFIX}0"
        node.assigned_variable_name = current
        return current

    next_index = int(getattr(props, "object_swap_variable_counter", 0) or 0)
    used_names = get_used_variable_names(context)
    while True:
        candidate = f"{OBJECT_SWAP_PREFIX}{next_index}"
        next_index += 1
        if candidate in used_names:
            continue
        used_names.add(candidate)
        props.object_swap_variable_counter = next_index
        props.allocated_variable_names_csv = _join_csv(sorted(used_names))
        node.assigned_variable_name = candidate
        return candidate


def allocate_shape_key_variable_name(shape_key_name: str, *, preferred: Optional[str] = None, context=None) -> str:
    props = _get_scene_global_properties(context)
    preferred_normalized = normalize_variable_name(preferred or "")
    used_names = get_used_variable_names(context)

    if preferred_normalized:
        if preferred_normalized not in used_names:
            mark_variable_name_used(preferred_normalized, context=context)
        return preferred_normalized

    base_name = _sanitize_name(shape_key_name, fallback="shape")
    candidate = f"{SHAPEKEY_PREFIX}{base_name}"
    if candidate not in used_names:
        mark_variable_name_used(candidate, context=context)
        return candidate

    suffix = 1
    while True:
        indexed = f"{candidate}_{suffix}"
        if indexed not in used_names:
            mark_variable_name_used(indexed, context=context)
            return indexed
        suffix += 1


def get_node_variable_name(node, context=None) -> str:
    assigned_name = ensure_object_swap_variable_name(node, context=context)
    manual_name = normalize_variable_name(getattr(node, "custom_var_name", ""))
    resolved = manual_name or assigned_name
    mark_variable_name_used(resolved, context=context)
    return f"${resolved}"
