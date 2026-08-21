import json
import os
import shutil
from dataclasses import dataclass


_STANDARD_MAP_NAMES = ("DiffuseMap", "LightMap", "NormalMap")


@dataclass(frozen=True)
class MergedAutoTextureBinding:
    """骨骼合并导出的自动槽位贴图；接口兼容 TextureMarkUpInfo。"""

    mark_name: str
    mark_slot: str
    mark_hash: str
    mark_filename: str
    source_path: str
    mark_type: str = "Slot"

    def get_resource_name(self):
        return "Resource-" + os.path.splitext(self.mark_filename)[0]


def _slot_number(slot_name: str) -> int | None:
    normalized = str(slot_name or "").strip().lower()
    if not normalized.startswith("ps-t"):
        return None
    try:
        return int(normalized[4:])
    except ValueError:
        return None


def _entry_area(entry: dict) -> int:
    try:
        return max(0, int(entry.get("width", 0))) * max(0, int(entry.get("height", 0)))
    except (TypeError, ValueError):
        return 0


def _is_srgb(entry: dict) -> bool:
    return "SRGB" in str(entry.get("format", "")).upper()


def _is_linear(entry: dict) -> bool:
    return not _is_srgb(entry)


def _is_normal_hint(entry: dict | None) -> bool:
    if entry is None:
        return False
    format_name = str(entry.get("format", "")).upper()
    return "BC5" in format_name or "NORMAL" in format_name


def _candidate_sequence(entries_by_slot: dict[int, dict], diffuse_slot: int):
    diffuse = entries_by_slot[diffuse_slot]
    next_one = entries_by_slot.get(diffuse_slot + 1)
    next_two = entries_by_slot.get(diffuse_slot + 2)

    light = None
    normal = None
    if next_one is not None and _is_linear(next_one):
        if next_two is not None and _is_linear(next_two):
            light = next_one
            normal = next_two
        else:
            normal = next_one
    elif next_two is not None and _is_normal_hint(next_two):
        normal = next_two

    bindings = [("DiffuseMap", diffuse_slot, diffuse)]
    if light is not None:
        bindings.append(("LightMap", diffuse_slot + 1, light))
    if normal is not None:
        normal_slot = diffuse_slot + (2 if light is not None else 1)
        bindings.append(("NormalMap", normal_slot, normal))
    return bindings


def _select_material_sequence(calls: dict) -> list[tuple[str, int, dict]]:
    candidates = []
    for call_name, raw_entries in calls.items():
        entries_by_slot = {}
        for entry in raw_entries if isinstance(raw_entries, list) else []:
            slot_number = _slot_number(entry.get("slot", ""))
            if slot_number is None or not entry.get("filename"):
                continue
            entries_by_slot[slot_number] = entry
        if not entries_by_slot:
            continue

        diffuse_slots = [
            slot_number for slot_number, entry in entries_by_slot.items()
            if _is_srgb(entry)
        ]
        if not diffuse_slots:
            # 极少数特效组件没有 sRGB 贴图；仍给它保留一张独立 DiffuseMap。
            diffuse_slots = list(entries_by_slot.keys())

        for diffuse_slot in diffuse_slots:
            bindings = _candidate_sequence(entries_by_slot, diffuse_slot)
            diffuse_entry = bindings[0][2]
            normal_entry = next(
                (entry for map_name, _, entry in bindings if map_name == "NormalMap"),
                None,
            )
            map_count = len(bindings)
            high_material_slot = int(diffuse_slot >= 10)
            normal_hint = int(_is_normal_hint(normal_entry))
            area = _entry_area(diffuse_entry)
            dimension_match = 0
            if normal_entry is not None and area:
                dimension_match = int(_entry_area(normal_entry) == area)
            try:
                call_id = int(call_name)
            except (TypeError, ValueError):
                call_id = 0

            # 优先完整的连续材质组，其次优先高位主材质槽和法线格式。
            # 同分时使用更大贴图，并偏向较低的材质起始槽，避免选到尾部通用占位贴图。
            score = (
                map_count,
                high_material_slot,
                normal_hint,
                dimension_match,
                area,
                -diffuse_slot,
                call_id,
            )
            candidates.append((score, bindings))

    if not candidates:
        return []
    return max(candidates, key=lambda item: item[0])[1]


def resolve_merged_auto_textures(
    texture_folder: str,
    unique_str: str,
    blocked_map_names: set[str] | None = None,
    blocked_slots: set[str] | None = None,
) -> list[MergedAutoTextureBinding]:
    """从 TextureSlots.json 自动选择一组独立的 Diffuse/Light/Normal。"""

    blocked_map_names = set(blocked_map_names or ())
    blocked_slots = {str(slot).lower() for slot in (blocked_slots or ())}
    slots_path = os.path.join(texture_folder, "TextureSlots.json")
    if not os.path.isfile(slots_path):
        return []

    try:
        with open(slots_path, "r", encoding="utf-8") as handle:
            texture_slots = json.load(handle)
    except (OSError, ValueError, TypeError):
        return []

    # 旧工作空间的 TextureSlots.json 聚合了相同 IB 的辅助绘制，不能安全自动绑定。
    # 必须由新版骨骼合并提取写入组件级过滤标记；普通流程文件也不会带此标记。
    if texture_slots.get("EFMIMergedComponentScoped") is not True:
        return []

    calls = texture_slots.get("calls", {})
    if not isinstance(calls, dict):
        return []

    bindings = []
    for map_name, slot_number, entry in _select_material_sequence(calls):
        slot_name = "ps-t" + str(slot_number)
        if map_name not in _STANDARD_MAP_NAMES:
            continue
        if map_name in blocked_map_names or slot_name.lower() in blocked_slots:
            continue

        source_filename = os.path.basename(str(entry.get("filename", "")))
        source_path = os.path.join(texture_folder, source_filename)
        if not os.path.isfile(source_path):
            continue
        extension = os.path.splitext(source_filename)[1].lower() or ".dds"
        output_filename = unique_str + "-" + map_name + extension
        canonical_source_path = os.path.join(texture_folder, output_filename)
        if os.path.isfile(canonical_source_path):
            source_path = canonical_source_path
        bindings.append(MergedAutoTextureBinding(
            mark_name=map_name,
            mark_slot=slot_name,
            mark_hash=str(entry.get("hash", "")).lower(),
            mark_filename=output_filename,
            source_path=source_path,
        ))
    return bindings


def copy_merged_auto_textures(
    bindings: list[MergedAutoTextureBinding],
    output_folder: str,
) -> int:
    """按 IB 独立复制贴图；目标已存在时与普通流程一样保留用户修改。"""

    os.makedirs(output_folder, exist_ok=True)
    copied_count = 0
    for binding in bindings:
        target_path = os.path.join(output_folder, binding.mark_filename)
        if os.path.exists(target_path):
            continue
        shutil.copy2(binding.source_path, target_path)
        copied_count += 1
    return copied_count
