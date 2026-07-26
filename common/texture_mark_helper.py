# -*- coding: utf-8 -*-
'''
贴图标记辅助模块 (尽量不依赖 bpy)。

负责:
- 从物体名称解析 unique_str (DrawIB-IndexCount-FirstIndex)
- 通过 submesh_metadata.check_and_get_submesh_json_path 定位 SubmeshJson 与 TYPE_ 提取目录
- 列出候选贴图 (提取器写出的 TextureSlots.json v1/v2 + 目录扫描)
- 按绘制调用分组列出候选贴图 (TextureSlots.json v2 "calls" 分组)
- 读取 / 写入 / 移除 SubmeshJson 中的 TextureMarkUpInfoList 标记条目

TextureSlots.json 兼容两个版本:
- v1: 顶层即槽位字典 {"ps-t<N>": [{"hash","filename","format","call_id"}...]}
- v2: {"version": 2,
       "slots": {"ps-t<N>": [{"hash","filename","format","call_id","width","height"}...]},
       "calls": {"<6位绘制调用id>": [{"slot","hash","filename","format","width","height"}...]}}

标记条目 Schema (与 texture_metadata_helper.py / m_ini_helper.py 消费端保持一致):
{
    "MarkName": str,            # DiffuseMap / NormalMap / LightMap / 自定义
    "MarkType": "Slot"|"Hash",
    "MarkHash": str,            # 8 位十六进制贴图 hash
    "MarkSlot": str,            # 例如 "ps-t3"
    "MarkFileName": str,        # 必须存在于 TYPE_ 提取目录中的文件名
                                # (SSMT4 风格命名副本 "<unique_str>-<MarkName>.<后缀>")
    "SourceFileName": str       # 标记时选中的原始候选贴图文件名 (仅面板内部使用，
                                # 消费端 texture_metadata_helper / submesh_json 会忽略多余键)
}
'''
import json
import os
import re
import shutil
import tempfile


TEXTURE_SLOTS_JSON_NAME = "TextureSlots.json"
TEXTURE_MARK_LIST_KEY = "TextureMarkUpInfoList"

_SLOT_TOKEN_PATTERN = re.compile(r"ps-t(\d+)")
_HASH_TOKEN_PATTERN = re.compile(r"[0-9a-fA-F]{8}")
_BLENDER_SUFFIX_PATTERN = re.compile(r"\.\d{3,}$")
_TEXTURE_EXT_SET = {".dds", ".jpg", ".jpeg", ".png", ".tga"}
_MARK_NAME_INVALID_CHAR_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
# SSMT4 风格命名副本主干: "<DrawIB>-<IndexCount>-<FirstIndex>-<MarkName>"
# (提取器文件名 "t-<8hex>-<FORMAT>" / "ps-t<N>-..." 首段不足 6 位字母数字，不会误判)
_NAMED_MARK_COPY_STEM_PATTERN = re.compile(r"^[0-9a-zA-Z]{6,}-\d+-\d+-[A-Za-z0-9_]+$")


class TextureMarkError(ValueError):
    '''贴图标记相关错误 (携带中文提示信息)'''
    pass


# ----------------------------------------------------------------------
# unique_str 解析与 SubmeshJson 定位
# ----------------------------------------------------------------------

def resolve_unique_str_from_object_name(object_name: str) -> str | None:
    '''从物体名称解析 unique_str。

    物体名称形如 "<DrawIB>-<IndexCount>-<FirstIndex>[.Alias][.001]"，
    优先复用 ObjectPrefixHelper 的前缀解析逻辑 (需要 bpy 环境)，
    在无 bpy 环境下退化为纯字符串解析。

    Returns:
        unique_str 字符串，无法解析时返回 None
    '''
    clean_name = (object_name or "").strip()
    if not clean_name:
        return None

    prefix = ""
    try:
        from .object_prefix_helper import ObjectPrefixHelper
    except Exception:
        ObjectPrefixHelper = None

    if ObjectPrefixHelper is not None:
        parsed = ObjectPrefixHelper.extract_prefix_info(clean_name)
        if parsed:
            prefix = parsed[0]

    if not prefix:
        # 退化解析: 去掉 ".Alias" / Blender 数字后缀后取前三个 '-' 段
        base_name = clean_name.split(".", 1)[0]
        base_name = _BLENDER_SUFFIX_PATTERN.sub("", base_name)
        prefix = base_name

    parts = [part.strip() for part in prefix.split("-") if part.strip()]
    if len(parts) < 3:
        return None

    draw_ib, index_count, first_index = parts[0], parts[1], parts[2]
    if len(draw_ib) < 6 or not draw_ib.isalnum():
        return None
    if not index_count.isdigit() or not first_index.isdigit():
        return None

    return draw_ib + "-" + index_count + "-" + first_index


def locate_submesh_json(unique_str: str) -> tuple[str, str]:
    '''定位 unique_str 对应的 SubmeshJson 文件与 TYPE_ 提取目录。

    Returns:
        (json_path, type_folder)

    Raises:
        TextureMarkError: 无法定位时抛出 (中文提示)
    '''
    from .submesh_metadata import check_and_get_submesh_json_path

    exists, error_msg, submesh_json_path = check_and_get_submesh_json_path(unique_str)
    if not exists:
        raise TextureMarkError(error_msg)

    return submesh_json_path, os.path.dirname(submesh_json_path)


# ----------------------------------------------------------------------
# 候选贴图列举
# ----------------------------------------------------------------------

def _slot_sort_key(slot_key: str) -> int:
    match = _SLOT_TOKEN_PATTERN.fullmatch(slot_key or "")
    if match:
        return int(match.group(1))
    return 9999


def _parse_tokens_from_filename(filename: str) -> tuple[str, str, str]:
    '''从提取器生成的文件名解析 (slot, hash, format) 令牌。

    当前提取器命名规则 (单令牌, 不含空格 / 等号 / 槽位号): "t-<8hex>-<FORMAT>.<dds|jpg>"。
    文件名刻意不含槽位号: 同一 hash 的贴图在不同绘制调用中可能绑定到不同槽位，
    槽位真值以 TextureSlots.json v2 的 "calls" 分组为准，因此新版命名返回 slot=""。

    兼容旧版命名 (仅作为兜底):
    - "ps-t<N>-<8hex>-<FORMAT>": 解析 hash 与 format；文件名中的槽位号仅在
      TextureSlots.json 完全缺失时作为最后手段的槽位提示返回
    - 更旧的 "ps-t<N> t=<8hex> <FORMAT>" (含空格 / 等号): 返回空字符串令牌

    "<unique_str>-DiffuseMap" 等不满足规则的文件同样返回空字符串令牌。
    '''
    stem = os.path.splitext(filename or "")[0]

    slot = ""
    tex_hash = ""
    format_name = ""

    # 新版命名 (无槽位令牌): "t-<8hex>-<FORMAT>"
    if stem.startswith("t-"):
        rest = stem[2:]
        hash_match = _HASH_TOKEN_PATTERN.match(rest)
        if hash_match and (hash_match.end() == len(rest) or rest[hash_match.end()] == "-"):
            tex_hash = hash_match.group(0).lower()
            format_name = rest[hash_match.end():].lstrip("-")
            return "", tex_hash, format_name

    # 旧版命名兜底: "ps-t<N>-<8hex>-<FORMAT>" (slot 令牌位于主干开头且后跟 '-' 或结尾，
    # 因此 "<unique_str>-DiffuseMap" 等文件不会误判出插槽信息)
    slot_match = _SLOT_TOKEN_PATTERN.match(stem)
    if slot_match and (slot_match.end() == len(stem) or stem[slot_match.end()] == "-"):
        slot = "ps-t" + slot_match.group(1)
        rest = stem[slot_match.end():].lstrip("-")
        hash_match = _HASH_TOKEN_PATTERN.match(rest)
        if hash_match and (hash_match.end() == len(rest) or rest[hash_match.end()] == "-"):
            tex_hash = hash_match.group(0).lower()
            format_name = rest[hash_match.end():].lstrip("-")

    return slot, tex_hash, format_name


def _load_texture_slots_json(type_folder: str) -> dict:
    slots_json_path = os.path.join(type_folder, TEXTURE_SLOTS_JSON_NAME)
    if not os.path.isfile(slots_json_path):
        return {}

    try:
        with open(slots_json_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
    except (OSError, ValueError):
        return {}

    if not isinstance(raw_json, dict):
        return {}
    return raw_json


def _is_v2_texture_slots(raw_json: dict) -> bool:
    '''判断 TextureSlots.json 是否为 v2 布局 ({"version","slots","calls"})。'''
    if not isinstance(raw_json, dict):
        return False
    if isinstance(raw_json.get("slots"), dict) or isinstance(raw_json.get("calls"), dict):
        return True
    return "version" in raw_json


def _extract_slot_map(raw_json: dict) -> dict:
    '''从 v1/v2 的 TextureSlots.json 内容中取出槽位字典 {"ps-tN": [...]}。'''
    if not isinstance(raw_json, dict):
        return {}
    if _is_v2_texture_slots(raw_json):
        slot_map = raw_json.get("slots")
        return slot_map if isinstance(slot_map, dict) else {}
    return raw_json


def _coerce_positive_int(value) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return 0
    return int_value if int_value > 0 else 0


def _normalize_call_id(raw_call_id) -> str:
    '''归一化绘制调用 id 为 6 位零填充字符串 (无法解析时原样返回字符串)。'''
    text = str(raw_call_id if raw_call_id is not None else "").strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(6)
    return text


def _read_jpeg_dimensions(file_obj) -> tuple[int, int]:
    '''扫描 JPEG SOF 段读取 (width, height)，失败返回 (0, 0)。'''
    if file_obj.read(2) != b'\xff\xd8':
        return 0, 0
    while True:
        marker = file_obj.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return 0, 0
        code = marker[1]
        while code == 0xFF:
            padded = file_obj.read(1)
            if not padded:
                return 0, 0
            code = padded[0]
        if code == 0x01 or 0xD0 <= code <= 0xD8:
            continue
        length_bytes = file_obj.read(2)
        if len(length_bytes) < 2:
            return 0, 0
        segment_length = int.from_bytes(length_bytes, 'big')
        if segment_length < 2:
            return 0, 0
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            body = file_obj.read(5)
            if len(body) < 5:
                return 0, 0
            height = int.from_bytes(body[1:3], 'big')
            width = int.from_bytes(body[3:5], 'big')
            return width, height
        file_obj.seek(segment_length - 2, os.SEEK_CUR)


def _read_texture_dimensions(file_path: str) -> tuple[int, int]:
    '''从贴图文件头读取 (width, height) (支持 DDS/PNG/TGA/JPEG)，失败返回 (0, 0)。'''
    ext = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, 'rb') as f:
            if ext == '.dds':
                header = f.read(20)
                if len(header) >= 20 and header[:4] == b'DDS ':
                    height = int.from_bytes(header[12:16], 'little')
                    width = int.from_bytes(header[16:20], 'little')
                    return width, height
            elif ext == '.png':
                header = f.read(24)
                if len(header) >= 24 and header[:8] == b'\x89PNG\r\n\x1a\n':
                    width = int.from_bytes(header[16:20], 'big')
                    height = int.from_bytes(header[20:24], 'big')
                    return width, height
            elif ext == '.tga':
                header = f.read(18)
                if len(header) >= 18:
                    width = int.from_bytes(header[12:14], 'little')
                    height = int.from_bytes(header[14:16], 'little')
                    return width, height
            elif ext in ('.jpg', '.jpeg'):
                return _read_jpeg_dimensions(f)
    except OSError:
        pass
    return 0, 0


def _resolve_entry_dimensions(entry: dict, file_path: str) -> tuple[int, int]:
    '''优先取 json 条目内的 width/height，缺失或非法时回退解析文件头。'''
    width = _coerce_positive_int(entry.get("width"))
    height = _coerce_positive_int(entry.get("height"))
    if not width or not height:
        width, height = _read_texture_dimensions(file_path)
    return width, height


def list_candidate_textures(type_folder: str) -> list[dict]:
    '''列出 TYPE_ 提取目录内的候选贴图 (扁平列表, 跨绘制调用去重)。

    数据来源:
    1. 提取器写出的 TextureSlots.json (v1 顶层槽位字典 / v2 "slots" 槽位聚合)
    2. 目录内的贴图文件扫描 (从 "t-<8hex>-<FORMAT>" / 旧版 "ps-t<N>-<8hex>-<FORMAT>"
       文件名解析令牌)

    Returns:
        list[dict]: [{"filename": str, "slot": str, "hash": str, "format": str,
                      "width": int, "height": int, "call_id": str}, ...]
        按文件名去重，TextureSlots.json 记录优先，槽位号升序。
        同一 hash 同时存在新旧命名文件时按 hash 去重 (json 引用的文件优先，
        目录扫描内新版 "t-" 命名文件优先)，避免重新提取覆盖旧工作空间后出现重复条目。
        width/height 未知时为 0；call_id 为贴图 dump 文件的调用 id (可能为空)。
    '''
    candidate_list: list[dict] = []
    seen_filenames: set[str] = set()
    json_hash_set: set[str] = set()

    if not type_folder or not os.path.isdir(type_folder):
        return candidate_list

    slot_map = _extract_slot_map(_load_texture_slots_json(type_folder))
    for slot_key in sorted(slot_map.keys(), key=_slot_sort_key):
        entry_list = slot_map.get(slot_key)
        if not isinstance(entry_list, list):
            continue
        for entry in entry_list:
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("filename", "") or "")
            if not filename or filename in seen_filenames:
                continue
            file_path = os.path.join(type_folder, filename)
            if not os.path.isfile(file_path):
                continue
            width, height = _resolve_entry_dimensions(entry, file_path)
            entry_hash = str(entry.get("hash", "") or "").lower()
            seen_filenames.add(filename)
            if entry_hash:
                json_hash_set.add(entry_hash)
            candidate_list.append({
                "filename": filename,
                "slot": slot_key,
                "hash": entry_hash,
                "format": str(entry.get("format", "") or ""),
                "width": width,
                "height": height,
                "call_id": _normalize_call_id(entry.get("call_id")),
            })

    # 目录扫描补充 (未被 TextureSlots.json 覆盖的贴图文件)
    try:
        dir_filenames = sorted(os.listdir(type_folder))
    except OSError:
        dir_filenames = []

    scanned_entries: list[tuple[str, str, str, str]] = []
    for filename in dir_filenames:
        if filename in seen_filenames:
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _TEXTURE_EXT_SET:
            continue
        # 标记生成的 SSMT4 风格命名副本 "<unique_str>-<MarkName>" 不是候选贴图
        if is_named_mark_copy_filename(filename):
            continue
        if not os.path.isfile(os.path.join(type_folder, filename)):
            continue
        slot, tex_hash, format_name = _parse_tokens_from_filename(filename)
        scanned_entries.append((filename, slot, tex_hash, format_name))

    # hash 级去重: 重新提取覆盖旧工作空间后，同一 hash 可能同时存在
    # 新版 "t-" 命名文件与残留的旧版 "ps-tN-" 命名文件
    # - json 已引用的 hash 优先 (跳过残留旧文件)
    # - 目录扫描内部同 hash 同时存在新旧命名时保留新版 "t-" 命名文件
    new_named_hash_set = {
        tex_hash for filename, _, tex_hash, _ in scanned_entries
        if tex_hash and filename.startswith("t-")
    }

    for filename, slot, tex_hash, format_name in scanned_entries:
        if tex_hash:
            if tex_hash in json_hash_set:
                continue
            if tex_hash in new_named_hash_set and not filename.startswith("t-"):
                continue
        width, height = _read_texture_dimensions(os.path.join(type_folder, filename))
        seen_filenames.add(filename)
        candidate_list.append({
            "filename": filename,
            "slot": slot,
            "hash": tex_hash,
            "format": format_name,
            "width": width,
            "height": height,
            "call_id": "",
        })

    return candidate_list


def list_candidate_textures_by_call(type_folder: str) -> dict[str, list[dict]]:
    '''按绘制调用分组列出 TYPE_ 提取目录内的候选贴图。

    数据来源为 TextureSlots.json v2 的 "calls" 分组 (每个绘制调用自己的完整
    ps-t 绑定集合)；v1 / 无 json / "calls" 缺失时退化为单个伪调用 "" ，
    其值为 list_candidate_textures 的扁平列表。

    Returns:
        dict[str, list[dict]]: {call_id: [{"filename": str, "slot": str, "hash": str,
                                           "format": str, "width": int, "height": int,
                                           "call_id": str}, ...]}
        调用 id 升序，组内槽位号升序；完全没有候选贴图时返回空 dict。
    '''
    if not type_folder or not os.path.isdir(type_folder):
        return {}

    raw_json = _load_texture_slots_json(type_folder)
    call_map_raw = raw_json.get("calls") if _is_v2_texture_slots(raw_json) else None

    call_map: dict[str, list[dict]] = {}
    if isinstance(call_map_raw, dict):
        for call_key in sorted(call_map_raw.keys(), key=str):
            entry_list = call_map_raw[call_key]
            if not isinstance(entry_list, list):
                continue
            candidate_list: list[dict] = []
            for entry in entry_list:
                if not isinstance(entry, dict):
                    continue
                filename = str(entry.get("filename", "") or "")
                if not filename:
                    continue
                file_path = os.path.join(type_folder, filename)
                if not os.path.isfile(file_path):
                    continue
                width, height = _resolve_entry_dimensions(entry, file_path)
                candidate_list.append({
                    "filename": filename,
                    "slot": str(entry.get("slot", "") or ""),
                    "hash": str(entry.get("hash", "") or "").lower(),
                    "format": str(entry.get("format", "") or ""),
                    "width": width,
                    "height": height,
                    "call_id": str(call_key),
                })
            if candidate_list:
                candidate_list.sort(key=lambda candidate: _slot_sort_key(candidate["slot"]))
                call_map[str(call_key)] = candidate_list

    if call_map:
        return call_map

    # v1 / 无 json / "calls" 为空: 退化为单个伪调用 ""
    flat_list = list_candidate_textures(type_folder)
    if flat_list:
        return {"": flat_list}
    return {}


# ----------------------------------------------------------------------
# SSMT4 风格命名副本
# ----------------------------------------------------------------------

def sanitize_mark_name(mark_name: str) -> str:
    '''将 MarkName 清洗为仅含 [A-Za-z0-9_] 的单令牌 (供命名副本文件名使用)。

    非法字符段收敛为单个下划线并去掉首尾下划线；清洗结果为空时抛出 TextureMarkError。
    '''
    sanitized = _MARK_NAME_INVALID_CHAR_PATTERN.sub("_", (mark_name or "").strip()).strip("_")
    if not sanitized:
        raise TextureMarkError(
            "标记名称 '" + str(mark_name) + "' 清洗后为空，"
            + "请使用仅含字母/数字/下划线的标记名称"
        )
    return sanitized


def is_named_mark_copy_filename(filename: str) -> bool:
    '''判断文件名是否为 SSMT4 风格命名副本 "<unique_str>-<MarkName>.<后缀>"。'''
    stem = os.path.splitext(filename or "")[0]
    return bool(_NAMED_MARK_COPY_STEM_PATTERN.fullmatch(stem))


def make_named_mark_copy(type_folder: str, source_filename: str, unique_str: str,
                         mark_name: str, source_folder: str | None = None) -> str:
    '''在 TYPE_ 提取目录内生成 SSMT4 风格命名副本并返回新文件名。

    将 <source_folder 或 type_folder>/<source_filename> 复制为
    <type_folder>/<unique_str>-<清洗后 MarkName><源文件后缀>，
    例如 "77ea19b6-15198-0-DiffuseMap.dds"。

    - 目标文件已存在时直接覆盖 (重新标记即替换副本)
    - 源路径与目标路径相同时不做任何操作
    - source_folder 用于多目标写入: 候选源文件可能仅存在于来源子网格的提取目录，
      此时从 source_folder 复制到目标子网格自己的 type_folder

    Returns:
        新文件名 "<unique_str>-<MarkName>.<后缀>"

    Raises:
        TextureMarkError: 参数非法 / 源文件缺失 / 复制失败时抛出 (中文提示)
    '''
    if not type_folder or not os.path.isdir(type_folder):
        raise TextureMarkError("贴图提取目录不存在: " + str(type_folder))
    if not source_filename:
        raise TextureMarkError("源贴图文件名为空，无法生成标记副本")
    if not unique_str or not unique_str.strip():
        raise TextureMarkError("unique_str 为空，无法生成标记副本文件名")

    source_path = os.path.join(source_folder or type_folder, source_filename)
    if not os.path.isfile(source_path):
        raise TextureMarkError("源贴图文件不存在: " + source_path)

    source_ext = os.path.splitext(source_filename)[1]
    target_filename = unique_str.strip() + "-" + sanitize_mark_name(mark_name) + source_ext
    target_path = os.path.join(type_folder, target_filename)

    if os.path.normcase(os.path.abspath(source_path)) == os.path.normcase(os.path.abspath(target_path)):
        return target_filename

    try:
        shutil.copyfile(source_path, target_path)
    except OSError as ex:
        raise TextureMarkError(
            "生成标记副本失败: " + source_path + " -> " + target_path + " (" + str(ex) + ")"
        )

    return target_filename


# ----------------------------------------------------------------------
# 标记读写
# ----------------------------------------------------------------------

def _load_json_dict(json_path: str) -> dict:
    if not os.path.isfile(json_path):
        raise TextureMarkError("SubmeshJson 文件不存在: " + json_path)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
    except (OSError, ValueError) as ex:
        raise TextureMarkError("SubmeshJson 读取失败: " + json_path + " (" + str(ex) + ")")

    if not isinstance(json_dict, dict):
        raise TextureMarkError("SubmeshJson 内容不是合法的 JSON 对象: " + json_path)
    return json_dict


def _atomic_write_json(json_path: str, json_dict: dict) -> None:
    '''原子写入 JSON: 先写同目录临时文件再 os.replace 覆盖，避免写坏原文件。'''
    dir_path = os.path.dirname(json_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(json_path) + ".", suffix=".tmp", dir=dir_path)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, json_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def read_marks(json_path: str) -> list[dict]:
    '''读取 SubmeshJson 中的 TextureMarkUpInfoList (仅返回 dict 条目的拷贝)。'''
    json_dict = _load_json_dict(json_path)
    mark_list = json_dict.get(TEXTURE_MARK_LIST_KEY, [])
    if not isinstance(mark_list, list):
        return []
    return [dict(mark) for mark in mark_list if isinstance(mark, dict)]


def find_mark_for_filename(mark_list: list[dict], filename: str) -> dict | None:
    '''在标记列表中按 MarkFileName 查找对应标记。'''
    if not filename:
        return None
    for mark in mark_list:
        if isinstance(mark, dict) and mark.get("MarkFileName", "") == filename:
            return mark
    return None


def find_mark_for_candidate(mark_list: list[dict], filename: str, tex_hash: str) -> dict | None:
    '''按候选贴图 (文件名 + hash) 查找对应标记。

    MarkFileName 为 SSMT4 风格命名副本后不再等于候选文件名，匹配优先级:
    1. MarkHash == 候选 hash (双方非空)
    2. SourceFileName == 候选文件名 (新版标记记录的原始候选文件名)
    3. MarkFileName == 候选文件名 (兼容旧版标记)
    '''
    normalized_hash = (tex_hash or "").strip().lower()
    if normalized_hash:
        for mark in mark_list:
            if not isinstance(mark, dict):
                continue
            mark_hash = str(mark.get("MarkHash", "") or "").strip().lower()
            if mark_hash and mark_hash == normalized_hash:
                return mark

    if filename:
        for mark in mark_list:
            if isinstance(mark, dict) and mark.get("SourceFileName", "") == filename:
                return mark

    return find_mark_for_filename(mark_list, filename)


def _mark_identity_matches(existing_mark: dict, new_mark: dict) -> bool:
    '''判断已有标记与新标记是否指向同一目标 (用于更新替换而非重复追加)。

    规则:
    - MarkFileName 相同 -> 同一贴图文件被重新标记
    - 同为 Slot 方式且 MarkSlot 相同 -> 同一插槽只保留一条标记
    - 同为 Hash 方式且 MarkHash 相同 -> 同一 hash 只保留一条标记
    '''
    new_filename = new_mark.get("MarkFileName", "")
    if new_filename and existing_mark.get("MarkFileName", "") == new_filename:
        return True

    new_mark_type = new_mark.get("MarkType", "")
    if new_mark_type and existing_mark.get("MarkType", "") == new_mark_type:
        if new_mark_type == "Slot":
            new_slot = new_mark.get("MarkSlot", "")
            return bool(new_slot) and existing_mark.get("MarkSlot", "") == new_slot
        if new_mark_type == "Hash":
            new_hash = new_mark.get("MarkHash", "")
            return bool(new_hash) and existing_mark.get("MarkHash", "") == new_hash

    return False


def write_mark(json_path: str, mark: dict) -> None:
    '''写入 (更新或追加) 一条贴图标记到 SubmeshJson 的 TextureMarkUpInfoList。

    保留 JSON 中的其他所有键，utf-8 + ensure_ascii=False + indent=4 原子写入。
    '''
    if not isinstance(mark, dict) or not mark.get("MarkName", ""):
        raise TextureMarkError("标记条目缺少 MarkName，无法写入")

    json_dict = _load_json_dict(json_path)
    mark_list = json_dict.get(TEXTURE_MARK_LIST_KEY, [])
    if not isinstance(mark_list, list):
        mark_list = []

    new_mark_list = [
        existing_mark for existing_mark in mark_list
        if not (isinstance(existing_mark, dict) and _mark_identity_matches(existing_mark, mark))
    ]
    new_mark_list.append(dict(mark))

    json_dict[TEXTURE_MARK_LIST_KEY] = new_mark_list
    _atomic_write_json(json_path, json_dict)


def remove_mark(json_path: str, mark_identity: dict) -> int:
    '''从 SubmeshJson 移除匹配的标记条目。

    匹配优先级: MarkFileName > MarkSlot > MarkHash (取 mark_identity 中第一个非空字段)。

    Returns:
        移除的条目数量
    '''
    identity_filename = (mark_identity or {}).get("MarkFileName", "")
    identity_slot = (mark_identity or {}).get("MarkSlot", "")
    identity_hash = (mark_identity or {}).get("MarkHash", "")

    def _matches(existing_mark) -> bool:
        if not isinstance(existing_mark, dict):
            return False
        if identity_filename:
            return existing_mark.get("MarkFileName", "") == identity_filename
        if identity_slot:
            return existing_mark.get("MarkSlot", "") == identity_slot
        if identity_hash:
            return existing_mark.get("MarkHash", "") == identity_hash
        return False

    json_dict = _load_json_dict(json_path)
    mark_list = json_dict.get(TEXTURE_MARK_LIST_KEY, [])
    if not isinstance(mark_list, list):
        return 0

    kept_mark_list = [existing_mark for existing_mark in mark_list if not _matches(existing_mark)]
    removed_count = len(mark_list) - len(kept_mark_list)
    if removed_count > 0:
        json_dict[TEXTURE_MARK_LIST_KEY] = kept_mark_list
        _atomic_write_json(json_path, json_dict)

    return removed_count
