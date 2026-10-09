# -*- coding: utf-8 -*-
'''
LoyalTools 绝区零 (ZZMI) DrawIB 提取核心

从 3dmigoto 帧分析 Dump 中按 DrawIB (IndexBuffer hash) 提取绝区零模型，
生成与终末地普通提取相同契约的 SSMT 风格工作空间:

    <workspace>/<DrawIB>-<IndexCount>-<FirstIndex>/TYPE_GPU-ZZMI/
        <unique_str>.json           SubmeshJson (GamePreset = ZZMI)
        <unique_str>-Index.ib       重定基后的索引 (R32_UINT 小端裸数据)
        <unique_str>-Position.buf   未蒙皮的 POSITION / NORMAL / TANGENT
        <unique_str>-Texcoord.buf   COLOR / TEXCOORD*
        <unique_str>-Blend.buf      BLENDWEIGHTS / BLENDINDICES
        t-<hash>-<FORMAT>.dds|.jpg  候选贴图 (文件名不含槽位号)
        TextureSlots.json           贴图槽位信息 v2 (slots + calls)
        <unique_str>-DiffuseMap.dds 启发式选择的漫反射贴图 (供自动材质导入)
    <workspace>/Import.json         {unique_str: gametype_name}
    <workspace>/Config.json         [{"DrawIB": ..., "Alias": ...}]

绝区零的角色网格每帧先做一次流输出 (Stream Output) 预蒙皮:
    Draw(VertexCount, 0)  pointlist  vb0=Position vb1=Texcoord vb2=Blend  SO0=DrawVB
之后的 DrawIndexed 使用 vb0=DrawVB (已蒙皮) vb1=Texcoord vb2=Blend vb3=DrawVB。
已蒙皮的 DrawVB 是当前姿势，不能当模型用；提取必须回到预蒙皮绘制取 vb0，
两次绘制通过 "SO 目标 hash == DrawIndexed 的 vb0 hash" 关联。
没有预蒙皮绘制的 DrawIB (静态网格) 直接取 DrawIndexed 绑定的顶点缓冲。

采集思路参考 gui_collect (Petrascyll, GPL-3.0) 与 XXMI-Tools 的 hash.json 约定
(draw_vb / position_vb / texcoord_vb / blend_vb / ib)，这些 hash 写入 SubmeshJson
的 CategoryHash / VertexLimitVB 和 ZZMIFrameExtract，供导出端生成 ini。

本模块不依赖 bpy，也不依赖 vendored 的 EFMI 帧模型，可在 Blender 外独立运行:
    py extract/zzmi_dump_extractor.py <dump目录> [ib_hash ...] [--workspace <输出目录>]
'''

import json
import os
import re
import shutil
import struct
import sys

from dataclasses import dataclass, field

import numpy


GAME_PRESET = "ZZMI"
DEFAULT_GAMETYPE = "GPU-ZZMI"
# SubmeshJson 中标识"由本提取器生成"的元数据键，导出端据此选择 XXMI 风格的 ini 写法
FRAME_EXTRACT_KEY = "ZZMIFrameExtract"
FRAME_EXTRACT_VERSION = 1

_TEXTURE_SUFFIXES = ('.dds', '.jpg')

# 导入/导出链路 (mesh_create_helper / obj_buffer_helper) 唯一支持的 TEXCOORD 格式
_VALID_TEXCOORD_FORMATS = ('R16G16_FLOAT', 'R32G32_FLOAT')

# D3D11_PRIMITIVE_TOPOLOGY
_TOPOLOGY_POINTLIST = 1
_TOPOLOGY_TRIANGLELIST = 4

# DXGI_FORMAT 枚举值 (IASetIndexBuffer 的 Format 参数是数字)
_IB_FORMAT_BY_ENUM = {57: "R16_UINT", 42: "R32_UINT"}
_IB_NUMPY_TYPES = {"R16_UINT": numpy.uint16, "R32_UINT": numpy.uint32}

# DDS 头里的 DXGI_FORMAT 枚举 -> 名称 (只列角色贴图会出现的格式，其余显示为 DXGI_<n>)
_DXGI_FORMAT_NAMES = {
    2: "R32G32B32A32_FLOAT", 10: "R16G16B16A16_FLOAT", 11: "R16G16B16A16_UNORM",
    24: "R10G10B10A2_UNORM", 26: "R11G11B10_FLOAT", 28: "R8G8B8A8_UNORM",
    29: "R8G8B8A8_UNORM_SRGB", 54: "R16_FLOAT", 56: "R16_UNORM", 61: "R8_UNORM",
    71: "BC1_UNORM", 72: "BC1_UNORM_SRGB", 74: "BC2_UNORM", 75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM", 78: "BC3_UNORM_SRGB", 80: "BC4_UNORM", 83: "BC5_UNORM",
    87: "B8G8R8A8_UNORM", 91: "B8G8R8A8_UNORM_SRGB", 95: "BC6H_UF16", 96: "BC6H_SF16",
    98: "BC7_UNORM", 99: "BC7_UNORM_SRGB",
}
_FOURCC_FORMAT_NAMES = {
    b"DXT1": "BC1_UNORM", b"DXT3": "BC2_UNORM", b"DXT5": "BC3_UNORM",
    b"ATI1": "BC4_UNORM", b"BC4U": "BC4_UNORM", b"ATI2": "BC5_UNORM", b"BC5U": "BC5_UNORM",
}

# 帧分析文件名: <调用号>-<资源>=[!污染标记!=]<hash>[-vs=..][-ps=..].<后缀>
_DUMP_FILE_RE = re.compile(
    r'^(?P<id>\d{6})-(?P<res>[a-z]+(?:-[a-z]+)?\d*|oD)=(?:![A-Za-z]+!=)?(?P<hash>[0-9a-f]{8})'
    r'(?P<shaders>(?:-[a-z]{2}=[0-9a-f]{16})*)\.(?P<ext>[A-Za-z0-9]+)$'
)
_SHADER_TOKEN_RE = re.compile(r'-([a-z]{2})=([0-9a-f]{16})')
_LOG_CALL_RE = re.compile(r'^(\d{6}) (\w+)\((.*)\)(?: hash=([0-9a-f]+))?\s*$')
# 槽位行。OMSetRenderTargets 的深度目标写作 "D:"
_LOG_SLOT_RE = re.compile(r'^\s+(\d+|D): (?:view=0x[0-9A-Fa-f]+ )?resource=0x[0-9A-Fa-f]+ hash=([0-9a-f]+)')
_LOG_ARG_RE = re.compile(r'(\w+):([^,]+)')
_LOG_TEXTURE_RE = re.compile(r'^\d{6} 3DMigoto Dumping Texture2D (.+?) -> (.+?)\s*$')
_DRAW_CALL_NAMES = ("Draw", "DrawInstanced", "DrawIndexed", "DrawIndexedInstanced")
_FORMAT_COMPONENT_RE = re.compile(r'[RGBAXD](\d+)')
_SUPPORTED_ELEMENT_FORMAT_RE = re.compile(
    r'^(?:(?:[RGBAD]8)+|(?:[RGBAD]16)+|(?:[RGBAD]32)+)_(?:FLOAT|UINT|SINT|UNORM|SNORM)$'
)


class ExtractError(ValueError):
    '''提取过程中的可预期错误 (带中文提示)'''
    pass


@dataclass
class DrawIBSummary:
    ib_hash: str
    draw_call_count: int
    total_index_count: int
    has_blend: bool
    texture_count: int


@dataclass
class ExtractResult:
    unique_strs: list
    json_paths: list
    warnings: list
    workspace_folder: str


@dataclass
class _Draw:
    '''log.txt 中的一次绘制及其绑定状态快照'''
    draw_id: str
    kind: str
    indexed: bool
    index_count: int = 0
    first_index: int = 0
    base_vertex: int = 0
    vertex_count: int = 0
    topology: int = 0
    vb: dict = field(default_factory=dict)      # {slot: hash}
    ib: str = ""
    ib_format: str = ""
    ib_offset: int = 0
    so: dict = field(default_factory=dict)      # {slot: hash}
    vs: str = ""
    ps: str = ""


@dataclass
class _Element:
    semantic_name: str
    semantic_index: int
    format: str
    byte_width: int

    @property
    def name(self) -> str:
        if self.semantic_index == 0:
            return self.semantic_name
        return self.semantic_name + str(self.semantic_index)


@dataclass
class _Category:
    '''一个顶点缓冲槽位的完整数据 (整个部件，未按 Part 切片)'''
    name: str
    slot: int
    vb_hash: str
    stride: int
    elements: list
    rows: numpy.ndarray = None      # shape (vertex_count, stride) uint8


@dataclass
class _TextureEntry:
    slot_id: int
    tex_hash: str
    format_name: str
    draw_id: str
    src_path: str
    width: int = 0
    height: int = 0
    filename: str = ""


@dataclass
class _Part:
    unique_str: str
    first_index: int
    index_count: int
    draws: list
    rebased_indices: numpy.ndarray = None
    used_vertices: numpy.ndarray = None
    texture_entries: list = field(default_factory=list)        # 按 hash 去重
    call_texture_entries: list = field(default_factory=list)   # 逐绘制完整绑定


@dataclass
class _Component:
    ib_hash: str
    draw_vb: str
    vertex_count: int
    categories: list
    parts: list
    pose_draw_id: str = ""
    root_vs: str = ""

    def category_hash(self, name: str) -> str:
        for category in self.categories:
            if category.name == name:
                return category.vb_hash
        return ""


def _format_byte_width(format_name: str) -> int:
    components = _FORMAT_COMPONENT_RE.findall(str(format_name).split("_", 1)[0])
    return sum(int(bits) for bits in components) // 8


def _read_txt_header(path: str):
    '''
    读取 3dmigoto vb/ib .txt 的头部 (到顶点/索引数据之前为止)。
    返回 (header 字典, element 字典列表)。
    '''
    header = {}
    elements = []
    current = None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("vertex-data:"):
                break
            if line.strip() == "":
                if header and not elements and "stride" not in header:
                    # ib .txt: 头部之后空一行就是索引数据
                    break
                continue
            if line.startswith("element["):
                current = {}
                elements.append(current)
                continue
            key, separator, value = line.partition(":")
            if not separator:
                # 不是 "键: 值" 形式，说明已经进入数据区
                break
            if line[:1].isspace() and current is not None:
                current[key.strip()] = value.strip()
            else:
                current = None
                header[key.strip()] = value.strip()
    return header, elements


def _read_image_info(path: str):
    '''读取 dds / jpg 头部，返回 (宽, 高, 格式名或空)。失败返回 (0, 0, "")'''
    try:
        suffix = os.path.splitext(path)[1].lower()
        with open(path, 'rb') as f:
            if suffix == '.dds':
                head = f.read(148)
                if len(head) < 128 or head[:4] != b"DDS ":
                    return 0, 0, ""
                height, width = struct.unpack_from("<II", head, 12)
                fourcc = head[84:88]
                format_name = ""
                if fourcc == b"DX10" and len(head) >= 132:
                    dxgi = struct.unpack_from("<I", head, 128)[0]
                    format_name = _DXGI_FORMAT_NAMES.get(dxgi, "DXGI_" + str(dxgi))
                elif fourcc in _FOURCC_FORMAT_NAMES:
                    format_name = _FOURCC_FORMAT_NAMES[fourcc]
                return int(width), int(height), format_name
            if suffix == '.jpg':
                if f.read(2) != b"\xff\xd8":
                    return 0, 0, ""
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return 0, 0, ""
                    if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                        continue
                    length_bytes = f.read(2)
                    if len(length_bytes) < 2:
                        return 0, 0, ""
                    length = struct.unpack(">H", length_bytes)[0]
                    if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                        frame = f.read(5)
                        height, width = struct.unpack(">HH", frame[1:5])
                        return int(width), int(height), ""
                    f.seek(length - 2, os.SEEK_CUR)
    except (OSError, struct.error):
        pass
    return 0, 0, ""


class ZZMIDumpExtractor:
    '''
    从绝区零帧分析 Dump 中按 DrawIB 提取模型并生成 SSMT 风格工作空间。

    公开接口 (UI 层依赖，签名与 DumpWorkspaceExtractor 的普通提取一致):
        __init__(dump_folder, verbose=False)
        list_draw_ibs() -> list[DrawIBSummary]
        list_skinned_draw_ibs() -> list[str]      "自动提取" 用: 整帧所有带骨骼权重的 IB
        extract(ib_hashes, workspace_folder, gametype_name='GPU-ZZMI', copy_textures=True,
                aliases=None) -> ExtractResult
    '''

    def __init__(self, dump_folder: str, verbose: bool = False):
        self.dump_folder = str(dump_folder)
        self.verbose = bool(verbose)

        if not os.path.isdir(self.dump_folder):
            raise ExtractError("无效的帧分析 Dump 目录 (目录不存在): " + self.dump_folder)
        if not os.path.isfile(os.path.join(self.dump_folder, "log.txt")):
            raise ExtractError(
                "无效的帧分析 Dump 目录: 未找到 log.txt 文件。\n"
                "请确认选择的是 3dmigoto 帧分析生成的 FrameAnalysis 目录: " + self.dump_folder
            )

        self._draws = None                 # list[_Draw]
        self._files = None                 # {draw_id: {resource: {ext: 文件名}}}
        self._file_hashes = None           # {draw_id: {resource: hash}}
        self._render_target_hashes = None  # set[str]
        self._texture_formats = {}         # {帧分析文件名: 格式名}
        self._header_cache = {}
        self._image_info_cache = {}
        self._component_cache = {}
        self._lossy_mark_names = set()     # 本次提取中因为只有 jpg 而没有自动标记的贴图名

    def _log(self, message: str):
        if self.verbose:
            print("[ZZMI提取] " + message)

    # ------------------------------------------------------------------
    # 帧分析目录索引与 log.txt 状态重放
    # ------------------------------------------------------------------

    def _index_dump_files(self):
        if self._files is not None:
            return
        files = {}
        file_hashes = {}
        render_target_hashes = set()
        with os.scandir(self.dump_folder) as iterator:
            for entry in iterator:
                match = _DUMP_FILE_RE.match(entry.name)
                if match is None:
                    continue
                draw_id = match.group("id")
                resource = match.group("res")
                extension = match.group("ext").lower()
                files.setdefault(draw_id, {}).setdefault(resource, {})[extension] = entry.name
                file_hashes.setdefault(draw_id, {})[resource] = match.group("hash")
                if resource.startswith("o") and (resource[1:].isdigit() or resource == "oD"):
                    render_target_hashes.add(match.group("hash"))
        self._files = files
        self._file_hashes = file_hashes
        self._render_target_hashes = render_target_hashes

    def _get_draws(self) -> list:
        '''
        顺序重放 log.txt 的 IA / SO / Shader 绑定，在每次绘制处留下状态快照。
        帧分析开始之前绑定的资源不在日志里，所以再用实际 dump 出的文件名校正 vb/ib:
        文件名记录的是绘制当时真正绑定的资源。DrawIndexed 的 vb2 (权重缓冲) 不在
        输入布局里，3dmigoto 不会 dump 它，这部分只能来自日志。
        '''
        if self._draws is not None:
            return self._draws
        self._index_dump_files()

        vb_state = {}
        so_state = {}
        ib_state = ["", "", 0]   # hash, format, offset
        vs_state = ""
        ps_state = ""
        topology_state = 0
        draws = []

        with open(os.path.join(self.dump_folder, "log.txt"), 'r', encoding='utf-8', errors='replace') as f:
            lines = f.read().splitlines()

        line_index = 0
        line_count = len(lines)
        while line_index < line_count:
            line = lines[line_index]
            line_index += 1
            if len(line) < 8 or not line[:6].isdigit():
                continue
            if line.startswith("3DMigoto", 7):
                texture_match = _LOG_TEXTURE_RE.match(line)
                if texture_match is not None:
                    deduped_stem = os.path.splitext(os.path.basename(texture_match.group(2)))[0]
                    if "-" in deduped_stem:
                        self._texture_formats[os.path.basename(texture_match.group(1))] = (
                            deduped_stem.split("-", 1)[1]
                        )
                continue
            call_match = _LOG_CALL_RE.match(line)
            if call_match is None:
                continue
            draw_id, call_name, args_text, trailing_hash = call_match.groups()

            slots = {}
            while line_index < line_count:
                slot_match = _LOG_SLOT_RE.match(lines[line_index])
                if slot_match is None:
                    break
                slot_key = slot_match.group(1)
                slots[int(slot_key) if slot_key.isdigit() else slot_key] = slot_match.group(2)
                line_index += 1

            if call_name == "OMSetRenderTargets":
                # 没有 dump_rt 时目录里没有 o* 文件，渲染目标只能从日志得知
                self._render_target_hashes.update(slots.values())
            elif call_name == "IASetVertexBuffers":
                args = dict(_LOG_ARG_RE.findall(args_text))
                start_slot = int(args.get("StartSlot", "0"))
                for slot in range(start_slot, start_slot + int(args.get("NumBuffers", "0"))):
                    if slot in slots:
                        vb_state[slot] = slots[slot]
                    else:
                        vb_state.pop(slot, None)
            elif call_name == "IASetIndexBuffer":
                args = dict(_LOG_ARG_RE.findall(args_text))
                ib_state[0] = trailing_hash or ""
                ib_state[1] = _IB_FORMAT_BY_ENUM.get(self._to_int(args.get("Format")), "")
                ib_state[2] = self._to_int(args.get("Offset"))
            elif call_name == "SOSetTargets":
                so_state = dict(slots)
            elif call_name == "VSSetShader":
                vs_state = trailing_hash or ""
            elif call_name == "PSSetShader":
                ps_state = trailing_hash or ""
            elif call_name == "IASetPrimitiveTopology":
                args = dict(_LOG_ARG_RE.findall(args_text))
                topology_state = self._to_int(args.get("Topology"))
            elif call_name in _DRAW_CALL_NAMES:
                args = dict(_LOG_ARG_RE.findall(args_text))
                indexed = call_name.startswith("DrawIndexed")
                draw = _Draw(
                    draw_id=draw_id,
                    kind=call_name,
                    indexed=indexed,
                    topology=topology_state,
                    vb=dict(vb_state),
                    ib=ib_state[0],
                    ib_format=ib_state[1],
                    ib_offset=ib_state[2],
                    so=dict(so_state),
                    vs=vs_state,
                    ps=ps_state,
                )
                if indexed:
                    draw.index_count = self._to_int(args.get("IndexCount", args.get("IndexCountPerInstance")))
                    draw.first_index = self._to_int(args.get("StartIndexLocation"))
                    draw.base_vertex = self._to_int(args.get("BaseVertexLocation"))
                else:
                    draw.vertex_count = self._to_int(args.get("VertexCount", args.get("VertexCountPerInstance")))
                self._apply_dumped_bindings(draw)
                draws.append(draw)

        self._draws = draws
        return draws

    @staticmethod
    def _to_int(value) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return 0

    def _apply_dumped_bindings(self, draw: _Draw):
        for resource, resource_hash in self._file_hashes.get(draw.draw_id, {}).items():
            if resource == "ib":
                if draw.indexed:
                    draw.ib = resource_hash
            elif resource.startswith("vb") and resource[2:].isdigit():
                draw.vb[int(resource[2:])] = resource_hash

    def _dump_file(self, draw_id: str, resource: str, extension: str) -> str:
        name = self._files.get(draw_id, {}).get(resource, {}).get(extension)
        if not name:
            return ""
        return os.path.join(self.dump_folder, name)

    def _get_header(self, path: str):
        cached = self._header_cache.get(path)
        if cached is None:
            cached = _read_txt_header(path)
            self._header_cache[path] = cached
        return cached

    # ------------------------------------------------------------------
    # DrawIB 列表
    # ------------------------------------------------------------------

    def _indexed_draws_by_ib(self) -> dict:
        grouped = {}
        for draw in self._get_draws():
            if draw.indexed and draw.ib and draw.index_count > 0:
                grouped.setdefault(draw.ib, []).append(draw)
        return grouped

    def list_draw_ibs(self) -> list:
        '''按 IB hash 聚合所有索引绘制；有预蒙皮 (带骨骼权重) 的排在前面，其次按索引总数降序'''
        summaries = []
        for ib_hash, draws in self._indexed_draws_by_ib().items():
            ranges = {(draw.first_index, draw.index_count) for draw in draws}
            texture_hashes = set()
            for draw in draws:
                for entry in self._collect_draw_textures(draw):
                    texture_hashes.add(entry.tex_hash)
            summaries.append(DrawIBSummary(
                ib_hash=ib_hash,
                draw_call_count=len(draws),
                total_index_count=sum(count for _, count in ranges),
                has_blend=self._find_pose_draw(self._pick_main_draws(draws)[0]) is not None,
                texture_count=len(texture_hashes),
            ))
        summaries.sort(key=lambda item: (not item.has_blend, -item.total_index_count, item.ib_hash))
        return summaries

    def list_skinned_draw_ibs(self) -> list:
        '''
        本帧所有经过流输出预蒙皮的 IB hash (即带骨骼权重的角色网格)，按索引总数降序。
        "自动提取" 用它代替手填的 DrawIB 列表；画面里有几个角色就会列出几个角色的部件。
        '''
        skinned = []
        for ib_hash, draws in self._indexed_draws_by_ib().items():
            if self._find_pose_draw(self._pick_main_draws(draws)[0]) is None:
                continue
            ranges = {(draw.first_index, draw.index_count) for draw in draws}
            skinned.append((-sum(count for _, count in ranges), ib_hash))
        return [ib_hash for _, ib_hash in sorted(skinned)]

    # ------------------------------------------------------------------
    # 提取入口
    # ------------------------------------------------------------------

    def extract(
        self,
        ib_hashes,
        workspace_folder: str,
        gametype_name: str = DEFAULT_GAMETYPE,
        copy_textures: bool = True,
        aliases: dict | None = None,
    ) -> ExtractResult:
        workspace_folder = str(workspace_folder)
        if not workspace_folder:
            raise ExtractError("未指定工作空间目录。")
        os.makedirs(workspace_folder, exist_ok=True)
        self._lossy_mark_names = set()

        normalized_hashes = []
        for ib_hash in ib_hashes or []:
            ib_hash = str(ib_hash).strip().lower()
            if ib_hash and ib_hash not in normalized_hashes:
                normalized_hashes.append(ib_hash)
        if not normalized_hashes:
            raise ExtractError("没有要提取的 DrawIB。")

        draws_by_ib = self._indexed_draws_by_ib()
        warnings = []
        unique_strs = []
        json_paths = []
        extracted_draw_ibs = []

        for ib_hash in normalized_hashes:
            draws = draws_by_ib.get(ib_hash)
            if not draws:
                warnings.append(
                    "DrawIB " + ib_hash + " 在帧分析中没有任何 DrawIndexed 绘制，"
                    "请确认 hash 填的是 IB (按小键盘 7/8 选中、9 复制的 IndexBuffer hash) 且角色在画面内。"
                )
                continue
            try:
                component = self._build_component(ib_hash, draws, warnings)
            except ExtractError as error:
                warnings.append("DrawIB " + ib_hash + " 提取失败: " + str(error))
                continue

            for part in component.parts:
                json_path = self._write_part(
                    component, part, workspace_folder, gametype_name, copy_textures, warnings,
                )
                unique_strs.append(part.unique_str)
                json_paths.append(json_path)
            extracted_draw_ibs.append(ib_hash)

        if unique_strs:
            self._update_workspace_root_files(
                workspace_folder, unique_strs, gametype_name, extracted_draw_ibs, aliases,
            )
        if self._lossy_mark_names:
            warnings.append(
                "/".join(sorted(self._lossy_mark_names)) + " 在这份帧分析里是 jpg (有损)，没有自动标记，"
                "游戏会继续用原贴图。要拿到无损的 dds: 在 d3dx.ini 的 analyse_options 里加上 dds 后重新按 F8 抓帧。"
            )

        return ExtractResult(
            unique_strs=unique_strs,
            json_paths=json_paths,
            warnings=warnings,
            workspace_folder=workspace_folder,
        )

    # ------------------------------------------------------------------
    # 部件构建
    # ------------------------------------------------------------------

    def _pick_main_draws(self, draws: list) -> list:
        '''
        同一 IB 的绘制按 vb0 分组，取绘制次数最多的一组作为主绘制，
        其余 (绑定了别的顶点缓冲的同 hash IB) 不参与提取。
        只比较 vb0: 其它槽位可能是之前的绘制留在状态里、本次绘制并不使用的旧绑定。
        '''
        signature_groups = {}
        for draw in draws:
            signature_groups.setdefault(draw.vb.get(0, ""), []).append(draw)
        return max(signature_groups.values(), key=len)

    def _find_pose_draw(self, main_draw: _Draw):
        '''
        找到写出 main_draw.vb0 的流输出预蒙皮绘制。
        流输出缓冲没有初始数据，顶点数相同的网格会算出同一个 hash，
        多个候选时用共享的 Texcoord / Blend 缓冲 hash 区分。
        '''
        draw_vb = main_draw.vb.get(0, "")
        if not draw_vb:
            return None
        candidates = [
            draw for draw in self._get_draws()
            if not draw.indexed and draw.so.get(0) == draw_vb and draw.vertex_count > 0
        ]
        if not candidates:
            return None
        if len(candidates) > 1:
            shared_hashes = {
                vb_hash for slot, vb_hash in main_draw.vb.items() if slot != 0 and vb_hash != draw_vb
            }
            narrowed = [
                draw for draw in candidates
                if shared_hashes.intersection(vb_hash for slot, vb_hash in draw.vb.items() if slot != 0)
            ]
            if narrowed:
                candidates = narrowed
        return candidates[0]

    def _build_component(self, ib_hash: str, draws: list, warnings: list) -> _Component:
        cached = self._component_cache.get(ib_hash)
        if cached is not None:
            return cached

        main_draws = self._pick_main_draws(draws)
        if len(main_draws) != len(draws):
            warnings.append(
                "DrawIB " + ib_hash + " 有 " + str(len(draws) - len(main_draws))
                + " 次绘制绑定了不同的顶点缓冲，已忽略 (只提取出现次数最多的那组绑定)。"
            )
        main_draw = main_draws[0]
        draw_vb = main_draw.vb.get(0, "")
        if not draw_vb:
            raise ExtractError("绘制 " + main_draw.draw_id + " 没有绑定 vb0。")

        pose_draw = self._find_pose_draw(main_draw)
        if pose_draw is not None:
            # 预蒙皮绘制的输入布局同时描述三个槽位，优先用它的 .txt；
            # 主绘制各渲染 Pass 的 .txt 作为补充候选 (只用于 Texcoord 等共享缓冲)。
            source_draws = [pose_draw] + main_draws
            slot_hashes = dict(pose_draw.vb)
            self._log("DrawIB " + ib_hash + " 预蒙皮绘制 " + pose_draw.draw_id + " -> DrawVB " + draw_vb)
        else:
            dumped_vb_hashes = [
                resource_hash
                for resource, resource_hash in self._file_hashes.get(main_draw.draw_id, {}).items()
                if resource.startswith("vb") and resource[2:].isdigit()
            ]
            if len(dumped_vb_hashes) != len(set(dumped_vb_hashes)):
                # 蒙皮网格的 DrawIndexed 会把同一块已蒙皮缓冲绑两次 (vb0 当前位置 + vb3 上一帧位置)。
                # 出现这种绑定却找不到流输出绘制，说明预蒙皮那次绘制没被抓到，vb0 是当前姿势。
                # (不能用日志里的绑定状态判断: 没被本次绘制用到的旧槽位会一直留在状态里)
                warnings.append(
                    "DrawIB " + ib_hash + " 没有找到流输出预蒙皮绘制，按静态网格提取: "
                    "得到的是抓帧时的姿势且没有骨骼权重。请确认帧分析是完整的一帧 (log.txt 含 SOSetTargets)。"
                )
            source_draws = main_draws
            slot_hashes = {}
            seen_hashes = set()
            for slot in sorted(main_draw.vb.keys()):
                vb_hash = main_draw.vb[slot]
                # vb3 与 vb0 是同一个缓冲 (上一帧位置)，不重复提取
                if vb_hash in seen_hashes:
                    continue
                seen_hashes.add(vb_hash)
                slot_hashes[slot] = vb_hash

        categories = []
        used_names = set()
        for slot in sorted(slot_hashes.keys()):
            category = self._build_category(slot, slot_hashes[slot], source_draws, warnings, ib_hash)
            if category is None:
                continue
            if category.name in used_names:
                warnings.append(
                    "DrawIB " + ib_hash + " 的 vb" + str(slot) + " 与已有槽位同为 "
                    + category.name + " 数据，已忽略该槽位。"
                )
                continue
            used_names.add(category.name)
            categories.append(category)

        if "Position" not in used_names:
            raise ExtractError("没有找到含 POSITION 的顶点缓冲 (请确认 analyse_options 含 dump_vb buf txt)。")

        vertex_count = len(next(category for category in categories if category.name == "Position").rows)
        for category in list(categories):
            if len(category.rows) != vertex_count:
                if category.name == "Position":
                    continue
                warnings.append(
                    "DrawIB " + ib_hash + " 的 " + category.name + " 缓冲顶点数 " + str(len(category.rows))
                    + " 与 Position 的 " + str(vertex_count) + " 不一致，已忽略该缓冲。"
                )
                categories.remove(category)

        # 导出端按 Position / Texcoord / Blend 的顺序读取分类
        order = {"Position": 0, "Texcoord": 1, "Blend": 2}
        categories.sort(key=lambda category: order.get(category.name, 9))

        index_data = self._read_index_buffer(main_draws, ib_hash)

        part_groups = {}
        for draw in main_draws:
            part_groups.setdefault((draw.first_index, draw.index_count), []).append(draw)

        parts = []
        previous_end = 0
        overlap_reported = False
        for (first_index, index_count) in sorted(part_groups.keys()):
            if first_index < previous_end and not overlap_reported:
                overlap_reported = True
                warnings.append(
                    "DrawIB " + ib_hash + " 的绘制区间有重叠 (first_index=" + str(first_index)
                    + " 落在前一个区间内)，重叠的区间会各自提取成一个子网格，导入后请删掉不需要的。"
                )
            previous_end = max(previous_end, first_index + index_count)
            part_draws = part_groups[(first_index, index_count)]
            unique_str = ib_hash + "-" + str(index_count) + "-" + str(first_index)
            try:
                part = self._build_part(
                    unique_str, first_index, index_count, part_draws, index_data, vertex_count,
                )
            except ExtractError as error:
                warnings.append(unique_str + " 已跳过: " + str(error))
                continue
            parts.append(part)

        if not parts:
            raise ExtractError("没有可提取的子网格。")

        component = _Component(
            ib_hash=ib_hash,
            draw_vb=draw_vb,
            vertex_count=vertex_count,
            categories=categories,
            parts=parts,
            pose_draw_id=pose_draw.draw_id if pose_draw is not None else "",
            root_vs=pose_draw.vs if pose_draw is not None else "",
        )
        self._component_cache[ib_hash] = component
        return component

    def _build_category(self, slot: int, vb_hash: str, source_draws: list, warnings: list, ib_hash: str):
        '''读取一个槽位的整块缓冲，并从 .txt 头部推导它的真实布局'''
        resource = "vb" + str(slot)
        buf_path = ""
        txt_paths = []
        for draw in source_draws:
            if self._file_hashes.get(draw.draw_id, {}).get(resource) != vb_hash:
                continue
            if not buf_path:
                buf_path = self._dump_file(draw.draw_id, resource, "buf")
            txt_path = self._dump_file(draw.draw_id, resource, "txt")
            if txt_path:
                txt_paths.append(txt_path)

        if not buf_path or not txt_paths:
            # DrawIndexed 的 vb2 等未进入输入布局的槽位不会被 dump，静态网格遇到时直接跳过
            self._log("DrawIB " + ib_hash + " 的 " + resource + "=" + vb_hash + " 没有 buf/txt，跳过")
            return None

        stride = 0
        best_elements = None
        best_covered = -1
        for txt_path in txt_paths:
            header, raw_elements = self._get_header(txt_path)
            txt_stride = self._to_int(header.get("stride"))
            if txt_stride <= 0:
                continue
            elements, covered = self._derive_slot_elements(raw_elements, slot, txt_stride)
            if covered > best_covered:
                stride, best_elements, best_covered = txt_stride, elements, covered
            if covered == txt_stride:
                break
        if best_elements is None or best_covered <= 0:
            return None

        semantic_names = {element.semantic_name for element in best_elements}
        if "POSITION" in semantic_names:
            category_name = "Position"
        elif semantic_names.intersection(("BLENDINDICES", "BLENDWEIGHT", "BLENDWEIGHTS")):
            category_name = "Blend"
        else:
            category_name = "Texcoord"

        if best_covered < stride:
            warnings.append(
                "DrawIB " + ib_hash + " 的 " + category_name + " 缓冲步长 " + str(stride)
                + " 字节中只有 " + str(best_covered) + " 字节能识别语义，其余按占位数据处理 (导出时填 0)。"
            )

        file_size = os.path.getsize(buf_path)
        if file_size == 0 or file_size % stride != 0:
            raise ExtractError(
                os.path.basename(buf_path) + " 大小 " + str(file_size) + " 不是步长 " + str(stride) + " 的整数倍。"
            )
        rows = numpy.fromfile(buf_path, dtype=numpy.uint8).reshape(-1, stride)
        return _Category(
            name=category_name, slot=slot, vb_hash=vb_hash, stride=stride, elements=best_elements, rows=rows,
        )

    def _derive_slot_elements(self, raw_elements: list, slot: int, stride: int):
        '''
        .txt 头部列出的是整条输入布局: 其它槽位的元素，以及着色器声明了但网格没有的输入
        (3dmigoto 把它们写成 InputSlot 0 / AlignedByteOffset 0 / R8G8B8A8_UNORM)。
        属于本槽位且偏移单调递增的才是真实元素；空隙和不支持的格式用 UNKNOWN 占位，
        保证各元素 ByteWidth 之和等于步长。返回 (元素列表, 可识别字节数)。
        '''
        elements = []
        cursor = 0
        covered = 0
        unknown_count = 0

        def append_unknown(width: int):
            nonlocal unknown_count
            if width <= 0:
                return
            if width % 4 == 0:
                placeholder_format = "R32_UINT"
            elif width % 2 == 0:
                placeholder_format = "R16_UINT"
            else:
                placeholder_format = "R8_UINT"
            elements.append(_Element(
                semantic_name="UNKNOWN" + ("" if slot == 0 else "VB" + str(slot)),
                semantic_index=unknown_count,
                format=placeholder_format,
                byte_width=width,
            ))
            unknown_count += 1

        for raw in raw_elements:
            if self._to_int(raw.get("InputSlot")) != slot:
                continue
            if raw.get("InputSlotClass", "per-vertex") != "per-vertex":
                continue
            format_name = str(raw.get("Format", "")).replace("DXGI_FORMAT_", "")
            width = _format_byte_width(format_name)
            offset = self._to_int(raw.get("AlignedByteOffset"))
            if width <= 0 or offset < cursor or offset + width > stride:
                continue

            append_unknown(offset - cursor)
            semantic_name = str(raw.get("SemanticName", "")).upper()
            semantic_index = self._to_int(raw.get("SemanticIndex"))
            supported = _SUPPORTED_ELEMENT_FORMAT_RE.match(format_name) is not None
            if supported and semantic_name == "TEXCOORD" and format_name not in _VALID_TEXCOORD_FORMATS:
                supported = False
            if supported:
                elements.append(_Element(semantic_name, semantic_index, format_name, width))
                covered += width
            else:
                append_unknown(width)
            cursor = offset + width

        append_unknown(stride - cursor)
        return elements, covered

    def _read_index_buffer(self, main_draws: list, ib_hash: str) -> numpy.ndarray:
        '''读取整块索引缓冲 (帧分析的 ib .buf 是完整缓冲，各 Part 再按 first_index 切片)'''
        for draw in main_draws:
            buf_path = self._dump_file(draw.draw_id, "ib", "buf")
            if not buf_path:
                continue
            format_name = draw.ib_format
            byte_offset = draw.ib_offset
            txt_path = self._dump_file(draw.draw_id, "ib", "txt")
            if txt_path:
                header, _ = self._get_header(txt_path)
                format_name = str(header.get("format", format_name)).replace("DXGI_FORMAT_", "") or format_name
                if "byte offset" in header:
                    byte_offset = self._to_int(header.get("byte offset"))
                topology = header.get("topology", "")
                if topology and topology != "trianglelist":
                    raise ExtractError("DrawIB " + ib_hash + " 的拓扑为 " + topology + "，只支持 trianglelist。")
            numpy_type = _IB_NUMPY_TYPES.get(format_name)
            if numpy_type is None:
                raise ExtractError("DrawIB " + ib_hash + " 的索引格式无法识别: " + str(format_name))
            return numpy.fromfile(buf_path, dtype=numpy_type, offset=byte_offset)
        raise ExtractError(
            "DrawIB " + ib_hash + " 没有 dump 出 ib 的 .buf 文件 (请确认 analyse_options 含 dump_ib buf txt)。"
        )

    def _build_part(
        self,
        unique_str: str,
        first_index: int,
        index_count: int,
        part_draws: list,
        index_data: numpy.ndarray,
        vertex_count: int,
    ) -> _Part:
        if index_count % 3 != 0:
            raise ExtractError("索引数 " + str(index_count) + " 不是 3 的倍数。")
        if first_index + index_count > len(index_data):
            raise ExtractError(
                "绘制区间 [" + str(first_index) + ", " + str(first_index + index_count)
                + ") 超出索引缓冲长度 " + str(len(index_data)) + "。"
            )
        for draw in part_draws:
            if draw.topology not in (0, _TOPOLOGY_TRIANGLELIST):
                raise ExtractError("绘制 " + draw.draw_id + " 的拓扑不是 trianglelist。")
        base_vertices = {draw.base_vertex for draw in part_draws}
        if len(base_vertices) != 1:
            raise ExtractError("各渲染 Pass 的 BaseVertexLocation 不一致: " + str(sorted(base_vertices)))

        absolute = index_data[first_index:first_index + index_count].astype(numpy.int64) + base_vertices.pop()
        if absolute.min() < 0 or absolute.max() >= vertex_count:
            raise ExtractError(
                "索引范围 " + str(int(absolute.min())) + ".." + str(int(absolute.max()))
                + " 超出顶点数 " + str(vertex_count) + "。"
            )
        # 只保留本 Part 实际引用的顶点 (保持原顺序)，索引重定基到 0..n-1
        used_vertices = numpy.unique(absolute)
        rebased = numpy.searchsorted(used_vertices, absolute).astype(numpy.uint32)

        part = _Part(
            unique_str=unique_str,
            first_index=first_index,
            index_count=index_count,
            draws=part_draws,
            rebased_indices=rebased,
            used_vertices=used_vertices,
        )
        seen_hashes = set()
        for draw in part_draws:
            for entry in self._collect_draw_textures(draw):
                part.call_texture_entries.append(entry)
                if entry.tex_hash not in seen_hashes:
                    seen_hashes.add(entry.tex_hash)
                    part.texture_entries.append(entry)
        return part

    # ------------------------------------------------------------------
    # 贴图
    # ------------------------------------------------------------------

    def _collect_draw_textures(self, draw: _Draw) -> list:
        '''一次绘制 dump 出的 ps-t* 贴图；本帧的渲染目标 (深度/颜色缓冲) 不是模型贴图，排除'''
        self._index_dump_files()
        entries = []
        for resource, by_extension in self._files.get(draw.draw_id, {}).items():
            if not resource.startswith("ps-t") or not resource[4:].isdigit():
                continue
            tex_hash = self._file_hashes[draw.draw_id][resource]
            if tex_hash in self._render_target_hashes:
                continue
            # 同一槽位同时 dump 了 dds 和 jpg 时保留 dds
            filename = by_extension.get("dds") or by_extension.get("jpg")
            if not filename:
                continue
            src_path = os.path.join(self.dump_folder, filename)
            info = self._image_info_cache.get(src_path)
            if info is None:
                info = _read_image_info(src_path)
                self._image_info_cache[src_path] = info
            width, height, header_format = info
            format_name = self._texture_formats.get(filename) or header_format or "UNKNOWN"
            entries.append(_TextureEntry(
                slot_id=int(resource[4:]),
                tex_hash=tex_hash,
                format_name=format_name,
                draw_id=draw.draw_id,
                src_path=src_path,
                width=width,
                height=height,
            ))
        entries.sort(key=lambda entry: entry.slot_id)
        return entries

    @staticmethod
    def _sanitize_format_token(format_name: str) -> str:
        '''格式令牌只保留字母数字与下划线 (贴图文件名必须是单令牌，供 ini 资源名使用)'''
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(format_name or ""))
        return token or "UNKNOWN"

    def _write_textures(self, part: _Part, type_folder: str, warnings: list):
        hash_to_filename = {}
        for entry in part.texture_entries:
            entry.filename = (
                "t-" + entry.tex_hash + "-" + self._sanitize_format_token(entry.format_name)
                + os.path.splitext(entry.src_path)[1].lower()
            )
            hash_to_filename[entry.tex_hash] = entry.filename
        for entry in part.call_texture_entries:
            entry.filename = hash_to_filename.get(entry.tex_hash, "")

        for entry in part.texture_entries:
            dst_path = os.path.join(type_folder, entry.filename)
            if os.path.exists(dst_path):
                continue
            try:
                shutil.copy2(entry.src_path, dst_path)
            except OSError as error:
                warnings.append(
                    part.unique_str + " 贴图复制失败 " + os.path.basename(entry.src_path) + ": " + repr(error)
                )

        slot_map = {}
        for entry in part.texture_entries:
            slot_map.setdefault(entry.slot_id, []).append({
                "hash": entry.tex_hash,
                "filename": entry.filename,
                "format": entry.format_name,
                "call_id": int(entry.draw_id),
                "width": entry.width,
                "height": entry.height,
            })
        call_map = {}
        for entry in part.call_texture_entries:
            call_map.setdefault(entry.draw_id, []).append({
                "slot": "ps-t" + str(entry.slot_id),
                "hash": entry.tex_hash,
                "filename": entry.filename,
                "format": entry.format_name,
                "width": entry.width,
                "height": entry.height,
            })
        texture_slots_json = {
            "version": 2,
            "slots": {"ps-t" + str(slot_id): slot_map[slot_id] for slot_id in sorted(slot_map.keys())},
            "calls": {draw_id: call_map[draw_id] for draw_id in sorted(call_map.keys())},
        }
        with open(os.path.join(type_folder, "TextureSlots.json"), 'w', encoding='utf-8') as f:
            json.dump(texture_slots_json, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------
    # 自动贴图标记
    # ------------------------------------------------------------------

    @staticmethod
    def _is_square(entry: _TextureEntry, minimum: int) -> bool:
        return entry.width == entry.height and entry.width >= minimum

    @classmethod
    def _is_diffuse(cls, entry) -> bool:
        return entry is not None and entry.format_name == "BC7_UNORM_SRGB" and cls._is_square(entry, 256)

    @classmethod
    def _is_light_or_material(cls, entry) -> bool:
        if entry is None:
            return False
        if entry.format_name == "BC6H_UF16":
            return cls._is_square(entry, 256)
        return entry.format_name == "BC7_UNORM" and cls._is_square(entry, 64)

    def _resolve_auto_marks(self, part: _Part) -> dict:
        '''
        按 ZZMI SlotFix (Core/ZZMI/Libraries/SlotFix) 识别角色着色器的槽位布局，
        返回 {MarkName: _TextureEntry}。SlotFix 在运行时按同一套规则把
        Resource\\ZZMI\\Diffuse 等分发到各渲染 Pass 的实际槽位，所以这里只需认出贴图本身:
            主着色 Pass:  t3 漫反射, t4 法线, t5 光照, t6 材质
                          (t5 被占用的变体: t6 光照, t7 材质)
            脸部:         t3 漫反射, t4 法线 (R16G16B16A16_FLOAT)
            只有描边 Pass: t2 漫反射, t3 光照
        '''
        by_draw = {}
        for entry in part.call_texture_entries:
            by_draw.setdefault(entry.draw_id, {})[entry.slot_id] = entry

        for draw_id in sorted(by_draw.keys()):
            slots = by_draw[draw_id]
            diffuse, normal = slots.get(3), slots.get(4)
            if not self._is_diffuse(diffuse) or normal is None:
                continue
            if normal.format_name == "R8G8B8A8_UNORM" and self._is_square(normal, 128):
                marks = {"DiffuseMap": diffuse, "NormalMap": normal}
                occupied = slots.get(5)
                if occupied is not None and occupied.format_name in ("R8G8B8A8_UNORM_SRGB", "BC7_UNORM_SRGB"):
                    light, material = slots.get(6), slots.get(7)
                else:
                    light, material = slots.get(5), slots.get(6)
                if self._is_light_or_material(light):
                    marks["LightMap"] = light
                if self._is_light_or_material(material):
                    marks["MaterialMap"] = material
                return marks
            if normal.format_name == "R16G16B16A16_FLOAT" and self._is_square(normal, 128):
                return {"DiffuseMap": diffuse, "NormalMap": normal}

        for draw_id in sorted(by_draw.keys()):
            slots = by_draw[draw_id]
            diffuse = slots.get(2)
            if not self._is_diffuse(diffuse):
                continue
            marks = {"DiffuseMap": diffuse}
            if self._is_light_or_material(slots.get(3)):
                marks["LightMap"] = slots.get(3)
            return marks
        return {}

    def _write_auto_marks(self, part: _Part, type_folder: str, existing_marks: list, warnings: list) -> list:
        '''
        把识别出的贴图写成和"标记贴图"面板相同的标记条目 (Slot 方式 + 命名副本)，
        面板里可以照常取消或改标。已有同名标记 (上次提取或手动标记留下的) 优先保留。

        只标记 dds。帧分析的 analyse_options 没有 dds 时，未压缩贴图 (法线贴图) 被存成 jpg:
        有损、没有 alpha，脸部的 R16G16B16A16_FLOAT 法线更是存不下数值范围。
        标记这样的文件会让导出的 Mod 用劣化的贴图替换游戏原图，所以跳过并在提取结果里提示
        改用 dds 抓帧 (用户明确选择无损方案，不做 jpg 转换)。
        '''
        marks = [mark for mark in existing_marks if isinstance(mark, dict)]
        marked_names = {str(mark.get("MarkName", "")) for mark in marks}
        for mark_name, entry in self._resolve_auto_marks(part).items():
            if mark_name in marked_names or not entry.filename:
                continue
            if not entry.filename.lower().endswith(".dds"):
                self._lossy_mark_names.add(mark_name)
                continue
            mark_filename = part.unique_str + "-" + mark_name + ".dds"
            try:
                shutil.copyfile(os.path.join(type_folder, entry.filename), os.path.join(type_folder, mark_filename))
            except OSError as error:
                warnings.append(part.unique_str + " 自动标记 " + mark_name + " 失败: " + repr(error))
                continue
            marks.append({
                "MarkName": mark_name,
                "MarkType": "Slot",
                "MarkHash": entry.tex_hash,
                "MarkSlot": "ps-t" + str(entry.slot_id),
                "MarkFileName": mark_filename,
                "SourceFileName": entry.filename,
            })
        return marks

    # ------------------------------------------------------------------
    # 写盘
    # ------------------------------------------------------------------

    def _write_part(
        self,
        component: _Component,
        part: _Part,
        workspace_folder: str,
        gametype_name: str,
        copy_textures: bool,
        warnings: list,
    ) -> str:
        unique_str = part.unique_str
        type_folder = os.path.join(workspace_folder, unique_str, "TYPE_" + gametype_name)
        os.makedirs(type_folder, exist_ok=True)

        index_filename = unique_str + "-Index.ib"
        with open(os.path.join(type_folder, index_filename), 'wb') as f:
            f.write(part.rebased_indices.tobytes())

        category_buffer_list = []
        for category in component.categories:
            category_filename = unique_str + "-" + category.name + ".buf"
            with open(os.path.join(type_folder, category_filename), 'wb') as f:
                f.write(numpy.ascontiguousarray(category.rows[part.used_vertices]).tobytes())
            category_buffer_list.append({
                "FileName": category_filename,
                "Type": "Normal",
                "D3D11ElementList": [
                    {
                        "SemanticName": element.semantic_name,
                        "SemanticIndex": int(element.semantic_index),
                        "Format": element.format,
                        "ByteWidth": int(element.byte_width),
                        "ExtractSlot": "vb" + str(category.slot),
                        "ExtractTechnique": "",
                        "Category": category.name,
                    }
                    for element in category.elements
                ],
            })

        json_path = os.path.join(type_folder, unique_str + ".json")
        # 重新提取时保留已有的贴图标记 (标记生成的命名副本仍在同一目录里)
        texture_markup_info_list = []
        existing_json = self._load_json_file(json_path, {})
        if isinstance(existing_json, dict) and isinstance(existing_json.get("TextureMarkUpInfoList"), list):
            texture_markup_info_list = existing_json["TextureMarkUpInfoList"]

        if copy_textures and part.texture_entries:
            self._write_textures(part, type_folder, warnings)
            texture_markup_info_list = self._write_auto_marks(
                part, type_folder, texture_markup_info_list, warnings,
            )

        vs_hash_list = []
        for draw in part.draws:
            if draw.vs and draw.vs not in vs_hash_list:
                vs_hash_list.append(draw.vs)

        has_blend = any(category.name == "Blend" for category in component.categories)
        submesh_json_dict = {
            "GamePreset": GAME_PRESET,
            "WorkGameType": gametype_name,
            "GPU-PreSkinning": bool(has_blend and component.pose_draw_id),
            "CategoryDrawCategoryMap": {category.name: category.name for category in component.categories},
            "CategoryHash": {category.name: category.vb_hash for category in component.categories},
            "VertexLimitVB": component.draw_vb,
            "VSHashList": vs_hash_list,
            "OriginalVertexCount": int(component.vertex_count),
            "PartName": str(part.first_index),
            "TextureMarkUpInfoList": texture_markup_info_list,
            "IndexBufferList": [{"DXGI_FORMAT": "R32_UINT", "FileName": index_filename}],
            "CategoryBufferList": category_buffer_list,
            FRAME_EXTRACT_KEY: {
                "Version": FRAME_EXTRACT_VERSION,
                "IB": component.ib_hash,
                "FirstIndex": int(part.first_index),
                "IndexCount": int(part.index_count),
                "DrawVB": component.draw_vb,
                "PositionVB": component.category_hash("Position"),
                "TexcoordVB": component.category_hash("Texcoord"),
                "BlendVB": component.category_hash("Blend"),
                "PreSkinned": bool(component.pose_draw_id),
                "PoseDrawId": component.pose_draw_id,
                "RootVS": component.root_vs,
                "ComponentVertexCount": int(component.vertex_count),
            },
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(submesh_json_dict, f, ensure_ascii=False, indent=4)
        return json_path

    @staticmethod
    def _load_json_file(path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return default

    def _update_workspace_root_files(
        self,
        workspace_folder: str,
        unique_strs: list,
        gametype_name: str,
        draw_ibs: list,
        aliases: dict | None = None,
    ):
        normalized_aliases = {}
        for ib_hash, alias in (aliases or {}).items():
            if ib_hash and str(ib_hash).strip() and alias is not None and str(alias).strip():
                normalized_aliases[str(ib_hash).strip().lower()] = str(alias).strip()

        # Import.json: {unique_str: gametype_name}，保留已有条目
        import_json_path = os.path.join(workspace_folder, "Import.json")
        import_json = self._load_json_file(import_json_path, {})
        if not isinstance(import_json, dict):
            import_json = {}
        for unique_str in unique_strs:
            import_json[unique_str] = gametype_name
        with open(import_json_path, 'w', encoding='utf-8') as f:
            json.dump(import_json, f, ensure_ascii=False, indent=4)

        # Config.json: [{"DrawIB": ..., "Alias": ...}]，追加缺失的 DrawIB，显式别名覆盖旧别名
        config_json_path = os.path.join(workspace_folder, "Config.json")
        config_json = self._load_json_file(config_json_path, [])
        if not isinstance(config_json, list):
            config_json = []
        existing_entries = {
            str(item.get("DrawIB", "")).lower(): item for item in config_json if isinstance(item, dict)
        }
        for draw_ib in draw_ibs:
            alias = normalized_aliases.get(draw_ib.lower())
            existing_entry = existing_entries.get(draw_ib.lower())
            if existing_entry is not None:
                if alias and existing_entry.get("Alias") != alias:
                    existing_entry["Alias"] = alias
                continue
            new_entry = {"DrawIB": draw_ib, "Alias": alias if alias else draw_ib}
            config_json.append(new_entry)
            existing_entries[draw_ib.lower()] = new_entry
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_json, f, ensure_ascii=False, indent=4)


# ----------------------------------------------------------------------
# 独立自测入口 (Blender 外手动测试用)
# ----------------------------------------------------------------------

def _standalone_main(argv: list):
    import argparse

    parser = argparse.ArgumentParser(description="LoyalTools 绝区零 DrawIB 提取自测工具")
    parser.add_argument("dump_folder", help="3dmigoto 帧分析 Dump 目录 (含 log.txt)")
    parser.add_argument("ib_hashes", nargs='*', help="要提取的 DrawIB hash 列表，留空则只列出所有 DrawIB")
    parser.add_argument("--all-skinned", action='store_true', help="提取整帧所有带骨骼权重的 DrawIB (不用填 hash)")
    parser.add_argument("--workspace", default="", help="输出工作空间目录 (提取时必填)")
    parser.add_argument("--gametype", default=DEFAULT_GAMETYPE, help="数据类型名称 (默认 " + DEFAULT_GAMETYPE + ")")
    parser.add_argument("--no-textures", action='store_true', help="不复制贴图")
    parser.add_argument("--alias", action='append', default=[], help="DrawIB 别名，格式 ib_hash=别名 (可多次指定)")
    parser.add_argument("--verbose", action='store_true', help="输出详细日志")
    args = parser.parse_args(argv)

    aliases = {}
    for alias_pair in args.alias:
        if '=' not in alias_pair:
            print("错误: --alias 参数格式应为 ib_hash=别名，实际: " + alias_pair)
            sys.exit(2)
        ib_hash, alias = alias_pair.split('=', 1)
        aliases[ib_hash] = alias

    extractor = ZZMIDumpExtractor(args.dump_folder, verbose=args.verbose)

    summaries = extractor.list_draw_ibs()
    print("共发现 " + str(len(summaries)) + " 个 DrawIB:")
    for summary in summaries:
        print(
            "  " + summary.ib_hash
            + "  draws=" + str(summary.draw_call_count)
            + "  indices=" + str(summary.total_index_count)
            + "  blend=" + str(summary.has_blend)
            + "  textures=" + str(summary.texture_count)
        )

    ib_hashes = list(args.ib_hashes)
    if args.all_skinned:
        ib_hashes = extractor.list_skinned_draw_ibs()
        print("带骨骼权重的 DrawIB: " + (", ".join(ib_hashes) or "无"))
    if not ib_hashes:
        return

    if not args.workspace:
        print("错误: 提取时必须通过 --workspace 指定输出目录。")
        sys.exit(2)

    result = extractor.extract(
        ib_hashes=ib_hashes,
        workspace_folder=args.workspace,
        gametype_name=args.gametype,
        copy_textures=not args.no_textures,
        aliases=aliases or None,
    )

    print("提取完成，工作空间: " + result.workspace_folder)
    for json_path in result.json_paths:
        print("  " + json_path)
    for warning in result.warnings:
        print("警告: " + warning)


if __name__ == '__main__':
    _standalone_main(sys.argv[1:])
