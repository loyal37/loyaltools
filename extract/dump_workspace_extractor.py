# -*- coding: utf-8 -*-
'''
LoyalTools DrawIB 提取核心

从 3dmigoto 帧分析 Dump (frame analysis dump) 中按 DrawIB (IndexBuffer hash) 提取模型，
并合成 TheHerta4 兼容的 SSMT 风格工作空间目录:

    <workspace>/<DrawIB>-<IndexCount>-<FirstIndex>/TYPE_<GameTypeName>/
        <unique_str>.json           SubmeshJson (供 ssmt_import_helper / efmi 导出使用)
        <unique_str>-Index.ib       重定基后的索引数据 (R32_UINT 小端裸数据)
        <unique_str>-<Category>.buf 分类顶点缓冲区 (Position / Blend / Texcoord...)
        t-<hash>-<FORMAT>.dds       本次绘制引用的 ps 贴图 (单令牌文件名, 不含槽位号, 供 ini 资源名使用)
        TextureSlots.json           贴图槽位信息 v2 (slots 槽位聚合 + calls 逐绘制分组, 供贴图标记面板使用)
        <unique_str>-DiffuseMap.dds 启发式选择的漫反射贴图 (供自动材质导入)
    <workspace>/Import.json         {unique_str: gametype_name}
    <workspace>/Config.json         [{"DrawIB": ..., "Alias": ...}]

本模块不依赖 bpy，可在 Blender 外独立测试:
    py extract/dump_workspace_extractor.py <dump目录> [ib_hash ...] [--workspace <输出目录>]

数据类型解析复用 vendored 的 EFMI-Tools 代码 (../efmi_extract/migoto_io)。
'''

import copy
import json
import os
import re
import shutil
import sys

from dataclasses import dataclass, field
from pathlib import Path

import numpy

if __package__ in (None, ''):
    # 直接以脚本运行 (py extract/dump_workspace_extractor.py ...) 时相对导入不可用。
    # 这里手动构造一个合成包并把执行委托给包内的同名模块，
    # 同时避免执行 LoyalTools/__init__.py (那里会 import bpy)。
    import importlib
    import types

    _root = Path(__file__).resolve().parent.parent
    _pkg_name = "_loyaltools_standalone"
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(_root)]
        sys.modules[_pkg_name] = _pkg

    _module = importlib.import_module(_pkg_name + ".extract.dump_workspace_extractor")
    _module._standalone_main(sys.argv[1:])
    sys.exit(0)

from ..efmi_extract.migoto_io.object_extractor.object_extractor import ObjectExtractor
from ..efmi_extract.migoto_io.object_extractor.migoto_object.migoto_object_builder import (
    MigotoObjectBuilder,
    MigotoObjectFilter,
)
from ..efmi_extract.migoto_io.object_extractor.raw_object.raw_object_extractor import (
    DrawCallFilter,
    RawObjectFilter,
)
from ..efmi_extract.migoto_io.data_model.byte_buffer import (
    BufferLayout,
    BufferSemantic,
    AbstractSemantic,
    Semantic,
)
from ..efmi_extract.migoto_io.data_model.dxgi_format import DXGIFormat
from ..efmi_extract.migoto_io.migoto_model.types import (
    SlotType,
    ShaderType,
    Topology,
    DXGI_FORMAT,
    ResourceSlot,
)
from ..efmi_extract.migoto_io.migoto_model.migoto_format import MigotoFormat
from ..efmi_extract.migoto_io.migoto_model.frame_model.calls import ShaderCall
from ..efmi_extract.migoto_io.migoto_model.frame_model.resources import (
    ConstantBuffer,
    IndexBuffer,
    Resource,
    VertexBuffer,
)
from ..efmi_extract.migoto_io.migoto_model.frame_model.api_calls.draw_calls import DrawIndexedInstanced
from ..efmi_extract.migoto_io.migoto_model.migoto_mesh import WeightingType
from ..efmi_extract.migoto_io.object_extractor.migoto_object.textures_descriptor import TextureFilter
from ..common.efmi_merged_skeleton import (
    MAX_VERTEX_GROUP_ID,
    PROFILE_FORMAT_VERSION,
    PROFILE_MODE,
    REQUIRED_EFMI_VERSION,
    make_submesh_metadata,
    write_profile,
)


# 3dmigoto 索引缓冲区 DXGI_FORMAT -> numpy 类型
_IB_NUMPY_TYPES = {
    DXGI_FORMAT.DXGI_FORMAT_R16_UINT: numpy.uint16,
    DXGI_FORMAT.DXGI_FORMAT_R32_UINT: numpy.uint32,
}

# 有效的贴图文件后缀 (dump_tex dds / jpg)
_TEXTURE_SUFFIXES = ('.dds', '.jpg')

# 与 EFMI-Tools 0.6.2 默认提取设置一致。该过滤只在骨骼合并提取中使用；
# 普通 DrawIB 提取仍保留现有的完整贴图候选，供用户手动标记。
_EFMI_MERGED_TEXTURE_MIN_FILE_SIZE = 256 * 1024

# 权重相关语义
_BLEND_SEMANTICS = (Semantic.Blendindices, Semantic.Blendweight, Semantic.Blendweights)

# 需要读取的 VB 槽位 (终末地布局固定使用 vb0-vb2)
_VB_IMPORT_SLOTS = (0, 1, 2)

# 导入/导出链路 (mesh_create_helper / obj_buffer_helper) 唯一支持的 TEXCOORD 格式:
# 2 分量 FLOAT。1 分量在导入时 IndexError，SNORM/UNORM 在导出时 Missing element data，
# 4 分量 (R32G32B32A32_FLOAT 等) 会破坏导出端 _parse_texcoord，均视为无效。
_VALID_TEXCOORD_FORMATS = ('R16G16_FLOAT', 'R32G32_FLOAT')

# 导出端语义分发按名称精确匹配这些语义，重命名 (如 NORMAL -> NORMAL1) 会导致 Fatal，
# 元素名冲突时不允许改名，只能把冲突的重复元素转换为 UNKNOWN 占位
_PROTECTED_SEMANTICS = (Semantic.Position, Semantic.Normal, Semantic.Tangent)


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
    unique_strs: list[str]
    json_paths: list[str]
    warnings: list[str]
    workspace_folder: str


@dataclass
class _DrawRecord:
    '''一次匹配的 DrawIndexedInstanced 绘制'''
    shader_call: ShaderCall
    draw_call: DrawIndexedInstanced
    ib: IndexBuffer


@dataclass
class _SlotData:
    '''一个 VB 槽位切片后的数据'''
    slot_id: int
    vb_hash: str
    category: str = ""
    stride: int = 0
    layout: BufferLayout = None
    raw_bytes: bytes = b""


@dataclass
class _TextureEntry:
    slot_id: int
    tex_hash: str
    format_name: str
    call_id: int
    draw_call_id: int
    src_path: Path
    filename: str = ""


@dataclass
class _SubmeshBuild:
    '''一个 unique_str 对应的全部产物 (写盘前的内存表示)'''
    unique_str: str
    ib_hash: str
    index_count: int
    first_index: int
    vertex_count: int
    rebased_indices: numpy.ndarray = None
    slot_datas: list = field(default_factory=list)
    vs_hash_list: list = field(default_factory=list)
    texture_entries: list = field(default_factory=list)
    call_texture_entries: list = field(default_factory=list)
    has_blend: bool = False


def _build_semantic_remap() -> dict:
    '''
    基于 vendored MigotoObjectBuilder 的终末地语义重映射表构建本地版本。

    唯一差异: NORMAL0(R32_FLOAT) 打包 TBN 数据不改名为 ENCODEDDATA0，
    而是保留 NORMAL0 并把 Format 改为 R32_UINT。
    原因: TheHerta4 的导入路径两个名字都支持 (mesh_create_helper)，
    但导出路径 (obj_buffer_helper._parse_normal) 只对 NORMAL + R32_UINT + EFMI
    进行 TBN 重编码，ENCODEDDATA 元素在导出时没有数据来源会直接报错。
    '''
    remap = {}
    for map_from, map_to in MigotoObjectBuilder.semantic_remap.items():
        map_from = copy.deepcopy(map_from)
        map_to = copy.deepcopy(map_to)
        if map_to.abstract.enum == Semantic.EncodedData:
            map_to = BufferSemantic(
                AbstractSemantic(Semantic.Normal, 0),
                format=DXGIFormat.R32_UINT,
                input_slot=map_to.input_slot,
            )
        remap[map_from] = map_to
    return remap


class DumpWorkspaceExtractor:
    '''
    从帧分析 Dump 中按 DrawIB 提取模型并生成 SSMT 风格工作空间。

    公开接口 (UI 层依赖，勿改签名):
        __init__(dump_folder, verbose=False)
        list_draw_ibs() -> list[DrawIBSummary]
        extract(ib_hashes, workspace_folder, gametype_name='GPU-EFMI', copy_textures=True,
                aliases=None) -> ExtractResult
    '''

    def __init__(self, dump_folder: str, verbose: bool = False):
        self.dump_folder = str(dump_folder)
        self.verbose = bool(verbose)

        self._dump_path = Path(self.dump_folder)
        if not self._dump_path.is_dir():
            raise ExtractError("无效的帧分析 Dump 目录 (目录不存在): " + self.dump_folder)
        if not (self._dump_path / "log.txt").is_file():
            raise ExtractError(
                "无效的帧分析 Dump 目录: 未找到 log.txt 文件。\n"
                "请确认选择的是 3dmigoto 帧分析生成的 FrameAnalysis 目录: " + self.dump_folder
            )

        self._model = None
        self._draw_records: list[_DrawRecord] | None = None
        self._txt_format_cache: dict[str, MigotoFormat] = {}
        self._texture_dimensions_cache: dict[str, tuple[int, int]] = {}
        self._semantic_remap = _build_semantic_remap()

    # ------------------------------------------------------------------
    # 帧模型构建
    # ------------------------------------------------------------------

    def _get_model(self):
        '''惰性构建帧模型 (只构建一次, 复用 EFMI-Tools 的跳过命令列表)'''
        if self._model is None:
            extractor = ObjectExtractor(verbose_logging=self.verbose)
            self._model = extractor.build_frame_model(self._dump_path)
        return self._model

    def _get_draw_records(self) -> list[_DrawRecord]:
        '''收集所有绑定了有效 IB 的 DrawIndexedInstanced 绘制'''
        if self._draw_records is not None:
            return self._draw_records

        model = self._get_model()
        records: list[_DrawRecord] = []

        for shader_call in model.calls:
            draw_call = shader_call.draw_call
            if not isinstance(draw_call, DrawIndexedInstanced):
                continue
            if shader_call.model_resources is None:
                continue

            ib = shader_call.model_resources.get_by_slot("ib")
            if ib is None or not isinstance(ib, IndexBuffer):
                continue
            if not ib.hash or str(ib.hash).startswith("UNKNOWN_") or str(ib.hash) == "None":
                continue
            if int(draw_call.index_count or 0) <= 0:
                continue

            records.append(_DrawRecord(shader_call=shader_call, draw_call=draw_call, ib=ib))

        self._draw_records = records
        return records

    # ------------------------------------------------------------------
    # DrawIB 列表
    # ------------------------------------------------------------------

    def list_draw_ibs(self) -> list[DrawIBSummary]:
        '''
        扫描所有 DrawIndexedInstanced 绘制，按 IB hash 聚合，
        按 total_index_count 从大到小排序。
        '''
        aggregates: dict[str, dict] = {}

        for record in self._get_draw_records():
            ib_hash = str(record.ib.hash).lower()
            agg = aggregates.get(ib_hash)
            if agg is None:
                agg = {
                    "draw_call_count": 0,
                    "total_index_count": 0,
                    "has_blend": False,
                    "texture_hashes": set(),
                }
                aggregates[ib_hash] = agg

            agg["draw_call_count"] += 1
            agg["total_index_count"] += int(record.draw_call.index_count or 0)

            if not agg["has_blend"]:
                agg["has_blend"] = self._call_has_blend(record.shader_call)

            for entry in self._collect_call_textures(record):
                agg["texture_hashes"].add(entry.tex_hash)

        summaries = [
            DrawIBSummary(
                ib_hash=ib_hash,
                draw_call_count=agg["draw_call_count"],
                total_index_count=agg["total_index_count"],
                has_blend=agg["has_blend"],
                texture_count=len(agg["texture_hashes"]),
            )
            for ib_hash, agg in aggregates.items()
        ]
        summaries.sort(key=lambda summary: summary.total_index_count, reverse=True)
        return summaries

    def _call_has_blend(self, shader_call: ShaderCall) -> bool:
        '''检查绘制的任意 VB 布局是否包含权重语义 (BLENDINDICES/BLENDWEIGHT)'''
        for slot_id in _VB_IMPORT_SLOTS:
            vb = shader_call.model_resources.get_by_slot("vb" + str(slot_id))
            if vb is None or not isinstance(vb, VertexBuffer):
                continue
            try:
                fmt = self._load_txt_format(vb)
            except Exception:
                continue
            if fmt is None or fmt.vb_layout is None:
                continue
            for semantic in fmt.vb_layout.semantics:
                if semantic.abstract.enum in _BLEND_SEMANTICS:
                    return True
        return False

    # ------------------------------------------------------------------
    # 提取主流程
    # ------------------------------------------------------------------

    def extract(
        self,
        ib_hashes: list[str],
        workspace_folder: str,
        gametype_name: str = 'GPU-EFMI',
        copy_textures: bool = True,
        aliases: dict[str, str] | None = None,
    ) -> ExtractResult:
        '''
        提取指定 DrawIB 的所有绘制，并写入 SSMT 风格工作空间。
        每个不同的 (ib_hash, index_count, first_index) 组合生成一个 unique_str 子目录。

        aliases: 可选的 {ib_hash: 别名} 映射，写入工作空间根 Config.json 的 Alias 字段。
        '''
        normalized_hashes = {h.strip().lower() for h in (ib_hashes or []) if h and h.strip()}
        if not normalized_hashes:
            raise ExtractError("未指定任何 DrawIB (ib_hashes 为空)，无法执行提取。")

        workspace_folder = os.path.abspath(str(workspace_folder))
        os.makedirs(workspace_folder, exist_ok=True)

        warnings: list[str] = []
        unique_strs: list[str] = []
        json_paths: list[str] = []
        extracted_ib_hashes: list[str] = []

        # 按 (ib_hash, index_count, first_index) 分组
        groups: dict[tuple, list[_DrawRecord]] = {}
        for record in self._get_draw_records():
            ib_hash = str(record.ib.hash).lower()
            if ib_hash not in normalized_hashes:
                continue
            key = (ib_hash, int(record.draw_call.index_count), int(record.draw_call.first_index))
            groups.setdefault(key, []).append(record)

        found_hashes = {key[0] for key in groups.keys()}
        for missing_hash in sorted(normalized_hashes - found_hashes):
            warnings.append("在 Dump 中没有找到 DrawIB 为 " + missing_hash + " 的 DrawIndexedInstanced 绘制。")

        for key in sorted(groups.keys()):
            ib_hash, index_count, first_index = key
            unique_str = ib_hash + "-" + str(index_count) + "-" + str(first_index)
            try:
                build = self._build_submesh(groups[key], unique_str, warnings)
            except ExtractError as e:
                warnings.append("跳过 " + unique_str + ": " + str(e))
                continue
            except Exception as e:
                warnings.append("跳过 " + unique_str + " (意外错误): " + repr(e))
                continue

            try:
                json_path = self._write_submesh(
                    build=build,
                    workspace_folder=workspace_folder,
                    gametype_name=gametype_name,
                    copy_textures=copy_textures,
                    warnings=warnings,
                )
            except Exception as e:
                warnings.append("写入 " + unique_str + " 失败: " + repr(e))
                continue

            unique_strs.append(unique_str)
            json_paths.append(json_path)
            if ib_hash not in extracted_ib_hashes:
                extracted_ib_hashes.append(ib_hash)

            if self.verbose:
                print("已提取: " + unique_str + " -> " + json_path)

        if unique_strs:
            self._update_workspace_root_files(
                workspace_folder=workspace_folder,
                unique_strs=unique_strs,
                gametype_name=gametype_name,
                draw_ibs=extracted_ib_hashes,
                aliases=aliases,
            )

        return ExtractResult(
            unique_strs=unique_strs,
            json_paths=json_paths,
            warnings=warnings,
            workspace_folder=workspace_folder,
        )

    def extract_merged_skeleton(
        self,
        workspace_folder: str,
        gametype_name: str = 'GPU-EFMI',
        copy_textures: bool = True,
    ) -> ExtractResult:
        '''
        自动识别当前帧中的主要显式权重对象，生成 EFMI 1.4.1 Merged
        Skeleton 工作空间。目录/JSON/物体主键仍沿用 LoyalTools 的
        ``IBHash-IndexCount-FirstIndex``，不会调用或修改普通 ``extract`` 流程。
        '''
        workspace_folder = os.path.abspath(str(workspace_folder))
        os.makedirs(workspace_folder, exist_ok=True)
        warnings: list[str] = []

        object_extractor = ObjectExtractor(verbose_logging=self.verbose)
        try:
            candidates = object_extractor.extract_objects(
                model=self._get_model(),
                draw_call_filter=DrawCallFilter(),
                raw_object_filter=RawObjectFilter(min_component_count=1),
                migoto_object_filter=MigotoObjectFilter(
                    skip_static_objects=True,
                    ignore_errors=True,
                ),
            )
        except Exception as exc:
            raise ExtractError("骨骼合并角色识别失败: " + repr(exc))

        explicit_candidates = []
        for candidate in candidates:
            gpu_components = [
                component for component in candidate.components
                if not component.mesh.cpu_posed
            ]
            if not gpu_components:
                continue
            weighting_types = [
                component.mesh.get_weighting_type()
                for component in gpu_components
            ]
            # EFMI-Tools 对象级 weighting_type 取组件中的最高等级：只要对象
            # 含 Explicit 组件就属于显式权重对象。真实角色可以同时包含
            # Explicit 与 Implicit 组件（隐式组件只有 BLENDINDICES，首项权重
            # 按 1.0 处理），不能要求每个组件都有 BLENDWEIGHTS。
            if WeightingType.Explicit not in weighting_types:
                continue
            if any(
                weighting_type not in (WeightingType.Explicit, WeightingType.Implicit)
                for weighting_type in weighting_types
            ):
                continue
            explicit_candidates.append(candidate)

        if not explicit_candidates:
            raise ExtractError(
                "帧分析中没有识别到可用于骨骼合并的显式权重角色。"
                "请在角色完整显示且未被菜单遮挡时重新抓帧。"
            )

        # EFMI-Tools 用 Character 标签识别多部件角色。若同一帧存在武器、NPC 等
        # 多个加权对象，优先 Character，其次按组件数和总顶点数选择主对象。
        selected_object = max(
            explicit_candidates,
            key=lambda obj: (
                1 if str(obj.id).startswith("Character") else 0,
                len(obj.components),
                sum(int(component.mesh.format.vertex_count) for component in obj.components),
            ),
        )
        if len(explicit_candidates) > 1:
            warnings.append(
                "帧中识别到 " + str(len(explicit_candidates))
                + " 个显式权重对象，已自动选择 " + str(selected_object.id) + "。"
            )

        component_vg_metadata = self._build_merged_skeleton_vg_metadata(selected_object)
        profile_components = []
        unique_strs = []
        seen_unique_strs = set()
        json_paths = []
        extracted_ib_hashes = []

        for source_component_id, component in enumerate(selected_object.components):
            records = []
            for shader_call in component.raw_data.shader_calls:
                draw_call = shader_call.draw_call
                resources = shader_call.model_resources
                if not isinstance(draw_call, DrawIndexedInstanced) or resources is None:
                    continue
                ib = resources.get_by_slot("ib")
                if not isinstance(ib, IndexBuffer):
                    continue
                records.append(_DrawRecord(shader_call, draw_call, ib))

            if not records:
                raise ExtractError(
                    "骨骼合并组件 " + str(source_component_id) + " 没有可导出的 DrawIndexedInstanced。"
                )

            primary_key = (
                str(records[0].ib.hash).lower(),
                int(records[0].draw_call.index_count),
                int(records[0].draw_call.first_index or 0),
            )
            matching_records = [
                record for record in records
                if (
                    str(record.ib.hash).lower(),
                    int(record.draw_call.index_count),
                    int(record.draw_call.first_index or 0),
                ) == primary_key
            ]
            if len(matching_records) != len(records):
                warnings.append(
                    "组件 " + str(source_component_id) + " 存在不同绘制范围，"
                    "已按首个主绘制生成 LoyalTools 子网格。"
                )

            ib_hash, index_count, first_index = primary_key
            unique_str = ib_hash + "-" + str(index_count) + "-" + str(first_index)
            if unique_str in seen_unique_strs:
                raise ExtractError(
                    "骨骼合并识别到重复的 LoyalTools 子网格主键: " + unique_str
                    + "。请重新抓取角色完整、绘制范围稳定的一帧。"
                )
            seen_unique_strs.add(unique_str)
            vg_metadata = component_vg_metadata[source_component_id]
            profile_component = {
                "component_id": len(profile_components),
                "source_component_id": source_component_id,
                "unique_str": unique_str,
                "ib_hash": ib_hash,
                "index_count": index_count,
                "first_index": first_index,
                "vertex_count": int(component.mesh.format.vertex_count),
                "cpu_posed": bool(component.mesh.cpu_posed),
                "vg_offset": int(vg_metadata["vg_offset"]),
                "vg_count": int(vg_metadata["vg_count"]),
                "vg_map": dict(vg_metadata["vg_map"]),
                "lods": [],
            }

            try:
                build = self._build_submesh(matching_records, unique_str, warnings)
                if copy_textures:
                    # MigotoComponent.textures 来自 EFMI-Tools 对当前组件的显式资源归属，
                    # 比按同 IB 的所有帧调用聚合更准确，可排除其他组件/阴影/屏幕通道贴图。
                    # 只替换骨骼合并 build 的贴图列表，不触碰普通提取路径。
                    (
                        build.texture_entries,
                        build.call_texture_entries,
                    ) = self._collect_merged_component_textures(
                        component=component,
                        unique_str=unique_str,
                        warnings=warnings,
                    )
                profile_component["vertex_count"] = int(build.vertex_count)
                json_path = self._write_submesh(
                    build=build,
                    workspace_folder=workspace_folder,
                    gametype_name=gametype_name,
                    copy_textures=copy_textures,
                    warnings=warnings,
                    merged_component=profile_component,
                )
            except Exception as exc:
                raise ExtractError(
                    "骨骼合并组件 " + str(source_component_id) + " (" + unique_str
                    + ") 提取失败: " + repr(exc)
                )

            profile_components.append(profile_component)
            unique_strs.append(unique_str)
            json_paths.append(json_path)
            if ib_hash not in extracted_ib_hashes:
                extracted_ib_hashes.append(ib_hash)

        profile = {
            "format_version": PROFILE_FORMAT_VERSION,
            "mode": PROFILE_MODE,
            "required_efmi_version": REQUIRED_EFMI_VERSION,
            "object_name": str(selected_object.id),
            "weighting_type": WeightingType.Explicit.value,
            "object_guid": sum(component["index_count"] for component in profile_components),
            "max_instance_count": 8,
            "components": profile_components,
        }
        write_profile(workspace_folder, profile)

        self._update_workspace_root_files(
            workspace_folder=workspace_folder,
            unique_strs=unique_strs,
            gametype_name=gametype_name,
            draw_ibs=extracted_ib_hashes,
            aliases=None,
        )
        return ExtractResult(
            unique_strs=unique_strs,
            json_paths=json_paths,
            warnings=warnings,
            workspace_folder=workspace_folder,
        )

    def _collect_merged_component_textures(
        self,
        component,
        unique_str: str,
        warnings: list[str],
    ) -> tuple[list[_TextureEntry], list[_TextureEntry]]:
        '''
        按 EFMI-Tools 的组件资源归属和默认过滤规则生成贴图条目。

        component.textures 只包含该 MigotoComponent 在显式资源调用中实际使用的
        ResourceSlot -> Resource 列表，因此不会把帧中相同 IB 或辅助通道的无关贴图
        混入当前 LoyalTools unique_str。不同组件使用同一 hash 时仍会各自在自己的
        TYPE_ 目录保留一份候选文件，满足每个 IB 独立编辑贴图的要求。
        '''
        texture_filter = TextureFilter(
            exclude_extensions=["jpg", "buf"],
            exclude_hashes=[],
            min_file_size=_EFMI_MERGED_TEXTURE_MIN_FILE_SIZE,
        )

        call_texture_entries: list[_TextureEntry] = []
        seen_call_entries = set()
        for slot, resources in component.textures.items():
            if slot.shader_type != ShaderType.Pixel or slot.slot_type != SlotType.Texture:
                continue
            for resource in resources:
                tex_hash = str(resource.hash or "").lower()
                if not tex_hash or tex_hash.startswith("unknown_"):
                    continue
                try:
                    if not texture_filter.is_valid_texture(resource):
                        continue
                except (OSError, ValueError, TypeError) as exc:
                    warnings.append(
                        unique_str + " 骨骼合并贴图过滤失败 " + tex_hash + ": " + repr(exc)
                    )
                    continue

                src_path = Path(resource.bin_path_deduped)
                format_name = "UNKNOWN"
                data_descriptor = resource.data_descriptor
                if data_descriptor is not None and getattr(data_descriptor, "data_format", None):
                    format_name = str(data_descriptor.data_format)

                usage_descriptor = resource.usage_descriptor
                call_id = 0
                if usage_descriptor is not None and getattr(usage_descriptor, "call_id", None) is not None:
                    call_id = int(usage_descriptor.call_id)
                dedupe_key = (call_id, int(slot.slot_id), tex_hash)
                if dedupe_key in seen_call_entries:
                    continue
                seen_call_entries.add(dedupe_key)
                call_texture_entries.append(_TextureEntry(
                    slot_id=int(slot.slot_id),
                    tex_hash=tex_hash,
                    format_name=format_name,
                    call_id=call_id,
                    draw_call_id=call_id,
                    src_path=src_path,
                ))

        call_texture_entries.sort(
            key=lambda entry: (entry.draw_call_id, entry.slot_id, entry.tex_hash)
        )
        texture_entries: list[_TextureEntry] = []
        seen_texture_entries = set()
        for entry in call_texture_entries:
            dedupe_key = (entry.slot_id, entry.tex_hash)
            if dedupe_key in seen_texture_entries:
                continue
            seen_texture_entries.add(dedupe_key)
            texture_entries.append(entry)
        texture_entries.sort(key=lambda entry: (entry.slot_id, entry.tex_hash))
        return texture_entries, call_texture_entries

    @staticmethod
    def _component_resources(component):
        for shader_call in component.raw_data.shader_calls:
            resources = shader_call.model_resources or shader_call.resources
            if resources is not None:
                yield resources

    def _get_component_skeleton_data(self, component) -> numpy.ndarray:
        instance_config_cb = None
        skeleton_resource = None
        for resources in self._component_resources(component):
            if instance_config_cb is None:
                for slot, constant_buffer in resources.constant_buffers.items():
                    if slot.shader_type == ShaderType.Vertex and constant_buffer.num_constants == 4096:
                        instance_config_cb = constant_buffer
                        break
            if skeleton_resource is None:
                skeleton_resource = resources.get_by_slot(
                    ResourceSlot(ShaderType.Vertex, SlotType.Texture, 0)
                )
            if instance_config_cb is not None and skeleton_resource is not None:
                break

        if not isinstance(instance_config_cb, ConstantBuffer):
            raise ExtractError("组件缺少 4096 常量的实例配置 VS-CB。")
        if not isinstance(skeleton_resource, Resource):
            raise ExtractError("组件缺少 vs-t0 骨骼数据资源。")

        raw_layout = MigotoFormat(vb_layout=BufferLayout([
            BufferSemantic(
                AbstractSemantic(Semantic.RawData, 0),
                DXGIFormat.R32G32B32A32_FLOAT,
                input_slot=0,
            ),
        ]))
        if instance_config_cb.buffer is None:
            instance_config_cb.build_numpy_buffer(raw_layout)
        config_data = instance_config_cb.buffer.get_field(0)
        config_offset = int(instance_config_cb.first_constant)
        instance_config = config_data[config_offset:config_offset + 16]
        if len(instance_config) < 6:
            raise ExtractError("组件的实例配置常量数据不完整。")
        skeleton_offset = int(instance_config[5][0:2].view(numpy.uint32)[0])
        if skeleton_offset == 0:
            raise ExtractError("组件实例配置没有主骨骼偏移。")

        if skeleton_resource.buffer is None:
            skeleton_resource.build_numpy_buffer(raw_layout)
        raw_data = skeleton_resource.buffer.get_field(0)
        data_offset = skeleton_offset + 3
        skeleton_raw = raw_data[data_offset:data_offset + 256 * 3]
        usable_size = (len(skeleton_raw) // 3) * 3
        if usable_size == 0:
            raise ExtractError("组件的 vs-t0 骨骼数据为空。")
        return skeleton_raw[:usable_size].reshape(-1, 12)

    def _is_valid_bone_source(self, component) -> bool:
        # EFMI-Tools v0.6.2 内置黑名单：Liino rocket boots 的骨骼矩阵不能作为
        # 重复骨骼的权威来源，但组件本身仍然参与合并。
        for resources in self._component_resources(component):
            if resources.get_by_hash("80aafa4b"):
                return False
        return True

    def _build_merged_skeleton_vg_metadata(self, migoto_object) -> list[dict]:
        '''移植 EFMI-Tools v0.6.2 的矩阵去重策略，构造全局 VG 地址空间。'''
        result = [
            {"vg_offset": 0, "vg_count": 0, "vg_map": {}}
            for _ in migoto_object.components
        ]
        vg_offset = 0
        bone_candidates: dict[tuple, list[dict]] = {}

        for component_id, component in enumerate(migoto_object.components):
            if component.mesh.cpu_posed:
                continue

            vg_ids = component.mesh.get_data(Semantic.Blendindices)
            vg_weights = component.mesh.get_data(Semantic.Blendweights)
            if vg_weights is None:
                vg_weights = component.mesh.get_data(Semantic.Blendweight)
            if vg_ids is None or vg_ids.size == 0:
                raise ExtractError("组件 " + str(component_id) + " 缺少 BLENDINDICES。")
            vg_ids = numpy.asarray(vg_ids, dtype=numpy.uint32)
            if vg_weights is None:
                vg_weights = numpy.zeros_like(vg_ids, dtype=numpy.float32)
                vg_weights[..., 0] = 1.0
            else:
                vg_weights = numpy.asarray(vg_weights, dtype=numpy.float32)

            vg_count = int(vg_ids.max()) + 1
            if vg_offset + vg_count - 1 > MAX_VERTEX_GROUP_ID:
                raise ExtractError(
                    "骨骼合并全局顶点组超过 R16_UINT 上限 "
                    + str(MAX_VERTEX_GROUP_ID) + "。"
                )
            weighted_vertex_counts = numpy.bincount(
                vg_ids[vg_weights != 0], minlength=vg_count
            )
            skeleton_buffer = self._get_component_skeleton_data(component)
            if len(skeleton_buffer) < vg_count:
                raise ExtractError(
                    "组件 " + str(component_id) + " 的骨骼只有 "
                    + str(len(skeleton_buffer)) + " 个，但网格声明了 "
                    + str(vg_count) + " 个顶点组。"
                )

            component_map = {
                str(local_vg_id): vg_offset + local_vg_id
                for local_vg_id in range(vg_count)
            }
            result[component_id] = {
                "vg_offset": vg_offset,
                "vg_count": vg_count,
                "vg_map": component_map,
            }
            valid_source = self._is_valid_bone_source(component)
            for local_vg_id in range(vg_count):
                bone_data = tuple(skeleton_buffer[local_vg_id].tolist())
                if all(value == 0 for value in bone_data):
                    continue
                bone_candidates.setdefault(bone_data, []).append({
                    "component_id": component_id,
                    "local_vg_id": local_vg_id,
                    "global_vg_id": vg_offset + local_vg_id,
                    "weighted_vertex_count": int(weighted_vertex_counts[local_vg_id]),
                    "is_valid_source": valid_source,
                })
            vg_offset += vg_count

        for candidates in bone_candidates.values():
            valid_candidates = [item for item in candidates if item["is_valid_source"]]
            source = max(
                valid_candidates,
                key=lambda item: item["weighted_vertex_count"],
            ) if valid_candidates else candidates[0]
            for candidate in candidates:
                result[candidate["component_id"]]["vg_map"][
                    str(candidate["local_vg_id"])
                ] = int(source["global_vg_id"])

        return result

    # ------------------------------------------------------------------
    # 索引缓冲区
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bin_path(resource) -> Path | None:
        '''优先使用 dump 根目录的 bin 文件，缺失时回退到 deduped 文件'''
        for path in (resource.bin_path, resource.bin_path_deduped):
            if path is not None and Path(path).is_file():
                return Path(path)
        return None

    def _build_index_data(self, record: _DrawRecord):
        '''
        按绘制范围切片 IB 并把索引重定基到 0。

        返回 (rebased_indices(uint32), vertex_offset, vertex_count)。
        其中 vertex_offset 已包含 BaseVertexLocation。

        注意: 终末地把所有网格放在巨型共享缓冲区中，3dmigoto 为每个绘制生成
        带独立 hash 的 view (efmi_extract migoto_calls.MigotoDumpFile)，view 上的
        format/byte_offset dataclass 字段是默认值，正确的绘制区间元数据在
        migoto_format (来自 deduped 文件名描述符) 中。因此优先走 vendored 的
        build_numpy_buffer 管线 (与 EFMI-Tools raw_object_extractor 相同)，
        仅在其不可用时回退到 IASetIndexBuffer 的原始字段。
        '''
        ib = record.ib
        draw = record.draw_call

        first_index = int(draw.first_index or 0)
        index_count = int(draw.index_count)

        # 拓扑校验: 仅支持 trianglelist (拓扑信息缺失时不拦截)
        topology = self._get_ib_topology(ib)
        if topology is not None and topology != Topology.TriangleList:
            raise ExtractError(
                "IB " + str(ib.hash) + " 图元拓扑为 " + str(topology) + ", 仅支持 trianglelist 拓扑绘制。"
            )
        if index_count % 3 != 0:
            raise ExtractError(
                "索引数量 " + str(index_count) + " 不是 3 的倍数, 不是有效的三角形列表绘制。"
            )

        # 主路径: vendored 管线 (deduped 描述符定位绘制区间)
        indices = None
        if getattr(ib, "migoto_format", None) is not None or getattr(ib, "data_descriptor", None) is not None:
            try:
                if getattr(ib, "buffer", None) is None:
                    ib.build_numpy_buffer()
                if getattr(ib, "buffer", None) is not None:
                    indices = ib.buffer.get_field(Semantic.Index).flatten()
            except Exception as e:
                if self.verbose:
                    print("IB " + str(ib.hash) + " vendored 读取失败，回退手动读取: " + repr(e))
                indices = None

        if indices is not None:
            if len(indices) == index_count:
                pass
            elif len(indices) >= first_index + index_count:
                # 描述符覆盖了比本次绘制更大的区间 (如整个缓冲区)，按绘制参数切片
                indices = indices[first_index:first_index + index_count]
            elif len(indices) >= index_count:
                indices = indices[:index_count]
            else:
                if self.verbose:
                    print("IB " + str(ib.hash) + " vendored 数据长度不足 ("
                          + str(len(indices)) + " < " + str(index_count) + ")，回退手动读取。")
                indices = None

        if indices is None:
            # 回退路径: 使用 IASetIndexBuffer 的原始字段直接读取 (非 view 情况)
            ib_format = ib.format
            if ib_format not in _IB_NUMPY_TYPES:
                parent = getattr(ib, "parent", None)
                if parent is not None and getattr(parent, "format", None) in _IB_NUMPY_TYPES:
                    ib_format = parent.format
            np_type = _IB_NUMPY_TYPES.get(ib_format)
            if np_type is None:
                raise ExtractError("不支持的索引缓冲区格式: " + str(ib.format))

            bin_path = self._resolve_bin_path(ib)
            if bin_path is None:
                raise ExtractError(
                    "IB " + str(ib.hash) + " 缺少二进制数据文件 (需要在 analyse_options 中包含 buf dump_ib 选项)。"
                )

            itemsize = numpy.dtype(np_type).itemsize
            byte_offset = int(ib.byte_offset or 0) + first_index * itemsize
            byte_size = index_count * itemsize

            file_size = bin_path.stat().st_size
            if byte_offset + byte_size > file_size:
                raise ExtractError(
                    "IB 文件长度不足: " + bin_path.name
                    + " 需要 " + str(byte_offset + byte_size) + " 字节, 实际 " + str(file_size) + " 字节。"
                )

            indices = numpy.fromfile(bin_path, dtype=np_type, count=index_count, offset=byte_offset)

        indices = indices.astype(numpy.uint32)

        vertex_offset = int(indices.min())
        vertex_count = int(indices.max()) - vertex_offset + 1

        # 重定基到 0 (与 EFMI-Tools raw_object_extractor 相同的处理)
        if vertex_offset > 0:
            indices = indices - numpy.uint32(vertex_offset)

        # BaseVertexLocation 会在绘制时自动加到每个索引上，
        # 读取 VB 时需要与 IB 隐含的偏移叠加
        base_vertex = int(getattr(draw, "first_vertex", 0) or 0)
        total_vertex_offset = vertex_offset + base_vertex

        rebased = numpy.ascontiguousarray(indices, dtype='<u4')
        return rebased, total_vertex_offset, vertex_count

    def _get_ib_topology(self, ib: IndexBuffer) -> Topology | None:
        '''尽力获取 IB 的图元拓扑 (依次尝试帧模型字段 / dump 文件名描述符 / .txt 头部)，未知时返回 None'''
        topology = getattr(ib, "topology", None)
        if isinstance(topology, Topology) and topology != Topology.Undefined:
            return topology

        data_descriptor = getattr(ib, "data_descriptor", None)
        if data_descriptor is not None:
            descriptor_topology = getattr(data_descriptor, "topology", None)
            if isinstance(descriptor_topology, Topology) and descriptor_topology != Topology.Undefined:
                return descriptor_topology

        for txt_path in (ib.txt_path, ib.txt_path_deduped):
            if txt_path is None:
                continue
            txt_path = Path(txt_path)
            if not txt_path.is_file():
                continue
            cache_key = str(txt_path)
            fmt = self._txt_format_cache.get(cache_key)
            if fmt is None:
                try:
                    with open(txt_path, 'r') as f:
                        fmt = MigotoFormat.from_txt_file(f)
                except Exception:
                    continue
                self._txt_format_cache[cache_key] = fmt
            if isinstance(fmt.topology, Topology) and fmt.topology != Topology.Undefined:
                return fmt.topology

        return None

    # ------------------------------------------------------------------
    # 顶点缓冲区
    # ------------------------------------------------------------------

    def _load_txt_format(self, vb: VertexBuffer) -> MigotoFormat | None:
        '''解析 VB 对应的 .txt 头部布局 (带缓存)'''
        for txt_path in (vb.txt_path, vb.txt_path_deduped):
            if txt_path is None:
                continue
            txt_path = Path(txt_path)
            if not txt_path.is_file():
                continue
            cache_key = str(txt_path)
            fmt = self._txt_format_cache.get(cache_key)
            if fmt is None:
                with open(txt_path, 'r') as f:
                    fmt = MigotoFormat.from_txt_file(f)
                self._txt_format_cache[cache_key] = fmt
            return fmt
        return None

    def _build_slot_data(
        self,
        vb: VertexBuffer,
        slot_id: int,
        vertex_offset: int,
        vertex_count: int,
        unknown_index_offset: int,
        warnings: list[str],
    ) -> tuple[_SlotData | None, int]:
        '''
        构建一个 VB 槽位的布局与顶点数据切片。
        返回 (slot_data 或 None, 新的 unknown 语义索引偏移)。
        '''
        fmt = self._load_txt_format(vb)
        if fmt is None:
            raise ExtractError(
                "vb" + str(slot_id) + " (hash=" + str(vb.hash) + ") 缺少 .txt 布局文件 "
                + "(需要在 analyse_options 中包含 txt dump_vb 选项)。"
            )
        if fmt.vb_layout is None:
            raise ExtractError("vb" + str(slot_id) + " 的 .txt 文件中没有元素布局信息。")

        slot_semantics = copy.deepcopy(fmt.vb_layout.get_elements_in_slot(slot_id))
        if not slot_semantics:
            # 此槽位在输入布局中没有元素，跳过
            return None, unknown_index_offset

        stride = int(fmt.stride or 0)
        if stride <= 0:
            print("警告: vb" + str(slot_id) + " (hash=" + str(vb.hash) + ") stride 为 0，跳过该槽位。")
            return None, unknown_index_offset

        layout = BufferLayout(
            semantics=slot_semantics,
            auto_offsets=False,
            auto_stride=False,
        )
        layout.stride = stride

        # 与 EFMI-Tools MigotoObjectBuilder 相同的布局清洗流程
        layout.sort()
        layout.remove_data_views()
        remapped_semantics = layout.remap_semantics(self._semantic_remap)
        if remapped_semantics and self.verbose:
            for map_from, map_to in remapped_semantics:
                print("语义重映射 vb" + str(slot_id) + ": " + str(map_from) + " -> " + str(map_to))
        layout.dedupe_semantics()

        # TEXCOORD 有效性检查: 导入/导出链路仅支持 2 分量 FLOAT 的 TEXCOORD，
        # 其余格式 (1/3/4 分量或 SNORM/UNORM 等) 转换为 UNKNOWN 占位元素，
        # 保持 ByteWidth 精确不变以免破坏 stride
        for semantic in layout.semantics:
            if semantic.abstract.enum != Semantic.TexCoord:
                continue
            if semantic.format.format in _VALID_TEXCOORD_FORMATS:
                continue
            original_name = self._element_name(semantic)
            original_format = semantic.format.format
            semantic.abstract = AbstractSemantic(Semantic.Unknown, unknown_index_offset)
            semantic.format = self._unknown_placeholder_format(semantic.stride)
            unknown_index_offset += 1
            warnings.append(
                "vb" + str(slot_id) + " 的 " + original_name + " 格式 " + original_format
                + " 不是 2 分量 FLOAT (导入/导出链路不支持)，已转换为 "
                + self._element_name(semantic) + " 占位元素。"
            )

        missing_semantics = layout.fill_missing_semantics(unknown_index_offset)
        if missing_semantics:
            unknown_index_offset += len(missing_semantics)
            if self.verbose:
                print("vb" + str(slot_id) + " 填充了 " + str(len(missing_semantics)) + " 个 UNKNOWN 占位语义。")

        semantics_stride = layout.calculate_stride()
        if semantics_stride != stride:
            raise ExtractError(
                "vb" + str(slot_id) + " 布局字节宽度 (" + str(semantics_stride)
                + ") 与 stride (" + str(stride) + ") 不一致，无法安全切片。"
            )

        bin_path = self._resolve_bin_path(vb)
        if bin_path is None:
            raise ExtractError(
                "vb" + str(slot_id) + " (hash=" + str(vb.hash) + ") 缺少二进制数据文件 "
                + "(需要在 analyse_options 中包含 buf dump_vb 选项)。"
            )

        # 字节窗口计算 (与 EFMI-Tools MigotoObjectBuilder.build_vertex_buffer 相同):
        # - 描述符的 vertex_count 与 IB 寻址顶点数一致时，说明 3dmigoto 已经按绘制
        #   区间生成了 byte offset (指向首个被寻址的顶点行)，直接用描述符自身的
        #   first_vertex 定位，不能再叠加 vertex_offset (否则双重偏移)；
        # - 不一致时 (描述符覆盖整个缓冲区等)，用 IB 推导的 vertex_offset 定位。
        vb_byte_offset = int(fmt.byte_offset or 0)
        fmt_first_vertex = int(fmt.first_vertex or 0)
        fmt_vertex_count = int(fmt.vertex_count or 0)
        if fmt_vertex_count == vertex_count:
            byte_start = vb_byte_offset + stride * fmt_first_vertex
        else:
            byte_start = vb_byte_offset + stride * vertex_offset
        byte_size = stride * vertex_count

        file_size = bin_path.stat().st_size
        if byte_start + byte_size > file_size:
            raise ExtractError(
                "vb" + str(slot_id) + " 文件长度不足: " + bin_path.name
                + " 需要 " + str(byte_start + byte_size) + " 字节, 实际 " + str(file_size) + " 字节。"
            )

        with open(bin_path, 'rb') as f:
            f.seek(byte_start)
            raw_bytes = f.read(byte_size)

        if len(raw_bytes) != byte_size:
            raise ExtractError("vb" + str(slot_id) + " 数据读取长度不符: " + bin_path.name)

        slot_data = _SlotData(
            slot_id=slot_id,
            vb_hash=str(vb.hash),
            stride=stride,
            layout=layout,
            raw_bytes=raw_bytes,
        )
        return slot_data, unknown_index_offset

    # ------------------------------------------------------------------
    # 子网格构建
    # ------------------------------------------------------------------

    def _build_submesh(self, records: list[_DrawRecord], unique_str: str, warnings: list[str]) -> _SubmeshBuild:
        record = records[0]
        draw = record.draw_call

        # 同一 (ib_hash, index_count, first_index) 分组内的绘制可能存在
        # BaseVertexLocation 或 VB 绑定不一致的情况 (实为不同的几何体)，
        # 只有与首条记录一致的绘制才参与贴图 / VS hash 合并
        base_identity = self._record_draw_identity(record)
        matched_records = [rec for rec in records if self._record_draw_identity(rec) == base_identity]
        if len(matched_records) != len(records):
            warnings.append(
                unique_str + " 分组内存在 " + str(len(records) - len(matched_records))
                + " 个 BaseVertexLocation 或 VB hash 不一致的绘制，已忽略其贴图与 VS 信息。"
            )
        records = matched_records

        rebased_indices, vertex_offset, vertex_count = self._build_index_data(record)

        # 逐槽位切片 VB
        slot_datas: list[_SlotData] = []
        unknown_index_offset = 0
        for slot_id in _VB_IMPORT_SLOTS:
            vb = record.shader_call.model_resources.get_by_slot("vb" + str(slot_id))
            if vb is None or not isinstance(vb, VertexBuffer):
                continue
            slot_data, unknown_index_offset = self._build_slot_data(
                vb=vb,
                slot_id=slot_id,
                vertex_offset=vertex_offset,
                vertex_count=vertex_count,
                unknown_index_offset=unknown_index_offset,
                warnings=warnings,
            )
            if slot_data is not None:
                slot_datas.append(slot_data)

        if not slot_datas:
            raise ExtractError("没有任何可用的 VB 槽位数据。")

        # 分类命名: 含 POSITION 的槽位 -> Position, 含权重语义 -> Blend, 其余 -> Texcoord...
        # ("Position" / "Blend" 名称在 TheHerta4 中是硬编码的，必须保持一致)
        has_blend = False
        position_assigned = False
        blend_assigned = False
        texcoord_counter = 0
        for slot_data in slot_datas:
            semantic_enums = {semantic.abstract.enum for semantic in slot_data.layout.semantics}
            is_position = (Semantic.Position in semantic_enums) and not position_assigned
            is_blend = any(enum in _BLEND_SEMANTICS for enum in semantic_enums) and not blend_assigned

            if is_position:
                slot_data.category = "Position"
                position_assigned = True
            elif is_blend:
                slot_data.category = "Blend"
                blend_assigned = True
                has_blend = True
            else:
                slot_data.category = "Texcoord" if texcoord_counter == 0 else "Texcoord" + str(texcoord_counter)
                texcoord_counter += 1

        if not position_assigned:
            raise ExtractError("没有找到包含 POSITION 语义的 VB 槽位，无法生成 Position 分类。")

        # 保证 ElementName (SemanticName+SemanticIndex) 在整个子网格中唯一，
        # 否则 numpy structured dtype 与 D3D11GameType 的字典会发生冲突
        used_element_names = set()
        for slot_data in slot_datas:
            for semantic in slot_data.layout.semantics:
                element_name = self._element_name(semantic)
                if element_name in used_element_names:
                    original_name = element_name
                    if semantic.abstract.enum in _PROTECTED_SEMANTICS:
                        # POSITION/NORMAL/TANGENT 不允许改名 (导出端语义分发按名称精确匹配)，
                        # 冲突的重复元素转换为等 ByteWidth 的 UNKNOWN 占位元素
                        semantic.abstract = AbstractSemantic(Semantic.Unknown, 0)
                        element_name = self._element_name(semantic)
                        while element_name in used_element_names:
                            semantic.abstract.index += 1
                            element_name = self._element_name(semantic)
                        semantic.format = self._unknown_placeholder_format(semantic.stride)
                        warnings.append(
                            unique_str + " 元素名冲突: 重复的 " + original_name
                            + " 已转换为 " + element_name + " 占位元素。"
                        )
                    else:
                        while element_name in used_element_names:
                            semantic.abstract.index += 1
                            element_name = self._element_name(semantic)
                        print("警告: 元素名冲突, " + original_name + " 已重命名为 " + element_name)
                used_element_names.add(element_name)

        # 收集所有匹配绘制的 VS hash
        vs_hash_list: list[str] = []
        for rec in records:
            for vs_hash in self._collect_call_vs_hashes(rec):
                if vs_hash and vs_hash not in vs_hash_list:
                    vs_hash_list.append(vs_hash)

        # 收集所有匹配绘制的 ps 贴图:
        # - call_texture_entries 保留每个绘制自己的完整 ps-t 绑定 (不去重, 供逐绘制分组)
        # - texture_entries 按 (槽位, hash) 去重 (供槽位聚合与文件落盘)
        texture_entries: list[_TextureEntry] = []
        call_texture_entries: list[_TextureEntry] = []
        seen_texture_keys = set()
        for rec in records:
            rec_entries = self._collect_call_textures(rec)
            rec_entries.sort(key=lambda entry: entry.slot_id)
            call_texture_entries.extend(rec_entries)
            for entry in rec_entries:
                dedupe_key = (entry.slot_id, entry.tex_hash)
                if dedupe_key in seen_texture_keys:
                    continue
                seen_texture_keys.add(dedupe_key)
                texture_entries.append(entry)
        texture_entries.sort(key=lambda entry: (entry.slot_id, entry.tex_hash))

        return _SubmeshBuild(
            unique_str=unique_str,
            ib_hash=str(record.ib.hash).lower(),
            index_count=int(draw.index_count),
            first_index=int(draw.first_index or 0),
            vertex_count=vertex_count,
            rebased_indices=rebased_indices,
            slot_datas=slot_datas,
            vs_hash_list=vs_hash_list,
            texture_entries=texture_entries,
            call_texture_entries=call_texture_entries,
            has_blend=has_blend,
        )

    @staticmethod
    def _element_name(semantic: BufferSemantic) -> str:
        '''与 D3D11Element.get_indexed_semantic_name 保持一致的元素名'''
        semantic_name = semantic.abstract.enum.value
        if semantic.abstract.index > 0:
            return semantic_name + str(semantic.abstract.index)
        return semantic_name

    @staticmethod
    def _unknown_placeholder_format(stride: int) -> DXGIFormat:
        '''为 UNKNOWN 占位元素选择能整除 ByteWidth 的 UINT 格式 (保证导入端 numpy dtype 解析可用)'''
        if stride % 4 == 0:
            return DXGIFormat.R32_UINT
        if stride % 2 == 0:
            return DXGIFormat.R16_UINT
        return DXGIFormat.R8_UINT

    def _record_draw_identity(self, record: _DrawRecord) -> tuple:
        '''绘制的几何体标识: (BaseVertexLocation, vb0-2 资源 hash)，用于识别同组内的冲突绘制'''
        identity = [int(getattr(record.draw_call, "first_vertex", 0) or 0)]
        for slot_id in _VB_IMPORT_SLOTS:
            vb = record.shader_call.model_resources.get_by_slot("vb" + str(slot_id))
            if vb is None or not isinstance(vb, VertexBuffer):
                identity.append(None)
            else:
                identity.append(str(vb.hash))
        return tuple(identity)

    @staticmethod
    def _collect_call_vs_hashes(record: _DrawRecord) -> list[str]:
        '''优先取本绘制显式设置的 VS，其次从 IB dump 文件名解析的着色器信息'''
        vs_hashes: list[str] = []
        for shader in (record.shader_call.shaders or []):
            if shader.type == ShaderType.Vertex and shader.hash:
                vs_hashes.append(str(shader.hash))
        usage_descriptor = record.ib.usage_descriptor
        if usage_descriptor is not None and usage_descriptor.shaders:
            vs_hash = usage_descriptor.shaders.get(ShaderType.Vertex)
            if vs_hash:
                vs_hashes.append(str(vs_hash))
        return vs_hashes

    def _collect_call_textures(self, record: _DrawRecord) -> list[_TextureEntry]:
        '''收集绘制时绑定的 ps-t* 贴图资源'''
        entries: list[_TextureEntry] = []
        textures = record.shader_call.model_resources.textures
        for slot, resource in textures.items():
            if slot.shader_type != ShaderType.Pixel:
                continue
            if slot.slot_type != SlotType.Texture:
                continue
            if resource.bin_path_deduped is None:
                continue
            src_path = Path(resource.bin_path_deduped)
            if src_path.suffix.lower() not in _TEXTURE_SUFFIXES:
                continue
            if not resource.hash or str(resource.hash).startswith("UNKNOWN_"):
                continue

            format_name = "UNKNOWN"
            data_descriptor = resource.data_descriptor
            if data_descriptor is not None and getattr(data_descriptor, "data_format", None):
                format_name = str(data_descriptor.data_format)

            # call_id: 贴图 dump 文件名中的调用号 (同 hash 贴图跨绘制去重后可能指向别的绘制)；
            # draw_call_id: 本次绘制自身的调用号 (log.txt call id, 供逐绘制分组使用)
            call_id = record.shader_call.id
            usage_descriptor = resource.usage_descriptor
            if usage_descriptor is not None and getattr(usage_descriptor, "call_id", None) is not None:
                call_id = int(usage_descriptor.call_id)

            entries.append(_TextureEntry(
                slot_id=int(slot.slot_id),
                tex_hash=str(resource.hash).lower(),
                format_name=format_name,
                call_id=call_id,
                draw_call_id=int(record.shader_call.id),
                src_path=src_path,
            ))
        return entries

    # ------------------------------------------------------------------
    # 写盘
    # ------------------------------------------------------------------

    def _write_submesh(
        self,
        build: _SubmeshBuild,
        workspace_folder: str,
        gametype_name: str,
        copy_textures: bool,
        warnings: list[str],
        merged_component: dict | None = None,
    ) -> str:
        unique_str = build.unique_str
        type_folder = os.path.join(workspace_folder, unique_str, "TYPE_" + gametype_name)
        os.makedirs(type_folder, exist_ok=True)

        # 索引文件: 重定基后的索引, R32_UINT 小端裸数据
        index_filename = unique_str + "-Index.ib"
        with open(os.path.join(type_folder, index_filename), 'wb') as f:
            f.write(build.rebased_indices.tobytes())

        # 分类缓冲区文件
        category_buffer_list = []
        for slot_data in build.slot_datas:
            category_filename = unique_str + "-" + slot_data.category + ".buf"
            with open(os.path.join(type_folder, category_filename), 'wb') as f:
                f.write(slot_data.raw_bytes)

            d3d11_element_list = []
            for semantic in slot_data.layout.semantics:
                d3d11_element_list.append({
                    "SemanticName": semantic.abstract.enum.value,
                    "SemanticIndex": int(semantic.abstract.index),
                    "Format": semantic.format.format,
                    "ByteWidth": int(semantic.stride),
                    "ExtractSlot": "vb" + str(slot_data.slot_id),
                    "ExtractTechnique": "",
                    "Category": slot_data.category,
                })

            category_buffer_list.append({
                "FileName": category_filename,
                "Type": "Normal",
                "D3D11ElementList": d3d11_element_list,
            })

        # 贴图
        if copy_textures and build.texture_entries:
            self._write_textures(
                build,
                type_folder,
                warnings,
                merged_component_scoped=merged_component is not None,
            )

        # SubmeshJson
        category_hash = {slot_data.category: slot_data.vb_hash for slot_data in build.slot_datas}
        vertex_limit_vb = ""
        for slot_data in build.slot_datas:
            if slot_data.slot_id == 0:
                vertex_limit_vb = slot_data.vb_hash
                break
        if not vertex_limit_vb:
            vertex_limit_vb = build.slot_datas[0].vb_hash

        submesh_json_dict = {
            "GamePreset": "EFMI",
            "WorkGameType": gametype_name,
            "GPU-PreSkinning": bool(build.has_blend),
            "CategoryDrawCategoryMap": {slot_data.category: slot_data.category for slot_data in build.slot_datas},
            "CategoryHash": category_hash,
            "VertexLimitVB": vertex_limit_vb,
            "VSHashList": list(build.vs_hash_list),
            "OriginalVertexCount": int(build.vertex_count),
            "PartName": str(build.first_index),
            "TextureMarkUpInfoList": [],
            "IndexBufferList": [
                {
                    "DXGI_FORMAT": "R32_UINT",
                    "FileName": index_filename,
                }
            ],
            "CategoryBufferList": category_buffer_list,
        }
        if merged_component is not None:
            submesh_json_dict["EFMIMergedSkeleton"] = make_submesh_metadata(merged_component)

        json_path = os.path.join(type_folder, unique_str + ".json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(submesh_json_dict, f, ensure_ascii=False, indent=4)

        return json_path

    def _write_textures(
        self,
        build: _SubmeshBuild,
        type_folder: str,
        warnings: list[str],
        merged_component_scoped: bool = False,
    ):
        '''
        复制贴图到 TYPE_ 目录:
        - 每个 hash 只保留一份文件, 命名 "t-<hash>-<FORMAT><后缀>"
          (单令牌文件名, 不含空格 / 等号: Slot 风格标记会用文件名主干生成 ini 资源名
          "Resource-<主干>"，3dmigoto 的 ini 解析器无法处理含空格或 '=' 的资源标识符。
          文件名中刻意不含槽位号: 同一 hash 的贴图在不同绘制中可能绑定到不同槽位,
          槽位信息以 TextureSlots.json 的 calls 分组为准, 避免文件名误导)
        - 写入 TextureSlots.json (v2, 含 slots 槽位聚合与 calls 逐绘制分组) 供贴图标记面板使用
        - 启发式选择漫反射贴图并额外复制为 "<unique_str>-DiffuseMap.dds"
        '''
        # 按 hash 去重决定落盘文件名
        hash_to_filename: dict[str, str] = {}
        for entry in build.texture_entries:
            existing_filename = hash_to_filename.get(entry.tex_hash)
            if existing_filename is None:
                filename = (
                    "t-" + entry.tex_hash
                    + "-" + self._sanitize_format_token(entry.format_name)
                    + entry.src_path.suffix.lower()
                )
                hash_to_filename[entry.tex_hash] = filename
                entry.filename = filename
            else:
                entry.filename = existing_filename

        # 逐绘制条目引用同 hash 的去重后文件名
        for entry in build.call_texture_entries:
            if not entry.filename:
                entry.filename = hash_to_filename.get(entry.tex_hash, "")

        # 复制文件
        for entry in build.texture_entries:
            dst_path = os.path.join(type_folder, entry.filename)
            if os.path.exists(dst_path):
                continue
            try:
                shutil.copy2(entry.src_path, dst_path)
            except OSError as e:
                warnings.append(
                    build.unique_str + " 贴图复制失败 " + entry.src_path.name + ": " + repr(e)
                )

        # TextureSlots.json v2
        # "slots": 按槽位聚合 (每槽位内按 hash 去重后的贴图列表)
        slot_map: dict[str, list] = {}
        for entry in build.texture_entries:
            slot_key = "ps-t" + str(entry.slot_id)
            width, height = self._get_texture_dimensions(entry.src_path)
            slot_map.setdefault(slot_key, []).append({
                "hash": entry.tex_hash,
                "filename": entry.filename,
                "format": entry.format_name,
                "call_id": entry.call_id,
                "width": width,
                "height": height,
            })
        slot_map = {slot_key: slot_map[slot_key] for slot_key in sorted(slot_map.keys(), key=self._slot_sort_key)}

        # "calls": 按绘制调用分组 (每个绘制自己的完整 ps-t 绑定, 槽位顺序)
        call_map: dict[str, list] = {}
        for entry in build.call_texture_entries:
            call_key = format(int(entry.draw_call_id), "06d")
            width, height = self._get_texture_dimensions(entry.src_path)
            call_map.setdefault(call_key, []).append({
                "slot": "ps-t" + str(entry.slot_id),
                "hash": entry.tex_hash,
                "filename": entry.filename,
                "format": entry.format_name,
                "width": width,
                "height": height,
            })
        call_map = {call_key: call_map[call_key] for call_key in sorted(call_map.keys())}

        texture_slots_json = {
            "version": 2,
            "slots": slot_map,
            "calls": call_map,
        }
        if merged_component_scoped:
            texture_slots_json["EFMIMergedComponentScoped"] = True
            texture_slots_json["EFMITextureFilter"] = {
                "exclude_extensions": ["jpg", "buf"],
                "min_file_size": _EFMI_MERGED_TEXTURE_MIN_FILE_SIZE,
                "square_only": True,
            }
        with open(os.path.join(type_folder, "TextureSlots.json"), 'w', encoding='utf-8') as f:
            json.dump(texture_slots_json, f, ensure_ascii=False, indent=4)

        if merged_component_scoped:
            # 骨骼合并工作空间直接准备与普通流程同名的独立贴图，Blender 材质导入、
            # 用户后续编辑以及自动导出都读取这些 canonical 文件。
            from ..common.efmi_merged_texture import (
                copy_merged_auto_textures,
                resolve_merged_auto_textures,
            )
            merged_bindings = resolve_merged_auto_textures(
                texture_folder=type_folder,
                unique_str=build.unique_str,
            )
            copy_merged_auto_textures(merged_bindings, type_folder)
            return

        # 漫反射启发式: 最小槽位的 sRGB 贴图优先, 否则最小槽位贴图
        diffuse_entry = None
        srgb_entries = [entry for entry in build.texture_entries if "SRGB" in entry.format_name.upper()]
        if srgb_entries:
            diffuse_entry = min(srgb_entries, key=lambda entry: entry.slot_id)
        elif build.texture_entries:
            diffuse_entry = min(build.texture_entries, key=lambda entry: entry.slot_id)

        if diffuse_entry is not None:
            diffuse_path = os.path.join(type_folder, build.unique_str + "-DiffuseMap.dds")
            if not os.path.exists(diffuse_path):
                try:
                    shutil.copy2(diffuse_entry.src_path, diffuse_path)
                except OSError as e:
                    warnings.append(build.unique_str + " 漫反射贴图复制失败: " + repr(e))

    def _get_texture_dimensions(self, src_path: Path) -> tuple[int, int]:
        '''读取贴图文件头部的宽高 (复用 vendored 的 DDS/JPEG 解析, 带缓存)，失败时返回 (0, 0)'''
        cache_key = str(src_path)
        dimensions = self._texture_dimensions_cache.get(cache_key)
        if dimensions is None:
            suffix = src_path.suffix.lower()
            try:
                if suffix == '.dds':
                    dimensions = TextureFilter.get_dds_dimensions(src_path)
                elif suffix == '.jpg':
                    dimensions = TextureFilter.get_jpg_dimensions(src_path)
                else:
                    dimensions = (0, 0)
                dimensions = (int(dimensions[0]), int(dimensions[1]))
            except Exception:
                dimensions = (0, 0)
            self._texture_dimensions_cache[cache_key] = dimensions
        return dimensions

    @staticmethod
    def _sanitize_format_token(format_name: str) -> str:
        '''格式令牌只保留字母数字与下划线 (文件名必须是单令牌, 供 ini 资源名与文件名解析使用)'''
        token = re.sub(r"[^A-Za-z0-9_]", "_", str(format_name or ""))
        return token or "UNKNOWN"

    @staticmethod
    def _slot_sort_key(slot_key: str) -> int:
        try:
            return int(slot_key.replace("ps-t", ""))
        except ValueError:
            return 9999

    # ------------------------------------------------------------------
    # 工作空间根文件
    # ------------------------------------------------------------------

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
        unique_strs: list[str],
        gametype_name: str,
        draw_ibs: list[str],
        aliases: dict[str, str] | None = None,
    ):
        # 规范化别名映射: {ib_hash(小写): 别名}，忽略空键 / 空值
        normalized_aliases: dict[str, str] = {}
        if aliases:
            for ib_hash, alias in aliases.items():
                if not ib_hash or not str(ib_hash).strip():
                    continue
                if alias is None or not str(alias).strip():
                    continue
                normalized_aliases[str(ib_hash).strip().lower()] = str(alias).strip()

        # Import.json: {unique_str: gametype_name}, 保留已有条目
        import_json_path = os.path.join(workspace_folder, "Import.json")
        import_json = self._load_json_file(import_json_path, {})
        if not isinstance(import_json, dict):
            import_json = {}
        for unique_str in unique_strs:
            import_json[unique_str] = gametype_name
        with open(import_json_path, 'w', encoding='utf-8') as f:
            json.dump(import_json, f, ensure_ascii=False, indent=4)

        # Config.json: [{"DrawIB": ..., "Alias": ...}], 追加缺失的 DrawIB；
        # 显式提供了别名时，同 DrawIB 的已有条目也会更新 Alias
        config_json_path = os.path.join(workspace_folder, "Config.json")
        config_json = self._load_json_file(config_json_path, [])
        if not isinstance(config_json, list):
            config_json = []
        existing_entries = {
            str(item.get("DrawIB", "")).lower(): item
            for item in config_json
            if isinstance(item, dict)
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

def _standalone_main(argv: list[str]):
    import argparse

    parser = argparse.ArgumentParser(description="LoyalTools DrawIB 提取自测工具")
    parser.add_argument("dump_folder", help="3dmigoto 帧分析 Dump 目录 (含 log.txt)")
    parser.add_argument("ib_hashes", nargs='*', help="要提取的 DrawIB hash 列表, 留空则只列出所有 DrawIB")
    parser.add_argument("--workspace", default="", help="输出工作空间目录 (提取时必填)")
    parser.add_argument("--gametype", default="GPU-EFMI", help="数据类型名称 (默认 GPU-EFMI)")
    parser.add_argument("--no-textures", action='store_true', help="不复制贴图")
    parser.add_argument("--alias", action='append', default=[],
                        help="DrawIB 别名, 格式 ib_hash=别名 (可多次指定)")
    parser.add_argument("--verbose", action='store_true', help="输出详细日志")
    args = parser.parse_args(argv)

    aliases: dict[str, str] = {}
    for alias_pair in args.alias:
        if '=' not in alias_pair:
            print("错误: --alias 参数格式应为 ib_hash=别名, 实际: " + alias_pair)
            sys.exit(2)
        ib_hash, alias = alias_pair.split('=', 1)
        aliases[ib_hash] = alias

    extractor = DumpWorkspaceExtractor(args.dump_folder, verbose=args.verbose)

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

    if not args.ib_hashes:
        return

    if not args.workspace:
        print("错误: 提取时必须通过 --workspace 指定输出目录。")
        sys.exit(2)

    result = extractor.extract(
        ib_hashes=args.ib_hashes,
        workspace_folder=args.workspace,
        gametype_name=args.gametype,
        copy_textures=not args.no_textures,
        aliases=aliases or None,
    )

    print("提取完成, 工作空间: " + result.workspace_folder)
    for json_path in result.json_paths:
        print("  " + json_path)
    for warning in result.warnings:
        print("警告: " + warning)


if __name__ == '__main__':
    # 正常情况下不会到达这里: 以脚本方式运行时,
    # 模块顶部的合成包引导逻辑已经完成委托并退出。
    _standalone_main(sys.argv[1:])
