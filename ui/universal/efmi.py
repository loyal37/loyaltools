from ...blueprint.model import BluePrintModel
from ...common.submesh_model import SubMeshModel
from ...common.drawib_model import DrawIBModel
from dataclasses import dataclass,field
from ...common.global_config import GlobalConfig
from ...common.global_properties import GlobalProterties
from ...blueprint.export_helper import BlueprintExportHelper

from ...common.buffer_export_helper import BufferExportHelper
from ...common.global_key_count_helper import GlobalKeyCountHelper
from ...common.m_ini_helper import M_IniHelper
from ...common.m_ini_helper_gui import M_IniHelperGUI
from ...common.m_ini_builder import M_IniBuilder,M_IniSection, M_SectionType
from .export_helper import ExportHelper
from ...utils.timer_utils import TimerUtils
from ...common.efmi_merged_skeleton import load_profile

import os
import re
import shutil
import bpy

@dataclass
class ExportEFMI:

    blueprint_model:BluePrintModel

    submesh_model_list:list[SubMeshModel] = field(default_factory=list,init=False)
    drawib_model_list:list[DrawIBModel] = field(default_factory=list,init=False)

    def _get_tagged_merged_skeleton_unique_strs(self) -> set[str]:
        tagged_unique_strs = set()
        objects = getattr(getattr(bpy, "data", None), "objects", None)
        get_object = getattr(objects, "get", None)
        if not callable(get_object):
            return tagged_unique_strs

        for draw_call_model in self.blueprint_model.ordered_draw_obj_data_model_list:
            obj = get_object(draw_call_model.get_blender_obj_name())
            if obj is not None and bool(obj.get("LoyalTools:EFMIMergedSkeleton", False)):
                tagged_unique_strs.add(draw_call_model.get_unique_str())
        return tagged_unique_strs

    def __post_init__(self):
        self.cross_ib_info_dict = self.blueprint_model.cross_ib_info_dict
        self.cross_ib_method_dict = self.blueprint_model.cross_ib_method_dict
        self.has_cross_ib = self.blueprint_model.has_cross_ib
        self.cross_ib_mapping_objects = self.blueprint_model.cross_ib_mapping_objects
        self.cross_ib_vb_condition_mapping = self.blueprint_model.cross_ib_vb_condition_mapping
        self.cross_ib_source_to_target_dict = self.blueprint_model.cross_ib_source_to_target_dict
        self.cross_ib_object_vb_condition = self.blueprint_model.cross_ib_object_vb_condition
        self.cross_ib_target_info = self.blueprint_model.cross_ib_target_info
        self.cross_ib_match_mode = self.blueprint_model.cross_ib_match_mode
        self.cross_ib_object_names = self.blueprint_model.cross_ib_object_names

        cross_ib_methods = {
            method for method in self.cross_ib_method_dict.values() if method
        }
        explicit_merged_skeleton = 'MERGED_SKELETON' in cross_ib_methods
        if explicit_merged_skeleton and cross_ib_methods != {'MERGED_SKELETON'}:
            raise ValueError(
                "同一次 EFMI 导出不能混用“一般跨 IB”和“骨骼合并”。"
                "请把蓝图中的 Cross IB 节点统一切换为同一种方式。"
            )

        tree = getattr(self.blueprint_model, "_tree", None)
        tree_merged_skeleton = bool(
            tree.get("LoyalTools:EFMIMergedSkeleton", False)
            if tree is not None and callable(getattr(tree, "get", None))
            else False
        )
        tagged_merged_unique_strs = self._get_tagged_merged_skeleton_unique_strs()
        automatic_merged_skeleton = (
            tree_merged_skeleton or bool(tagged_merged_unique_strs)
        )
        if (
            automatic_merged_skeleton
            and cross_ib_methods
            and cross_ib_methods != {'MERGED_SKELETON'}
        ):
            raise ValueError(
                "当前蓝图包含骨骼合并物体，但 Cross IB 节点仍为“一般跨 IB”。"
                "请将节点切换为“骨骼合并”，或改用普通流程导入的物体。"
            )

        self.merged_skeleton_mode = (
            explicit_merged_skeleton or automatic_merged_skeleton
        )
        self.merged_skeleton_profile = None
        merged_gpu_unique_strs = None
        if self.merged_skeleton_mode:
            self.merged_skeleton_profile = load_profile(
                GlobalConfig.path_workspace_folder(), required=True
            )
            merged_gpu_unique_strs = {
                component["unique_str"]
                for component in self.merged_skeleton_profile["components"]
                if not component["cpu_posed"]
            }
            exported_unique_strs = {
                draw_call_model.get_unique_str()
                for draw_call_model in self.blueprint_model.ordered_draw_obj_data_model_list
            }
            unexpected_unique_strs = sorted(
                exported_unique_strs - merged_gpu_unique_strs
            )
            if unexpected_unique_strs:
                raise ValueError(
                    "蓝图包含不属于当前骨骼合并 profile 的 GPU 子网格: "
                    + ", ".join(unexpected_unique_strs)
                )
            detection_reason = []
            if explicit_merged_skeleton:
                detection_reason.append("Cross IB 节点")
            if tree_merged_skeleton:
                detection_reason.append("蓝图标记")
            if tagged_merged_unique_strs:
                detection_reason.append("物体标记")
            print(
                "[EFMI骨骼合并] 已自动启用，依据: "
                + "、".join(detection_reason)
            )

        self.submesh_model_list = ExportHelper.parse_submesh_model_list_from_blueprint_model(
            self.blueprint_model,
            efmi_merged_skeleton=self.merged_skeleton_mode,
            efmi_merged_skeleton_unique_strs=merged_gpu_unique_strs,
        )
        # EFMI 直接复用已经解析好的 SubMeshModel，避免同一轮导出把几何解析做两遍。
        self.drawib_model_list = ExportHelper.parse_drawib_model_list_from_submesh_model_list(
            submesh_model_list=self.submesh_model_list,
            combine_ib=False,
        )
        print("SubMeshModel列表初始化完成，共有 " + str(len(self.submesh_model_list)) + " 个SubMeshModel")

        print(f"[CrossIB EFMI] 初始化: has_cross_ib={self.has_cross_ib}")
        print(f"[CrossIB EFMI] cross_ib_info_dict={self.cross_ib_info_dict}")
        print(f"[CrossIB EFMI] cross_ib_object_names={self.cross_ib_object_names}")
        print(f"[CrossIB EFMI] cross_ib_mapping_objects={self.cross_ib_mapping_objects}")

    def generate_buffer_files(self):
        buf_output_folder = GlobalConfig.path_generatemod_buffer_folder()

        for submesh_model in self.submesh_model_list:
            print("ExportEFMI: 导出SubMeshModel，Unique标识: " + submesh_model.unique_str)

            ib_filename = submesh_model.unique_str + "-Index.buf"
            ib_filepath = os.path.join(buf_output_folder, ib_filename)
            BufferExportHelper.write_buf_ib_r32_uint(submesh_model.ib, ib_filepath)

            for category, category_buf in submesh_model.category_buffer_dict.items():
                category_buf_filename = submesh_model.unique_str + "-" + category + ".buf"
                category_buf_filepath = os.path.join(buf_output_folder, category_buf_filename)
                with open(category_buf_filepath, 'wb') as f:
                    category_buf.tofile(f)

    def _get_submesh_ib_key(self, submesh_model):
        if self.cross_ib_match_mode == 'INDEX_COUNT':
            return f"indexcount_{submesh_model.match_index_count}"
        else:
            return f"{submesh_model.match_draw_ib}_{submesh_model.match_first_index}"

    def _get_all_cross_ib_identifiers(self):
        all_identifiers = set()

        if self.cross_ib_match_mode == 'INDEX_COUNT':
            for source_key, target_key_list in self.cross_ib_info_dict.items():
                if source_key.startswith('indexcount_'):
                    index_count = source_key.replace('indexcount_', '')
                    all_identifiers.add(index_count)
                for target_key in target_key_list:
                    if target_key.startswith('indexcount_'):
                        index_count = target_key.replace('indexcount_', '')
                        all_identifiers.add(index_count)

            for submesh_model in self.submesh_model_list:
                if submesh_model.match_index_count:
                    all_identifiers.add(submesh_model.match_index_count)
        else:
            for source_ib, target_ib_list in self.cross_ib_info_dict.items():
                source_hash = source_ib.split("_")[0]
                all_identifiers.add(source_hash)
                for target_ib in target_ib_list:
                    target_hash = target_ib.split("_")[0]
                    all_identifiers.add(target_hash)

            for drawib_model in self.drawib_model_list:
                all_identifiers.add(drawib_model.draw_ib)

        return all_identifiers

    def _get_vb_condition_for_mapping(self, source_ib_key, target_ib_key, condition_type='source'):
        mapping_key = (source_ib_key, target_ib_key)
        condition_info = self.cross_ib_vb_condition_mapping.get(mapping_key, {})
        if condition_type == 'source':
            return condition_info.get('source', "if vs == 200 || vs == 201")
        else:
            return condition_info.get('target', "if vs == 202 || vs == 203")

    def _get_vb_condition_for_object(self, obj_name, source_ib_key, target_ib_key, condition_type='source'):
        object_mapping_key = (obj_name, source_ib_key, target_ib_key)
        condition_info = self.cross_ib_object_vb_condition.get(object_mapping_key, {})
        if condition_type == 'source':
            return condition_info.get('source', "if vs == 200 || vs == 201")
        else:
            return condition_info.get('target', "if vs == 202 || vs == 203")

    # 源块分支路由 (依据 fix-efmi-cross-ib skill / 实测有效Mod结构):
    # 200 = 捕获/prepass通道 -> ExtractCaptureCB1 提取, 绑到 vs-cb1
    # 201 = 常规蒙皮/简单阴影通道 -> ExtractCB1 提取, 绑到 vs-cb2
    # 其余编号(含204未知通道)保守沿用旧行为 ExtractCB1 + vs-cb1
    _CROSS_IB_SOURCE_ROUTING = {
        200: ("CustomShader_ExtractCaptureCB1", "vs-cb1"),
        201: ("CustomShader_ExtractCB1", "vs-cb2"),
    }

    # 仅由 "if vs == N [|| vs == M ...]" 组成的简单条件才可安全拆分
    _SIMPLE_VS_CONDITION_PATTERN = re.compile(
        r'^\s*if\s+vs\s*==\s*\d+(?:\s*\|\|\s*vs\s*==\s*\d+)*\s*$'
    )

    def _append_cross_ib_source_branches(self, lines, vb_condition, source_identifier, drawindexed_str_list):
        '''
        生成源块的跨IB分支。
        依据 fix-efmi-cross-ib skill: 合并的 "if vs == 200 || vs == 201" 分支会让
        两条通道共用同一提取shader与CB槽位，导致本体消失/变黑/丢阴影，
        必须按 filter 编号拆成独立分支 (200走CaptureCB1+vs-cb1, 201走CB1+vs-cb2)，
        每个分支内完整复制跨IB绘制命令。
        无法解析的自定义条件保留旧的单分支行为。
        '''
        drawindexed_str_list = [line for line in drawindexed_str_list if line.strip()]

        vs_numbers = []
        if vb_condition and self._SIMPLE_VS_CONDITION_PATTERN.match(vb_condition):
            seen = set()
            for match in re.findall(r'vs\s*==\s*(\d+)', vb_condition):
                vs_number = int(match)
                if vs_number not in seen:
                    seen.add(vs_number)
                    vs_numbers.append(vs_number)

        if vs_numbers:
            for vs_number in vs_numbers:
                extract_shader, cb_slot = self._CROSS_IB_SOURCE_ROUTING.get(
                    vs_number, ("CustomShader_ExtractCB1", "vs-cb1")
                )
                lines.append(f"if vs == {vs_number}")
                lines.append(f"    run = {extract_shader}")
                lines.append(f"    cs-t2 = ResourceID_{source_identifier}")
                lines.append(f"    run = CustomShader_RecordBones_{source_identifier}")
                lines.append(f"    run = CustomShader_RedirectCB1_{source_identifier}")
                lines.append(f"    vs-t0 = ResourceFakeT0_SRV_{source_identifier}")
                lines.append(f"    {cb_slot} = ResourceFakeCB1_{source_identifier}")
                lines.append(";所有需要跨 Ib 的物体引用")
                lines.extend(drawindexed_str_list)
                lines.append("endif")
        else:
            # 自定义/不可解析条件: 保留单分支旧行为 (ExtractCB1 + vs-cb1)
            # 条件为空 (源槽位全部未勾选) 时不能输出孤立的 endif，
            # 此时跳过 if/endif 包裹，绘制命令无条件执行
            has_condition = bool(vb_condition and vb_condition.strip())
            if has_condition:
                lines.append(vb_condition)
            lines.append("    run = CustomShader_ExtractCB1")
            lines.append(f"    cs-t2 = ResourceID_{source_identifier}")
            lines.append(f"    run = CustomShader_RecordBones_{source_identifier}")
            lines.append(f"    run = CustomShader_RedirectCB1_{source_identifier}")
            lines.append(f"    vs-t0 = ResourceFakeT0_SRV_{source_identifier}")
            lines.append(f"    vs-cb1 = ResourceFakeCB1_{source_identifier}")
            lines.append(";所有需要跨 Ib 的物体引用")
            lines.extend(drawindexed_str_list)
            if has_condition:
                lines.append("endif")

    def _split_drawcalls_by_cross_ib(self, drawcall_model_list, source_ib_key=None, target_ib_key=None):
        cross_ib_drawcalls = []
        non_cross_ib_drawcalls = []

        cross_ib_mapping_objects = self.cross_ib_mapping_objects

        for drawcall_model in drawcall_model_list:
            obj_name = drawcall_model.obj_name if hasattr(drawcall_model, 'obj_name') else str(drawcall_model)

            is_cross_ib = False
            if source_ib_key:
                if target_ib_key:
                    mapping_key = (source_ib_key, target_ib_key)
                    if mapping_key in cross_ib_mapping_objects:
                        if obj_name in cross_ib_mapping_objects[mapping_key]:
                            is_cross_ib = True
                else:
                    for (src_key, tgt_key), obj_names in cross_ib_mapping_objects.items():
                        if src_key == source_ib_key and obj_name in obj_names:
                            is_cross_ib = True
                            break
            else:
                if obj_name in self.cross_ib_object_names:
                    is_cross_ib = True

            if is_cross_ib:
                cross_ib_drawcalls.append(drawcall_model)
            else:
                non_cross_ib_drawcalls.append(drawcall_model)

        return cross_ib_drawcalls, non_cross_ib_drawcalls

    def _group_drawcalls_by_cross_ib_target(self, drawcall_model_list, source_ib_key, target_ib_keys):
        grouped = {}
        cross_ib_mapping_objects = self.cross_ib_mapping_objects

        for drawcall_model in drawcall_model_list:
            obj_name = drawcall_model.obj_name if hasattr(drawcall_model, 'obj_name') else str(drawcall_model)

            for target_ib_key in target_ib_keys:
                mapping_key = (source_ib_key, target_ib_key)
                if mapping_key in cross_ib_mapping_objects:
                    if obj_name in cross_ib_mapping_objects[mapping_key]:
                        vb_condition = self._get_vb_condition_for_object(obj_name, source_ib_key, target_ib_key, 'source')
                        group_key = (target_ib_key, vb_condition)
                        if group_key not in grouped:
                            grouped[group_key] = []
                        grouped[group_key].append(drawcall_model)
                        break

        return grouped

    def _generate_cross_ib_block_for_source(self, source_identifier, drawcall_model_list, source_ib_key=None, target_ib_key=None):
        lines = []

        cross_ib_drawcalls, non_cross_ib_drawcalls = self._split_drawcalls_by_cross_ib(
            drawcall_model_list,
            source_ib_key=source_ib_key
        )

        target_ib_keys = self.cross_ib_source_to_target_dict.get(source_ib_key, [])
        if target_ib_key and target_ib_key not in target_ib_keys:
            target_ib_keys.append(target_ib_key)

        grouped_drawcalls = self._group_drawcalls_by_cross_ib_target(cross_ib_drawcalls, source_ib_key, target_ib_keys)

        for (tgt_ib_key, vb_condition), objects in grouped_drawcalls.items():
            if not objects:
                continue

            lines.append(";跨 iB 区域")
            self._append_cross_ib_source_branches(
                lines, vb_condition, source_identifier,
                M_IniHelper.get_drawindexed_instanced_str_list(objects),
            )

        lines.append(";不需要跨 Ib 的物体引用")

        if non_cross_ib_drawcalls:
            drawindexed_str_list = M_IniHelper.get_drawindexed_instanced_str_list(non_cross_ib_drawcalls)
            for drawindexed_str in drawindexed_str_list:
                if drawindexed_str.strip():
                    lines.append(drawindexed_str)

        lines.append("")
        lines.append(f"post vs-cb1 = null")
        lines.append(f"post vs-cb2 = null")
        lines.append(f"post vs-t0 = null")
        lines.append(f"post cs-t2 = null")

        return lines

    def _add_cross_ib_present_section(self, ini_builder):
        if not self.has_cross_ib:
            return

        present_section = M_IniSection(M_SectionType.CrossIBPresent)
        present_section.append(";特殊追加固定区域")
        present_section.append("[Present]")
        present_section.append("ResourcePrev_SRV = ResourceFakeT0_SRV")
        present_section.new_line()

        present_section.append("[ResourceDumpedCB1_UAV]")
        present_section.append("type = RWStructuredBuffer")
        present_section.append("stride = 16")
        present_section.append("array = 4096")
        present_section.new_line()

        present_section.append("[ResourceDumpedCB1_SRV]")
        present_section.append("type = Buffer")
        present_section.append("stride = 16")
        present_section.append("array = 4096")
        present_section.new_line()

        all_identifiers = self._get_all_cross_ib_identifiers()

        for identifier in sorted(all_identifiers):
            present_section.append(f"[ResourceFakeCB1_UAV_{identifier}]")
            present_section.append("type = RWStructuredBuffer")
            present_section.append("stride = 16")
            present_section.append("array = 4096")
            present_section.new_line()

            present_section.append(f"[ResourceFakeCB1_{identifier}]")
            present_section.append("type = Buffer")
            present_section.append("stride = 16")
            present_section.append("format = R32G32B32A32_UINT")
            present_section.append("array = 4096")
            present_section.new_line()

            present_section.append(f"[ResourceFakeT0_UAV_{identifier}]")
            present_section.append("type = RWStructuredBuffer")
            present_section.append("stride = 16")
            present_section.append("array = 200000")
            present_section.new_line()

            present_section.append(f"[ResourceFakeT0_SRV_{identifier}]")
            present_section.append("type = StructuredBuffer")
            present_section.append("stride = 16")
            present_section.append("array = 200000")
            present_section.new_line()

        present_section.append("[ResourceFakeT0_UAV]")
        present_section.append("type = RWStructuredBuffer")
        present_section.append("stride = 16")
        present_section.append("array = 200000")
        present_section.new_line()

        present_section.append("[ResourceFakeT0_SRV]")
        present_section.append("type = StructuredBuffer")
        present_section.append("stride = 16")
        present_section.append("array = 200000")
        present_section.new_line()

        present_section.append("[ResourcePrev_SRV]")
        present_section.append("type = StructuredBuffer")
        present_section.append("stride = 16")
        present_section.append("array = 200000")
        present_section.new_line()

        present_section.append("[CustomShader_ExtractCB1]")
        present_section.append("vs = ./res/extract_cb1_vs.hlsl")
        present_section.append("ps = ./res/extract_cb1_ps.hlsl")
        present_section.append("ps-u7 = ResourceDumpedCB1_UAV")
        present_section.append("depth_enable = false")
        present_section.append("blend = ADD SRC_ALPHA INV_SRC_ALPHA")
        present_section.append("cull = none")
        present_section.append("topology = point_list")
        present_section.append("draw = 4096, 0")
        present_section.append("ps-u7 = null")
        present_section.append("ResourceDumpedCB1_SRV = copy ResourceDumpedCB1_UAV")
        present_section.new_line()

        # 200 捕获通道专用提取shader (capture cbuffer 位于 register(b1)，
        # 见 res/extract_capture_cb1_vs.hlsl；依据 fix-efmi-cross-ib skill)
        present_section.append("[CustomShader_ExtractCaptureCB1]")
        present_section.append("vs = ./res/extract_capture_cb1_vs.hlsl")
        present_section.append("ps = ./res/extract_cb1_ps.hlsl")
        present_section.append("ps-u7 = ResourceDumpedCB1_UAV")
        present_section.append("depth_enable = false")
        present_section.append("blend = ADD SRC_ALPHA INV_SRC_ALPHA")
        present_section.append("cull = none")
        present_section.append("topology = point_list")
        present_section.append("draw = 4096, 0")
        present_section.append("ps-u7 = null")
        present_section.append("ResourceDumpedCB1_SRV = copy ResourceDumpedCB1_UAV")
        present_section.new_line()

        for identifier in sorted(all_identifiers):
            present_section.append(f"[CustomShader_RecordBones_{identifier}]")
            present_section.append("cs = ./res/record_bones_cs.hlsl")
            present_section.append("cs-t0 = vs-t0")
            present_section.append("cs-t1 = ResourceDumpedCB1_SRV")
            present_section.append(f"cs-u1 = ResourceFakeT0_UAV_{identifier}")
            present_section.append("dispatch = 12, 1, 1")
            present_section.append("cs-u1 = null")
            present_section.append("cs-t0 = null")
            present_section.append("cs-t1 = null")
            present_section.append(f"ResourceFakeT0_SRV_{identifier} = copy ResourceFakeT0_UAV_{identifier}")
            present_section.new_line()

            present_section.append(f"[CustomShader_RedirectCB1_{identifier}]")
            present_section.append("cs = ./res/redirect_cb1_cs.hlsl")
            present_section.append("cs-t0 = ResourceDumpedCB1_SRV")
            present_section.append(f"ResourceFakeCB1_UAV_{identifier} = copy ResourceDumpedCB1_SRV")
            present_section.append(f"cs-u0 = ResourceFakeCB1_UAV_{identifier}")
            present_section.append("dispatch = 4, 1, 1")
            present_section.append("cs-u0 = null")
            present_section.append("cs-t0 = null")
            present_section.append(f"ResourceFakeCB1_{identifier} = copy ResourceFakeCB1_UAV_{identifier}")
            present_section.new_line()

        # VS Hash -> filter_index 路由基线 (fix-efmi-cross-ib skill hash-routing.md，
        # 与实测有效Mod一致；200=捕获/prepass提取CB1, 201=常规蒙皮/简单阴影提取CB2,
        # 202/203=跨IB回放通道(经vs-cb2回放), 204=已知但暂不处理的通道)
        shader_overrides = [
            ("ShaderOverridevs1000", "f11c7e1dbf876a69", "200"),
            ("ShaderOverridevs1001", "303f45d5266d0369", "201"),
            ("ShaderOverridevs1002", "7b3a141f99cd9b39", "201"),
            ("ShaderOverridevs1003", "1479b2b594b9c91a", "202"),
            ("ShaderOverridevs1004", "c6e55aaa8f4b3218", "202"),
            ("ShaderOverridevs1005", "784f11ae11c97112", "203"),
            ("ShaderOverridevs1006", "f1b10202c73c72c3", "204"),
            ("ShaderOverridevs1007", "12ad3cc5f56f853c", "204"),
            ("ShaderOverridevs1008", "86cb3bc0a3e2e013", "204"),
            ("ShaderOverridevs1009", "906a3976f3e33cfb", "204"),
            ("ShaderOverridevs1010", "0ba16985f9f74f8d", "204"),
            ("ShaderOverridevs1011", "06c94dd56f447210", "204"),
            ("ShaderOverridevs1012", "f47b1f797f5831d0", "204"),
        ]

        for name, hash_val, filter_idx in shader_overrides:
            present_section.append(f"[{name}]")
            present_section.append(f"hash = {hash_val}")
            present_section.append(f"filter_index = {filter_idx}")
            present_section.append("allow_duplicate_hash = overrule")
            present_section.new_line()

        ini_builder.append_section(present_section)

    def _add_cross_ib_resource_id_sections(self, ini_builder):
        if not self.has_cross_ib:
            return

        resource_id_section = M_IniSection(M_SectionType.ResourceID)
        resource_id_section.append(";特殊追加身份证区域")

        all_identifiers = set()

        if self.cross_ib_match_mode == 'INDEX_COUNT':
            for source_key, target_key_list in self.cross_ib_info_dict.items():
                if source_key.startswith('indexcount_'):
                    index_count = source_key.replace('indexcount_', '')
                    all_identifiers.add(index_count)
                for target_key in target_key_list:
                    if target_key.startswith('indexcount_'):
                        index_count = target_key.replace('indexcount_', '')
                        all_identifiers.add(index_count)

            for submesh_model in self.submesh_model_list:
                if submesh_model.match_index_count:
                    all_identifiers.add(submesh_model.match_index_count)
        else:
            for source_ib, target_ib_list in self.cross_ib_info_dict.items():
                source_hash = source_ib.split("_")[0]
                all_identifiers.add(source_hash)
                for target_ib in target_ib_list:
                    target_hash = target_ib.split("_")[0]
                    all_identifiers.add(target_hash)

            for drawib_model in self.drawib_model_list:
                all_identifiers.add(drawib_model.draw_ib)

        sorted_identifiers = sorted(list(all_identifiers))

        for idx, identifier in enumerate(sorted_identifiers):
            resource_id_section.append(f"[ResourceID_{identifier}]")
            resource_id_section.append("type = Buffer")
            resource_id_section.append("format = R32_FLOAT")
            resource_id_section.append(f"data = {idx * 1000}.0")
            resource_id_section.new_line()

        ini_builder.append_section(resource_id_section)

    def _find_source_submesh_by_ib_key(self, source_ib_key):
        for submesh_model in self.submesh_model_list:
            submesh_ib_key = self._get_submesh_ib_key(submesh_model)
            if submesh_ib_key == source_ib_key:
                return submesh_model
        return None

    def _find_source_drawib_by_ib_key(self, source_ib_key):
        if self.cross_ib_match_mode == 'INDEX_COUNT':
            index_count = source_ib_key.replace('indexcount_', '') if source_ib_key.startswith('indexcount_') else None
            if index_count:
                for drawib_model in self.drawib_model_list:
                    for submesh in drawib_model.submesh_model_list:
                        if submesh.match_index_count == index_count:
                            return drawib_model
            return None
        else:
            source_hash = source_ib_key.split("_")[0]
            for drawib_model in self.drawib_model_list:
                if drawib_model.draw_ib == source_hash:
                    return drawib_model
            return None

    @staticmethod
    def _merged_resource_prefix(submesh_model):
        return "Resource_" + submesh_model.unique_str.replace("-", "_")

    def _append_merged_buffer_bindings(self, section, submesh_model, indent=""):
        prefix = self._merged_resource_prefix(submesh_model)
        section.append(indent + "ib = " + prefix + "_Index")
        for category in submesh_model.category_buffer_dict.keys():
            slot = submesh_model.d3d11_game_type.CategoryExtractSlotDict.get(
                category, "unknown_slot"
            )
            section.append(indent + slot + " = " + prefix + "_" + category)
        if "Position" in submesh_model.category_buffer_dict:
            section.append(indent + "vb3 = " + prefix + "_Position")

    def _append_merged_slot_texture_bindings(
        self, section, submesh_model, drawib_model, indent=""
    ):
        if GlobalProterties.forbid_auto_texture_ini() or drawib_model is None:
            return
        texture_markup_info_list = drawib_model.get_submesh_texture_markup_info_list(
            submesh_model
        )
        if GlobalProterties.use_rabbitfx_slot():
            for texture_info in texture_markup_info_list:
                if getattr(texture_info, "mark_type", "") != "Slot":
                    continue
                resource_name = texture_info.get_resource_name()
                if texture_info.mark_name == "DiffuseMap":
                    section.append(indent + "Resource\\RabbitFx\\Diffuse = ref " + resource_name)
                elif texture_info.mark_name == "LightMap":
                    section.append(indent + "Resource\\RabbitFx\\LightMap = ref " + resource_name)
                elif texture_info.mark_name == "NormalMap":
                    section.append(indent + "Resource\\RabbitFx\\NormalMap = ref " + resource_name)
            section.append(indent + "run = CommandList\\RabbitFx\\SetTextures")
            for texture_info in texture_markup_info_list:
                if getattr(texture_info, "mark_type", "") != "Slot":
                    continue
                if texture_info.mark_name in ("DiffuseMap", "LightMap", "NormalMap"):
                    continue
                slot = texture_info.mark_slot
                if slot and not slot.lower().startswith("ps-t"):
                    number = re.search(r"\d+", slot)
                    slot = "ps-t" + (number.group() if number else slot)
                section.append(indent + slot + " = " + texture_info.get_resource_name())
        else:
            for texture_info in texture_markup_info_list:
                if getattr(texture_info, "mark_type", "") != "Slot":
                    continue
                section.append(
                    indent + texture_info.mark_slot + " = " + texture_info.get_resource_name()
                )

    def _append_merged_draw_lines(self, section, drawcalls):
        for line in M_IniHelper.get_drawindexed_str_list(drawcalls):
            section.append(line)

    def _append_merged_incoming_cross_ib_draws(
        self,
        section,
        target_key,
        submesh_by_unique,
        drawib_drawibmodel_dict,
    ):
        for source_key in self.cross_ib_target_info.get(target_key, []):
            source_submesh = self._find_source_submesh_by_ib_key(source_key)
            if source_submesh is None:
                continue
            incoming_drawcalls, _ = self._split_drawcalls_by_cross_ib(
                source_submesh.drawcall_model_list,
                source_ib_key=source_key,
                target_ib_key=target_key,
            )
            if not incoming_drawcalls:
                continue
            section.append(
                "; 骨骼合并跨 IB: "
                + source_submesh.unique_str
                + " -> "
                + target_key
            )
            self._append_merged_buffer_bindings(section, source_submesh)
            self._append_merged_slot_texture_bindings(
                section,
                source_submesh,
                drawib_drawibmodel_dict.get(source_submesh.match_draw_ib),
            )
            self._append_merged_draw_lines(section, incoming_drawcalls)

    def _generate_merged_skeleton_ini_file(self):
        profile = self.merged_skeleton_profile
        if profile is None:
            raise ValueError("缺少 EFMI 骨骼合并工作空间配置。")
        if self.has_cross_ib and self.cross_ib_match_mode != 'INDEX_COUNT':
            raise ValueError("EFMI 骨骼合并 Cross IB 必须使用 IndexCount 识别模式。")

        profile_unique_strs = {
            component["unique_str"] for component in profile["components"]
        }
        unexpected_submeshes = [
            submesh.unique_str
            for submesh in self.submesh_model_list
            if submesh.unique_str not in profile_unique_strs
        ]
        if unexpected_submeshes:
            raise ValueError(
                "蓝图包含不属于当前骨骼合并 profile 的子网格: "
                + ", ".join(unexpected_submeshes)
            )

        cpu_component_keys = {
            "indexcount_" + str(component["index_count"])
            for component in profile["components"]
            if component["cpu_posed"]
        }
        cpu_cross_ib_keys = set()
        for source_key, target_keys in self.cross_ib_info_dict.items():
            if source_key in cpu_component_keys:
                cpu_cross_ib_keys.add(source_key)
            cpu_cross_ib_keys.update(
                target_key for target_key in target_keys
                if target_key in cpu_component_keys
            )
        if cpu_cross_ib_keys:
            raise ValueError(
                "CPU posed 组件不支持骨骼合并跨 IB，自定义映射涉及: "
                + ", ".join(sorted(cpu_cross_ib_keys))
            )

        ini_builder = M_IniBuilder()
        drawib_drawibmodel_dict = {
            drawib_model.draw_ib: drawib_model
            for drawib_model in self.drawib_model_list
        }
        draw_ib_active_index_dict = {
            drawib_model.draw_ib: index
            for index, drawib_model in enumerate(self.drawib_model_list)
        }
        submesh_by_unique = {
            submesh.unique_str: submesh for submesh in self.submesh_model_list
        }
        workspace_name = GlobalConfig.get_workspace_name().replace('"', "'")

        M_IniHelper.generate_hash_style_texture_ini(
            ini_builder=ini_builder,
            drawib_drawibmodel_dict=drawib_drawibmodel_dict,
        )
        self._integrate_object_swap_ini_hook(ini_builder)

        constants = M_IniSection(M_SectionType.Constants)
        constants.append("[Constants]")
        constants.append("global $required_efmi_version = 1.41")
        constants.append("global $object_guid = " + str(profile["object_guid"]))
        constants.append(
            "global $mesh_vertex_count = "
            + str(sum(submesh.vertex_count for submesh in self.submesh_model_list))
        )
        constants.append("global $component_count = " + str(profile["component_count"]))
        constants.append("global $max_instance_count = " + str(profile["max_instance_count"]))
        constants.append("global $bones_count = " + str(profile["bones_count"]))
        constants.append("global $mod_id = -1000")
        constants.append("global $mod_enabled = 0")
        constants.append("global $object_detected = 0")
        constants.append("global $lod_level = 0")
        constants.append("global $merged_skeleton_initialized = 0")
        ini_builder.append_section(constants)

        present = M_IniSection(M_SectionType.Present)
        present.append("[Present]")
        present.append("if $object_detected")
        present.append("    if $mod_enabled")
        present.append("        post $object_detected = 0")
        present.append("    else")
        present.append("        if $mod_id == -1000")
        present.append("            run = CommandListRegisterMod")
        present.append("        endif")
        present.append("    endif")
        present.append("endif")
        ini_builder.append_section(present)

        command_lists = M_IniSection(M_SectionType.CommandList)
        command_lists.append("[CommandListRegisterMod]")
        command_lists.append("$\\EFMIv1\\required_version = $required_efmi_version")
        command_lists.append("$\\EFMIv1\\object_guid = $object_guid")
        command_lists.append("Resource\\EFMIv1\\ModName = ref ResourceModName")
        command_lists.append("Resource\\EFMIv1\\ModAuthor = ref ResourceModAuthor")
        command_lists.append("Resource\\EFMIv1\\ModDesc = ref ResourceModDesc")
        command_lists.append("Resource\\EFMIv1\\ModLink = ref ResourceModLink")
        command_lists.append("Resource\\EFMIv1\\ModLogo = ref ResourceModLogo")
        command_lists.append("run = CommandList\\EFMIv1\\RegisterMod")
        command_lists.append("$mod_id = $\\EFMIv1\\mod_id")
        command_lists.append("if $mod_id >= 0")
        command_lists.append("    $mod_enabled = 1")
        command_lists.append("endif")
        command_lists.new_line()

        for component in profile["components"]:
            component_id = component["component_id"]
            submesh = submesh_by_unique.get(component["unique_str"])
            command_lists.append("[CommandList_Draw_Component" + str(component_id) + "]")
            command_lists.append("run = CommandList\\EFMIv1\\OverrideTextures")
            if component["cpu_posed"]:
                command_lists.append("; 当前组件仅参与骨骼读取，没有可替换的 GPU 网格")
                command_lists.new_line()
                continue

            current_key = "indexcount_" + str(component["index_count"])
            if submesh is not None:
                self._append_merged_buffer_bindings(command_lists, submesh)
                self._append_merged_slot_texture_bindings(
                    command_lists,
                    submesh,
                    drawib_drawibmodel_dict.get(submesh.match_draw_ib),
                )
                current_key = self._get_submesh_ib_key(submesh)
                if current_key in self.cross_ib_info_dict:
                    _, own_drawcalls = self._split_drawcalls_by_cross_ib(
                        submesh.drawcall_model_list,
                        source_ib_key=current_key,
                    )
                else:
                    own_drawcalls = submesh.drawcall_model_list
                self._append_merged_draw_lines(command_lists, own_drawcalls)
            else:
                command_lists.append("; 当前组件没有蓝图自定义网格，保留用于骨骼与跨 IB 目标回调")
            self._append_merged_incoming_cross_ib_draws(
                command_lists,
                current_key,
                submesh_by_unique,
                drawib_drawibmodel_dict,
            )
            command_lists.new_line()

        command_lists.append("[CommandList_Component_DrawInstances]")
        command_lists.append("handling = skip")
        command_lists.append("$\\EFMIv1\\component_count = $component_count")
        command_lists.append("$\\EFMIv1\\bones_count = $bones_count")
        command_lists.append("$\\EFMIv1\\instance_count = $max_instance_count")
        command_lists.append("run = CommandList\\EFMIv1\\Object_ReadConfig")
        command_lists.append("$\\EFMIv1\\lod_level = $lod_level")
        command_lists.append("$\\EFMIv1\\custom_mesh_scale = 1.00")
        command_lists.append("run = CommandList\\EFMIv1\\Component_ReadConfig")
        command_lists.append(
            "Pool\\EFMIv1\\Input_ObjectSpatialIdentity = ref Pool_ObjectSpatialIdentity"
        )
        command_lists.append(
            "run = CommandList\\EFMIv1\\SpatialIdentity_IdentifyComponentInstances"
        )
        command_lists.append(
            "CommandList\\EFMIv1\\Callback_MergedSkeleton_ConnectComponent = "
            "ref CommandList_MergedSkeleton_ConnectComponent"
        )
        command_lists.append("run = CommandList\\EFMIv1\\Component_DrawInstances")
        command_lists.new_line()

        command_lists.append("[CommandList_MergedSkeleton_ConnectComponent]")
        command_lists.append("if !$merged_skeleton_initialized")
        command_lists.append("    $merged_skeleton_initialized = 1")
        command_lists.append("    run = CommandListInitializeMergedSkeleton")
        command_lists.append("endif")
        command_lists.append(
            "Pool\\EFMIv1\\Input_MergedSkeleton_Component_VertexGroupOffsets = "
            "ref Pool_MergedSkeleton_Component_VertexGroupOffsets"
        )
        command_lists.append(
            "Pool\\EFMIv1\\Input_MergedSkeleton_Component_VertexGroupCounts = "
            "ref Pool_MergedSkeleton_Component_VertexGroupCounts"
        )
        command_lists.append(
            "Pool\\EFMIv1\\Input_MergedSkeleton_Component_LodRemaps = "
            "ref Pool_MergedSkeleton_Component_LodRemaps"
        )
        command_lists.append(
            "Pool\\EFMIv1\\Input_MergedSkeleton_Instance_UpdateFrame = "
            "ref Pool_MergedSkeleton_Instance_UpdateFrame"
        )
        command_lists.append(
            "Pool\\EFMIv1\\Input_MergedSkeleton_Instance_LodLevel = "
            "ref Pool_MergedSkeleton_Instance_LodLevel"
        )
        command_lists.append(
            "Resource\\EFMIv1\\Output_MergedSkeleton = ref ResourceMergedSkeletonDataRW"
        )
        command_lists.append("run = CommandList\\EFMIv1\\MergedSkeleton_AttachComponent")
        command_lists.append(
            "vb2->ElementFormat(BLENDINDICES, 0) = R16G16B16A16_UINT"
        )
        command_lists.new_line()

        command_lists.append("[CommandListInitializeMergedSkeleton]")
        command_lists.append(
            "Resource\\EFMIv1\\OutputMergedSkeleton_Template = "
            "ref ResourceMergedSkeletonDataRW"
        )
        command_lists.append("run = CommandList\\EFMIv1\\InitializeMergedSkeleton")
        command_lists.append("local $lod_level_count = $\\EFMIv1\\cfg_ms_max_lod_level_count")
        command_lists.append("local $component_id")
        for component in profile["components"]:
            if component["cpu_posed"]:
                continue
            component_id = component["component_id"]
            command_lists.append("$component_id = " + str(component_id))
            command_lists.append(
                "$Pool_MergedSkeleton_Component_VertexGroupOffsets[$component_id] = "
                + str(component["vg_offset"])
            )
            command_lists.append(
                "$Pool_MergedSkeleton_Component_VertexGroupCounts[$component_id] = "
                + str(component["vg_count"])
            )
            command_lists.append(
                "Pool_MergedSkeleton_Component_LodRemaps"
                "[$component_id*$lod_level_count+0] = null"
            )
        ini_builder.append_section(command_lists)

        entrypoints = M_IniSection(M_SectionType.TextureOverrideIB)
        for component in profile["components"]:
            component_id = component["component_id"]
            submesh = submesh_by_unique.get(component["unique_str"])
            entrypoints.append(
                "[TextureOverride_EntryPoint_Component" + str(component_id) + "]"
            )
            entrypoints.append("hash = " + component["ib_hash"])
            entrypoints.append("match_first_index = " + str(component["first_index"]))
            entrypoints.append("match_index_count = " + str(component["index_count"]))
            entrypoints.append("$object_detected = 1")
            entrypoints.append("if $mod_enabled && DRAW_TYPE == 4")
            entrypoints.append(
                "    $\\EFMIv1\\component_id = " + str(component_id)
            )
            entrypoints.append(
                "    $\\EFMIv1\\gpu_posed = " + str(int(not component["cpu_posed"]))
            )
            if submesh is None or component["cpu_posed"]:
                entrypoints.append("    $\\EFMIv1\\skip_skeleton_override = 1")
            entrypoints.append(
                "    CommandList\\EFMIv1\\Callback_Component_DrawCustom = "
                "ref CommandList_Draw_Component" + str(component_id)
            )
            entrypoints.append("    run = CommandList_Component_DrawInstances")
            if submesh is not None and self.blueprint_model.keyname_mkey_dict:
                active_index = draw_ib_active_index_dict.get(submesh.match_draw_ib, 0)
                entrypoints.append("    $active" + str(active_index) + " = 1")
                if GlobalProterties.generate_branch_mod_gui():
                    entrypoints.append("    $ActiveCharacter = 1")
            entrypoints.append("endif")
            entrypoints.new_line()
        ini_builder.append_section(entrypoints)

        resources = M_IniSection(M_SectionType.ResourceBuffer)
        resources.append("[Pool_ObjectSpatialIdentity]")
        resources.append(
            "pool_size = $max_instance_count * $\\EFMIv1\\cfg_spatial_instance_load_ratio"
        )
        resources.append("pool_index_type = spatial")
        resources.append("pool_spatial_radius = $\\EFMIv1\\cfg_spatial_base_radius")
        resources.append(
            "pool_expiration_timeout_frames = $\\EFMIv1\\cfg_spatial_expiration_frames"
        )
        resources.append(
            "pool_expiration_reset_elements = $\\EFMIv1\\cfg_spatial_expiration_reset"
        )
        resources.append(
            "pool_expiration_refresh_on_read = $\\EFMIv1\\cfg_spatial_expiration_read_refresh"
        )
        resources.append(
            "pool_variable_default_value = $\\EFMIv1\\cfg_spatial_detault_value"
        )
        resources.new_line()
        resources.append("[Pool_MergedSkeleton_Component_VertexGroupOffsets]")
        resources.append("pool_size = $component_count")
        resources.append("[Pool_MergedSkeleton_Component_VertexGroupCounts]")
        resources.append("pool_size = $component_count")
        resources.append("[Pool_MergedSkeleton_Component_LodRemaps]")
        resources.append(
            "pool_size = $component_count * $\\EFMIv1\\cfg_ms_max_lod_level_count"
        )
        resources.append("[Pool_MergedSkeleton_Instance_UpdateFrame]")
        resources.append("pool_size = $component_count * $max_instance_count")
        resources.append("[Pool_MergedSkeleton_Instance_LodLevel]")
        resources.append("pool_size = $component_count * $max_instance_count")
        resources.append("[ResourceMergedSkeletonDataRW]")
        resources.append("type = RWBuffer")
        resources.append("format = R32G32B32A32_FLOAT")
        resources.append(
            "array = ($\\EFMIv1\\cfg_ms_implicit_bones_count + "
            "$\\EFMIv1\\cfg_ms_skeletons_count * $bones_count * $max_instance_count) "
            "* $\\EFMIv1\\cfg_ms_bone_entry_size"
        )
        resources.new_line()
        buffer_folder_name = BlueprintExportHelper.get_current_buffer_folder_name()
        for submesh in self.submesh_model_list:
            prefix = self._merged_resource_prefix(submesh)
            resources.append("[" + prefix + "_Index]")
            resources.append("type = Buffer")
            resources.append("format = DXGI_FORMAT_R32_UINT")
            resources.append(
                "filename = " + buffer_folder_name + "\\"
                + submesh.unique_str + "-Index.buf"
            )
            resources.new_line()
            for category in submesh.category_buffer_dict.keys():
                resources.append("[" + prefix + "_" + category + "]")
                resources.append("type = Buffer")
                resources.append(
                    "stride = "
                    + str(submesh.d3d11_game_type.CategoryStrideDict.get(category, 0))
                )
                resources.append(
                    "filename = " + buffer_folder_name + "\\"
                    + submesh.unique_str + "-" + category + ".buf"
                )
                resources.new_line()
        ini_builder.append_section(resources)

        if not GlobalProterties.forbid_auto_texture_ini():
            texture_resources = M_IniSection(M_SectionType.ResourceTexture)
            appended_names = set()
            for drawib_model in self.drawib_model_list:
                for submesh in drawib_model.submesh_model_list:
                    for texture_info in drawib_model.get_submesh_texture_markup_info_list(
                        submesh
                    ):
                        if getattr(texture_info, "mark_type", "") != "Slot":
                            continue
                        resource_name = texture_info.get_resource_name()
                        if resource_name in appended_names:
                            continue
                        appended_names.add(resource_name)
                        texture_resources.append("[" + resource_name + "]")
                        texture_resources.append(
                            "filename = Textures/" + texture_info.mark_filename
                        )
                        texture_resources.new_line()
            ini_builder.append_section(texture_resources)

        mod_info = M_IniSection(M_SectionType.ResourceModInfo)
        mod_info.append("[ResourceModName]")
        mod_info.append('type = Buffer')
        mod_info.append('data = "' + workspace_name + '"')
        mod_info.append("[ResourceModAuthor]")
        mod_info.append("[ResourceModDesc]")
        mod_info.append("[ResourceModLink]")
        mod_info.append("[ResourceModLogo]")
        ini_builder.append_section(mod_info)

        for drawib_model in self.drawib_model_list:
            M_IniHelper.move_slot_style_textures(draw_ib_model=drawib_model)
        GlobalKeyCountHelper.generated_mod_number = len(self.drawib_model_list)
        M_IniHelper.add_branch_key_sections(
            ini_builder=ini_builder,
            key_name_mkey_dict=self.blueprint_model.keyname_mkey_dict,
        )
        M_IniHelperGUI.add_branch_mod_gui_section(
            ini_builder=ini_builder,
            key_name_mkey_dict=self.blueprint_model.keyname_mkey_dict,
        )

        ini_filepath = os.path.join(
            GlobalConfig.path_generate_mod_folder(),
            GlobalConfig.get_workspace_name() + ".ini",
        )
        ini_builder.save_to_file(ini_filepath)

    def generate_ini_file(self):
        if self.merged_skeleton_mode:
            return self._generate_merged_skeleton_ini_file()

        ini_builder = M_IniBuilder()
        drawib_drawibmodel_dict = {
            drawib_model.draw_ib: drawib_model
            for drawib_model in self.drawib_model_list
        }
        draw_ib_active_index_dict = {
            drawib_model.draw_ib: index
            for index, drawib_model in enumerate(self.drawib_model_list)
        }

        if self.has_cross_ib:
            for node_name, cross_ib_method in self.cross_ib_method_dict.items():
                if cross_ib_method != 'END_FIELD':
                    print(f"[CrossIB] 警告: 节点 {node_name} 使用的跨 IB 方式 '{cross_ib_method}' 不适用于 EFMI 模式")
                    self.has_cross_ib = False
                    break

        if self.has_cross_ib:
            self._add_cross_ib_present_section(ini_builder)
            self._add_cross_ib_resource_id_sections(ini_builder)

        M_IniHelper.generate_hash_style_texture_ini(
            ini_builder=ini_builder,
            drawib_drawibmodel_dict=drawib_drawibmodel_dict,
        )

        self._integrate_object_swap_ini_hook(ini_builder)

        texture_override_ib_section = M_IniSection(M_SectionType.TextureOverrideIB)

        for submesh_model in self.submesh_model_list:
            drawib_model = drawib_drawibmodel_dict.get(submesh_model.match_draw_ib)
            active_index = draw_ib_active_index_dict.get(submesh_model.match_draw_ib, 0)

            current_ib_key = self._get_submesh_ib_key(submesh_model)

            is_source_ib = current_ib_key in self.cross_ib_info_dict
            source_ib_list_for_target = self.cross_ib_target_info.get(current_ib_key, [])
            is_target_ib = len(source_ib_list_for_target) > 0

            if self.cross_ib_match_mode == 'INDEX_COUNT':
                current_identifier = submesh_model.match_index_count
            else:
                current_identifier = submesh_model.match_draw_ib

            texture_override_ib_section.append("[TextureOverride_" + submesh_model.unique_str.replace("-","_") + "]")
            texture_override_ib_section.append("hash = " + submesh_model.match_draw_ib)
            texture_override_ib_section.append("match_first_index = " + submesh_model.match_first_index)
            texture_override_ib_section.append("match_index_count = " + submesh_model.match_index_count)
            texture_override_ib_section.append("handling = skip")

            if is_target_ib:
                texture_override_ib_section.append("analyse_options = deferred_ctx_immediate dump_rt dump_cb dump_vb dump_ib buf txt dds dump_tex dds symlink")

            texture_override_ib_section.append("run = CommandList\\EFMIv1\\OverrideTextures")

            ib_resource_name = "Resource_" + submesh_model.unique_str.replace("-","_") + "_Index"
            texture_override_ib_section.append("ib = " + ib_resource_name)

            for category in submesh_model.category_buffer_dict.keys():
                category_slot = submesh_model.d3d11_game_type.CategoryExtractSlotDict.get(category,"unknown_slot")
                category_resource_name = "Resource_" + submesh_model.unique_str.replace("-","_")  + "_" + category
                texture_override_ib_section.append(category_slot + " = " + category_resource_name)

            unique_str = submesh_model.unique_str
            texture_override_ib_section.append("vb3 = Resource_" + unique_str.replace('-', '_') + "_Position")

            if not GlobalProterties.forbid_auto_texture_ini() and drawib_model is not None:
                texture_markup_info_list = drawib_model.get_submesh_texture_markup_info_list(submesh_model)
                if GlobalProterties.use_rabbitfx_slot():
                    for texture_markup_info in texture_markup_info_list:
                        if getattr(texture_markup_info, "mark_type", "") != "Slot":
                            continue
                        if texture_markup_info.mark_name == "DiffuseMap":
                            texture_override_ib_section.append("Resource\\RabbitFx\\Diffuse = ref " + texture_markup_info.get_resource_name())
                        elif texture_markup_info.mark_name == "LightMap":
                            texture_override_ib_section.append("Resource\\RabbitFx\\LightMap = ref " + texture_markup_info.get_resource_name())
                        elif texture_markup_info.mark_name == "NormalMap":
                            texture_override_ib_section.append("Resource\\RabbitFx\\NormalMap = ref " + texture_markup_info.get_resource_name())
                    
                    texture_override_ib_section.append("run = CommandList\\RabbitFx\\SetTextures")
                    
                    for texture_markup_info in texture_markup_info_list:
                        if getattr(texture_markup_info, "mark_type", "") != "Slot":
                            continue
                        if texture_markup_info.mark_name in ["DiffuseMap", "LightMap", "NormalMap"]:
                            pass
                        else:
                            slot = texture_markup_info.mark_slot
                            if slot and not slot.lower().startswith("ps-t"):
                                num_match = re.search(r'\d+', slot)
                                if num_match:
                                    slot = "ps-t" + num_match.group()
                                else:
                                    slot = "ps-t" + slot
                            texture_override_ib_section.append(slot + " = " + texture_markup_info.get_resource_name())
                else:
                    for texture_markup_info in texture_markup_info_list:
                        if getattr(texture_markup_info, "mark_type", "") != "Slot":
                            continue
                        texture_override_ib_section.append(texture_markup_info.mark_slot + " = " + texture_markup_info.get_resource_name())

            is_both_source_and_target = is_source_ib and is_target_ib and self.has_cross_ib

            if is_both_source_and_target:
                cross_ib_drawcalls, non_cross_ib_drawcalls = self._split_drawcalls_by_cross_ib(
                    submesh_model.drawcall_model_list,
                    source_ib_key=current_ib_key
                )

                target_ib_keys = self.cross_ib_source_to_target_dict.get(current_ib_key, [])
                grouped_source_drawcalls = self._group_drawcalls_by_cross_ib_target(
                    cross_ib_drawcalls, current_ib_key, target_ib_keys
                )

                for (target_ib_key, vb_condition), objects in grouped_source_drawcalls.items():
                    if not objects:
                        continue

                    texture_override_ib_section.append(";跨 iB 区域")
                    source_branch_lines = []
                    self._append_cross_ib_source_branches(
                        source_branch_lines, vb_condition, current_identifier,
                        M_IniHelper.get_drawindexed_instanced_str_list(objects),
                    )
                    for source_branch_line in source_branch_lines:
                        texture_override_ib_section.append(source_branch_line)

                texture_override_ib_section.append(";不需要跨 Ib 的物体引用")

                if non_cross_ib_drawcalls:
                    drawindexed_str_list = M_IniHelper.get_drawindexed_instanced_str_list(non_cross_ib_drawcalls)
                    for drawindexed_str in drawindexed_str_list:
                        if drawindexed_str.strip():
                            texture_override_ib_section.append(drawindexed_str)

                if is_target_ib and source_ib_list_for_target:
                    self._append_target_cross_ib_blocks(
                        texture_override_ib_section, source_ib_list_for_target, current_ib_key
                    )

                texture_override_ib_section.append("")
                texture_override_ib_section.append("post vs-cb1 = null")
                texture_override_ib_section.append("post vs-cb2 = null")
                texture_override_ib_section.append("post vs-t0 = null")
                texture_override_ib_section.append("post cs-t2 = null")

            elif is_source_ib and self.has_cross_ib:
                target_ib_keys = self.cross_ib_source_to_target_dict.get(current_ib_key, [])
                target_ib_key = target_ib_keys[0] if target_ib_keys else None
                cross_ib_lines = self._generate_cross_ib_block_for_source(
                    current_identifier, submesh_model.drawcall_model_list,
                    source_ib_key=current_ib_key, target_ib_key=target_ib_key
                )
                for line in cross_ib_lines:
                    texture_override_ib_section.append(line)

            elif is_target_ib and self.has_cross_ib and source_ib_list_for_target:
                all_target_drawcalls = submesh_model.drawcall_model_list
                if all_target_drawcalls:
                    drawindexed_str_list = M_IniHelper.get_drawindexed_instanced_str_list(all_target_drawcalls)
                    for drawindexed_str in drawindexed_str_list:
                        if drawindexed_str.strip():
                            texture_override_ib_section.append(drawindexed_str)

                self._append_target_cross_ib_blocks(
                    texture_override_ib_section, source_ib_list_for_target, current_ib_key
                )

                texture_override_ib_section.append("")
                texture_override_ib_section.append("post vs-cb1 = null")
                texture_override_ib_section.append("post vs-cb2 = null")
                texture_override_ib_section.append("post vs-t0 = null")
                texture_override_ib_section.append("post cs-t2 = null")

            else:
                for draw_line in M_IniHelper.get_drawindexed_instanced_str_list(submesh_model.drawcall_model_list):
                    texture_override_ib_section.append(draw_line)

            if len(self.blueprint_model.keyname_mkey_dict.keys()) != 0:
                texture_override_ib_section.append("$active" + str(active_index) + " = 1")
                if GlobalProterties.generate_branch_mod_gui():
                    texture_override_ib_section.append("$ActiveCharacter = 1")

            texture_override_ib_section.new_line()

        ini_builder.append_section(texture_override_ib_section)

        resource_buffer_section = M_IniSection(M_SectionType.ResourceBuffer)
        buffer_folder_name = BlueprintExportHelper.get_current_buffer_folder_name()
        for submesh_model in self.submesh_model_list:
            ib_resource_name = "Resource_" + submesh_model.unique_str.replace("-","_") + "_Index"
            resource_buffer_section.append("[" + ib_resource_name + "]")
            resource_buffer_section.append("type = Buffer")
            resource_buffer_section.append("format = DXGI_FORMAT_R32_UINT")
            resource_buffer_section.append("filename = " + buffer_folder_name + "\\" + submesh_model.unique_str + "-Index.buf")
            resource_buffer_section.new_line()

            for category in submesh_model.category_buffer_dict.keys():
                category_resource_name = "Resource_" + submesh_model.unique_str.replace("-","_")  + "_" + category
                stride = submesh_model.d3d11_game_type.CategoryStrideDict.get(category,0)
                resource_buffer_section.append("[" + category_resource_name + "]")
                resource_buffer_section.append("type = Buffer")
                resource_buffer_section.append("stride = " + str(stride))
                resource_buffer_section.append("filename = " + buffer_folder_name + "\\" + submesh_model.unique_str + "-" + category + ".buf")
                resource_buffer_section.new_line()

        if not GlobalProterties.forbid_auto_texture_ini():
            resource_texture_section = M_IniSection(M_SectionType.ResourceTexture)
            appended_resource_names = set()
            for drawib_model in self.drawib_model_list:
                for submesh_model in drawib_model.submesh_model_list:
                    for texture_markup_info in drawib_model.get_submesh_texture_markup_info_list(submesh_model):
                        if getattr(texture_markup_info, "mark_type", "") != "Slot":
                            continue
                        resource_name = texture_markup_info.get_resource_name()
                        if resource_name in appended_resource_names:
                            continue
                        appended_resource_names.add(resource_name)
                        resource_texture_section.append("[" + texture_markup_info.get_resource_name() + "]")
                        resource_texture_section.append("filename = Textures/" + texture_markup_info.mark_filename)
                        resource_texture_section.new_line()
            ini_builder.append_section(resource_texture_section)

        ini_builder.append_section(resource_buffer_section)

        for drawib_model in self.drawib_model_list:
            M_IniHelper.move_slot_style_textures(draw_ib_model=drawib_model)

        GlobalKeyCountHelper.generated_mod_number = len(self.drawib_model_list)
        M_IniHelper.add_branch_key_sections(
            ini_builder=ini_builder,
            key_name_mkey_dict=self.blueprint_model.keyname_mkey_dict,
        )
        M_IniHelperGUI.add_branch_mod_gui_section(
            ini_builder=ini_builder,
            key_name_mkey_dict=self.blueprint_model.keyname_mkey_dict,
        )

        ini_filepath = os.path.join(GlobalConfig.path_generate_mod_folder(), GlobalConfig.get_workspace_name() + ".ini")
        ini_builder.save_to_file(ini_filepath)

        if self.has_cross_ib:
            self._copy_cross_ib_hlsl_files()

    def _append_target_cross_ib_blocks(self, section, source_ib_list_for_target, current_ib_key):
        for source_ib_key in source_ib_list_for_target:
            if self.cross_ib_match_mode == 'INDEX_COUNT':
                source_identifier = source_ib_key.replace('indexcount_', '') if source_ib_key.startswith('indexcount_') else source_ib_key.split("_")[0]
            else:
                source_hash = source_ib_key.split("_")[0]
                source_identifier = source_hash

            source_submesh = self._find_source_submesh_by_ib_key(source_ib_key)
            source_drawib_model = self._find_source_drawib_by_ib_key(source_ib_key)

            if not source_submesh or not source_drawib_model:
                continue

            cross_drawcalls, _ = self._split_drawcalls_by_cross_ib(
                source_submesh.drawcall_model_list,
                source_ib_key=source_ib_key,
                target_ib_key=current_ib_key
            )

            if not cross_drawcalls:
                continue

            grouped_cross_drawcalls = {}
            for drawcall_model in cross_drawcalls:
                obj_name = drawcall_model.obj_name if hasattr(drawcall_model, 'obj_name') else str(drawcall_model)
                vb_condition_target = self._get_vb_condition_for_object(obj_name, source_ib_key, current_ib_key, 'target')
                if vb_condition_target not in grouped_cross_drawcalls:
                    grouped_cross_drawcalls[vb_condition_target] = []
                grouped_cross_drawcalls[vb_condition_target].append(drawcall_model)

            for vb_condition_target, objects in grouped_cross_drawcalls.items():
                if not objects:
                    continue

                section.append(f";跨 IB 身份块,绘制 {source_identifier} 需要跨 Ib 的物体引用")
                if vb_condition_target:
                    section.append(vb_condition_target)
                section.append(f"    cs-t2 = ResourceID_{source_identifier}")
                section.append(f"    run = CustomShader_RedirectCB1_{source_identifier}")
                section.append(f"    vs-t0 = ResourceFakeT0_SRV_{source_identifier}")
                # 202/203 回放通道通过 vs-cb2 重放记录的本体数据 (fix-efmi-cross-ib skill)
                section.append(f"    vs-cb2 = ResourceFakeCB1_{source_identifier}")
                section.append("    ;跨 IB 块数据区域")

                source_unique_str = source_submesh.unique_str
                section.append(f"    vb0 = Resource_{source_unique_str.replace('-', '_')}_Position")
                section.append(f"    vb1 = Resource_{source_unique_str.replace('-', '_')}_Texcoord")
                section.append(f"    vb2 = Resource_{source_unique_str.replace('-', '_')}_Blend")
                section.append(f"    vb3 = Resource_{source_unique_str.replace('-', '_')}_Position")
                src_ib_resource_name = "Resource_" + source_unique_str.replace('-', '_') + "_Index"
                section.append(f"    ib = {src_ib_resource_name}")

                section.append(";所有需要跨 Ib 的物体引用")

                drawindexed_str_list = M_IniHelper.get_drawindexed_instanced_str_list(objects)
                for drawindexed_str in drawindexed_str_list:
                    if drawindexed_str.strip():
                        section.append(drawindexed_str)

                # 条件为空 (目标槽位全部未勾选) 时上方没有输出 if，不能输出孤立的 endif
                if vb_condition_target:
                    section.append("endif")

    def _copy_cross_ib_hlsl_files(self):
        addon_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        source_dir = os.path.join(addon_dir, "Toolset", "old")

        if not os.path.exists(source_dir):
            print(f"[CrossIB] 警告: Toolset目录不存在: {source_dir}")
            return

        hlsl_files = [
            'extract_cb1_ps.hlsl',
            'extract_cb1_vs.hlsl',
            'extract_capture_cb1_vs.hlsl',
            'record_bones_cs.hlsl',
            'redirect_cb1_cs.hlsl'
        ]

        mod_export_path = GlobalConfig.path_generate_mod_folder()
        res_dir = os.path.join(mod_export_path, "res")
        os.makedirs(res_dir, exist_ok=True)

        copied_count = 0
        for hlsl_file in hlsl_files:
            source_file = os.path.join(source_dir, hlsl_file)
            target_file = os.path.join(res_dir, hlsl_file)

            if os.path.exists(source_file):
                # 框架 HLSL 必须与生成的 ini 结构保持一致，始终覆盖旧副本
                # (否则修复 shader 后重新生成 Mod 仍会残留过期版本，如旧的 b1 版 extract_cb1_vs)
                shutil.copy2(source_file, target_file)
                print(f"[CrossIB] 已复制: {hlsl_file}")
                copied_count += 1
            else:
                print(f"[CrossIB] 警告: 源文件不存在: {source_file}")

        print(f"[CrossIB] 共复制 {copied_count} 个HLSL文件到 {res_dir}")

    def _integrate_object_swap_ini_hook(self, ini_builder: M_IniBuilder):
        try:
            from ...blueprint.node_swap_ini import SwapKeyINIIntegrator
            from ...blueprint.export_helper import BlueprintExportHelper

            blueprint_tree = BlueprintExportHelper.get_current_blueprint_tree()
            if not blueprint_tree:
                return

            registry = getattr(self.blueprint_model, '_swap_key_registry', None)

            SwapKeyINIIntegrator.integrate_to_export(ini_builder, blueprint_tree, registry=registry)

        except ImportError:
            pass
        except Exception as e:
            from ...utils.log_utils import LOG
            LOG.warning(f"⚠️ 物体切换节点 INI 集成钩子执行失败: {e}")

    def export(self):
        TimerUtils.start_stage("缓冲文件生成")
        self.generate_buffer_files()
        TimerUtils.end_stage("缓冲文件生成")

        TimerUtils.start_stage("INI配置生成")
        self.generate_ini_file()
        TimerUtils.end_stage("INI配置生成")

    def export_buffers_only(self):
        """只导出 Buffer 文件，不生成 INI 配置"""
        TimerUtils.start_stage("缓冲文件生成")
        self.generate_buffer_files()
        TimerUtils.end_stage("缓冲文件生成")
