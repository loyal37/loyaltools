import os

import numpy

from ...common.global_config import GlobalConfig
from ...common.global_properties import GlobalProterties
from ...common.global_key_count_helper import GlobalKeyCountHelper
from ...common.m_ini_helper import M_IniHelper
from ...common.m_ini_helper_gui import M_IniHelperGUI
from ...common.m_ini_builder import M_IniBuilder, M_IniSection, M_SectionType
from .unity import ExportUnity
from ...common.zzmi_outline import compute_outline_texcoord
from ...extract.zzmi_dump_extractor import FRAME_EXTRACT_KEY
from ...utils.timer_utils import TimerUtils


class ZZMITextureMarkName:
    DiffuseMap = "DiffuseMap"
    NormalMap = "NormalMap"
    LightMap = "LightMap"
    MaterialMap = "MaterialMap"
    StockingMap = "StockingMap"
    GlowMap = "GlowMap"
    GlowGradient = "GlowGradient"
    WengineFx = "WengineFx"


class ExportZZMI(ExportUnity):
    # 绝区零的贴图统一用 ZZMI SlotFix 写法: 把贴图交给 Resource\ZZMI\<名称>，
    # 再 run = CommandList\ZZMI\SetTextures，由 SlotFix 分发到各渲染 Pass 的实际槽位。
    SLOT_FIX_RESOURCE_NAME_DICT = {
        ZZMITextureMarkName.DiffuseMap: r"Resource\ZZMI\Diffuse",
        ZZMITextureMarkName.NormalMap: r"Resource\ZZMI\NormalMap",
        ZZMITextureMarkName.LightMap: r"Resource\ZZMI\LightMap",
        ZZMITextureMarkName.MaterialMap: r"Resource\ZZMI\MaterialMap",
        ZZMITextureMarkName.StockingMap: r"Resource\ZZMI\WengineFx",
        ZZMITextureMarkName.GlowMap: r"Resource\ZZMI\GlowMap",
        ZZMITextureMarkName.GlowGradient: r"Resource\ZZMI\GlowGradient",
        ZZMITextureMarkName.WengineFx: r"Resource\ZZMI\WengineFx",
    }

    def __init__(self, blueprint_model):
        super().__init__(blueprint_model)

        self.cross_ib_info_dict = blueprint_model.cross_ib_info_dict
        self.cross_ib_method_dict = blueprint_model.cross_ib_method_dict
        self.has_cross_ib = blueprint_model.has_cross_ib
        self.cross_ib_object_names = blueprint_model.cross_ib_object_names

        print(f"[CrossIB ZZMI] 初始化: has_cross_ib={self.has_cross_ib}")
        print(f"[CrossIB ZZMI] cross_ib_info_dict={self.cross_ib_info_dict}")
        print(f"[CrossIB ZZMI] cross_ib_object_names={self.cross_ib_object_names}")

    def _get_submesh_ib_key(self, submesh_model, draw_ib):
        return f"{draw_ib}_{submesh_model.match_first_index}"

    @staticmethod
    def _get_frame_extract_info(drawib_model) -> dict | None:
        '''LoyalTools 帧分析提取器写入 SubmeshJson 的元数据；SSMT4 提取的工作空间没有该键'''
        frame_extract = getattr(drawib_model, "import_json_dict", {}).get(FRAME_EXTRACT_KEY)
        return frame_extract if isinstance(frame_extract, dict) else None

    @staticmethod
    def _get_category_element_offsets(d3d11_game_type, category_name: str) -> dict:
        '''{元素名: (分类缓冲内的字节偏移, Format)}'''
        offsets = {}
        byte_offset = 0
        for d3d11_element in d3d11_game_type.D3D11ElementList:
            if d3d11_element.Category != category_name:
                continue
            offsets[d3d11_element.ElementName] = (byte_offset, d3d11_element.Format)
            byte_offset += d3d11_element.ByteWidth
        return offsets

    def _optimize_outline(self, drawib_model):
        '''
        按导出后的几何重算描边数据并写回 Texcoord 缓冲的 TEXCOORD1 (算法见 common/zzmi_outline.py)。
        布局里缺少所需元素或格式不同的模型保持原样。
        '''
        d3d11_game_type = drawib_model.d3d11GameType
        if d3d11_game_type is None:
            return
        position_offsets = self._get_category_element_offsets(d3d11_game_type, "Position")
        texcoord_offsets = self._get_category_element_offsets(d3d11_game_type, "Texcoord")
        required_elements = (
            (position_offsets, "POSITION", "R32G32B32_FLOAT"),
            (position_offsets, "NORMAL", "R32G32B32_FLOAT"),
            (position_offsets, "TANGENT", "R32G32B32A32_FLOAT"),
            (texcoord_offsets, "TEXCOORD1", "R32G32_FLOAT"),
        )
        for offsets, element_name, element_format in required_elements:
            if offsets.get(element_name, (0, ""))[1] != element_format:
                print("[ZZMI] " + drawib_model.draw_ib + " 没有 " + element_format + " 的 " + element_name + "，跳过轮廓线优化")
                return

        position_buffer = drawib_model.category_buffer_dict.get("Position")
        texcoord_buffer = drawib_model.category_buffer_dict.get("Texcoord")
        if position_buffer is None or texcoord_buffer is None:
            return
        if drawib_model.combine_ib:
            index_list = list(drawib_model.ib)
        else:
            index_list = [index for ib in drawib_model.submesh_ib_dict.values() for index in ib]
        if not index_list:
            return

        position_stride = d3d11_game_type.CategoryStrideDict["Position"]
        texcoord_stride = d3d11_game_type.CategoryStrideDict["Texcoord"]
        position_rows = numpy.frombuffer(
            numpy.ascontiguousarray(position_buffer).tobytes(), dtype=numpy.uint8,
        ).reshape(-1, position_stride)

        def read_floats(element_name: str, component_count: int) -> numpy.ndarray:
            byte_offset = position_offsets[element_name][0]
            return numpy.ascontiguousarray(
                position_rows[:, byte_offset:byte_offset + component_count * 4]
            ).view(numpy.float32).reshape(-1, component_count)

        outline_texcoord = compute_outline_texcoord(
            positions=read_floats("POSITION", 3),
            normals=read_floats("NORMAL", 3),
            tangents=read_floats("TANGENT", 4),
            indices=numpy.asarray(index_list, dtype=numpy.int64),
        )

        texcoord_rows = numpy.frombuffer(
            numpy.ascontiguousarray(texcoord_buffer).tobytes(), dtype=numpy.uint8,
        ).reshape(-1, texcoord_stride).copy()
        if len(texcoord_rows) != len(outline_texcoord):
            return
        byte_offset = texcoord_offsets["TEXCOORD1"][0]
        texcoord_rows[:, byte_offset:byte_offset + 8] = outline_texcoord.view(numpy.uint8).reshape(-1, 8)
        drawib_model.category_buffer_dict["Texcoord"] = texcoord_rows.reshape(-1)

    def generate_buffer_files(self, output_folder: str):
        if GlobalProterties.zzz_outline_optimization():
            for drawib_model in self.drawib_model_list:
                self._optimize_outline(drawib_model)
        super().generate_buffer_files(output_folder)

    def add_frame_extract_vb_sections(self, ini_builder: M_IniBuilder, drawib_model, frame_extract: dict):
        '''
        帧分析提取的模型按 XXMI-Tools 的 Zenless Zone Zero 模板覆盖顶点缓冲。

        权重缓冲 (vb2) 在预蒙皮绘制和之后的 DrawIndexed 里都绑着，所以一切从它的
        TextureOverride 出发，再用 DRAW_TYPE 区分两种绘制:
            DRAW_TYPE 1 (Draw, 流输出预蒙皮): 换入 Position / Blend 并按新顶点数重画
            DRAW_TYPE 2/4 (DrawIndexed):       换入 Texcoord / Blend 并转去检查 IB
        ZZMI 关闭 $is_legacy 后只会对 vb0 / vb2 调用 checktextureoverride，
        这种写法不依赖 Position、Texcoord、IB 被单独检查。
        '''
        d3d11_game_type = drawib_model.d3d11GameType
        draw_ib = drawib_model.draw_ib
        slot_dict = d3d11_game_type.CategoryExtractSlotDict
        section_name_prefix = "[TextureOverride_VB_" + draw_ib + "_" + drawib_model.draw_ib_alias + "_"

        section = M_IniSection(M_SectionType.TextureOverrideVB)
        section.append("; " + draw_ib)

        def append_active_lines():
            # $active0 是所有物体切换节点共用的激活参数
            if len(self.blueprint_model.keyname_mkey_dict.keys()) != 0:
                section.append("$active0 = 1")
                if GlobalProterties.generate_branch_mod_gui():
                    section.append("$ActiveCharacter = 1")

        blend_hash = drawib_model.category_hash_dict.get("Blend", "")
        if frame_extract.get("PreSkinned") and blend_hash and "Blend" in slot_dict:
            section.append(section_name_prefix + "Blend]")
            section.append("hash = " + blend_hash)
            section.append("handling = skip")
            append_active_lines()
            section.append("if DRAW_TYPE == 2 || DRAW_TYPE == 4")
            if "Texcoord" in slot_dict:
                section.append("\t" + slot_dict["Texcoord"] + " = Resource" + draw_ib + "Texcoord")
            section.append("\t" + slot_dict["Blend"] + " = Resource" + draw_ib + "Blend")
            section.append("\tchecktextureoverride = ib")
            section.append("elif DRAW_TYPE == 1")
            section.append("\t" + slot_dict["Position"] + " = Resource" + draw_ib + "Position")
            section.append("\t" + slot_dict["Blend"] + " = Resource" + draw_ib + "Blend")
            section.append("\tdraw = " + str(drawib_model.draw_number) + ", 0")
            section.append("endif")
            section.new_line()

            texcoord_hash = drawib_model.category_hash_dict.get("Texcoord", "")
            if texcoord_hash and "Texcoord" in slot_dict:
                section.append(section_name_prefix + "Texcoord]")
                section.append("hash = " + texcoord_hash)
                section.append(slot_dict["Texcoord"] + " = Resource" + draw_ib + "Texcoord")
                section.new_line()
        else:
            # 没有预蒙皮的静态网格: DrawIndexed 直接使用这些顶点缓冲
            section.append(section_name_prefix + "Position]")
            section.append("hash = " + drawib_model.category_hash_dict.get("Position", ""))
            for category_name in d3d11_game_type.OrderedCategoryNameList:
                section.append(slot_dict[category_name] + " = Resource" + draw_ib + category_name)
            append_active_lines()
            section.append("checktextureoverride = ib")
            section.new_line()

        ini_builder.append_section(section)

    def add_unity_vs_texture_override_ib_sections(self, ini_builder: M_IniBuilder, drawib_model):
        texture_override_ib_section = M_IniSection(M_SectionType.TextureOverrideIB)
        draw_ib = drawib_model.draw_ib
        frame_extract = self._get_frame_extract_info(drawib_model)

        print(f"[CrossIB ZZMI] 处理 draw_ib={draw_ib}, has_cross_ib={self.has_cross_ib}")

        texture_override_ib_section.append("[TextureOverride_IB_" + draw_ib + "]")
        texture_override_ib_section.append("hash = " + draw_ib)
        texture_override_ib_section.append("handling = skip")
        texture_override_ib_section.new_line()

        for submesh_model in drawib_model.submesh_model_list:
            texture_override_name_suffix = drawib_model.get_submesh_texture_override_suffix(submesh_model)
            ib_resource_name = drawib_model.get_submesh_ib_resource_name(submesh_model)

            current_ib_key = self._get_submesh_ib_key(submesh_model, draw_ib)
            is_cross_ib_source = current_ib_key in self.cross_ib_info_dict
            is_cross_ib_target = any(current_ib_key in targets for targets in self.cross_ib_info_dict.values())

            print(f"[CrossIB ZZMI] submesh={submesh_model.unique_str}, ib_key={current_ib_key}, is_source={is_cross_ib_source}, is_target={is_cross_ib_target}")

            source_ib_list_for_target = []
            if is_cross_ib_target:
                for source_ib, target_ib_list in self.cross_ib_info_dict.items():
                    if current_ib_key in target_ib_list:
                        source_ib_list_for_target.append(source_ib)

            if is_cross_ib_source:
                texture_override_ib_section.append("[ResourceBodyVB_" + draw_ib + "_" + str(submesh_model.match_first_index) + "]")

            texture_override_ib_section.append("[TextureOverride_" + texture_override_name_suffix + "]")
            texture_override_ib_section.append("hash = " + draw_ib)
            texture_override_ib_section.append("match_first_index = " + str(submesh_model.match_first_index))
            # 与 XXMI-Tools 一致: 同时按索引数匹配，避免同一 first_index 的其它绘制区间被当成本部件
            if frame_extract is not None and int(submesh_model.match_index_count) > 0:
                texture_override_ib_section.append("match_index_count = " + str(submesh_model.match_index_count))

            if is_cross_ib_source:
                texture_override_ib_section.append("ResourceBodyVB_" + draw_ib + "_" + str(submesh_model.match_first_index) + " = copy vb0")

            ib_buf = drawib_model.submesh_ib_dict.get(submesh_model.unique_str, None)
            if ib_buf is None or len(ib_buf) == 0:
                texture_override_ib_section.append("ib = null")
                texture_override_ib_section.new_line()
                continue

            texture_override_ib_section.append("ib = " + ib_resource_name)

            texture_markup_info_list = drawib_model.get_submesh_texture_markup_info_list(submesh_model)
            if not GlobalProterties.forbid_auto_texture_ini() and texture_markup_info_list:
                uses_slot_fix = False

                for texture_markup_info in texture_markup_info_list:
                    if texture_markup_info.mark_type != "Slot":
                        continue

                    slot_fix_resource_name = self.SLOT_FIX_RESOURCE_NAME_DICT.get(texture_markup_info.mark_name)
                    if slot_fix_resource_name is not None:
                        texture_override_ib_section.append(
                            slot_fix_resource_name + " = ref " + texture_markup_info.get_resource_name()
                        )
                        uses_slot_fix = True
                    else:
                        # SlotFix 不认识的自定义标记名只能按标记时的槽位直接绑定
                        texture_override_ib_section.append(
                            texture_markup_info.mark_slot + " = " + texture_markup_info.get_resource_name()
                        )

                if uses_slot_fix:
                    texture_override_ib_section.append(r"run = CommandList\ZZMI\SetTextures")

            # CommandListSkinTexture 负责触发按贴图 hash 的覆盖。帧分析提取的模型只有
            # Hash 方式的标记才需要它 (XXMI-Tools 的模板里没有这一行)。
            if frame_extract is None:
                needs_skin_texture = bool(texture_markup_info_list)
            else:
                needs_skin_texture = any(
                    texture_markup_info.mark_type == "Hash" for texture_markup_info in texture_markup_info_list
                )
            if needs_skin_texture:
                texture_override_ib_section.append("run = CommandListSkinTexture")

            if is_cross_ib_source:
                non_cross_ib_drawcalls = []
                for drawcall_model in submesh_model.drawcall_model_list:
                    obj_name = drawcall_model.obj_name if hasattr(drawcall_model, 'obj_name') else str(drawcall_model)
                    if obj_name not in self.cross_ib_object_names:
                        non_cross_ib_drawcalls.append(drawcall_model)

                print(f"[CrossIB ZZMI] 源块绘制非跨IB物体: {len(non_cross_ib_drawcalls)} 个")

                for drawindexed_str in M_IniHelper.get_drawindexed_str_list(
                    non_cross_ib_drawcalls,
                    obj_name_draw_offset_dict=drawib_model.obj_name_draw_offset,
                ):
                    texture_override_ib_section.append(drawindexed_str)
            else:
                print(f"[CrossIB ZZMI] 非源块绘制物体: {len(submesh_model.drawcall_model_list)} 个")

                for drawindexed_str in M_IniHelper.get_drawindexed_str_list(
                    submesh_model.drawcall_model_list,
                    obj_name_draw_offset_dict=drawib_model.obj_name_draw_offset,
                ):
                    texture_override_ib_section.append(drawindexed_str)

            if is_cross_ib_target and source_ib_list_for_target:
                print(f"[CrossIB ZZMI] 目标块处理: source_ib_list={source_ib_list_for_target}")

                for source_ib_key in source_ib_list_for_target:
                    source_parts = source_ib_key.split("_")
                    source_hash = source_parts[0]
                    source_first_index = int(source_parts[1]) if len(source_parts) > 1 else 0

                    print(f"[CrossIB ZZMI] 查找源块: hash={source_hash}, first_index={source_first_index}")

                    source_drawib_model = None
                    for dib_model in self.drawib_model_list:
                        if dib_model.draw_ib == source_hash:
                            source_drawib_model = dib_model
                            print(f"[CrossIB ZZMI] 找到源 DrawIBModel: {dib_model.draw_ib}")
                            break

                    source_submesh = None
                    if source_drawib_model:
                        for sm in source_drawib_model.submesh_model_list:
                            if str(sm.match_first_index) == str(source_first_index):
                                source_submesh = sm
                                print(f"[CrossIB ZZMI] 找到源 submesh: {sm.unique_str}")
                                break

                    if source_submesh:
                        source_ib_resource_name = source_drawib_model.get_submesh_ib_resource_name(source_submesh)
                        texture_override_ib_section.append("ib = " + source_ib_resource_name)

                        texture_override_ib_section.append("vb0 = ResourceBodyVB_" + source_hash + "_" + str(source_first_index))
                        texture_override_ib_section.append("vb1 = Resource" + source_hash + "Texcoord")
                        texture_override_ib_section.append("vb2 = Resource" + source_hash + "Blend")
                        texture_override_ib_section.append("vb3 = ResourceBodyVB_" + source_hash + "_" + str(source_first_index))

                        cross_ib_drawcalls = []
                        for drawcall_model in source_submesh.drawcall_model_list:
                            obj_name = drawcall_model.obj_name if hasattr(drawcall_model, 'obj_name') else str(drawcall_model)
                            if obj_name in self.cross_ib_object_names:
                                cross_ib_drawcalls.append(drawcall_model)

                        print(f"[CrossIB ZZMI] 跨IB物体数量: {len(cross_ib_drawcalls)}")

                        if cross_ib_drawcalls:
                            for drawindexed_str in M_IniHelper.get_drawindexed_str_list(
                                cross_ib_drawcalls,
                                obj_name_draw_offset_dict=source_drawib_model.obj_name_draw_offset,
                            ):
                                texture_override_ib_section.append(drawindexed_str)
                    else:
                        print(f"[CrossIB ZZMI] 警告: 未找到源块 submesh for {source_ib_key}")

        ini_builder.append_section(texture_override_ib_section)

    def export(self):
        TimerUtils.start_stage("缓冲文件生成")
        self.generate_buffer_files(GlobalConfig.path_generatemod_buffer_folder())
        TimerUtils.end_stage("缓冲文件生成")

        if self.has_cross_ib:
            for node_name, cross_ib_method in self.cross_ib_method_dict.items():
                if cross_ib_method and cross_ib_method != 'VB_COPY':
                    print(f"[CrossIB] ❌ 错误: 节点 '{node_name}' 使用的跨 IB 方式 '{cross_ib_method}' 不适用于 ZZMI 模式")
                    print(f"[CrossIB] ZZMI 模式只支持 'VB_COPY' (VB 复制) 方式")
                    print(f"[CrossIB] 请在 Cross IB 节点中将跨 IB 方式改为 'VB_COPY'")
                    self.has_cross_ib = False
                    break

        print(f"[CrossIB ZZMI] export: has_cross_ib={self.has_cross_ib}")

        TimerUtils.start_stage("INI配置生成")
        ini_builder = M_IniBuilder()
        drawib_drawibmodel_dict = {drawib_model.draw_ib: drawib_model for drawib_model in self.drawib_model_list}

        M_IniHelper.generate_hash_style_texture_ini(ini_builder=ini_builder, drawib_drawibmodel_dict=drawib_drawibmodel_dict)
        self._integrate_object_swap_ini_hook(ini_builder)
        for drawib_model in self.drawib_model_list:
            frame_extract = self._get_frame_extract_info(drawib_model)
            if frame_extract is not None:
                self.add_unity_vs_texture_override_vlr_section(
                    ini_builder=ini_builder, drawib_model=drawib_model, include_uav_byte_stride=False,
                )
                self.add_frame_extract_vb_sections(
                    ini_builder=ini_builder, drawib_model=drawib_model, frame_extract=frame_extract,
                )
            else:
                self.add_unity_vs_texture_override_vlr_section(ini_builder=ini_builder, drawib_model=drawib_model)
                self.add_unity_vs_texture_override_vb_sections(ini_builder=ini_builder, drawib_model=drawib_model)
            self.add_unity_vs_texture_override_ib_sections(ini_builder=ini_builder, drawib_model=drawib_model)
            self.add_unity_vs_resource_vb_sections(ini_builder=ini_builder, drawib_model=drawib_model)
            self.add_resource_texture_sections(ini_builder=ini_builder, drawib_model=drawib_model)
            M_IniHelper.move_slot_style_textures(draw_ib_model=drawib_model)
            GlobalKeyCountHelper.generated_mod_number = GlobalKeyCountHelper.generated_mod_number + 1

        M_IniHelper.add_branch_key_sections(ini_builder=ini_builder, key_name_mkey_dict=self.blueprint_model.keyname_mkey_dict)
        M_IniHelper.add_shapekey_ini_sections(ini_builder=ini_builder, drawib_drawibmodel_dict=drawib_drawibmodel_dict)
        M_IniHelperGUI.add_branch_mod_gui_section(ini_builder=ini_builder, key_name_mkey_dict=self.blueprint_model.keyname_mkey_dict)
        ini_builder.save_to_file(os.path.join(GlobalConfig.path_generate_mod_folder(), GlobalConfig.get_workspace_name() + ".ini"))
        TimerUtils.end_stage("INI配置生成")


ModModelZZMI = ExportZZMI
