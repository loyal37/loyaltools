'''
绝区零 (ZZMI) 骨骼合并导出。

运行时原理见 common/zzmi_merged_skeleton.py。本模块负责:
    1. 把 Blender 里的全局骨骼编号 (规范编号) 改写成骨骼池编号:
       绘制部件自己有的骨骼读自己的那份，其余骨骼读提供它的部件那份；
    2. 判断每个合并部件是否真的用到了别的部件的骨骼:
       没用到的照常用游戏自己的骨骼表蒙皮 (权重缓冲写本地编号)，
       用到的在蒙皮前由 Mod 自带的计算着色器生成合并调色板;
    3. 生成运行时需要的 ini 段 (导入骨骼、生成调色板) 和 HLSL 文件。

骨骼是在各部件自己的预蒙皮绘制前一刻才上传的。和 EFMI 的骨骼合并一样，
每个部件只在自己的预蒙皮绘制上导入骨骼、生成调色板: 用到的骨骼如果来自本帧
更晚蒙皮的部件，用的是那个部件上一帧的骨骼 (经锚点换算，误差只是这些骨骼
相对锚点在一帧内的移动)。不重新绑定流输出目标 (3dmigoto 设置 so0 时会多释放
一次缓冲区引用)，同一角色同屏出现多次时每个实例照常写自己的流输出缓冲。

预蒙皮分支都带 VERTEX_COUNT 判断: 后处理的全屏绘制 (Draw(3)) 会残留角色的
vb2，ZZMI 对所有 DRAW_TYPE 1 的绘制都检查 vb2，不加判断就会在那些绘制上误触发。
'''

import math
import os

import numpy

from ...common.global_config import GlobalConfig
from ...common.m_ini_builder import M_IniSection, M_SectionType
from ...common import zzmi_merged_skeleton as zms


def _ceil_groups(count: int) -> int:
    return max(1, int(math.ceil(int(count) / 64.0)))


def _ini_name(ib_hash: str) -> str:
    return str(ib_hash).replace("-", "_")


class ZZMIMergedSkeletonExport:
    INDEX_FORMATS = {
        "R32G32B32A32_SINT": numpy.int32, "R32G32B32A32_UINT": numpy.uint32,
        "R32_SINT": numpy.int32, "R32_UINT": numpy.uint32,
    }

    def __init__(self, blueprint_model, drawib_model_list):
        self.blueprint_model = blueprint_model
        self.drawib_model_list = drawib_model_list
        self.enabled = self._detect()
        self.profile = None
        self.component_by_ib = {}
        self.modes = {}            # 合并部件 ib -> "local" | "merged"
        self.foreign_slots = {}    # ib -> set[int]
        self.dependencies = {}     # 用合并调色板的部件 ib -> set[部件 ib]
        self.participants = []     # 需要把骨骼导入骨骼池的部件 ib (按抓帧顺序)
        self.warnings = []
        if not self.enabled:
            return

        self.profile = zms.load_profile(GlobalConfig.path_workspace_folder(), required=True)
        self.component_by_ib = zms.get_component_by_ib(self.profile)
        unknown = sorted(
            drawib_model.draw_ib for drawib_model in drawib_model_list
            if drawib_model.draw_ib not in self.component_by_ib
        )
        if unknown:
            raise ValueError(
                "蓝图包含不属于当前骨骼合并配置的部件: " + ", ".join(unknown)
                + "。请把它们移出蓝图，或用骨骼合并模式重新提取。"
            )
        print("[ZZMI骨骼合并] 已启用，骨骼池 " + str(self.profile["bones_count"]) + " 根骨骼")

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------

    def _detect(self) -> bool:
        tree = getattr(self.blueprint_model, "_tree", None)
        if tree is not None and callable(getattr(tree, "get", None)):
            if bool(tree.get(zms.TREE_MARKER_MERGED_SKELETON, False)):
                return True
        try:
            import bpy
        except ImportError:
            return False
        objects = getattr(getattr(bpy, "data", None), "objects", None)
        get_object = getattr(objects, "get", None)
        if not callable(get_object):
            return False
        for draw_call_model in self.blueprint_model.ordered_draw_obj_data_model_list:
            obj = get_object(draw_call_model.get_blender_obj_name())
            if obj is not None and bool(obj.get(zms.OBJECT_MARKER_MERGED_SKELETON, False)):
                return True
        return False

    def role(self, ib_hash: str):
        if not self.enabled:
            return None
        component = self.component_by_ib.get(ib_hash)
        return component["role"] if component is not None else None

    def is_shape_keyed(self, ib_hash: str) -> bool:
        return self.role(ib_hash) == zms.ROLE_SHAPE_KEYED

    # ------------------------------------------------------------------
    # 权重缓冲
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_layout(game_type):
        offset = 0
        indices = weights = None
        for element in game_type.D3D11ElementList:
            if element.Category != "Blend":
                continue
            if int(element.SemanticIndex) == 0:
                if element.SemanticName == "BLENDINDICES":
                    indices = (offset, element)
                elif element.SemanticName in ("BLENDWEIGHT", "BLENDWEIGHTS"):
                    weights = (offset, element)
            offset += int(element.ByteWidth)
        return indices, weights, offset

    def prepare_blend_buffers(self):
        if not self.enabled:
            return
        bones_count = int(self.profile["bones_count"])
        for drawib_model in self.drawib_model_list:
            component = self.component_by_ib[drawib_model.draw_ib]
            if component["role"] != zms.ROLE_MERGED:
                continue
            game_type = drawib_model.d3d11GameType
            indices_info, weights_info, stride = self._blend_layout(game_type)
            blend_buffer = drawib_model.category_buffer_dict.get("Blend")
            if indices_info is None or blend_buffer is None:
                raise ValueError(drawib_model.draw_ib + " 没有 BLENDINDICES，无法按骨骼合并导出。")
            index_offset, index_element = indices_info
            index_dtype = self.INDEX_FORMATS.get(index_element.Format)
            if index_dtype is None:
                raise ValueError(
                    drawib_model.draw_ib + " 的 BLENDINDICES 格式 " + index_element.Format
                    + " 不是 32 位整数，骨骼合并导出不支持。"
                )
            if stride != int(game_type.CategoryStrideDict["Blend"]):
                raise ValueError(drawib_model.draw_ib + " 的 Blend 布局与步长不一致。")

            rows = numpy.ascontiguousarray(numpy.asarray(blend_buffer, dtype=numpy.uint8)).reshape(-1, stride).copy()
            width = int(index_element.ByteWidth)
            indices = numpy.ascontiguousarray(rows[:, index_offset:index_offset + width]).view(index_dtype)
            indices = indices.reshape(len(rows), -1).astype(numpy.int64)
            if weights_info is not None and weights_info[1].Format == "R32G32B32A32_FLOAT" and indices.shape[1] == 4:
                weight_offset = weights_info[0]
                weights = numpy.ascontiguousarray(rows[:, weight_offset:weight_offset + 16]).view(numpy.float32).reshape(-1, 4)
                active = weights > 0
            else:
                active = numpy.ones(indices.shape, dtype=bool)

            if active.any():
                used = indices[active]
                if int(used.min()) < 0 or int(used.max()) >= bones_count:
                    raise ValueError(
                        drawib_model.draw_ib + " 的权重用了编号 " + str(int(used.max()))
                        + " 的顶点组，超出合并骨骼的 " + str(bones_count)
                        + " 个顶点组。请检查是否混入了不是骨骼合并导入的物体。"
                    )

            lookup = numpy.arange(bones_count, dtype=numpy.int64)
            for canonical, slot in zms.build_private_remap(component).items():
                lookup[canonical] = slot
            vg_offset = int(component["vg_offset"])
            vg_end = vg_offset + int(component["vg_count"])
            rewritten = numpy.full(indices.shape, vg_offset, dtype=numpy.int64)
            rewritten[active] = lookup[indices[active]]
            foreign_mask = active & ((rewritten < vg_offset) | (rewritten >= vg_end))
            foreign = set(numpy.unique(rewritten[foreign_mask]).tolist())

            if foreign:
                mode = "merged"
                output = rewritten
                owners = sorted(
                    self.profile["components"][owner_id]["ib_hash"]
                    for owner_id in zms.slot_owner_ids(self.profile, foreign)
                )
                print(
                    "[ZZMI骨骼合并] " + drawib_model.draw_ib + ": "
                    + str(int(foreign_mask.any(axis=1).sum())) + " 个顶点用到其它部件的骨骼 ("
                    + ", ".join(owners) + "，共 " + str(len(foreign)) + " 根)，使用合并骨骼调色板"
                )
            else:
                mode = "local"
                output = rewritten - vg_offset
                print("[ZZMI骨骼合并] " + drawib_model.draw_ib + ": 只用到自己的骨骼，照常蒙皮")

            rows[:, index_offset:index_offset + width] = (
                output.astype(index_dtype).view(numpy.uint8).reshape(len(rows), width)
            )
            drawib_model.category_buffer_dict["Blend"] = rows.reshape(-1)
            self.modes[drawib_model.draw_ib] = mode
            self.foreign_slots[drawib_model.draw_ib] = foreign

        self._plan_runtime()

    # ------------------------------------------------------------------
    # 运行时规划
    # ------------------------------------------------------------------

    def _plan_runtime(self):
        components = self.profile["components"]
        participants = set()
        for ib_hash, mode in self.modes.items():
            if mode != "merged":
                continue
            component = self.component_by_ib[ib_hash]
            dependency_ids = zms.palette_dependencies(
                self.profile, component["component_id"], self.foreign_slots[ib_hash]
            )
            dependency_ibs = {components[component_id]["ib_hash"] for component_id in dependency_ids}
            self.dependencies[ib_hash] = dependency_ibs
            participants.add(ib_hash)
            participants.update(dependency_ibs)
        self.participants = sorted(participants, key=lambda ib_hash: self.component_by_ib[ib_hash]["pose_order"])

        # 按抓帧时的蒙皮顺序提示: 比绘制部件更晚上传骨骼的部件，其骨骼会慢一帧
        for ib_hash, dependency_ibs in sorted(self.dependencies.items()):
            own_order = self.component_by_ib[ib_hash]["pose_order"]
            late = sorted(
                dependency for dependency in dependency_ibs
                if self.component_by_ib[dependency]["pose_order"] > own_order
            )
            if late:
                self.warnings.append(
                    ib_hash + " 用到的 " + ", ".join(late) + " 的骨骼在抓帧里比它晚上传，"
                    "游戏里这部分用的是上一帧的骨骼 (与 EFMI 骨骼合并相同)。"
                )
        for warning in self.warnings:
            print("[ZZMI骨骼合并][提示] " + warning)

    def uses_runtime(self) -> bool:
        return bool(self.participants)

    # ------------------------------------------------------------------
    # ini
    # ------------------------------------------------------------------

    def draw_type1_lines(self, drawib_model, slot_dict: dict) -> list | None:
        '''Blend 覆盖段里预蒙皮分支的内容；None 表示用原来的写法'''
        if not self.enabled or drawib_model.draw_ib not in self.participants:
            return None
        return ["run = CommandListZZMS_Skin_" + _ini_name(drawib_model.draw_ib)]

    def _skin_command_list(self, ib_hash: str, exported_drawib, slot_dict: dict) -> list:
        name = _ini_name(ib_hash)
        lines = ["[CommandListZZMS_Skin_" + name + "]"]
        lines.append("ResourceZZMS_Input = ref vs-t0")
        lines.append("run = CustomShaderZZMS_Import_" + name)
        if exported_drawib is not None:
            if self.modes.get(ib_hash) == "merged":
                lines.append("run = CustomShaderZZMS_Palette_" + name)
                lines.append("vs-t0 = ref ResourceZZMS_Palette_" + name)
            lines.append(slot_dict["Position"] + " = Resource" + ib_hash + "Position")
            lines.append(slot_dict["Blend"] + " = Resource" + ib_hash + "Blend")
            lines.append("draw = " + str(exported_drawib.draw_number) + ", 0")
        lines.append("")
        return lines

    def hidden_component_ibs(self) -> list:
        '''参与合并、但不在蓝图里的部件 (和 EFMI 骨骼合并一样: 不再绘制，只在需要时提供骨骼)'''
        if not self.enabled:
            return []
        exported = {drawib_model.draw_ib for drawib_model in self.drawib_model_list}
        return [
            component["ib_hash"] for component in sorted(
                zms.merged_components(self.profile), key=lambda component: component["pose_order"],
            )
            if component["ib_hash"] not in exported
        ]

    def _add_hidden_component_sections(self, ini_builder):
        '''
        蓝图里没有的合并部件: 它的索引绘制全部跳过 (按 IB hash)，通过权重缓冲的覆盖段
        转去检查 IB (ZZMI 关闭 $is_legacy 时不会自己检查 IB)；别的部件用到它的骨骼时，
        只在它真正的预蒙皮绘制上导入骨骼，那次预蒙皮照常进行。
        '''
        blend_sections = M_IniSection(M_SectionType.TextureOverrideVB)
        ib_sections = M_IniSection(M_SectionType.TextureOverrideIB)
        for ib_hash in self.hidden_component_ibs():
            component = self.component_by_ib[ib_hash]
            provides_bones = ib_hash in self.participants
            pose_vertex_count = component["pose_vertex_count"] or component["vertex_count"]
            if not component["blend_vb"] or (provides_bones and not pose_vertex_count):
                raise ValueError(ib_hash + " 缺少权重缓冲 hash 或顶点数，无法隐藏或提供骨骼。请重新提取。")
            print(
                "[ZZMI骨骼合并] 蓝图里没有 " + ib_hash + "，隐藏它的绘制 (与 EFMI 骨骼合并相同)"
                + ("，仍为其它部件提供骨骼" if provides_bones else "")
            )
            blend_sections.append("; " + ib_hash + (" (隐藏，只提供骨骼)" if provides_bones else " (隐藏)"))
            blend_sections.append("[TextureOverride_ZZMS_Hidden_" + _ini_name(ib_hash) + "]")
            blend_sections.append("hash = " + component["blend_vb"])
            if provides_bones:
                blend_sections.append("if DRAW_TYPE == 1 && VERTEX_COUNT == " + str(pose_vertex_count))
                blend_sections.append("\trun = CommandListZZMS_Skin_" + _ini_name(ib_hash))
                blend_sections.append("elif DRAW_TYPE == 2 || DRAW_TYPE == 4")
            else:
                blend_sections.append("if DRAW_TYPE == 2 || DRAW_TYPE == 4")
            blend_sections.append("\tchecktextureoverride = ib")
            blend_sections.append("endif")
            blend_sections.new_line()
            ib_sections.append("[TextureOverride_ZZMS_Hidden_IB_" + _ini_name(ib_hash) + "]")
            ib_sections.append("hash = " + ib_hash)
            ib_sections.append("handling = skip")
            ib_sections.new_line()
        ini_builder.append_section(blend_sections)
        ini_builder.append_section(ib_sections)

    def add_runtime_sections(self, ini_builder):
        if not self.enabled:
            return
        self._add_hidden_component_sections(ini_builder)
        if not self.uses_runtime():
            return
        exported_by_ib = {drawib_model.draw_ib: drawib_model for drawib_model in self.drawib_model_list}
        bones_count = int(self.profile["bones_count"])

        command_lists = M_IniSection(M_SectionType.CommandList)
        command_lists.append("; Merged skeleton runtime (Zenless Zone Zero)")
        for ib_hash in self.participants:
            exported = exported_by_ib.get(ib_hash)
            slot_dict = exported.d3d11GameType.CategoryExtractSlotDict if exported is not None else {}
            for line in self._skin_command_list(ib_hash, exported, slot_dict):
                command_lists.append(line)

        shader_folder = "Shaders"
        for ib_hash in self.participants:
            name = _ini_name(ib_hash)
            component = self.component_by_ib[ib_hash]
            command_lists.append("[CustomShaderZZMS_Import_" + name + "]")
            command_lists.append("cs = " + shader_folder + "/" + zms.IMPORT_SHADER_FILENAME)
            command_lists.append("cs-t0 = ref ResourceZZMS_Input")
            command_lists.append("cs-t1 = ref ResourceZZMS_Components")
            command_lists.append("cs-t2 = ref ResourceZZMS_Params_" + name)
            command_lists.append("cs-u0 = ref ResourceZZMS_Pool")
            command_lists.append("dispatch = " + str(_ceil_groups(component["vg_count"])) + ", 1, 1")
            command_lists.append("cs-u0 = null")
            command_lists.append("cs-t0 = null")
            command_lists.append("cs-t1 = null")
            command_lists.append("cs-t2 = null")
            command_lists.append("")
        for ib_hash, mode in self.modes.items():
            if mode != "merged":
                continue
            name = _ini_name(ib_hash)
            command_lists.append("[CustomShaderZZMS_Palette_" + name + "]")
            command_lists.append("cs = " + shader_folder + "/" + zms.PALETTE_SHADER_FILENAME)
            command_lists.append("cs-t0 = ref ResourceZZMS_Pool")
            command_lists.append("cs-t1 = ref ResourceZZMS_Components")
            command_lists.append("cs-t2 = ref ResourceZZMS_Params_" + name)
            command_lists.append("cs-u0 = ref ResourceZZMS_Palette_" + name)
            command_lists.append("dispatch = " + str(_ceil_groups(bones_count)) + ", 1, 1")
            command_lists.append("cs-u0 = null")
            command_lists.append("cs-t0 = null")
            command_lists.append("cs-t1 = null")
            command_lists.append("cs-t2 = null")
            command_lists.append("")
        ini_builder.append_section(command_lists)

        resources = M_IniSection(M_SectionType.ResourceBuffer)
        resources.append("[ResourceZZMS_Pool]")
        resources.append("type = RWBuffer")
        resources.append("format = R32G32B32A32_FLOAT")
        resources.append("array = " + str(bones_count * 3))
        resources.append("bind_flags = shader_resource unordered_access")
        resources.new_line()
        for ib_hash, mode in self.modes.items():
            if mode != "merged":
                continue
            resources.append("[ResourceZZMS_Palette_" + _ini_name(ib_hash) + "]")
            resources.append("type = RWBuffer")
            resources.append("format = R32G32B32A32_FLOAT")
            resources.append("array = " + str(bones_count * 3))
            resources.append("bind_flags = shader_resource unordered_access")
            resources.new_line()
        resources.append("[ResourceZZMS_Components]")
        resources.append("type = Buffer")
        resources.append("format = R32G32B32A32_UINT")
        resources.append("data = " + " ".join(str(value) for value in zms.runtime_component_table(self.profile)))
        resources.new_line()
        for ib_hash in self.participants:
            component = self.component_by_ib[ib_hash]
            resources.append("[ResourceZZMS_Params_" + _ini_name(ib_hash) + "]")
            resources.append("type = Buffer")
            resources.append("format = R32G32B32A32_UINT")
            resources.append(
                "data = " + " ".join(str(value) for value in zms.runtime_params(self.profile, component["component_id"]))
            )
            resources.new_line()
        resources.append("[ResourceZZMS_Input]")
        resources.new_line()
        ini_builder.append_section(resources)

    def write_shaders(self, mod_folder: str):
        if not self.uses_runtime():
            return
        zms.write_runtime_shaders(os.path.join(mod_folder, "Shaders"))
