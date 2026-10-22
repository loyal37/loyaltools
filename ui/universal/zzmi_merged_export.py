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

统一延迟一帧 (生成 Mod 节点上的「统一延迟一帧(骨骼合并)」选项，和 WWMI 的骨骼合并相同): 整个角色都用
上一帧的骨骼蒙皮，各部分之间不再有一帧的错位。骨骼池在 [Present] 整份复制一份，
调色板从复制的那份生成；只用自己骨骼的部件、脸和独立部件也把自己的骨骼导入池里
(脸和独立部件放在合并骨骼之后)，蒙皮前换成自己上一帧的骨骼表。脸的顶点 (表情)
仍是游戏当帧算出来的。刚出现的第一帧还没有上一帧的骨骼，照常用当帧的。
各部件的骨骼表在自己根骨骼的空间里，游戏用这个根当帧的物体矩阵 (vs-cb1) 摆放部件，
而这些根是骨骼 (胯、胸、头)。所以自己绘制的部件画的时候也换成上一帧的物体矩阵: 整个
角色画成上一帧的样子 (位置也晚一帧)。游戏只给每一遍绘制写物体常量的第 0~3 行，
上一帧的矩阵用 Mod 自己的记录。
'''

import math
import os

import numpy

from ...common.global_config import GlobalConfig
from ...common.global_properties import GlobalProterties
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
        self.frame_delay = False
        self.profile = None
        self.component_by_ib = {}
        self.modes = {}            # 合并部件 ib -> "local" | "merged"
        self.foreign_slots = {}    # ib -> set[int]
        self.dependencies = {}     # 用合并调色板的部件 ib -> set[部件 ib]
        self.participants = []     # 需要把骨骼导入骨骼池的部件 ib (按抓帧顺序)
        self.delayed_layout = []   # 统一延迟: 脸和独立部件在骨骼池里的区间
        self.delayed_copies = {}   # 统一延迟: 蒙皮前换成自己上一帧骨骼表的部件 ib -> (起始骨骼, 骨骼数)
        self.placed_ibs = []       # 统一延迟: 画的时候换成上一帧物体矩阵的部件 (蓝图里的部件、脸)
        self.face_parts_written = set()  # 统一延迟: 已经写了绘制行的 (脸 ib, 首索引)
        self.warnings = []
        if not self.enabled:
            return
        self.frame_delay = bool(GlobalProterties.zzz_merged_frame_delay())

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
        if self.frame_delay:
            print("[ZZMI骨骼合并] 统一延迟一帧: 整个角色 (含脸) 用上一帧的骨骼蒙皮")

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

        if self.frame_delay:
            self._plan_frame_delay(participants)
            self.participants = sorted(participants, key=lambda ib_hash: self.component_by_ib[ib_hash]["pose_order"])
            return
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

    def _plan_frame_delay(self, participants: set):
        '''
        统一延迟: 蓝图里的合并部件全部参与 (只用自己骨骼的也要换成上一帧的骨骼表)，
        脸 (不论在不在蓝图里，游戏都照常画它) 和蓝图里的独立部件把骨骼导入骨骼池末尾。
        '''
        exported = {drawib_model.draw_ib for drawib_model in self.drawib_model_list}
        for ib_hash, mode in self.modes.items():
            participants.add(ib_hash)
            if mode == "local":
                component = self.component_by_ib[ib_hash]
                self.delayed_copies[ib_hash] = (int(component["vg_offset"]), int(component["vg_count"]))

        extras = []
        for component in sorted(self.profile["components"], key=lambda component: component["pose_order"]):
            role = component["role"]
            if role == zms.ROLE_MERGED:
                continue
            if role == zms.ROLE_STANDALONE and component["ib_hash"] not in exported:
                continue
            pose_vertex_count = component["pose_vertex_count"] or component["vertex_count"]
            if not component["blend_vb"] or not pose_vertex_count or not component["palette_bone_count"]:
                print(
                    "[ZZMI骨骼合并][提示] " + component["ib_hash"]
                    + " 缺少权重缓冲 hash、顶点数或骨骼表大小，不做延迟 (仍用当帧骨骼)。请重新提取。"
                )
                continue
            extras.append(component)
        self.delayed_layout = zms.delayed_pool_layout(
            self.profile, [component["component_id"] for component in extras]
        )
        for component, entry in zip(extras, self.delayed_layout):
            participants.add(component["ib_hash"])
            self.delayed_copies[component["ib_hash"]] = (entry["offset"], entry["count"])
            print(
                "[ZZMI骨骼合并] " + component["ib_hash"]
                + (" (脸部)" if component["role"] == zms.ROLE_SHAPE_KEYED else " (独立部件)")
                + " 也换成上一帧的骨骼表"
            )

        # 每个部件的骨骼表在它自己根骨骼的空间里，游戏用这个根当帧的物体矩阵摆放部件，而这些
        # 根是骨骼 (胯、胸、头)。只延迟骨骼表时，上一帧的姿势会被根这一帧的移动和转动带着晃，
        # 各部件之间也对不上；所以画的部件都换成上一帧的物体矩阵，整个角色画成上一帧的样子。
        self.placed_ibs = [
            ib_hash for ib_hash in self.participants_order(participants)
            if (ib_hash in self.modes or ib_hash in self.delayed_copies)
            and (ib_hash in exported or self.component_by_ib[ib_hash]["role"] == zms.ROLE_SHAPE_KEYED)
        ]
        print("[ZZMI骨骼合并] 画的时候也用上一帧的物体矩阵: " + ", ".join(self.placed_ibs))

    def participants_order(self, ib_hashes) -> list:
        return sorted(ib_hashes, key=lambda ib_hash: self.component_by_ib[ib_hash]["pose_order"])

    def _pool_bone_total(self) -> int:
        return int(self.profile["bones_count"]) + sum(entry["count"] for entry in self.delayed_layout)

    def _import_bone_count(self, ib_hash: str) -> int:
        component = self.component_by_ib[ib_hash]
        if component["role"] == zms.ROLE_MERGED:
            return int(component["vg_count"])
        return self.delayed_copies[ib_hash][1]

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

    # 游戏画完一个物体不会重新绑定 vs-cb1 (只改写同一块缓冲)，所以换上的物体常量必须紧贴着
    # 这次绘制换回去，否则后面的绘制都会用它。ZZMI 的 legacy 模式下 IB 覆盖段一次绘制会被
    # 检查两次，所以换上、绘制、换回放在同一个 IB 覆盖段里，每次执行都成对。

    def place_begin_lines(self, ib_hash: str) -> list:
        '''统一延迟: 我们自己绘制的部件，IB 覆盖段里 drawindexed 之前的行'''
        if not self.enabled or not self.frame_delay or ib_hash not in self.placed_ibs:
            return []
        return ["run = CommandListZZMS_PlaceBegin_" + _ini_name(ib_hash)]

    def place_end_lines(self, ib_hash: str) -> list:
        '''统一延迟: 我们自己绘制的部件，IB 覆盖段里 drawindexed 之后的行'''
        if not self.enabled or not self.frame_delay or ib_hash not in self.placed_ibs:
            return []
        return ["run = CommandListZZMS_PlaceEnd_" + _ini_name(ib_hash)]

    def face_draw_lines(self, ib_hash: str, first_index: int) -> list:
        '''
        统一延迟: 脸 (游戏自己画) 每个部分的 IB 覆盖段末尾要加的行: 换上物体常量，
        原样重发这次绘制 (draw = from_caller)，换回，跳过原绘制。贴图行要在它前面。
        '''
        if not self.enabled or not self.frame_delay or ib_hash not in self.placed_ibs:
            return []
        self.face_parts_written.add((ib_hash, int(first_index)))
        return [
            "run = CommandListZZMS_PlaceBegin_" + _ini_name(ib_hash),
            "if $zzms_ready",
            "\tdraw = from_caller",
            "\thandling = skip",
            "endif",
            "run = CommandListZZMS_PlaceEnd_" + _ini_name(ib_hash),
        ]

    def _add_face_part_sections(self, ini_builder):
        '''没有贴图覆盖段的脸部分 (脸不在蓝图里，或这部分没有标记贴图) 单独生成 IB 覆盖段'''
        sections = M_IniSection(M_SectionType.TextureOverrideIB)
        for ib_hash in self.placed_ibs:
            component = self.component_by_ib[ib_hash]
            if component["role"] != zms.ROLE_SHAPE_KEYED:
                continue
            for unique_str in component["unique_strs"]:
                _, index_count, first_index = unique_str.rsplit("-", 2)
                if (ib_hash, int(first_index)) in self.face_parts_written:
                    continue
                lines = self.face_draw_lines(ib_hash, int(first_index))
                sections.append("[TextureOverride_ZZMS_Face_" + _ini_name(ib_hash) + "_" + first_index + "]")
                sections.append("hash = " + ib_hash)
                sections.append("match_first_index = " + first_index)
                sections.append("match_index_count = " + index_count)
                for line in lines:
                    sections.append(line)
                sections.new_line()
        ini_builder.append_section(sections)

    def _skin_command_list(self, ib_hash: str, exported_drawib, slot_dict: dict) -> list:
        name = _ini_name(ib_hash)
        lines = ["[CommandListZZMS_Skin_" + name + "]"]
        lines.append("ResourceZZMS_Input = ref vs-t0")
        lines.append("run = CustomShaderZZMS_Import_" + name)
        if self.frame_delay:
            lines.append("$zzms_imported = 1")
            if exported_drawib is not None and self.modes.get(ib_hash) == "merged":
                lines.append("if $zzms_ready")
                lines.append("\trun = CustomShaderZZMS_PaletteDelayed_" + name)
                lines.append("else")
                lines.append("\trun = CustomShaderZZMS_Palette_" + name)
                lines.append("endif")
                lines.append("vs-t0 = ref ResourceZZMS_Palette_" + name)
            elif ib_hash in self.delayed_copies:
                # 第一帧还没有上一帧的骨骼: 不换，照常用游戏当帧的骨骼表
                lines.append("if $zzms_ready")
                lines.append("\trun = CustomShaderZZMS_Delay_" + name)
                lines.append("\tvs-t0 = ref ResourceZZMS_Delayed_" + name)
                lines.append("endif")
            if exported_drawib is not None:
                lines.append(slot_dict["Position"] + " = Resource" + ib_hash + "Position")
                lines.append(slot_dict["Blend"] + " = Resource" + ib_hash + "Blend")
                lines.append("draw = " + str(exported_drawib.draw_number) + ", 0")
            lines.append("")
            return lines
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

    def _add_face_delay_sections(self, ini_builder):
        '''
        统一延迟下的脸 (形态键部件): 游戏照常画它，只在它自己的预蒙皮绘制上导入骨骼，
        并把骨骼表换成上一帧的那份；顶点 (表情) 仍是游戏当帧算出来的。
        '''
        sections = M_IniSection(M_SectionType.TextureOverrideVB)
        for ib_hash in self.participants:
            component = self.component_by_ib[ib_hash]
            if component["role"] != zms.ROLE_SHAPE_KEYED:
                continue
            pose_vertex_count = component["pose_vertex_count"] or component["vertex_count"]
            sections.append("; " + ib_hash + " (脸部，统一延迟)")
            sections.append("[TextureOverride_ZZMS_Delay_" + _ini_name(ib_hash) + "]")
            sections.append("hash = " + component["blend_vb"])
            sections.append("if DRAW_TYPE == 1 && VERTEX_COUNT == " + str(pose_vertex_count))
            sections.append("\trun = CommandListZZMS_Skin_" + _ini_name(ib_hash))
            sections.append("endif")
            sections.new_line()
        ini_builder.append_section(sections)

    def _add_frame_delay_present(self, ini_builder):
        '''$zzms_ready: 骨骼池上一帧那份已经可用；角色有一帧没出现就重新从当帧开始'''
        constants = M_IniSection(M_SectionType.Constants)
        constants.SectionName = "Constants"
        constants.append("global $zzms_imported = 0")
        constants.append("global $zzms_ready = 0")
        ini_builder.append_section(constants)

        present = M_IniSection(M_SectionType.Present)
        present.SectionName = "Present"
        present.append("if $zzms_imported")
        present.append("\trun = CustomShaderZZMS_Advance")
        for ib_hash in self.placed_ibs:
            present.append("\trun = CustomShaderZZMS_PlaceAdvance_" + _ini_name(ib_hash))
        if self.placed_ibs:
            present.append("\trun = CustomShaderZZMS_ShiftClear")
        present.append("\t$zzms_ready = 1")
        present.append("else")
        present.append("\t$zzms_ready = 0")
        present.append("endif")
        present.append("$zzms_imported = 0")
        ini_builder.append_section(present)

    @staticmethod
    def _custom_shader(lines: list, name: str, shader_file: str, sources: list, target: str, groups: int):
        lines.append("[" + name + "]")
        lines.append("cs = Shaders/" + shader_file)
        for slot, resource in enumerate(sources):
            lines.append("cs-t" + str(slot) + " = ref " + resource)
        lines.append("cs-u0 = ref " + target)
        lines.append("dispatch = " + str(groups) + ", 1, 1")
        lines.append("cs-u0 = null")
        for slot in range(len(sources)):
            lines.append("cs-t" + str(slot) + " = null")
        lines.append("")

    def _place_lines(self, ib_hash: str) -> list:
        '''
        统一延迟: 画这个部件时换掉它的物体矩阵 (vs-cb1)。第一帧还没有上一帧的骨骼，
        照常用游戏的矩阵，只记下它给下一帧用。
        ResourceZZMS_Shift 是这一帧整个角色挪动的距离，所有部件共用，在 [Present] 清零。
        '''
        name = _ini_name(ib_hash)
        lines = [
            "[CommandListZZMS_PlaceBegin_" + name + "]",
            "if $zzms_ready",
            "\trun = CustomShaderZZMS_Place_" + name,
            "\tResourceZZMS_GameCb = ref vs-cb1",
            "\tvs-cb1 = copy ResourceZZMS_PlaceCb_" + name,
            "else",
            "\trun = CustomShaderZZMS_PlaceRecord_" + name,
            "endif",
            "",
            "[CommandListZZMS_PlaceEnd_" + name + "]",
            "if $zzms_ready",
            "\tvs-cb1 = ref ResourceZZMS_GameCb",
            "endif",
            "",
        ]
        for section_name, mode, shift in (("CustomShaderZZMS_Place_", "ResourceZZMS_PlaceOn", True),
                                          ("CustomShaderZZMS_PlaceRecord_", "ResourceZZMS_PlaceOff", False)):
            lines += [
                "[" + section_name + name + "]",
                "cs = Shaders/" + zms.PLACE_SHADER_FILENAME,
                "cs-cb0 = ref vs-cb1",
                "cs-t0 = ref ResourceZZMS_History_" + name,
                "cs-t1 = ref " + mode,
                "cs-u0 = ref ResourceZZMS_PlaceCb_" + name,
                "cs-u1 = ref ResourceZZMS_Record_" + name,
            ]
            if shift:
                lines.append("cs-u2 = ref ResourceZZMS_Shift")
            lines += [
                "dispatch = 1, 1, 1",
                "cs-u0 = null",
                "cs-u1 = null",
            ]
            if shift:
                lines.append("cs-u2 = null")
            lines += [
                "cs-t0 = null",
                "cs-t1 = null",
                "cs-cb0 = null",
                "",
            ]
        self._custom_shader(
            lines, "CustomShaderZZMS_PlaceAdvance_" + name, zms.DELAY_SHADER_FILENAME,
            ["ResourceZZMS_Record_" + name, "ResourceZZMS_PlaceAdvanceParams"],
            "ResourceZZMS_History_" + name, 1,
        )
        return lines

    @staticmethod
    def _rw_buffer(resources, name: str, rows: int):
        resources.append("[" + name + "]")
        resources.append("type = RWBuffer")
        resources.append("format = R32G32B32A32_FLOAT")
        resources.append("array = " + str(rows))
        resources.append("bind_flags = shader_resource unordered_access")
        resources.new_line()

    @staticmethod
    def _uint4_data(resources, name: str, values):
        resources.append("[" + name + "]")
        resources.append("type = Buffer")
        resources.append("format = R32G32B32A32_UINT")
        resources.append("data = " + " ".join(str(value) for value in values))
        resources.new_line()

    def _add_frame_delay_runtime_sections(self, ini_builder, exported_by_ib: dict):
        bones_count = int(self.profile["bones_count"])
        pool_total = self._pool_bone_total()
        layout = self.delayed_layout
        self._add_face_delay_sections(ini_builder)
        self._add_face_part_sections(ini_builder)
        self._add_frame_delay_present(ini_builder)

        command_lists = M_IniSection(M_SectionType.CommandList)
        command_lists.append("; Merged skeleton runtime (Zenless Zone Zero), skinned with the bones of the previous frame")
        for ib_hash in self.participants:
            # 脸在蓝图里也不导出网格，游戏照常画它
            exported = None if self.is_shape_keyed(ib_hash) else exported_by_ib.get(ib_hash)
            slot_dict = exported.d3d11GameType.CategoryExtractSlotDict if exported is not None else {}
            for line in self._skin_command_list(ib_hash, exported, slot_dict):
                command_lists.append(line)

        lines = []
        for ib_hash in self.participants:
            name = _ini_name(ib_hash)
            self._custom_shader(
                lines, "CustomShaderZZMS_Import_" + name, zms.IMPORT_SHADER_FILENAME,
                ["ResourceZZMS_Input", "ResourceZZMS_Components", "ResourceZZMS_Params_" + name],
                "ResourceZZMS_Pool", _ceil_groups(self._import_bone_count(ib_hash)),
            )
        for ib_hash, mode in self.modes.items():
            if mode != "merged":
                continue
            name = _ini_name(ib_hash)
            for section_name, pool in (("CustomShaderZZMS_Palette_", "ResourceZZMS_Pool"),
                                       ("CustomShaderZZMS_PaletteDelayed_", "ResourceZZMS_PoolPrev")):
                self._custom_shader(
                    lines, section_name + name, zms.PALETTE_SHADER_FILENAME,
                    [pool, "ResourceZZMS_Components", "ResourceZZMS_Params_" + name],
                    "ResourceZZMS_Palette_" + name, _ceil_groups(bones_count),
                )
        for ib_hash, (_, count) in self.delayed_copies.items():
            name = _ini_name(ib_hash)
            self._custom_shader(
                lines, "CustomShaderZZMS_Delay_" + name, zms.DELAY_SHADER_FILENAME,
                ["ResourceZZMS_PoolPrev", "ResourceZZMS_DelayParams_" + name],
                "ResourceZZMS_Delayed_" + name, _ceil_groups(count),
            )
        self._custom_shader(
            lines, "CustomShaderZZMS_Advance", zms.DELAY_SHADER_FILENAME,
            ["ResourceZZMS_Pool", "ResourceZZMS_AdvanceParams"],
            "ResourceZZMS_PoolPrev", _ceil_groups(pool_total),
        )
        for ib_hash in self.placed_ibs:
            lines += self._place_lines(ib_hash)
        if self.placed_ibs:
            lines += [
                "[CustomShaderZZMS_ShiftClear]",
                "cs = Shaders/" + zms.PLACE_SHADER_FILENAME,
                "cs-t1 = ref ResourceZZMS_ShiftClearParams",
                "cs-u2 = ref ResourceZZMS_Shift",
                "dispatch = 1, 1, 1",
                "cs-u2 = null",
                "cs-t1 = null",
                "",
            ]
        for line in lines:
            command_lists.append(line)
        ini_builder.append_section(command_lists)

        resources = M_IniSection(M_SectionType.ResourceBuffer)
        self._rw_buffer(resources, "ResourceZZMS_Pool", pool_total * 3)
        self._rw_buffer(resources, "ResourceZZMS_PoolPrev", pool_total * 3)
        for ib_hash, mode in self.modes.items():
            if mode == "merged":
                self._rw_buffer(resources, "ResourceZZMS_Palette_" + _ini_name(ib_hash), bones_count * 3)
        for ib_hash, (first, count) in self.delayed_copies.items():
            self._rw_buffer(resources, "ResourceZZMS_Delayed_" + _ini_name(ib_hash), count * 3)
            self._uint4_data(resources, "ResourceZZMS_DelayParams_" + _ini_name(ib_hash), [first, count, 0, 0])
        self._uint4_data(resources, "ResourceZZMS_AdvanceParams", [0, pool_total, 0, 0])
        self._uint4_data(resources, "ResourceZZMS_Components", zms.runtime_component_table(self.profile, layout))
        for ib_hash in self.participants:
            component = self.component_by_ib[ib_hash]
            self._uint4_data(
                resources, "ResourceZZMS_Params_" + _ini_name(ib_hash),
                zms.runtime_params(self.profile, component["component_id"], layout),
            )
        if self.placed_ibs:
            # 物体常量 29 行；记录 8 行，按 3 行一组复制，所以留 9 行
            for ib_hash in self.placed_ibs:
                name = _ini_name(ib_hash)
                self._rw_buffer(resources, "ResourceZZMS_PlaceCb_" + name, zms.OBJECT_CB_REGISTERS)
                self._rw_buffer(resources, "ResourceZZMS_Record_" + name, 9)
                self._rw_buffer(resources, "ResourceZZMS_History_" + name, 9)
            self._uint4_data(resources, "ResourceZZMS_PlaceOn", [1, 0, 0, 0])
            self._uint4_data(resources, "ResourceZZMS_PlaceOff", [0, 0, 0, 0])
            self._uint4_data(resources, "ResourceZZMS_PlaceAdvanceParams", [0, 3, 0, 0])
            self._uint4_data(resources, "ResourceZZMS_ShiftClearParams", [2, 0, 0, 0])
            resources.append("[ResourceZZMS_Shift]")
            resources.append("type = RWBuffer")
            resources.append("format = R32_FLOAT")
            resources.append("array = 4")
            resources.append("bind_flags = shader_resource unordered_access")
            resources.new_line()
            resources.append("[ResourceZZMS_GameCb]")
            resources.new_line()
        resources.append("[ResourceZZMS_Input]")
        resources.new_line()
        ini_builder.append_section(resources)

    def add_runtime_sections(self, ini_builder):
        if not self.enabled:
            return
        self._add_hidden_component_sections(ini_builder)
        if not self.uses_runtime():
            return
        exported_by_ib = {drawib_model.draw_ib: drawib_model for drawib_model in self.drawib_model_list}
        if self.frame_delay:
            self._add_frame_delay_runtime_sections(ini_builder, exported_by_ib)
            return
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
        zms.write_runtime_shaders(os.path.join(mod_folder, "Shaders"), frame_delay=self.frame_delay)
