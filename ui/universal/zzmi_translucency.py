"""Translate the shared translucency node to ZZMI's TTLib 2.3 draw API."""

import os
import re
import shutil

from ...blueprint.node_translucency import MASK_DIFFUSE_A, MASK_TEXTURE
from ...common.global_config import GlobalConfig
from ...common.m_ini_builder import M_IniSection, M_SectionType
from ...common.m_ini_helper import M_IniHelper


class ZZMITranslucency:
    def __init__(self):
        self.mask_textures = {}

    def _mask_resource(self, drawcall):
        translucency = drawcall.translucency
        source_path = translucency["texture"]
        if not source_path:
            raise ValueError(
                "半透明节点选择了 " + "RGB"[translucency["channel"]]
                + " 通道，但没有填遮罩贴图 (" + drawcall.obj_name + ")。"
            )
        if not os.path.isfile(source_path):
            raise ValueError("找不到半透明遮罩贴图: " + source_path)
        filename = os.path.basename(source_path)
        source_key = os.path.normcase(os.path.abspath(source_path))
        record = self.mask_textures.get(filename.lower())
        if record is not None:
            if record["source_key"] != source_key:
                raise ValueError(
                    "两张不同的半透明遮罩贴图同名，导出到 Textures 会冲突: "
                    + record["source_path"] + " / " + source_path
                )
            return record["resource"]
        used_names = {record["resource"].lower() for record in self.mask_textures.values()}
        base = "Resource-TTLMask-" + re.sub(r"[^0-9A-Za-z_.-]", "_", os.path.splitext(filename)[0])
        resource = base
        suffix = 2
        while resource.lower() in used_names:
            resource = base + "-" + str(suffix)
            suffix += 1
        self.mask_textures[filename.lower()] = {
            "source_key": source_key, "source_path": source_path,
            "filename": filename, "resource": resource,
        }
        return resource

    def _draw_lines(self, drawcall, offsets, diffuse_resource):
        translucency = drawcall.translucency
        native_draw = drawcall.get_drawindexed_str(offsets)
        if not translucency:
            return [native_draw]

        mask_resource = ""
        if translucency["mask"] == MASK_DIFFUSE_A:
            if not diffuse_resource:
                raise ValueError(
                    "ZZMI 半透明节点选择了 A 通道，但绘制部件没有可用的 Slot 漫射贴图标记 ("
                    + drawcall.obj_name + ")。请标记 DiffuseMap 并开启自动贴图。"
                )
            mask_resource = diffuse_resource
        elif translucency["mask"] == MASK_TEXTURE:
            mask_resource = self._mask_resource(drawcall)

        # SlotFix classifies the current pass even when there are no texture marks.
        lines = [r"run = CommandList\ZZMI\SetTextures"]
        if mask_resource:
            lines += [
                r"Resource\TTL\TransparencyTex = ref " + mask_resource,
                r"$\TTL\mask_channel = " + str(translucency["channel"]),
                # The node uses opacity masks; TTLib uses transparency masks and
                # multiplies by alpha even when a texture was supplied.
                r"$\TTL\mask_invert = 1",
                r"$\TTL\mask_discard_threshold = 1",
                r"$\TTL\alpha = 1",
            ]
        else:
            lines += [
                r"Resource\TTL\TransparencyTex = null",
                r"$\TTL\alpha = " + format(float(translucency["alpha"]), ".3f").rstrip("0").rstrip("."),
            ]
        offset = drawcall.index_offset if offsets is None else offsets.get(drawcall.obj_name, drawcall.index_offset)
        lines += [
            r"$\TTL\use_auto_draw = 0",
            r"$\TTL\_1 = " + str(drawcall.index_count),
            r"$\TTL\_2 = " + str(offset),
            r"$\TTL\_3 = 0",
            r"run = CommandList\TTL\Draw",
            # TTLib resets material parameters, but not its public texture/args.
            r"Resource\TTL\TransparencyTex = null",
            r"$\TTL\_1 = 0",
            r"$\TTL\_2 = 0",
            r"$\TTL\_3 = 0",
            # Material postprocessing locates FX resets by this draw marker.
            "; ZZMI TTL: " + native_draw,
        ]
        return lines

    def draw_lines(self, drawcalls, obj_name_draw_offset_dict=None, diffuse_resource=""):
        if not any(getattr(drawcall, "translucency", None) for drawcall in drawcalls):
            return M_IniHelper.get_drawindexed_str_list(drawcalls, obj_name_draw_offset_dict)
        groups = {}
        for drawcall in drawcalls:
            groups.setdefault(drawcall.get_condition_str(), []).append(drawcall)
        lines = []
        for condition, group in groups.items():
            indent = "  " if condition else ""
            if condition:
                lines.append("if " + condition)
            for drawcall in group:
                display_name = str(drawcall.obj_name or getattr(drawcall, "display_name", "") or "")
                lines.append(indent + "; [mesh:" + display_name + "] [vertex_count:" + str(drawcall.vertex_count) + "]")
                lines.extend(indent + line for line in self._draw_lines(drawcall, obj_name_draw_offset_dict, diffuse_resource))
            if condition:
                lines.append("endif")
            lines.append("")
        return lines

    def append_resources(self, ini_builder):
        section = M_IniSection(M_SectionType.ResourceTexture)
        for record in self.mask_textures.values():
            section.append("[" + record["resource"] + "]")
            section.append("filename = Textures/" + record["filename"])
            section.new_line()
        ini_builder.append_section(section)

    def copy_textures(self):
        if not self.mask_textures:
            return
        folder = GlobalConfig.path_generatemod_texture_folder(draw_ib="ZZMITTL")
        for record in self.mask_textures.values():
            destination = os.path.join(folder, record["filename"])
            # Like the existing EFMI node, preserve textures edited in the mod.
            if not os.path.exists(destination):
                shutil.copy2(record["source_path"], destination)
