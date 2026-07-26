# -*- coding: utf-8 -*-
'''
LoyalTools 贴图标记面板。

功能: 为提取并导入的模型标记贴图 (DiffuseMap/NormalMap/LightMap/自定义)，
写入对应 SubmeshJson 的 TextureMarkUpInfoList，
供导出管线 (texture_metadata_helper.py + m_ini_helper.py) 生成贴图 ini 并拷贝贴图文件。

界面参考 SSMT4 的贴图标记页: 候选贴图按绘制调用分组 (每个 DrawIB 由多个
渲染通道绘制，几何区域相同但各通道绑定自己的 ps-t 贴图集)，
列表条目带缩略图 / 插槽 / 尺寸，列表下方展示选中贴图的大图预览与详细信息。
'''
import json
import os
import subprocess

import bpy
import bpy.utils.previews

from ..common import texture_mark_helper
from ..common.global_config import GlobalConfig


# ----------------------------------------------------------------------
# 模块级缓存
# ----------------------------------------------------------------------

# 绘制调用枚举缓存: Blender 动态枚举 items 回调返回的字符串必须保持引用，
# 否则会触发已知的字符串被垃圾回收导致乱码/崩溃问题，
# 因此与 ui_panel_sword / global_properties 中的动态枚举一样使用模块级缓存。
_CALL_FILTER_ALL_ITEM = ('ALL', "全部", "显示所有绘制调用的候选贴图 (按 文件名+插槽 去重)")
_texmark_call_enum_items_cache: list[tuple[str, str, str]] = [_CALL_FILTER_ALL_ITEM]
_texmark_call_enum_cache_key = None

# 目标模型枚举缓存 (切换 DrawIB 用): 同样使用模块级缓存保持字符串引用，
# 缓存键为当前工作空间目录路径 (工作空间切换 / 点击刷新时重建)。
_TEXMARK_SUBMESH_ACTIVE_ITEM = ('ACTIVE', "跟随选中物体", "跟随当前选中物体解析出的 unique_str (默认行为)")
_texmark_submesh_enum_items_cache: list[tuple[str, str, str]] = [_TEXMARK_SUBMESH_ACTIVE_ITEM]
_texmark_submesh_enum_cache_key = None

# 缩略图预览集合 (bpy.utils.previews): register() 创建, unregister() 移除。
# 预览条目无法重复加载，因此以贴图绝对路径 (normcase) 为键，仅加载新键。
_texmark_preview_collection = None

# 刷新时记录的 {文件名: icon_id}，供 UIList 行与详情大图读取
_texmark_icon_id_map: dict[str, int] = {}


# ----------------------------------------------------------------------
# 内部工具函数
# ----------------------------------------------------------------------

def _resolve_object_submesh(obj) -> tuple[str, str, str]:
    '''从物体解析 (unique_str, json_path, type_folder)。

    Raises:
        ValueError: 携带中文提示信息
    '''
    if obj is None:
        raise ValueError("请先选中一个已导入的模型对象")

    unique_str = texture_mark_helper.resolve_unique_str_from_object_name(obj.name)
    if not unique_str:
        raise ValueError(
            "物体 '" + obj.name + "' 名称无法解析出 unique_str，"
            + "请选中名称形如 DrawIB-IndexCount-FirstIndex 的已导入模型对象"
        )

    try:
        json_path, type_folder = texture_mark_helper.locate_submesh_json(unique_str)
    except Exception as ex:
        raise ValueError("无法定位工作空间提取数据（请先提取并导入模型）: " + str(ex))

    return unique_str, json_path, type_folder


def _collect_selected_unique_strs(context) -> tuple[list[str], list[str]]:
    '''收集所有选中物体各自的 unique_str (去重，活动物体优先)。

    Returns:
        (unique_str 列表, 无法解析的物体名称列表)
    '''
    ordered_objects = []
    if context.active_object is not None:
        ordered_objects.append(context.active_object)
    for obj in context.selected_objects:
        if obj not in ordered_objects:
            ordered_objects.append(obj)

    unique_str_list: list[str] = []
    unresolved_names: list[str] = []
    for obj in ordered_objects:
        unique_str = texture_mark_helper.resolve_unique_str_from_object_name(obj.name)
        if not unique_str:
            unresolved_names.append(obj.name)
            continue
        if unique_str not in unique_str_list:
            unique_str_list.append(unique_str)

    return unique_str_list, unresolved_names


def _get_target_submesh_value(context) -> str:
    '''读取面板目标模型选择 ('ACTIVE' 或工作空间内的某个 unique_str)。'''
    try:
        return context.scene.loyal_texmark_props.target_submesh or 'ACTIVE'
    except Exception:
        return 'ACTIVE'


def _peek_target_unique_str(context):
    '''解析面板当前目标的 unique_str (不校验提取数据，失败返回 None)。

    target_submesh 为 'ACTIVE' 时从活动物体名称解析，否则直接返回所选 unique_str。
    '''
    target = _get_target_submesh_value(context)
    if target != 'ACTIVE':
        return target

    active_object = getattr(context, "active_object", None) if context is not None else None
    if active_object is None:
        return None
    return texture_mark_helper.resolve_unique_str_from_object_name(active_object.name)


def _resolve_target_submesh(context) -> tuple[str, str, str]:
    '''解析面板目标的 (unique_str, json_path, type_folder)。

    target_submesh 为 'ACTIVE' 时跟随活动物体 (当前默认行为)，
    否则直接使用所选 unique_str，无需选中任何物体。

    Raises:
        ValueError: 携带中文提示信息
    '''
    target = _get_target_submesh_value(context)
    if target == 'ACTIVE':
        return _resolve_object_submesh(context.active_object)

    try:
        json_path, type_folder = texture_mark_helper.locate_submesh_json(target)
    except Exception as ex:
        raise ValueError("无法定位工作空间提取数据（请刷新贴图列表或重新提取模型）: " + str(ex))

    return target, json_path, type_folder


def _rebuild_submesh_enum_cache() -> None:
    '''重建目标模型枚举缓存 (来自当前工作空间的 Import.json，回退为子目录扫描)。

    显示标签为 unique_str，工作空间 Config.json 中配置了别名时附加 "(别名)"。
    '''
    global _texmark_submesh_enum_items_cache, _texmark_submesh_enum_cache_key

    workspace_folder = ""
    try:
        workspace_folder = GlobalConfig.path_workspace_folder() or ""
    except Exception:
        workspace_folder = ""

    unique_str_list: list[str] = []
    if workspace_folder and os.path.isdir(workspace_folder):
        # 优先 Import.json ({unique_str: gametype_name})
        try:
            import_json_path = os.path.join(workspace_folder, "Import.json")
            if os.path.isfile(import_json_path):
                with open(import_json_path, 'r', encoding='utf-8') as f:
                    import_json = json.load(f)
                if isinstance(import_json, dict):
                    unique_str_list = [
                        str(key) for key in import_json.keys()
                        if len(str(key).split("-")) >= 3
                    ]
        except Exception:
            unique_str_list = []

        # 回退: 扫描符合 "<DrawIB>-<IndexCount>-<FirstIndex>" 命名的子目录
        if not unique_str_list:
            try:
                for dir_entry in os.scandir(workspace_folder):
                    if dir_entry.is_dir() and len(dir_entry.name.split("-")) >= 3:
                        unique_str_list.append(dir_entry.name)
            except OSError:
                pass

    drawib_aliasname_dict: dict = {}
    try:
        from ..common.workspace_helper import WorkSpaceHelper
        drawib_aliasname_dict = WorkSpaceHelper.get_drawib_aliasname_dict()
    except Exception:
        drawib_aliasname_dict = {}

    enum_items = [_TEXMARK_SUBMESH_ACTIVE_ITEM]
    for unique_str in sorted(set(unique_str_list)):
        alias_name = str(drawib_aliasname_dict.get(unique_str.split("-")[0], "")).strip()
        display_label = unique_str + " (" + alias_name + ")" if alias_name else unique_str
        enum_items.append((
            unique_str,
            display_label,
            "以工作空间子网格 " + unique_str + " 为标记目标 (无需选中物体)",
        ))

    _texmark_submesh_enum_items_cache = enum_items
    _texmark_submesh_enum_cache_key = os.path.normcase(os.path.normpath(workspace_folder)) if workspace_folder else ""


def _get_target_submesh_enum_items(self, context):
    '''目标模型枚举 items 回调 (返回模块级缓存，工作空间变化时重建)。'''
    global _texmark_submesh_enum_items_cache

    try:
        workspace_folder = ""
        try:
            workspace_folder = GlobalConfig.path_workspace_folder() or ""
        except Exception:
            workspace_folder = ""
        cache_key = os.path.normcase(os.path.normpath(workspace_folder)) if workspace_folder else ""
        if cache_key != _texmark_submesh_enum_cache_key:
            _rebuild_submesh_enum_cache()
    except Exception:
        pass

    return _texmark_submesh_enum_items_cache


def _on_target_submesh_update(self, context):
    '''切换目标模型后立即刷新候选贴图列表 (失败时静默，交由刷新按钮报告)。'''
    try:
        _refresh_texmark_items(context)
    except Exception:
        pass


def _slot_sort_value(slot: str) -> int:
    '''"ps-t<N>" -> N，无插槽信息的排最后。'''
    if slot and slot.startswith("ps-t"):
        digits = slot[4:]
        if digits.isdigit():
            return int(digits)
    return 9999


def _rebuild_call_enum_cache(unique_str, type_folder: str) -> None:
    '''重建绘制调用枚举缓存 (调用 id 升序，附带每组贴图数量提示)。'''
    global _texmark_call_enum_items_cache, _texmark_call_enum_cache_key

    enum_items = [_CALL_FILTER_ALL_ITEM]
    try:
        if type_folder:
            call_map = texture_mark_helper.list_candidate_textures_by_call(type_folder)
            for call_id in call_map.keys():
                if not call_id:
                    continue
                enum_items.append((
                    call_id,
                    "绘制调用 " + call_id,
                    "绘制调用 " + call_id + " 绑定的 " + str(len(call_map[call_id])) + " 张 ps-t 贴图",
                ))
    except Exception:
        pass

    _texmark_call_enum_items_cache = enum_items
    _texmark_call_enum_cache_key = unique_str


def _get_call_filter_enum_items(self, context):
    '''绘制调用筛选枚举 items 回调 (返回模块级缓存，目标模型变化时重建)。'''
    global _texmark_call_enum_items_cache

    try:
        unique_str = _peek_target_unique_str(context) if context is not None else None

        if unique_str != _texmark_call_enum_cache_key:
            type_folder = ""
            if unique_str:
                try:
                    _, type_folder = texture_mark_helper.locate_submesh_json(unique_str)
                except Exception:
                    type_folder = ""
            _rebuild_call_enum_cache(unique_str, type_folder)
    except Exception:
        pass

    return _texmark_call_enum_items_cache


def _on_call_filter_update(self, context):
    '''切换绘制调用后立即刷新候选贴图列表 (失败时静默，交由刷新按钮报告)。'''
    try:
        _refresh_texmark_items(context)
    except Exception:
        pass


def _load_texture_thumbnail(type_folder: str, filename: str) -> int:
    '''加载贴图缩略图到预览集合，返回 icon_id (失败返回 0)。

    预览条目无法重新加载，同一路径复用已有条目；
    DDS (BC7 等压缩格式) 能否解码取决于 Blender 构建，失败时回退无缩略图。
    '''
    if _texmark_preview_collection is None:
        return 0

    file_path = os.path.join(type_folder, filename)
    if not os.path.isfile(file_path):
        return 0

    preview_key = os.path.normcase(os.path.abspath(file_path))
    try:
        if preview_key in _texmark_preview_collection:
            preview = _texmark_preview_collection[preview_key]
        else:
            preview = _texmark_preview_collection.load(preview_key, file_path, 'IMAGE')
        return preview.icon_id
    except Exception:
        return 0


def _refresh_texmark_items(context) -> tuple[str, int]:
    '''刷新贴图候选列表 (以面板目标模型为准，按绘制调用筛选)。

    Returns:
        (unique_str, 候选贴图数量)

    Raises:
        ValueError: 携带中文提示信息
    '''
    scene = context.scene
    settings = scene.loyal_texmark_props
    _rebuild_submesh_enum_cache()
    unique_str, json_path, type_folder = _resolve_target_submesh(context)

    _rebuild_call_enum_cache(unique_str, type_folder)
    call_map = texture_mark_helper.list_candidate_textures_by_call(type_folder)

    selected_call = 'ALL'
    try:
        selected_call = settings.call_filter or 'ALL'
    except Exception:
        pass

    if selected_call != 'ALL' and selected_call in call_map:
        candidate_list = list(call_map[selected_call])
    else:
        # 全部: 跨绘制调用聚合，按 (文件名, 插槽) 去重 (调用 id 升序，先出现者优先)
        candidate_list = []
        seen_keys: set[tuple[str, str]] = set()
        for call_id in call_map.keys():
            for candidate in call_map[call_id]:
                dedupe_key = (candidate.get("filename", ""), candidate.get("slot", ""))
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                candidate_list.append(candidate)

    candidate_list.sort(
        key=lambda candidate: (_slot_sort_value(candidate.get("slot", "")), candidate.get("filename", ""))
    )

    mark_list = texture_mark_helper.read_marks(json_path)

    _texmark_icon_id_map.clear()
    scene.loyal_texmark_items.clear()
    for candidate in candidate_list:
        item = scene.loyal_texmark_items.add()
        item.filename = candidate["filename"]
        item.slot = candidate["slot"]
        item.hash = candidate["hash"]
        item.format = candidate["format"]
        item.width = int(candidate.get("width", 0) or 0)
        item.height = int(candidate.get("height", 0) or 0)
        item.call_id = str(candidate.get("call_id", "") or "")
        # MarkFileName 为 SSMT4 风格命名副本，不再等于候选文件名:
        # 按 hash > SourceFileName > 旧版 MarkFileName 优先级匹配已标记状态
        matched_mark = texture_mark_helper.find_mark_for_candidate(
            mark_list, candidate["filename"], candidate["hash"]
        )
        item.marked_name = matched_mark.get("MarkName", "") if matched_mark else ""

        icon_id = _load_texture_thumbnail(type_folder, candidate["filename"])
        if icon_id:
            _texmark_icon_id_map[candidate["filename"]] = icon_id

    if scene.loyal_texmark_index >= len(scene.loyal_texmark_items):
        scene.loyal_texmark_index = max(0, len(scene.loyal_texmark_items) - 1)

    return unique_str, len(candidate_list)


def _get_selected_item(context):
    '''获取列表中当前选中的贴图条目，未选中时返回 None。'''
    scene = context.scene
    items = scene.loyal_texmark_items
    index = scene.loyal_texmark_index
    if index < 0 or index >= len(items):
        return None
    return items[index]


# ----------------------------------------------------------------------
# 属性组
# ----------------------------------------------------------------------

class LoyalTexMarkItem(bpy.types.PropertyGroup):
    filename: bpy.props.StringProperty(name="文件名", default="")  # type: ignore
    slot: bpy.props.StringProperty(name="插槽", default="")  # type: ignore
    hash: bpy.props.StringProperty(name="Hash", default="")  # type: ignore
    format: bpy.props.StringProperty(name="格式", default="")  # type: ignore
    width: bpy.props.IntProperty(name="宽度", default=0)  # type: ignore
    height: bpy.props.IntProperty(name="高度", default=0)  # type: ignore
    call_id: bpy.props.StringProperty(name="绘制调用", default="")  # type: ignore
    marked_name: bpy.props.StringProperty(name="已标记名称", default="")  # type: ignore


class LoyalTexMarkSettings(bpy.types.PropertyGroup):
    target_submesh: bpy.props.EnumProperty(
        name="目标模型",
        description="选择贴图标记的目标模型: 跟随选中物体，或直接指定当前工作空间内的某个子网格 (无需选中物体)",
        items=_get_target_submesh_enum_items,
        update=_on_target_submesh_update,
    )  # type: ignore
    call_filter: bpy.props.EnumProperty(
        name="绘制调用",
        description="按绘制调用筛选候选贴图 (每个绘制调用绑定自己的 ps-t 贴图集)",
        items=_get_call_filter_enum_items,
        update=_on_call_filter_update,
    )  # type: ignore
    mark_name: bpy.props.EnumProperty(
        name="标记名称",
        description="写入 TextureMarkUpInfoList 的 MarkName",
        items=[
            ('DiffuseMap', 'DiffuseMap', '漫反射贴图'),
            ('NormalMap', 'NormalMap', '法线贴图'),
            ('LightMap', 'LightMap', '光照贴图'),
            ('CUSTOM', '自定义', '使用自定义标记名称'),
        ],
        default='DiffuseMap',
    )  # type: ignore
    custom_mark_name: bpy.props.StringProperty(
        name="自定义标记名称",
        description="选择「自定义」时使用的 MarkName",
        default="",
    )  # type: ignore
    mark_type: bpy.props.EnumProperty(
        name="标记方式",
        description="写入 TextureMarkUpInfoList 的 MarkType",
        items=[
            ('Slot', 'Slot 插槽方式', '按插槽替换贴图 (ini 中直接绑定 ps-t 插槽)'),
            ('Hash', 'Hash 方式', '按贴图 Hash 生成 TextureOverride 小节'),
        ],
        default='Slot',
    )  # type: ignore


# ----------------------------------------------------------------------
# UIList
# ----------------------------------------------------------------------

class LOYAL_UL_TexMarkList(bpy.types.UIList):
    '''候选贴图列表 (插槽优先展示，与 SSMT4 贴图标记页一致)。

    行布局: [缩略图] [ps-t<N> 插槽 (来自 calls 分组的真实插槽)] [hash] [宽x高] [已标记]。
    文件名不在列表行中展示 (避免误导)，仅在列表下方的详情框内展示。
    '''

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        icon_id = _texmark_icon_id_map.get(item.filename, 0)
        # 插槽优先；无插槽信息时退化为 hash / 文件名，避免整行空白
        primary_text = item.slot or item.hash or item.filename
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if icon_id:
                row.label(text="", icon_value=icon_id)
            else:
                row.label(text="", icon='TEXTURE')
            row.label(text=primary_text)
            if item.slot and item.hash:
                row.label(text=item.hash)
            if item.width > 0 and item.height > 0:
                row.label(text=str(item.width) + "x" + str(item.height))
            if item.marked_name:
                row.label(text="已标记: " + item.marked_name, icon='CHECKMARK')
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            if icon_id:
                layout.label(text=primary_text, icon_value=icon_id)
            else:
                layout.label(text=primary_text, icon='TEXTURE')


# ----------------------------------------------------------------------
# 操作符
# ----------------------------------------------------------------------

class LOYAL_OT_TexMarkRefresh(bpy.types.Operator):
    bl_idname = "loyal.texmark_refresh"
    bl_label = "刷新贴图列表"
    bl_description = "根据目标模型 (跟随选中物体或面板顶部指定的模型) 读取提取目录中的候选贴图并刷新列表 (按绘制调用筛选)"

    def execute(self, context):
        try:
            unique_str, candidate_count = _refresh_texmark_items(context)
        except ValueError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        except Exception as ex:
            self.report({'ERROR'}, "刷新贴图列表失败: " + str(ex))
            return {'CANCELLED'}

        if candidate_count == 0:
            self.report({'WARNING'}, "未在 " + unique_str + " 的提取目录中找到候选贴图")
        else:
            self.report({'INFO'}, "已刷新贴图列表: " + unique_str + "，共 " + str(candidate_count) + " 张候选贴图")
        return {'FINISHED'}


class LOYAL_OT_TexMarkMark(bpy.types.Operator):
    bl_idname = "loyal.texmark_mark"
    bl_label = "标记所选贴图"
    bl_description = "将选中的贴图标记写入目标模型的 SubmeshJson (TextureMarkUpInfoList)；跟随选中物体时写入所有选中物体各自的 SubmeshJson"

    def execute(self, context):
        scene = context.scene
        settings = scene.loyal_texmark_props

        item = _get_selected_item(context)
        if item is None:
            self.report({'ERROR'}, "请先刷新贴图列表并在列表中选择一个贴图条目")
            return {'CANCELLED'}

        if settings.mark_name == 'CUSTOM':
            mark_name = settings.custom_mark_name.strip()
            if not mark_name:
                self.report({'ERROR'}, "自定义标记名称不能为空")
                return {'CANCELLED'}
        else:
            mark_name = settings.mark_name

        if settings.mark_type == 'Slot' and not item.slot:
            self.report({'ERROR'}, "该贴图缺少插槽信息，无法使用 Slot 插槽方式标记，请改用 Hash 方式或选择带 ps-t 插槽信息的贴图")
            return {'CANCELLED'}
        if settings.mark_type == 'Hash' and not item.hash:
            self.report({'ERROR'}, "该贴图缺少 Hash 信息，无法使用 Hash 方式标记，请改用 Slot 插槽方式或选择带 Hash 信息的贴图")
            return {'CANCELLED'}

        target_submesh = _get_target_submesh_value(context)
        if target_submesh == 'ACTIVE':
            unique_str_list, unresolved_names = _collect_selected_unique_strs(context)
            if not unique_str_list:
                self.report({'ERROR'}, "请先选中一个已导入的模型对象（名称形如 DrawIB-IndexCount-FirstIndex），或在面板顶部指定目标模型")
                return {'CANCELLED'}
        else:
            # 面板顶部指定了目标模型: 仅写入该子网格的 SubmeshJson，无需选中物体
            unique_str_list, unresolved_names = [target_submesh], []

        # 候选列表来自面板目标 (跟随活动物体或指定子网格) 的提取目录，
        # 该目录即命名副本的复制来源 (候选源文件可能仅存在于此目录)
        try:
            _, _, source_type_folder = _resolve_target_submesh(context)
        except ValueError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}

        success_count = 0
        error_message_list = list(unresolved_names)
        marked_filename_example = ""
        for unique_str in unique_str_list:
            try:
                json_path, type_folder = texture_mark_helper.locate_submesh_json(unique_str)
                # 每个目标子网格在自己的提取目录内以自己的 unique_str 生成 SSMT4 风格
                # 命名副本 "<unique_str>-<MarkName>.<后缀>" (已存在则覆盖)；
                # 候选源文件不在目标目录时从来源提取目录跨目录复制
                copy_source_folder = None
                if not os.path.isfile(os.path.join(type_folder, item.filename)):
                    copy_source_folder = source_type_folder
                mark_filename = texture_mark_helper.make_named_mark_copy(
                    type_folder, item.filename, unique_str, mark_name,
                    source_folder=copy_source_folder,
                )
                mark = {
                    "MarkName": mark_name,
                    "MarkType": settings.mark_type,
                    "MarkHash": item.hash,
                    "MarkSlot": item.slot,
                    "MarkFileName": mark_filename,
                    "SourceFileName": item.filename,
                }
                texture_mark_helper.write_mark(json_path, mark)
                success_count += 1
                if not marked_filename_example:
                    marked_filename_example = mark_filename
            except Exception as ex:
                error_message_list.append(unique_str + ": " + str(ex))

        try:
            _refresh_texmark_items(context)
        except Exception:
            pass

        if success_count == 0:
            self.report({'ERROR'}, "标记写入失败: " + "；".join(error_message_list))
            return {'CANCELLED'}

        for error_message in error_message_list:
            print("LOYAL_OT_TexMarkMark: 部分标记写入失败: " + error_message)

        message = (
            "已标记 " + mark_name + " -> " + item.filename
            + "（命名副本 " + marked_filename_example
            + "，写入 " + str(success_count) + " 个 SubmeshJson）"
        )
        if error_message_list:
            message = message + "，另有 " + str(len(error_message_list)) + " 个失败（详见控制台）"
        self.report({'INFO'}, message)
        return {'FINISHED'}


class LOYAL_OT_TexMarkUnmark(bpy.types.Operator):
    bl_idname = "loyal.texmark_unmark"
    bl_label = "取消标记"
    bl_description = "从目标模型的 SubmeshJson 中移除选中贴图对应的标记条目；跟随选中物体时处理所有选中物体各自的 SubmeshJson"

    def execute(self, context):
        item = _get_selected_item(context)
        if item is None:
            self.report({'ERROR'}, "请先刷新贴图列表并在列表中选择一个贴图条目")
            return {'CANCELLED'}

        target_submesh = _get_target_submesh_value(context)
        if target_submesh == 'ACTIVE':
            unique_str_list, unresolved_names = _collect_selected_unique_strs(context)
            if not unique_str_list:
                self.report({'ERROR'}, "请先选中一个已导入的模型对象（名称形如 DrawIB-IndexCount-FirstIndex），或在面板顶部指定目标模型")
                return {'CANCELLED'}
        else:
            unique_str_list, unresolved_names = [target_submesh], []

        removed_count = 0
        error_message_list = list(unresolved_names)
        for unique_str in unique_str_list:
            try:
                json_path, _ = texture_mark_helper.locate_submesh_json(unique_str)
                # MarkFileName 为 SSMT4 风格命名副本，不再等于候选文件名:
                # 先按 hash / SourceFileName / 旧版 MarkFileName 匹配出选中候选对应的标记，
                # 再按该标记自身的 MarkFileName (兜底 Slot/Hash) 身份移除 json 条目；
                # 不删除命名副本文件本身 (用户可能已手动编辑过)
                mark_list = texture_mark_helper.read_marks(json_path)
                matched_mark = texture_mark_helper.find_mark_for_candidate(mark_list, item.filename, item.hash)
                if matched_mark is None:
                    continue
                mark_identity = {
                    "MarkFileName": matched_mark.get("MarkFileName", ""),
                    "MarkSlot": matched_mark.get("MarkSlot", ""),
                    "MarkHash": matched_mark.get("MarkHash", ""),
                }
                removed_count += texture_mark_helper.remove_mark(json_path, mark_identity)
            except Exception as ex:
                error_message_list.append(unique_str + ": " + str(ex))

        try:
            _refresh_texmark_items(context)
        except Exception:
            pass

        for error_message in error_message_list:
            print("LOYAL_OT_TexMarkUnmark: 部分标记移除失败: " + error_message)

        if removed_count == 0:
            self.report({'WARNING'}, "未找到 " + item.filename + " 对应的标记条目")
            return {'FINISHED'}

        self.report({'INFO'}, "已移除 " + item.filename + " 的标记（共移除 " + str(removed_count) + " 条）")
        return {'FINISHED'}


class LOYAL_OT_TexMarkOpenFolder(bpy.types.Operator):
    bl_idname = "loyal.texmark_open_folder"
    bl_label = "打开贴图文件夹"
    bl_description = "在文件管理器中打开目标模型对应的 TYPE_ 提取目录"

    def execute(self, context):
        try:
            _, _, type_folder = _resolve_target_submesh(context)
        except ValueError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}

        try:
            if os.name == 'nt':
                os.startfile(type_folder)
            else:
                subprocess.Popen(["xdg-open", type_folder])
        except OSError as ex:
            self.report({'ERROR'}, "打开贴图文件夹失败: " + str(ex))
            return {'CANCELLED'}

        self.report({'INFO'}, "已打开贴图文件夹: " + type_folder)
        return {'FINISHED'}


class LOYAL_OT_TexMarkPreview(bpy.types.Operator):
    bl_idname = "loyal.texmark_preview"
    bl_label = "预览"
    bl_description = "加载选中的贴图并在图像编辑器中预览"

    def execute(self, context):
        item = _get_selected_item(context)
        if item is None:
            self.report({'ERROR'}, "请先刷新贴图列表并在列表中选择一个贴图条目")
            return {'CANCELLED'}

        try:
            _, _, type_folder = _resolve_target_submesh(context)
        except ValueError as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}

        texture_file_path = os.path.join(type_folder, item.filename)
        if not os.path.isfile(texture_file_path):
            self.report({'ERROR'}, "贴图文件不存在: " + texture_file_path)
            return {'CANCELLED'}

        try:
            image = bpy.data.images.load(texture_file_path, check_existing=True)
        except Exception as ex:
            self.report({'ERROR'}, "贴图加载失败: " + str(ex))
            return {'CANCELLED'}

        shown_in_editor = False
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = image
                    area.tag_redraw()
                    shown_in_editor = True
                    break
            if shown_in_editor:
                break

        if shown_in_editor:
            self.report({'INFO'}, "已在图像编辑器中预览: " + item.filename)
        else:
            self.report({'INFO'}, "已加载图像数据块: " + image.name + "（当前没有打开的图像编辑器，可手动切换查看）")
        return {'FINISHED'}


# ----------------------------------------------------------------------
# 面板
# ----------------------------------------------------------------------

class LOYAL_PT_TexMarkPanel(bpy.types.Panel):
    bl_label = "标记贴图"
    bl_idname = "VIEW3D_PT_LoyalTools_TexMark"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LoyalTools'
    bl_order = 2

    @classmethod
    def poll(cls, context):
        if not hasattr(context.scene, 'herta_show_toolkit'):
            return True
        return not context.scene.herta_show_toolkit

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.loyal_texmark_props

        layout.prop(settings, "target_submesh", text="目标模型")

        unique_str = _peek_target_unique_str(context)
        if unique_str:
            layout.label(text="当前模型: " + unique_str, icon='OBJECT_DATA')
        else:
            layout.label(text="请选中一个已导入的模型对象，或在上方指定目标模型", icon='INFO')

        layout.prop(settings, "call_filter", text="绘制调用")
        layout.operator(LOYAL_OT_TexMarkRefresh.bl_idname, text="刷新贴图列表", icon='FILE_REFRESH')

        layout.template_list(
            "LOYAL_UL_TexMarkList", "",
            scene, "loyal_texmark_items",
            scene, "loyal_texmark_index",
            rows=6,
        )

        # 选中贴图详情 (大图预览 + 信息)
        item = _get_selected_item(context)
        if item is not None:
            detail_box = layout.box()
            icon_id = _texmark_icon_id_map.get(item.filename, 0)
            if icon_id:
                detail_box.template_icon(icon_value=icon_id, scale=8.0)
            else:
                detail_box.label(text="无法生成缩略图 (可用「预览」按钮在图像编辑器中查看)", icon='IMAGE_DATA')
            detail_box.label(text="文件名: " + item.filename, icon='FILE_IMAGE')
            if item.slot:
                detail_box.label(text="插槽: " + item.slot)
            if item.width > 0 and item.height > 0:
                detail_box.label(text="尺寸: " + str(item.width) + "x" + str(item.height))
            if item.format:
                detail_box.label(text="格式: " + item.format)
            if item.hash:
                detail_box.label(text="Hash: " + item.hash)
            if item.call_id:
                detail_box.label(text="绘制调用: " + item.call_id)
            if item.marked_name:
                detail_box.label(text="已标记: " + item.marked_name, icon='CHECKMARK')

        mark_box = layout.box()
        mark_box.prop(settings, "mark_name", text="标记名称")
        if settings.mark_name == 'CUSTOM':
            mark_box.prop(settings, "custom_mark_name", text="自定义名称")
        mark_box.prop(settings, "mark_type", text="标记方式")

        mark_row = layout.row(align=True)
        mark_row.operator(LOYAL_OT_TexMarkMark.bl_idname, text="标记所选贴图", icon='CHECKMARK')
        mark_row.operator(LOYAL_OT_TexMarkUnmark.bl_idname, text="取消标记", icon='X')

        utility_row = layout.row(align=True)
        utility_row.operator(LOYAL_OT_TexMarkOpenFolder.bl_idname, text="打开贴图文件夹", icon='FILE_FOLDER')
        utility_row.operator(LOYAL_OT_TexMarkPreview.bl_idname, text="预览", icon='IMAGE_DATA')


# ----------------------------------------------------------------------
# 注册
# ----------------------------------------------------------------------

_classes = (
    LoyalTexMarkItem,
    LoyalTexMarkSettings,
    LOYAL_UL_TexMarkList,
    LOYAL_OT_TexMarkRefresh,
    LOYAL_OT_TexMarkMark,
    LOYAL_OT_TexMarkUnmark,
    LOYAL_OT_TexMarkOpenFolder,
    LOYAL_OT_TexMarkPreview,
    LOYAL_PT_TexMarkPanel,
)


def register():
    global _texmark_preview_collection

    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.loyal_texmark_items = bpy.props.CollectionProperty(type=LoyalTexMarkItem)
    bpy.types.Scene.loyal_texmark_index = bpy.props.IntProperty(name="贴图列表索引", default=0)
    bpy.types.Scene.loyal_texmark_props = bpy.props.PointerProperty(type=LoyalTexMarkSettings)

    if _texmark_preview_collection is None:
        _texmark_preview_collection = bpy.utils.previews.new()


def unregister():
    global _texmark_preview_collection, _texmark_call_enum_cache_key
    global _texmark_call_enum_items_cache
    global _texmark_submesh_enum_items_cache, _texmark_submesh_enum_cache_key

    if _texmark_preview_collection is not None:
        bpy.utils.previews.remove(_texmark_preview_collection)
        _texmark_preview_collection = None
    _texmark_icon_id_map.clear()
    _texmark_call_enum_items_cache = [_CALL_FILTER_ALL_ITEM]
    _texmark_call_enum_cache_key = None
    _texmark_submesh_enum_items_cache = [_TEXMARK_SUBMESH_ACTIVE_ITEM]
    _texmark_submesh_enum_cache_key = None

    if hasattr(bpy.types.Scene, 'loyal_texmark_props'):
        del bpy.types.Scene.loyal_texmark_props
    if hasattr(bpy.types.Scene, 'loyal_texmark_index'):
        del bpy.types.Scene.loyal_texmark_index
    if hasattr(bpy.types.Scene, 'loyal_texmark_items'):
        del bpy.types.Scene.loyal_texmark_items

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
