# -*- coding: utf-8 -*-
'''
LoyalTools DrawIB 提取面板。
功能：按行填写 DrawIB 和别名 (SSMT4风格表格)，从 3dmigoto 帧分析 Dump 中提取模型，
生成 SSMT 风格工作空间并导入 Blender，导入后自动接入当前工作空间的 SSMT 蓝图节点树。
提取核心逻辑位于 extract/dump_workspace_extractor.py (与 Blender 解耦，可独立运行)。
'''
import os
import re

import bpy

from ..common.global_config import GlobalConfig
from ..common.global_properties import GlobalProterties
from ..common.logic_name import LogicName
from ..common.non_mirror_workflow import NonMirrorWorkflowHelper
from ..common.ssmt_import_helper import SSMTImportHelper
from ..common.workspace_helper import WorkSpaceHelper
from ..utils.collection_utils import CollectionColor, CollectionUtils
from .ui_prefix_quick_ops import PrefixQuickOpsHelper


# DrawIB 必须是 8 位十六进制 (3dmigoto CRC32C hash)
DRAWIB_PATTERN = re.compile(r'^[0-9a-fA-F]{8}$')


def parse_draw_ib_input(raw_text: str) -> list[str]:
    '''
    解析用户输入的 DrawIB 字符串，支持逗号/分号/空格分隔多个。
    非法条目 (非 8 位十六进制) 会抛出 ValueError (中文消息)。
    (旧版单行输入的解析逻辑，现仅用于把旧字段内容迁移到分行列表)
    '''
    parts = [part for part in re.split(r'[,;\s]+', (raw_text or "").strip()) if part]
    if not parts:
        raise ValueError("请先填写 DrawIB (8位十六进制 hash，可填多个，用逗号或空格分隔)。")

    ib_hashes = []
    for part in parts:
        if not DRAWIB_PATTERN.match(part):
            raise ValueError("DrawIB 格式错误: \"" + part + "\" (必须是8位十六进制，例如 a1b2c3d4)。")
        ib_hash = part.lower()
        if ib_hash not in ib_hashes:
            ib_hashes.append(ib_hash)
    return ib_hashes


def parse_optional_draw_ib_input(raw_text: str, field_label: str) -> list[str]:
    '''解析可留空的 DrawIB 列表；用于骨骼合并的游戏原网格覆盖。'''
    raw_text = (raw_text or "").strip()
    if raw_text == "":
        return []
    try:
        return parse_draw_ib_input(raw_text)
    except ValueError as exc:
        raise ValueError(field_label + "格式错误：" + str(exc)) from exc


def migrate_legacy_draw_ib_field(scene, props) -> None:
    '''
    向后兼容：DrawIB 列表为空且旧版单行 draw_ib 字段非空时，
    把旧字段内容解析后迁移为列表行 (旧字段内容非法时抛出 ValueError)。
    '''
    items = scene.loyal_extract_ib_items
    if len(items) > 0:
        return
    legacy_text = (props.draw_ib or "").strip()
    if legacy_text == "":
        return
    for ib_hash in parse_draw_ib_input(legacy_text):
        item = items.add()
        item.draw_ib = ib_hash
    scene.loyal_extract_ib_index = 0


def gather_ib_rows_input(scene, props) -> tuple[list[str], dict[str, str]]:
    '''
    汇总 DrawIB 列表的所有行：跳过空行、逐行校验格式、去重，
    返回 (ib_hashes, {ib_hash: 别名})，别名字典只含填写了别名的行。
    非法行抛出 ValueError (中文消息，标明行号)；没有任何有效行时也抛出 ValueError。
    '''
    migrate_legacy_draw_ib_field(scene, props)

    ib_hashes: list[str] = []
    aliases: dict[str, str] = {}
    for row_index, item in enumerate(scene.loyal_extract_ib_items):
        raw_ib = (item.draw_ib or "").strip()
        if raw_ib == "":
            # 空行跳过 (允许用户预留空行)
            continue
        if not DRAWIB_PATTERN.match(raw_ib):
            raise ValueError(
                "DrawIB列表第 " + str(row_index + 1) + " 行格式错误: \"" + raw_ib
                + "\" (必须是8位十六进制，例如 a1b2c3d4)。"
            )
        ib_hash = raw_ib.lower()
        if ib_hash not in ib_hashes:
            ib_hashes.append(ib_hash)
        alias = (item.alias or "").strip()
        if alias and ib_hash not in aliases:
            aliases[ib_hash] = alias

    if not ib_hashes:
        raise ValueError("请先在DrawIB列表中添加至少一行有效的DrawIB (8位十六进制 hash)，别名可留空。")
    return ib_hashes, aliases


def resolve_dump_folder(props) -> str:
    '''解析帧分析Dump目录为绝对路径，未填写或不存在时抛出 ValueError (中文消息)。'''
    dump_folder = (props.frame_dump_folder or "").strip()
    if dump_folder == "":
        raise ValueError("请先选择帧分析Dump目录 (包含log.txt的FrameAnalysis文件夹)。")
    dump_folder = bpy.path.abspath(dump_folder)
    if not os.path.isdir(dump_folder):
        raise ValueError("帧分析Dump目录不存在: " + dump_folder)
    return dump_folder


def get_or_create_workspace_import_collection(context):
    '''
    获取 (或创建) 以工作空间命名的红色导入集合。
    命名和颜色与"一键导入SSMT工作空间内容"保持一致
    (红色 COLOR_01 = SSMT识别用的工作空间集合色，见 WorkSpaceHelper.create_and_get_workspace_collection)；
    已存在同名集合时直接复用 (并确保已链接进当前场景)，避免重复提取产生 "xxx.001" 副本集合。
    '''
    collection_name = GlobalConfig.get_workspace_name()
    if not collection_name:
        workspace_folder = GlobalConfig.path_workspace_folder()
        if workspace_folder:
            collection_name = os.path.basename(workspace_folder.rstrip("\\/"))
    if not collection_name:
        raise ValueError("未解析到工作空间名称，无法创建导入集合。")

    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = CollectionUtils.create_new_collection(
            collection_name=collection_name,
            color_tag=CollectionColor.Red,
        )
        context.scene.collection.children.link(collection)
        return collection

    # 复用已有集合：确保颜色为红色，且已链接进当前场景 (可能只存在于其它场景)
    collection.color_tag = CollectionColor.Red
    scene_root_collection = context.scene.collection
    if collection not in scene_root_collection.children_recursive:
        scene_root_collection.children.link(collection)
    return collection


def create_new_workspace_import_collection(context):
    '''
    总是新建一个红色集合（不复用已有同名集合）。
    Blender 的 collections.new() 在名称已存在时会自动追加 .001/.002 等数字后缀，
    保证每次"导入"都产生独立的集合。
    '''
    collection_name = GlobalConfig.get_workspace_name()
    if not collection_name:
        workspace_folder = GlobalConfig.path_workspace_folder()
        if workspace_folder:
            collection_name = os.path.basename(workspace_folder.rstrip("\\/"))
    if not collection_name:
        raise ValueError("未解析到工作空间名称，无法创建导入集合。")

    collection = bpy.data.collections.new(collection_name)  # Blender 自动编号
    collection.color_tag = CollectionColor.Red
    context.scene.collection.children.link(collection)
    return collection


# 帧模型解析较慢 (大型Dump约10-20秒)，同一个Dump目录缓存一个提取器实例，
# log.txt 的修改时间变化时自动失效 (只保留最近一个，避免占用过多内存)
_EXTRACTOR_CACHE: dict = {}


def get_cached_extractor(dump_folder: str):
    '''获取 (或复用缓存的) DumpWorkspaceExtractor 实例'''
    from ..extract.dump_workspace_extractor import DumpWorkspaceExtractor

    log_path = os.path.join(dump_folder, "log.txt")
    try:
        cache_key = (dump_folder, os.path.getmtime(log_path))
    except OSError:
        cache_key = (dump_folder, None)

    cached = _EXTRACTOR_CACHE.get("entry")
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    extractor = DumpWorkspaceExtractor(dump_folder)
    _EXTRACTOR_CACHE["entry"] = (cache_key, extractor)
    return extractor


# ----------------------------------------------------------------------
# 蓝图节点自动连接 (节点填写方式与 ui_func_import_ssmt 的一键导入保持一致，
# 区别是这里按需幂等更新 (重复提取同一DrawIB不会重复创建节点)，
# 且所有物体接入同一个分组节点，不按DrawIB拆分多个分组)
# ----------------------------------------------------------------------

# 各类节点的横向摆放位置：物体信息节点 -> 分组节点 -> 输出节点
_BLUEPRINT_INFO_NODE_X = 0.0
_BLUEPRINT_GROUP_NODE_X = 400.0
_BLUEPRINT_OUTPUT_NODE_X = 800.0
_BLUEPRINT_NODE_Y_GAP = 200.0


def _append_socket_into_group(tree, from_socket, group_node):
    '''
    把一个输出口接到分组节点的最后一个空闲输入口。
    输入口不足时先扩充一个 (和 SSMTNode_Object_Group.update 的扩充逻辑一致)。
    '''
    if not group_node.inputs:
        group_node.inputs.new('SSMTSocketObject', "Input 1")
    if group_node.inputs[-1].is_linked:
        group_node.inputs.new('SSMTSocketObject', "Input " + str(len(group_node.inputs) + 1))
    tree.links.new(from_socket, group_node.inputs[-1])


def _resolve_single_object_group(tree, output_node, group_label, group_y):
    '''
    解析本次连线使用的唯一分组节点 (所有导入的物体都接入同一个分组，由它连到输出节点)：
    1. 优先复用已直连输出节点输入口的分组节点 (保持用户现有链路)；
    2. 其次复用树中已有的第一个分组节点；
    3. 都没有时新建一个 (标签 = 工作空间名，摆在物体信息节点和输出节点之间)。
    '''
    if output_node is not None and output_node.inputs and output_node.inputs[0].is_linked:
        upstream_node = output_node.inputs[0].links[0].from_node
        if getattr(upstream_node, "bl_idname", "") == 'SSMTNode_Object_Group':
            return upstream_node

    for node in tree.nodes:
        if getattr(node, "bl_idname", "") == 'SSMTNode_Object_Group':
            return node

    group_node = tree.nodes.new('SSMTNode_Object_Group')
    group_node.label = group_label
    group_node.location = (_BLUEPRINT_GROUP_NODE_X, group_y)
    return group_node


def wire_objects_into_blueprint(context, imported_entries, drawib_aliasname_dict,
                               tree_name_override=None, merged_skeleton=False):
    '''
    (提取自动导入成功后调用) 把导入的物体接入当前工作空间的 SSMT 蓝图节点树。
    树的命名和选中方式与"一键导入SSMT工作空间内容"保持一致
    (树名 = 工作空间名，选中后"基础信息"面板的"生成所选蓝图Mod"可直接使用)。

    所有导入的物体都接入同一个分组节点 (Object_Group)，由它直连输出节点 (Generate Mod)，
    不按DrawIB拆分多个分组、也不插入Merge汇总组。

    幂等设计：物体信息节点按 object_name 复用 (仅更新 object_id 等属性)、
    分组节点复用树中已有的 (优先取已直连输出节点的那个)、输出节点在树中缺失时才创建；
    新节点从已有节点下方开始摆放，避免堆叠在 (0, 0)。

    imported_entries: [(unique_str, obj), ...]
    返回蓝图名称；任何失败只打印中文警告并返回 None (不影响已完成的导入)。
    '''
    try:
        if tree_name_override is not None:
            # 强制新建蓝图树（用于"导入"按钮的重复导入场景）：
            # tree_name_override 是本次新建集合的实际名称（已含 .001 等后缀），
            # 直接用它命名新树，不查找也不复用已有树。
            tree_name = tree_name_override
            tree = None
        else:
            tree_name = GlobalConfig.get_workspace_name()
            if not tree_name:
                workspace_folder = GlobalConfig.path_workspace_folder()
                if workspace_folder:
                    tree_name = os.path.basename(workspace_folder.rstrip("\\/"))
            if not tree_name:
                print("[LoyalTools提取] 未解析到工作空间名称，跳过蓝图节点自动连接。")
                return None

            tree = bpy.data.node_groups.get(tree_name)
            if tree is not None and getattr(tree, "bl_idname", "") != 'SSMTBlueprintTreeType':
                print("[LoyalTools提取] 已存在同名的非SSMT蓝图节点树 \"" + tree_name + "\"，跳过蓝图节点自动连接。")
                return None

        if tree is None:
            tree = bpy.data.node_groups.new(name=tree_name, type='SSMTBlueprintTreeType')
            tree.use_fake_user = True

        # 将制作流程写入蓝图本身，导出时不再依赖用户额外添加 Cross IB 节点。
        # 物体上的合并标记仍作为旧蓝图的兼容检测依据。
        tree["LoyalTools:EFMIMergedSkeleton"] = bool(merged_skeleton)

        # 和打开蓝图窗口/一键导入相同的选中方式，保证"生成所选蓝图Mod"立即可用
        try:
            from ..blueprint.export_helper import BlueprintExportHelper
            BlueprintExportHelper.set_runtime_blueprint_tree(tree)
        except Exception as e:
            print("[LoyalTools提取] 记录运行时蓝图失败 (可忽略): " + repr(e))
        global_properties = getattr(getattr(context, "scene", None), "global_properties", None)
        if global_properties and getattr(global_properties, "selected_blueprint_name", "") != tree.name:
            global_properties.selected_blueprint_name = tree.name

        # 收集已有节点用于复用，保证重复提取时幂等
        info_nodes_by_object_name = {}
        output_node = None
        for node in tree.nodes:
            node_idname = getattr(node, "bl_idname", "")
            if node_idname == 'SSMTNode_Object_Info':
                info_nodes_by_object_name.setdefault(node.object_name, node)
            elif node_idname == 'SSMTNode_Result_Output' and output_node is None:
                output_node = node

        # 新节点从已有节点最低处的下方开始摆放
        if len(tree.nodes) > 0:
            start_y = min(node.location.y for node in tree.nodes) - _BLUEPRINT_NODE_Y_GAP
        else:
            start_y = 0.0
        info_y = start_y

        # 所有物体统一接入这一个分组节点 (新建时标签 = 工作空间名)
        group_node = _resolve_single_object_group(tree, output_node, tree_name, start_y)

        object_count = 0
        for unique_str, obj in imported_entries:
            if obj is None or getattr(obj, "type", "") != 'MESH':
                continue
            name_parts = unique_str.split('-')
            draw_ib = name_parts[0]

            info_node = info_nodes_by_object_name.get(obj.name)
            if info_node is None:
                info_node = tree.nodes.new('SSMTNode_Object_Info')
                info_node.location = (_BLUEPRINT_INFO_NODE_X, info_y)
                info_y -= _BLUEPRINT_NODE_Y_GAP
                info_nodes_by_object_name[obj.name] = info_node

            # 节点信息填写方式和一键导入保持一致 (已有节点也刷新一遍 object_id 等)
            info_node.object_name = obj.name
            info_node.object_id = str(obj.as_pointer())
            info_node.draw_ib = draw_ib
            info_node.component = name_parts[1] if len(name_parts) >= 2 else "1"
            info_node.alias_name = WorkSpaceHelper.get_object_display_name(
                unique_str,
                drawib_aliasname_dict=drawib_aliasname_dict,
            )
            info_node.label = obj.name
            object_count += 1

            # 输出口已有连线时视为已接入蓝图，不重复连接
            if info_node.outputs and not info_node.outputs[0].is_linked:
                _append_socket_into_group(tree, info_node.outputs[0], group_node)

        if output_node is None:
            output_node = tree.nodes.new('SSMTNode_Result_Output')
            output_node.label = "Generate Mod"
            output_node.location = (_BLUEPRINT_OUTPUT_NODE_X, start_y)

        # 确保分组直连输出节点；输出口已被其它节点占用 (用户自己的连线) 时不改动
        if group_node.outputs and output_node.inputs:
            output_input = output_node.inputs[0]
            if not output_input.is_linked:
                tree.links.new(group_node.outputs[0], output_input)
            elif output_input.links[0].from_node != group_node:
                print(
                    "[LoyalTools提取] 输出节点的输入口已被其它连线占用，保留现有连线，"
                    "分组 \"" + (group_node.label or group_node.name) + "\" 未自动接入输出节点。"
                )

        # 触发分组节点扩充空输入口，保持和手动连线后一致的外观
        if hasattr(group_node, "update"):
            group_node.update()

        print(
            "[LoyalTools提取] 蓝图 \"" + tree.name + "\" 节点已连接: "
            + str(object_count) + " 个物体接入分组 \"" + (group_node.label or group_node.name) + "\""
        )
        return tree.name
    except Exception as e:
        print("[LoyalTools提取] 蓝图节点自动连接失败 (不影响已导入的模型): " + repr(e))
        import traceback
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------
# 属性
# ----------------------------------------------------------------------

class LoyalExtractIBItem(bpy.types.PropertyGroup):
    '''DrawIB提取列表的一行: DrawIB + 别名'''
    draw_ib: bpy.props.StringProperty(
        name="DrawIB",
        description="要提取的DrawIB (索引缓冲区hash，8位十六进制，例如 a1b2c3d4)",
        default="",
    ) # type: ignore

    alias: bpy.props.StringProperty(
        name="别名",
        description="该DrawIB的别名 (可留空)。填写后导入的物体会重命名为 \"前缀.别名\"，并写入工作空间Config.json",
        default="",
    ) # type: ignore


class LoyalExtractProperties(bpy.types.PropertyGroup):
    workflow_mode: bpy.props.EnumProperty(
        name="制作流程",
        description="普通制作保持原流程；骨骼合并会自动从整帧识别角色并建立全局顶点组",
        items=[
            ('STANDARD', "普通制作", "按 DrawIB 列表提取，使用 LoyalTools 原有制作流程"),
            ('MERGED_SKELETON', "骨骼合并", "自动识别帧中的角色组件，使用 EFMI 1.4.1 骨骼合并流程"),
        ],
        default='STANDARD',
    ) # type: ignore

    frame_dump_folder: bpy.props.StringProperty(
        name="帧分析Dump目录",
        description="3dmigoto帧分析生成的FrameAnalysis-xxx文件夹路径 (需包含log.txt，且dump选项开启buf+txt)",
        subtype='DIR_PATH',
        default="",
    ) # type: ignore

    merged_original_mesh_ibs: bpy.props.StringProperty(
        name="游戏原网格 IB",
        description=(
            "可选；填写需要按 EFMI CPU posed 方式处理的 IB hash，多个用逗号或空格分隔。"
            "这些组件不导出自定义网格/权重，游戏中始终绘制原网格，但仍可自动替换贴图"
        ),
        default="",
    ) # type: ignore

    # 旧版单行DrawIB输入字段，仅为向后兼容保留 (面板上已不显示)：
    # 提取时若分行列表为空而此字段非空，会自动把内容迁移为列表行
    draw_ib: bpy.props.StringProperty(
        name="DrawIB",
        description="旧版单行DrawIB输入 (已由分行列表替代，仅作兼容保留)",
        default="",
        options={'HIDDEN'},
    ) # type: ignore

    # 以下两个属性已不在 UI 中显示，保留仅为向后兼容 (.blend 文件不报错)
    copy_textures: bpy.props.BoolProperty(
        name="提取贴图",
        description="(已废弃 UI 控件，提取始终复制贴图)",
        default=True,
        options={'HIDDEN'},
    ) # type: ignore

    auto_import: bpy.props.BoolProperty(
        name="提取后自动导入",
        description="(已废弃 UI 控件，提取始终自动导入)",
        default=True,
        options={'HIDDEN'},
    ) # type: ignore

    # 仅由操作符写入，用于在面板上展示最近一次提取结果 (可含多行，用\n分隔)
    last_report: bpy.props.StringProperty(
        name="最近提取结果",
        default="",
    ) # type: ignore


# ----------------------------------------------------------------------
# UIList 和行操作符
# ----------------------------------------------------------------------

class LOYAL_UL_ExtractIBList(bpy.types.UIList):
    '''DrawIB提取列表: 每行两列可编辑字段 (DrawIB | 别名)，双击字段即可编辑'''

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            split = layout.split(factor=0.55, align=True)
            split.prop(item, "draw_ib", text="", emboss=False, icon='MESH_DATA')
            split.prop(item, "alias", text="", emboss=False, icon='SORTALPHA')
        elif self.layout_type in {'GRID'}:
            layout.alignment = 'CENTER'
            layout.prop(item, "draw_ib", text="", emboss=False)


class LoyalExtractIBAdd(bpy.types.Operator):
    bl_idname = "loyal.extract_ib_add"
    bl_label = "添加DrawIB行"
    bl_description = "在DrawIB列表末尾添加一行"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        items = scene.loyal_extract_ib_items
        items.add()
        scene.loyal_extract_ib_index = len(items) - 1
        return {'FINISHED'}


class LoyalExtractIBRemove(bpy.types.Operator):
    bl_idname = "loyal.extract_ib_remove"
    bl_label = "删除DrawIB行"
    bl_description = "删除DrawIB列表中当前选中的行"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        items = getattr(context.scene, 'loyal_extract_ib_items', None)
        return items is not None and len(items) > 0

    def execute(self, context):
        scene = context.scene
        items = scene.loyal_extract_ib_items
        index = scene.loyal_extract_ib_index
        if not (0 <= index < len(items)):
            return {'CANCELLED'}
        items.remove(index)
        scene.loyal_extract_ib_index = max(0, min(index, len(items) - 1))
        return {'FINISHED'}


class LoyalExtractIBMove(bpy.types.Operator):
    bl_idname = "loyal.extract_ib_move"
    bl_label = "移动DrawIB行"
    bl_description = "上下移动DrawIB列表中当前选中的行"
    bl_options = {'REGISTER', 'UNDO'}

    direction: bpy.props.EnumProperty(
        name="方向",
        items=[
            ('UP', "上移", "把选中行向上移动一位"),
            ('DOWN', "下移", "把选中行向下移动一位"),
        ],
        default='UP',
    ) # type: ignore

    @classmethod
    def poll(cls, context):
        items = getattr(context.scene, 'loyal_extract_ib_items', None)
        return items is not None and len(items) > 1

    def execute(self, context):
        scene = context.scene
        items = scene.loyal_extract_ib_items
        index = scene.loyal_extract_ib_index
        if not (0 <= index < len(items)):
            return {'CANCELLED'}
        new_index = index - 1 if self.direction == 'UP' else index + 1
        if not (0 <= new_index < len(items)):
            return {'CANCELLED'}
        items.move(index, new_index)
        scene.loyal_extract_ib_index = new_index
        return {'FINISHED'}


# ----------------------------------------------------------------------
# 主操作符
# ----------------------------------------------------------------------

class LoyalExtractFromDump(bpy.types.Operator):
    bl_idname = "loyal.extract_from_dump"
    bl_label = "提取"
    bl_description = "从帧分析Dump中提取列表中所有DrawIB的模型和贴图，写入当前工作空间，并可自动导入到场景、接入蓝图节点"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        props = scene.loyal_extract_props
        merged_skeleton_mode = props.workflow_mode == 'MERGED_SKELETON'

        # 1.校验输入。普通模式仍读取 DrawIB 表；骨骼合并模式由整帧自动识别角色。
        try:
            dump_folder = resolve_dump_folder(props)
            if merged_skeleton_mode:
                ib_hashes, aliases = [], {}
                original_mesh_ib_hashes = parse_optional_draw_ib_input(
                    props.merged_original_mesh_ibs,
                    "游戏原网格 IB",
                )
            else:
                ib_hashes, aliases = gather_ib_rows_input(scene, props)
                original_mesh_ib_hashes = []
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        # 2.解析工作空间 (和其它操作符一样，先刷新一次全局配置)
        GlobalConfig.read_from_main_json_ssmt4()
        workspace_folder = GlobalConfig.path_workspace_folder()
        if workspace_folder == "":
            self.report({'ERROR'}, "未解析到工作空间，请在\"基础信息\"面板中把工作空间来源切换为自定义目录并填写路径。")
            return {'CANCELLED'}

        # 3.提取是终末地(EFMI)专用功能，其它预设下警告但仍允许执行
        if GlobalConfig.logic_name != "" and GlobalConfig.logic_name != LogicName.EFMI:
            if merged_skeleton_mode:
                self.report({'ERROR'}, "骨骼合并仅支持终末地(EFMI)，当前游戏预设为 " + GlobalConfig.logic_name + "。")
                return {'CANCELLED'}
            self.report({'WARNING'}, "当前游戏预设为 " + GlobalConfig.logic_name + "，DrawIB提取是终末地(EFMI)专用功能，结果可能不正确。")

        # 4.延迟导入提取模块并执行提取
        try:
            from ..extract.dump_workspace_extractor import DumpWorkspaceExtractor, ExtractError
        except Exception as e:
            self.report({'ERROR'}, "无法加载提取模块: " + repr(e))
            return {'CANCELLED'}

        try:
            extractor = get_cached_extractor(dump_folder)
            if merged_skeleton_mode:
                result = extractor.extract_merged_skeleton(
                    workspace_folder=workspace_folder,
                    copy_textures=True,
                    original_mesh_ib_hashes=original_mesh_ib_hashes,
                )
            else:
                result = extractor.extract(
                    ib_hashes=ib_hashes,
                    workspace_folder=workspace_folder,
                    copy_textures=True,
                    aliases=aliases or None,
                )
        except ExtractError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, "提取失败: " + repr(e))
            return {'CANCELLED'}

        for warning in result.warnings:
            print("[LoyalTools提取警告] " + warning)

        if not result.unique_strs:
            message = "没有提取到任何子网格。"
            if result.warnings:
                message += " 首条警告: " + result.warnings[0]
            props.last_report = message
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        # 5.导入到以工作空间命名的红色集合 (已存在则复用)，并接入蓝图节点
        imported_objects = []
        imported_entries = []  # [(unique_str, obj)] 供蓝图连线使用
        import_error_count = 0
        blueprint_tree_name = None
        collection_name = ""

        # 本次填写的别名已由extract()写入工作空间Config.json，这里读回合并后的
        # 完整别名表 (含之前提取留下的别名)，保证重命名/蓝图分组和"一键导入"行为一致
        drawib_aliasname_dict = {}
        try:
            drawib_aliasname_dict = WorkSpaceHelper.get_drawib_aliasname_dict()
        except Exception as e:
            print("[LoyalTools提取] 读取别名配置失败 (按无别名处理): " + repr(e))
        drawib_aliasname_dict.update(aliases)

        # 集合命名/颜色与"一键导入"一致：工作空间名 + 红色，重复提取时复用同一个集合
        try:
            collection = get_or_create_workspace_import_collection(context)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        collection_name = collection.name

        for json_path in result.json_paths:
            unique_str = os.path.splitext(os.path.basename(json_path))[0]
            try:
                # create_mesh_from_json 会以unique_str作为物体名称，
                # 该前缀是后续生成Mod时定位工作空间文件夹的关键，不能改名去掉
                imported_obj = SSMTImportHelper.create_mesh_from_json(
                    json_file_path=json_path,
                    import_collection=collection,
                    merged_skeleton=merged_skeleton_mode,
                )
                if imported_obj is not None:
                    # 物体名称保持纯 unique_str (drawib-indexcount-firstindex)，
                    # 不追加别名后缀；别名仅用于蓝图分组标签与工作空间Config.json
                    if imported_obj.name != unique_str:
                        imported_obj.name = unique_str
                        imported_obj.data.name = imported_obj.name
                    imported_objects.append(imported_obj)
                    imported_entries.append((unique_str, imported_obj))
            except Exception as e:
                import_error_count += 1
                print("[LoyalTools提取] 导入失败 " + json_path + ": " + repr(e))

        if imported_objects:
            if GlobalProterties.enable_non_mirror_workflow():
                NonMirrorWorkflowHelper.process_imported_objects(imported_objects)
            # 用户习惯导入后为全选状态，同时注册前缀快捷操作条目方便后续导出
            CollectionUtils.select_collection_objects(collection)
            PrefixQuickOpsHelper.merge_prefixes_from_objects(context, imported_objects)

            # 物体导入后自动接入蓝图节点 (失败只打印警告，不影响导入结果)
            blueprint_tree_name = wire_objects_into_blueprint(
                context, imported_entries, drawib_aliasname_dict,
                merged_skeleton=merged_skeleton_mode,
            )

        # 6.汇总结果
        # 贴图数量仅作展示，统计各子网格文件夹下的贴图文件
        texture_count = 0
        for unique_str in result.unique_strs:
            submesh_folder = os.path.join(result.workspace_folder, unique_str)
            if os.path.isdir(submesh_folder):
                for _, _, filenames in os.walk(submesh_folder):
                    for filename in filenames:
                        if filename.lower().endswith((".dds", ".jpg", ".png")):
                            texture_count += 1

        workflow_label = "骨骼合并" if merged_skeleton_mode else "普通"
        report_lines = [workflow_label + "提取完成: " + str(len(result.unique_strs)) + " 个子网格, 贴图 " + str(texture_count) + " 张"]
        report_lines.append("已导入 " + str(len(imported_objects)) + " 个物体到集合 \"" + collection_name + "\"")
        if import_error_count > 0:
            report_lines.append("有 " + str(import_error_count) + " 个子网格导入失败，详见控制台")
        if blueprint_tree_name:
            report_lines.append("蓝图节点已连接: \"" + blueprint_tree_name + "\"，可在基础信息面板直接生成所选蓝图Mod")
        elif imported_objects:
            report_lines.append("蓝图节点连接失败，详见控制台 (不影响手动连线)")
        if result.warnings:
            report_lines.append("警告 " + str(len(result.warnings)) + " 条 (详见控制台):")
            for warning in result.warnings[:5]:
                report_lines.append("  " + warning)
            if len(result.warnings) > 5:
                report_lines.append("  ... 其余 " + str(len(result.warnings) - 5) + " 条省略")

        props.last_report = "\n".join(report_lines)
        self.report({'INFO'}, report_lines[0])
        return {'FINISHED'}


# ----------------------------------------------------------------------
# 独立导入操作符（不重新提取，仅把工作空间现有文件导入到场景）
# ----------------------------------------------------------------------

class LoyalImportWorkspace(bpy.types.Operator):
    bl_idname = "loyal.import_workspace"
    bl_label = "导入"
    bl_description = "把当前工作空间中已提取的所有子网格导入到场景，可多次重复执行（不会重新提取）"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import json as _json

        props = context.scene.loyal_extract_props
        merged_skeleton_mode = props.workflow_mode == 'MERGED_SKELETON'
        GlobalConfig.read_from_main_json_ssmt4()
        workspace_folder = GlobalConfig.path_workspace_folder()
        if workspace_folder == "":
            self.report({'ERROR'}, "未解析到工作空间，请在\"基础信息\"面板中把工作空间来源切换为自定义目录并填写路径。")
            return {'CANCELLED'}

        import_json_path = os.path.join(workspace_folder, "Import.json")
        if not os.path.isfile(import_json_path):
            self.report({'ERROR'}, "工作空间中没有找到 Import.json，请先执行提取。")
            return {'CANCELLED'}

        try:
            with open(import_json_path, 'r', encoding='utf-8') as f:
                import_map = _json.load(f)  # {unique_str: gametype_name}
        except Exception as e:
            self.report({'ERROR'}, "读取 Import.json 失败: " + repr(e))
            return {'CANCELLED'}

        if not import_map:
            self.report({'WARNING'}, "Import.json 为空，工作空间中没有已提取的子网格，请先执行提取。")
            return {'CANCELLED'}

        # 普通模式按 Import.json；骨骼合并模式严格按 profile 的组件顺序导入，
        # 避免同一工作空间中旧的普通提取数据混入全局 VG 地址空间。
        ordered_unique_strs = list(import_map.keys())
        if merged_skeleton_mode:
            try:
                from ..common.efmi_merged_skeleton import load_profile
                merged_profile = load_profile(workspace_folder, required=True)
                ordered_unique_strs = [
                    component["unique_str"]
                    for component in merged_profile["components"]
                ]
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

        # 根据工作空间契约拼出各子网格的 SubmeshJson 路径
        json_paths = []
        for unique_str in ordered_unique_strs:
            gametype_name = import_map.get(unique_str, 'GPU-EFMI')
            json_path = os.path.join(
                workspace_folder, unique_str,
                "TYPE_" + gametype_name,
                unique_str + ".json",
            )
            if os.path.isfile(json_path):
                json_paths.append(json_path)
            else:
                print("[LoyalTools导入] 找不到子网格JSON: " + json_path)

        if not json_paths:
            self.report({'ERROR'}, "工作空间中没有找到有效的子网格 JSON 文件，请先执行提取。")
            return {'CANCELLED'}

        # 读取别名配置
        drawib_aliasname_dict = {}
        try:
            drawib_aliasname_dict = WorkSpaceHelper.get_drawib_aliasname_dict()
        except Exception as e:
            print("[LoyalTools导入] 读取别名配置失败 (按无别名处理): " + repr(e))

        # 每次导入都新建一个集合（Blender 自动追加 .001/.002 等后缀），
        # 避免重复导入的物体混入同一个集合
        try:
            collection = create_new_workspace_import_collection(context)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        collection_name = collection.name

        # 逐个导入
        imported_objects = []
        imported_entries = []
        import_error_count = 0
        for json_path in json_paths:
            unique_str = os.path.splitext(os.path.basename(json_path))[0]
            try:
                imported_obj = SSMTImportHelper.create_mesh_from_json(
                    json_file_path=json_path,
                    import_collection=collection,
                    merged_skeleton=merged_skeleton_mode,
                )
                if imported_obj is not None:
                    if imported_obj.name != unique_str:
                        imported_obj.name = unique_str
                        imported_obj.data.name = imported_obj.name
                    imported_objects.append(imported_obj)
                    imported_entries.append((unique_str, imported_obj))
            except Exception as e:
                import_error_count += 1
                print("[LoyalTools导入] 导入失败 " + json_path + ": " + repr(e))

        if imported_objects:
            if GlobalProterties.enable_non_mirror_workflow():
                NonMirrorWorkflowHelper.process_imported_objects(imported_objects)
            CollectionUtils.select_collection_objects(collection)
            PrefixQuickOpsHelper.merge_prefixes_from_objects(context, imported_objects)

        blueprint_tree_name = wire_objects_into_blueprint(
            context, imported_entries, drawib_aliasname_dict,
            tree_name_override=collection_name,
            merged_skeleton=merged_skeleton_mode,
        )

        msg = "已导入 " + str(len(imported_objects)) + " 个物体到集合 \"" + collection_name + "\""
        if import_error_count > 0:
            msg += "，" + str(import_error_count) + " 个失败 (详见控制台)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ----------------------------------------------------------------------
# 面板
# ----------------------------------------------------------------------

class LOYAL_PT_ExtractPanel(bpy.types.Panel):
    bl_label = "提取模型 (DrawIB)"
    bl_idname = "LOYAL_PT_ExtractPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LoyalTools'
    bl_order = 1

    @classmethod
    def poll(cls, context):
        if not hasattr(context.scene, 'herta_show_toolkit'):
            return True
        return not context.scene.herta_show_toolkit

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.loyal_extract_props

        layout.prop(props, "workflow_mode", expand=True)
        layout.prop(props, "frame_dump_folder")

        if props.workflow_mode == 'STANDARD':
            # DrawIB分行列表 (SSMT4风格表格: DrawIB | 别名)
            header_split = layout.split(factor=0.55, align=True)
            header_split.label(text="DrawIB", icon='MESH_DATA')
            header_split.label(text="别名 (可留空)")

            list_row = layout.row()
            list_row.template_list(
                "LOYAL_UL_ExtractIBList", "",
                scene, "loyal_extract_ib_items",
                scene, "loyal_extract_ib_index",
                rows=3,
            )
            button_column = list_row.column(align=True)
            button_column.operator(LoyalExtractIBAdd.bl_idname, text="", icon='ADD')
            button_column.operator(LoyalExtractIBRemove.bl_idname, text="", icon='REMOVE')
            button_column.separator()
            button_column.operator(LoyalExtractIBMove.bl_idname, text="", icon='TRIA_UP').direction = 'UP'
            button_column.operator(LoyalExtractIBMove.bl_idname, text="", icon='TRIA_DOWN').direction = 'DOWN'
        else:
            merged_box = layout.box()
            merged_box.label(text="自动识别当前帧的主要角色与全部组件", icon='ARMATURE_DATA')
            merged_box.label(text="输出仍为 IBHash-IndexCount-FirstIndex；暂不处理 LOD")
            merged_box.label(text="导入时自动把各组件本地顶点组映射为全局顶点组")
            merged_box.prop(props, "merged_original_mesh_ibs")
            merged_box.label(text="特殊眉毛/睫毛可填 IB；保留游戏原网格与自动贴图", icon='INFO')

        extract_row = layout.row()
        extract_row.scale_y = 1.5
        extract_row.operator(LoyalExtractFromDump.bl_idname, text="提取", icon='IMPORT')

        import_row = layout.row()
        import_row.scale_y = 1.2
        import_row.operator(LoyalImportWorkspace.bl_idname, text="导入", icon='MESH_DATA')

        if props.last_report:
            report_box = layout.box()
            for line in props.last_report.split("\n"):
                report_box.label(text=line)


classes = (
    LoyalExtractIBItem,
    LoyalExtractProperties,
    LOYAL_UL_ExtractIBList,
    LoyalExtractIBAdd,
    LoyalExtractIBRemove,
    LoyalExtractIBMove,
    LoyalExtractFromDump,
    LoyalImportWorkspace,
    LOYAL_PT_ExtractPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.loyal_extract_props = bpy.props.PointerProperty(type=LoyalExtractProperties)
    bpy.types.Scene.loyal_extract_ib_items = bpy.props.CollectionProperty(type=LoyalExtractIBItem)
    bpy.types.Scene.loyal_extract_ib_index = bpy.props.IntProperty(
        name="DrawIB列表选中行",
        default=0,
        min=0,
    )


def unregister():
    for prop_name in ('loyal_extract_ib_index', 'loyal_extract_ib_items', 'loyal_extract_props'):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
