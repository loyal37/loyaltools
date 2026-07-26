"""
Object swap node support.
"""

from dataclasses import dataclass

import bpy

from .node_base import SSMTNodeBase
from .variable_registry import ensure_object_swap_variable_name, mark_variable_name_used, normalize_variable_name


@dataclass
class SwapKeyConfig:
    node_id: str = ""
    index: int = 0
    hotkey: str = "No_Modifiers Numpad3"
    swap_type: str = "cycle"
    option_count: int = 2
    comment: str = ""
    custom_var_name: str = ""
    assigned_variable_name: str = ""

    def get_swap_key_name(self) -> str:
        if self.custom_var_name:
            return f"${self.custom_var_name}"
        if self.assigned_variable_name:
            return f"${self.assigned_variable_name}"
        return f"$swapkey{self.index}"

    def get_key_swap_section_name(self) -> str:
        return f"KeySwap_{self.index}"

    def get_active_param_name(self) -> str:
        return f"$active{self.index}"

    def get_condition_value(self, option_index: int) -> str:
        return str(option_index)

    def get_condition_str(self, value: int = 1) -> str:
        return f"{self.get_swap_key_name()} == {value}"


class SSMTNode_ObjectSwap(SSMTNodeBase):
    bl_idname = "SSMTNode_ObjectSwap"
    bl_label = "物体切换"
    bl_icon = "SHADERFX"

    def update_all_properties(self, context):
        normalized = normalize_variable_name(self.custom_var_name)
        if normalized != str(self.custom_var_name or "").strip().lstrip("$"):
            self.custom_var_name = normalized
            return
        if normalized:
            mark_variable_name_used(normalized, context=context)
        ensure_object_swap_variable_name(self, context=context)
        self.update_node_width([self.comment, self.hotkey, self.custom_var_name, self.assigned_variable_name])

    comment: bpy.props.StringProperty(
        name="备注",
        description="生成到 KeySwap 段落中的注释。",
        default="",
        update=update_all_properties,
    )

    custom_var_name: bpy.props.StringProperty(
        name="变量名",
        description="节点创建时会自动填入预分配变量名，你可以直接复制或手动修改它。",
        default="",
        update=update_all_properties,
    )

    assigned_variable_name: bpy.props.StringProperty(
        name="Assigned Variable Name",
        description="Preallocated global variable name for this node.",
        default="",
        options={"HIDDEN"},
    )

    hotkey: bpy.props.StringProperty(
        name="快捷键",
        description="按键格式: Modifier KeyName，例如 No_Modifiers Numpad3。",
        default="No_Modifiers Numpad3",
        update=update_all_properties,
    )

    swap_type: bpy.props.EnumProperty(
        name="切换类型",
        description="切换模式。",
        items=[
            ("cycle", "循环切换", "循环切换所有选项"),
            ("toggle", "开关切换", "在两个选项间切换"),
            ("hold", "按住生效", "按住时激活"),
        ],
        default="cycle",
        update=update_all_properties,
    )

    def update_input_slot_count(self, context):
        self._update_input_sockets()

    input_slot_count: bpy.props.IntProperty(
        name="输入口数量",
        description="每个输入口对应一个切换选项。",
        min=1,
        max=1024,
        default=2,
        update=update_input_slot_count,
    )

    condition_operator: bpy.props.EnumProperty(
        name="条件运算符",
        description="多个切换条件之间的逻辑运算符。",
        items=[
            ("&&", "AND (&&)", "所有条件都满足时执行"),
            ("||", "OR (||)", "任一条件满足时执行"),
        ],
        default="&&",
        update=update_all_properties,
    )

    description_expanded: bpy.props.BoolProperty(
        name="展开说明",
        description="展开节点说明菜单。",
        default=False,
    )

    def init(self, context):
        ensure_object_swap_variable_name(self, context=context)
        self.outputs.new("SSMTSocketObject", "Output")
        self._update_input_sockets()
        self.width = 320

    def copy(self, node):
        self.assigned_variable_name = ""
        ensure_object_swap_variable_name(self)
        self.custom_var_name = self.assigned_variable_name

    def _update_input_sockets(self):
        current_count = len(self.inputs)
        target_count = self.input_slot_count

        while current_count > target_count:
            self.inputs.remove(self.inputs[-1])
            current_count -= 1

        while current_count < target_count:
            self.inputs.new("SSMTSocketObject", f"选项_{current_count}")
            current_count += 1

        for idx, inp in enumerate(self.inputs):
            inp.name = f"选项_{idx}"

    def update(self):
        ensure_object_swap_variable_name(self)
        self._update_input_sockets()

    def draw_buttons(self, context, layout):
        layout.prop(self, "comment", text="备注")
        layout.prop(self, "custom_var_name", text="变量名")
        if not str(self.custom_var_name or "").strip() and str(self.assigned_variable_name or "").strip():
            layout.label(text=f"预分配变量: ${self.assigned_variable_name}", icon="INFO")
        layout.prop(self, "hotkey", text="按键")

        layout.separator()
        layout.label(text="选项数量:")
        row = layout.row(align=True)
        row.prop(self, "input_slot_count", text="")
        if self.input_slot_count >= 2:
            row.operator("ssmt.add_swap_option", text="", icon="ADD").node_name = self.name
        if self.input_slot_count > 1:
            row.operator("ssmt.remove_swap_option", text="", icon="REMOVE").node_name = self.name

        layout.separator()
        row = layout.row()
        icon = "TRIA_DOWN" if self.description_expanded else "TRIA_RIGHT"
        row.prop(self, "description_expanded", text="节点说明", icon=icon, emboss=True)

        if self.description_expanded:
            col = layout.column()
            col.scale_y = 0.8
            col.prop(self, "swap_type", text="类型")
            col.prop(self, "condition_operator", text="运算符")
            col.separator()
            self._draw_node_description(col)

    def _draw_node_description(self, layout):
        var_display = f"${self.custom_var_name}" if self.custom_var_name else f"${self.assigned_variable_name}"
        option_seq = ", ".join(str(i) for i in range(self.input_slot_count))

        layout.label(text="本节点会追加以下配置段落:", icon="INFO")
        layout.label(text="  [KeySwap_*]", icon="NONE")
        layout.label(text="  [Constants]", icon="NONE")
        layout.label(text="  [Present]", icon="NONE")
        layout.label(text="  [TextureOverride_*]", icon="NONE")

        layout.separator()
        layout.label(text="当前节点参数:", icon="FILE_TEXT")
        layout.label(text=f"  备注: {self.comment if self.comment else '(未设置)'}", icon="NONE")
        layout.label(text=f"  变量: {var_display}", icon="NONE")
        layout.label(text=f"  按键: {self.hotkey}", icon="NONE")
        layout.label(text=f"  类型: {self.swap_type}", icon="NONE")
        layout.label(text=f"  运算符: {self.condition_operator}", icon="NONE")

        layout.separator()
        layout.label(text="选项配置:", icon="TRACKING")
        layout.label(text=f"  选项值: {option_seq}", icon="NONE")
        layout.label(text=f"  条件格式: {var_display} == 选项值", icon="NONE")

        layout.separator()
        layout.label(text="生成示例:", icon="INFO")
        layout.label(text="  [KeySwap_*]", icon="NONE")
        if self.comment:
            layout.label(text=f"  ; {self.comment}", icon="NONE")
        layout.label(text="  condition = $active0 == 1", icon="NONE")
        layout.label(text=f"  key = {self.hotkey}", icon="NONE")
        layout.label(text=f"  type = {self.swap_type}", icon="NONE")
        layout.label(text=f"  {var_display} = {option_seq},", icon="NONE")


class SSMT_OT_AddSwapOption(bpy.types.Operator):
    bl_idname = "ssmt.add_swap_option"
    bl_label = "添加选项"
    bl_options = {"REGISTER", "UNDO"}

    node_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return context.active_node is not None

    def execute(self, context):
        tree = context.space_data.edit_tree
        node = tree.nodes.get(self.node_name)
        if node and node.bl_idname == "SSMTNode_ObjectSwap" and node.input_slot_count < 1024:
            node.input_slot_count += 1
        return {"FINISHED"}


class SSMT_OT_RemoveSwapOption(bpy.types.Operator):
    bl_idname = "ssmt.remove_swap_option"
    bl_label = "移除选项"
    bl_options = {"REGISTER", "UNDO"}

    node_name: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return context.active_node is not None

    def execute(self, context):
        tree = context.space_data.edit_tree
        node = tree.nodes.get(self.node_name)
        if node and node.bl_idname == "SSMTNode_ObjectSwap" and node.input_slot_count > 1:
            node.input_slot_count -= 1
        return {"FINISHED"}


class ObjectSwapDebugger:
    @staticmethod
    def _get_node_unique_key(node) -> str:
        tree_name = node.id_data.name if hasattr(node, "id_data") and node.id_data else ""
        return f"{tree_name}::{node.name}"

    @staticmethod
    def generate_chain_detail(chain, registry=None) -> list[str]:
        if not chain.swap_node_option_values:
            return []

        lines = []
        for swap_key, option_val in chain.swap_node_option_values.items():
            swap_node = None
            for node in chain.node_path:
                if ObjectSwapDebugger._get_node_unique_key(node) == swap_key:
                    swap_node = node
                    break

            if swap_node is not None:
                swap_key_index = 0
                if registry and hasattr(registry, "node_swapkey_map"):
                    swap_key_index = registry.node_swapkey_map.get(swap_key, 0)
                node_index = 0
                for idx, path_node in enumerate(chain.node_path):
                    if ObjectSwapDebugger._get_node_unique_key(path_node) == swap_key:
                        node_index = idx
                        break
                lines.extend(ObjectSwapDebugger.generate_debug_detail(swap_node, node_index, swap_key_index))
            else:
                lines.append(f"物体切换: {swap_key} -> 选项 {option_val + 1} (索引 {option_val})")

        return lines

    @staticmethod
    def generate_debug_detail(swap_node: bpy.types.Node, node_index: int, swap_key_index: int) -> list[str]:
        resolved_name = normalize_variable_name(getattr(swap_node, "custom_var_name", "") or "")
        if not resolved_name:
            resolved_name = normalize_variable_name(getattr(swap_node, "assigned_variable_name", "") or "")
        var_name = f"${resolved_name}" if resolved_name else f"$swapkey{swap_key_index}"
        config = SwapKeyConfig(
            index=swap_key_index,
            comment=getattr(swap_node, "comment", ""),
            custom_var_name=normalize_variable_name(getattr(swap_node, "custom_var_name", "") or ""),
            assigned_variable_name=normalize_variable_name(getattr(swap_node, "assigned_variable_name", "") or ""),
        )

        lines = [
            "",
            f"物体切换节点 #{node_index + 1}",
            f"   备注: {getattr(swap_node, 'comment', '未设置')}",
            f"   变量: {var_name}",
            f"   快捷键: {getattr(swap_node, 'hotkey', 'N/A')}",
            f"   切换类型: {getattr(swap_node, 'swap_type', 'N/A')}",
            f"   选项数量: {getattr(swap_node, 'input_slot_count', 1)}",
            "",
            "   配置段落:",
            f"   [{config.get_key_swap_section_name()}]",
        ]

        if config.comment:
            lines.append(f"   ; {config.comment}")
        lines.extend(
            [
                "   condition = $active0 == 1",
                f"   key = {getattr(swap_node, 'hotkey', 'N/A')}",
                f"   type = {getattr(swap_node, 'swap_type', 'N/A')}",
                f"   {var_name} = {','.join(str(i) for i in range(getattr(swap_node, 'input_slot_count', 1)))},",
                "",
                "   常量声明:",
                f"   {var_name} = 0",
                "",
                "   初始化参数:",
                "   post $active0 = 0",
            ]
        )
        return lines


classes = (
    SSMTNode_ObjectSwap,
    SSMT_OT_AddSwapOption,
    SSMT_OT_RemoveSwapOption,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
