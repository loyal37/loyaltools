from __future__ import annotations

import os

import bpy

from ..common.global_properties import GlobalProterties
from ..utils.log_utils import LOG
from ..utils.timer_utils import TimerUtils
from .export_helper import BlueprintExportHelper
from .node_base import SSMTNodeBase


class SSMT_OT_GenerateNTMIModImpBlueprint(bpy.types.Operator):
    bl_idname = "ssmt.generate_ntmi_modimp_blueprint"
    bl_label = "Generate NTMI ModImp"
    bl_description = "Generate mod_importer compatible NTMI buffers and INI from this blueprint"
    bl_options = {'REGISTER', 'UNDO'}

    node_name: bpy.props.StringProperty(default="", options={'SKIP_SAVE'})  # type: ignore
    blueprint_name: bpy.props.StringProperty(default="", options={'SKIP_SAVE'})  # type: ignore

    def _resolve_tree_and_node(self, context):
        requested_tree_name = str(getattr(self, "blueprint_name", "") or "").strip()
        if requested_tree_name:
            tree = BlueprintExportHelper.get_selected_blueprint_tree(
                selected_name=requested_tree_name,
                context=context,
            )
        else:
            tree = BlueprintExportHelper.get_current_blueprint_tree(context=context)

        if not tree:
            return None, None

        node_name = str(getattr(self, "node_name", "") or "").strip()
        node = tree.nodes.get(node_name) if node_name else None
        if node is None:
            for candidate in tree.nodes:
                if candidate.bl_idname == "SSMTNode_Result_Output_NTMIModImp":
                    node = candidate
                    break
        return tree, node

    def execute(self, context):
        from .ntmi_export_modimp import (
            execute_ntmi_modimp_export,
            resolve_ntmi_modimp_output_dir,
        )

        LOG.start_collecting()
        TimerUtils.start_session("NTMI ModImp Export")

        try:
            tree, node = self._resolve_tree_and_node(context)
            if tree is None or node is None:
                self.report({'ERROR'}, "No NTMI ModImp output node found in the current blueprint.")
                return {'CANCELLED'}

            BlueprintExportHelper.set_runtime_blueprint_tree(tree)
            execute_ntmi_modimp_export(context=context, tree=tree, node=node)

            self.report({'INFO'}, "Generate NTMI ModImp Success!")
            if GlobalProterties.open_mod_folder_after_generate_mod():
                output_dir = resolve_ntmi_modimp_output_dir(node)
                os.startfile(output_dir)
            return {'FINISHED'}
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            TimerUtils.print_summary()
            LOG.stop_collecting()
            LOG.save_to_text_editor("NTMI_ModImp_Export_Log")


class SSMT_OT_CheckNTMIModImpDependency(bpy.types.Operator):
    bl_idname = "ssmt.check_ntmi_modimp_dependency"
    bl_label = "Check NTMI ModImp Dependency"
    bl_description = "Check whether the mod_importer-main dependency is installed or resolvable"
    bl_options = {'REGISTER'}

    node_name: bpy.props.StringProperty(default="", options={'SKIP_SAVE'})  # type: ignore
    blueprint_name: bpy.props.StringProperty(default="", options={'SKIP_SAVE'})  # type: ignore

    def execute(self, context):
        from ..ui.ntmi_modimp.modimp_core import detect_mod_importer_dependency

        tree = BlueprintExportHelper.get_selected_blueprint_tree(
            selected_name=str(getattr(self, "blueprint_name", "") or "").strip(),
            context=context,
        )
        node = tree.nodes.get(self.node_name) if tree and self.node_name else None
        configured_root = str(getattr(node, "mod_importer_root", "") or "").strip() if node else ""
        status = detect_mod_importer_dependency(configured_root)
        if status.available:
            self.report({'INFO'}, f"{status.message}: {status.root}")
            return {'FINISHED'}

        self.report({'ERROR'}, status.message)
        return {'CANCELLED'}


class SSMTNode_Result_Output_NTMIModImp(SSMTNodeBase):
    bl_idname = "SSMTNode_Result_Output_NTMIModImp"
    bl_label = "NTMI ModImp Output"
    bl_icon = 'EXPORT'
    bl_width_min = 420

    use_custom_export_dir: bpy.props.BoolProperty(
        name="Use Custom Export Directory",
        description="Enable this to export to the directory selected below instead of the default output directory",
        default=True,
    )  # type: ignore

    export_dir: bpy.props.StringProperty(
        name="Export Directory",
        description="Directory used when manual export directory is enabled",
        default="",
        subtype='DIR_PATH',
    )  # type: ignore

    mod_importer_root: bpy.props.StringProperty(
        name="mod_importer-main",
        description="Optional override path to the mod_importer-main plugin root. Leave empty to auto-detect installed add-on.",
        default="",
        subtype='DIR_PATH',
    )  # type: ignore

    generate_ini: bpy.props.BoolProperty(
        name="Generate INI",
        default=True,
    )  # type: ignore

    force_buffer_only_when_contract_missing: bpy.props.BoolProperty(
        name="Missing Contract -> Buffer Only",
        default=True,
    )  # type: ignore

    keep_temp_collection_tree: bpy.props.BoolProperty(
        name="Keep Temp Collections",
        default=False,
    )  # type: ignore

    run_postprocess_nodes: bpy.props.BoolProperty(
        name="Run Compatible Postprocess",
        default=True,
    )  # type: ignore

    flip_uv_v: bpy.props.BoolProperty(
        name="Flip UV V",
        default=True,
    )  # type: ignore

    default_mirror_flip: bpy.props.BoolProperty(
        name="Default Mirror Flip",
        default=False,
    )  # type: ignore

    export_runtime_shapekeys: bpy.props.BoolProperty(
        name="Runtime ShapeKeys",
        default=False,
    )  # type: ignore

    runtime_shapekey_names: bpy.props.StringProperty(
        name="Runtime ShapeKey Names",
        description="Comma separated runtime shapekey names",
        default="",
    )  # type: ignore

    show_advanced: bpy.props.BoolProperty(
        name="Advanced",
        default=False,
    )  # type: ignore

    def init(self, context):
        self.inputs.new('SSMTSocketObject', "Group 1")
        self.outputs.new('SSMTSocketPostProcess', "Post Process")
        self.width = 420

    def draw_buttons(self, context, layout):
        operator = layout.operator("ssmt.generate_ntmi_modimp_blueprint", text="Generate NTMI ModImp", icon='EXPORT')
        operator.node_name = self.name
        operator.blueprint_name = self.id_data.name if self.id_data else ""

        output_box = layout.box()
        output_box.label(text="导出目录")
        output_box.prop(self, "use_custom_export_dir", text="手动指定导出目录")
        if self.use_custom_export_dir:
            output_box.prop(self, "export_dir", text="目录")
        else:
            output_box.label(text="默认：.blend 同目录/用户目录", icon='INFO')

        layout.prop(self, "mod_importer_root", text="前置插件路径")
        try:
            from ..ui.ntmi_modimp.modimp_core import detect_mod_importer_dependency

            dependency_status = detect_mod_importer_dependency(self.mod_importer_root)
            dep_box = layout.box()
            if dependency_status.available:
                dep_box.label(text=dependency_status.message, icon='CHECKMARK')
                dep_box.label(text=dependency_status.root)
            else:
                dep_box.label(text=dependency_status.message, icon='ERROR')
                dep_box.label(text="可安装 Mod Importer 前置插件，或手动填写其插件根目录。")
            check_operator = dep_box.operator("ssmt.check_ntmi_modimp_dependency", text="检测前置插件", icon='VIEWZOOM')
            check_operator.node_name = self.name
            check_operator.blueprint_name = self.id_data.name if self.id_data else ""
        except Exception as exc:
            layout.label(text=f"前置插件检测失败: {exc}", icon='ERROR')

        layout.prop(self, "generate_ini", text="生成 INI")
        layout.prop(self, "force_buffer_only_when_contract_missing", text="缺少合同字段时只导出 Buffer")
        layout.prop(self, "flip_uv_v", text="翻转 UV V")
        layout.prop(self, "run_postprocess_nodes", text="运行兼容后处理节点")

    def update(self):
        if self.inputs and self.inputs[-1].is_linked:
            self.inputs.new('SSMTSocketObject', f"Group {len(self.inputs) + 1}")

        if len(self.inputs) > 1 and not self.inputs[-1].is_linked and not self.inputs[-2].is_linked:
            self.inputs.remove(self.inputs[-1])


classes = (
    SSMT_OT_GenerateNTMIModImpBlueprint,
    SSMT_OT_CheckNTMIModImpDependency,
    SSMTNode_Result_Output_NTMIModImp,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
