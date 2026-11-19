"""Headless tests of production switch gates, without importing Blender's UI.

These tests execute extracted production statements; fake scene objects fail on
unexpected reads. Full mesh/buffer export and .blend reload remain Blender tests.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1]


def method(path, owner, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def function(body, namespace, *, name="execute", args=("self",)):
    definition = ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=arg) for arg in args],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=copy.deepcopy(body), decorator_list=[],
    )
    tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(tree, "production-switch-gate", "exec"), namespace)
    return namespace[name]


class Untouchable:
    def __getattr__(self, name):
        raise AssertionError("Disabled module read " + name)

    def __len__(self):
        raise AssertionError("Disabled module read the registered objects")


class TaggedOnly:
    """Only ID-property reads are legal; modifier access is a failure."""
    def __init__(self, merged):
        self.merged = merged

    def get(self, key, default=None):
        return self.merged if key == "LoyalTools:EFMIMergedSkeleton" else default

    @property
    def modifiers(self):
        raise AssertionError("Unrelated mesh modifiers were visited")


def strip_method(settings, *, logic="EFMI"):
    code = method("blueprint/preprocess.py", "PreProcessHelper", "_strip_efmi_physics_preview_modifiers")
    return function(code.body, {
        "bpy": NS(context=NS(scene=NS(loyaltools_efmi_physics=settings))),
        "GlobalConfig": NS(logic_name=logic), "LogicName": NS(EFMI="EFMI"),
        "LOG": NS(info=lambda _message: None),
    }, args=("obj",))


class PhysicsSwitchTests(unittest.TestCase):
    def test_scene_switch_defaults_false_and_is_blend_persistent(self):
        tree = ast.parse((ROOT / "efmi_physics/properties.py").read_text(encoding="utf-8"))
        settings = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EFMIPhysicsSettings")
        enabled = next(node.annotation for node in settings.body if isinstance(node, ast.AnnAssign) and node.target.id == "enabled")
        keywords = {item.arg: item.value for item in enabled.keywords}
        self.assertIs(ast.literal_eval(keywords["default"]), False)
        if "options" in keywords:
            self.assertNotIn("SKIP_SAVE", ast.literal_eval(keywords["options"]))

    def test_off_preprocess_never_reads_mesh_or_ownership(self):
        settings = NS(enabled=False, objects=Untouchable())
        self.assertEqual(strip_method(settings)(Untouchable()), 0)
        self.assertEqual(strip_method(None)(Untouchable()), 0)

    def test_nonmerged_and_other_games_never_visit_modifiers(self):
        settings = NS(enabled=True, objects=Untouchable())
        self.assertEqual(strip_method(settings)(TaggedOnly(False)), 0)
        self.assertEqual(strip_method(settings, logic="ZZMI")(TaggedOnly(True)), 0)

    def test_off_export_initialization_skips_optional_imports_and_records(self):
        code = method("ui/universal/efmi.py", "ExportEFMI", "__post_init__")
        def assignment(node, attribute):
            return isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == attribute for target in node.targets)
        start = next(index for index, node in enumerate(code.body) if assignment(node, "efmi_physics_plan"))
        end = next(index for index, node in enumerate(code.body[start:], start) if assignment(node, "submesh_model_list"))
        # Disabled export may read normal export targets for its safety check,
        # and registry bindings read-only, but must not collect/migrate records
        # or construct the optional runtime.
        preflight_calls = []
        namespace = {"bpy": NS(context=NS(scene=NS(loyaltools_efmi_physics=NS(enabled=False, objects=[])))),
                     "merged_gpu_unique_strs": None,
                     "reject_disabled_physics_weights": lambda targets, **kwargs: preflight_calls.append(targets)}
        execute = function(code.body[start:end], namespace)
        target = NS(blueprint_model=NS(ordered_draw_obj_data_model_list=[]), merged_skeleton_mode=True)
        execute(target)
        self.assertIsNone(target.efmi_physics_plan)
        self.assertIsNone(target.efmi_physics_runtime)
        self.assertEqual(preflight_calls, [[]])

    def test_off_callback_keeps_native_draw_command(self):
        code = method("ui/universal/efmi.py", "ExportEFMI", "_merged_draw_callback_name")
        execute = function(code.body, {}, args=("self", "component_id"))
        for target in (NS(efmi_physics_runtime=None), NS()):
            self.assertEqual(execute(target, 3), "CommandList_Draw_Component3")

    def test_full_ini_export_off_does_not_publish_or_reference_physics_files(self):
        code = method("ui/universal/efmi.py", "ExportEFMI", "_generate_merged_skeleton_ini_file")
        gate_start = next(index for index, node in enumerate(code.body)
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id == "runtime" for target in node.targets))
        with tempfile.TemporaryDirectory(prefix="efmi_switch_") as directory:
            output = Path(directory)
            active = output / "character.ini"
            # A prior enabled export's inert files may remain. INI is the only
            # executable entry point; turning off must replace its references.
            shader = output / "res/EFMIPhysics/Shaders/prepare.hlsl"
            shader.parent.mkdir(parents=True)
            shader.write_text("prior reviewed shader", encoding="utf-8")
            active.write_text("[CustomShaderEFMIPhysics_Prepare]\ncs = res/EFMIPhysics/Shaders/prepare.hlsl\n", encoding="utf-8")
            calls = []
            def save(path):
                calls.append(path)
                Path(path).write_text("[CommandList_Draw_Component3]\ndrawindexed = auto\n", encoding="utf-8")
            def unexpected_write(*_args, **_kwargs):
                raise AssertionError("Off export wrote physics runtime files")
            namespace = {"ini_builder": NS(save_to_file=save), "ini_filepath": str(active),
                         "write_runtime_files": unexpected_write, "split_runtime_sections": unexpected_write,
                         "atomic_ini_target": unexpected_write}
            execute = function(code.body[gate_start:], namespace)
            for target in (NS(efmi_physics_runtime=None), NS()):
                execute(target)
            self.assertEqual(calls, [str(active), str(active)])
            self.assertNotIn("EFMIPhysics", active.read_text(encoding="utf-8"))
            self.assertEqual(shader.read_text(encoding="utf-8"), "prior reviewed shader")


if __name__ == "__main__":
    unittest.main()
