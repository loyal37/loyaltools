"""Check generated draw selectors without changing any user project or mod."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from LoyalTools.common.efmi_merged_skeleton import validate_profile
from LoyalTools.common.global_config import GlobalConfig
from LoyalTools.common.m_ini_helper import M_IniHelper
from LoyalTools.common.m_ini_helper_gui import M_IniHelperGUI
from LoyalTools.ui.universal.efmi import ExportEFMI
from LoyalTools.ui.universal import efmi as efmi_module


def component(number, main_hash, lod_hash, main_base, lod_base):
    main = {
        "component_id": number, "ib_hash": main_hash,
        "unique_str": main_hash + "-540-3", "index_count": 540,
        "first_index": 3, "vertex_count": 360, "vg_offset": number,
        "vg_count": 1, "vg_map": {"0": number}, "cpu_posed": False,
        "lods": [{
            "lod_object_name": "LOD1", "ib_hash": lod_hash,
            "index_count": 540, "first_index": 3, "vertex_count": 360,
        }],
    }
    if main_base is not None:
        main["first_vertex"] = main_base
    if lod_base is not None:
        main["lods"][0]["first_vertex"] = lod_base
    return main


def sections(text):
    result, current = {}, None
    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            result[current] = []
        elif current is not None:
            result[current].append(line)
    return result


def main():
    profile = validate_profile({
        "mode": "EFMI_MERGED_SKELETON", "format_version": 1,
        "components": [
            component(0, "aaaaaaaa", "aaaaaaaa", 7, 9),
            component(1, "bbbbbbbb", "cccccccc", -4, -8),
            component(2, "dddddddd", "eeeeeeee", None, None),
            component(3, "ffffffff", "ffffffff", 0, 0),
            component(4, "12345678", "12345678", 0, 0),
        ],
    })
    # A same-hash LoD with a different index window also needs its own entry.
    profile["components"][3]["lods"][0]["first_index"] = 21
    exporter = ExportEFMI.__new__(ExportEFMI)
    exporter.merged_skeleton_profile = profile
    exporter.has_cross_ib = False
    exporter.cross_ib_info_dict = {}
    exporter.submesh_model_list = []  # Bone-only GPU entries must stay present.
    exporter.drawib_model_list = []
    exporter.merged_connected_cpu_unique_strs = set()
    exporter.merged_lod_variant_buffers = {}
    exporter.merged_lod_blend_remaps = {}
    exporter.merged_auto_texture_binding_dict = {}
    exporter.blueprint_model = SimpleNamespace(keyname_mkey_dict={})
    exporter._integrate_object_swap_ini_hook = lambda builder: None
    exporter._copy_merged_auto_texture_files = lambda: None
    exporter._append_merged_incoming_cross_ib_draws = lambda *args: None

    output = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="loyal-entrypoints-") as folder:
        with contextlib.ExitStack() as stack:
            for target, name, value in (
                (GlobalConfig, "get_workspace_name", "SelectorRegression"),
                (GlobalConfig, "path_generate_mod_folder", folder),
                (efmi_module.GlobalProterties, "forbid_auto_texture_ini", True),
                (M_IniHelper, "generate_hash_style_texture_ini", None),
                (M_IniHelper, "add_branch_key_sections", None),
                (M_IniHelperGUI, "add_branch_mod_gui_section", None),
            ):
                stack.enter_context(patch.object(target, name, return_value=value))
            with contextlib.redirect_stdout(output):
                exporter._generate_merged_skeleton_ini_file()
            with open(Path(folder) / "SelectorRegression.ini") as file:
                ini = file.read()

    parsed = sections(ini)
    entries = {name: lines for name, lines in parsed.items()
               if name.startswith("TextureOverride_EntryPoint_")}
    assert len(entries) == 9, entries.keys()
    assert "TextureOverride_EntryPoint_Component4" in entries
    assert "TextureOverride_EntryPoint_Component4_LOD1" not in entries
    for component_id, expected in enumerate(((7, 9), (-4, -8), (0, 0), (0, 0))):
        for level, base in enumerate(expected):
            name = "TextureOverride_EntryPoint_Component" + str(component_id)
            if level:
                name += "_LOD1"
            lines = entries[name]
            assert "match_first_vertex = " + str(base) in lines, (name, lines)
            first = 21 if component_id == 3 and level else 3
            assert "match_first_index = " + str(first) in lines
            assert "match_index_count = 540" in lines
            assert lines.count("match_first_vertex = " + str(base)) == 1
    assert output.getvalue().count("缺少原始绘制 first_vertex") == 2
    assert "重新提取主帧并更新 LOD" in output.getvalue()
    assert ini.count("Legacy profile: first_vertex is assumed 0") == 2
    # No change to shared skip/identity/skeleton coordination.
    draw = parsed["CommandList_Component_DrawInstances"]
    assert draw[0] == "handling = skip"
    assert "run = CommandList\\EFMIv1\\SpatialIdentity_IdentifyComponentInstances" in draw
    assert "run = CommandList\\EFMIv1\\Component_DrawInstances" in draw
    same = profile["components"][0]
    assert ExportEFMI._merged_entrypoint_signature(same) != ExportEFMI._merged_entrypoint_signature(same["lods"][0])
    assert ExportEFMI._merged_entrypoint_signature(same) == ExportEFMI._merged_entrypoint_signature(dict(same))
    print("MERGED_ENTRYPOINT_REGRESSION=PASS")


if __name__ == "__main__":
    main()
