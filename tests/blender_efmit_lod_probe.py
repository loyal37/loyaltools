# -*- coding: utf-8 -*-
"""Run the upstream EFMI-Tools v0.6.2 LoD matcher on a copied object folder.

This probe is intentionally independent from LoyalTools.  It is useful when
checking that our profile data agrees with the upstream reference behavior.
"""

import importlib
import json
import sys
from pathlib import Path

import bpy


def _script_args():
    if "--" not in sys.argv:
        raise RuntimeError("Expected: -- <object folder> <LoD FrameAnalysis folder>")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 2:
        raise RuntimeError("Expected: <object folder> <LoD FrameAnalysis folder>")
    return Path(args[0]), Path(args[1])


def main():
    object_folder, lod_dump = _script_args()
    bpy.ops.preferences.addon_enable(module="EFMI-Tools")

    # Some valid FrameAnalysis captures contain unrelated draw calls whose IB
    # pointer was logged but whose resource was not dumped.  Upstream 0.6.2
    # aborts the entire LoD scan on those calls, so the probe skips only that
    # incomplete call and leaves the matcher itself unchanged.
    raw_module = importlib.import_module(
        "EFMI-Tools.migoto_io.object_extractor.raw_object.raw_object_extractor"
    )
    original_register = raw_module.RawObjectExtractor.register_shader_call

    def register_if_complete(self, extracted_object, shader_call, gpu_posed):
        try:
            return original_register(self, extracted_object, shader_call, gpu_posed)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            print(
                "EFMIT_LOD_PROBE_SKIPPED_CALL="
                + str(shader_call.id)
                + ":"
                + repr(exc)
            )
            return None

    raw_module.RawObjectExtractor.register_shader_call = register_if_complete

    cfg = bpy.context.scene.efmi_tools_settings
    cfg.object_source_folder = str(object_folder)
    cfg.lod_frame_dump_folder = str(lod_dump)
    cfg.allow_lod_overwrite = True
    cfg.import_matched_lod_objects = False
    cfg.verbose_logging = False

    module = importlib.import_module("EFMI-Tools.extract_frame_data.extract_frame_data")
    module.extract_frame_data(bpy.context, cfg, extract_lods=True)

    metadata = json.loads((object_folder / "Metadata.json").read_text(encoding="utf-8"))
    summary = []
    for component_id, component in enumerate(metadata["components"]):
        lods = component.get("lods", [])
        summary.append(
            {
                "component_id": component_id,
                "full_ib": component["ib_hash"],
                "full_vertex_count": component["vertex_count"],
                "lods": lods,
            }
        )

    print("EFMIT_LOD_PROBE=PASS")
    print("EFMIT_LOD_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
