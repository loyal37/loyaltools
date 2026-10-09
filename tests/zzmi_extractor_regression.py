# -*- coding: utf-8 -*-
"""Regression for the Zenless Zone Zero frame analysis extractor (no Blender needed)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR))

import zzmi_synthetic_dump as synthetic  # noqa: E402


def _load_module(relative_path):
    module_path = TESTS_DIR.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_path.stem + "_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_extractor_module():
    return _load_module("extract/zzmi_dump_extractor.py")


def _load_submesh(workspace, unique_str):
    folder = os.path.join(workspace, unique_str, "TYPE_GPU-ZZMI")
    with open(os.path.join(folder, unique_str + ".json"), encoding="utf-8") as handle:
        return folder, json.load(handle)


def _element_summary(submesh_json):
    return {
        category["D3D11ElementList"][0]["Category"]: [
            (element["SemanticName"], element["SemanticIndex"], element["Format"], element["ByteWidth"], element["ExtractSlot"])
            for element in category["D3D11ElementList"]
        ]
        for category in submesh_json["CategoryBufferList"]
    }


def _read(folder, unique_str, category, dtype):
    return numpy.fromfile(os.path.join(folder, unique_str + "-" + category + ".buf"), dtype=dtype)


def main() -> None:
    extractor_module = _load_extractor_module()

    with tempfile.TemporaryDirectory(prefix="loyal_zzmi_extract_") as temp_folder:
        dump_folder = os.path.join(temp_folder, "FrameAnalysis-synthetic")
        workspace = os.path.join(temp_folder, "workplace")
        os.makedirs(dump_folder)
        source = synthetic.build_dump(dump_folder)

        extractor = extractor_module.ZZMIDumpExtractor(dump_folder)

        summaries = {summary.ib_hash: summary for summary in extractor.list_draw_ibs()}
        assert set(summaries) == {synthetic.BODY["ib"], synthetic.FACE["ib"], synthetic.PROP["ib"]}
        assert summaries[synthetic.BODY["ib"]].has_blend
        assert summaries[synthetic.FACE["ib"]].has_blend
        assert not summaries[synthetic.PROP["ib"]].has_blend
        assert summaries[synthetic.BODY["ib"]].draw_call_count == 3
        assert summaries[synthetic.BODY["ib"]].total_index_count == 102
        # Skinned objects are listed before static ones.
        assert [s.has_blend for s in extractor.list_draw_ibs()] == [True, True, False]
        # What "extract everything" works from: every pre-skinned index buffer, nothing static.
        assert extractor.list_skinned_draw_ibs() == [synthetic.BODY["ib"], synthetic.FACE["ib"]]

        result = extractor.extract(
            ib_hashes=[synthetic.BODY["ib"], synthetic.FACE["ib"].upper(), synthetic.PROP["ib"], "deadbeef"],
            workspace_folder=workspace,
            aliases={synthetic.BODY["ib"]: "body"},
        )
        body_grid = synthetic.BODY["ib"] + "-96-0"
        body_quad = synthetic.BODY["ib"] + "-6-96"
        face = synthetic.FACE["ib"] + "-102-0"
        prop = synthetic.PROP["ib"] + "-6-6"
        assert result.unique_strs == [body_grid, body_quad, face, prop], result.unique_strs
        # Two notes: the unknown hash, and (once, not per part) that the normal maps of this dump
        # are lossy jpg files and were therefore left unmarked.
        assert len(result.warnings) == 2, result.warnings
        assert "deadbeef" in result.warnings[0]
        assert "NormalMap" in result.warnings[1] and "dds" in result.warnings[1]

        with open(os.path.join(workspace, "Import.json"), encoding="utf-8") as handle:
            assert json.load(handle) == {unique_str: "GPU-ZZMI" for unique_str in result.unique_strs}
        with open(os.path.join(workspace, "Config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        assert config[0] == {"DrawIB": synthetic.BODY["ib"], "Alias": "body"}
        assert [entry["DrawIB"] for entry in config] == [synthetic.BODY["ib"], synthetic.FACE["ib"], synthetic.PROP["ib"]]

        # --- body: metadata, layout and the un-posed buffers of the stream-output draw -------
        folder, body_json = _load_submesh(workspace, body_grid)
        assert body_json["GamePreset"] == "ZZMI"
        assert body_json["WorkGameType"] == "GPU-ZZMI"
        assert body_json["GPU-PreSkinning"] is True
        assert body_json["CategoryHash"] == {
            "Position": synthetic.BODY["position"],
            "Texcoord": synthetic.BODY["texcoord"],
            "Blend": synthetic.BODY["blend"],
        }
        assert body_json["VertexLimitVB"] == synthetic.BODY["draw"]
        assert body_json["VSHashList"] == [synthetic.VS_COLOR, synthetic.VS_OUTLINE]
        assert body_json["OriginalVertexCount"] == 29
        frame_extract = body_json[extractor_module.FRAME_EXTRACT_KEY]
        assert frame_extract["PreSkinned"] is True
        assert frame_extract["PoseDrawId"] == "000001"
        assert frame_extract["RootVS"] == synthetic.VS_POSE
        assert (frame_extract["FirstIndex"], frame_extract["IndexCount"]) == (0, 96)
        assert _element_summary(body_json) == {
            "Position": [
                ("POSITION", 0, "R32G32B32_FLOAT", 12, "vb0"),
                ("NORMAL", 0, "R32G32B32_FLOAT", 12, "vb0"),
                ("TANGENT", 0, "R32G32B32A32_FLOAT", 16, "vb0"),
            ],
            "Texcoord": [
                ("COLOR", 0, "R8G8B8A8_UNORM", 4, "vb1"),
                ("TEXCOORD", 0, "R16G16_FLOAT", 4, "vb1"),
                ("TEXCOORD", 1, "R32G32_FLOAT", 8, "vb1"),
                ("TEXCOORD", 2, "R16G16_FLOAT", 4, "vb1"),
                ("TEXCOORD", 3, "R16G16_FLOAT", 4, "vb1"),
            ],
            "Blend": [
                ("BLENDWEIGHTS", 0, "R32G32B32A32_FLOAT", 16, "vb2"),
                ("BLENDINDICES", 0, "R32G32B32A32_UINT", 16, "vb2"),
            ],
        }
        body = source["body"]
        assert _read(folder, body_grid, "Position", synthetic.POSITION_DTYPE).tobytes() == body["position"][:25].tobytes()
        assert _read(folder, body_grid, "Texcoord", synthetic.BODY_TEXCOORD_DTYPE).tobytes() == body["texcoord"][:25].tobytes()
        assert _read(folder, body_grid, "Blend", synthetic.BODY_BLEND_DTYPE).tobytes() == body["blend"][:25].tobytes()
        grid_index = numpy.fromfile(os.path.join(folder, body_grid + "-Index.ib"), dtype=numpy.uint32)
        assert grid_index.tolist() == body["grid_indices"].tolist()

        # The second part only keeps the four vertices it references, re-based to 0..3.
        folder, quad_json = _load_submesh(workspace, body_quad)
        assert quad_json["PartName"] == "96"
        assert _read(folder, body_quad, "Position", synthetic.POSITION_DTYPE).tobytes() == body["position"][25:].tobytes()
        quad_index = numpy.fromfile(os.path.join(folder, body_quad + "-Index.ib"), dtype=numpy.uint32)
        assert quad_index.tolist() == (body["quad_indices"] - 25).tolist()

        # --- body textures: candidates, render target exclusion, automatic marks --------------
        folder, _ = _load_submesh(workspace, body_grid)
        with open(os.path.join(folder, "TextureSlots.json"), encoding="utf-8") as handle:
            texture_slots = json.load(handle)
        assert texture_slots["version"] == 2
        assert sorted(texture_slots["calls"]) == ["000010", "000012"]
        assert [entry["slot"] for entry in texture_slots["calls"]["000010"]] == ["ps-t3", "ps-t4", "ps-t5", "ps-t6"]
        assert [entry["slot"] for entry in texture_slots["calls"]["000012"]] == ["ps-t2", "ps-t3"]
        diffuse_entry = texture_slots["calls"]["000010"][0]
        assert diffuse_entry == {
            "slot": "ps-t3", "hash": synthetic.TEX_DIFFUSE,
            "filename": "t-" + synthetic.TEX_DIFFUSE + "-BC7_UNORM_SRGB.dds",
            "format": "BC7_UNORM_SRGB", "width": 256, "height": 256,
        }
        assert texture_slots["calls"]["000010"][1]["filename"] == "t-" + synthetic.TEX_NORMAL + "-R8G8B8A8_UNORM.jpg"
        assert not any(synthetic.RENDER_TARGET in name for name in os.listdir(folder))

        # The normal map was dumped as a lossy jpg, so it is offered but not marked.
        marks = {mark["MarkName"]: mark for mark in body_json["TextureMarkUpInfoList"]}
        assert sorted(marks) == ["DiffuseMap", "LightMap", "MaterialMap"]
        assert marks["DiffuseMap"] == {
            "MarkName": "DiffuseMap", "MarkType": "Slot", "MarkHash": synthetic.TEX_DIFFUSE,
            "MarkSlot": "ps-t3", "MarkFileName": body_grid + "-DiffuseMap.dds",
            "SourceFileName": "t-" + synthetic.TEX_DIFFUSE + "-BC7_UNORM_SRGB.dds",
        }
        assert marks["LightMap"]["MarkSlot"] == "ps-t5" and marks["LightMap"]["MarkHash"] == synthetic.TEX_LIGHT
        assert marks["MaterialMap"]["MarkSlot"] == "ps-t6" and marks["MaterialMap"]["MarkHash"] == synthetic.TEX_MATERIAL
        for mark in marks.values():
            assert os.path.isfile(os.path.join(folder, mark["MarkFileName"]))

        # --- face: same stream-output hash as the body, resolved through its own buffers ------
        folder, face_json = _load_submesh(workspace, face)
        assert face_json["CategoryHash"]["Position"] == synthetic.FACE["position"]
        assert face_json[extractor_module.FRAME_EXTRACT_KEY]["PoseDrawId"] == "000002"
        assert _element_summary(face_json)["Blend"] == [("BLENDINDICES", 0, "R32_UINT", 4, "vb2")]
        assert _element_summary(face_json)["Texcoord"][0] == ("COLOR", 0, "R32G32B32A32_FLOAT", 16, "vb1")
        assert _read(folder, face, "Position", synthetic.POSITION_DTYPE).tobytes() == source["face"]["position"].tobytes()
        assert _read(folder, face, "Blend", synthetic.FACE_BLEND_DTYPE).tobytes() == source["face"]["blend"].tobytes()
        assert [mark["MarkName"] for mark in face_json["TextureMarkUpInfoList"]] == ["DiffuseMap"]

        # --- prop: static mesh, BaseVertexLocation applied, one interleaved category ----------
        folder, prop_json = _load_submesh(workspace, prop)
        assert prop_json["GPU-PreSkinning"] is False
        assert prop_json[extractor_module.FRAME_EXTRACT_KEY]["PreSkinned"] is False
        assert list(prop_json["CategoryHash"]) == ["Position"]
        assert _read(folder, prop, "Position", synthetic.PROP_DTYPE).tobytes() == source["prop"]["buffer"].tobytes()
        prop_index = numpy.fromfile(os.path.join(folder, prop + "-Index.ib"), dtype=numpy.uint32)
        assert prop_index.tolist() == source["prop"]["indices"].tolist()

        # --- re-extracting keeps manual marks and does not bring back a replaced automatic one --
        folder, _ = _load_submesh(workspace, body_grid)
        json_path = os.path.join(folder, body_grid + ".json")
        body_json["TextureMarkUpInfoList"] = [
            {
                "MarkName": "DiffuseMap", "MarkType": "Slot", "MarkHash": "12345678", "MarkSlot": "ps-t2",
                "MarkFileName": body_grid + "-DiffuseMap.dds", "SourceFileName": "manual.dds",
            },
            {
                "MarkName": "GlowMap", "MarkType": "Slot", "MarkHash": "87654321", "MarkSlot": "ps-t8",
                "MarkFileName": body_grid + "-GlowMap.dds", "SourceFileName": "glow.dds",
            },
        ]
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(body_json, handle)
        with open(os.path.join(folder, body_grid + "-DiffuseMap.dds"), "wb") as handle:
            handle.write(b"manual")
        extractor_module.ZZMIDumpExtractor(dump_folder).extract([synthetic.BODY["ib"]], workspace)
        _, body_json = _load_submesh(workspace, body_grid)
        marks = {mark["MarkName"]: mark for mark in body_json["TextureMarkUpInfoList"]}
        assert sorted(marks) == ["DiffuseMap", "GlowMap", "LightMap", "MaterialMap"]
        assert marks["DiffuseMap"]["MarkHash"] == "12345678"
        with open(os.path.join(folder, body_grid + "-DiffuseMap.dds"), "rb") as handle:
            assert handle.read() == b"manual"

        # --- lossless dumps: a normal map dumped as dds (analyse_options with "dds") is marked ---
        color_shaders = "-vs=" + synthetic.VS_COLOR + "-ps=" + synthetic.PS_COLOR
        dds_names = []
        for draw_id, tex_hash, dxgi_format, contaminated in (
            (10, synthetic.TEX_NORMAL, 28, True),        # R8G8B8A8_UNORM, body part 1
            (11, synthetic.TEX_NORMAL, 28, True),        # body part 2
            (13, synthetic.TEX_FACE_NORMAL, 10, False),  # R16G16B16A16_FLOAT, face
        ):
            jpg_name = "%06d-ps-t4=%s%s%s.jpg" % (draw_id, "!S!=" if contaminated else "", tex_hash, color_shaders)
            os.remove(os.path.join(dump_folder, jpg_name))
            dds_names.append(jpg_name[:-4] + ".dds")
            with open(os.path.join(dump_folder, dds_names[-1]), "wb") as handle:
                handle.write(synthetic._dds(256, 256, dxgi_format, 1024))
        lossless_workspace = os.path.join(temp_folder, "lossless")
        lossless = extractor_module.ZZMIDumpExtractor(dump_folder).extract(
            [synthetic.BODY["ib"], synthetic.FACE["ib"]], lossless_workspace,
        )
        assert not lossless.warnings, lossless.warnings
        folder, lossless_json = _load_submesh(lossless_workspace, body_grid)
        marks = {mark["MarkName"]: mark for mark in lossless_json["TextureMarkUpInfoList"]}
        assert list(marks) == ["DiffuseMap", "NormalMap", "LightMap", "MaterialMap"]
        assert marks["NormalMap"] == {
            "MarkName": "NormalMap", "MarkType": "Slot", "MarkHash": synthetic.TEX_NORMAL,
            "MarkSlot": "ps-t4", "MarkFileName": body_grid + "-NormalMap.dds",
            "SourceFileName": "t-" + synthetic.TEX_NORMAL + "-R8G8B8A8_UNORM.dds",
        }
        # The marked copy is the dumped file itself, byte for byte.
        with open(os.path.join(folder, body_grid + "-NormalMap.dds"), "rb") as handle:
            assert handle.read() == synthetic._dds(256, 256, 28, 1024)
        _, lossless_face_json = _load_submesh(lossless_workspace, face)
        face_marks = {mark["MarkName"]: mark for mark in lossless_face_json["TextureMarkUpInfoList"]}
        assert list(face_marks) == ["DiffuseMap", "NormalMap"]
        assert face_marks["NormalMap"]["SourceFileName"] == "t-" + synthetic.TEX_FACE_NORMAL + "-R16G16B16A16_FLOAT.dds"
        for name in dds_names:
            os.remove(os.path.join(dump_folder, name))
            with open(os.path.join(dump_folder, name[:-4] + ".jpg"), "wb") as handle:
                handle.write(synthetic._jpg(256, 256))

        # --- a draw range overlapping the parts is extracted too, with one warning ------------
        log_path = os.path.join(dump_folder, "log.txt")
        with open(log_path, encoding="utf-8") as handle:
            original_log = handle.read()
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(original_log + "\n".join([
                "000020 IASetVertexBuffers(StartSlot:0, NumBuffers:4, ppVertexBuffers:0x0, pStrides:0x0, pOffsets:0x0)",
                "       0: resource=0x00000215B3BA37E0 hash=" + synthetic.BODY["draw"],
                "       1: resource=0x00000215B3BA37E0 hash=" + synthetic.BODY["texcoord"],
                "       2: resource=0x00000215B3BA37E0 hash=" + synthetic.BODY["blend"],
                "       3: resource=0x00000215B3BA37E0 hash=" + synthetic.BODY["draw"],
                "000020 IASetIndexBuffer(pIndexBuffer:0x000002153EF21860, Format:57, Offset:0) hash=" + synthetic.BODY["ib"],
                "000020 DrawIndexed(IndexCount:102, StartIndexLocation:0, BaseVertexLocation:0)",
                "",
            ]))
        overlap_workspace = os.path.join(temp_folder, "overlap")
        overlap = extractor_module.ZZMIDumpExtractor(dump_folder).extract([synthetic.BODY["ib"]], overlap_workspace)
        assert overlap.unique_strs == [body_grid, synthetic.BODY["ib"] + "-102-0", body_quad], overlap.unique_strs
        assert [warning for warning in overlap.warnings if "重叠" in warning] == overlap.warnings[:1]
        assert len(overlap.warnings) == 2 and "NormalMap" in overlap.warnings[1], overlap.warnings

        # --- without the stream-output draw the posed buffers are all there is: say so --------
        kept_lines = []
        skip_slot_lines = False
        for line in original_log.splitlines():
            if "SOSetTargets(" in line:
                skip_slot_lines = True
                continue
            if skip_slot_lines and line.startswith("   "):
                continue
            skip_slot_lines = False
            kept_lines.append(line)
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(kept_lines) + "\n")
        posed_workspace = os.path.join(temp_folder, "posed")
        posed = extractor_module.ZZMIDumpExtractor(dump_folder).extract([synthetic.BODY["ib"]], posed_workspace)
        assert posed.unique_strs == [body_grid, body_quad]
        assert any("预蒙皮" in warning for warning in posed.warnings), posed.warnings
        _, posed_json = _load_submesh(posed_workspace, body_grid)
        assert posed_json["GPU-PreSkinning"] is False
        assert posed_json[extractor_module.FRAME_EXTRACT_KEY]["PreSkinned"] is False
        assert "Blend" not in posed_json["CategoryHash"]
        # The indexed draws only describe POSITION and NORMAL of the posed buffer.
        assert [name for name, *_ in _element_summary(posed_json)["Position"]] == ["POSITION", "NORMAL", "UNKNOWN"]
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(original_log)

        # --- clear errors for folders that are not frame analysis dumps -----------------------
        try:
            extractor_module.ZZMIDumpExtractor(temp_folder)
        except extractor_module.ExtractError as error:
            assert "log.txt" in str(error)
        else:
            raise AssertionError("a folder without log.txt must be rejected")

    print("ZZMI_EXTRACTOR_REGRESSION=PASS")


if __name__ == "__main__":
    main()
