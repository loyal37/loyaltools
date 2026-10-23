# -*- coding: utf-8 -*-
"""Blender regression: Zenless Zone Zero merged skeleton, from extraction to the mod.

Run with:
    blender --background --factory-startup --python-exit-code 1 \
        --python tests/blender_zzmi_merged_export_regression.py

1. "Merged skeleton" extraction of a synthetic dump: merged parts share one set
   of vertex groups, the face keeps its own and the bangboo is left out.
2. An unchanged export needs no runtime: every part skins with its own bones
   and the face keeps the game mesh (textures only).
3. A copy of the hair joined into the body: the body is skinned with a merged
   palette built from the bone pool, the hair imports its bones into the pool
   when it is skinned, and the exported vertices, run through the palette,
   land where the game puts them.
4. The hair and the accessory taken out of the blueprint: as in EFMI's merged
   skeleton they are no longer drawn; the hair still provides its bones.

Every skinning branch is limited to the component's own skinning draw
(``VERTEX_COUNT``): full-screen passes keep a character's vb2 bound and must not
trigger it.  No stream-output target is ever re-bound.

Each case is also exported with the "uniform one-frame delay" option: the whole
character, the face included, is skinned with the bones of the previous frame
(the pool is copied at [Present]), the buffers stay byte-identical, and
switching the option off again gives the ordinary export.

And with the "same-frame skinning" option: a part that uses bones of other
components only imports its bones at its skinning draw and is skinned by the
mod at its first draw of the frame, into a buffer of the mod bound as vb0 (the
previous result as vb3); the skinning parameters decode the exported buffers.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import bpy
import numpy

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parents[1]))
sys.path.insert(0, str(TESTS_DIR))

import zzmi_merged_synthetic_dump as synthetic  # noqa: E402

# The game's object constants as the game bound them for the draw (a part of a shared buffer).
OBJECT_WINDOW_CB0 = "cs-cb0 = ref vs-cb1->Region($zzms_cb_offset, $zzms_cb_size)"


def _section(ini_text, header):
    assert header in ini_text, header + " missing"
    lines = ini_text.split(header, 1)[1].split("\n[", 1)[0].splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith(";")]


def _export(tree_name, output_dir, frame_delay=False, same_frame=False):
    props = bpy.context.scene.global_properties
    props.generate_mod_folder_path = output_dir
    props.zzz_merged_frame_delay = frame_delay
    props.zzz_merged_same_frame = same_frame
    os.makedirs(output_dir, exist_ok=True)
    bpy.ops.ssmt.generate_mod_blueprint(blueprint_name=tree_name)
    ini_files = [name for name in os.listdir(output_dir) if name.lower().endswith(".ini")]
    assert len(ini_files) == 1, ini_files
    with open(os.path.join(output_dir, ini_files[0]), encoding="utf-8") as handle:
        return handle.read()


def _read_body_buffers(mod_folder, ib_hash):
    meshes = os.path.join(mod_folder, "Meshes")
    position = numpy.fromfile(os.path.join(meshes, ib_hash + "-Position.buf"), dtype=numpy.float32).reshape(-1, 10)
    blend = numpy.fromfile(os.path.join(meshes, ib_hash + "-Blend.buf"), dtype=numpy.uint8).reshape(-1, 32)
    weights = blend[:, :16].copy().view(numpy.float32)
    indices = blend[:, 16:].copy().view(numpy.uint32).astype(numpy.int64)
    return position, weights, indices


def _draw_count(mod_folder, ib_hash):
    return os.path.getsize(os.path.join(mod_folder, "Meshes", ib_hash + "-Position.buf")) // 40


def _same_meshes(mod_a, mod_b):
    meshes_a, meshes_b = os.path.join(mod_a, "Meshes"), os.path.join(mod_b, "Meshes")
    names = sorted(os.listdir(meshes_a))
    assert names == sorted(os.listdir(meshes_b)), (names, os.listdir(meshes_b))
    for name in names:
        with open(os.path.join(meshes_a, name), "rb") as a, open(os.path.join(meshes_b, name), "rb") as b:
            assert a.read() == b.read(), name


def _check_delay_frame(ini_text, zms, profile, parts, pool_total, placed):
    """Sections every delayed export has: the [Present] copies, the face hook and the placements.

    ``placed``: ib hashes of the parts drawn through their own object matrix, which are
    drawn with their matrix of the previous frame, moved by the frame's shared distance.
    """
    face = zms.get_component_by_ib(profile)[parts["face"]["spec"]["ib"]]
    face_name = face["ib_hash"]
    constants = _section(ini_text, "[Constants]")
    assert "global $zzms_imported = 0" in constants and "global $zzms_ready = 0" in constants, constants
    present = _section(ini_text, "[Present]")
    assert present[:2] == ["if $zzms_imported", "run = CustomShaderZZMS_Advance"], present
    assert present[2:2 + len(placed)] == ["run = CustomShaderZZMS_PlaceAdvance_" + ib for ib in placed], present
    assert present[2 + len(placed):] == [
        "run = CustomShaderZZMS_ShiftClear",
        "$zzms_ready = 1", "else", "$zzms_ready = 0", "endif", "$zzms_imported = 0"], present
    for ib in placed:
        _check_place(ini_text, zms, ib)
    # Every part of a frame is moved by the same distance, cleared at [Present].
    assert _section(ini_text, "[CustomShaderZZMS_ShiftClear]") == [
        "cs = Shaders/" + zms.PLACE_SHADER_FILENAME,
        "cs-t1 = ref ResourceZZMS_ShiftClearParams", "cs-u2 = ref ResourceZZMS_Shift",
        "dispatch = 1, 1, 1", "cs-u2 = null", "cs-t1 = null"]
    assert _section(ini_text, "[ResourceZZMS_ShiftClearParams]")[-1] == "data = 2 0 0 0"
    assert _section(ini_text, "[ResourceZZMS_Shift]") == [
        "type = RWBuffer", "format = R32_FLOAT", "array = 4", "bind_flags = shader_resource unordered_access"]
    assert _section(ini_text, "[CustomShaderZZMS_Advance]") == [
        "cs = Shaders/" + zms.DELAY_SHADER_FILENAME,
        "cs-t0 = ref ResourceZZMS_Pool", "cs-t1 = ref ResourceZZMS_AdvanceParams",
        "cs-u0 = ref ResourceZZMS_PoolPrev",
        "dispatch = " + str((pool_total + 63) // 64) + ", 1, 1",
        "cs-u0 = null", "cs-t0 = null", "cs-t1 = null"]
    assert _section(ini_text, "[ResourceZZMS_AdvanceParams]")[-1] == "data = 0 " + str(pool_total) + " 0 0"
    for name in ("[ResourceZZMS_Pool]", "[ResourceZZMS_PoolPrev]"):
        assert _section(ini_text, name) == [
            "type = RWBuffer", "format = R32G32B32A32_FLOAT", "array = " + str(pool_total * 3),
            "bind_flags = shader_resource unordered_access"], name
    # The face keeps the game's mesh and expression; only its bone table is the previous one.
    assert _section(ini_text, "[TextureOverride_ZZMS_Delay_" + face_name + "]") == [
        "hash = " + parts["face"]["spec"]["blend"],
        "if DRAW_TYPE == 1 && VERTEX_COUNT == " + str(face["pose_vertex_count"]),
        "run = CommandListZZMS_Skin_" + face_name,
        "endif"]
    # The game draws the face; each face part is drawn again with the placed object constants
    # after its textures are bound, the original draw is skipped and vs-cb1 is put back.
    face_part = _section(ini_text, "[TextureOverride_" + parts["face"]["unique_str"].replace("-", "_") + "]")
    assert face_part[-6:] == _face_draw_lines(face_name), face_part
    assert face_part.index(r"run = CommandList\ZZMI\SetTextures") < len(face_part) - 6, face_part
    assert _section(ini_text, "[CommandListZZMS_Skin_" + face_name + "]") == [
        "ResourceZZMS_Input = ref vs-t0",
        "run = CustomShaderZZMS_Import_" + face_name,
        "$zzms_imported = 1",
        "if $zzms_ready",
        "run = CustomShaderZZMS_Delay_" + face_name,
        "vs-t0 = ref ResourceZZMS_Delayed_" + face_name,
        "endif"], _section(ini_text, "[CommandListZZMS_Skin_" + face_name + "]")
    assert _section(ini_text, "[ResourceZZMS_DelayParams_" + face_name + "]")[-1] == (
        "data = " + str(profile["bones_count"]) + " " + str(face["palette_bone_count"]) + " 0 0")
    assert "so0" not in ini_text and "handling = skip" not in _section(
        ini_text, "[TextureOverride_ZZMS_Delay_" + face_name + "]")
    return face


def _face_draw_lines(ib):
    return [
        "run = CommandListZZMS_PlaceBegin_" + ib,
        "if $zzms_ready", "draw = from_caller", "handling = skip", "endif",
        "run = CommandListZZMS_PlaceEnd_" + ib,
    ]


def _check_placed_draw(ini_text, part_unique_str, ib):
    """A part we draw: its object constants are swapped in right before its drawindexed and back right after."""
    lines = _section(ini_text, "[TextureOverride_" + part_unique_str.replace("-", "_") + "]")
    draws = [index for index, line in enumerate(lines) if line.startswith("drawindexed")]
    assert draws, lines
    assert lines[draws[0] - 1] == "run = CommandListZZMS_PlaceBegin_" + ib, lines
    assert lines[-1] == "run = CommandListZZMS_PlaceEnd_" + ib and draws[-1] == len(lines) - 2, lines


def _check_place(ini_text, zms, ib):
    """A placed part: its object constants are replaced when it is drawn."""
    assert _section(ini_text, "[CommandListZZMS_PlaceBegin_" + ib + "]") == [
        "if $zzms_ready",
        "run = CustomShaderZZMS_Place_" + ib,
        "ResourceZZMS_GameCb = ref vs-cb1",
        "vs-cb1 = copy ResourceZZMS_PlaceCb_" + ib,
        "else",
        "run = CustomShaderZZMS_PlaceRecord_" + ib,
        "endif"], ib
    assert _section(ini_text, "[CommandListZZMS_PlaceEnd_" + ib + "]") == [
        "if $zzms_ready", "vs-cb1 = ref ResourceZZMS_GameCb", "endif"], ib
    for section, mode, shift in (("[CustomShaderZZMS_Place_", "ResourceZZMS_PlaceOn", True),
                                 ("[CustomShaderZZMS_PlaceRecord_", "ResourceZZMS_PlaceOff", False)):
        assert _section(ini_text, section + ib + "]") == [
            "cs = Shaders/" + zms.PLACE_SHADER_FILENAME,
            "cs-cb0 = ref vs-cb1",
            "cs-t0 = ref ResourceZZMS_History_" + ib,
            "cs-t1 = ref " + mode,
            "cs-u0 = ref ResourceZZMS_PlaceCb_" + ib,
            "cs-u1 = ref ResourceZZMS_Record_" + ib,
        ] + (["cs-u2 = ref ResourceZZMS_Shift"] if shift else []) + [
            "dispatch = 1, 1, 1",
            "cs-u0 = null", "cs-u1 = null",
        ] + (["cs-u2 = null"] if shift else []) + [
            "cs-t0 = null", "cs-t1 = null", "cs-cb0 = null"], ib
    assert _section(ini_text, "[ResourceZZMS_PlaceCb_" + ib + "]")[2] == "array = 29"
    # The previous frame's matrices are this mod's own records: copied at [Present].
    assert _section(ini_text, "[CustomShaderZZMS_PlaceAdvance_" + ib + "]")[1:4] == [
        "cs-t0 = ref ResourceZZMS_Record_" + ib, "cs-t1 = ref ResourceZZMS_PlaceAdvanceParams",
        "cs-u0 = ref ResourceZZMS_History_" + ib], ib


def _check_same_frame(ini_text, zms, profile, mod_folder, ib, unique_str, layout, participants, needed):
    """A part skinned by the mod at the first draw of each copy of the character, into that copy's buffers.

    ``participants``: ib hashes that import their bones (into the slot of each copy's stream-output
    buffer) and record where they are when drawn; ``needed``: those whose bones this part reads.
    """
    draw_count = _draw_count(mod_folder, ib)
    by_ib = zms.get_component_by_ib(profile)
    participant_ids = [by_ib[value]["component_id"] for value in participants]
    slots = zms.instance_slot_count(len(participants))
    rows = zms.instance_slot_rows(profile, participant_ids)
    lines = _section(ini_text, "[TextureOverride_" + unique_str.replace("-", "_") + "]")
    draws = [index for index, line in enumerate(lines) if line.startswith("drawindexed")]
    # Where it is is recorded before vb0 is replaced (its identity is that of the game's buffer); the outputs
    # skinned are the ones drawn (found once, by number).
    assert draws and lines[draws[0] - 6:draws[0]] == [
        "run = CommandListZZMS_Record", "run = CommandListZZMS_SkinNow_" + ib,
        "ResourceZZMS_GameVb0 = ref vb0", "ResourceZZMS_GameVb3 = ref vb3",
        "vb0 = ref PoolZZMS_Out_" + ib + "[$zzms_o0]", "vb3 = ref PoolZZMS_Out_" + ib + "[$zzms_o3]"], lines
    assert lines[draws[-1] + 1:] == ["vb0 = ref ResourceZZMS_GameVb0", "vb3 = ref ResourceZZMS_GameVb3"], lines
    # Slots are claimed before IniParams are saved (a claim saves them itself); marks are kept by slot.
    # The copy is known by the identities of both of its buffers (owner entries), which the game alternates.
    # Both outputs are found before either is used (a lookup can take the other's buffer).
    assert _section(ini_text, "[CommandListZZMS_SkinNow_" + ib + "]") == [
        "$zzms_host0 = @vb0", "$zzms_host3 = @vb3",
        "$zzms_prev3 = $zzms_host0",
        "if $zzms_host0 != 0",
        "$zzms_claim = $zzms_host0", "run = CommandListZZMS_Claim", "$zzms_s0 = $zzms_claimed",
        "$zzms_k3 = $zzms_host0",
        "if $zzms_host3 != 0",
        "$zzms_claim = $zzms_host3", "run = CommandListZZMS_Claim", "$zzms_k3 = $zzms_host3",
        "if $PoolZZMS_SkinnedAt[$zzms_claimed] == FRAME_NUMBER - 1", "$zzms_prev3 = $zzms_host3", "endif",
        "endif",
        "$zzms_o0 = PoolZZMS_OutSlot_" + ib + "[$zzms_host0]->Index",
        "$zzms_o3 = $zzms_o0",
        "if $zzms_prev3 != $zzms_host0",
        "$zzms_o3 = PoolZZMS_OutSlot_" + ib + "[$zzms_prev3]->Index",
        "if $zzms_o3 == $zzms_o0", "$zzms_o0 = PoolZZMS_OutSlot_" + ib + "[$zzms_host0]->Index", "endif",
        "endif",
        # Skinned again when either output passed to another buffer since (both numbers are below 64).
        "if $PoolZZMS_SkinnedAt[$zzms_s0] != FRAME_NUMBER || $PoolZZMS_OutAt[$zzms_s0] != $zzms_o0 * 64 + $zzms_o3",
        "$PoolZZMS_SkinnedAt[$zzms_s0] = FRAME_NUMBER", "$PoolZZMS_OutAt[$zzms_s0] = $zzms_o0 * 64 + $zzms_o3",
        "run = CommandListZZMS_ObjectWindow",
        "run = CommandListZZMS_BackupIniParams",
        "x0 = $zzms_s0", "y0 = FRAME_NUMBER", "z0 = $zzms_host0", "w0 = $zzms_k3",
        "run = CustomShaderZZMS_Match_" + ib, "run = CustomShaderZZMS_Gather_" + ib,
        "run = CustomShaderZZMS_Palette_" + ib,
        "if $zzms_prev3 == $zzms_host0", "w0 = 0", "endif",
        "run = CustomShaderZZMS_Skin_" + ib,
        "run = CommandListZZMS_RestoreIniParams",
        "endif",
        "endif"]
    # Several objects can share the game's buffer, each bound as a part of it: that part is bound.
    assert _section(ini_text, "[CustomShaderZZMS_Match_" + ib + "]") == [
        "cs = Shaders/" + zms.INSTANCE_MATCH_SHADER_FILENAME, OBJECT_WINDOW_CB0,
        "cs-t0 = ref ResourceZZMS_Store", "cs-t1 = ref ResourceZZMS_Records", "cs-t2 = ref ResourceZZMS_Components",
        "cs-t3 = ref ResourceZZMS_InstanceParams_" + ib, "cs-t4 = ref ResourceZZMS_Needed_" + ib,
        "cs-u0 = ref ResourceZZMS_Choice", "cs-u1 = ref ResourceZZMS_Owners", "dispatch = 1, 1, 1",
        "cs-u0 = null", "cs-u1 = null",
        "cs-t0 = null", "cs-t1 = null", "cs-t2 = null", "cs-t3 = null", "cs-t4 = null", "cs-cb0 = null"]
    assert _section(ini_text, "[CustomShaderZZMS_Gather_" + ib + "]") == [
        "cs = Shaders/" + zms.INSTANCE_GATHER_SHADER_FILENAME,
        "cs-t0 = ref ResourceZZMS_Store", "cs-t1 = ref ResourceZZMS_Components",
        "cs-t2 = ref ResourceZZMS_InstanceParams_" + ib, "cs-t3 = ref ResourceZZMS_Choice",
        "cs-u0 = ref ResourceZZMS_Pool", "dispatch = " + str((profile["bones_count"] + 63) // 64) + ", 1, 1",
        "cs-u0 = null", "cs-t0 = null", "cs-t1 = null", "cs-t2 = null", "cs-t3 = null"]
    # The previous output (vb3) is bound too: when it is not a valid result it gets this frame's.  Without one
    # the slot is cleared (it can hold a buffer of the game or of another mod; one buffer bound to two
    # UAV slots is dropped from both by D3D11).
    assert _section(ini_text, "[CustomShaderZZMS_Skin_" + ib + "]") == [
        "cs = Shaders/" + zms.SKIN_SHADER_FILENAME,
        "cs-t0 = ref ResourceZZMS_Palette_" + ib, "cs-t1 = ref ResourceZZMS_SkinPosition_" + ib,
        "cs-t2 = ref ResourceZZMS_SkinBlend_" + ib, "cs-t3 = ref ResourceZZMS_SkinParams_" + ib,
        "cs-t4 = ref ResourceZZMS_Choice", "cs-u0 = ref PoolZZMS_Out_" + ib + "[$zzms_o0]",
        "if $zzms_o3 != $zzms_o0", "cs-u1 = ref PoolZZMS_Out_" + ib + "[$zzms_o3]", "else", "cs-u1 = null", "endif",
        "dispatch = " + str((draw_count + 63) // 64) + ", 1, 1", "cs-u0 = null", "cs-u1 = null",
        "cs-t0 = null", "cs-t1 = null", "cs-t2 = null", "cs-t3 = null", "cs-t4 = null"]
    for gone in ("[CustomShaderZZMS_SkinFrame_", "[CustomShaderZZMS_SkinFirst_"):
        assert gone not in ini_text, gone
    assert "cs-t0 = ref ResourceZZMS_Pool" in _section(ini_text, "[CustomShaderZZMS_Palette_" + ib + "]")
    for category in ("Position", "Blend"):
        assert _section(ini_text, "[ResourceZZMS_Skin" + category + "_" + ib + "]") == [
            "type = ByteAddressBuffer", "filename = Meshes/" + ib + "-" + category + ".buf"], category
    assert _section(ini_text, "[ResourceZZMS_SkinParams_" + ib + "]")[-1] == "data = " + " ".join(
        str(value) for value in zms.same_frame_skin_params(draw_count, layout))
    assert _section(ini_text, "[ResourceZZMS_Needed_" + ib + "]")[-1] == "data = " + " ".join(
        str(value) for value in zms.instance_needed(profile, {by_ib[value]["component_id"] for value in needed}))
    assert _section(ini_text, "[ResourceZZMS_InstanceParams_" + ib + "]")[-1] == "data = " + " ".join(
        str(number) for number in zms.instance_params(profile, by_ib[ib]["component_id"], slots, rows))
    # Outputs: numbers given by buffer identity, buffers taken by number.  One element after the vertices
    # holds whether they are a valid result and whose.  No expiration: XXMI resets an expired resource
    # element to null for good.
    assert _section(ini_text, "[PoolZZMS_OutSlot_" + ib + "]") == [
        "pool_size = 32", "pool_index_type = fifo", "pool_allocate_slot_on_missing = 1"]
    assert _section(ini_text, "[PoolZZMS_Out_" + ib + "]") == [
        "pool_size = 32", "pool_index_type = ring",
        "type = RWByteAddressBuffer", "stride = 40",
        "array = " + str(draw_count + 1), "bind_flags = vertex_buffer shader_resource unordered_access"]

    # Every participant imports into the slot of its stream-output buffer (one import for all, the part
    # in $zzms_table) and records where it is (one record for all: the part is the one imported there).
    for value in participants:
        component_id = by_ib[value]["component_id"]
        assert _section(ini_text, "[CommandListZZMS_Skin_" + value + "]")[:3] == [
            "ResourceZZMS_Input = ref vs-t0",
            "$zzms_table = " + str(zms.runtime_component_index(profile, component_id)),
            "run = CommandListZZMS_Import"], value
        for gone in ("[CustomShaderZZMS_Import_" + value + "]", "[CommandListZZMS_Record_" + value + "]",
                     "[CustomShaderZZMS_Record_" + value + "]"):
            assert gone not in ini_text, gone
        if value not in (ib,):
            assert "[ResourceZZMS_InstanceParams_" + value + "]" not in ini_text, value
    params = "ResourceZZMS_InstanceParams_"
    assert _section(ini_text, "[CommandListZZMS_Import]") == [
        "$zzms_claim = @so0", "run = CommandListZZMS_Claim",
        "run = CommandListZZMS_BackupIniParams", "x0 = $zzms_claimed", "y0 = FRAME_NUMBER", "z0 = $zzms_table",
        "run = CustomShaderZZMS_Import", "run = CommandListZZMS_RestoreIniParams"]
    import_lines = _section(ini_text, "[CustomShaderZZMS_Import]")
    groups = max((by_ib[value]["vg_count"] + 63) // 64 for value in participants)
    # Any part's parameters do (slot count, rows per slot and component count are shared).
    assert import_lines[3].startswith("cs-t2 = ref " + params), import_lines
    assert import_lines[:3] + import_lines[4:] == [
        "cs = Shaders/" + zms.INSTANCE_IMPORT_SHADER_FILENAME, "cs-t0 = ref ResourceZZMS_Input",
        "cs-t1 = ref ResourceZZMS_Components", "cs-u0 = ref ResourceZZMS_Store",
        "dispatch = " + str(groups) + ", 1, 1",
        "cs-u0 = null", "cs-t0 = null", "cs-t1 = null", "cs-t2 = null"], import_lines
    assert _section(ini_text, "[CommandListZZMS_Record]") == [
        "$zzms_key3 = @vb3", "$zzms_claim = @vb0",
        "if $zzms_claim != 0",
        "run = CommandListZZMS_Claim", "$zzms_s0 = $zzms_claimed",
        "if $PoolZZMS_RecordedAt[$zzms_s0] != FRAME_NUMBER",
        "$zzms_s3 = $zzms_s0",
        "if $zzms_key3 != 0",
        "$PoolZZMS_RecordedAt[$zzms_s0] = FRAME_NUMBER",
        "$zzms_claim = $zzms_key3", "run = CommandListZZMS_Claim", "$zzms_s3 = $zzms_claimed",
        "endif",
        "run = CommandListZZMS_ObjectWindow",
        "run = CommandListZZMS_BackupIniParams",
        "x0 = $zzms_s0", "y0 = $zzms_s3", "z0 = FRAME_NUMBER",
        "run = CustomShaderZZMS_Record", "run = CommandListZZMS_RestoreIniParams",
        "endif",
        "endif"]
    record_lines = _section(ini_text, "[CustomShaderZZMS_Record]")
    assert record_lines[2].startswith("cs-t0 = ref " + params), record_lines
    assert record_lines[:2] + record_lines[3:] == [
        "cs = Shaders/" + zms.INSTANCE_RECORD_SHADER_FILENAME, OBJECT_WINDOW_CB0,
        "cs-t1 = ref ResourceZZMS_Store", "cs-u0 = ref ResourceZZMS_Records", "dispatch = 1, 1, 1",
        "cs-u0 = null", "cs-t0 = null", "cs-t1 = null", "cs-cb0 = null"], record_lines
    assert _section(ini_text, "[ResourceZZMS_Store]")[2] == "array = " + str(slots * rows)
    assert _section(ini_text, "[ResourceZZMS_Records]")[2] == "array = " + str(slots * 5)
    assert _section(ini_text, "[ResourceZZMS_Owners]") == [
        "type = RWBuffer", "format = R32G32B32A32_FLOAT",
        "array = " + str(zms.instance_owner_rows(slots, len(zms.merged_components(profile)))),
        "bind_flags = shader_resource unordered_access"]
    assert _section(ini_text, "[ResourceZZMS_Choice]") == [
        "type = RWBuffer", "format = R32G32B32A32_UINT", "array = " + str(1 + len(zms.merged_components(profile))),
        "bind_flags = shader_resource unordered_access"]
    # One pool gives slots by buffer identity; the rest is kept by slot (ring), with the slot's buffer.
    assert _section(ini_text, "[PoolZZMS_Slot]") == [
        "pool_size = " + str(slots), "pool_index_type = fifo", "pool_allocate_slot_on_missing = 1",
        "pool_expiration_timeout_frames = 600", "pool_expiration_refresh_on_read = 1"]
    for pool_name in ("PoolZZMS_SlotKey", "PoolZZMS_RecordedAt", "PoolZZMS_SkinnedAt", "PoolZZMS_OutAt"):
        assert _section(ini_text, "[" + pool_name + "]") == [
            "pool_size = " + str(slots), "pool_index_type = ring"], pool_name
    for gone in ("[PoolZZMS_Recorded]", "[PoolZZMS_Skinned]"):
        assert gone not in ini_text, gone
    # A slot that passes to another buffer is cleared (marks, and on the GPU its header, records, owner entries).
    assert _section(ini_text, "[CommandListZZMS_Claim]") == [
        "$zzms_claimed = PoolZZMS_Slot[$zzms_claim]->Index",
        "if $PoolZZMS_SlotKey[$zzms_claimed] != $zzms_claim",
        "$PoolZZMS_SlotKey[$zzms_claimed] = $zzms_claim",
        "$PoolZZMS_RecordedAt[$zzms_claimed] = 0", "$PoolZZMS_SkinnedAt[$zzms_claimed] = 0",
        "$PoolZZMS_OutAt[$zzms_claimed] = -1",
        "run = CommandListZZMS_BackupIniParams", "x0 = $zzms_claimed", "run = CustomShaderZZMS_ClearSlot",
        "run = CommandListZZMS_RestoreIniParams",
        "endif"]
    clear_lines = _section(ini_text, "[CustomShaderZZMS_ClearSlot]")
    assert clear_lines[1].startswith("cs-t0 = ref " + params), clear_lines
    assert clear_lines[:1] + clear_lines[2:] == [
        "cs = Shaders/" + zms.INSTANCE_CLEAR_SHADER_FILENAME,
        "cs-u0 = ref ResourceZZMS_Store", "cs-u1 = ref ResourceZZMS_Records", "cs-u2 = ref ResourceZZMS_Owners",
        "dispatch = 1, 1, 1", "cs-u0 = null", "cs-u1 = null", "cs-u2 = null", "cs-t0 = null"], clear_lines
    assert _section(ini_text, "[CommandListZZMS_BackupIniParams]") == [
        "$zzms_bak_x0 = x0", "$zzms_bak_y0 = y0", "$zzms_bak_z0 = z0", "$zzms_bak_w0 = w0"]
    assert _section(ini_text, "[CommandListZZMS_RestoreIniParams]") == [
        "x0 = $zzms_bak_x0", "y0 = $zzms_bak_y0", "z0 = $zzms_bak_z0", "w0 = $zzms_bak_w0"]
    assert _section(ini_text, "[CommandListZZMS_ObjectWindow]") == [
        "$zzms_cb_offset = vs-cb1->offset", "$zzms_cb_size = vs-cb1->size"]
    constants = _section(ini_text, "[Constants]")
    for variable in ("$zzms_table", "$zzms_key3", "$zzms_host0", "$zzms_host3", "$zzms_prev3",
                     "$zzms_claim", "$zzms_claimed", "$zzms_s0", "$zzms_s3", "$zzms_k3", "$zzms_o0", "$zzms_o3",
                     "$zzms_cb_offset", "$zzms_cb_size", "$zzms_bak_x0"):
        assert "global " + variable + " = 0" in constants, constants
    for filename in (zms.SKIN_SHADER_FILENAME, zms.INSTANCE_IMPORT_SHADER_FILENAME, zms.INSTANCE_RECORD_SHADER_FILENAME,
                     zms.INSTANCE_MATCH_SHADER_FILENAME, zms.INSTANCE_GATHER_SHADER_FILENAME,
                     zms.INSTANCE_CLEAR_SHADER_FILENAME, zms.PALETTE_SHADER_FILENAME):
        assert os.path.isfile(os.path.join(mod_folder, "Shaders", filename)), filename
    # Only the shaders it uses.
    assert sorted(os.listdir(os.path.join(mod_folder, "Shaders"))) == sorted([
        zms.SKIN_SHADER_FILENAME, zms.INSTANCE_IMPORT_SHADER_FILENAME, zms.INSTANCE_RECORD_SHADER_FILENAME,
        zms.INSTANCE_MATCH_SHADER_FILENAME, zms.INSTANCE_GATHER_SHADER_FILENAME, zms.INSTANCE_CLEAR_SHADER_FILENAME,
        zms.PALETTE_SHADER_FILENAME])
    assert not any(line.strip().startswith("so0") for line in ini_text.splitlines()), "no stream-output target is re-bound"
    for forbidden in ("zzms_ready", "Place", "ZZMS_Delay", "zzms_primed", "SkinnedPrev", "[Present]"):
        assert forbidden not in ini_text, forbidden


def _check_same_frame_skin(zms, profile, source, mod_folder, drawing, layout):
    """The skinning parameters decode the exported buffers: same result as skinning them directly."""
    meshes = os.path.join(mod_folder, "Meshes")
    ib = source["components"][drawing]["spec"]["ib"]
    with open(os.path.join(meshes, ib + "-Position.buf"), "rb") as handle:
        position_bytes = handle.read()
    with open(os.path.join(meshes, ib + "-Blend.buf"), "rb") as handle:
        blend_bytes = handle.read()
    parts = source["components"]
    pool = numpy.concatenate([parts[name]["palette"] for name in ("body", "hair", "accessory")]).astype(numpy.float64)
    palette = zms.reference_palette(profile, pool, zms.get_component_by_ib(profile)[ib]["component_id"])
    position, weights, indices = _read_body_buffers(mod_folder, ib)
    params = zms.same_frame_skin_params(len(position), layout)
    skinned = zms.reference_same_frame_skin(palette.reshape(-1, 4), position_bytes, blend_bytes, params)
    matrices = numpy.zeros((len(position), 3, 4))
    for k in range(4):
        matrices += weights[:, k, None, None] * palette[indices[:, k]]
    expected = numpy.einsum("vij,vj->vi", matrices, numpy.c_[position[:, :3], numpy.ones(len(position))])
    error = float(numpy.abs(skinned[:, :3] - expected).max())
    assert error < 1e-4, error
    return error


def _delayed_skin_lines(name, draw_count):
    return [
        "ResourceZZMS_Input = ref vs-t0",
        "run = CustomShaderZZMS_Import_" + name,
        "$zzms_imported = 1",
        "if $zzms_ready",
        "run = CustomShaderZZMS_Delay_" + name,
        "vs-t0 = ref ResourceZZMS_Delayed_" + name,
        "endif",
        "vb0 = Resource" + name + "Position",
        "vb2 = Resource" + name + "Blend",
        "draw = " + str(draw_count) + ", 0",
    ]


def _check_runtime(zms, profile, source, position, weights, indices, drawing, provider):
    """Skin the exported vertices with the palette the runtime builds and compare with the game."""
    parts = source["components"]
    merged = zms.merged_components(profile)
    pool = numpy.concatenate([parts[name]["palette"] for name in ("body", "hair", "accessory")]).astype(numpy.float64)
    assert [component["ib_hash"] for component in merged] == [parts[name]["spec"]["ib"] for name in ("body", "hair", "accessory")]
    drawing_component = zms.get_component_by_ib(profile)[parts[drawing]["spec"]["ib"]]
    palette = zms.reference_palette(profile, pool, drawing_component["component_id"])
    matrices = numpy.zeros((len(position), 3, 4))
    for k in range(4):
        matrices += weights[:, k, None, None] * palette[indices[:, k]]
    skinned = numpy.einsum("vij,vj->vi", matrices, numpy.c_[position[:, :3], numpy.ones(len(position))])

    to_drawing = numpy.linalg.inv(parts[drawing]["root"]) @ parts[provider]["root"]
    expected = {}
    for name, transform in ((drawing, numpy.eye(4)), (provider, to_drawing)):
        rest = parts[name]["position"]["POSITION"].astype(numpy.float32)
        posed = parts[name]["posed"]["POSITION"].astype(numpy.float64)
        for vertex in range(len(rest)):
            moved = (transform @ numpy.r_[posed[vertex], 1.0])[:3]
            expected.setdefault(rest[vertex].tobytes(), []).append((name, moved))
    worst = {drawing: 0.0, provider: 0.0}
    counts = {drawing: 0, provider: 0}
    for vertex in range(len(position)):
        candidates = expected[position[vertex, :3].tobytes()]
        errors = [(float(numpy.abs(skinned[vertex] - moved).max()), name) for name, moved in candidates]
        error, name = min(errors)
        worst[name] = max(worst[name], error)
        counts[name] += 1
    assert counts[drawing] and counts[provider], counts
    assert worst[drawing] < 1e-5 and worst[provider] < 1e-4, worst
    return worst, counts


def main():
    if not hasattr(bpy.context.scene, "global_properties"):
        bpy.ops.preferences.addon_enable(module="LoyalTools")

    import LoyalTools
    assert LoyalTools.__file__.startswith(str(TESTS_DIR.parents[1])), LoyalTools.__file__
    from LoyalTools.common.global_config import GlobalConfig
    from LoyalTools.common.logic_name import LogicName
    from LoyalTools.common import zzmi_merged_skeleton as zms
    from LoyalTools.common.global_key_count_helper import GlobalKeyCountHelper

    with tempfile.TemporaryDirectory(prefix="loyal_zzmi_merged_export_") as temp_folder:
        dump_folder = os.path.join(temp_folder, "FrameAnalysis-merged")
        os.makedirs(dump_folder)
        source = synthetic.build_dump(dump_folder)
        parts = source["components"]
        ib = {name: data["spec"]["ib"] for name, data in parts.items()}

        props = bpy.context.scene.global_properties
        props.workspace_source_mode = "CUSTOM"
        props.custom_workspace_folder_path = os.path.join(temp_folder, "zzmi_merged")
        props.force_standalone_preset = True
        props.standalone_game_preset = LogicName.ZZMI
        props.use_specific_generate_mod_folder_path = True
        props.open_mod_folder_after_generate_mod = False
        props.zzz_outline_optimization = False
        GlobalConfig.read_from_main_json_ssmt4()
        workspace = GlobalConfig.path_workspace_folder()

        # 1. extraction
        scene = bpy.context.scene
        scene.loyal_extract_props.workflow_mode = 'MERGED_SKELETON'
        scene.loyal_extract_props.frame_dump_folder = dump_folder
        assert bpy.ops.loyal.extract_from_dump() == {'FINISHED'}
        assert scene.loyal_extract_props.last_report.startswith("骨骼合并提取完成"), scene.loyal_extract_props.last_report
        profile = zms.load_profile(workspace)
        tree_name = props.selected_blueprint_name
        tree = bpy.data.node_groups[tree_name]
        assert tree.get(zms.TREE_MARKER_MERGED_SKELETON) and not tree.get("LoyalTools:EFMIMergedSkeleton")
        from LoyalTools.blueprint.node_obj import _is_zzmi_merged_tree
        assert _is_zzmi_merged_tree(tree), "the Generate Mod node must offer the frame-delay option"

        objects = {name: bpy.data.objects.get(parts[name]["unique_str"]) for name in parts}
        assert objects["bangboo"] is None
        for name in ("body", "hair", "accessory"):
            assert len(objects[name].vertex_groups) == profile["bones_count"] == 15, name
            assert objects[name][zms.OBJECT_MARKER_ROLE] == zms.ROLE_MERGED
        assert len(objects["face"].vertex_groups) <= 3
        assert objects["face"][zms.OBJECT_MARKER_ROLE] == zms.ROLE_SHAPE_KEYED
        # The hair's head weights land on the body's head group (canonical id 6).
        head_group = objects["hair"].vertex_groups["6"].index
        assert any(group.group == head_group for vertex in objects["hair"].data.vertices for group in vertex.groups)

        # The import button rebuilds the same scene from the profile.
        assert bpy.ops.loyal.import_workspace() == {'FINISHED'}
        copies = [obj for obj in bpy.data.objects if obj.name.startswith(parts["hair"]["unique_str"] + ".")]
        assert copies and len(copies[0].vertex_groups) == 15
        for obj in [obj for obj in bpy.data.objects if "." in obj.name and obj.name.split(".")[0] in
                    {data["unique_str"] for data in parts.values()}]:
            bpy.data.objects.remove(obj, do_unlink=True)
        props.selected_blueprint_name = tree_name

        # 2. unchanged export
        mod_a = os.path.join(temp_folder, "mod_unchanged")
        ini_a = _export(tree_name, mod_a)
        # $active0..N are numbered per ini: body, hair, accessory and the face
        assert GlobalKeyCountHelper.generated_mod_number == 4, GlobalKeyCountHelper.generated_mod_number
        assert "ZZMS" not in ini_a, "an unchanged export needs no runtime and hides nothing"
        assert not os.path.isdir(os.path.join(mod_a, "Shaders"))
        body_blend_lines = _section(ini_a, "[TextureOverride_VB_" + ib["body"] + "_" + ib["body"] + "_Blend]")
        assert "draw = " + str(_draw_count(mod_a, ib["body"])) + ", 0" in body_blend_lines, body_blend_lines
        assert "vb0 = Resource" + ib["body"] + "Position" in body_blend_lines, body_blend_lines
        assert "[TextureOverride_IB_" + ib["face"] + "]" not in ini_a
        assert "hash = " + parts["face"]["spec"]["blend"] not in ini_a
        assert not os.path.exists(os.path.join(mod_a, "Meshes", ib["face"] + "-Position.buf"))
        face_lines = _section(ini_a, "[TextureOverride_" + parts["face"]["unique_str"].replace("-", "_") + "]")
        assert face_lines[:3] == ["hash = " + ib["face"], "match_first_index = 0",
                                  "match_index_count = " + str(len(parts["face"]["indices"]))], face_lines
        assert face_lines[-1] == r"run = CommandList\ZZMI\SetTextures", face_lines
        assert any(line.startswith(r"Resource\ZZMI\Diffuse = ref ") for line in face_lines), face_lines
        assert not any(line.startswith(("ib =", "handling", "drawindexed")) for line in face_lines), face_lines
        _, _, body_indices = _read_body_buffers(mod_a, ib["body"])
        assert body_indices.max() < 8, "unchanged body must keep its own (local) bone ids"
        _, hair_weights, hair_indices = _read_body_buffers(mod_a, ib["hair"])
        assert hair_indices[hair_weights > 0].max() < 5, "unchanged hair must keep its own (local) bone ids"

        # 2b. the same blueprint, uniform one-frame delay: every part reads its own previous bones
        by_ib = zms.get_component_by_ib(profile)
        face_component = by_ib[ib["face"]]
        delayed_layout = zms.delayed_pool_layout(profile, [face_component["component_id"]])
        pool_total = profile["bones_count"] + face_component["palette_bone_count"]
        mod_a2 = os.path.join(temp_folder, "mod_unchanged_delayed")
        ini_a2 = _export(tree_name, mod_a2, frame_delay=True)
        assert GlobalKeyCountHelper.generated_mod_number == 4, GlobalKeyCountHelper.generated_mod_number
        _same_meshes(mod_a, mod_a2)
        # Every drawn part, the face included, is drawn with its object matrix of the previous frame.
        by_order = lambda names: sorted((ib[name] for name in names), key=lambda value: by_ib[value]["pose_order"])
        _check_delay_frame(ini_a2, zms, profile, parts, pool_total, by_order(("body", "hair", "accessory", "face")))
        for name in ("body", "hair", "accessory"):
            component = by_ib[ib[name]]
            assert _section(ini_a2, "[CommandListZZMS_Skin_" + ib[name] + "]") == _delayed_skin_lines(
                ib[name], _draw_count(mod_a2, ib[name])), name
            assert _section(ini_a2, "[ResourceZZMS_DelayParams_" + ib[name] + "]")[-1] == "data = %d %d 0 0" % (
                component["vg_offset"], component["vg_count"]), name
            blend_lines = _section(ini_a2, "[TextureOverride_VB_" + ib[name] + "_" + ib[name] + "_Blend]")
            assert "run = CommandListZZMS_Skin_" + ib[name] in blend_lines, blend_lines
            assert not any("Place" in line for line in blend_lines), blend_lines
            _check_placed_draw(ini_a2, parts[name]["unique_str"], ib[name])
        assert "CustomShaderZZMS_Palette" not in ini_a2, "no part uses the bones of another one"
        assert _section(ini_a2, "[ResourceZZMS_Components]")[-1] == "data = " + " ".join(
            str(value) for value in zms.runtime_component_table(profile, delayed_layout))
        for filename in (zms.IMPORT_SHADER_FILENAME, zms.PALETTE_SHADER_FILENAME, zms.DELAY_SHADER_FILENAME):
            assert os.path.isfile(os.path.join(mod_a2, "Shaders", filename)), filename

        # 2c. same-frame skinning with no part using bones of another one: the ordinary export
        mod_a3 = os.path.join(temp_folder, "mod_unchanged_same_frame")
        ini_a3 = _export(tree_name, mod_a3, same_frame=True)
        assert ini_a3 == ini_a, "same-frame skinning changes nothing when no part uses foreign bones"
        _same_meshes(mod_a, mod_a3)
        assert not os.path.isdir(os.path.join(mod_a3, "Shaders"))

        # 3. a copy of the hair joined into the body
        hair_obj, body_obj = objects["hair"], objects["body"]
        hair_copy = hair_obj.copy()
        hair_copy.data = hair_obj.data.copy()
        scene.collection.objects.link(hair_copy)
        bpy.ops.object.select_all(action='DESELECT')
        hair_copy.select_set(True)
        body_obj.select_set(True)
        bpy.context.view_layer.objects.active = body_obj
        bpy.ops.object.join()
        mod_b = os.path.join(temp_folder, "mod_hair_in_body")
        ini_b = _export(tree_name, mod_b)
        assert GlobalKeyCountHelper.generated_mod_number == 4, "the counter must restart for every export"
        body, hair = ib["body"], ib["hair"]
        body_vertices = len(parts["body"]["position"])
        hair_vertices = len(parts["hair"]["position"])
        body_blend_b = _section(ini_b, "[TextureOverride_VB_" + body + "_" + body + "_Blend]")
        assert body_blend_b[-3:] == [
            "elif DRAW_TYPE == 1 && VERTEX_COUNT == " + str(body_vertices),
            "run = CommandListZZMS_Skin_" + body, "endif"], body_blend_b
        assert _section(ini_b, "[CommandListZZMS_Skin_" + body + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + body,
            "run = CustomShaderZZMS_Palette_" + body,
            "vs-t0 = ref ResourceZZMS_Palette_" + body,
            "vb0 = Resource" + body + "Position",
            "vb2 = Resource" + body + "Blend",
            "draw = " + str(_draw_count(mod_b, body)) + ", 0",
        ]
        assert _section(ini_b, "[CommandListZZMS_Skin_" + hair + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + hair,
            "vb0 = Resource" + hair + "Position",
            "vb2 = Resource" + hair + "Blend",
            "draw = " + str(_draw_count(mod_b, hair)) + ", 0",
        ]
        hair_blend_b = _section(ini_b, "[TextureOverride_VB_" + hair + "_" + hair + "_Blend]")
        assert "elif DRAW_TYPE == 1 && VERTEX_COUNT == " + str(hair_vertices) in hair_blend_b, hair_blend_b
        for forbidden in ("so0", "FRAME_NUMBER", "Reskin", "$zzms_seen"):
            assert forbidden not in ini_b, forbidden
        assert "dispatch = 1, 1, 1" in _section(ini_b, "[CustomShaderZZMS_Palette_" + body + "]")
        assert _section(ini_b, "[ResourceZZMS_Components]")[-1] == "data = " + " ".join(
            str(value) for value in zms.runtime_component_table(profile))
        assert _section(ini_b, "[ResourceZZMS_Pool]") == [
            "type = RWBuffer", "format = R32G32B32A32_FLOAT", "array = 45",
            "bind_flags = shader_resource unordered_access"]
        assert "[CommandListZZMS_Skin_" + ib["accessory"] + "]" not in ini_b
        for filename in (zms.IMPORT_SHADER_FILENAME, zms.PALETTE_SHADER_FILENAME):
            assert os.path.isfile(os.path.join(mod_b, "Shaders", filename)), filename
        position, weights, indices = _read_body_buffers(mod_b, body)
        active = weights > 0
        assert set(numpy.unique(indices[active]).tolist()) <= set(range(13))
        assert (indices[active] >= 9).any(), "hair bones must come from the hair's range"
        worst, counts = _check_runtime(zms, profile, source, position, weights, indices, "body", "hair")
        print("hair in body: vertices " + repr(counts) + ", worst error " + repr(worst))
        assert not os.path.isfile(os.path.join(mod_b, "Shaders", zms.DELAY_SHADER_FILENAME))

        # 3b. hair in body, uniform delay: the body's merged palette is built from the previous pool
        mod_b2 = os.path.join(temp_folder, "mod_hair_in_body_delayed")
        ini_b2 = _export(tree_name, mod_b2, frame_delay=True)
        _same_meshes(mod_b, mod_b2)
        _check_delay_frame(ini_b2, zms, profile, parts, pool_total, by_order(("body", "hair", "accessory", "face")))
        assert _section(ini_b2, "[CommandListZZMS_Skin_" + body + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + body,
            "$zzms_imported = 1",
            "if $zzms_ready",
            "run = CustomShaderZZMS_PaletteDelayed_" + body,
            "else",
            "run = CustomShaderZZMS_Palette_" + body,
            "endif",
            "vs-t0 = ref ResourceZZMS_Palette_" + body,
            "vb0 = Resource" + body + "Position",
            "vb2 = Resource" + body + "Blend",
            "draw = " + str(_draw_count(mod_b2, body)) + ", 0",
        ]
        assert "cs-t0 = ref ResourceZZMS_PoolPrev" in _section(ini_b2, "[CustomShaderZZMS_PaletteDelayed_" + body + "]")
        assert "cs-t0 = ref ResourceZZMS_Pool" in _section(ini_b2, "[CustomShaderZZMS_Palette_" + body + "]")
        assert _section(ini_b2, "[CommandListZZMS_Skin_" + hair + "]") == _delayed_skin_lines(
            hair, _draw_count(mod_b2, hair))
        assert "[ResourceZZMS_Delayed_" + body + "]" not in ini_b2
        assert _section(ini_b2, "[ResourceZZMS_Palette_" + body + "]")[2] == "array = 45"

        # 3c. hair in body, same-frame skinning: the body is skinned by the mod at its first draw
        body_layout = (32, 0, 1, 16, False)  # BLENDWEIGHTS float4 @0, BLENDINDICES uint4 @16
        mod_b3 = os.path.join(temp_folder, "mod_hair_in_body_same_frame")
        ini_b3 = _export(tree_name, mod_b3, same_frame=True)
        _same_meshes(mod_b, mod_b3)
        _check_same_frame(ini_b3, zms, profile, mod_b3, body, parts["body"]["unique_str"], body_layout,
                          by_order(("body", "hair")), by_order(("body", "hair")))
        assert body_blend_b == _section(ini_b3, "[TextureOverride_VB_" + body + "_" + body + "_Blend]")
        assert _section(ini_b3, "[CustomShaderZZMS_Palette_" + body + "]") == _section(ini_b, "[CustomShaderZZMS_Palette_" + body + "]")
        assert "PoolZZMS_Out_" + hair not in ini_b3, "the hair uses only its own bones"
        # The hair is drawn as usual and records where it is (its copies are matched by it).
        hair_lines = _section(ini_b3, "[TextureOverride_" + parts["hair"]["unique_str"].replace("-", "_") + "]")
        hair_draw = next(index for index, line in enumerate(hair_lines) if line.startswith("drawindexed"))
        assert hair_lines[hair_draw - 1] == "run = CommandListZZMS_Record", hair_lines
        assert _section(ini_b3, "[CommandListZZMS_Skin_" + hair + "]")[3:] == [
            "vb0 = Resource" + hair + "Position", "vb2 = Resource" + hair + "Blend",
            "draw = " + str(_draw_count(mod_b3, hair)) + ", 0"]
        skin_error = _check_same_frame_skin(zms, profile, source, mod_b3, "body", body_layout)
        print("hair in body, same frame: worst error " + repr(skin_error))
        # Both options ticked: same-frame skinning wins.
        mod_b4 = os.path.join(temp_folder, "mod_hair_in_body_both")
        assert _export(tree_name, mod_b4, frame_delay=True, same_frame=True) == ini_b3

        # 4. the hair and the accessory taken out of the blueprint are hidden; the hair still provides bones
        removed = {parts["hair"]["unique_str"], parts["accessory"]["unique_str"]}
        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") == 'SSMTNode_Object_Info' and node.object_name in removed:
                tree.nodes.remove(node)
        mod_c = os.path.join(temp_folder, "mod_hair_provider")
        ini_c = _export(tree_name, mod_c)
        assert GlobalKeyCountHelper.generated_mod_number == 2, GlobalKeyCountHelper.generated_mod_number
        accessory = ib["accessory"]
        assert "[TextureOverride_VB_" + hair + "_" + hair + "_Blend]" not in ini_c
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_" + hair + "]") == [
            "hash = " + parts["hair"]["spec"]["blend"],
            "if DRAW_TYPE == 1 && VERTEX_COUNT == " + str(hair_vertices),
            "run = CommandListZZMS_Skin_" + hair,
            "elif DRAW_TYPE == 2 || DRAW_TYPE == 4",
            "checktextureoverride = ib",
            "endif"]
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_IB_" + hair + "]") == ["hash = " + hair, "handling = skip"]
        assert _section(ini_c, "[CommandListZZMS_Skin_" + hair + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + hair,
        ]
        # The accessory provides no bone to the body: hidden only, its skinning draw is left alone.
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_" + accessory + "]") == [
            "hash = " + parts["accessory"]["spec"]["blend"],
            "if DRAW_TYPE == 2 || DRAW_TYPE == 4",
            "checktextureoverride = ib",
            "endif"]
        assert _section(ini_c, "[TextureOverride_ZZMS_Hidden_IB_" + accessory + "]") == [
            "hash = " + accessory, "handling = skip"]
        assert "[CommandListZZMS_Skin_" + accessory + "]" not in ini_c
        # The face is not merged: leaving it out of the blueprint never hides it.
        assert "Hidden_" + ib["face"] not in ini_c
        assert "so0" not in ini_c
        assert not os.path.exists(os.path.join(mod_c, "Meshes", hair + "-Position.buf"))
        assert "zzms_ready" not in ini_c and "ZZMS_Delay" not in ini_c, "with the option off the export is the ordinary one"

        # 4b. the same, uniform delay: the hidden hair still only imports its bones
        mod_c2 = os.path.join(temp_folder, "mod_hair_provider_delayed")
        ini_c2 = _export(tree_name, mod_c2, frame_delay=True)
        assert GlobalKeyCountHelper.generated_mod_number == 2, GlobalKeyCountHelper.generated_mod_number
        _same_meshes(mod_c, mod_c2)
        _check_delay_frame(ini_c2, zms, profile, parts, pool_total, by_order(("body", "face")))
        _check_placed_draw(ini_c2, parts["body"]["unique_str"], body)
        for name in ("[TextureOverride_ZZMS_Hidden_" + hair + "]", "[TextureOverride_ZZMS_Hidden_IB_" + hair + "]",
                     "[TextureOverride_ZZMS_Hidden_" + accessory + "]"):
            assert _section(ini_c2, name) == _section(ini_c, name), name
        assert _section(ini_c2, "[CommandListZZMS_Skin_" + hair + "]") == [
            "ResourceZZMS_Input = ref vs-t0",
            "run = CustomShaderZZMS_Import_" + hair,
            "$zzms_imported = 1",
        ]
        assert "[CommandListZZMS_Skin_" + accessory + "]" not in ini_c2

        # 4c. the same, same-frame skinning: the hidden hair imports its bones, the body is skinned at its draw
        mod_c3 = os.path.join(temp_folder, "mod_hair_provider_same_frame")
        ini_c3 = _export(tree_name, mod_c3, same_frame=True)
        _same_meshes(mod_c, mod_c3)
        _check_same_frame(ini_c3, zms, profile, mod_c3, body, parts["body"]["unique_str"], body_layout,
                          by_order(("body", "hair")), by_order(("body", "hair")))
        for name in ("[TextureOverride_ZZMS_Hidden_" + hair + "]", "[TextureOverride_ZZMS_Hidden_" + accessory + "]",
                     "[TextureOverride_ZZMS_Hidden_IB_" + accessory + "]"):
            assert _section(ini_c3, name) == _section(ini_c, name), name
        # The hidden hair is not drawn, but records where it is at its draws.
        assert _section(ini_c3, "[TextureOverride_ZZMS_Hidden_IB_" + hair + "]") == [
            "hash = " + hair, "handling = skip", "run = CommandListZZMS_Record"]

        # 5. the face taken out of the blueprint too: the game still draws it, so it is still delayed and placed
        for node in list(tree.nodes):
            if getattr(node, "bl_idname", "") == 'SSMTNode_Object_Info' and node.object_name == parts["face"]["unique_str"]:
                tree.nodes.remove(node)
        mod_d = os.path.join(temp_folder, "mod_face_left_out_delayed")
        ini_d = _export(tree_name, mod_d, frame_delay=True)
        face_ib = ib["face"]
        assert "[TextureOverride_" + parts["face"]["unique_str"].replace("-", "_") + "]" not in ini_d
        face_first, face_count = parts["face"]["unique_str"].split("-")[2], parts["face"]["unique_str"].split("-")[1]
        assert _section(ini_d, "[TextureOverride_ZZMS_Face_" + face_ib + "_" + face_first + "]") == [
            "hash = " + face_ib, "match_first_index = " + face_first, "match_index_count = " + face_count,
        ] + _face_draw_lines(face_ib)
        assert "[CommandListZZMS_PlaceBegin_" + face_ib + "]" in ini_d
        assert "[TextureOverride_ZZMS_Delay_" + face_ib + "]" in ini_d

    print("BLENDER_ZZMI_MERGED_EXPORT_REGRESSION=PASS")


main()
